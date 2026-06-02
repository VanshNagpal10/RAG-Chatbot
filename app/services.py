"""
services.py — Core extraction logic for YouTube and Instagram videos.

Architecture notes:
─────────────────────────────────────────────────────────────────────
ASYNC STRATEGY:
  yt-dlp, youtube-transcript-api, and faster-whisper are all synchronous.
  FastAPI runs on an asyncio event loop — blocking it kills concurrency.

  Solution: Every blocking call is wrapped in `asyncio.to_thread()` which pushes the
  work onto the default ThreadPoolExecutor. This means:
    1. The event loop stays free to handle other requests.
    2. We get natural parallelism — both videos extract concurrently via asyncio.gather().
    3. No need for a custom thread pool unless we hit >40 concurrent extractions
       (the default pool size).

YOUTUBE TRANSCRIPT FALLBACK CHAIN:
  1. youtube-transcript-api (manual captions)  →  fastest, highest quality
  2. youtube-transcript-api (auto-generated)   →  still fast, lower quality
  3. yt-dlp audio download → local Whisper     →  slowest, but always works & FREE

INSTAGRAM RESILIENCE:
  Meta's CDN is hostile to scrapers. We mitigate with:
  - cookies.txt from an authenticated browser session
  - Desktop Chrome User-Agent
  - Audio-only download (saves bandwidth, reduces detection surface)
  - Generous retry/timeout config in yt-dlp

LOCAL WHISPER (faster-whisper):
  We use CTranslate2-based faster-whisper instead of the paid OpenAI API.
  - Model: "base" (~150MB, downloads once, cached forever)
  - 4x faster than original openai-whisper on CPU
  - Zero API cost — critical for keeping the project in budget
  - Trade-off: ~30s per 1-minute reel on CPU (acceptable for prototyping)
  - For production scale: swap to GPU instance or hosted Whisper service

ERROR PHILOSOPHY:
  We never let one video's failure crash the other. Each extraction is independent.
  Errors are collected into a warnings list and returned alongside whatever data
  we successfully extracted. The caller decides what's acceptable.
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

import yt_dlp
from faster_whisper import WhisperModel
from youtube_transcript_api import YouTubeTranscriptApi

from app.schemas import (
    TranscriptChunk,
    TranscriptResult,
    VideoExtractionResult,
    VideoMetadata,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PLATFORM DETECTION
# ─────────────────────────────────────────────


def detect_platform(url: str) -> Literal["youtube", "instagram"]:
    """
    Determine the platform from the URL.

    We parse the hostname rather than regex-matching the full URL because:
    - YouTube has multiple valid domains (youtube.com, youtu.be, m.youtube.com)
    - Instagram has reels under instagram.com/reel/ and instagram.com/p/
    - This is more resilient to URL parameter variations
    """
    host = urlparse(url).hostname or ""
    # Strip 'www.' and 'm.' prefixes for normalization
    host = re.sub(r"^(www\.|m\.)", "", host)

    if host in ("youtube.com", "youtu.be"):
        return "youtube"
    elif host == "instagram.com":
        return "instagram"
    else:
        raise ValueError(
            f"Unsupported platform for URL: {url}. Only YouTube and Instagram are supported."
        )


def extract_youtube_video_id(url: str) -> str:
    """
    Extract the 11-char video ID from any YouTube URL format.

    Handles:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - https://m.youtube.com/watch?v=VIDEO_ID
    """
    parsed = urlparse(url)
    host = re.sub(r"^(www\.|m\.)", "", parsed.hostname or "")

    if host == "youtu.be":
        return parsed.path.lstrip("/")

    if host == "youtube.com":
        # /shorts/VIDEO_ID or /embed/VIDEO_ID
        if parsed.path.startswith(("/shorts/", "/embed/")):
            return parsed.path.split("/")[2]
        # Standard ?v=VIDEO_ID
        from urllib.parse import parse_qs
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]

    raise ValueError(f"Could not extract YouTube video ID from: {url}")


# ─────────────────────────────────────────────
# YT-DLP BASE OPTIONS
# ─────────────────────────────────────────────


def _yt_dlp_base_opts(
    cookies_path: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    audio_only: bool = False,
) -> dict:
    """
    Shared yt-dlp options. Centralized here so every callsite uses the same
    resilient config — no copy-paste drift.

    Why these specific options:
    - quiet/no_warnings: We handle logging ourselves; yt-dlp's stdout is noisy.
    - socket_timeout: Meta's CDN sometimes hangs; 30s prevents infinite waits.
    - retries: Transient 403s from Instagram often resolve on retry.
    - User-Agent: Instagram blocks the default yt-dlp UA. Chrome desktop works.

    Cookie priority:
    1. cookies_from_browser (e.g. "chrome") — auto-extracts from browser, zero setup
    2. cookies_path (cookies.txt file) — manual but works everywhere
    3. No cookies — works for some YouTube videos, fails for most Instagram
    """
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 3,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
        },
    }

    # cookies_from_browser takes priority — it's the easiest setup.
    # yt-dlp natively supports: chrome, firefox, safari, edge, opera, brave, etc.
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
        logger.info("Using cookies from browser: %s", cookies_from_browser)
    elif cookies_path and Path(cookies_path).exists():
        opts["cookiefile"] = cookies_path
        logger.info("Using cookies file: %s", cookies_path)

    if audio_only:
        opts["format"] = "bestaudio/best"
        # Post-process to wav for Whisper compatibility
        opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",  # best quality
            }
        ]

    return opts


# ─────────────────────────────────────────────
# METADATA EXTRACTION
# ─────────────────────────────────────────────


def _extract_metadata_sync(
    url: str,
    video_id: Literal["A", "B"],
    platform: Literal["youtube", "instagram"],
    cookies_path: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
) -> VideoMetadata:
    """
    Synchronous metadata extraction via yt-dlp.

    yt-dlp's extract_info with download=False fetches only metadata — no audio/video
    download. This is fast (~1-3s for YouTube, ~3-8s for Instagram).

    Field mapping is platform-specific because yt-dlp normalizes *some* fields
    but not all. Instagram's `comment_count` vs YouTube's `comment_count` are
    consistent, but `channel_follower_count` only exists for YouTube.
    """
    opts = _yt_dlp_base_opts(cookies_path=cookies_path, cookies_from_browser=cookies_from_browser)

    # CRITICAL: We only need metadata, not a download URL.
    # yt-dlp's default format selector (bestvideo*+bestaudio/best) can fail
    # when Chrome cookies cause YouTube to return a different format list.
    # Setting skip_download + ignore_no_formats_error + a permissive format
    # prevents format-related errors from killing metadata extraction.
    opts["skip_download"] = True
    opts["ignore_no_formats_error"] = True
    # 'best' always resolves — it picks any single pre-merged stream.
    # This is irrelevant since we don't download, but yt-dlp still validates it.
    opts["format"] = "best"

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if not info:
        raise RuntimeError(f"yt-dlp returned no info for {url}")

    # Parse upload_date: yt-dlp returns "YYYYMMDD" string
    upload_date = None
    raw_date = info.get("upload_date")
    if raw_date:
        try:
            upload_date = datetime.strptime(raw_date, "%Y%m%d")
        except ValueError:
            logger.warning("Could not parse upload_date: %s", raw_date)

    # Extract hashtags from description if not provided directly
    hashtags = info.get("tags") or []
    description = info.get("description") or ""
    if not hashtags:
        # Fallback: pull #hashtags from description text
        hashtags = re.findall(r"#(\w+)", description)

    metadata = VideoMetadata(
        video_id=video_id,
        platform=platform,
        source_url=url,
        title=info.get("title"),
        creator=info.get("uploader") or info.get("channel"),
        follower_count=info.get("channel_follower_count"),
        views=info.get("view_count") or info.get("play_count"),
        likes=info.get("like_count"),
        comments=info.get("comment_count"),
        hashtags=hashtags,
        upload_date=upload_date,
        duration_seconds=info.get("duration"),
    )

    # Compute engagement rate
    # Guard against division by zero and missing fields
    if metadata.views and metadata.views > 0:
        likes = metadata.likes or 0
        comments = metadata.comments or 0
        metadata.engagement_rate = round(
            ((likes + comments) / metadata.views) * 100, 4
        )
    elif metadata.likes and metadata.likes > 0:
        # Instagram Reels often hide view counts. In that case, we can't compute
        # a true engagement rate. We flag it as None and let the frontend show "N/A".
        # The LLM can still compare likes/comments directly.
        logger.warning(
            "No view count for video %s (%s) — engagement rate unavailable. "
            "Likes: %s, Comments: %s",
            video_id, platform, metadata.likes, metadata.comments,
        )

    logger.info(
        "Extracted metadata for video %s (%s): %s",
        video_id,
        platform,
        metadata.title,
    )
    return metadata


async def extract_metadata(
    url: str,
    video_id: Literal["A", "B"],
    platform: Literal["youtube", "instagram"],
    cookies_path: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
) -> VideoMetadata:
    """
    Async wrapper — pushes the blocking yt-dlp call to a thread.

    Why asyncio.to_thread instead of run_in_executor?
    - Cleaner API (Python 3.9+)
    - Automatically uses the default ThreadPoolExecutor
    - No need to manage executor lifecycle
    """
    return await asyncio.to_thread(
        _extract_metadata_sync, url, video_id, platform, cookies_path, cookies_from_browser
    )


# ─────────────────────────────────────────────
# TRANSCRIPT EXTRACTION — YOUTUBE
# ─────────────────────────────────────────────


def _fetch_youtube_transcript_sync(
    url: str,
    video_id: Literal["A", "B"],
) -> Optional[TranscriptResult]:
    """
    Attempt to fetch YouTube transcript via the youtube-transcript-api.

    Strategy:
    1. Try English first (most useful for our RAG pipeline)
    2. If English unavailable, list ALL available transcripts and grab the first one.
       A Korean transcript is better than no transcript — Whisper fallback is expensive.
    3. If nothing works, return None so the caller falls back to Whisper.
    """
    yt_video_id = extract_youtube_video_id(url)

    try:
        ytt_api = YouTubeTranscriptApi()

        # Attempt 1: Fetch English transcript
        try:
            transcript = ytt_api.fetch(yt_video_id, languages=["en"])
        except Exception:
            # Attempt 2: Fetch ANY available transcript
            logger.info(
                "English transcript unavailable for video %s, trying any language...",
                video_id,
            )
            transcript_list = ytt_api.list(yt_video_id)
            # transcript_list is iterable — grab the first available
            first_available = None
            for t in transcript_list:
                first_available = t
                break

            if first_available is None:
                logger.warning("No transcripts at all for video %s", video_id)
                return None

            transcript = ytt_api.fetch(
                yt_video_id, languages=[first_available.language_code]
            )

        chunks = [
            TranscriptChunk(
                video_id=video_id,
                text=snippet.text,
                start=snippet.start,
                end=snippet.start + snippet.duration,
            )
            for snippet in transcript.snippets
        ]

        logger.info(
            "Fetched YouTube transcript for video %s: %d chunks via API (lang=%s)",
            video_id,
            len(chunks),
            transcript.language,
        )

        return TranscriptResult(
            video_id=video_id,
            source="youtube_api",
            language=transcript.language,
            chunks=chunks,
        )

    except Exception as exc:
        logger.warning(
            "youtube-transcript-api failed for video %s (%s): %s. "
            "Falling back to Whisper.",
            video_id,
            yt_video_id,
            exc,
        )
        return None


async def fetch_youtube_transcript(
    url: str,
    video_id: Literal["A", "B"],
) -> Optional[TranscriptResult]:
    """Async wrapper for YouTube transcript fetch."""
    return await asyncio.to_thread(_fetch_youtube_transcript_sync, url, video_id)


# ─────────────────────────────────────────────
# TRANSCRIPT EXTRACTION — WHISPER FALLBACK
# ─────────────────────────────────────────────


def _download_audio_sync(
    url: str,
    cookies_path: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
) -> str:
    """
    Download audio to a temp file via yt-dlp.

    Returns the path to the downloaded .wav file.

    Why WAV instead of MP3?
    - Whisper works with WAV natively — no transcoding overhead on the API side.
    - WAV is lossless, so we get slightly better transcription accuracy.
    - File size is larger but we're uploading to an API, not storing long-term.

    Why tempfile?
    - We don't want to pollute the working directory.
    - The caller is responsible for cleanup after Whisper processes the file.
    """
    tmp_dir = tempfile.mkdtemp(prefix="rag_audio_")
    output_template = os.path.join(tmp_dir, "audio.%(ext)s")

    opts = _yt_dlp_base_opts(cookies_path=cookies_path, cookies_from_browser=cookies_from_browser, audio_only=True)
    opts["outtmpl"] = output_template

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        # Retry with a simpler format selector if the default fails.
        # This handles cases where Chrome cookies cause YouTube to return
        # format lists that don't match 'bestaudio/best'.
        logger.warning(
            "Audio download failed with bestaudio format, retrying with 'best': %s",
            exc,
        )
        opts["format"] = "best"
        opts.pop("postprocessors", None)  # skip FFmpeg conversion on fallback
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

    # Find the downloaded file — accept any audio/video format.
    # On fallback (format='best'), yt-dlp may download mp4/mkv containers.
    # Whisper API accepts all of these natively.
    for f in Path(tmp_dir).iterdir():
        if f.suffix in (".wav", ".m4a", ".mp3", ".webm", ".ogg", ".mp4", ".mkv", ".opus"):
            logger.info("Downloaded audio: %s (%.1f MB)", f, f.stat().st_size / 1e6)
            return str(f)

    raise FileNotFoundError(f"No audio file found in {tmp_dir} after yt-dlp download.")


# ── Singleton Whisper model ──
# Loaded once on first transcription call, reused forever.
# "base" model is ~150MB — good balance of speed/quality for short reels.
# On CPU: ~30s per 1-minute reel. On GPU: ~3s.
_whisper_model: WhisperModel | None = None


def _get_whisper_model() -> WhisperModel:
    """Lazy-load the Whisper model (singleton pattern)."""
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading local Whisper model (base)... first run downloads ~150MB")
        _whisper_model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8",  # Quantized — 2x faster on CPU, minimal quality loss
        )
        logger.info("Whisper model loaded successfully")
    return _whisper_model


def _transcribe_with_whisper_sync(
    audio_path: str,
    video_id: Literal["A", "B"],
) -> TranscriptResult:
    """
    Transcribe an audio file using local faster-whisper (FREE, no API key).

    Why faster-whisper instead of the paid OpenAI Whisper API?
    - Zero cost — no OPENAI_API_KEY needed
    - 4x faster than original openai-whisper on CPU (CTranslate2 backend)
    - int8 quantization gives another 2x speedup with <1% quality loss
    - Model downloads once (~150MB), cached in ~/.cache/huggingface/
    - Works fully offline after first download

    Trade-off: ~30s per 1-minute reel on CPU. For production at 1000 creators/day,
    scale vertically (GPU instance) or horizontally (worker queue + GPU pool).
    """
    model = _get_whisper_model()

    # Transcribe with word-level timestamps
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,  # Beam search for better accuracy
        vad_filter=True,  # Voice Activity Detection — skips silence, 2x faster
    )

    chunks: list[TranscriptChunk] = []

    for segment in segments:
        text = segment.text.strip()
        if text:  # Skip empty segments
            chunks.append(
                TranscriptChunk(
                    video_id=video_id,
                    text=text,
                    start=segment.start,
                    end=segment.end,
                )
            )

    # If no segments found, create a single empty chunk
    if not chunks:
        chunks.append(
            TranscriptChunk(
                video_id=video_id,
                text="[No speech detected]",
                start=0.0,
                end=0.0,
            )
        )

    logger.info(
        "Local Whisper transcribed video %s: %d segments, language=%s (%.1fs audio)",
        video_id,
        len(chunks),
        info.language,
        info.duration,
    )

    return TranscriptResult(
        video_id=video_id,
        source="local_whisper",
        language=info.language,
        chunks=chunks,
    )


async def transcribe_with_whisper(
    url: str,
    video_id: Literal["A", "B"],
    cookies_path: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
) -> TranscriptResult:
    """
    Full Whisper pipeline: download audio → transcribe → cleanup.

    This is the universal fallback that works for both YouTube (when captions
    are disabled) and Instagram (which never has native captions).
    """
    audio_path: Optional[str] = None
    try:
        audio_path = await asyncio.to_thread(
            _download_audio_sync, url, cookies_path, cookies_from_browser
        )
        result = await asyncio.to_thread(
            _transcribe_with_whisper_sync, audio_path, video_id
        )
        return result
    finally:
        # Always clean up the temp audio file
        if audio_path and Path(audio_path).exists():
            parent = Path(audio_path).parent
            try:
                Path(audio_path).unlink()
                parent.rmdir()
                logger.debug("Cleaned up temp audio: %s", audio_path)
            except OSError:
                logger.warning("Failed to clean up temp audio: %s", audio_path)


# ─────────────────────────────────────────────
# ORCHESTRATOR — SINGLE VIDEO PIPELINE
# ─────────────────────────────────────────────


async def extract_single_video(
    url: str,
    video_id: Literal["A", "B"],
    cookies_path: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
) -> tuple[VideoExtractionResult, list[str]]:
    """
    Full extraction pipeline for a single video.

    Returns (result, warnings) — we never raise; we collect errors.

    Pipeline:
    1. Detect platform
    2. Extract metadata (via yt-dlp)
    3. Extract transcript:
       - YouTube: try native API first, fall back to Whisper
       - Instagram: go straight to Whisper (no native captions)
    4. Compute engagement rate (done inside metadata extraction)

    Metadata and transcript are extracted sequentially for a single video
    because the transcript fallback path depends on metadata (we need to know
    the platform). However, the TWO videos are extracted in parallel from
    the caller (main.py) via asyncio.gather().
    """
    warnings: list[str] = []

    # Step 1: Platform detection
    platform = detect_platform(url)
    logger.info("Detected platform '%s' for video %s", platform, video_id)

    # Step 2: Metadata extraction
    try:
        metadata = await extract_metadata(url, video_id, platform, cookies_path, cookies_from_browser)
    except Exception as exc:
        logger.error("Metadata extraction failed for video %s: %s", video_id, exc)
        warnings.append(f"[Video {video_id}] Metadata extraction failed: {exc}")
        # Create a skeleton metadata so we can still return something
        metadata = VideoMetadata(
            video_id=video_id,
            platform=platform,
            source_url=url,
        )

    # Step 3: Transcript extraction
    transcript: Optional[TranscriptResult] = None

    if platform == "youtube":
        # Try native captions first
        try:
            transcript = await fetch_youtube_transcript(url, video_id)
        except Exception as exc:
            logger.warning(
                "YouTube transcript API failed for video %s: %s", video_id, exc
            )
            warnings.append(
                f"[Video {video_id}] YouTube captions unavailable: {exc}"
            )

        # Fallback to Whisper if native captions failed
        if transcript is None:
            logger.info(
                "Falling back to Whisper for YouTube video %s", video_id
            )
            try:
                transcript = await transcribe_with_whisper(
                    url, video_id, cookies_path, cookies_from_browser
                )
            except Exception as exc:
                logger.error(
                    "Whisper fallback failed for video %s: %s", video_id, exc
                )
                warnings.append(
                    f"[Video {video_id}] Whisper transcription failed: {exc}"
                )

    elif platform == "instagram":
        # Instagram never has native captions — go straight to Whisper
        try:
            transcript = await transcribe_with_whisper(
                url, video_id, cookies_path
            )
        except Exception as exc:
            logger.error(
                "Whisper transcription failed for Instagram video %s: %s",
                video_id,
                exc,
            )
            warnings.append(
                f"[Video {video_id}] Instagram transcription failed: {exc}"
            )

    # If transcript is still None, create an empty one
    if transcript is None:
        transcript = TranscriptResult(
            video_id=video_id,
            source="whisper_fallback",
            chunks=[],
        )
        warnings.append(
            f"[Video {video_id}] No transcript could be extracted."
        )

    result = VideoExtractionResult(metadata=metadata, transcript=transcript)
    return result, warnings
