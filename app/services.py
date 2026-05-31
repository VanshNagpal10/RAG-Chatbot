"""
services.py — Core extraction logic for YouTube and Instagram videos.

Architecture notes:
─────────────────────────────────────────────────────────────────────
ASYNC STRATEGY:
  yt-dlp, youtube-transcript-api, and the OpenAI Whisper client are all synchronous.
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
  3. yt-dlp audio download → Whisper API       →  slowest, but always works

INSTAGRAM RESILIENCE:
  Meta's CDN is hostile to scrapers. We mitigate with:
  - cookies.txt from an authenticated browser session
  - Desktop Chrome User-Agent
  - Audio-only download (saves bandwidth, reduces detection surface)
  - Generous retry/timeout config in yt-dlp

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
from openai import OpenAI
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
        views=info.get("view_count"),
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


def _transcribe_with_whisper_sync(
    audio_path: str,
    video_id: Literal["A", "B"],
) -> TranscriptResult:
    """
    Transcribe an audio file using OpenAI's Whisper API.

    Why the OpenAI API instead of local faster-whisper?
    - No GPU dependency — works on any machine.
    - No model download (faster-whisper's large-v3 is ~3GB).
    - Consistent quality regardless of hardware.
    - Trade-off: costs money and requires network. For a dev project, this is fine.
      Swap to faster-whisper for production/air-gapped environments.

    We use the verbose_json response format to get word-level timestamps,
    then aggregate into segment-level chunks.
    """
    client = OpenAI()  # Reads OPENAI_API_KEY from env

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    chunks: list[TranscriptChunk] = []

    # verbose_json returns segments with start/end times
    if hasattr(response, "segments") and response.segments:
        for seg in response.segments:
            chunks.append(
                TranscriptChunk(
                    video_id=video_id,
                    text=seg.get("text", "").strip() if isinstance(seg, dict) else seg.text.strip(),
                    start=seg.get("start", 0.0) if isinstance(seg, dict) else seg.start,
                    end=seg.get("end", 0.0) if isinstance(seg, dict) else seg.end,
                )
            )
    else:
        # Fallback: if no segments, use full text as single chunk
        chunks.append(
            TranscriptChunk(
                video_id=video_id,
                text=response.text.strip(),
                start=0.0,
                end=0.0,
            )
        )

    logger.info(
        "Whisper transcribed video %s: %d segments, language=%s",
        video_id,
        len(chunks),
        getattr(response, "language", "unknown"),
    )

    return TranscriptResult(
        video_id=video_id,
        source="whisper_fallback",
        language=getattr(response, "language", None),
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
