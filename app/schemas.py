"""
schemas.py — Pydantic V2 models for the data extraction pipeline.

Design decisions:
- Every field that *might* be unavailable from a scrape is Optional with a sensible default.
  Social platforms are notoriously inconsistent: Instagram might not expose follower_count
  through yt-dlp, YouTube might not expose it for topic channels, etc.
- engagement_rate is computed downstream (in services.py), NOT accepted from the client.
- TranscriptChunk carries its own video_id so chunks are self-describing once they leave
  this layer and enter the vector DB in later sprints.
- We use Literal["A", "B"] instead of a free-form str to enforce the two-video constraint
  at the type level. If a chunk says video_id="C", Pydantic rejects it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, model_validator


# ─────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────


class VideoInput(BaseModel):
    """A single video URL with its assigned label."""

    url: HttpUrl
    video_id: Literal["A", "B"]


class ExtractionRequest(BaseModel):
    """
    Top-level request body for the /extract endpoint.
    Exactly two videos — one labeled A, one labeled B.
    """

    videos: list[VideoInput] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Exactly two video inputs labeled A and B.",
    )

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "ExtractionRequest":
        """Guarantee we have exactly one A and one B — no duplicates."""
        ids = {v.video_id for v in self.videos}
        if ids != {"A", "B"}:
            raise ValueError(
                "Must provide exactly one video with video_id='A' and one with video_id='B'."
            )
        return self


# ─────────────────────────────────────────────
# METADATA MODELS
# ─────────────────────────────────────────────


class VideoMetadata(BaseModel):
    """
    Normalized metadata common to both YouTube and Instagram.

    Fields are Optional because extraction is best-effort:
    - Instagram might not expose follower_count via yt-dlp
    - Some YouTube videos hide like counts
    - Hashtags may not exist on every video

    We never want a missing optional field to crash the pipeline.
    """

    video_id: Literal["A", "B"]
    platform: Literal["youtube", "instagram"]
    source_url: str
    title: Optional[str] = None
    creator: Optional[str] = None
    follower_count: Optional[int] = None
    views: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    hashtags: list[str] = Field(default_factory=list)
    upload_date: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Computed downstream — never accepted from raw extraction
    engagement_rate: Optional[float] = None


# ─────────────────────────────────────────────
# TRANSCRIPT MODELS
# ─────────────────────────────────────────────


class TranscriptChunk(BaseModel):
    """
    A single timestamped segment of a transcript.

    Why separate start/end instead of a single timestamp?
    Because downstream RAG retrieval benefits from knowing the *span* of a chunk —
    it lets the frontend highlight the exact video segment in the player.
    """

    video_id: Literal["A", "B"]
    text: str
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")


class TranscriptResult(BaseModel):
    """Full transcript for a single video."""

    video_id: Literal["A", "B"]
    source: Literal["youtube_api", "whisper_fallback"]
    language: Optional[str] = None
    chunks: list[TranscriptChunk]


# ─────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────


class VideoExtractionResult(BaseModel):
    """Complete extraction result for a single video."""

    metadata: VideoMetadata
    transcript: TranscriptResult


class ExtractionResponse(BaseModel):
    """
    Top-level response from the /extract endpoint.
    Always contains exactly two results (A and B), even if one partially failed.
    """

    results: list[VideoExtractionResult]
    errors: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings or partial failures encountered during extraction.",
    )


# ─────────────────────────────────────────────
# INGESTION MODELS (Day 2)
# ─────────────────────────────────────────────


class IngestionStats(BaseModel):
    """Stats for a single video's ingestion into the vector store."""

    video_id: Literal["A", "B"]
    chunks_stored: int
    title: Optional[str] = None
    platform: Optional[str] = None
    metadata: Optional[VideoMetadata] = Field(
        default=None,
        description="Full extracted metadata for the video (views, likes, engagement, etc.).",
    )



class IngestionResponse(BaseModel):
    """
    Response from the /ingest endpoint.
    Covers the full pipeline: extract → chunk → embed → store.
    """

    videos: list[IngestionStats]
    total_chunks: int
    collection_stats: dict = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────
# CHAT MODELS (Day 3)
# ─────────────────────────────────────────────


class ChatMessage(BaseModel):
    """A single message in chat history (OpenAI-compatible format)."""

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """
    Request body for the /chat endpoint.

    The frontend sends:
    {
        "query": "Why did Video A get more engagement?",
        "chat_history": [
            {"role": "user", "content": "What's the engagement rate of A?"},
            {"role": "assistant", "content": "Video A has an engagement rate of 5.5%"}
        ]
    }

    chat_history is optional — first message in a conversation won't have any.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question about the videos.",
    )
    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous messages in the conversation.",
    )

