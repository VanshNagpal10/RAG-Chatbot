from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, model_validator

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



class VideoMetadata(BaseModel):

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



class TranscriptChunk(BaseModel):
    

    video_id: Literal["A", "B"]
    text: str
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")


class TranscriptResult(BaseModel):
    """Full transcript for a single video."""

    video_id: Literal["A", "B"]
    source: Literal["youtube_api", "whisper_fallback", "local_whisper"]
    language: Optional[str] = None
    chunks: list[TranscriptChunk]



class VideoExtractionResult(BaseModel):
    """Complete extraction result for a single video."""

    metadata: VideoMetadata
    transcript: TranscriptResult


class ExtractionResponse(BaseModel):
    

    results: list[VideoExtractionResult]
    errors: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings or partial failures encountered during extraction.",
    )


class IngestionStats(BaseModel):


    video_id: Literal["A", "B"]
    chunks_stored: int
    title: Optional[str] = None
    platform: Optional[str] = None
    metadata: Optional[VideoMetadata] = Field(
        default=None,
        description="Full extracted metadata for the video (views, likes, engagement, etc.).",
    )



class IngestionResponse(BaseModel):
 

    videos: list[IngestionStats]
    total_chunks: int
    collection_stats: dict = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)



class ChatMessage(BaseModel):


    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
 

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

