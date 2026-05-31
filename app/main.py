"""
main.py — FastAPI application entry point.

Responsibilities:
- Single /extract endpoint that accepts two video URLs
- Parallel extraction of both videos via asyncio.gather()
- Health check endpoint for infra monitoring
- Structured logging configuration
- Clean error responses (never raw tracebacks to the client)

Why FastAPI?
- Native async support (critical for our threaded extraction pipeline)
- Pydantic V2 integration for request/response validation
- Auto-generated OpenAPI docs at /docs
- Minimal boilerplate
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ExtractionRequest, ExtractionResponse
from app.services import extract_single_video

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Load .env before anything else touches os.environ
load_dotenv()

# Configure structured logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# APP LIFECYCLE
# ─────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown lifecycle.

    Currently just validates that required config is present.
    In future sprints this will initialize:
    - Vector DB connections (Pinecone/ChromaDB)
    - LangChain/LangGraph chains
    - Whisper model cache (if using local inference)
    """
    # Validate critical env vars at startup, not at first request
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "sk-your-key-here":
        logger.warning(
            "OPENAI_API_KEY is not set or is the placeholder value. "
            "Whisper transcription will fail. Set it in .env"
        )

    cookies_path = os.getenv("COOKIES_PATH", "./cookies.txt")
    cookies_from_browser = os.getenv("COOKIES_FROM_BROWSER", "").strip() or None

    if cookies_from_browser:
        logger.info(
            "Cookie source: browser (%s). "
            "yt-dlp will auto-extract cookies for YouTube & Instagram.",
            cookies_from_browser,
        )
    elif os.path.exists(cookies_path):
        logger.info("Cookie source: file (%s)", cookies_path)
    else:
        logger.warning(
            "No cookie source configured. Set COOKIES_FROM_BROWSER=chrome in .env "
            "or provide a cookies.txt file. Some videos may fail with bot detection."
        )

    logger.info("RAG Chat extraction service started.")
    yield
    logger.info("RAG Chat extraction service shutting down.")


# ─────────────────────────────────────────────
# APP FACTORY
# ─────────────────────────────────────────────

app = FastAPI(
    title="RAG Chat — Data Extraction API",
    description=(
        "Extracts metadata and transcripts from YouTube and Instagram Reels "
        "for downstream RAG ingestion."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — permissive for dev, lock down in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────


@app.get("/health")
async def health_check():
    """
    Minimal health check for load balancers and monitoring.
    Returns 200 if the process is alive and the event loop isn't blocked.
    """
    return {"status": "healthy", "service": "rag-chat-extraction"}


@app.post(
    "/extract",
    response_model=ExtractionResponse,
    summary="Extract metadata and transcripts from two videos",
    description=(
        "Accepts exactly two video URLs (one YouTube, one Instagram Reel), "
        "extracts metadata and transcripts in parallel, computes engagement rates, "
        "and returns structured results tagged with video_id A and B."
    ),
)
async def extract_videos(request: ExtractionRequest):
    """
    Main extraction endpoint.

    Flow:
    1. Parse and validate the two video URLs (Pydantic handles this)
    2. Launch both extractions concurrently via asyncio.gather()
       - Each extraction is internally async (blocking calls run in threads)
       - If one fails, the other still completes (return_exceptions=False
         is fine because extract_single_video never raises — it collects errors)
    3. Merge results and warnings into a single response
    """
    cookies_path = os.getenv("COOKIES_PATH", "./cookies.txt")
    cookies_from_browser = os.getenv("COOKIES_FROM_BROWSER", "").strip() or None

    try:
        # Launch both extractions in parallel
        # This is where the async architecture pays off — both videos
        # extract simultaneously instead of sequentially.
        results = await asyncio.gather(
            *[
                extract_single_video(
                    url=str(v.url),
                    video_id=v.video_id,
                    cookies_path=cookies_path,
                    cookies_from_browser=cookies_from_browser,
                )
                for v in request.videos
            ]
        )

        # Unpack results and aggregate warnings
        extraction_results = []
        all_warnings: list[str] = []

        for result, warnings in results:
            extraction_results.append(result)
            all_warnings.extend(warnings)

        return ExtractionResponse(
            results=extraction_results,
            errors=all_warnings,
        )

    except Exception as exc:
        # This should rarely fire because extract_single_video catches everything.
        # But defense in depth — we never want a raw 500 with a traceback.
        logger.exception("Unhandled error in /extract endpoint")
        raise HTTPException(
            status_code=500,
            detail=f"Extraction pipeline failed: {str(exc)}",
        ) from exc


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on file changes during development
        log_level=log_level.lower(),
    )
