"""
main.py — FastAPI application entry point.

Responsibilities:
- /extract endpoint: Extract metadata and transcripts from two video URLs
- /ingest endpoint: Full pipeline — extract + chunk + embed + store in ChromaDB
- /health endpoint: Health check for monitoring
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
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.schemas import (
    ChatRequest,
    ExtractionRequest,
    ExtractionResponse,
    IngestionResponse,
    IngestionStats,
)
from app.chat import stream_rag_response
from app.services import extract_single_video
from app.vector_store import (
    VectorStoreService,
    chunk_transcript,
    get_vector_store_service,
)

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

    Validates config and initializes the vector store eagerly so the first
    request doesn't pay the cold-start cost.
    """
    # No API key needed for embeddings — BAAI/bge-small-en-v1.5 runs locally.
    # OPENAI_API_KEY is optional — only used for Whisper transcription fallback.
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key != "sk-your-key-here":
        logger.info("OpenAI API key found — Whisper transcription fallback enabled.")
    else:
        logger.info(
            "No OpenAI API key set — Whisper fallback disabled. "
            "Most YouTube videos have captions, so this is fine."
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

    # Eagerly initialize the vector store so /ingest doesn't have cold-start latency
    try:
        vs = get_vector_store_service()
        stats = vs.get_collection_stats()
        logger.info("Vector store ready: %s", stats)
    except Exception as exc:
        logger.warning("Vector store init failed (will retry on first request): %s", exc)

    logger.info("RAG Chat service started.")
    yield
    logger.info("RAG Chat service shutting down.")


# ─────────────────────────────────────────────
# APP FACTORY
# ─────────────────────────────────────────────

app = FastAPI(
    title="RAG Chat — Video Analysis API",
    description=(
        "Extracts metadata and transcripts from YouTube and Instagram Reels, "
        "chunks and embeds them into ChromaDB for downstream RAG retrieval."
    ),
    version="0.2.0",
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
    return {"status": "healthy", "service": "rag-chat"}


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
    Extraction-only endpoint (Day 1).
    Returns raw metadata + transcripts without storing anything.
    """
    cookies_path = os.getenv("COOKIES_PATH", "./cookies.txt")
    cookies_from_browser = os.getenv("COOKIES_FROM_BROWSER", "").strip() or None

    try:
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
        logger.exception("Unhandled error in /extract endpoint")
        raise HTTPException(
            status_code=500,
            detail=f"Extraction pipeline failed: {str(exc)}",
        ) from exc


@app.post(
    "/ingest",
    response_model=IngestionResponse,
    summary="Extract, chunk, embed, and store two videos",
    description=(
        "Full ingestion pipeline: extracts metadata/transcripts from two video URLs, "
        "chunks the transcripts, embeds them via OpenAI text-embedding-3-small, "
        "and stores everything in ChromaDB. Idempotent — re-ingesting the same "
        "video replaces old chunks instead of duplicating them."
    ),
)
async def ingest_videos(
    request: ExtractionRequest,
    vs: VectorStoreService = Depends(get_vector_store_service),
):
    """
    Full pipeline endpoint (Day 2).

    Flow:
    1. Extract metadata + transcripts for both videos (parallel)
    2. Chunk each transcript into LangChain Documents with rich metadata
    3. Embed and store chunks in ChromaDB (async, non-blocking)
    4. Return stats about what was stored

    Idempotency: If Video A was already ingested, its old chunks are deleted
    before the new ones are inserted. No duplicates.
    """
    cookies_path = os.getenv("COOKIES_PATH", "./cookies.txt")
    cookies_from_browser = os.getenv("COOKIES_FROM_BROWSER", "").strip() or None
    all_warnings: list[str] = []

    try:
        # ── Step 1: Extract both videos in parallel ──
        extraction_results_raw = await asyncio.gather(
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

        extraction_results = []
        for result, warnings in extraction_results_raw:
            extraction_results.append(result)
            all_warnings.extend(warnings)

        # ── Step 2: Chunk transcripts into Documents ──
        all_ingestion_stats: list[IngestionStats] = []
        total_chunks = 0

        for extraction_result in extraction_results:
            video_id = extraction_result.metadata.video_id

            try:
                # Chunk the transcript
                documents = chunk_transcript(extraction_result)
                logger.info(
                    "Created %d document chunks for video %s",
                    len(documents),
                    video_id,
                )

                # ── Step 3: Embed and store in ChromaDB ──
                stored_count = await vs.ingest_documents(
                    documents=documents,
                    video_id=video_id,
                )

                all_ingestion_stats.append(
                    IngestionStats(
                        video_id=video_id,
                        chunks_stored=stored_count,
                        title=extraction_result.metadata.title,
                        platform=extraction_result.metadata.platform,
                        metadata=extraction_result.metadata,
                    )
                )
                total_chunks += stored_count

            except Exception as exc:
                logger.error(
                    "Ingestion failed for video %s: %s", video_id, exc
                )
                all_warnings.append(
                    f"[Video {video_id}] Ingestion failed: {exc}"
                )
                all_ingestion_stats.append(
                    IngestionStats(
                        video_id=video_id,
                        chunks_stored=0,
                        title=extraction_result.metadata.title,
                        platform=extraction_result.metadata.platform,
                        metadata=extraction_result.metadata,
                    )
                )

        # ── Step 4: Return results ──
        return IngestionResponse(
            videos=all_ingestion_stats,
            total_chunks=total_chunks,
            collection_stats=vs.get_collection_stats(),
            errors=all_warnings,
        )

    except Exception as exc:
        logger.exception("Unhandled error in /ingest endpoint")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion pipeline failed: {str(exc)}",
        ) from exc


@app.get(
    "/vector-store/stats",
    summary="Get vector store statistics",
)
async def vector_store_stats(
    vs: VectorStoreService = Depends(get_vector_store_service),
):
    """Returns current ChromaDB collection stats."""
    return vs.get_collection_stats()


@app.post(
    "/chat",
    summary="Chat with the RAG pipeline (SSE streaming)",
    description=(
        "Send a question about the ingested videos and receive a streaming response. "
        "The response is streamed as Server-Sent Events (SSE) with three event types: "
        "'token' (LLM output), 'sources' (cited documents), and 'done' (stream end)."
    ),
)
async def chat(
    request: ChatRequest,
    vs: VectorStoreService = Depends(get_vector_store_service),
):
    """
    RAG chat endpoint with SSE streaming.

    Flow:
    1. Reformulate the query using chat history (history-aware retriever)
    2. Search ChromaDB for relevant transcript chunks
    3. Stream the LLM's answer token-by-token as SSE events
    4. Emit source documents as the final SSE event

    SSE event format:
        data: {"type": "token", "content": "The"}
        data: {"type": "token", "content": " engagement"}
        ...
        data: {"type": "sources", "content": [{"video_id": "A", ...}]}
        data: {"type": "done"}

    Requires GOOGLE_API_KEY in .env for the Gemini LLM.
    """
    # Validate that the LLM API key is configured
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        raise HTTPException(
            status_code=500,
            detail=(
                "GOOGLE_API_KEY is not set. The /chat endpoint requires a Gemini API key. "
                "Get a free one at https://aistudio.google.com/apikey"
            ),
        )

    # Check that we have documents in the vector store
    stats = vs.get_collection_stats()
    if stats.get("total_documents", 0) == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents in the vector store. Run /ingest first.",
        )

    # Convert ChatRequest history to the format expected by the streaming generator
    raw_history = [msg.model_dump() for msg in request.chat_history]

    logger.info("Chat query: '%s' (history: %d messages)", request.query[:80], len(raw_history))

    # Return a StreamingResponse with SSE content type.
    # The generator yields SSE-formatted events as the LLM produces tokens.
    return StreamingResponse(
        stream_rag_response(
            query=request.query,
            chat_history=raw_history,
            vs=vs,
        ),
        media_type="text/event-stream",
        headers={
            # Prevent proxy/CDN buffering — SSE must be delivered immediately
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx
        },
    )


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
