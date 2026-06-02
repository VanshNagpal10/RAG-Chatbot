from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.schemas import VideoExtractionResult, VideoMetadata, TranscriptResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Chunk size tuned for embedding quality.
# bge-small-en-v1.5 has a 512-token context window (~2000 chars).
# 500 chars is well within that limit.
CHUNK_SIZE = 500

# Overlap prevents semantic loss at chunk boundaries.
# 50 chars ≈ 8-12 words — enough to carry a clause across the boundary.
CHUNK_OVERLAP = 50

# Persistent storage directory for ChromaDB.
# Relative to project root — survives server restarts.
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# Collection name — single collection for all videos.
# We filter by video_id in metadata, not by collection.
CHROMA_COLLECTION_NAME = "rag_chat_transcripts"

# ─── Embedding Model Config ───
# BAAI/bge-small-en-v1.5:
#   - 384 dimensions (compact, fast similarity search)
#   - ~130MB download (cached after first run in ~/.cache/huggingface)
#   - English-optimized, top MTEB scores for its size class
#   - FREE: no API key, no rate limits, no cost per embed
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE models perform better when queries are prefixed with "Represent this sentence:"
# The LangChain HuggingFaceEmbeddings class handles this via encode_kwargs.
EMBEDDING_MODEL_KWARGS = {"device": "cpu"}  # Use "cuda" if GPU available
EMBEDDING_ENCODE_KWARGS = {"normalize_embeddings": True}  # L2 normalize for cosine sim


# ─────────────────────────────────────────────
# TEXT SPLITTER (stateless, reusable)
# ─────────────────────────────────────────────

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    # These separators preserve natural text boundaries.
    # Order matters: try paragraph breaks first, then lines, then words.
    separators=["\n\n", "\n", ". ", " ", ""],
    is_separator_regex=False,
)


# ─────────────────────────────────────────────
# CHUNKING + DOCUMENT CONVERSION
# ─────────────────────────────────────────────


def _build_metadata_dict(metadata: VideoMetadata, chunk_index: int) -> dict:
    """
    Build the metadata dict that every Document chunk carries.

    Why include engagement stats on every chunk?
    Because when the LLM retrieves a chunk to answer "What's the engagement rate
    of Video A?", it needs the stats RIGHT THERE in the chunk metadata — it can't
    make a separate DB query. This is the standard RAG metadata enrichment pattern.
    """
    return {
        # Identity — required for filtering and idempotency
        "video_id": metadata.video_id,
        "chunk_index": chunk_index,
        # Source info — for citations
        "platform": metadata.platform,
        "source_url": metadata.source_url,
        "title": metadata.title or "Unknown",
        "creator": metadata.creator or "Unknown",
        # Engagement stats — so the LLM can answer stats questions
        "views": metadata.views or 0,
        "likes": metadata.likes or 0,
        "comments": metadata.comments or 0,
        "engagement_rate": metadata.engagement_rate or 0.0,
        "follower_count": metadata.follower_count or 0,
        "duration_seconds": metadata.duration_seconds or 0.0,
        # Hashtags as comma-separated string (ChromaDB metadata values must be
        # str, int, float, or bool — no lists allowed)
        "hashtags": ", ".join(metadata.hashtags) if metadata.hashtags else "",
    }


def chunk_transcript(
    extraction_result: VideoExtractionResult,
) -> list[Document]:
    """
    Convert a VideoExtractionResult into LangChain Documents ready for embedding.

    Pipeline:
    1. Concatenate all transcript chunks into a single text with timestamps
    2. Split the concatenated text using RecursiveCharacterTextSplitter
    3. Tag each chunk with rich metadata from the video

    Why concatenate first, then split?
    Because youtube-transcript-api returns chunks of ~2-5 seconds each (very short).
    We want semantic chunks of ~500 chars, which span multiple transcript segments.
    Concatenating first lets the splitter find natural boundaries across segments.

    We also inject timestamp markers into the text so the LLM can reference
    specific moments: "[00:43] Never gonna give you up..."
    """
    metadata = extraction_result.metadata
    transcript = extraction_result.transcript

    if not transcript.chunks:
        logger.warning(
            "No transcript chunks for video %s — creating metadata-only document",
            metadata.video_id,
        )
        # Even with no transcript, create a single document with metadata
        # so stats queries ("What's the engagement rate of A?") still work.
        return [
            Document(
                page_content=(
                    f"Video {metadata.video_id}: {metadata.title or 'Unknown title'} "
                    f"by {metadata.creator or 'Unknown creator'} on {metadata.platform}. "
                    f"Views: {metadata.views or 0}, Likes: {metadata.likes or 0}, "
                    f"Comments: {metadata.comments or 0}, "
                    f"Engagement Rate: {metadata.engagement_rate or 0.0}%."
                ),
                metadata=_build_metadata_dict(metadata, chunk_index=0),
            )
        ]

    # Step 1: Concatenate transcript with timestamp markers
    # Format: [MM:SS] text\n
    lines: list[str] = []
    for chunk in transcript.chunks:
        minutes = int(chunk.start // 60)
        seconds = int(chunk.start % 60)
        timestamp = f"[{minutes:02d}:{seconds:02d}]"
        lines.append(f"{timestamp} {chunk.text}")

    full_text = "\n".join(lines)

    # Step 2: Split into semantically meaningful chunks
    text_chunks = _splitter.split_text(full_text)

    # Step 3: Convert to LangChain Documents with rich metadata
    documents: list[Document] = []
    for i, chunk_text in enumerate(text_chunks):
        # Generate a deterministic ID for this chunk.
        # Used by ChromaDB for deduplication.
        chunk_id = hashlib.sha256(
            f"{metadata.video_id}:{i}:{chunk_text[:50]}".encode()
        ).hexdigest()[:16]

        doc = Document(
            page_content=chunk_text,
            metadata={
                **_build_metadata_dict(metadata, chunk_index=i),
                "chunk_id": chunk_id,
                # Store the source language for potential translation context
                "transcript_language": transcript.language or "unknown",
                "transcript_source": transcript.source,
            },
        )
        documents.append(doc)

    logger.info(
        "Chunked video %s transcript into %d documents "
        "(from %d raw segments, %d total chars)",
        metadata.video_id,
        len(documents),
        len(transcript.chunks),
        len(full_text),
    )

    return documents


# ─────────────────────────────────────────────
# VECTOR STORE SERVICE
# ─────────────────────────────────────────────


class VectorStoreService:
    """
    Manages the ChromaDB vector store with local HuggingFace embeddings.

    Key design decisions:

    1. LOCAL EMBEDDINGS (BAAI/bge-small-en-v1.5):
       The model weights are downloaded once to ~/.cache/huggingface on first use.
       After that, initialization takes ~2-3s (loading weights into RAM).
       We do this ONCE at startup via the singleton pattern, not per-request.

    2. EVENT LOOP PROTECTION:
       Embedding is CPU-bound (matrix multiplication over 384 dimensions).
       If we run it in an async handler, it blocks the event loop and all
       concurrent requests stall. EVERY embedding + DB write operation is
       wrapped in asyncio.to_thread() to run in a separate OS thread.

    3. SINGLETON via get_vector_store_service():
       The embedding model + ChromaDB client are initialized once and shared
       across all requests. No per-request overhead.
    """

    def __init__(self):
        self._store: Optional[Chroma] = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """
        Lazy-init the HuggingFace embedding model on first access.

        First call triggers a ~130MB download (cached permanently after).
        Subsequent calls return the already-loaded model instantly.

        The model runs entirely on CPU — no GPU required.
        For GPU acceleration, change model_kwargs to {"device": "cuda"}.
        """
        if self._embeddings is None:
            logger.info(
                "Loading embedding model: %s (first load may download ~130MB)...",
                EMBEDDING_MODEL_NAME,
            )
            self._embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs=EMBEDDING_MODEL_KWARGS,
                encode_kwargs=EMBEDDING_ENCODE_KWARGS,
            )
            logger.info(
                "Embedding model loaded: %s (384-dim, CPU, normalized)",
                EMBEDDING_MODEL_NAME,
            )
        return self._embeddings

    @property
    def store(self) -> Chroma:
        """
        Lazy-init the ChromaDB persistent store on first access.

        Why persistent storage?
        - Survives server restarts — no re-embedding on every boot
        - Good enough for dev/prototyping
        - Swap to a managed service (Pinecone, Weaviate) for production
        """
        if self._store is None:
            persist_dir = str(Path(CHROMA_PERSIST_DIR).resolve())
            self._store = Chroma(
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=persist_dir,
            )
            logger.info(
                "Initialized ChromaDB store: collection=%s, dir=%s",
                CHROMA_COLLECTION_NAME,
                persist_dir,
            )
        return self._store

    def _purge_video(self, video_id: str) -> int:
        """
        Delete all existing chunks for a given video_id.

        This is the idempotency mechanism: before inserting new chunks,
        we wipe all old ones for the same video. This prevents duplicates
        when re-ingesting and handles cases where chunking parameters changed.

        Returns the number of documents deleted.
        """
        try:
            existing = self.store.get(where={"video_id": video_id})
            if existing and existing["ids"]:
                count = len(existing["ids"])
                self.store.delete(ids=existing["ids"])
                logger.info(
                    "Purged %d existing chunks for video %s", count, video_id
                )
                return count
            return 0
        except Exception as exc:
            logger.warning(
                "Failed to purge existing chunks for video %s: %s",
                video_id,
                exc,
            )
            return 0

    def _sync_ingest(self, documents: list[Document], video_id: str) -> int:
        """
        SYNCHRONOUS embedding + insertion — runs in a worker thread.

        ┌─────────────────────────────────────────────────────────────┐
        │ WHY THIS IS A SEPARATE SYNC METHOD:                        │
        │                                                            │
        │ HuggingFace embedding is CPU-bound (matrix math).          │
        │ If we run it in the async event loop, it blocks ALL        │
        │ concurrent requests for 2-10 seconds.                      │
        │                                                            │
        │ By keeping this synchronous and calling it via             │
        │ asyncio.to_thread() from the async wrapper, the CPU work   │
        │ happens in a separate OS thread. The event loop stays      │
        │ free to handle /health checks, other API calls, etc.       │
        │                                                            │
        │ This is the standard pattern for CPU-bound work in ASGI    │
        │ frameworks (FastAPI, Starlette).                           │
        └─────────────────────────────────────────────────────────────┘
        """
        # Step 1: Idempotency — remove old chunks
        self._purge_video(video_id)

        # Step 2: Generate deterministic IDs
        ids = [
            f"{video_id}_chunk_{doc.metadata.get('chunk_index', i)}"
            for i, doc in enumerate(documents)
        ]

        # Step 3: Embed + insert (this is the CPU-intensive part)
        # add_documents() calls self.embeddings.embed_documents() internally,
        # which runs the BAAI/bge-small-en-v1.5 forward pass on CPU.
        # For ~300 documents, this takes ~3-8 seconds on a modern MacBook.
        self.store.add_documents(documents=documents, ids=ids)

        logger.info(
            "Ingested %d documents for video %s into ChromaDB (local embeddings)",
            len(documents),
            video_id,
        )
        return len(documents)

    async def ingest_documents(
        self,
        documents: list[Document],
        video_id: str,
    ) -> int:
        """
        Async wrapper — offloads CPU-bound embedding to a worker thread.

        This is the public API. Route handlers call this method.
        It delegates to _sync_ingest() via asyncio.to_thread() so the
        FastAPI event loop is never blocked by the embedding computation.

        No batching or rate limiting needed — local model, no API quotas.
        """
        if not documents:
            logger.warning("No documents to ingest for video %s", video_id)
            return 0

        logger.info(
            "Starting ingestion for video %s: %d documents (offloading to thread)...",
            video_id,
            len(documents),
        )

        # asyncio.to_thread() runs _sync_ingest in the default ThreadPoolExecutor.
        # The event loop continues handling other requests while embedding runs.
        count = await asyncio.to_thread(
            self._sync_ingest, documents, video_id
        )

        return count

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        video_id: Optional[str] = None,
    ) -> list[Document]:
        """
        Search for relevant chunks. Optionally filter by video_id.

        Also offloaded to a thread because the embedding model needs to
        encode the query string (CPU-bound) before ChromaDB can search.
        """
        filter_dict = {"video_id": video_id} if video_id else None

        def _sync_search() -> list[Document]:
            return self.store.similarity_search(
                query=query,
                k=k,
                filter=filter_dict,
            )

        return await asyncio.to_thread(_sync_search)

    async def balanced_search(
        self,
        query: str,
        k: int = 8,
    ) -> list[Document]:
        """
        Retrieve k/2 chunks from Video A and k/2 from Video B.

        Why not just similarity_search(k=8)?
        Because cosine similarity doesn't care about balance — if Video B's
        transcript uses more similar vocabulary to the query, ALL 8 results
        could come from Video B. The LLM then says "I don't have info about
        Video A" which is a terrible user experience for a comparison app.

        This method guarantees both videos are always represented.
        """
        per_video = k // 2

        # Fetch from both videos concurrently
        docs_a, docs_b = await asyncio.gather(
            self.similarity_search(query=query, k=per_video, video_id="A"),
            self.similarity_search(query=query, k=per_video, video_id="B"),
        )

        # Interleave: A1, B1, A2, B2, ... (better for LLM context)
        merged = []
        for i in range(max(len(docs_a), len(docs_b))):
            if i < len(docs_a):
                merged.append(docs_a[i])
            if i < len(docs_b):
                merged.append(docs_b[i])

        return merged

    def get_collection_stats(self) -> dict:
        """Return basic stats about what's in the vector store."""
        try:
            collection = self.store._collection
            count = collection.count()
            return {
                "collection_name": CHROMA_COLLECTION_NAME,
                "total_documents": count,
                "persist_directory": str(Path(CHROMA_PERSIST_DIR).resolve()),
                "embedding_model": EMBEDDING_MODEL_NAME,
                "embedding_dimensions": 384,
                "cost_per_embed": "$0.00 (local)",
            }
        except Exception as exc:
            return {"error": str(exc)}


# ─────────────────────────────────────────────
# SINGLETON + FASTAPI DEPENDENCY
# ─────────────────────────────────────────────

# Module-level singleton.
# Created once when the module is first imported, shared across all requests.
_vector_store_service: Optional[VectorStoreService] = None


def get_vector_store_service() -> VectorStoreService:
    """
    FastAPI dependency — returns the singleton VectorStoreService.

    Usage in routes:
        @app.post("/ingest")
        async def ingest(vs: VectorStoreService = Depends(get_vector_store_service)):
            ...

    Why a function instead of just importing the global?
    - FastAPI's Depends() requires a callable
    - Testable — you can override this dependency in tests
    - Lazy initialization — the service isn't created until the first request
    """
    global _vector_store_service
    if _vector_store_service is None:
        _vector_store_service = VectorStoreService()
    return _vector_store_service
