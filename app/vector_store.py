"""
vector_store.py — Chunking, embedding, and vector storage service.

Architecture notes:
─────────────────────────────────────────────────────────────────────
CHUNKING STRATEGY:
  We use RecursiveCharacterTextSplitter with:
  - chunk_size=500: Small enough for precise retrieval (the LLM gets focused
    context), large enough to carry semantic meaning. 500 chars ≈ 75-100 words
    ≈ 3-5 sentences, which is the sweet spot for embedding quality.
  - chunk_overlap=50: Prevents losing context at chunk boundaries. If a sentence
    like "The hook in the first 5 seconds grabbed attention" gets split, the
    overlap ensures both chunks carry enough context for meaningful retrieval.
  - RecursiveCharacterTextSplitter splits on ["\n\n", "\n", " ", ""] in order,
    which preserves paragraph/sentence boundaries better than a naive char split.

DOCUMENT METADATA:
  Every chunk carries rich metadata so the RAG chain can cite sources precisely:
  - video_id: "A" or "B" — required for filtering and comparison queries
  - title, creator, platform: For source citations in responses
  - views, likes, comments, engagement_rate: So the LLM can answer stats questions
    directly from metadata without needing the transcript
  - chunk_index, start_time, end_time: For frontend video player integration

IDEMPOTENCY:
  If you ingest Video A twice, we don't want duplicate chunks in ChromaDB.
  Strategy: Before inserting, delete ALL existing documents with the same video_id.
  This is a "delete-then-insert" pattern — simpler and more reliable than upserting
  individual chunks, because the chunking itself might change (different overlap,
  different text) between runs.

ASYNC INSERTION:
  ChromaDB's LangChain integration provides `aadd_documents()` which runs the
  embedding + insertion in a non-blocking way. Critical because embedding 50+
  chunks via OpenAI's API takes 2-5 seconds — we can't block the event loop.
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.schemas import VideoExtractionResult, VideoMetadata, TranscriptResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Chunk size tuned for embedding quality.
# text-embedding-3-small has a context window of 8191 tokens (~32K chars),
# so 500 chars is well within limits. Smaller chunks = more precise retrieval.
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

# Embedding model — small is fast + cheap + good enough for transcript chunks.
# Upgrade to text-embedding-3-large if retrieval quality becomes an issue.
EMBEDDING_MODEL = "text-embedding-3-small"


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
    Manages the ChromaDB vector store lifecycle.

    Why a class instead of module-level functions?
    - Encapsulates the DB connection and embedding model as state
    - Makes dependency injection straightforward (FastAPI Depends())
    - Testable — you can mock the class or swap to a different vector DB
    - Lazy initialization — the DB connection isn't created until first use

    Why a singleton pattern via get_vector_store_service()?
    - ChromaDB persistent client should be created ONCE per process
    - Embedding model should be initialized ONCE (loads config, validates API key)
    - Multiple route handlers share the same instance
    """

    def __init__(self):
        self._store: Optional[Chroma] = None
        self._embeddings: Optional[OpenAIEmbeddings] = None

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """Lazy-init the embedding model on first access."""
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                # text-embedding-3-small returns 1536-dim vectors by default.
                # We could reduce with `dimensions=512` for speed, but 1536
                # gives better retrieval quality for our use case.
            )
            logger.info("Initialized OpenAI embeddings: %s", EMBEDDING_MODEL)
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
            # ChromaDB's get() with where filter returns matching doc IDs
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

    async def ingest_documents(
        self,
        documents: list[Document],
        video_id: str,
    ) -> int:
        """
        Embed and store documents in ChromaDB.

        Flow:
        1. Purge existing chunks for this video_id (idempotency)
        2. Add new documents via aadd_documents (async, non-blocking)
        3. Return the number of documents stored

        Why aadd_documents instead of add_documents?
        add_documents is synchronous — it calls the OpenAI embedding API
        and writes to ChromaDB sequentially. For 20+ chunks, that's 2-5 seconds
        of blocking. aadd_documents runs this in a thread pool, keeping the
        FastAPI event loop free to handle other requests.
        """
        if not documents:
            logger.warning("No documents to ingest for video %s", video_id)
            return 0

        # Step 1: Idempotency — remove old chunks
        self._purge_video(video_id)

        # Step 2: Generate deterministic IDs for ChromaDB
        # ChromaDB requires unique string IDs. We use video_id + chunk_index
        # so the same content always gets the same ID.
        ids = [
            f"{video_id}_chunk_{doc.metadata.get('chunk_index', i)}"
            for i, doc in enumerate(documents)
        ]

        # Step 3: Async embed + insert
        # aadd_documents calls the OpenAI embedding API and writes to ChromaDB
        # without blocking the event loop.
        await self.store.aadd_documents(documents=documents, ids=ids)

        logger.info(
            "Ingested %d documents for video %s into ChromaDB",
            len(documents),
            video_id,
        )
        return len(documents)

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        video_id: Optional[str] = None,
    ) -> list[Document]:
        """
        Search for relevant chunks. Optionally filter by video_id.

        This will be used by the RAG chain in Day 3.
        Exposed here so the chain doesn't need to know about ChromaDB internals.
        """
        filter_dict = {"video_id": video_id} if video_id else None
        results = await self.store.asimilarity_search(
            query=query,
            k=k,
            filter=filter_dict,
        )
        return results

    def get_collection_stats(self) -> dict:
        """Return basic stats about what's in the vector store."""
        try:
            collection = self.store._collection
            count = collection.count()
            return {
                "collection_name": CHROMA_COLLECTION_NAME,
                "total_documents": count,
                "persist_directory": str(Path(CHROMA_PERSIST_DIR).resolve()),
                "embedding_model": EMBEDDING_MODEL,
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
