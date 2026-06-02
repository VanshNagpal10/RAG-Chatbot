from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from app.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query reformulator. Given the chat history and the latest "
            "user question, reformulate it into a standalone question that can be "
            "understood WITHOUT the chat history. Do NOT answer the question — "
            "just reformulate it. If it's already standalone, return it as-is.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


QA_SYSTEM_PROMPT = """\
You are an expert social media analyst. Your job is to compare Video A and Video B \
using ONLY the provided context below.

RULES:
1. Base ALL your answers strictly on the provided context. Never make up information.
2. If asked about engagement rates, views, likes, or comments — use the metadata \
values shown for each chunk (video_id, views, likes, engagement_rate, etc.).
3. If asked about the first 5 seconds or the "hook" of a video, look explicitly at \
the earliest timestamps ([00:00], [00:01], etc.) in the context chunks.
4. When comparing videos, clearly label which is Video A and which is Video B.
5. Always cite your sources — mention the video ID and timestamp when referencing \
specific content from the transcripts.
6. If the answer is NOT in the context, say "I don't have enough information in the \
provided transcripts to answer that."

CONTEXT:
{context}
"""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

def _convert_chat_history(
    raw_history: list[dict],
) -> list[HumanMessage | AIMessage]:
    """
    Convert frontend chat history to LangChain message objects.

    Frontend: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    LangChain: [HumanMessage("..."), AIMessage("...")]

    We cap at 10 messages to keep the context window manageable.
    """
    MAX_HISTORY = 10
    messages = []

    for msg in raw_history[-MAX_HISTORY:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            messages.append(AIMessage(content=content))

    return messages


def _format_docs_as_context(docs: list[Document]) -> str:
    if not docs:
        return "No relevant documents found."

    parts = []
    for doc in docs:
        meta = doc.metadata
        header = (
            f"─── Video {meta.get('video_id', '?')} | "
            f"Chunk {meta.get('chunk_index', '?')} | "
            f"Title: {meta.get('title', 'Unknown')} | "
            f"Creator: {meta.get('creator', 'Unknown')} | "
            f"Views: {meta.get('views', 0):,} | "
            f"Likes: {meta.get('likes', 0):,} | "
            f"Comments: {meta.get('comments', 0):,} | "
            f"Engagement Rate: {meta.get('engagement_rate', 0.0)}% ───"
        )
        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n".join(parts)


def _serialize_sources(docs: list[Document]) -> list[dict]:
    sources = []
    seen = set()

    for doc in docs:
        content_key = doc.page_content[:100]
        if content_key in seen:
            continue
        seen.add(content_key)

        meta = doc.metadata
        sources.append(
            {
                "video_id": meta.get("video_id", "?"),
                "title": meta.get("title", "Unknown"),
                "creator": meta.get("creator", "Unknown"),
                "platform": meta.get("platform", "unknown"),
                "views": meta.get("views", 0),
                "likes": meta.get("likes", 0),
                "engagement_rate": meta.get("engagement_rate", 0.0),
                "chunk_index": meta.get("chunk_index", 0),
                "text_preview": doc.page_content[:200],
            }
        )

    return sources


async def stream_rag_response(
    query: str,
    chat_history: list[dict],
    vs: VectorStoreService,
    model_name: str = "llama-3.3-70b-versatile",
) -> AsyncGenerator[str, None]:
   
    # Initialize Groq LLM — reads GROQ_API_KEY from env
    llm = ChatGroq(
        model=model_name,
        streaming=True,
        temperature=0.3,  # Low temp for factual analysis
    )

    lc_history = _convert_chat_history(chat_history)

    try:
        search_query = query
        if lc_history:
            # Use LCEL: prompt | llm | parser
            reformulate_chain = CONTEXTUALIZE_Q_PROMPT | llm | StrOutputParser()
            search_query = await reformulate_chain.ainvoke(
                {"input": query, "chat_history": lc_history}
            )
            logger.info(
                "Reformulated query: '%s' → '%s'", query[:60], search_query[:60]
            )

        source_docs = await vs.balanced_search(
            query=search_query,
            k=8,  # 4 from Video A + 4 from Video B
        )
        logger.info(
            "Retrieved %d chunks for query: '%s'", len(source_docs), search_query[:60]
        )

        context_text = _format_docs_as_context(source_docs)

        # Build the QA prompt with context injected
        qa_messages = await QA_PROMPT.ainvoke(
            {
                "context": context_text,
                "chat_history": lc_history,
                "input": query,  # Use original query, not reformulated
            }
        )

        full_answer = ""

        async for chunk in llm.astream(qa_messages):
            token = chunk.content
            if token:
                full_answer += token
                event = json.dumps({"type": "token", "content": token})
                yield f"data: {event}\n\n"

        sources_payload = _serialize_sources(source_docs)
        sources_event = json.dumps({"type": "sources", "content": sources_payload})
        yield f"data: {sources_event}\n\n"

        # Signal stream completion
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

        logger.info(
            "Chat completed: %d chars, %d sources, query='%s'",
            len(full_answer),
            len(sources_payload),
            query[:60],
        )

    except Exception as exc:
        logger.exception("Error during RAG streaming: %s", exc)
        error_event = json.dumps(
            {"type": "error", "content": f"Generation failed: {str(exc)}"}
        )
        yield f"data: {error_event}\n\n"
