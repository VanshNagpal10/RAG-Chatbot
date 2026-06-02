---
title: CHROMA Video RAG Analytics
sdk: docker
app_port: 7860
---

# CHROMA Video RAG Analytics

CHROMA is a full-stack RAG chatbot for comparing two social media videos.

The app takes two video URLs, extracts their metadata and transcripts, stores
the transcript chunks in a vector database, and lets a creator chat with the
videos. The main goal is to help answer practical creator questions like:

- Why did Video A perform better than Video B?
- What is the engagement rate of each video?
- How do the hooks compare in the first few seconds?
- Who is the creator of Video B, and how many followers do they have?
- What can Video B improve based on what worked in Video A?

The project is built to be simple to use from the frontend, but the backend does
the heavier work: video extraction, transcript fallback, chunking, embedding,
retrieval, and streaming LLM responses.

## Project Links

- Deployed app: `https://vansh1027-chroma-rag-backend.hf.space`
- GitHub repo: `https://github.com/VanshNagpal10/RAG-Chatbot`

## What The App Does

The workflow is:

1. The user pastes two video URLs.
2. The backend labels them as `Video A` and `Video B`.
3. The backend detects whether each URL is from YouTube or Instagram.
4. Metadata is extracted using `yt-dlp`.
5. Transcripts are fetched.
6. If a YouTube transcript is not available, the app falls back to local
   `faster-whisper`.
7. Instagram Reels are transcribed with local `faster-whisper`.
8. Engagement rate is calculated for each video.
9. Transcripts are chunked into small text pieces.
10. Every chunk is embedded using a local BGE embedding model.
11. Chunks are stored in ChromaDB with rich metadata and a `video_id` tag.
12. The creator chats with the videos through a streaming RAG interface.
13. Answers include cited sources, such as the video ID and chunk used.

## Task Requirements Covered

| Requirement | How It Is Implemented |
| --- | --- |
| Take two social media video URLs | The frontend accepts exactly two URLs, labeled Video A and Video B. |
| Support YouTube and Instagram Reels | The backend detects `youtube.com`, `youtu.be`, and `instagram.com` URLs. |
| Pull transcript and metadata | Metadata comes from `yt-dlp`; transcripts come from YouTube captions or `faster-whisper`. |
| Extract views, likes, comments, creator, followers, hashtags, upload date, duration | These fields are mapped into the `VideoMetadata` schema. |
| Compute engagement rate | `(likes + comments) / views * 100`, with safe handling for missing views. |
| Chunk transcripts | LangChain `RecursiveCharacterTextSplitter` chunks transcript text. |
| Embed transcripts | `BAAI/bge-small-en-v1.5` runs locally through `HuggingFaceEmbeddings`. |
| Store in vector DB | ChromaDB is used as the persistent vector store. |
| Tag chunks with `video_id` | Every chunk contains `video_id` as metadata, either `A` or `B`. |
| RAG chat interface | LangChain prompt chains retrieve context and generate answers with Groq. |
| Stream responses | FastAPI returns Server-Sent Events, and the frontend parses tokens as they arrive. |
| Cite sources | The backend sends source chunks after the answer finishes streaming. |
| Maintain memory | The frontend sends chat history, and the backend uses recent messages for query reformulation. |
| Side-by-side video cards plus chat panel | The Next.js frontend shows both video cards and a chat panel in one workspace. |

## Core Features

### Two-Video Comparison

The app is built around comparing two videos, not just analyzing one video in
isolation. Each video is given a stable ID:

- `A` for the first video
- `B` for the second video

This makes the RAG system more reliable because every transcript chunk, source,
and metadata record can be traced back to the correct video.

### Metadata Extraction

For each video, the backend extracts:

- Platform
- Source URL
- Title
- Creator or uploader
- Follower count, when available
- Views
- Likes
- Comments
- Hashtags
- Upload date
- Duration
- Engagement rate

The metadata is not only shown in the UI. It is also attached to every vector DB
chunk, so the LLM can answer questions about stats without needing a separate
database lookup.

### Engagement Rate Calculation

Engagement rate is calculated as:

```text
(likes + comments) / views * 100
```

If views are missing, the app does not fake the number. It keeps the engagement
rate unavailable and lets the frontend show `N/A`.

### Transcript Extraction

The transcript pipeline is designed to handle the messy reality of social video:

- YouTube videos are checked for captions first.
- If captions are unavailable, the app falls back to local Whisper transcription.
- Instagram Reels do not expose normal captions, so the app uses Whisper after
  downloading the audio.

The local transcription model is `faster-whisper`, which gives free local speech
to text without needing a paid transcription API.

### Chunking And Embedding

Once the transcript is available, the backend turns it into useful RAG context.

The chunking setup is:

- Chunk size: `500` characters
- Chunk overlap: `50` characters
- Splitter: LangChain `RecursiveCharacterTextSplitter`

Each chunk includes:

- Transcript text
- Timestamp markers like `[00:05]`
- `video_id`
- Chunk index
- Title
- Creator
- Platform
- Views
- Likes
- Comments
- Engagement rate
- Follower count
- Hashtags
- Transcript source

This makes retrieval much more useful. When the user asks about a hook, the
system can retrieve early timestamped chunks. When the user asks about metrics,
the retrieved chunks already contain those metrics.

### Local Embeddings

The app uses:

```text
BAAI/bge-small-en-v1.5
```

This model runs locally on CPU and creates 384-dimensional embeddings. It is a
good fit for this project because it is small, fast, and free to run.

The embedding model is loaded once and reused through a singleton service, so it
does not reload on every request.

### ChromaDB Vector Store

The vector database is ChromaDB.

The collection name is:

```text
rag_chat_transcripts
```

The vector store is persistent locally, so during local development the indexed
chunks survive server restarts. Before inserting new chunks for a video, the app
deletes old chunks with the same `video_id`, which prevents duplicate chunks
when the same video is ingested again.

### Balanced Retrieval

A normal similarity search can accidentally retrieve chunks from only one video.
That is bad for a comparison app.

To avoid that, the backend performs balanced search:

- Retrieve relevant chunks from Video A
- Retrieve relevant chunks from Video B
- Merge them together before sending context to the LLM

This helps the model compare both videos instead of over-focusing on whichever
one has more similar wording.

### Streaming RAG Chat

The chat endpoint streams responses with Server-Sent Events.

The stream sends:

- `token` events while the answer is being generated
- `sources` event with retrieved chunks
- `done` event when the answer is complete
- `error` event if generation fails

This makes the frontend feel responsive because users see the answer appear
token by token instead of waiting for the whole response.

### Chat Memory

The frontend sends previous chat messages with each new question. The backend
uses recent history to reformulate follow-up questions into standalone search
queries.

For example:

```text
User: Why did A perform better?
User: What about the first 5 seconds?
```

The backend can understand that the second question is still about comparing
Video A and Video B.

### Source Citations

After streaming the answer, the backend sends source chunks to the frontend.
The UI shows clickable source badges such as:

```text
Video A - Chunk 3
Video B - Chunk 1
```

This keeps the answer grounded in the actual retrieved transcript chunks.

## Tech Stack

### Frontend

- Next.js
- React
- TypeScript
- Tailwind CSS
- Server-Sent Events parsing for streaming chat

The frontend is intentionally practical and fast. It has two main areas:

- Video cards on the left
- RAG chat panel on the right

### Backend

- FastAPI
- Pydantic
- Uvicorn
- `yt-dlp`
- `youtube-transcript-api`
- `faster-whisper`
- LangChain
- ChromaDB
- Hugging Face sentence transformers
- Groq LLM API

### Models

- Embeddings: `BAAI/bge-small-en-v1.5`
- Transcription: `faster-whisper` `base`
- Chat model: Groq `llama-3.3-70b-versatile`

### Deployment

- Docker
- Hugging Face Spaces
- Free CPU hardware

## Architecture

```text
User
  |
  v
Next.js frontend
  |
  | /api/ingest
  v
FastAPI backend
  |
  |-- detect platform
  |-- extract metadata with yt-dlp
  |-- fetch captions or transcribe audio
  |-- calculate engagement rate
  |-- chunk transcript
  |-- embed chunks with BGE
  |-- store chunks in ChromaDB
  |
  v
Frontend displays video cards
  |
  | /api/chat
  v
FastAPI RAG chat
  |
  |-- reformulate question with memory
  |-- retrieve balanced chunks from Video A and Video B
  |-- generate answer with Groq
  |-- stream tokens and sources back to UI
  v
User sees answer with citations
```

## Folder Structure

```text
.
├── app
│   ├── main.py          # FastAPI routes and app lifecycle
│   ├── services.py      # Video metadata, transcript, and Whisper pipeline
│   ├── vector_store.py  # Chunking, embeddings, ChromaDB, retrieval
│   ├── chat.py          # LangChain RAG chat and streaming logic
│   ├── schemas.py       # Pydantic request and response models
│   └── __init__.py
├── frontend
│   ├── src
│   │   ├── app
│   │   │   ├── page.tsx
│   │   │   ├── layout.tsx
│   │   │   └── globals.css
│   │   ├── components
│   │   │   ├── ChatPanel.tsx
│   │   │   └── VideoCard.tsx
│   │   └── lib
│   │       ├── types.ts
│   │       └── utils.ts
│   ├── package.json
│   └── next.config.ts
├── Dockerfile
├── requirements.txt
├── scripts
│   └── start-space.sh
└── README.md
```

## API Overview

### `GET /health`

Simple health check.

### `POST /extract`

Extracts metadata and transcripts for two videos without storing them in the
vector database.

### `POST /ingest`

Runs the full pipeline:

```text
extract -> chunk -> embed -> store
```

This is the main endpoint used by the frontend when the user clicks
`Process Videos`.

### `POST /chat`

Streams a RAG answer using the ingested video chunks.

### `GET /vector-store/stats`

Returns basic information about the current ChromaDB collection.

## Environment Variables

Create a `.env` file for local development:

```env
GROQ_API_KEY=gsk_your_key_here
LOG_LEVEL=INFO
CHROMA_PERSIST_DIR=./chroma_db

# Optional for local development when a platform requires login/session access.
# This reads cookies from your local browser profile.
COOKIES_FROM_BROWSER=chrome

# Alternative local cookie file option.
# COOKIES_PATH=./cookies.txt
```

Important:

- `GROQ_API_KEY` is required for the chat endpoint.
- BGE embeddings run locally, so no embedding API key is needed.
- `faster-whisper` runs locally, so no transcription API key is needed.
- Browser cookies are useful for local development, but they should be handled
  carefully because cookies can contain login session data.

## Local Development

### 1. Start The Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend runs at:

```text
http://localhost:8000
```

### 2. Start The Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at:

```text
http://localhost:3000
```

The frontend calls `/api/*`. During development, `frontend/next.config.ts`
rewrites those requests to the FastAPI backend on port `8000`.

## Hugging Face Deployment

This repo is configured as a Docker Space.

The metadata at the top of this README tells Hugging Face:

```yaml
sdk: docker
app_port: 7860
```

The Docker container runs:

- FastAPI internally on port `8000`
- Next.js on the exposed Space port `7860`

### Required Space Settings

In Hugging Face Spaces, add:

```text
Secret:
GROQ_API_KEY = your_groq_key
```

Optional variables:

```text
LOG_LEVEL = INFO
CHROMA_PERSIST_DIR = /home/node/app/chroma_db
```

### Push To Hugging Face

```bash
git remote add space https://huggingface.co/spaces/<your-user>/<your-space>
git push space HEAD:main
```

If the Space already has a starter commit, you may need:

```bash
git fetch space main
git push --force-with-lease space HEAD:main
```

## Notes About Hosted Video Extraction

The app works best locally because local browser cookies can help `yt-dlp`
access videos that platforms may otherwise block.

On a hosted service like Hugging Face Spaces, the backend runs on a remote
server. It cannot automatically read the user's browser cookies. That is normal
browser security behavior.

For a public production app, the safer path would be:

- Avoid personal browser cookies.
- Prefer public videos with captions.
- Add clear limits for long videos.
- Move transcription to a queue and worker system.
- Use GPU workers or a managed transcription API for scale.
- Store vectors in a managed vector DB if persistence is required.

## Performance Choices

Several parts of the backend are written to avoid unnecessary work:

- Embedding model is loaded once and reused.
- Whisper model is loaded lazily and reused.
- ChromaDB service is a singleton.
- CPU-heavy embedding and search work is moved off the async event loop with
  `asyncio.to_thread`.
- Ingestion is idempotent, so reprocessing the same video replaces old chunks
  instead of duplicating them.
- The frontend uses streaming so the user sees answers as they are generated.

## Limitations

This is a strong demo and prototype, but there are real-world platform limits:

- YouTube and Instagram can block anonymous server-side requests.
- Long videos can be slow to transcribe on free CPU hardware.
- Free Hugging Face Spaces do not provide reliable persistent disk for ChromaDB.
- Instagram extraction may require authentication depending on the Reel.
- For production, video processing should move to background jobs instead of a
  single long HTTP request.

## Future Improvements

Good next steps would be:

- Add a background job queue for ingestion.
- Add upload/progress status for long-running video processing.
- Add a managed vector database such as Qdrant, Pinecone, or Weaviate.
- Add user accounts and saved comparison sessions.
- Add a safer transcript provider for production use.
- Add better fallback behavior when captions are unavailable.
- Add tests around URL validation, metadata extraction, chunking, and retrieval.

## Summary

CHROMA is a practical RAG application for creators. It combines video metadata,
transcripts, vector search, and streaming chat into one workflow. The app is
designed so a creator can paste two videos, compare their performance, ask
natural questions, and get grounded answers with citations from the actual
transcript chunks.
