---
title: CHROMA Video RAG Analytics
sdk: docker
app_port: 7860
---

# CHROMA Video RAG Analytics

Compare two YouTube or Instagram videos, extract metadata and transcripts, embed
the transcripts with local BGE embeddings, store them in ChromaDB, and chat over
the resulting RAG context with Groq.

## Architecture

- `app/`: FastAPI backend with `yt-dlp`, `faster-whisper`, BGE embeddings, ChromaDB, and Groq chat streaming.
- `frontend/`: Next.js UI. It calls `/api/*`, and `frontend/next.config.ts` rewrites those calls to the backend.
- `Dockerfile`: Hugging Face Docker Space deployment. It runs FastAPI on port `8000` internally and Next on the exposed Space port `7860`.

## Local Development

Backend:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## Deploy On Hugging Face Spaces For Free

Create a new Space and choose **Docker** as the SDK. Push this repository to the
Space repo; the YAML block at the top of this README tells Hugging Face to build
the Docker image and expose port `7860`.

Set these in the Space **Settings**:

- Secret: `GROQ_API_KEY`
- Variable: `LOG_LEVEL=INFO`
- Variable: `CHROMA_PERSIST_DIR=/home/node/app/chroma_db`
- Optional secret: `COOKIES_TXT` with the full Netscape-format `cookies.txt`
  content if YouTube or Instagram requires cookies.

Then push:

```bash
git remote add space https://huggingface.co/spaces/<your-user>/<your-space>
git push space main
```

Free Spaces use CPU, so Whisper fallback is expected to be slow. YouTube videos
with captions will be much faster because the app uses captions before falling
back to Whisper.

## Deployment Notes

- Do not commit `.env` or `cookies.txt`.
- ChromaDB data is local to the running Space container. On free hardware it can
  disappear when the Space sleeps, restarts, or rebuilds.
- `COOKIES_FROM_BROWSER=chrome` is for local development only. Hosted Spaces do
  not have your browser profile, so use `COOKIES_TXT` instead.
- The Docker build pre-downloads `BAAI/bge-small-en-v1.5` and faster-whisper's
  `base` model so the first user request does not pay the model download cost.

## Health Check

The backend health endpoint is available internally at:

```bash
curl http://localhost:8000/health
```

In the deployed app, the browser hits Next on `7860`; Next proxies `/api/*` to
FastAPI.
