FROM node:22-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    BACKEND_PORT=8000 \
    CHROMA_PERSIST_DIR=/home/node/app/chroma_db \
    HF_HOME=/home/node/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/node/.cache/huggingface/sentence-transformers

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    libgomp1 \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

USER node
WORKDIR /home/node/app

COPY --chown=node:node requirements.txt ./
RUN python3 -m venv /home/node/venv
ENV PATH=/home/node/venv/bin:$PATH
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY --chown=node:node frontend/package*.json ./frontend/
WORKDIR /home/node/app/frontend
RUN npm ci

WORKDIR /home/node/app
COPY --chown=node:node . .

WORKDIR /home/node/app/frontend
RUN npm run build

WORKDIR /home/node/app
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5'); from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

EXPOSE 7860

CMD ["bash", "scripts/start-space.sh"]
