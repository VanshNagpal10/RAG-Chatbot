// ─── Backend Response Types ───

export interface VideoMetadata {
  video_id: 'A' | 'B';
  platform: string;
  source_url: string;
  title: string | null;
  creator: string | null;
  views: number | null;
  likes: number | null;
  comments: number | null;
  engagement_rate: number | null;
  follower_count: number | null;
  duration_seconds: number | null;
  hashtags: string[];
  upload_date: string | null;
}

export interface TranscriptChunk {
  video_id: string;
  text: string;
  start: number;
  end: number;
}

export interface TranscriptResult {
  video_id: string;
  source: string;
  language: string | null;
  chunks: TranscriptChunk[];
}

export interface VideoExtractionResult {
  metadata: VideoMetadata;
  transcript: TranscriptResult;
}

export interface ExtractionResponse {
  results: VideoExtractionResult[];
  errors: string[];
}

export interface IngestionStats {
  video_id: 'A' | 'B';
  chunks_stored: number;
  title: string | null;
  platform: string | null;
  metadata: VideoMetadata | null; // Full metadata returned from /ingest
}

export interface IngestionResponse {
  videos: IngestionStats[];
  total_chunks: number;
  collection_stats: Record<string, unknown>;
  errors: string[];
}

// ─── Chat Types ───

export interface SourceDoc {
  video_id: string;
  title: string;
  creator: string;
  platform: string;
  views: number;
  likes: number;
  engagement_rate: number;
  chunk_index: number;
  text_preview: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceDoc[];
  isStreaming?: boolean;
  isError?: boolean;
}

export interface SSEEvent {
  type: 'token' | 'sources' | 'done' | 'error';
  content: string | SourceDoc[];
}

// ─── UI State Types ───

export type ProcessingStatus = 'idle' | 'processing' | 'done' | 'error';
