'use client';

import { useState, useCallback } from 'react';
import {
  VideoMetadata,
  IngestionResponse,
  ProcessingStatus,
} from '@/lib/types';
import VideoCard from '@/components/VideoCard';
import ChatPanel from '@/components/ChatPanel';

// ─── Skeleton Loader ───
// Matches the exact dimensions of VideoCard to prevent layout shift.
function VideoCardSkeleton() {
  return (
    <div className="bg-zinc-900/80 border border-zinc-800 rounded-2xl overflow-hidden p-4 space-y-3">
      <div className="flex items-center gap-2">
        <div className="skeleton w-7 h-7 rounded-lg" />
        <div className="skeleton h-4 flex-1 rounded" />
      </div>
      <div className="skeleton w-full aspect-video rounded-xl" />
      <div className="skeleton h-8 w-32 rounded-lg" />
      <div className="space-y-2">
        {[1, 2, 3, 4, 5].map(i => (
          <div key={i} className="flex justify-between">
            <div className="skeleton h-3 w-16 rounded" />
            <div className="skeleton h-3 w-20 rounded" />
          </div>
        ))}
      </div>
    </div>
  );
}

export default function HomePage() {
  // ─── State ───
  const [urlA, setUrlA] = useState('');
  const [urlB, setUrlB] = useState('');
  const [status, setStatus] = useState<ProcessingStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [videoA, setVideoA] = useState<VideoMetadata | null>(null);
  const [videoB, setVideoB] = useState<VideoMetadata | null>(null);
  const [ingestionInfo, setIngestionInfo] = useState<string | null>(null);

  // ─── Process Videos (single API call) ───
  const processVideos = useCallback(async () => {
    if (!urlA.trim() || !urlB.trim()) {
      setError('Both video URLs are required');
      return;
    }

    setStatus('processing');
    setError(null);
    setVideoA(null);
    setVideoB(null);
    setIngestionInfo(null);

    try {
      // Single call to /ingest — extracts, chunks, embeds, stores,
      // AND returns full VideoMetadata for both videos.
      const res = await fetch('/api/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          videos: [
            { url: urlA.trim(), video_id: 'A' },
            { url: urlB.trim(), video_id: 'B' },
          ],
        }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Server error ${res.status}: ${text.slice(0, 300)}`);
      }

      const data: IngestionResponse = await res.json();

      // Show non-fatal warnings
      if (data.errors && data.errors.length > 0) {
        setError(data.errors.join('\n'));
      }

      // Pull full VideoMetadata from the ingestion response
      for (const video of data.videos) {
        if (video.metadata) {
          if (video.video_id === 'A') setVideoA(video.metadata);
          if (video.video_id === 'B') setVideoB(video.metadata);
        }
      }

      setIngestionInfo(
        `${data.total_chunks} chunks stored · ${data.videos.length} videos processed`
      );
      setStatus('done');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setStatus('error');
    }
  }, [urlA, urlB]);

  const isReady = status === 'done' && videoA !== null && videoB !== null;

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* ─── Top Bar ─── */}
      <header className="shrink-0 border-b border-zinc-800 bg-zinc-950/90 backdrop-blur-sm">
        <div className="max-w-[1600px] mx-auto px-4 py-3">
          {/* Brand */}
          <div className="flex items-center gap-3 mb-3">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-violet-600 to-fuchsia-500 flex items-center justify-center">
                <span className="text-white text-xs font-bold">C</span>
              </div>
              <h1 className="text-lg font-bold tracking-tight">
                <span className="text-violet-400">CHROMA</span>
                <span className="text-zinc-500 font-normal text-sm ml-2">
                  Video RAG Analytics
                </span>
              </h1>
            </div>
            {ingestionInfo && (
              <span className="ml-auto text-xs text-emerald-400 bg-emerald-400/10 px-2 py-1 rounded-md">
                ✓ {ingestionInfo}
              </span>
            )}
          </div>

          {/* URL Inputs */}
          <div className="flex flex-col sm:flex-row gap-2">
            <input
              type="url"
              value={urlA}
              onChange={e => setUrlA(e.target.value)}
              placeholder="Video A URL (YouTube)"
              disabled={status === 'processing'}
              className="flex-1 bg-zinc-900 border border-zinc-800 rounded-xl px-3 py-2 text-sm text-zinc-200 placeholder-zinc-600 outline-none focus:border-violet-500/50 disabled:opacity-50 transition-colors"
            />
            <input
              type="url"
              value={urlB}
              onChange={e => setUrlB(e.target.value)}
              placeholder="Video B URL (YouTube / Instagram)"
              disabled={status === 'processing'}
              className="flex-1 bg-zinc-900 border border-zinc-800 rounded-xl px-3 py-2 text-sm text-zinc-200 placeholder-zinc-600 outline-none focus:border-violet-500/50 disabled:opacity-50 transition-colors"
            />
            <button
              onClick={processVideos}
              disabled={status === 'processing' || !urlA.trim() || !urlB.trim()}
              className="px-5 py-2 rounded-xl bg-violet-600 hover:bg-violet-500 text-white text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
            >
              {status === 'processing' ? (
                <span className="flex items-center gap-2">
                  <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Processing…
                </span>
              ) : (
                'Process Videos'
              )}
            </button>
          </div>

          {/* Error Banner */}
          {error && (
            <div className="mt-2 px-3 py-2 rounded-xl bg-red-500/10 border border-red-500/20 text-xs text-red-400 flex items-start gap-2">
              <svg className="w-4 h-4 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
              </svg>
              <pre className="whitespace-pre-wrap font-sans">{error}</pre>
            </div>
          )}
        </div>
      </header>

      {/* ─── Dashboard Workspace ─── */}
      <main className="flex-1 overflow-hidden">
        <div className="h-full max-w-[1600px] mx-auto p-4 grid grid-cols-1 lg:grid-cols-5 gap-4">
          {/* Left Panel: Video Cards (3/5 width on desktop) */}
          <div className="lg:col-span-3 overflow-y-auto pr-1 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {status === 'processing' ? (
                <>
                  <VideoCardSkeleton />
                  <VideoCardSkeleton />
                </>
              ) : videoA && videoB ? (
                <>
                  <VideoCard metadata={videoA} label="A" />
                  <VideoCard metadata={videoB} label="B" />
                </>
              ) : (
                <div className="md:col-span-2 flex items-center justify-center py-20">
                  <div className="text-center space-y-3 max-w-sm">
                    <div className="w-14 h-14 mx-auto rounded-2xl bg-zinc-800 border border-zinc-700 flex items-center justify-center">
                      <svg className="w-7 h-7 text-zinc-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3.375 19.5h17.25m-17.25 0a1.125 1.125 0 01-1.125-1.125M3.375 19.5h1.5C5.496 19.5 6 18.996 6 18.375m-2.625 0V5.625m0 0A1.125 1.125 0 014.5 4.5h15a1.125 1.125 0 011.125 1.125v12.75M18 19.5h1.5m-1.5 0a1.125 1.125 0 01-1.125-1.125M18 19.5v-1.125m0 0A1.125 1.125 0 0019.5 17.25h1.125" />
                      </svg>
                    </div>
                    <p className="text-sm text-zinc-500">
                      Paste two video URLs above and click{' '}
                      <span className="text-violet-400">Process Videos</span> to begin
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right Panel: Chat (2/5 width on desktop) */}
          <div className="lg:col-span-2 min-h-[400px] lg:min-h-0">
            <ChatPanel isReady={isReady} />
          </div>
        </div>
      </main>
    </div>
  );
}
