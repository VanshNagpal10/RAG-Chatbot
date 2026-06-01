'use client';

import { VideoMetadata } from '@/lib/types';
import { formatNumber, formatDuration, getYouTubeEmbedUrl } from '@/lib/utils';

interface VideoCardProps {
  metadata: VideoMetadata;
  label: string; // "A" or "B"
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-zinc-800/50 last:border-0">
      <span className="text-xs text-zinc-500 uppercase tracking-wider">{label}</span>
      <span className="text-sm font-medium text-zinc-200">{value}</span>
    </div>
  );
}

export default function VideoCard({ metadata, label }: VideoCardProps) {
  const embedUrl = getYouTubeEmbedUrl(metadata.source_url);
  const engRate = metadata.engagement_rate;

  // Color-code engagement rate
  const engColor =
    engRate != null && engRate >= 5
      ? 'from-emerald-500 to-emerald-400'
      : engRate != null && engRate >= 2
        ? 'from-amber-500 to-yellow-400'
        : 'from-red-500 to-red-400';

  return (
    <div className="bg-zinc-900/80 border border-zinc-800 rounded-2xl overflow-hidden backdrop-blur-sm">
      {/* Video Label Badge */}
      <div className="flex items-center gap-2 px-4 pt-4 pb-2">
        <span className="inline-flex items-center justify-center w-7 h-7 rounded-lg bg-violet-600 text-white text-xs font-bold">
          {label}
        </span>
        <h3 className="text-sm font-semibold text-zinc-100 truncate flex-1" title={metadata.title || 'Untitled'}>
          {metadata.title || 'Untitled Video'}
        </h3>
      </div>

      {/* Video Embed */}
      <div className="relative mx-4 rounded-xl overflow-hidden bg-zinc-950 aspect-video">
        {embedUrl ? (
          <iframe
            src={embedUrl}
            className="absolute inset-0 w-full h-full"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
            loading="lazy"
            title={`Video ${label}`}
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-zinc-600">
            <div className="text-center">
              <svg className="w-10 h-10 mx-auto mb-2 opacity-40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15.91 11.672a.375.375 0 010 .656l-5.603 3.113a.375.375 0 01-.557-.328V8.887c0-.286.307-.466.557-.327l5.603 3.112z" />
              </svg>
              <p className="text-xs">Embed not available</p>
            </div>
          </div>
        )}
      </div>

      {/* Engagement Rate Badge */}
      <div className="px-4 pt-3">
        <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gradient-to-r ${engColor} bg-opacity-10`}>
          <span className="text-[10px] uppercase tracking-widest text-white/70">Engagement</span>
          <span className="text-lg font-bold text-white">
            {engRate != null ? `${engRate.toFixed(2)}%` : '—'}
          </span>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="px-4 py-3 space-y-0">
        <MetricRow label="Creator" value={metadata.creator || '—'} />
        <MetricRow label="Views" value={formatNumber(metadata.views)} />
        <MetricRow label="Likes" value={formatNumber(metadata.likes)} />
        <MetricRow label="Comments" value={formatNumber(metadata.comments)} />
        <MetricRow label="Followers" value={formatNumber(metadata.follower_count)} />
        <MetricRow label="Duration" value={formatDuration(metadata.duration_seconds)} />
        <MetricRow label="Platform" value={metadata.platform} />
      </div>
    </div>
  );
}
