'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { ChatMessage, SourceDoc } from '@/lib/types';
import { generateId, parseSSEStream } from '@/lib/utils';

interface ChatPanelProps {
  isReady: boolean;
}

// ─── Citation Badge ───
// Clicking toggles an expanded preview of the transcript chunk that the LLM used.
function SourceBadge({ source }: { source: SourceDoc }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="inline-block">
      <button
        onClick={() => setExpanded(!expanded)}
        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-sky-500/10 border border-sky-500/20 text-[11px] text-sky-300 hover:bg-sky-500/20 transition-colors cursor-pointer"
        title="Click to view source transcript"
      >
        <svg className="w-3 h-3 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.172 13.828a4 4 0 005.656 0l4-4a4 4 0 10-5.656-5.656l-1.102 1.101" />
        </svg>
        Video {source.video_id} · Chunk {source.chunk_index}
        <svg
          className={`w-2.5 h-2.5 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Expanded transcript preview */}
      {expanded && (
        <div className="mt-1.5 px-3 py-2 rounded-lg bg-zinc-800/90 border border-zinc-700/50 text-[11px] text-zinc-400 leading-relaxed max-w-sm">
          <p className="text-[10px] text-zinc-500 mb-1 uppercase tracking-wider">
            {source.title} · {source.creator}
          </p>
          <p className="text-zinc-300">{source.text_preview}</p>
        </div>
      )}
    </div>
  );
}

// ─── Message Bubble ───
function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[85%] space-y-2 ${isUser ? 'order-1' : ''}`}>
        {/* Role label */}
        <p className={`text-[10px] uppercase tracking-widest ${
          isUser ? 'text-right text-sky-400' : 'text-zinc-500'
        }`}>
          {isUser ? 'You' : 'CHROMA AI'}
        </p>

        {/* Message body */}
        <div
          className={`px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
            message.isError
              ? 'bg-red-500/10 text-red-300 border border-red-500/20 rounded-bl-md'
              : isUser
                ? 'bg-sky-600 text-white rounded-br-md'
                : 'bg-zinc-800/80 text-zinc-200 border border-zinc-700/50 rounded-bl-md'
          }`}
        >
          {message.content}
          {message.isStreaming && (
            <span className="inline-block w-1.5 h-4 ml-0.5 bg-sky-400 animate-pulse rounded-sm" />
          )}
        </div>

        {/* Source citations — only shown after streaming completes */}
        {!message.isStreaming && message.sources && message.sources.length > 0 && (
          <div className="flex flex-wrap gap-1.5 pt-1">
            {message.sources.map((src, i) => (
              <SourceBadge key={`${src.video_id}-${src.chunk_index}-${i}`} source={src} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Main Chat Panel ───
export default function ChatPanel({ isReady }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Smart auto-scroll: tracks whether the user has manually scrolled up.
  // If they have, we don't force-scroll to bottom on new tokens.
  // This prevents the jarring experience of reading old messages while the
  // AI is still streaming — the user stays where they scrolled.
  const shouldAutoScroll = useRef(true);

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    // If user is within 80px of bottom, re-enable auto-scroll.
    // If they've scrolled up more than 80px, disable it.
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    shouldAutoScroll.current = atBottom;
  }, []);

  // Scroll to bottom only if the user hasn't scrolled up
  useEffect(() => {
    if (shouldAutoScroll.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // ─── Core: Send message and handle SSE stream ───
  const sendMessage = useCallback(async () => {
    const query = input.trim();
    if (!query || isLoading) return;

    const userMsg: ChatMessage = {
      id: generateId(),
      role: 'user',
      content: query,
    };

    const aiMsg: ChatMessage = {
      id: generateId(),
      role: 'assistant',
      content: '',
      isStreaming: true,
    };

    // Snapshot chat_history BEFORE adding current exchange.
    // This is what the backend needs for conversational context.
    const chatHistory = messages
      .filter(m => !m.isError)
      .map(m => ({
        role: m.role as 'user' | 'assistant',
        content: m.content,
      }));

    setMessages(prev => [...prev, userMsg, aiMsg]);
    setInput('');
    setIsLoading(true);
    // Re-enable auto-scroll when the user sends a new message
    shouldAutoScroll.current = true;

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          chat_history: chatHistory,
        }),
      });

      if (!response.ok) {
        // Graceful degradation: show the error inline, don't break state
        let errorText: string;
        try {
          errorText = await response.text();
        } catch {
          errorText = 'Unable to read error response';
        }
        throw new Error(
          response.status === 500
            ? `Server error — the backend encountered an issue. ${errorText.slice(0, 200)}`
            : `Request failed (${response.status}): ${errorText.slice(0, 200)}`
        );
      }

      if (!response.body) {
        throw new Error('No response body — SSE streaming not supported by this connection');
      }

      // Parse the SSE stream token-by-token.
      // The parseSSEStream generator handles chunk-boundary buffering
      // so partial JSON never causes a parse error.
      let accumulatedContent = '';
      let sources: SourceDoc[] = [];

      for await (const event of parseSSEStream(response)) {
        if (event.type === 'token') {
          accumulatedContent += event.content as string;
          setMessages(prev => {
            const updated = [...prev];
            const lastIdx = updated.length - 1;
            updated[lastIdx] = {
              ...updated[lastIdx],
              content: accumulatedContent,
              isStreaming: true,
            };
            return updated;
          });
        } else if (event.type === 'sources') {
          sources = event.content as SourceDoc[];
        } else if (event.type === 'done') {
          // Finalize: attach sources, clear streaming flag
          setMessages(prev => {
            const updated = [...prev];
            const lastIdx = updated.length - 1;
            updated[lastIdx] = {
              ...updated[lastIdx],
              content: accumulatedContent,
              isStreaming: false,
              sources,
            };
            return updated;
          });
        } else if (event.type === 'error') {
          // Backend sent an error event mid-stream
          throw new Error(event.content as string);
        }
      }

      // Safety net: if stream ended without a 'done' event, finalize anyway
      setMessages(prev => {
        const last = prev[prev.length - 1];
        if (last?.isStreaming) {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...last,
            content: accumulatedContent || 'No response received.',
            isStreaming: false,
            sources,
          };
          return updated;
        }
        return prev;
      });

    } catch (err) {
      // Graceful degradation: show the error as a styled message
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setMessages(prev => {
        const updated = [...prev];
        const lastIdx = updated.length - 1;
        if (lastIdx >= 0 && updated[lastIdx].role === 'assistant') {
          updated[lastIdx] = {
            ...updated[lastIdx],
            content: errorMessage,
            isStreaming: false,
            isError: true,
          };
        }
        return updated;
      });
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }, [input, isLoading, messages]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full bg-zinc-900/60 border border-zinc-800 rounded-2xl overflow-hidden backdrop-blur-sm">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-zinc-800">
        <div className={`w-2 h-2 rounded-full ${isLoading ? 'bg-amber-400' : 'bg-emerald-400'} animate-pulse`} />
        <h2 className="text-sm font-semibold text-zinc-200">RAG Chat Engine</h2>
        {isLoading && (
          <span className="text-[10px] text-amber-400/80 animate-pulse">thinking…</span>
        )}
        <span className="ml-auto text-[10px] text-zinc-600">
          {messages.filter(m => m.role === 'user').length} queries
        </span>
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-4 space-y-4"
      >
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center space-y-3 max-w-xs">
              <div className="w-12 h-12 mx-auto rounded-2xl bg-sky-500/10 border border-sky-500/20 flex items-center justify-center">
                <svg className="w-6 h-6 text-sky-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z" />
                </svg>
              </div>
              <p className="text-sm text-zinc-400">
                {isReady
                  ? 'Ask anything about your videos. Try: "Compare the hooks in the first 5 seconds"'
                  : 'Process videos first to start chatting'}
              </p>
            </div>
          </div>
        )}
        {messages.map(msg => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
      </div>

      {/* Input Bar — locked during streaming to prevent race conditions */}
      <div className="p-3 border-t border-zinc-800">
        <div className={`flex items-center gap-2 rounded-xl px-3 py-2 border transition-colors ${
          isLoading
            ? 'bg-zinc-800/30 border-zinc-700/30'
            : 'bg-zinc-800/60 border-zinc-700/50 focus-within:border-sky-500/50'
        }`}>
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              isLoading
                ? 'Waiting for response…'
                : isReady
                  ? 'Ask about your videos…'
                  : 'Process videos first…'
            }
            disabled={!isReady || isLoading}
            className="flex-1 bg-transparent text-sm text-zinc-200 placeholder-zinc-600 outline-none disabled:opacity-40 disabled:cursor-not-allowed"
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || isLoading || !isReady}
            className="p-1.5 rounded-lg bg-sky-500 hover:bg-sky-400 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            aria-label="Send message"
          >
            {isLoading ? (
              <svg className="w-4 h-4 text-white animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            ) : (
              <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
              </svg>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
