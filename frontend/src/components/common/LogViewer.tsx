import React, { useEffect, useRef, useState } from 'react';
import type { JobEvent } from '../../api/types';
import { ChevronDown, ChevronRight, Copy, Check } from 'lucide-react';

interface LogViewerProps {
  events: JobEvent[];
  maxHeight?: string;
  autoScroll?: boolean;
}

export const LogViewer: React.FC<LogViewerProps> = ({
  events,
  maxHeight = '350px',
  autoScroll = true,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set());
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (autoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [events, autoScroll]);

  const toggleExpand = (id: number) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const copyAll = () => {
    const text = events
      .map(
        (e) =>
          `[${e.timestamp}] [${e.phase}] [${e.status}] ${e.message}${
            e.details ? '\n' + JSON.stringify(e.details, null, 2) : ''
          }`
      )
      .join('\n');
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (events.length === 0) {
    return (
      <div className="bg-surface rounded-lg border border-surface-border p-6 text-center text-content-secondary font-mono text-xs">
        Waiting for execution events...
      </div>
    );
  }

  return (
    <div className="relative rounded-lg border border-surface-border bg-[#0B0B0E] overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 bg-surface border-b border-surface-border text-xs text-content-secondary font-mono">
        <div className="flex items-center space-x-2">
          <span className="w-2 h-2 rounded-full bg-accent-red animate-ping" />
          <span>LIVE EVENT STREAM ({events.length} events)</span>
        </div>
        <button
          onClick={copyAll}
          className="flex items-center space-x-1 hover:text-content-primary transition-colors"
          title="Copy log contents"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>

      <div
        ref={containerRef}
        style={{ maxHeight }}
        className="p-3 font-mono text-xs overflow-y-auto space-y-1.5"
      >
        {events.map((evt) => {
          const isExpanded = expandedIds.has(evt.event_id);
          const hasDetails = evt.details && Object.keys(evt.details).length > 0;

          return (
            <div
              key={evt.event_id || `${evt.sequence}-${evt.timestamp}`}
              className="flex flex-col hover:bg-surface/50 rounded px-1.5 py-0.5 transition-colors"
            >
              <div className="flex items-start space-x-2">
                <span className="text-zinc-600 select-none">
                  {String(evt.sequence).padStart(2, '0')}
                </span>
                <span className="text-zinc-500 whitespace-nowrap select-none">
                  {evt.timestamp ? new Date(evt.timestamp).toLocaleTimeString() : ''}
                </span>
                <span
                  className={`px-1.5 py-0.2 rounded text-[10px] uppercase font-semibold tracking-wider ${
                    evt.status === 'FAILED'
                      ? 'bg-red-500/20 text-red-400'
                      : evt.status === 'SUCCESS'
                      ? 'bg-emerald-500/20 text-emerald-400'
                      : 'bg-zinc-800 text-zinc-300'
                  }`}
                >
                  {evt.phase}
                </span>
                <span className="text-content-primary flex-1 break-words">
                  {evt.message}
                </span>

                {hasDetails && (
                  <button
                    onClick={() => toggleExpand(evt.event_id)}
                    className="text-zinc-500 hover:text-zinc-300 text-[11px] flex items-center space-x-0.5 select-none"
                  >
                    <span>json</span>
                    {isExpanded ? (
                      <ChevronDown className="w-3 h-3" />
                    ) : (
                      <ChevronRight className="w-3 h-3" />
                    )}
                  </button>
                )}
              </div>

              {hasDetails && isExpanded && (
                <pre className="mt-1 ml-16 p-2 rounded bg-black/60 text-zinc-400 text-[11px] overflow-x-auto border border-zinc-800">
                  {JSON.stringify(evt.details, null, 2)}
                </pre>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
