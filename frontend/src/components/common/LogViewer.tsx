import React, { useEffect, useRef, useState } from 'react';
import type { JobEvent } from '../../api/types';
import { Copy, Check, ChevronDown, ChevronUp } from 'lucide-react';

interface LogViewerProps {
  events: JobEvent[];
  maxHeight?: string;
  autoScroll?: boolean;
  defaultCollapsed?: boolean;
}

export const LogViewer: React.FC<LogViewerProps> = ({
  events,
  maxHeight = '280px',
  autoScroll = true,
  defaultCollapsed = false,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set());
  const [copied, setCopied] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);

  useEffect(() => {
    if (autoScroll && !isCollapsed && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [events, autoScroll, isCollapsed]);

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
      .map((e) => `[${e.timestamp}] [${e.phase}] [${e.status}] ${e.message}${e.details ? '\n' + JSON.stringify(e.details, null, 2) : ''}`)
      .join('\n');
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (events.length === 0) {
    return (
      <div className="rounded-lg border border-surface-border bg-surface-deep p-5 text-center text-xs text-content-muted font-mono">
        Waiting for events...
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-surface-border bg-surface-deep overflow-hidden">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-surface-border text-xs font-mono">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-accent-red animate-pulse" />
          <span className="text-content-muted">Live Stream</span>
          <span className="text-content-muted">·</span>
          <span className="text-content-muted">{events.length} events</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={copyAll}
            className="flex items-center gap-1 text-content-muted hover:text-content-primary transition-colors"
            title="Copy log"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            <span className="hidden sm:inline">{copied ? 'Copied' : 'Copy'}</span>
          </button>
          <button
            onClick={() => setIsCollapsed(v => !v)}
            className="text-content-muted hover:text-content-primary transition-colors"
          >
            {isCollapsed ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronUp className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Log content */}
      {!isCollapsed && (
        <div
          ref={containerRef}
          style={{ maxHeight }}
          className="p-3 font-mono text-xs overflow-y-auto space-y-1"
        >
          {events.map((evt) => {
            const isExpanded = expandedIds.has(evt.event_id);
            const hasDetails = evt.details && Object.keys(evt.details).length > 0;
            const phaseColor =
              evt.status === 'FAILED' ? 'bg-red-500/20 text-red-400 border-red-500/20' :
              evt.status === 'SUCCEEDED' || evt.phase === 'COMPLETED' ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/20' :
              'bg-surface-elevated text-content-secondary border-surface-border';

            return (
              <div
                key={evt.event_id || `${evt.sequence}-${evt.timestamp}`}
                className="flex items-start gap-2 hover:bg-surface/30 rounded px-1.5 py-1 transition-colors"
              >
                <span className="text-content-muted select-none w-5 text-right shrink-0">{String(evt.sequence).padStart(2, '0')}</span>
                <span className="text-content-muted text-[10px] whitespace-nowrap select-none shrink-0 mt-px">
                  {evt.timestamp ? new Date(evt.timestamp).toLocaleTimeString() : ''}
                </span>
                <span className={`px-1.5 py-px rounded text-[10px] uppercase font-semibold tracking-wider border shrink-0 ${phaseColor}`}>
                  {evt.phase}
                </span>
                <span className="text-content-primary flex-1 break-words leading-relaxed">{evt.message}</span>
                {hasDetails && (
                  <button
                    onClick={() => toggleExpand(evt.event_id)}
                    className="text-content-muted hover:text-content-secondary text-[10px] shrink-0 flex items-center gap-0.5"
                  >
                    json{isExpanded ? <ChevronUp className="w-2.5 h-2.5" /> : <ChevronDown className="w-2.5 h-2.5" />}
                  </button>
                )}
              </div>
            );
          })}
          {/* Inline expanded JSON */}
          {events.map((evt) => {
            const isExpanded = expandedIds.has(evt.event_id);
            const hasDetails = evt.details && Object.keys(evt.details).length > 0;
            return hasDetails && isExpanded ? (
              <pre key={`json-${evt.event_id}`} className="ml-16 p-2 rounded-lg bg-black/50 text-content-muted text-[11px] overflow-x-auto border border-surface-border">
                {JSON.stringify(evt.details, null, 2)}
              </pre>
            ) : null;
          })}
        </div>
      )}
    </div>
  );
};
