import type { JobEvent } from './types';

export interface SseOptions {
  onEvent: (event: JobEvent) => void;
  onError?: (error: Error) => void;
  onComplete?: () => void;
  maxReconnectAttempts?: number;
}

export function subscribeToJobEvents(
  jobId: string,
  options: SseOptions
): () => void {
  let eventSource: EventSource | null = null;
  let lastEventId = 0;
  const processedEventIds = new Set<number>();
  let reconnectAttempts = 0;
  const maxAttempts = options.maxReconnectAttempts ?? 5;
  let isClosed = false;
  let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;

  function connect() {
    if (isClosed) return;

    // Use from_event_id query param so backend SQLite replay resumes exactly after last received
    const url = `/api/v1/jobs/${encodeURIComponent(jobId)}/events?from_event_id=${lastEventId}`;
    eventSource = new EventSource(url);

    function handleEvent(e: MessageEvent) {
      if (isClosed) return;
      try {
        const parsed: JobEvent = JSON.parse(e.data);
        if (parsed.event_id && processedEventIds.has(parsed.event_id)) {
          return; // Prevent duplicate processing
        }
        if (parsed.event_id) {
          processedEventIds.add(parsed.event_id);
          lastEventId = Math.max(lastEventId, parsed.event_id);
        }

        options.onEvent(parsed);

        // Check terminal state: only job-level terminal states, not step-level SUCCESS
        const isTerminal =
          parsed.status === 'SUCCEEDED' ||
          parsed.status === 'FAILED' ||
          parsed.status === 'CANCELLED' ||
          parsed.phase === 'COMPLETED' ||
          e.type === 'job_completed' ||
          e.type === 'job_failed' ||
          e.type === 'job_cancelled';

        if (isTerminal) {
          cleanup();
          options.onComplete?.();
        }
      } catch (err: any) {
        console.warn('Failed to parse SSE event payload:', err, e.data);
      }
    }

    // Register all standard event types sent by ROCmHub backend
    const eventTypes = [
      'message',
      'job_queued',
      'job_started',
      'job_progress',
      'job_completed',
      'job_failed',
      'job_cancelled',
    ];

    eventTypes.forEach((type) => {
      eventSource?.addEventListener(type, handleEvent);
    });

    eventSource.onopen = () => {
      reconnectAttempts = 0;
    };

    eventSource.onerror = async () => {
      if (isClosed) return;
      eventSource?.close();
      eventSource = null;

      // Check if job completed cleanly on backend before triggering reconnection backoff
      try {
        const checkRes = await fetch(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
        if (checkRes.ok) {
          const jobData = await checkRes.json();
          if (['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(jobData.status)) {
            cleanup();
            options.onComplete?.();
            return;
          }
        }
      } catch {
        // proceed with reconnection attempt
      }

      if (reconnectAttempts < maxAttempts) {
        reconnectAttempts++;
        const backoffMs = Math.min(1000 * Math.pow(1.5, reconnectAttempts), 5000);
        reconnectTimeout = setTimeout(() => {
          connect();
        }, backoffMs);
      } else {
        options.onError?.(new Error('SSE stream disconnected after maximum retry attempts'));
      }
    };
  }

  function cleanup() {
    isClosed = true;
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout);
      reconnectTimeout = null;
    }
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
  }

  connect();
  return cleanup;
}
