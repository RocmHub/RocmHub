import type { JobEvent } from './types';

export interface SseOptions {
  onEvent: (event: JobEvent) => void;
  onError?: (error: Error) => void;
  onComplete?: () => void;
  /** Transport health only; an SSE failure is not a domain job failure. */
  onDegraded?: (degraded: boolean) => void;
  maxReconnectAttempts?: number;
  pollIntervalMs?: number;
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
  let pollingTimeout: ReturnType<typeof setTimeout> | null = null;
  let isDegraded = false;
  let pollAttempt = 0;
  let completed = false;

  function finish() {
    if (completed || isClosed) return;
    completed = true;
    cleanup();
    options.onComplete?.();
  }

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
        // Transport/event labels are hints, not domain state. Only an explicit
        // terminal job status may stop observation; a late phase event can race
        // the transaction that persists the result.
        const isTerminal =
          parsed.status === 'SUCCEEDED' ||
          parsed.status === 'FAILED' ||
          parsed.status === 'CANCELLED';

        if (isTerminal) {
          finish();
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
      if (isDegraded) {
        isDegraded = false;
        options.onDegraded?.(false);
      }
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
            finish();
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
        if (!isDegraded) {
          isDegraded = true;
          options.onDegraded?.(true);
          options.onError?.(new Error('Live updates are interrupted; checking job status directly.'));
        }
        scheduleStatusCheck();
        // Keep a low-frequency SSE recovery attempt alive alongside REST polling.
        reconnectTimeout = setTimeout(connect, 30000);
      }
    };
  }

  function scheduleStatusCheck() {
    if (isClosed || pollingTimeout) return;
    const base = options.pollIntervalMs ?? 1000;
    const delay = Math.min(base * Math.pow(1.7, pollAttempt++), 10000);
    pollingTimeout = setTimeout(async () => {
      pollingTimeout = null;
      if (isClosed) return;
      try {
        const response = await fetch(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
        if (response.ok) {
          const job = await response.json();
          if (['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(job.status)) {
            finish();
            return;
          }
        }
      } catch {
        // Keep the bounded-frequency status check alive through transient API outages.
      }
      scheduleStatusCheck();
    }, delay);
  }

  function cleanup() {
    isClosed = true;
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout);
      reconnectTimeout = null;
    }
    if (pollingTimeout) {
      clearTimeout(pollingTimeout);
      pollingTimeout = null;
    }
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
  }

  connect();
  return cleanup;
}
