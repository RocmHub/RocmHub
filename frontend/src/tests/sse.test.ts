import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { subscribeToJobEvents } from '../api/sse';
import type { JobEvent } from '../api/types';

describe('SSE Streaming Client', () => {
  let listeners: Record<string, Function> = {};
  let mockClose: ReturnType<typeof vi.fn>;

  class MockEventSource {
    static instances: MockEventSource[] = [];
    url: string;
    onopen: Function | null = null;
    onerror: Function | null = null;

    constructor(url: string) {
      this.url = url;
      mockClose = vi.fn();
      MockEventSource.instances.push(this);
    }

    addEventListener(event: string, callback: Function) {
      listeners[event] = callback;
    }

    close() {
      mockClose();
    }
  }

  beforeEach(() => {
    listeners = {};
    mockClose = vi.fn();
    MockEventSource.instances = [];
    vi.stubGlobal('EventSource', MockEventSource);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('subscribes to job events and passes parsed events', () => {
    const onEvent = vi.fn();
    const cleanup = subscribeToJobEvents('job_test_1', { onEvent });

    const sampleEvent: JobEvent = {
      event_id: 1,
      job_id: 'job_test_1',
      sequence: 1,
      timestamp: new Date().toISOString(),
      phase: 'PLANNING',
      status: 'RUNNING',
      message: 'Plan created',
      error_code: null,
      details: null,
    };

    listeners['job_progress']({
      type: 'job_progress',
      data: JSON.stringify(sampleEvent),
    });

    expect(onEvent).toHaveBeenCalledWith(sampleEvent);
    cleanup();
    expect(mockClose).toHaveBeenCalled();
  });

  it('deduplicates events with identical event_id', () => {
    const onEvent = vi.fn();
    const cleanup = subscribeToJobEvents('job_test_2', { onEvent });

    const sampleEvent: JobEvent = {
      event_id: 42,
      job_id: 'job_test_2',
      sequence: 1,
      timestamp: new Date().toISOString(),
      phase: 'BUILDING',
      status: 'RUNNING',
      message: 'Processing weights',
      error_code: null,
      details: null,
    };

    // Emit event twice
    listeners['job_progress']({
      type: 'job_progress',
      data: JSON.stringify(sampleEvent),
    });
    listeners['job_progress']({
      type: 'job_progress',
      data: JSON.stringify(sampleEvent),
    });

    expect(onEvent).toHaveBeenCalledTimes(1);
    cleanup();
  });

  it('triggers onComplete and closes on terminal status event', () => {
    const onEvent = vi.fn();
    const onComplete = vi.fn();
    subscribeToJobEvents('job_test_3', { onEvent, onComplete });

    const completeEvent: JobEvent = {
      event_id: 99,
      job_id: 'job_test_3',
      sequence: 5,
      timestamp: new Date().toISOString(),
      phase: 'FINALIZING',
      status: 'SUCCEEDED',
      message: 'Job completed',
      error_code: null,
      details: null,
    };

    listeners['job_completed']({
      type: 'job_completed',
      data: JSON.stringify(completeEvent),
    });

    expect(onEvent).toHaveBeenCalledWith(completeEvent);
    expect(onComplete).toHaveBeenCalled();
    expect(mockClose).toHaveBeenCalled();
  });

  it('does not treat a completed phase with a non-terminal job status as terminal', () => {
    const onComplete = vi.fn();
    const cleanup = subscribeToJobEvents('job_phase_race', { onEvent: vi.fn(), onComplete });
    const event: JobEvent = {
      event_id: 100,
      job_id: 'job_phase_race',
      sequence: 5,
      timestamp: new Date().toISOString(),
      phase: 'COMPLETED',
      status: 'RUNNING',
      message: 'Finalizing result',
      error_code: null,
      details: null,
    };
    listeners.job_completed({ type: 'job_completed', data: JSON.stringify(event) });

    expect(onComplete).not.toHaveBeenCalled();
    cleanup();
  });

  it.each(['FAILED', 'CANCELLED'] as const)('reconciles true %s terminal events', (status) => {
    const onComplete = vi.fn();
    subscribeToJobEvents(`job_${status.toLowerCase()}`, { onEvent: vi.fn(), onComplete });
    const event: JobEvent = {
      event_id: 101,
      job_id: `job_${status.toLowerCase()}`,
      sequence: 5,
      timestamp: new Date().toISOString(),
      phase: 'FINALIZING',
      status,
      message: `Job ${status.toLowerCase()}`,
      error_code: null,
      details: null,
    };
    listeners.job_failed({ type: 'job_failed', data: JSON.stringify(event) });
    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it('reconciles a lost terminal event through REST after reconnect exhaustion', async () => {
    vi.useFakeTimers();
    const onComplete = vi.fn();
    const onDegraded = vi.fn();
    vi.stubGlobal('fetch', vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ status: 'RUNNING' }) })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ status: 'SUCCEEDED' }) }));
    subscribeToJobEvents('job_lost_terminal', { onEvent: vi.fn(), onComplete, onDegraded, maxReconnectAttempts: 0, pollIntervalMs: 10 });

    // The browser loses the stream after the final SSE event was missed.
    await MockEventSource.instances[0].onerror?.();
    await vi.advanceTimersByTimeAsync(10);

    expect(onDegraded).toHaveBeenCalledWith(true);
    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it('does not complete while REST still reports RUNNING and stops polling on cleanup', async () => {
    vi.useFakeTimers();
    const onComplete = vi.fn();
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ status: 'RUNNING' }) });
    vi.stubGlobal('fetch', fetchMock);
    const cleanup = subscribeToJobEvents('job_still_running', { onEvent: vi.fn(), onComplete, maxReconnectAttempts: 0, pollIntervalMs: 10 });
    await MockEventSource.instances[0].onerror?.();
    await vi.advanceTimersByTimeAsync(10);

    expect(onComplete).not.toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    cleanup();
    await vi.advanceTimersByTimeAsync(10000);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
