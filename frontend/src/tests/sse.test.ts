import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { subscribeToJobEvents } from '../api/sse';
import type { JobEvent } from '../api/types';

describe('SSE Streaming Client', () => {
  let listeners: Record<string, Function> = {};
  let mockClose: ReturnType<typeof vi.fn>;

  class MockEventSource {
    url: string;
    onopen: Function | null = null;
    onerror: Function | null = null;

    constructor(url: string) {
      this.url = url;
      mockClose = vi.fn();
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
    vi.stubGlobal('EventSource', MockEventSource);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
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
});
