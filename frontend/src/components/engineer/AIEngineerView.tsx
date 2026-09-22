import React, { useState } from 'react';
import type { JobResponse, JobResultResponse, JobEvent } from '../../api/types';
import { createJob, fetchJobResult, cancelJob } from '../../api/client';
import { subscribeToJobEvents } from '../../api/sse';
import { StatusBadge } from '../common/StatusBadge';
import { DomainStatusTag } from '../common/DomainStatusTag';
import { LogViewer } from '../common/LogViewer';
import { Bot, Play, XCircle, CheckCircle2 } from 'lucide-react';

interface AIEngineerViewProps {
  onJobCreated?: (jobId: string) => void;
}

export const AIEngineerView: React.FC<AIEngineerViewProps> = ({ onJobCreated }) => {
  const [modelId, setModelId] = useState('Qwen/Qwen2.5-0.5B-Instruct');
  const [objective, setObjective] = useState('BASE_PREPARATION');
  const [maxAttempts, setMaxAttempts] = useState(5);
  const [timeoutMinutes, setTimeoutMinutes] = useState(10);
  const [maxDiskGb, setMaxDiskGb] = useState(10);

  const [activeJob, setActiveJob] = useState<JobResponse | null>(null);
  const [jobEvents, setJobEvents] = useState<JobEvent[]>([]);
  const [jobResult, setJobResult] = useState<JobResultResponse | null>(null);
  const [isStarting, setIsStarting] = useState(false);

  const handleLaunch = async () => {
    setIsStarting(true);
    setJobEvents([]);
    setJobResult(null);

    try {
      const job = await createJob({
        job_type: 'ENGINEER',
        model_id: modelId,
        objective,
        max_attempts: maxAttempts,
        timeout_seconds: timeoutMinutes * 60,
        max_disk_gb: maxDiskGb,
        allow_full_weights: false,
      });

      setActiveJob(job);
      onJobCreated?.(job.job_id);

      subscribeToJobEvents(job.job_id, {
        onEvent: (event) => {
          setJobEvents((prev) => [...prev, event]);
          if (event.status === 'SUCCEEDED' || event.status === 'FAILED' || event.status === 'CANCELLED') {
            setActiveJob((prev) => (prev ? { ...prev, status: event.status as any } : prev));
          }
        },
        onComplete: async () => {
          try {
            const res = await fetchJobResult(job.job_id);
            setJobResult(res);
            setActiveJob((prev) =>
              prev
                ? {
                    ...prev,
                    status: res.job_status,
                    domain_status: res.domain_status,
                  }
                : prev
            );
          } catch (e) {
            console.warn('Failed to fetch engineer job result:', e);
          }
        },
      });
    } catch (err: any) {
      alert(`Launch failed: ${err.message}`);
    } finally {
      setIsStarting(false);
    }
  };

  const handleCancel = async () => {
    if (!activeJob) return;
    try {
      await cancelJob(activeJob.job_id);
      setActiveJob((prev) => (prev ? { ...prev, status: 'CANCELLED' } : prev));
    } catch (e: any) {
      alert(`Cancel failed: ${e.message}`);
    }
  };

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h1 className="text-xl font-bold text-content-primary">Autonomous AI Engineer</h1>
        <p className="text-xs text-content-secondary mt-0.5">
          Specialized agent executing iterative model preparation loops, diagnostics, and build repairs.
        </p>
      </div>

      {/* Task Creation Form */}
      <div className="p-5 rounded-lg bg-surface border border-surface-border space-y-4">
        <h2 className="text-sm font-semibold text-content-primary flex items-center space-x-2">
          <Bot className="w-4 h-4 text-accent-red" />
          <span>Engineer Session Configuration</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-1">
            <label className="text-xs font-mono text-content-secondary">Target Model ID</label>
            <input
              type="text"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              className="w-full bg-background border border-surface-border rounded px-3 py-2 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red"
            />
          </div>

          <div className="space-y-1">
            <label className="text-xs font-mono text-content-secondary">Optimization Objective</label>
            <select
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              className="w-full bg-background border border-surface-border rounded px-3 py-2 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red"
            >
              <option value="BASE_PREPARATION">BASE_PREPARATION (Baseline preparation)</option>
              <option value="FULL_PREPARATION">FULL_PREPARATION (Full weights preparation)</option>
              <option value="AMD_EXECUTION">AMD_EXECUTION (Execution on AMD hardware)</option>
              <option value="MAX_THROUGHPUT">MAX_THROUGHPUT (Tokens per second)</option>
              <option value="MIN_LATENCY">MIN_LATENCY (Time to first token)</option>
            </select>
          </div>
        </div>

        {/* Budget Limits */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2 border-t border-surface-border">
          <div className="space-y-1">
            <label className="text-xs font-mono text-content-secondary">Max Retry Attempts</label>
            <input
              type="number"
              min={1}
              max={10}
              value={maxAttempts}
              onChange={(e) => setMaxAttempts(parseInt(e.target.value) || 3)}
              className="w-full bg-background border border-surface-border rounded px-3 py-2 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red"
            />
          </div>

          <div className="space-y-1">
            <label className="text-xs font-mono text-content-secondary">Execution Budget (Minutes)</label>
            <input
              type="number"
              min={1}
              max={60}
              value={timeoutMinutes}
              onChange={(e) => setTimeoutMinutes(parseInt(e.target.value) || 10)}
              className="w-full bg-background border border-surface-border rounded px-3 py-2 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red"
            />
          </div>

          <div className="space-y-1">
            <label className="text-xs font-mono text-content-secondary">Max Disk Limit (GB)</label>
            <input
              type="number"
              min={1}
              max={50}
              value={maxDiskGb}
              onChange={(e) => setMaxDiskGb(parseInt(e.target.value) || 10)}
              className="w-full bg-background border border-surface-border rounded px-3 py-2 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red"
            />
          </div>
        </div>

        <div className="flex items-center justify-between pt-2 border-t border-surface-border">
          <div className="text-[11px] font-mono text-zinc-500">
            Safety Boundary: Zero arbitrary shell access • Structured tool actions only
          </div>
          <button
            onClick={handleLaunch}
            disabled={isStarting || activeJob?.status === 'RUNNING'}
            className="flex items-center space-x-2 px-4 py-2 rounded bg-accent-red hover:bg-accent-red-hover text-white text-xs font-medium transition-colors disabled:opacity-50"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>{isStarting ? 'Initiating Agent...' : 'Launch AI Engineer'}</span>
          </button>
        </div>
      </div>

      {/* Active Session & Actions View */}
      {activeJob && (
        <div className="p-5 rounded-lg bg-surface border border-surface-border space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <h3 className="text-sm font-semibold text-content-primary">Agent Action Progress</h3>
              <span className="text-xs font-mono text-zinc-400">ID: {activeJob.job_id}</span>
              <StatusBadge status={activeJob.status} />
              <DomainStatusTag status={activeJob.domain_status} />
            </div>

            {(activeJob.status === 'RUNNING' || activeJob.status === 'QUEUED') && (
              <button
                onClick={handleCancel}
                className="flex items-center space-x-1.5 px-3 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs font-mono transition-colors"
              >
                <XCircle className="w-3.5 h-3.5 text-zinc-400" />
                <span>Cancel Execution</span>
              </button>
            )}
          </div>

          {/* Structured Actions Stream */}
          <LogViewer events={jobEvents} />

          {/* Final Agent Report */}
          {jobResult && (
            <div className="p-4 rounded bg-[#131317] border border-surface-border space-y-3 text-xs font-mono">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-emerald-400 flex items-center space-x-1.5">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>Agent Loop Finished — Status: {jobResult.domain_status || 'CONFIG_ONLY'}</span>
                </span>
                <span className="text-zinc-500">
                  {jobResult.completed_at ? new Date(jobResult.completed_at).toLocaleTimeString() : ''}
                </span>
              </div>

              {jobResult.result?.summary && (
                <div className="p-3 rounded bg-surface-elevated border border-surface-border text-zinc-300">
                  <div className="text-zinc-500 text-[11px] uppercase mb-1">Agent Executive Summary</div>
                  {jobResult.result.summary}
                </div>
              )}

              {jobResult.result?.iterations && Array.isArray(jobResult.result.iterations) && (
                <div className="space-y-1.5 pt-2">
                  <div className="text-zinc-400 uppercase text-[11px]">Iterations Executed:</div>
                  {jobResult.result.iterations.map((iter: any, idx: number) => (
                    <div
                      key={idx}
                      className="p-2 rounded bg-surface-elevated border border-surface-border flex items-center justify-between"
                    >
                      <span className="text-zinc-300">Attempt {idx + 1}: {iter.action || 'Execute step'}</span>
                      <span className="text-zinc-500">{iter.status || 'OK'}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
