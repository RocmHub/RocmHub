import React, { useState, useEffect } from 'react';
import type { JobResponse, JobResultResponse, JobEvent } from '../../api/types';
import { createJob, fetchJob, fetchJobResult, cancelJob } from '../../api/client';
import { subscribeToJobEvents } from '../../api/sse';
import { StatusBadge } from '../common/StatusBadge';
import { DomainStatusTag } from '../common/DomainStatusTag';
import { LogViewer } from '../common/LogViewer';
import { Bot, Play, XCircle, CheckCircle2, Clock, RotateCcw, AlertCircle, Sparkles, Cpu, Target, Shield, Check } from 'lucide-react';

import aiEngineerHeroImg from '../../assets/visuals/ai_engineer_hero.svg';

interface AIEngineerViewProps {
  selectedJobId?: string | null;
  onJobCreated?: (jobId: string) => void;
}

export const AIEngineerView: React.FC<AIEngineerViewProps> = ({ selectedJobId, onJobCreated }) => {
  const [modelId, setModelId] = useState('Qwen/Qwen2.5-0.5B-Instruct');
  const [objective, setObjective] = useState('BASE_PREPARATION');
  const [maxAttempts, setMaxAttempts] = useState(5);
  const [timeoutMinutes, setTimeoutMinutes] = useState(10);
  const [maxDiskGb, setMaxDiskGb] = useState(10);

  const [activeJob, setActiveJob] = useState<JobResponse | null>(null);
  const [jobEvents, setJobEvents] = useState<JobEvent[]>([]);
  const [jobResult, setJobResult] = useState<JobResultResponse | null>(null);
  const [isStarting, setIsStarting] = useState(false);

  // Load existing selected job if requested from Dashboard
  useEffect(() => {
    if (!selectedJobId) return;
    let unsubscribe: (() => void) | null = null;
    let isCancelled = false;

    async function loadSelectedJob() {
      try {
        const job = await fetchJob(selectedJobId!);
        if (isCancelled) return;
        setActiveJob(job);
        if (job.model_id) setModelId(job.model_id);

        if (job.status === 'SUCCEEDED' || job.status === 'FAILED' || job.status === 'CANCELLED') {
          const res = await fetchJobResult(selectedJobId!);
          if (!isCancelled) setJobResult(res);
        } else {
          unsubscribe = subscribeToJobEvents(selectedJobId!, {
            onEvent: (evt) => {
              if (!isCancelled) {
                setJobEvents((prev) => [...prev, evt]);
                const isTerminal = ['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(evt.status) || evt.phase === 'COMPLETED';
                if (isTerminal) {
                  if (['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(evt.status)) {
                    setActiveJob((prev) => (prev ? { ...prev, status: evt.status as any } : prev));
                  }
                  fetchJobResult(selectedJobId!)
                    .then((res) => {
                      if (!isCancelled) {
                        setJobResult(res);
                        setActiveJob((prev) =>
                          prev ? { ...prev, status: res.job_status, domain_status: res.domain_status } : prev
                        );
                      }
                    })
                    .catch(() => {});
                }
              }
            },
            onComplete: async () => {
              try {
                const res = await fetchJobResult(selectedJobId!);
                if (!isCancelled) setJobResult(res);
              } catch (e) {
                console.warn(e);
              }
            },
          });
        }
      } catch (err) {
        console.warn('Failed to load selected job:', err);
      }
    }

    loadSelectedJob();
    return () => {
      isCancelled = true;
      if (unsubscribe) unsubscribe();
    };
  }, [selectedJobId]);

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
          const isTerminal = ['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(event.status) || event.phase === 'COMPLETED';
          if (isTerminal) {
            if (['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(event.status)) {
              setActiveJob((prev) => (prev ? { ...prev, status: event.status as any } : prev));
            }
            fetchJobResult(job.job_id)
              .then((res) => {
                setJobResult(res);
                setActiveJob((prev) =>
                  prev ? { ...prev, status: res.job_status, domain_status: res.domain_status } : prev
                );
              })
              .catch(() => {});
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

  const OBJECTIVES = [
    { id: 'BASE_PREPARATION', label: 'Base Preparation', desc: 'Baseline runtime & metadata verification' },
    { id: 'MAX_THROUGHPUT', label: 'Max Throughput', desc: 'Optimize generation tokens per second' },
    { id: 'MIN_LATENCY', label: 'Min Latency', desc: 'Minimize time-to-first-token (TTFT)' },
    { id: 'FULL_PREPARATION', label: 'Full Preparation', desc: 'Prepare weights when hardware confirmed' },
  ];

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Visual Header Banner */}
      <div className="relative rounded-xl overflow-hidden border border-surface-border bg-surface shadow-xl">
        <div
          className="absolute inset-0 bg-cover bg-center opacity-40 mix-blend-luminosity"
          style={{ backgroundImage: `url(${aiEngineerHeroImg})` }}
        />
        <div className="absolute inset-0 hero-overlay" />

        <div className="relative p-6 space-y-2">
          <div className="inline-flex items-center space-x-2 px-2.5 py-1 rounded-full bg-accent-red/10 border border-accent-red/30 text-[11px] font-mono text-red-400">
            <Bot className="w-3.5 h-3.5 text-accent-red" />
            <span>Autonomous Closed-Loop Optimization</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Autonomous AI Engineer</h1>
          <p className="text-xs text-zinc-300 max-w-2xl font-sans leading-relaxed">
            Specialized agent executing iterative model preparation loops, diagnostics, and build repairs.
          </p>
        </div>
      </div>

      {/* Task Creation Form */}
      <div className="p-6 rounded-xl bg-surface border border-surface-border shadow-sm space-y-5">
        <h2 className="text-sm font-semibold text-content-primary flex items-center space-x-2">
          <Target className="w-4 h-4 text-accent-red" />
          <span>Engineer Session Configuration</span>
        </h2>

        <div className="space-y-3">
          <label className="text-xs font-mono text-content-secondary">Optimization Objective</label>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
            {OBJECTIVES.map((obj) => {
              const isSelected = objective === obj.id;
              return (
                <button
                  key={obj.id}
                  type="button"
                  onClick={() => setObjective(obj.id)}
                  className={`p-3 rounded-lg border text-left transition-all font-mono ${
                    isSelected
                      ? 'bg-accent-red/10 border-accent-red text-white shadow-sm'
                      : 'bg-surface-elevated/70 border-surface-border text-zinc-400 hover:text-zinc-200 hover:border-zinc-700'
                  }`}
                >
                  <div className="flex items-center justify-between text-xs font-semibold mb-1">
                    <span>{obj.label}</span>
                    {isSelected && <Check className="w-3.5 h-3.5 text-accent-red" />}
                  </div>
                  <div className="text-[10px] text-zinc-500 font-sans leading-tight">{obj.desc}</div>
                </button>
              );
            })}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
          <div className="space-y-1.5">
            <label className="text-xs font-mono text-content-secondary">Target Model ID</label>
            <input
              type="text"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              className="w-full bg-background border border-surface-border rounded-lg px-3.5 py-2.5 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red focus:ring-1 focus:ring-accent-red transition-all"
            />
          </div>

          <div className="grid grid-cols-3 gap-2.5">
            <div className="space-y-1.5">
              <label className="text-xs font-mono text-content-secondary">Max Attempts</label>
              <input
                type="number"
                min={1}
                max={10}
                value={maxAttempts}
                onChange={(e) => setMaxAttempts(parseInt(e.target.value) || 5)}
                className="w-full bg-background border border-surface-border rounded-lg px-3 py-2.5 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-mono text-content-secondary">Timeout (m)</label>
              <input
                type="number"
                min={1}
                max={60}
                value={timeoutMinutes}
                onChange={(e) => setTimeoutMinutes(parseInt(e.target.value) || 10)}
                className="w-full bg-background border border-surface-border rounded-lg px-3 py-2.5 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-mono text-content-secondary">Disk (GB)</label>
              <input
                type="number"
                min={1}
                max={100}
                value={maxDiskGb}
                onChange={(e) => setMaxDiskGb(parseInt(e.target.value) || 10)}
                className="w-full bg-background border border-surface-border rounded-lg px-3 py-2.5 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red"
              />
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-surface-border">
          <div className="flex items-center space-x-2 text-[11px] font-mono text-zinc-500">
            <Shield className="w-3.5 h-3.5 text-zinc-400" />
            <span>Guards: Isolated workspace, step quotas, fail-closed preflight.</span>
          </div>
          <button
            onClick={handleLaunch}
            disabled={isStarting || activeJob?.status === 'RUNNING'}
            className="flex items-center space-x-2 px-5 py-2.5 rounded-lg bg-accent-red hover:bg-accent-red-hover text-white text-xs font-semibold transition-all disabled:opacity-50 shadow-md shadow-red-950/40 hover:scale-[1.02] active:scale-[0.98]"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>{isStarting ? 'Launching Agent...' : 'Launch AI Engineer Session'}</span>
          </button>
        </div>
      </div>

      {/* Active Session & Actions View */}
      {activeJob && (
        <div className="p-6 rounded-xl bg-surface border border-surface-border shadow-lg space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-2.5">
              <h3 className="text-sm font-semibold text-content-primary">Agent Action Progress</h3>
              <span className="text-xs font-mono text-zinc-400 bg-surface-elevated px-2 py-0.5 rounded border border-surface-border">
                {activeJob.job_id}
              </span>
              <StatusBadge status={activeJob.status} />
              <DomainStatusTag status={activeJob.domain_status} />
            </div>

            {(activeJob.status === 'RUNNING' || activeJob.status === 'QUEUED') && (
              <button
                onClick={handleCancel}
                className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs font-mono transition-colors border border-zinc-700"
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
            <div className="p-5 rounded-xl bg-[#121216] border border-surface-border space-y-4 text-xs font-mono">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-surface-border pb-3">
                <span className="font-semibold text-emerald-400 flex items-center space-x-1.5 text-sm">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  <span>Agent Session Finished — Domain Status: {jobResult.domain_status || 'CONFIG_ONLY'}</span>
                </span>
                <span className="text-zinc-500 text-[11px]">
                  {jobResult.completed_at ? new Date(jobResult.completed_at).toLocaleTimeString() : ''}
                </span>
              </div>

              {/* Report Metrics Bar */}
              {jobResult.result && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="p-3 rounded-lg bg-surface-elevated border border-surface-border space-y-1">
                    <div className="text-zinc-500 text-[10px] uppercase flex items-center space-x-1">
                      <Clock className="w-3 h-3" />
                      <span>Duration</span>
                    </div>
                    <div className="text-zinc-200 font-bold text-sm">
                      {jobResult.result.total_duration_seconds !== undefined
                        ? `${Number(jobResult.result.total_duration_seconds).toFixed(2)}s`
                        : 'N/A'}
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-surface-elevated border border-surface-border space-y-1">
                    <div className="text-zinc-500 text-[10px] uppercase flex items-center space-x-1">
                      <RotateCcw className="w-3 h-3" />
                      <span>Attempts</span>
                    </div>
                    <div className="text-zinc-200 font-bold text-sm">
                      {jobResult.result.attempts_used !== undefined
                        ? `${jobResult.result.attempts_used}`
                        : 'N/A'}
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-surface-elevated border border-surface-border space-y-1">
                    <div className="text-zinc-500 text-[10px] uppercase flex items-center space-x-1">
                      <Cpu className="w-3 h-3" />
                      <span>Target GPU</span>
                    </div>
                    <div className="text-zinc-200 font-bold text-sm truncate">
                      {jobResult.result.target_gpu || 'Auto / None'}
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-surface-elevated border border-surface-border space-y-1">
                    <div className="text-zinc-500 text-[10px] uppercase flex items-center space-x-1">
                      <Sparkles className="w-3 h-3 text-emerald-400" />
                      <span>Build Status</span>
                    </div>
                    <div className="text-emerald-400 font-bold text-sm">
                      {jobResult.result.build_manifest?.status || 'CONFIG_ONLY'}
                    </div>
                  </div>
                </div>
              )}

              {/* Executive Reasons List */}
              {jobResult.result?.reasons && Array.isArray(jobResult.result.reasons) && jobResult.result.reasons.length > 0 && (
                <div className="p-4 rounded-lg bg-surface border border-surface-border space-y-2">
                  <div className="text-zinc-400 text-[11px] uppercase font-semibold">Decision Rationale &amp; Executive Summary</div>
                  <ul className="space-y-1.5 text-zinc-300">
                    {jobResult.result.reasons.map((reason: string, idx: number) => (
                      <li key={idx} className="flex items-start space-x-2">
                        <span className="text-accent-red font-bold">›</span>
                        <span className="leading-relaxed">{reason}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Errors Handled */}
              {jobResult.result?.errors_encountered && Array.isArray(jobResult.result.errors_encountered) && jobResult.result.errors_encountered.length > 0 && (
                <div className="p-4 rounded-lg bg-amber-500/10 border border-amber-500/20 space-y-2 text-amber-300">
                  <div className="text-amber-400 text-[11px] uppercase font-semibold flex items-center space-x-1.5">
                    <AlertCircle className="w-3.5 h-3.5" />
                    <span>Diagnostics &amp; Handled Warnings:</span>
                  </div>
                  <ul className="space-y-1 text-[11px]">
                    {jobResult.result.errors_encountered.map((err: string, idx: number) => (
                      <li key={idx} className="flex items-start space-x-2">
                        <span>•</span>
                        <span>{err}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Action Trajectory Trace */}
              {jobResult.result?.trajectory && Array.isArray(jobResult.result.trajectory) && jobResult.result.trajectory.length > 0 && (
                <div className="space-y-2 pt-2 border-t border-surface-border">
                  <div className="text-zinc-400 uppercase text-[11px] font-semibold">Agent Action Trajectory ({jobResult.result.trajectory.length} steps):</div>
                  <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
                    {jobResult.result.trajectory.map((step: any, idx: number) => (
                      <div
                        key={idx}
                        className="p-3 rounded-lg bg-surface border border-surface-border text-xs space-y-1.5 shadow-sm"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <span className="px-1.5 py-0.5 rounded bg-black/50 text-zinc-400 text-[10px] font-mono border border-zinc-800">
                              Step {step.step_index ?? idx + 1}
                            </span>
                            <span className="px-1.5 py-0.5 rounded bg-accent-red/15 text-red-400 text-[10px] font-mono border border-accent-red/20 font-semibold">
                              {step.phase}
                            </span>
                            <span className="font-semibold text-zinc-200">{step.action}</span>
                          </div>
                          {step.duration_seconds !== undefined && (
                            <span className="text-zinc-500 text-[11px]">{Number(step.duration_seconds).toFixed(2)}s</span>
                          )}
                        </div>
                        {step.observation && (
                          <div className="text-zinc-300 text-[11px] pl-2.5 border-l-2 border-zinc-700 font-sans">
                            {step.observation}
                          </div>
                        )}
                        {step.rationale && (
                          <div className="text-zinc-500 text-[10px] pl-2.5 italic">
                            Thought: {step.rationale}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
