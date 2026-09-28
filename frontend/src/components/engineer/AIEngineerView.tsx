import React, { useState, useEffect } from 'react';
import type { JobResponse, JobResultResponse, JobEvent } from '../../api/types';
import { createJob, fetchJob, fetchJobResult, cancelJob } from '../../api/client';
import { subscribeToJobEvents } from '../../api/sse';
import { StatusBadge } from '../common/StatusBadge';
import { DomainStatusTag } from '../common/DomainStatusTag';
import { LogViewer } from '../common/LogViewer';
import { useToast } from '../common/Toast';
import { domainStatusLabel } from '../../ui/presentation';
import {
  Play, XCircle, CheckCircle2, Clock, RotateCcw,
  AlertCircle, ChevronDown, ChevronUp,
  Shield, Zap, Timer, Layers,
} from 'lucide-react';

import aiEngineerHeroImg from '../../assets/visuals/ai_engineer_hero.svg';

interface AIEngineerViewProps {
  selectedJobId?: string | null;
  onJobCreated?: (jobId: string) => void;
}

const OBJECTIVES = [
  {
    id: 'BASE_PREPARATION',
    label: 'Base Preparation',
    desc: 'Verify runtime compatibility and validate model configuration',
    Icon: Shield,
  },
  {
    id: 'MAX_THROUGHPUT',
    label: 'Max Throughput',
    desc: 'Optimize for highest generation tokens per second',
    Icon: Zap,
  },
  {
    id: 'MIN_LATENCY',
    label: 'Min Latency',
    desc: 'Minimize time-to-first-token for responsive inference',
    Icon: Timer,
  },
  {
    id: 'FULL_PREPARATION',
    label: 'Full Preparation',
    desc: 'Complete weight materialization when hardware is confirmed',
    Icon: Layers,
  },
];

// Map backend phase labels to product-level descriptions
const PHASE_LABELS: Record<string, string> = {
  PREFLIGHT: 'Inspecting model',
  ENVIRONMENT_CHECK: 'Checking environment',
  PLANNING: 'Planning session',
  FORGE: 'Preparing configuration',
  VALIDATION: 'Evaluating result',
  COMPLETED: 'Session complete',
  RECOMMENDATION: 'Generating recommendations',
  ERROR: 'Handling error',
};

export const AIEngineerView: React.FC<AIEngineerViewProps> = ({ selectedJobId, onJobCreated }) => {
  const toast = useToast();

  const [modelId, setModelId] = useState('Qwen/Qwen2.5-0.5B-Instruct');
  const [objective, setObjective] = useState('BASE_PREPARATION');
  const [maxAttempts, setMaxAttempts] = useState(5);
  const [timeoutMinutes, setTimeoutMinutes] = useState(10);
  const [maxDiskGb, setMaxDiskGb] = useState(10);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [activeJob, setActiveJob] = useState<JobResponse | null>(null);
  const [jobEvents, setJobEvents] = useState<JobEvent[]>([]);
  const [jobResult, setJobResult] = useState<JobResultResponse | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);

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
                        setActiveJob((prev) => prev ? { ...prev, status: res.job_status, domain_status: res.domain_status } : prev);
                      }
                    })
                    .catch(() => {});
                }
              }
            },
            onError: () => {
              if (!isCancelled) toast.warning('Live updates are interrupted. The same job is being checked directly.');
            },
            onComplete: async () => {
              try {
                const res = await fetchJobResult(selectedJobId!);
                if (!isCancelled) setJobResult(res);
              } catch (e) { console.warn(e); }
            },
          });
        }
      } catch (err) {
        console.warn('Failed to load selected job:', err);
      }
    }

    loadSelectedJob();
    return () => { isCancelled = true; if (unsubscribe) unsubscribe(); };
  }, [selectedJobId]);

  const handleLaunch = async () => {
    setIsStarting(true);
    setLaunchError(null);
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
                setActiveJob((prev) => prev ? { ...prev, status: res.job_status, domain_status: res.domain_status } : prev);
              })
              .catch(() => {});
          }
        },
        onError: () => toast.warning('Live updates are interrupted. The same session is being checked directly.'),
        onComplete: async () => {
          try {
            const res = await fetchJobResult(job.job_id);
            setJobResult(res);
            setActiveJob((prev) => prev ? { ...prev, status: res.job_status, domain_status: res.domain_status } : prev);
          } catch (e) { console.warn('Failed to fetch engineer job result:', e); }
        },
      });
    } catch (err: any) {
      const message = `Session launch failed: ${err.message}`;
      setLaunchError(message);
      toast.error(message);
    } finally {
      setIsStarting(false);
    }
  };

  const handleCancel = async () => {
    if (!activeJob) return;
    try {
      await cancelJob(activeJob.job_id);
      const job = await fetchJob(activeJob.job_id);
      setActiveJob(job);
      if (['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(job.status)) {
        const result = await fetchJobResult(activeJob.job_id);
        setJobResult(result);
        setActiveJob((prev) => prev ? { ...prev, status: result.job_status, domain_status: result.domain_status } : prev);
      }
      if (job.status === 'CANCELLED') setLaunchError('The session was cancelled. Your objective and model are preserved for retry.');
      else if (job.status !== 'SUCCEEDED') toast.info('Cancellation requested. The job is still being reconciled.');
    } catch (e: any) {
      toast.error(`Cancel failed: ${e.message}`);
    }
  };

  const isRunning = activeJob?.status === 'RUNNING' || activeJob?.status === 'QUEUED';
  const failed = jobResult?.job_status === 'FAILED' || jobResult?.domain_status === 'FAILED' || jobResult?.result?.status === 'FAILED' || activeJob?.status === 'FAILED' || activeJob?.domain_status === 'FAILED';
  const cancelled = jobResult?.job_status === 'CANCELLED' || activeJob?.status === 'CANCELLED';
  const selectedObj = OBJECTIVES.find(o => o.id === objective);
  const latestActivity = jobEvents.at(-1);
  const rawFailure = String(jobResult?.error_message || jobResult?.result?.errors_encountered?.[0] || jobResult?.result?.reasons?.[0] || launchError || 'The engineer stopped before producing a verified recommendation.');
  const failureMessage = rawFailure.includes('MODEL_NOT_FOUND') ? 'The model could not be resolved. Check repository access and the model ID, then retry.' : rawFailure;

  return (
    <div className="page-fade">
      {/* ── HERO ─────────────────────────────────────────────────────── */}
      <div className="relative overflow-hidden border-b hairline min-h-[290px] flex items-center">
        <div
          className="absolute inset-0 flex items-center justify-end opacity-25 pointer-events-none"
          style={{ overflow: 'hidden' }}
        >
          <img src={aiEngineerHeroImg} alt="" className="h-[280px] mr-12 opacity-70" />
        </div>
        <div className="absolute inset-0 bg-gradient-to-r from-[#09090b] via-[#09090b]/95 to-transparent" />
        <div className="relative z-10 page-shell py-12">
          <p className="eyebrow mb-5">Intelligent model preparation</p>
          <h1 className="page-title">Tell us the outcome.<br/><span className="text-zinc-500">Engineer the path.</span></h1>
          <p className="text-base text-zinc-500 mt-5 max-w-xl">Set an objective and let ROCmHub inspect, plan, and recommend the safest path for your model.</p>
        </div>
      </div>

      <div className="page-shell py-10 space-y-8">
        {/* ── OBJECTIVE SELECTION ──────────────────────────────────── */}
        <div className="space-y-3">
          <h2 className="text-2xl font-semibold tracking-[-.03em]">What do you want to accomplish?</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {OBJECTIVES.map((obj) => {
              const isSelected = objective === obj.id;
              const Icon = obj.Icon;
              return (
                <button
                  key={obj.id}
                  type="button"
                  onClick={() => setObjective(obj.id)}
                  className={`p-6 min-h-[190px] rounded-2xl border text-left transition-all ${
                    isSelected
                      ? 'bg-red-500/[.08] border-red-500/40 shadow-sm'
                      : 'card hover:border-zinc-600 hover:bg-surface-elevated'
                  }`}
                >
                  <div className={`mb-8 ${isSelected ? 'text-red-400' : 'text-content-muted'}`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <div className={`text-sm font-semibold mb-1 ${isSelected ? 'text-content-primary' : 'text-content-secondary'}`}>
                    {obj.label.replace('Base Preparation','Prepare this model').replace('Max Throughput','Optimize throughput').replace('Min Latency','Reduce latency').replace('Full Preparation','Analyze compatibility')}
                  </div>
                  <p className="text-[11px] text-content-muted leading-relaxed">{obj.desc}</p>
                  {isSelected && (
                    <div className="mt-2 w-full h-0.5 bg-gradient-to-r from-red-500/50 to-transparent rounded" />
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* ── MODEL + LAUNCH ──────────────────────────────────────── */}
        <div className="focus-panel p-6 md:p-8 space-y-5">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 items-end">
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-content-muted">Model ID</label>
              <input
                type="text"
                value={modelId}
                onChange={(e) => { setModelId(e.target.value); setLaunchError(null); setActiveJob(null); setJobResult(null); setJobEvents([]); }}
                className="input-field font-mono text-xs"
                placeholder="org/model-name"
              />
            </div>
            <div className="flex items-end gap-3">
              <button
                onClick={handleLaunch}
                disabled={isStarting || isRunning || !modelId.trim()}
                className="btn-primary flex-1"
              >
                {isStarting ? (
                  <><span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />Launching...</>
                ) : (
                  <><Play className="w-4 h-4 fill-current" />Start Session</>
                )}
              </button>
            </div>
          </div>

          {/* Advanced toggle */}
          <div className="pt-2 border-t border-surface-border">
            <button
              type="button"
              onClick={() => setShowAdvanced(v => !v)}
              className="flex items-center gap-1.5 text-xs text-content-muted hover:text-content-secondary transition-colors"
            >
              {showAdvanced ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              Advanced Settings
            </button>

            {showAdvanced && (
              <div className="mt-3 grid grid-cols-3 gap-3">
                <div className="space-y-1">
                  <label className="text-[11px] text-content-muted">Max Attempts</label>
                  <input
                    type="number" min={1} max={10} value={maxAttempts}
                    onChange={(e) => setMaxAttempts(parseInt(e.target.value) || 5)}
                    className="input-field text-xs font-mono"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-[11px] text-content-muted">Timeout (min)</label>
                  <input
                    type="number" min={1} max={60} value={timeoutMinutes}
                    onChange={(e) => setTimeoutMinutes(parseInt(e.target.value) || 10)}
                    className="input-field text-xs font-mono"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-[11px] text-content-muted">Disk Limit (GB)</label>
                  <input
                    type="number" min={1} max={100} value={maxDiskGb}
                    onChange={(e) => setMaxDiskGb(parseInt(e.target.value) || 10)}
                    className="input-field text-xs font-mono"
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        {launchError && !jobResult && (
          <div className="p-5 rounded-2xl bg-red-500/[.07] border border-red-500/20 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex gap-3"><AlertCircle className="w-5 h-5 text-red-400 shrink-0"/><div><div className="text-sm font-semibold text-red-200">Engineer needs your attention</div><p className="text-xs text-red-300/70 mt-1">{launchError}</p></div></div>
            <button onClick={handleLaunch} disabled={!modelId.trim()} className="btn-secondary shrink-0"><RotateCcw className="w-3.5 h-3.5"/>Retry session</button>
          </div>
        )}

        {/* ── ACTIVE SESSION ──────────────────────────────────────── */}
        {activeJob && (
          <div className="flex flex-col gap-4">
            {isRunning && !jobResult && (
              <div className="focus-panel p-6 md:p-8 order-1" aria-label="Engineer activity">
                <div className="flex items-center justify-between gap-4"><div><div className="eyebrow mb-2">Engineer activity</div><h3 className="text-xl font-semibold">Building a recommendation</h3></div><span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse"/></div>
                <div className="mt-6 rounded-xl border hairline bg-white/[.02] p-4" role="status"><div className="flex items-center gap-3"><span className="w-2 h-2 rounded-full bg-red-400 animate-pulse"/><span className="text-sm text-zinc-200">{latestActivity?.message || 'Waiting for the first activity update'}</span></div><p className="text-xs text-zinc-600 mt-2">Live activity · no completion percentage is estimated</p></div>
              </div>
            )}
            {/* Session header */}
            <div className="card p-5 order-2">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="text-sm font-semibold text-content-primary">{isRunning ? 'Engineer at work' : 'Session record'}</h3>
                  <StatusBadge status={activeJob.status} />
                  <DomainStatusTag status={activeJob.domain_status} />
                </div>
                {isRunning && (
                  <button onClick={handleCancel} className="btn-danger shrink-0">
                    <XCircle className="w-3.5 h-3.5" />
                    Cancel
                  </button>
                )}
              </div>
              <details className="mb-3"><summary className="text-[11px] text-content-muted cursor-pointer">Technical details</summary><p className="mt-2 font-mono text-[10px] text-content-muted break-all">Run ID: {activeJob.job_id}</p></details>

              {/* Agent activity — product-level view of events */}
              {jobEvents.length > 0 && (
                <div className="space-y-1.5">
                  <div className="text-xs text-content-muted font-medium mb-2">Agent Activity</div>
                  {jobEvents.slice(-6).map((evt) => (
                    <div
                      key={evt.event_id || evt.sequence}
                      className="flex items-start gap-2.5 text-xs py-1.5"
                    >
                      <div className={`w-1.5 h-1.5 rounded-full mt-1 shrink-0 ${
                        evt.status === 'FAILED' ? 'bg-red-500' :
                        evt.status === 'SUCCEEDED' || evt.phase === 'COMPLETED' ? 'bg-emerald-400' :
                        'bg-blue-400 animate-pulse'
                      }`} />
                      <div className="flex-1 min-w-0">
                        <span className="text-[11px] font-medium text-content-secondary">
                          {PHASE_LABELS[evt.phase] || evt.phase}
                        </span>
                        <span className="text-content-muted mx-1.5">—</span>
                        <span className="text-content-secondary">{evt.message}</span>
                      </div>
                      <span className="text-[10px] text-content-muted font-mono shrink-0">
                        {evt.timestamp ? new Date(evt.timestamp).toLocaleTimeString() : ''}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* Expandable full log */}
              {jobEvents.length > 0 && (
                <details className="mt-3">
                  <summary className="text-[11px] text-content-muted cursor-pointer hover:text-content-secondary transition-colors list-none flex items-center gap-1.5">
                    <ChevronDown className="w-3 h-3" />
                    View full event log ({jobEvents.length} events)
                  </summary>
                  <div className="mt-2">
                    <LogViewer events={jobEvents} />
                  </div>
                </details>
              )}
            </div>

            {/* ── RESULT ──────────────────────────────────────────── */}
            {jobResult && (failed || cancelled) && (
              <div className="focus-panel p-6 md:p-8 space-y-5 order-1">
                <div className="flex items-start gap-4"><div className="w-11 h-11 rounded-xl bg-amber-500/10 text-amber-400 flex items-center justify-center shrink-0"><AlertCircle className="w-5 h-5"/></div><div><div className="eyebrow mb-2">Recommendation unavailable</div><h3 className="text-xl font-semibold">{cancelled ? 'Session cancelled' : 'We couldn’t complete this session'}</h3><p className="text-sm text-zinc-500 mt-2 leading-relaxed">{failureMessage}</p>{rawFailure!==failureMessage&&<details className="mt-3"><summary className="text-xs text-zinc-600 cursor-pointer">Technical details</summary><p className="font-mono text-[10px] text-zinc-600 mt-2 break-all">{rawFailure}</p></details>}</div></div>
                <div className="flex flex-wrap justify-end gap-2 border-t hairline pt-5"><button onClick={()=>{setActiveJob(null);setJobResult(null);setJobEvents([]);setLaunchError(null)}} className="btn-ghost">Edit objective</button><button onClick={handleLaunch} className="btn-primary"><RotateCcw className="w-3.5 h-3.5"/>Retry session</button></div>
              </div>
            )}
            {jobResult && !failed && !cancelled && (
              <div className="card p-5 space-y-4 order-1">
                {/* Completion header */}
                <div className="flex flex-wrap items-center justify-between gap-2 pb-4 border-b border-surface-border">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                    <span className="text-sm font-semibold text-emerald-400">
                      Recommendation ready
                    </span>
                  </div>
                  <span className="text-[11px] text-content-muted font-mono">
                    {jobResult.completed_at ? new Date(jobResult.completed_at).toLocaleTimeString() : ''}
                  </span>
                </div>

                {/* Metrics row */}
                {jobResult.result && (
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <div className="p-3 rounded-lg bg-surface-elevated border border-surface-border">
                      <div className="flex items-center gap-1.5 text-[11px] text-content-muted mb-1">
                        <Clock className="w-3 h-3" /> Duration
                      </div>
                      <div className="text-sm font-bold text-content-primary font-mono">
                        {jobResult.result.total_duration_seconds !== undefined
                          ? `${Number(jobResult.result.total_duration_seconds).toFixed(1)}s`
                          : '—'}
                      </div>
                    </div>
                    <div className="p-3 rounded-lg bg-surface-elevated border border-surface-border">
                      <div className="flex items-center gap-1.5 text-[11px] text-content-muted mb-1">
                        <RotateCcw className="w-3 h-3" /> Attempts
                      </div>
                      <div className="text-sm font-bold text-content-primary font-mono">
                        {jobResult.result.attempts_used ?? '—'}
                      </div>
                    </div>
                    <div className="p-3 rounded-lg bg-surface-elevated border border-surface-border">
                      <div className="text-[11px] text-content-muted mb-1">Goal</div>
                      <div className="text-xs font-bold text-content-primary">
                        {(selectedObj?.label || objective).replace('Base Preparation', 'Prepare this model').replace('Max Throughput', 'Optimize throughput').replace('Min Latency', 'Reduce latency').replace('Full Preparation', 'Analyze compatibility')}
                      </div>
                    </div>
                    <div className="p-3 rounded-lg bg-surface-elevated border border-surface-border">
                      <div className="text-[11px] text-content-muted mb-1">Preparation</div>
                      <div className="text-xs font-bold text-emerald-400">
                        {domainStatusLabel(jobResult.result.build_manifest?.status || jobResult.domain_status) || 'Recommendation ready'}
                      </div>
                    </div>
                  </div>
                )}

                {/* Recommendations */}
                {jobResult.result?.reasons && Array.isArray(jobResult.result.reasons) && jobResult.result.reasons.length > 0 && (
                  <div className="p-4 rounded-xl bg-surface-elevated border border-surface-border space-y-2">
                    <div className="text-xs font-semibold text-content-secondary uppercase tracking-wide">Recommendations</div>
                    <ul className="space-y-2">
                      {jobResult.result.reasons.map((reason: string, idx: number) => (
                        <li key={idx} className="flex items-start gap-2 text-xs text-content-secondary">
                          <span className="text-accent-red mt-0.5 shrink-0">›</span>
                          <span className="leading-relaxed">{reason.replace('Model configuration prepared in CONFIG_ONLY mode (weights not materialized). Real AMD execution was not performed.', 'Configuration is ready. Model weights were not materialized, and AMD execution was not performed.')}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Errors */}
                {jobResult.result?.errors_encountered && Array.isArray(jobResult.result.errors_encountered) && jobResult.result.errors_encountered.length > 0 && (
                  <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-500/20 space-y-2">
                    <div className="flex items-center gap-1.5 text-xs font-semibold text-amber-400">
                      <AlertCircle className="w-3.5 h-3.5" />
                      Handled During Session
                    </div>
                    <ul className="space-y-1 text-xs text-amber-300/80">
                      {jobResult.result.errors_encountered.map((err: string, idx: number) => (
                        <li key={idx} className="flex items-start gap-2">
                          <span className="shrink-0 mt-0.5">•</span>
                          <span>{err}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Trajectory — collapsed by default */}
                {jobResult.result?.trajectory && Array.isArray(jobResult.result.trajectory) && jobResult.result.trajectory.length > 0 && (
                  <details>
                    <summary className="text-[11px] text-content-muted cursor-pointer hover:text-content-secondary transition-colors list-none flex items-center gap-1.5">
                      <ChevronDown className="w-3 h-3" />
                      Session Trajectory ({jobResult.result.trajectory.length} steps)
                    </summary>
                    <div className="mt-3 space-y-2 max-h-72 overflow-y-auto">
                      {jobResult.result.trajectory.map((step: any, idx: number) => (
                        <div key={idx} className="p-3 rounded-lg bg-surface-elevated border border-surface-border text-xs space-y-1">
                          <div className="flex items-center justify-between gap-2">
                            <div className="flex items-center gap-2">
                              <span className="text-[10px] font-mono text-content-muted px-1.5 py-0.5 rounded bg-surface-deep border border-surface-border">
                                {step.step_index ?? idx + 1}
                              </span>
                              <span className="text-[10px] font-mono font-semibold text-violet-400">
                                {PHASE_LABELS[step.phase] || step.phase}
                              </span>
                              <span className="font-medium text-content-primary">{step.action}</span>
                            </div>
                            {step.duration_seconds !== undefined && (
                              <span className="text-content-muted text-[10px] font-mono shrink-0">
                                {Number(step.duration_seconds).toFixed(2)}s
                              </span>
                            )}
                          </div>
                          {step.observation && (
                            <div className="text-content-secondary text-[11px] pl-10 leading-relaxed border-l-2 border-surface-border">
                              {step.observation}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
