import React, { useState, useEffect } from 'react';
import type { JobResponse, JobResultResponse, JobEvent } from '../../api/types';
import { createJob, fetchJob, fetchJobResult, cancelJob } from '../../api/client';
import { subscribeToJobEvents } from '../../api/sse';
import { StatusBadge } from '../common/StatusBadge';
import { DomainStatusTag } from '../common/DomainStatusTag';
import { LogViewer } from '../common/LogViewer';
import { useToast } from '../common/Toast';
import { DOMAIN_STATUS_COPY, domainStatusLabel, jobProgressCopy } from '../../ui/presentation';
import {
  Play, XCircle, CheckCircle2, RotateCcw,
  AlertCircle, ChevronDown, ChevronUp,
  Shield, Zap, Timer, Layers,
} from 'lucide-react';


interface AIEngineerViewProps {
  selectedJobId?: string | null;
  onJobCreated?: (jobId: string) => void;
  amdComputeAvailable?: boolean | null;
}

const OBJECTIVES = [
  {
    id: 'BASE_PREPARATION',
    label: 'Prepare this model',
    desc: 'Create a model setup for AMD hardware. No inference is run.',
    Icon: Shield,
  },
  {
    id: 'MAX_THROUGHPUT',
    label: 'Improve throughput',
    desc: 'Explore configurations aimed at generating more tokens per second.',
    Icon: Zap,
  },
  {
    id: 'MIN_LATENCY',
    label: 'Reduce latency',
    desc: 'Explore configurations aimed at faster responses.',
    Icon: Timer,
  },
  {
    id: 'FULL_PREPARATION',
    label: 'Check compatibility',
    desc: 'Understand model requirements. AMD hardware validation is not performed.',
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

export const AIEngineerView: React.FC<AIEngineerViewProps> = ({ selectedJobId, onJobCreated, amdComputeAvailable = null }) => {
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
      const message = err?.message || 'Unable to start this session.';
      setLaunchError(message);
      toast.error('The session couldn’t start. Your model and selected outcome are unchanged.');
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
    } catch {
      toast.error('The session could not be cancelled. Check Activity or retry.');
    }
  };

  const isRunning = activeJob?.status === 'RUNNING' || activeJob?.status === 'QUEUED';
  const failed = jobResult?.job_status === 'FAILED' || jobResult?.domain_status === 'FAILED' || jobResult?.result?.status === 'FAILED' || activeJob?.status === 'FAILED' || activeJob?.domain_status === 'FAILED';
  const cancelled = jobResult?.job_status === 'CANCELLED' || activeJob?.status === 'CANCELLED';
  const selectedObj = OBJECTIVES.find(o => o.id === objective);
  const latestActivity = jobEvents.at(-1);
  const rawFailure = String(jobResult?.error_message || jobResult?.result?.errors_encountered?.[0] || jobResult?.result?.reasons?.[0] || launchError || 'The engineer stopped before producing a verified recommendation.');

  return (
    <div className="page-fade engineer-page">
      <header className="engineer-header page-shell"><div><div className="eyebrow">Guided model workflow</div><h1>AI Engineer</h1><p>Tell us what you want to achieve. Review the recommendation before preparing anything.</p></div></header>
      <div className="page-shell py-8 space-y-8">
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
                  disabled={(obj.id === 'MAX_THROUGHPUT' || obj.id === 'MIN_LATENCY') && amdComputeAvailable !== true}
                  aria-pressed={isSelected}
                  onClick={() => setObjective(obj.id)}
                  className={`option-selector objective-selector p-6 min-h-[190px] text-left ${isSelected ? 'is-selected' : ''}`}
                >
                  <div className={`mb-8 ${isSelected ? 'text-red-400' : 'text-content-muted'}`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <div className={`text-sm font-semibold mb-1 ${isSelected ? 'text-content-primary' : 'text-content-secondary'}`}>
                  {obj.label}
                </div>
                  <p className="text-sm text-content-muted leading-relaxed">{obj.desc}</p>
                  {(obj.id === 'MAX_THROUGHPUT' || obj.id === 'MIN_LATENCY') && amdComputeAvailable !== true && <span id={`objective-availability-${obj.id}`} className="availability-note">{amdComputeAvailable === false ? 'Requires AMD compute' : 'Compute availability unknown'}</span>}
                  {isSelected && (
                    <div className="mt-2 w-full selection-indicator" />
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* ── MODEL + LAUNCH ──────────────────────────────────────── */}
        <div className="focus-panel engineer-request p-6 md:p-8 space-y-5">
          <div className="engineer-selected-outcome" aria-live="polite"><span className="eyebrow">Selected outcome</span><strong>{selectedObj?.label || 'Prepare this model'}</strong><p>{selectedObj?.desc}</p></div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 items-end">
            <div className="space-y-1.5">
              <label className="text-sm font-medium text-content-muted">Public Hugging Face model ID</label>
              <input
                type="text"
                value={modelId}
                onChange={(e) => { setModelId(e.target.value); setLaunchError(null); setActiveJob(null); setJobResult(null); setJobEvents([]); }}
                className="input-field font-mono text-sm"
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
                  <><Play className="w-4 h-4 fill-current" />Get recommendation</>
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
            <div className="flex gap-3"><AlertCircle className="w-5 h-5 text-red-400 shrink-0"/><div><div className="text-sm font-semibold text-red-200">The session couldn’t start</div><p className="text-sm text-zinc-500 mt-1">Your model and selected outcome are unchanged. You can retry or adjust the request.</p><details className="technical-details"><summary>Technical details</summary><p>{launchError}</p></details></div></div>
            <button onClick={handleLaunch} disabled={!modelId.trim()} className="btn-secondary shrink-0"><RotateCcw className="w-3.5 h-3.5"/>Retry session</button>
          </div>
        )}

        {/* ── ACTIVE SESSION ──────────────────────────────────────── */}
        {activeJob && (
          <div className="flex flex-col gap-4">
            {isRunning && !jobResult && (
              <div className="focus-panel engineer-progress p-6 md:p-8 order-1" aria-label="Engineer activity">
                <div className="flex items-center justify-between gap-4"><div><div className="eyebrow mb-2">Progress</div><h3 className="text-xl font-semibold">Preparing your recommendation</h3></div><span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse"/></div>
                <div className="engineer-live-status mt-6 rounded-xl border hairline bg-white/[.02] p-4" role="status"><div className="flex items-center gap-3"><span className="w-2 h-2 rounded-full bg-red-400 animate-pulse"/><span className="text-sm text-zinc-200">{activeJob.status === 'QUEUED' ? 'Waiting for compatible compute. Your request is saved.' : latestActivity ? `${jobProgressCopy(latestActivity.status, latestActivity.phase)} your request.` : 'Your request is in progress. Checking for the first update.'}</span></div><p className="text-sm text-zinc-600 mt-2">Live updates may pause temporarily; the same request is reconciled. No completion percentage is estimated.</p></div>
              </div>
            )}
            {/* Session header */}
            <div className="card engineer-session p-5 order-2">
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

              {/* Worker event feed, expressed as product-level progress */}
              {jobEvents.length > 0 && (
                <div className="space-y-1.5">
                  <div className="text-sm text-content-muted font-medium mb-2">Recent updates</div>
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
                          {PHASE_LABELS[evt.phase] || 'Working on your request'}
                        </span>
                        <span className="text-content-muted mx-1.5">—</span>
                        <span className="text-content-secondary">{jobProgressCopy(evt.status, evt.phase)}</span>
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
              <div className="focus-panel engineer-executive-result p-6 md:p-8 space-y-5 order-1">
                <div className="flex items-start gap-4"><div className="w-11 h-11 rounded-xl bg-amber-500/10 text-amber-400 flex items-center justify-center shrink-0"><AlertCircle className="w-5 h-5"/></div><div><div className="eyebrow mb-2">Recommendation unavailable</div><h3 className="text-xl font-semibold">{cancelled ? 'Session cancelled' : 'We couldn’t complete this session'}</h3><p className="text-sm text-zinc-500 mt-2 leading-relaxed">The session ended before a recommendation was recorded. Your model and selected outcome are preserved for retry.</p><details className="technical-details mt-3"><summary>Technical details</summary><p>{rawFailure}</p></details></div></div>
                <div className="flex flex-wrap justify-end gap-2 border-t hairline pt-5"><button onClick={()=>{setActiveJob(null);setJobResult(null);setJobEvents([]);setLaunchError(null)}} className="btn-ghost">Edit objective</button><button onClick={handleLaunch} className="btn-primary"><RotateCcw className="w-3.5 h-3.5"/>Retry session</button></div>
              </div>
            )}
            {jobResult && !failed && !cancelled && (
              <div className="card engineer-executive-result p-5 space-y-4 order-1">
                {/* Completion header */}
                <div className="flex flex-wrap items-center justify-between gap-2 pb-4 border-b border-surface-border">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                    <span className="text-sm font-semibold text-content-primary">
                      Recommendation ready
                    </span>
                  </div>
                  <span className="text-[11px] text-content-muted font-mono">
                    {jobResult.completed_at ? new Date(jobResult.completed_at).toLocaleTimeString() : ''}
                  </span>
                </div>

                  <div className="engineer-result-summary p-5 border-l-2 border-red-500/50 bg-surface-elevated">
                  <div className="text-sm font-semibold text-content-primary">{selectedObj?.label || 'Guidance'}</div>
                  <p className="text-sm text-content-secondary mt-1">Review the guidance below. A recommendation does not itself run, validate, or measure the model.</p>
                  {jobResult.domain_status && <div className="text-sm text-content-secondary mt-3">{domainStatusLabel(jobResult.domain_status)} · {DOMAIN_STATUS_COPY[jobResult.domain_status].explanation}</div>}
                </div>
                {jobResult.result && <details className="technical-details"><summary>Technical session details</summary><dl className="mt-3 grid sm:grid-cols-3 gap-3"><div><dt>Session duration</dt><dd>{jobResult.result.total_duration_seconds !== undefined ? `${Number(jobResult.result.total_duration_seconds).toFixed(1)}s` : 'Not reported'}</dd></div><div><dt>Attempts</dt><dd>{jobResult.result.attempts_used ?? 'Not reported'}</dd></div><div><dt>Recorded status</dt><dd>{domainStatusLabel(jobResult.result.build_manifest?.status || jobResult.domain_status) || 'Not reported'}</dd></div></dl></details>}

                {/* Recommendations */}
                {jobResult.result?.reasons && Array.isArray(jobResult.result.reasons) && jobResult.result.reasons.length > 0 && (
                  <div className="p-4 rounded-xl bg-surface-elevated border border-surface-border space-y-2">
                    <div className="text-sm font-semibold text-content-secondary">Recommended next steps</div>
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
                  <details className="technical-details">
                    <summary><span className="inline-flex items-center gap-1.5"><AlertCircle className="w-3.5 h-3.5"/>Technical details · {jobResult.result.errors_encountered.length}</span></summary>
                    <ul className="mt-3 space-y-1 text-xs text-content-secondary">
                      {jobResult.result.errors_encountered.map((err: string, idx: number) => (
                        <li key={idx} className="flex items-start gap-2">
                          <span className="shrink-0 mt-0.5">•</span>
                          <span>{err}</span>
                        </li>
                      ))}
                    </ul>
                  </details>
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
