import React, { useState, useEffect } from 'react';
import type { JobResponse, JobResultResponse, JobEvent } from '../../api/types';
import { createJob, fetchJob, fetchJobResult, cancelJob } from '../../api/client';
import { subscribeToJobEvents } from '../../api/sse';
import { StatusBadge } from '../common/StatusBadge';
import { DomainStatusTag } from '../common/DomainStatusTag';
import { LogViewer } from '../common/LogViewer';
import { useToast } from '../common/Toast';
import {
  Play, XCircle, CheckCircle2, BarChart3, Info, Check, ChevronDown,
} from 'lucide-react';

import optimizationLabHeroImg from '../../assets/visuals/optimization_lab_hero.svg';

interface OptimizationLabViewProps {
  selectedJobId?: string | null;
  onJobCreated?: (jobId: string) => void;
}

const OBJECTIVES = [
  { id: 'MAX_THROUGHPUT', label: 'Max Throughput', desc: 'Tokens per second' },
  { id: 'MIN_LATENCY', label: 'Min Latency', desc: 'Time to first token' },
  { id: 'BALANCED', label: 'Balanced', desc: 'Throughput + VRAM footprint' },
];

const STRATEGIES = [
  { id: 'fp16', label: 'FP16', desc: 'Half precision · standard' },
  { id: 'bf16', label: 'BF16', desc: 'Bfloat16 · CDNA / RDNA3 native' },
  { id: 'fp32', label: 'FP32', desc: 'Single precision · reference' },
];

export const OptimizationLabView: React.FC<OptimizationLabViewProps> = ({ selectedJobId, onJobCreated }) => {
  const toast = useToast();

  const [modelId, setModelId] = useState('Qwen/Qwen2.5-0.5B-Instruct');
  const [objective, setObjective] = useState('MAX_THROUGHPUT');
  const [strategies, setStrategies] = useState<string[]>(['fp16', 'bf16']);

  const [activeJob, setActiveJob] = useState<JobResponse | null>(null);
  const [jobEvents, setJobEvents] = useState<JobEvent[]>([]);
  const [jobResult, setJobResult] = useState<JobResultResponse | null>(null);
  const [isStarting, setIsStarting] = useState(false);

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

  const toggleStrategy = (strat: string) => {
    if (strategies.includes(strat)) {
      if (strategies.length > 1) setStrategies(strategies.filter((s) => s !== strat));
    } else {
      setStrategies([...strategies, strat]);
    }
  };

  const handleLaunch = async () => {
    setIsStarting(true);
    setJobEvents([]);
    setJobResult(null);

    try {
      const job = await createJob({
        job_type: 'OPTIMIZATION',
        model_id: modelId,
        objective,
        strategies,
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
        onComplete: async () => {
          try {
            const res = await fetchJobResult(job.job_id);
            setJobResult(res);
            setActiveJob((prev) => prev ? { ...prev, status: res.job_status, domain_status: res.domain_status } : prev);
          } catch (e) { console.warn(e); }
        },
      });
    } catch (err: any) {
      toast.error(`Optimization failed: ${err.message}`);
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
      toast.error(`Cancel failed: ${e.message}`);
    }
  };

  const isRunning = activeJob?.status === 'RUNNING' || activeJob?.status === 'QUEUED';

  return (
    <div className="page-fade">
      {/* ── HERO ─────────────────────────────────────────────────────── */}
      <div className="relative overflow-hidden border-b border-surface-border">
        <div className="absolute inset-0 flex items-center justify-end pr-16 opacity-20 pointer-events-none">
          <img src={optimizationLabHeroImg} alt="" className="h-40 opacity-80" />
        </div>
        <div className="absolute inset-0 hero-overlay" />
        <div className="relative z-10 px-8 sm:px-10 py-8">
          <p className="text-xs font-mono font-medium text-emerald-400/90 uppercase tracking-wider mb-1.5">
            Strategy Comparison
          </p>
          <h1 className="text-2xl font-bold tracking-tight text-white mb-1">Optimization Lab</h1>
          <p className="text-sm text-content-secondary">
            Compare precision strategies side by side. Results reflect truthful NOT_MEASURED on non-AMD hosts.
          </p>
        </div>
      </div>

      <div className="px-8 sm:px-10 py-7 space-y-5 max-w-5xl">
        {/* ── WORKSPACE SETUP ─────────────────────────────────────── */}
        <div className="card p-6 space-y-5">
          <h2 className="text-sm font-semibold text-content-primary">Optimization Workspace</h2>

          {/* Model + Objective row */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-content-muted">Model ID</label>
              <input
                type="text"
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                className="input-field font-mono text-xs"
                placeholder="org/model-name"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-content-muted">Optimization Goal</label>
              <div className="grid grid-cols-3 gap-2">
                {OBJECTIVES.map((obj) => {
                  const isSelected = objective === obj.id;
                  return (
                    <button
                      key={obj.id}
                      type="button"
                      onClick={() => setObjective(obj.id)}
                      title={obj.desc}
                      className={`py-2 px-2 rounded-lg border text-[11px] font-medium transition-all ${
                        isSelected
                          ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-300'
                          : 'bg-surface-elevated border-surface-border text-content-secondary hover:border-zinc-600'
                      }`}
                    >
                      {obj.label}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Strategies */}
          <div className="space-y-2 pt-2 border-t border-surface-border">
            <label className="text-xs font-medium text-content-muted">Compare Strategies</label>
            <div className="flex flex-wrap gap-2">
              {STRATEGIES.map((st) => {
                const isSelected = strategies.includes(st.id);
                return (
                  <button
                    key={st.id}
                    type="button"
                    onClick={() => toggleStrategy(st.id)}
                    title={st.desc}
                    className={`flex items-center gap-2 px-3.5 py-2 rounded-lg border text-xs font-medium transition-all ${
                      isSelected
                        ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-300'
                        : 'bg-surface-elevated border-surface-border text-content-secondary hover:border-zinc-600 hover:text-content-primary'
                    }`}
                  >
                    {isSelected && <Check className="w-3 h-3 text-emerald-400" />}
                    {st.label}
                    <span className="text-[10px] text-content-muted">{st.desc.split('·')[0].trim()}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Launch */}
          <div className="flex items-center justify-between pt-2 border-t border-surface-border">
            <div className="text-[11px] text-content-muted">
              {strategies.length} strategies · {OBJECTIVES.find(o => o.id === objective)?.label}
            </div>
            <button
              onClick={handleLaunch}
              disabled={isStarting || isRunning}
              className="btn-primary"
            >
              {isStarting ? (
                <><span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />Submitting...</>
              ) : (
                <><Play className="w-4 h-4 fill-current" />Run Optimization</>
              )}
            </button>
          </div>
        </div>

        {/* ── ACTIVE JOB ──────────────────────────────────────────── */}
        {activeJob && (
          <div className="card p-5 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="text-sm font-semibold text-content-primary">Optimization Pipeline</h3>
                <span className="text-[11px] font-mono text-content-muted bg-surface-elevated px-2 py-0.5 rounded border border-surface-border">
                  {activeJob.job_id}
                </span>
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

            {/* Collapsed log */}
            <details>
              <summary className="text-[11px] text-content-muted cursor-pointer hover:text-content-secondary list-none flex items-center gap-1.5">
                <ChevronDown className="w-3 h-3" />
                Event log ({jobEvents.length})
              </summary>
              <div className="mt-2">
                <LogViewer events={jobEvents} />
              </div>
            </details>

            {/* Results */}
            {jobResult && (
              <div className="space-y-4 pt-2 border-t border-surface-border">
                {/* Completion + hardware notice */}
                <div className="flex flex-wrap items-center gap-3">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    <span className="text-sm font-semibold text-emerald-400">
                      Optimization Complete — {jobResult.domain_status || 'CONFIG_ONLY'}
                    </span>
                  </div>
                </div>

                {/* Hardware truth notice — prominent */}
                <div className="flex items-start gap-3 p-3.5 rounded-xl bg-surface-elevated border border-surface-border text-xs">
                  <Info className="w-4 h-4 text-content-muted shrink-0 mt-0.5" />
                  <p className="text-content-secondary leading-relaxed">
                    On hosts without a physical AMD GPU, throughput and latency are reported as{' '}
                    <span className="font-mono font-semibold text-amber-400">NOT_MEASURED</span>.
                    Real hardware benchmarks require a ROCm-capable GPU.
                  </p>
                </div>

                {/* Candidate Comparison Table */}
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-content-muted" />
                    <h4 className="text-sm font-semibold text-content-primary">Candidate Comparison Table</h4>
                    {jobResult.result?.best_candidate_id && (
                      <span className="px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 text-[11px] font-mono border border-emerald-500/25">
                        Best: {jobResult.result.best_candidate_id}
                      </span>
                    )}
                  </div>

                  <div className="overflow-x-auto rounded-xl border border-surface-border bg-surface-deep">
                    <table className="w-full text-left text-xs">
                      <thead className="border-b border-surface-border">
                        <tr>
                          <th className="px-4 py-3 text-content-muted font-medium">Role</th>
                          <th className="px-4 py-3 text-content-muted font-medium">Strategy</th>
                          <th className="px-4 py-3 text-content-muted font-medium">Status</th>
                          <th className="px-4 py-3 text-content-muted font-medium">Throughput</th>
                          <th className="px-4 py-3 text-content-muted font-medium">TTFT</th>
                          <th className="px-4 py-3 text-content-muted font-medium">Verdict</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-surface-border">
                        {/* Baseline */}
                        <tr className="hover:bg-surface-elevated/40 transition-colors">
                          <td className="px-4 py-3 font-semibold text-content-secondary">Baseline</td>
                          <td className="px-4 py-3 font-mono text-content-primary uppercase">
                            {jobResult.result?.baseline?.precision?.toUpperCase() || 'FP16'}
                          </td>
                          <td className="px-4 py-3">
                            <DomainStatusTag status={jobResult.result?.baseline?.status || jobResult.domain_status || 'CONFIG_ONLY'} />
                          </td>
                          <td className="px-4 py-3">
                            {jobResult.result?.baseline?.benchmark_result?.throughput_tokens_per_sec != null
                              ? <span className="font-mono">{Number(jobResult.result.baseline.benchmark_result.throughput_tokens_per_sec).toFixed(1)} tok/s</span>
                              : <DomainStatusTag status="NOT_MEASURED" />}
                          </td>
                          <td className="px-4 py-3">
                            {jobResult.result?.baseline?.benchmark_result?.ttft_ms != null
                              ? <span className="font-mono">{Number(jobResult.result.baseline.benchmark_result.ttft_ms).toFixed(1)} ms</span>
                              : <DomainStatusTag status="NOT_MEASURED" />}
                          </td>
                          <td className="px-4 py-3 text-content-muted font-mono text-[11px]">REFERENCE</td>
                        </tr>

                        {/* Candidates */}
                        {jobResult.result?.candidates && Array.isArray(jobResult.result.candidates) && jobResult.result.candidates.length > 0
                          ? jobResult.result.candidates.map((cand: any, idx: number) => {
                              const comp = jobResult.result?.comparisons?.find((c: any) => c.candidate_id === cand.candidate_id);
                              const isBest = cand.candidate_id === jobResult.result?.best_candidate_id;
                              return (
                                <tr key={cand.candidate_id || idx} className={`hover:bg-surface-elevated/40 transition-colors ${isBest ? 'bg-emerald-500/5' : ''}`}>
                                  <td className="px-4 py-3 font-semibold text-content-primary font-mono">
                                    {cand.candidate_id}
                                    {isBest && <span className="ml-2 text-[10px] text-emerald-400">★ Best</span>}
                                  </td>
                                  <td className="px-4 py-3 font-mono text-content-primary uppercase">{cand.strategy}</td>
                                  <td className="px-4 py-3"><DomainStatusTag status={cand.status || 'CONFIG_ONLY'} /></td>
                                  <td className="px-4 py-3">
                                    {cand.benchmark_result?.throughput_tokens_per_sec != null
                                      ? <span className="font-mono">{Number(cand.benchmark_result.throughput_tokens_per_sec).toFixed(1)} tok/s</span>
                                      : <DomainStatusTag status="NOT_MEASURED" />}
                                  </td>
                                  <td className="px-4 py-3">
                                    {cand.benchmark_result?.ttft_ms != null
                                      ? <span className="font-mono">{Number(cand.benchmark_result.ttft_ms).toFixed(1)} ms</span>
                                      : <DomainStatusTag status="NOT_MEASURED" />}
                                  </td>
                                  <td className="px-4 py-3 text-content-secondary font-mono text-[11px]">
                                    {comp?.verdict || (cand.measured ? 'EVALUATED' : 'NOT_MEASURED')}
                                  </td>
                                </tr>
                              );
                            })
                          : strategies.map((strat, idx) => (
                              <tr key={strat} className="hover:bg-surface-elevated/40 transition-colors">
                                <td className="px-4 py-3 font-semibold text-content-secondary font-mono">Candidate #{idx + 1}</td>
                                <td className="px-4 py-3 font-mono text-content-primary uppercase">{strat}</td>
                                <td className="px-4 py-3"><DomainStatusTag status="CONFIG_ONLY" /></td>
                                <td className="px-4 py-3"><DomainStatusTag status="NOT_MEASURED" /></td>
                                <td className="px-4 py-3"><DomainStatusTag status="NOT_MEASURED" /></td>
                                <td className="px-4 py-3 text-content-muted font-mono text-[11px]">NOT_MEASURED</td>
                              </tr>
                            ))
                        }
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
