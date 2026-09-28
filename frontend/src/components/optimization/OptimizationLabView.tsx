import React, { useState, useEffect } from 'react';
import type { JobResponse, JobResultResponse, JobEvent } from '../../api/types';
import { createJob, fetchJob, fetchJobResult, cancelJob } from '../../api/client';
import { subscribeToJobEvents } from '../../api/sse';
import { StatusBadge } from '../common/StatusBadge';
import { DomainStatusTag } from '../common/DomainStatusTag';
import { DOMAIN_STATUS_COPY } from '../../ui/presentation';
import { resultLabel } from '../../ui/presentation';
import { LogViewer } from '../common/LogViewer';
import { useToast } from '../common/Toast';
import {
  Play, XCircle, CheckCircle2, BarChart3, Info, Check, ChevronDown,
} from 'lucide-react';

import optimizationLabHeroImg from '../../assets/visuals/optimization_lab_hero.svg';

interface OptimizationLabViewProps {
  selectedJobId?: string | null;
  onJobCreated?: (jobId: string) => void;
  amdComputeAvailable: boolean | null;
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

const friendlyError = (message: string) => message.includes('MODEL_NOT_FOUND')
  ? 'The model could not be resolved. Check repository access and the model ID, then retry.'
  : message;

export const OptimizationLabView: React.FC<OptimizationLabViewProps> = ({ selectedJobId, onJobCreated, amdComputeAvailable }) => {
  const toast = useToast();

  const [modelId, setModelId] = useState('Qwen/Qwen2.5-0.5B-Instruct');
  const [objective, setObjective] = useState('MAX_THROUGHPUT');
  const [strategies, setStrategies] = useState<string[]>(['fp16', 'bf16']);

  const [activeJob, setActiveJob] = useState<JobResponse | null>(null);
  const [jobEvents, setJobEvents] = useState<JobEvent[]>([]);
  const [jobResult, setJobResult] = useState<JobResultResponse | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);

  // This screen currently has no weight-download consent/materialization step.
  // Keep its jobs preparation-only even when AMD compute is detected.
  const allowFullWeights = false;
  const canMeasureOnAmd = amdComputeAvailable === true && allowFullWeights;
  const capabilityMessage = amdComputeAvailable === false
    ? 'No AMD compute is available to this service. You can prepare a comparison study, but it will not run inference or collect hardware performance measurements. Metrics will be marked Not measured.'
    : amdComputeAvailable === true
      ? 'AMD compute is detected, but this study does not download model weights or start inference. It prepares comparison configurations only; no hardware performance measurements are collected.'
      : 'AMD compute availability could not be confirmed. This action prepares comparison configurations only and does not start inference; metrics will be marked Not measured.';

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

  const toggleStrategy = (strat: string) => {
    if (strategies.includes(strat)) {
      if (strategies.length > 1) setStrategies(strategies.filter((s) => s !== strat));
    } else {
      setStrategies([...strategies, strat]);
    }
  };

  const handleLaunch = async () => {
    setIsStarting(true);
    setLaunchError(null);
    setJobEvents([]);
    setJobResult(null);

    try {
      const job = await createJob({
        job_type: 'OPTIMIZATION',
        model_id: modelId,
        objective,
        strategies,
        allow_full_weights: allowFullWeights,
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
        onError: () => toast.warning('Live updates are interrupted. The same experiment is being checked directly.'),
        onComplete: async () => {
          try {
            const res = await fetchJobResult(job.job_id);
            setJobResult(res);
            setActiveJob((prev) => prev ? { ...prev, status: res.job_status, domain_status: res.domain_status } : prev);
          } catch (e) { console.warn(e); }
        },
      });
    } catch (err: any) {
      const message = `Optimization failed: ${err.message}`;
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
      if (job.status === 'CANCELLED') setLaunchError('The experiment was cancelled. The study design is preserved for retry.');
      else if (job.status !== 'SUCCEEDED') toast.info('Cancellation requested. The job is still being reconciled.');
    } catch (e: any) {
      toast.error(`Cancel failed: ${e.message}`);
    }
  };

  const isRunning = activeJob?.status === 'RUNNING' || activeJob?.status === 'QUEUED';
  const failed = jobResult?.job_status === 'FAILED' || jobResult?.domain_status === 'FAILED' || jobResult?.result?.status === 'FAILED' || activeJob?.status === 'FAILED' || activeJob?.domain_status === 'FAILED';
  const cancelled = jobResult?.job_status === 'CANCELLED' || activeJob?.status === 'CANCELLED';
  const failureDetail = jobResult?.error_message || launchError || 'The experiment stopped before baseline and candidate preparation completed.';

  return (
    <div className="page-fade">
      {/* ── HERO ─────────────────────────────────────────────────────── */}
      <div className="relative overflow-hidden border-b hairline min-h-[260px] flex items-center">
        <div className="absolute inset-0 flex items-center justify-end pr-16 opacity-20 pointer-events-none">
          <img src={optimizationLabHeroImg} alt="" className="h-40 opacity-80" />
        </div>
        <div className="absolute inset-0 bg-gradient-to-r from-[#09090b] via-[#09090b]/95 to-transparent" />
        <div className="relative z-10 page-shell py-12">
          <p className="eyebrow mb-5">Experiment workspace</p>
          <h1 className="page-title">Compare strategies.<br/><span className="text-zinc-500">Keep the evidence.</span></h1>
          <p className="text-base text-zinc-500 mt-5 max-w-xl">Design a fair baseline and candidate study before moving it onto AMD hardware.</p>
        </div>
      </div>

      <div className="page-shell py-10 space-y-6">
        {/* ── WORKSPACE SETUP ─────────────────────────────────────── */}
        <div className="focus-panel p-6 md:p-8 space-y-7">
          <div role="note" aria-label="Hardware measurement availability" className="rounded-xl border border-amber-500/20 bg-amber-500/[.05] p-4 md:p-5">
            <div className="text-sm font-semibold text-amber-200">Comparison preparation only</div>
            <p className="text-sm text-zinc-400 mt-1 leading-relaxed">{capabilityMessage}</p>
          </div>
          <div><div className="eyebrow mb-3">Study design</div><h2 className="text-2xl font-semibold tracking-[-.03em]">Baseline versus candidates</h2></div>

          {/* Model + Objective row */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
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
            <div className="grid sm:grid-cols-3 gap-3">
              {STRATEGIES.map((st) => {
                const isSelected = strategies.includes(st.id);
                return (
                  <button
                    key={st.id}
                    type="button"
                    onClick={() => toggleStrategy(st.id)}
                    title={st.desc}
                    className={`min-h-[130px] flex flex-col items-start justify-between gap-4 p-5 rounded-2xl border text-sm font-medium transition-all ${
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
              disabled={isStarting || isRunning || !modelId.trim()}
              className="btn-primary"
            >
              {isStarting ? (
                <><span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />Submitting...</>
              ) : (
                <><Play className="w-4 h-4 fill-current" />{canMeasureOnAmd ? 'Run comparison on AMD' : 'Prepare comparison'}</>
              )}
            </button>
          </div>
        </div>

        {launchError && !jobResult && (
          <div className="p-5 rounded-2xl bg-red-500/[.07] border border-red-500/20 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex gap-3"><Info className="w-5 h-5 text-red-400 shrink-0"/><div><div className="text-sm font-semibold text-red-200">Experiment needs attention</div><p className="text-xs text-red-300/70 mt-1">{launchError}</p></div></div>
            <button onClick={handleLaunch} disabled={!modelId.trim()} className="btn-secondary shrink-0">Retry experiment</button>
          </div>
        )}

        {/* ── ACTIVE JOB ──────────────────────────────────────────── */}
        {activeJob && (
          <div className="card p-5 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="text-sm font-semibold text-content-primary">{isRunning ? 'Experiment running' : 'Experiment record'}</h3>
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
            <details><summary className="text-[11px] text-content-muted cursor-pointer">Technical details</summary><p className="mt-2 font-mono text-[10px] text-content-muted break-all">Run ID: {activeJob.job_id}</p></details>

            {isRunning && (
              <div className="grid md:grid-cols-3 gap-3" aria-label="Experiment activity">
                {[{name:'Baseline',strategy:'FP16'},...strategies.slice(0,2).map((strategy,index)=>({name:`Candidate ${String.fromCharCode(65+index)}`,strategy:strategy.toUpperCase()}))].map((lane,index)=><div key={lane.name} className="p-5 rounded-2xl bg-white/[.025] border hairline"><div className="flex items-center justify-between"><span className="text-sm font-semibold">{lane.name}</span><span className="font-mono text-[10px] text-zinc-500">{lane.strategy}</span></div><div className="flex items-center gap-2 mt-7 text-[11px] text-zinc-400"><span className="w-1.5 h-1.5 rounded-full bg-emerald-400/80 animate-pulse"/>{index===0?'Preparing reference':'Preparing candidate'}</div><div className="text-[11px] text-zinc-600 mt-2">Waiting for result</div></div>)}
              </div>
            )}

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
                    {jobResult.job_status === 'FAILED'
                      ? <Info className="w-4 h-4 text-amber-400" />
                      : <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
                    <span className={`text-sm font-semibold ${jobResult.job_status === 'FAILED' ? 'text-amber-300' : 'text-emerald-400'}`}>
                      {jobResult.job_status === 'FAILED'
                        ? 'The optimization study could not complete'
                        : `Optimization study completed${jobResult.domain_status ? ` · ${DOMAIN_STATUS_COPY[jobResult.domain_status].label}` : ''}`}
                    </span>
                  </div>
                </div>

                {(failed || cancelled) && (
                  <div className="p-6 rounded-2xl bg-amber-500/[.06] border border-amber-500/20 flex flex-col md:flex-row md:items-center justify-between gap-5"><div><div className="text-lg font-semibold">{cancelled?'Experiment cancelled':'No comparison was produced'}</div><p className="text-sm text-zinc-500 mt-2 max-w-2xl leading-relaxed">{friendlyError(failureDetail)}</p>{friendlyError(failureDetail)!==failureDetail&&<details className="mt-3"><summary className="text-[11px] text-zinc-500 cursor-pointer">Technical details</summary><p className="mt-2 max-w-2xl font-mono text-[10px] text-zinc-600 break-all">{failureDetail}</p></details>}</div><button onClick={handleLaunch} className="btn-secondary shrink-0">Retry experiment</button></div>
                )}

                {/* Hardware truth notice — prominent */}
                {!failed && !cancelled && <>
                <div className="flex items-start gap-3 p-3.5 rounded-xl bg-surface-elevated border border-surface-border text-xs">
                  <Info className="w-4 h-4 text-content-muted shrink-0 mt-0.5" />
                  <p className="text-content-secondary leading-relaxed">
                    On hosts without a physical AMD GPU, throughput and latency are reported as{' '}
                    <span className="font-semibold text-amber-400">not measured</span>.
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
                          <th className="px-2 sm:px-4 py-3 text-content-muted font-medium">Role</th>
                          <th className="px-2 sm:px-4 py-3 text-content-muted font-medium">Strategy</th>
                          <th className="px-2 sm:px-4 py-3 text-content-muted font-medium">Status</th>
                          <th className="hidden sm:table-cell px-4 py-3 text-content-muted font-medium">Throughput</th>
                          <th className="hidden sm:table-cell px-4 py-3 text-content-muted font-medium">TTFT</th>
                          <th className="hidden sm:table-cell px-4 py-3 text-content-muted font-medium">Verdict</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-surface-border">
                        {/* Baseline */}
                        <tr className="hover:bg-surface-elevated/40 transition-colors">
                          <td className="px-2 sm:px-4 py-3 font-semibold text-content-secondary">Baseline<div className="sm:hidden mt-1 text-[9px] font-normal text-content-muted">Reference metrics</div></td>
                          <td className="px-2 sm:px-4 py-3 font-mono text-content-primary uppercase">
                            {jobResult.result?.baseline?.precision?.toUpperCase() || 'FP16'}
                          </td>
                          <td className="px-2 sm:px-4 py-3">
                            <DomainStatusTag status={jobResult.result?.baseline?.status || jobResult.domain_status || 'CONFIG_ONLY'} />
                          </td>
                          <td className="hidden sm:table-cell px-4 py-3">
                            {jobResult.result?.baseline?.benchmark_result?.throughput_tokens_per_sec != null
                              ? <span className="font-mono">{Number(jobResult.result.baseline.benchmark_result.throughput_tokens_per_sec).toFixed(1)} tok/s</span>
                              : <DomainStatusTag status="NOT_MEASURED" />}
                          </td>
                          <td className="hidden sm:table-cell px-4 py-3">
                            {jobResult.result?.baseline?.benchmark_result?.ttft_ms != null
                              ? <span className="font-mono">{Number(jobResult.result.baseline.benchmark_result.ttft_ms).toFixed(1)} ms</span>
                              : <DomainStatusTag status="NOT_MEASURED" />}
                          </td>
                          <td className="hidden sm:table-cell px-4 py-3 text-content-muted text-[11px]">Reference</td>
                        </tr>

                        {/* Candidates */}
                        {jobResult.result?.candidates && Array.isArray(jobResult.result.candidates) && jobResult.result.candidates.length > 0
                          ? jobResult.result.candidates.map((cand: any, idx: number) => {
                              const comp = jobResult.result?.comparisons?.find((c: any) => c.candidate_id === cand.candidate_id);
                              const isBest = cand.candidate_id === jobResult.result?.best_candidate_id;
                              return (
                                <tr key={cand.candidate_id || idx} className={`hover:bg-surface-elevated/40 transition-colors ${isBest ? 'bg-emerald-500/5' : ''}`}>
                                  <td className="px-2 sm:px-4 py-3 font-semibold text-content-primary font-mono break-all">
                                    {cand.candidate_id}
                                    {isBest && <span className="ml-2 text-[10px] text-emerald-400">★ Best</span>}
                                    <div className="sm:hidden mt-1 text-[9px] font-normal text-content-muted break-normal">{cand.measured?'Measured candidate':'Metrics pending hardware'}</div>
                                  </td>
                                  <td className="px-2 sm:px-4 py-3 font-mono text-content-primary uppercase">{cand.strategy}</td>
                                  <td className="px-2 sm:px-4 py-3"><DomainStatusTag status={cand.status || 'CONFIG_ONLY'} /></td>
                                  <td className="hidden sm:table-cell px-4 py-3">
                                    {cand.benchmark_result?.throughput_tokens_per_sec != null
                                      ? <span className="font-mono">{Number(cand.benchmark_result.throughput_tokens_per_sec).toFixed(1)} tok/s</span>
                                      : <DomainStatusTag status="NOT_MEASURED" />}
                                  </td>
                                  <td className="hidden sm:table-cell px-4 py-3">
                                    {cand.benchmark_result?.ttft_ms != null
                                      ? <span className="font-mono">{Number(cand.benchmark_result.ttft_ms).toFixed(1)} ms</span>
                                      : <DomainStatusTag status="NOT_MEASURED" />}
                                  </td>
                                  <td className="hidden sm:table-cell px-4 py-3 text-content-secondary font-mono text-[11px]">
                                    {resultLabel(comp?.verdict) || (cand.measured ? 'Evaluated' : 'Not measured')}
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
                                <td className="px-4 py-3 text-content-muted text-[11px]">Not measured</td>
                              </tr>
                            ))
                        }
                      </tbody>
                    </table>
                  </div>
                </div>
                </>}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
