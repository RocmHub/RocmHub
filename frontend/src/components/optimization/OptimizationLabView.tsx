import React, { useState, useEffect } from 'react';
import type { JobResponse, JobResultResponse, JobEvent } from '../../api/types';
import { createJob, fetchJob, fetchJobResult, cancelJob } from '../../api/client';
import { subscribeToJobEvents } from '../../api/sse';
import { StatusBadge } from '../common/StatusBadge';
import { DomainStatusTag } from '../common/DomainStatusTag';
import { LogViewer } from '../common/LogViewer';
import { Gauge, Play, XCircle, BarChart3, Sliders, Info, ShieldCheck, Check, Sparkles } from 'lucide-react';

import optimizationLabHeroImg from '../../assets/visuals/optimization_lab_hero.svg';

interface OptimizationLabViewProps {
  selectedJobId?: string | null;
  onJobCreated?: (jobId: string) => void;
}

export const OptimizationLabView: React.FC<OptimizationLabViewProps> = ({ selectedJobId, onJobCreated }) => {
  const [modelId, setModelId] = useState('Qwen/Qwen2.5-0.5B-Instruct');
  const [objective, setObjective] = useState('MAX_THROUGHPUT');
  const [strategies, setStrategies] = useState<string[]>(['fp16', 'bf16']);

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

  const toggleStrategy = (strat: string) => {
    if (strategies.includes(strat)) {
      if (strategies.length > 1) {
        setStrategies(strategies.filter((s) => s !== strat));
      }
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
            console.warn('Failed to fetch optimization job result:', e);
          }
        },
      });
    } catch (err: any) {
      alert(`Optimization job failed: ${err.message}`);
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

  const CANDIDATE_STRATEGIES = [
    { id: 'fp16', label: 'FP16 (Half Precision)', desc: 'Standard float16 tensor execution' },
    { id: 'bf16', label: 'BF16 (Bfloat16)', desc: 'RDNA3 & CDNA native bfloat16 range' },
    { id: 'fp32', label: 'FP32 (Single Precision)', desc: 'Reference baseline single float' },
  ];

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Visual Header Banner */}
      <div className="relative rounded-xl overflow-hidden border border-surface-border bg-surface shadow-xl">
        <div
          className="absolute inset-0 bg-cover bg-center opacity-40 mix-blend-luminosity"
          style={{ backgroundImage: `url(${optimizationLabHeroImg})` }}
        />
        <div className="absolute inset-0 hero-overlay" />

        <div className="relative p-6 space-y-2">
          <div className="inline-flex items-center space-x-2 px-2.5 py-1 rounded-full bg-accent-red/10 border border-accent-red/30 text-[11px] font-mono text-red-400">
            <Gauge className="w-3.5 h-3.5 text-accent-red" />
            <span>Comparative Dyno &amp; Precision Profiler</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Optimization Lab</h1>
          <p className="text-xs text-zinc-300 max-w-2xl font-sans leading-relaxed">
            Generate, benchmark, and compare execution candidate strategies against baseline on AMD ROCm. On non-AMD hosts, status reflects truthful NOT_MEASURED.
          </p>
        </div>
      </div>

      {/* Task Creation Form */}
      <div className="p-6 rounded-xl bg-surface border border-surface-border shadow-sm space-y-5">
        <h2 className="text-sm font-semibold text-content-primary flex items-center space-x-2">
          <Sliders className="w-4 h-4 text-accent-red" />
          <span>Optimization Run Configuration</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <label className="text-xs font-mono text-content-secondary">Model ID</label>
            <input
              type="text"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              className="w-full bg-background border border-surface-border rounded-lg px-3.5 py-2.5 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red focus:ring-1 focus:ring-accent-red transition-all"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-mono text-content-secondary">Primary Target Metric</label>
            <select
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              className="w-full bg-background border border-surface-border rounded-lg px-3 py-2.5 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red focus:ring-1 focus:ring-accent-red transition-all"
            >
              <option value="MAX_THROUGHPUT">MAX_THROUGHPUT (Tokens / sec)</option>
              <option value="MIN_LATENCY">MIN_LATENCY (Time to First Token - TTFT)</option>
              <option value="BALANCED">BALANCED (Throughput &amp; VRAM Footprint)</option>
            </select>
          </div>
        </div>

        {/* Strategies multi-selection cards */}
        <div className="space-y-2 pt-2 border-t border-surface-border">
          <label className="text-xs font-mono text-content-secondary">Candidate Exploration Strategies</label>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
            {CANDIDATE_STRATEGIES.map((st) => {
              const isSelected = strategies.includes(st.id);
              return (
                <button
                  key={st.id}
                  type="button"
                  onClick={() => toggleStrategy(st.id)}
                  className={`p-3 rounded-lg border text-left transition-all font-mono ${
                    isSelected
                      ? 'bg-accent-red/10 border-accent-red text-white shadow-sm'
                      : 'bg-surface-elevated/70 border-surface-border text-zinc-400 hover:text-zinc-200 hover:border-zinc-700'
                  }`}
                >
                  <div className="flex items-center justify-between text-xs font-semibold mb-1">
                    <span>{st.label}</span>
                    {isSelected && <Check className="w-3.5 h-3.5 text-accent-red" />}
                  </div>
                  <div className="text-[10px] text-zinc-500 font-sans leading-tight">{st.desc}</div>
                </button>
              );
            })}
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-surface-border">
          <div className="flex items-center space-x-2 text-[11px] font-mono text-zinc-500">
            <ShieldCheck className="w-3.5 h-3.5 text-zinc-400" />
            <span>Integrity: Zero synthetic performance metrics. Fail-closed on non-AMD hosts.</span>
          </div>
          <button
            onClick={handleLaunch}
            disabled={isStarting || activeJob?.status === 'RUNNING'}
            className="flex items-center space-x-2 px-5 py-2.5 rounded-lg bg-accent-red hover:bg-accent-red-hover text-white text-xs font-semibold transition-all disabled:opacity-50 shadow-md shadow-red-950/40 hover:scale-[1.02] active:scale-[0.98]"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>{isStarting ? 'Submitting Run...' : 'Run Optimization Job'}</span>
          </button>
        </div>
      </div>

      {/* Active Job & SSE Progress */}
      {activeJob && (
        <div className="p-6 rounded-xl bg-surface border border-surface-border shadow-lg space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-2.5">
              <h3 className="text-sm font-semibold text-content-primary">Optimization Pipeline</h3>
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
                <span>Cancel</span>
              </button>
            )}
          </div>

          <LogViewer events={jobEvents} />

          {/* Results Comparison Table */}
          {jobResult && (
            <div className="space-y-4 pt-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center space-x-2">
                  <BarChart3 className="w-4 h-4 text-accent-red" />
                  <h4 className="text-xs font-mono font-semibold text-content-primary uppercase tracking-wider">
                    Candidate Comparison Table
                  </h4>
                  {jobResult.result?.best_candidate_id && (
                    <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 text-[10px] font-mono border border-emerald-500/30 flex items-center space-x-1">
                      <Sparkles className="w-3 h-3" />
                      <span>Best: {jobResult.result.best_candidate_id}</span>
                    </span>
                  )}
                </div>
                <span className="text-xs font-mono text-zinc-500">
                  Domain Status: {jobResult.domain_status || 'CONFIG_ONLY'}
                </span>
              </div>

              {/* Informative Hardware Callout */}
              <div className="p-3.5 rounded-lg bg-surface-elevated/70 border border-surface-border text-xs flex items-start space-x-3">
                <Info className="w-4 h-4 text-zinc-400 shrink-0 mt-0.5" />
                <div className="text-[11px] text-zinc-400 leading-relaxed font-sans">
                  On hosts without physical AMD GPU accelerators, throughput (tok/s) and TTFT are reported strictly as{' '}
                  <span className="font-mono text-amber-400 font-semibold">NOT_MEASURED</span>. Actual hardware benchmarking is verified through{' '}
                  <span className="font-mono text-zinc-200">FIRST_AMD_RUN.md</span> on physical ROCm hardware.
                </div>
              </div>

              {/* Table */}
              <div className="overflow-x-auto rounded-xl border border-surface-border bg-[#121216] shadow-sm">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-[#181820] text-zinc-400 border-b border-surface-border">
                    <tr>
                      <th className="px-4 py-3">Role</th>
                      <th className="px-4 py-3">Strategy</th>
                      <th className="px-4 py-3">Status</th>
                      <th className="px-4 py-3">Throughput</th>
                      <th className="px-4 py-3">TTFT</th>
                      <th className="px-4 py-3">Verdict</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-surface-border">
                    {/* Baseline row */}
                    <tr className="hover:bg-surface-elevated/40 transition-colors">
                      <td className="px-4 py-3 font-semibold text-zinc-200 flex items-center space-x-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-zinc-500" />
                        <span>Baseline</span>
                      </td>
                      <td className="px-4 py-3 text-zinc-400">
                        {jobResult.result?.baseline?.precision?.toUpperCase() || 'FP16'} (Standard)
                      </td>
                      <td className="px-4 py-3">
                        <DomainStatusTag status={jobResult.result?.baseline?.status || jobResult.domain_status || 'CONFIG_ONLY'} />
                      </td>
                      <td className="px-4 py-3 text-zinc-500">
                        {jobResult.result?.baseline?.benchmark_result?.throughput_tokens_per_sec != null
                          ? `${Number(jobResult.result.baseline.benchmark_result.throughput_tokens_per_sec).toFixed(1)} tok/s`
                          : <DomainStatusTag status="NOT_MEASURED" />}
                      </td>
                      <td className="px-4 py-3 text-zinc-500">
                        {jobResult.result?.baseline?.benchmark_result?.ttft_ms != null
                          ? `${Number(jobResult.result.baseline.benchmark_result.ttft_ms).toFixed(1)} ms`
                          : <DomainStatusTag status="NOT_MEASURED" />}
                      </td>
                      <td className="px-4 py-3 text-zinc-400">REFERENCE</td>
                    </tr>

                    {/* Explored Candidates */}
                    {jobResult.result?.candidates && Array.isArray(jobResult.result.candidates) && jobResult.result.candidates.length > 0
                      ? jobResult.result.candidates.map((cand: any, idx: number) => {
                          const comp = jobResult.result?.comparisons?.find(
                            (c: any) => c.candidate_id === cand.candidate_id
                          );
                          return (
                            <tr key={cand.candidate_id || idx} className="hover:bg-surface-elevated/40 transition-colors">
                              <td className="px-4 py-3 font-semibold text-zinc-300">
                                {cand.candidate_id}
                              </td>
                              <td className="px-4 py-3 uppercase text-zinc-200">{cand.strategy}</td>
                              <td className="px-4 py-3">
                                <DomainStatusTag status={cand.status || 'CONFIG_ONLY'} />
                              </td>
                              <td className="px-4 py-3 text-zinc-500">
                                {cand.benchmark_result?.throughput_tokens_per_sec != null
                                  ? `${Number(cand.benchmark_result.throughput_tokens_per_sec).toFixed(1)} tok/s`
                                  : <DomainStatusTag status="NOT_MEASURED" />}
                              </td>
                              <td className="px-4 py-3 text-zinc-500">
                                {cand.benchmark_result?.ttft_ms != null
                                  ? `${Number(cand.benchmark_result.ttft_ms).toFixed(1)} ms`
                                  : <DomainStatusTag status="NOT_MEASURED" />}
                              </td>
                              <td className="px-4 py-3 text-zinc-300">
                                {comp?.verdict || (cand.measured ? 'EVALUATED' : 'NOT_MEASURED')}
                              </td>
                            </tr>
                          );
                        })
                      : strategies.map((strat, idx) => (
                          <tr key={strat} className="hover:bg-surface-elevated/40 transition-colors">
                            <td className="px-4 py-3 font-semibold text-zinc-300">Candidate #{idx + 1}</td>
                            <td className="px-4 py-3 uppercase text-zinc-200">{strat}</td>
                            <td className="px-4 py-3">
                              <DomainStatusTag status="CONFIG_ONLY" />
                            </td>
                            <td className="px-4 py-3 text-zinc-500">
                              <DomainStatusTag status="NOT_MEASURED" />
                            </td>
                            <td className="px-4 py-3 text-zinc-500">
                              <DomainStatusTag status="NOT_MEASURED" />
                            </td>
                            <td className="px-4 py-3 text-zinc-400">NOT_MEASURED</td>
                          </tr>
                        ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
