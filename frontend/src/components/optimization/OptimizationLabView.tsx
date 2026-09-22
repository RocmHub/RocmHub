import React, { useState } from 'react';
import type { JobResponse, JobResultResponse, JobEvent } from '../../api/types';
import { createJob, fetchJobResult, cancelJob } from '../../api/client';
import { subscribeToJobEvents } from '../../api/sse';
import { StatusBadge } from '../common/StatusBadge';
import { DomainStatusTag } from '../common/DomainStatusTag';
import { LogViewer } from '../common/LogViewer';
import { Gauge, Play, XCircle } from 'lucide-react';

interface OptimizationLabViewProps {
  onJobCreated?: (jobId: string) => void;
}

export const OptimizationLabView: React.FC<OptimizationLabViewProps> = ({ onJobCreated }) => {
  const [modelId, setModelId] = useState('Qwen/Qwen2.5-0.5B-Instruct');
  const [objective, setObjective] = useState('MAX_THROUGHPUT');
  const [strategies, setStrategies] = useState<string[]>(['fp16', 'bf16']);
  const [maxCandidates] = useState(3);

  const [activeJob, setActiveJob] = useState<JobResponse | null>(null);
  const [jobEvents, setJobEvents] = useState<JobEvent[]>([]);
  const [jobResult, setJobResult] = useState<JobResultResponse | null>(null);
  const [isStarting, setIsStarting] = useState(false);

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
        max_candidates: maxCandidates,
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
            console.warn('Failed to fetch optimization result:', e);
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

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h1 className="text-xl font-bold text-content-primary">Optimization Lab</h1>
        <p className="text-xs text-content-secondary mt-0.5">
          Generate, evaluate, and compare execution variants on AMD ROCm. On non-AMD hosts, status reflects NOT_MEASURED.
        </p>
      </div>

      {/* Task Creation Form */}
      <div className="p-5 rounded-lg bg-surface border border-surface-border space-y-4">
        <h2 className="text-sm font-semibold text-content-primary flex items-center space-x-2">
          <Gauge className="w-4 h-4 text-accent-red" />
          <span>Optimization Run Configuration</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-1">
            <label className="text-xs font-mono text-content-secondary">Model ID</label>
            <input
              type="text"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              className="w-full bg-background border border-surface-border rounded px-3 py-2 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red"
            />
          </div>

          <div className="space-y-1">
            <label className="text-xs font-mono text-content-secondary">Objective</label>
            <select
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              className="w-full bg-background border border-surface-border rounded px-3 py-2 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red"
            >
              <option value="MAX_THROUGHPUT">MAX_THROUGHPUT (Tokens / sec)</option>
              <option value="MIN_LATENCY">MIN_LATENCY (Time to First Token)</option>
              <option value="BALANCED">BALANCED</option>
            </select>
          </div>
        </div>

        {/* Strategies multi-selection */}
        <div className="space-y-2 pt-2 border-t border-surface-border">
          <label className="text-xs font-mono text-content-secondary">Candidate Exploration Strategies</label>
          <div className="flex flex-wrap gap-2">
            {[
              { id: 'fp16', label: 'FP16 (Half Precision)' },
              { id: 'bf16', label: 'BF16 (Bfloat16)' },
              { id: 'fp32', label: 'FP32 (Baseline Single)' },
            ].map((st) => {
              const isSelected = strategies.includes(st.id);
              return (
                <button
                  key={st.id}
                  type="button"
                  onClick={() => toggleStrategy(st.id)}
                  className={`px-3 py-1.5 rounded text-xs font-mono transition-colors border ${
                    isSelected
                      ? 'bg-accent-red text-white border-accent-red'
                      : 'bg-surface-elevated text-zinc-400 border-surface-border hover:text-zinc-200'
                  }`}
                >
                  {st.label}
                </button>
              );
            })}
          </div>
        </div>

        <div className="flex items-center justify-between pt-2 border-t border-surface-border">
          <div className="text-[11px] font-mono text-zinc-500">
            Benchmarking Integrity: Real hardware measurements only. No synthetic numbers.
          </div>
          <button
            onClick={handleLaunch}
            disabled={isStarting || activeJob?.status === 'RUNNING'}
            className="flex items-center space-x-2 px-4 py-2 rounded bg-accent-red hover:bg-accent-red-hover text-white text-xs font-medium transition-colors disabled:opacity-50"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>{isStarting ? 'Queuing Lab Run...' : 'Run Optimization Job'}</span>
          </button>
        </div>
      </div>

      {/* Active Job & SSE Progress */}
      {activeJob && (
        <div className="p-5 rounded-lg bg-surface border border-surface-border space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <h3 className="text-sm font-semibold text-content-primary">Optimization Pipeline</h3>
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
                <span>Cancel</span>
              </button>
            )}
          </div>

          <LogViewer events={jobEvents} />

          {/* Results Comparison Table */}
          {jobResult && (
            <div className="space-y-3 pt-2">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-mono font-semibold text-content-primary uppercase tracking-wider">
                  Candidate Comparison Table
                </h4>
                <span className="text-xs font-mono text-zinc-500">
                  Status: {jobResult.domain_status || 'CONFIG_ONLY'}
                </span>
              </div>

              {/* Table */}
              <div className="overflow-x-auto rounded border border-surface-border bg-[#131317]">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-surface-elevated text-zinc-400 border-b border-surface-border">
                    <tr>
                      <th className="px-3 py-2">Role</th>
                      <th className="px-3 py-2">Strategy</th>
                      <th className="px-3 py-2">Status</th>
                      <th className="px-3 py-2">Throughput</th>
                      <th className="px-3 py-2">TTFT</th>
                      <th className="px-3 py-2">Quality Gate</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-surface-border">
                    {/* Baseline row */}
                    <tr className="hover:bg-surface-elevated/40">
                      <td className="px-3 py-2.5 font-semibold text-zinc-200">Baseline</td>
                      <td className="px-3 py-2.5 text-zinc-400">Default (FP32)</td>
                      <td className="px-3 py-2.5">
                        <DomainStatusTag status={jobResult.domain_status || 'CONFIG_ONLY'} />
                      </td>
                      <td className="px-3 py-2.5 text-zinc-500">
                        <DomainStatusTag status="NOT_MEASURED" />
                      </td>
                      <td className="px-3 py-2.5 text-zinc-500">
                        <DomainStatusTag status="NOT_MEASURED" />
                      </td>
                      <td className="px-3 py-2.5 text-zinc-400">PASS</td>
                    </tr>

                    {/* Explored Candidates */}
                    {strategies.map((strat, idx) => (
                      <tr key={strat} className="hover:bg-surface-elevated/40">
                        <td className="px-3 py-2.5 font-semibold text-zinc-300">Candidate #{idx + 1}</td>
                        <td className="px-3 py-2.5 uppercase text-zinc-200">{strat}</td>
                        <td className="px-3 py-2.5">
                          <DomainStatusTag status="CONFIG_ONLY" />
                        </td>
                        <td className="px-3 py-2.5 text-zinc-500">
                          <DomainStatusTag status="NOT_MEASURED" />
                        </td>
                        <td className="px-3 py-2.5 text-zinc-500">
                          <DomainStatusTag status="NOT_MEASURED" />
                        </td>
                        <td className="px-3 py-2.5 text-zinc-400">PENDING</td>
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
