import React, { useState, useEffect } from 'react';
import type { ForgePlan, JobResponse, JobResultResponse, JobEvent } from '../../api/types';
import { createForgePlan, createJob, fetchJobResult, cancelJob } from '../../api/client';
import { subscribeToJobEvents } from '../../api/sse';
import { StatusBadge } from '../common/StatusBadge';
import { DomainStatusTag } from '../common/DomainStatusTag';
import { LogViewer } from '../common/LogViewer';
import { Play, XCircle, CheckCircle2 } from 'lucide-react';

interface ForgeStudioViewProps {
  initialModelId?: string;
  initialRevision?: string;
  selectedJobId?: string | null;
  onJobCreated?: (jobId: string) => void;
}

export const ForgeStudioView: React.FC<ForgeStudioViewProps> = ({
  initialModelId = 'Qwen/Qwen2.5-0.5B-Instruct',
  initialRevision = 'main',
  onJobCreated,
}) => {
  const [modelId, setModelId] = useState(initialModelId);
  const [revision, setRevision] = useState(initialRevision);
  const [targetGpu, setTargetGpu] = useState('gfx90a');
  const [precision, setPrecision] = useState('fp16');
  const [plan, setPlan] = useState<ForgePlan | null>(null);
  const [isPlanning, setIsPlanning] = useState(false);
  const [planError, setPlanError] = useState<string | null>(null);

  // Active Job State
  const [activeJob, setActiveJob] = useState<JobResponse | null>(null);
  const [jobEvents, setJobEvents] = useState<JobEvent[]>([]);
  const [jobResult, setJobResult] = useState<JobResultResponse | null>(null);
  const [isStartingBuild, setIsStartingBuild] = useState(false);

  useEffect(() => {
    if (initialModelId) setModelId(initialModelId);
    if (initialRevision) setRevision(initialRevision);
  }, [initialModelId, initialRevision]);

  // Handle plan generation
  const handleGeneratePlan = async () => {
    setIsPlanning(true);
    setPlanError(null);
    try {
      const generatedPlan = await createForgePlan({
        model_id: modelId,
        revision: revision || 'main',
        precision,
        target_gpu: targetGpu,
      });
      setPlan(generatedPlan);
    } catch (err: any) {
      setPlanError(err.message || 'Failed to generate Forge plan');
      setPlan(null);
    } finally {
      setIsPlanning(false);
    }
  };

  // Launch Forge Build Job
  const handleStartBuild = async () => {
    setIsStartingBuild(true);
    setJobEvents([]);
    setJobResult(null);

    try {
      const job = await createJob({
        job_type: 'FORGE_BUILD',
        model_id: modelId,
        revision: plan?.revision || revision,
        precision,
        target_gpu: targetGpu,
        allow_full_weights: false, // CONFIG_ONLY default
      });
      setActiveJob(job);
      onJobCreated?.(job.job_id);

      // Subscribe to SSE
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
            console.warn('Failed to fetch job result:', e);
          }
        },
      });
    } catch (err: any) {
      alert(`Build launch failed: ${err.message}`);
    } finally {
      setIsStartingBuild(false);
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
        <h1 className="text-xl font-bold text-content-primary">Forge Studio</h1>
        <p className="text-xs text-content-secondary mt-0.5">
          Deterministic build planning and reproducible model preparation for AMD ROCm GPUs.
        </p>
      </div>

      {/* Configuration & Plan Generator */}
      <div className="p-5 rounded-lg bg-surface border border-surface-border space-y-4">
        <h2 className="text-sm font-semibold text-content-primary">1. Model & Target Configuration</h2>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <div className="md:col-span-2 space-y-1">
            <label className="text-xs font-mono text-content-secondary">Model ID</label>
            <input
              type="text"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              className="w-full bg-background border border-surface-border rounded px-3 py-2 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red"
            />
          </div>

          <div className="space-y-1">
            <label className="text-xs font-mono text-content-secondary">Target AMD GPU</label>
            <select
              value={targetGpu}
              onChange={(e) => setTargetGpu(e.target.value)}
              className="w-full bg-background border border-surface-border rounded px-3 py-2 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red"
            >
              <option value="gfx90a">gfx90a (MI210 / MI250X)</option>
              <option value="gfx942">gfx942 (MI300X)</option>
              <option value="gfx1100">gfx1100 (Radeon RX 7900 XTX)</option>
              <option value="gfx1030">gfx1030 (Radeon RX 6800 / 6900)</option>
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-xs font-mono text-content-secondary">Precision</label>
            <select
              value={precision}
              onChange={(e) => setPrecision(e.target.value)}
              className="w-full bg-background border border-surface-border rounded px-3 py-2 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red"
            >
              <option value="fp16">FP16 (Half Precision)</option>
              <option value="bf16">BF16 (Bfloat16)</option>
              <option value="fp32">FP32 (Single Precision)</option>
            </select>
          </div>
        </div>

        <div className="flex items-center justify-between pt-2 border-t border-surface-border">
          <div className="text-[11px] font-mono text-zinc-500">
            Selected Runtime: <span className="text-zinc-300">pytorch_transformers_hip</span> (PyTorch + ROCm)
          </div>
          <button
            onClick={handleGeneratePlan}
            disabled={isPlanning}
            className="px-4 py-2 rounded bg-surface-elevated hover:bg-zinc-700 text-content-primary text-xs font-medium transition-colors border border-surface-border disabled:opacity-50"
          >
            {isPlanning ? 'Planning...' : 'Generate Forge Plan'}
          </button>
        </div>
      </div>

      {/* Plan Error */}
      {planError && (
        <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-xs text-red-400">
          {planError}
        </div>
      )}

      {/* Deterministic Plan Display */}
      {plan && (
        <div className="p-5 rounded-lg bg-surface border border-surface-border space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xs font-mono text-zinc-500 uppercase">Deterministic Plan</span>
                <span className="font-mono text-xs text-content-primary bg-black/40 px-2 py-0.5 rounded border border-zinc-800">
                  {plan.plan_id}
                </span>
              </div>
              <div className="text-[11px] font-mono text-zinc-400 mt-1">
                Resolved Commit SHA: <span className="text-zinc-200">{plan.revision}</span>
              </div>
            </div>

            <button
              onClick={handleStartBuild}
              disabled={isStartingBuild || activeJob?.status === 'RUNNING'}
              className="flex items-center space-x-2 px-4 py-2 rounded bg-accent-red hover:bg-accent-red-hover text-white text-xs font-medium transition-colors disabled:opacity-50 shadow-sm"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>{isStartingBuild ? 'Submitting...' : 'Run Forge Build (CONFIG_ONLY)'}</span>
            </button>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-3 rounded bg-[#131317] border border-surface-border text-xs font-mono">
            <div>
              <span className="text-zinc-500">Recipe:</span>
              <div className="text-zinc-200 font-semibold">{plan.recipe_id} v{plan.recipe_version}</div>
            </div>
            <div>
              <span className="text-zinc-500">Target GPU:</span>
              <div className="text-zinc-200 font-semibold">{plan.target_gpu || 'Host Default'}</div>
            </div>
            <div>
              <span className="text-zinc-500">Precision:</span>
              <div className="text-zinc-200 font-semibold uppercase">{plan.precision}</div>
            </div>
            <div>
              <span className="text-zinc-500">Est. Space:</span>
              <div className="text-zinc-200 font-semibold">
                {(plan.estimated_disk_space_bytes / 1e9).toFixed(2)} GB
              </div>
            </div>
          </div>

          {/* Ordered Build Steps */}
          <div className="space-y-2">
            <span className="text-xs font-mono text-zinc-400 font-semibold uppercase">Ordered Build Steps</span>
            <div className="space-y-1.5">
              {plan.steps.map((step, idx) => (
                <div
                  key={idx}
                  className="px-3 py-2 rounded bg-surface-elevated border border-surface-border flex items-center justify-between text-xs font-mono"
                >
                  <div className="flex items-center space-x-2">
                    <span className="text-zinc-500">{idx + 1}.</span>
                    <span className="text-zinc-200 font-medium">{step.name}</span>
                  </div>
                  <span className="text-zinc-400 text-[11px]">{step.description}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Active Job Execution Progress */}
      {activeJob && (
        <div className="p-5 rounded-lg bg-surface border border-surface-border space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <h3 className="text-sm font-semibold text-content-primary">Execution Progress</h3>
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
                <span>Cancel Job</span>
              </button>
            )}
          </div>

          {/* Monospace Event Log Stream */}
          <LogViewer events={jobEvents} />

          {/* Completed Job Result Artifacts */}
          {jobResult && (
            <div className="p-4 rounded bg-[#131317] border border-surface-border space-y-3 text-xs font-mono">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-emerald-400 flex items-center space-x-1.5">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>Build Finished — Domain Status: {jobResult.domain_status || 'CONFIG_ONLY'}</span>
                </span>
                <span className="text-zinc-500">
                  {jobResult.completed_at ? new Date(jobResult.completed_at).toLocaleTimeString() : ''}
                </span>
              </div>

              {jobResult.result?.artifacts && (
                <div className="space-y-1 pt-2 border-t border-surface-border">
                  <span className="text-zinc-400 uppercase text-[11px]">Verified Artifact Digests:</span>
                  <div className="space-y-1">
                    {Object.entries(jobResult.result.artifacts).map(([file, sha]) => (
                      <div key={file} className="flex items-center justify-between text-zinc-300 py-0.5">
                        <span className="text-zinc-300">{file}</span>
                        <span className="text-zinc-500 font-mono text-[11px] truncate max-w-[280px]">
                          SHA256: {String(sha)}
                        </span>
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
