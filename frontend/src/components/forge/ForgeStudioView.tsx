import React, { useState, useEffect } from 'react';
import type { ForgePlan, JobResponse, JobResultResponse, JobEvent } from '../../api/types';
import { createForgePlan, createJob, fetchJob, fetchJobResult, cancelJob } from '../../api/client';
import { subscribeToJobEvents } from '../../api/sse';
import { StatusBadge } from '../common/StatusBadge';
import { DomainStatusTag } from '../common/DomainStatusTag';
import { LogViewer } from '../common/LogViewer';
import { Play, XCircle, CheckCircle2, Hammer, Cpu, Terminal, FileCode, Check, ShieldAlert } from 'lucide-react';

import forgeStudioHeroImg from '../../assets/visuals/forge_studio_hero.jpg';

interface ForgeStudioViewProps {
  initialModelId?: string;
  initialRevision?: string;
  selectedJobId?: string | null;
  onJobCreated?: (jobId: string) => void;
}

export const ForgeStudioView: React.FC<ForgeStudioViewProps> = ({
  initialModelId = 'Qwen/Qwen2.5-0.5B-Instruct',
  initialRevision = 'main',
  selectedJobId,
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
        if (job.revision) setRevision(job.revision);

        if (job.status === 'SUCCEEDED' || job.status === 'FAILED' || job.status === 'CANCELLED') {
          const res = await fetchJobResult(selectedJobId!);
          if (!isCancelled) setJobResult(res);
        } else {
          unsubscribe = subscribeToJobEvents(selectedJobId!, {
            onEvent: (evt) => {
              if (!isCancelled) {
                setJobEvents((prev) => [...prev, evt]);
                if (['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(evt.status)) {
                  setActiveJob((prev) => (prev ? { ...prev, status: evt.status as any } : prev));
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

  const PIPELINE_STEPS = [
    { id: 'check_prerequisites', label: 'Prerequisites', desc: 'Disk & environment validation' },
    { id: 'materialize_model', label: 'Materialization', desc: 'Immutable SHA acquisition' },
    { id: 'configure_runtime', label: 'Configuration', desc: 'ROCm recipe synthesis' },
    { id: 'generate_launch_scripts', label: 'Launch Scripts', desc: 'run_inference.py generation' },
    { id: 'verify_build', label: 'Verification', desc: 'SHA256 checksum validation' },
  ];

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Visual Header Banner */}
      <div className="relative rounded-xl overflow-hidden border border-surface-border bg-surface shadow-xl">
        <div
          className="absolute inset-0 bg-cover bg-center opacity-35 mix-blend-luminosity"
          style={{ backgroundImage: `url(${forgeStudioHeroImg})` }}
        />
        <div className="absolute inset-0 hero-overlay" />

        <div className="relative p-6 space-y-2">
          <div className="inline-flex items-center space-x-2 px-2.5 py-1 rounded-full bg-accent-red/10 border border-accent-red/30 text-[11px] font-mono text-red-400">
            <Hammer className="w-3.5 h-3.5 text-accent-red" />
            <span>Deterministic Model Packaging Harness</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Forge Studio</h1>
          <p className="text-xs text-zinc-300 max-w-2xl font-sans leading-relaxed">
            Generate immutable build plans, materialize config manifests, and generate standalone launchers for AMD ROCm GPUs.
          </p>
        </div>
      </div>

      {/* 5-Stage Stepper Overview */}
      <div className="p-4 rounded-xl bg-surface border border-surface-border shadow-sm">
        <div className="text-[11px] font-mono text-zinc-400 font-semibold uppercase mb-3 flex items-center space-x-2">
          <Terminal className="w-3.5 h-3.5 text-accent-red" />
          <span>Deterministic Build Lifecycle</span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
          {PIPELINE_STEPS.map((step, idx) => (
            <div
              key={step.id}
              className="p-2.5 rounded-lg bg-surface-elevated/70 border border-surface-border flex flex-col justify-between space-y-1 text-xs font-mono"
            >
              <div className="flex items-center justify-between text-zinc-500 text-[10px]">
                <span>0{idx + 1}</span>
                <span className="w-1.5 h-1.5 rounded-full bg-zinc-600" />
              </div>
              <div className="font-semibold text-content-primary text-[11px]">{step.label}</div>
              <div className="text-[10px] text-zinc-500 truncate">{step.desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Configuration & Plan Generator */}
      <div className="p-6 rounded-xl bg-surface border border-surface-border shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-content-primary flex items-center space-x-2">
            <Cpu className="w-4 h-4 text-zinc-400" />
            <span>1. Target Architecture &amp; Precision</span>
          </h2>
          <span className="text-[11px] font-mono text-zinc-500">pytorch_transformers_hip</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-3.5">
          <div className="md:col-span-2 space-y-1.5">
            <label className="text-xs font-mono text-content-secondary">Model ID</label>
            <input
              type="text"
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              className="w-full bg-background border border-surface-border rounded-lg px-3.5 py-2.5 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red focus:ring-1 focus:ring-accent-red transition-all"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-mono text-content-secondary">Target AMD GPU</label>
            <select
              value={targetGpu}
              onChange={(e) => setTargetGpu(e.target.value)}
              className="w-full bg-background border border-surface-border rounded-lg px-3 py-2.5 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red focus:ring-1 focus:ring-accent-red transition-all"
            >
              <option value="gfx1100">gfx1100 (Radeon RX 7900 XTX)</option>
              <option value="gfx942">gfx942 (Instinct MI300X)</option>
              <option value="gfx90a">gfx90a (Instinct MI210 / MI250X)</option>
              <option value="gfx1030">gfx1030 (Radeon RX 6800 / 6900)</option>
            </select>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-mono text-content-secondary">Precision</label>
            <select
              value={precision}
              onChange={(e) => setPrecision(e.target.value)}
              className="w-full bg-background border border-surface-border rounded-lg px-3 py-2.5 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red focus:ring-1 focus:ring-accent-red transition-all"
            >
              <option value="fp16">FP16 (Half Precision)</option>
              <option value="bf16">BF16 (Bfloat16)</option>
              <option value="fp32">FP32 (Single Precision)</option>
            </select>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-surface-border">
          <div className="text-[11px] font-mono text-zinc-500">
            Runtime Recipe: <span className="text-zinc-300 font-semibold">pytorch_transformers_hip (v1.0.0)</span>
          </div>
          <button
            onClick={handleGeneratePlan}
            disabled={isPlanning}
            className="px-5 py-2.5 rounded-lg bg-surface-elevated hover:bg-zinc-700 text-content-primary text-xs font-semibold transition-all border border-surface-border disabled:opacity-50 shadow-sm"
          >
            {isPlanning ? 'Generating Forge Plan...' : 'Generate Forge Plan'}
          </button>
        </div>
      </div>

      {/* Plan Error */}
      {planError && (
        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/25 text-xs text-red-400 flex items-start space-x-2">
          <ShieldAlert className="w-4 h-4 shrink-0 mt-0.5 text-red-400" />
          <span>{planError}</span>
        </div>
      )}

      {/* Deterministic Plan Display */}
      {plan && (
        <div className="p-6 rounded-xl bg-surface border border-surface-border shadow-lg space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="space-y-1">
              <div className="flex items-center space-x-2">
                <span className="text-xs font-mono text-zinc-500 uppercase font-semibold">Deterministic Plan ID</span>
                <span className="font-mono text-xs text-content-primary bg-black/60 px-2 py-0.5 rounded border border-zinc-800">
                  {plan.plan_id}
                </span>
              </div>
              <div className="text-[11px] font-mono text-zinc-400">
                Resolved Commit SHA: <span className="text-zinc-200 font-semibold">{plan.revision}</span>
              </div>
            </div>

            <button
              onClick={handleStartBuild}
              disabled={isStartingBuild || activeJob?.status === 'RUNNING'}
              className="flex items-center space-x-2 px-5 py-2.5 rounded-lg bg-accent-red hover:bg-accent-red-hover text-white text-xs font-semibold transition-all disabled:opacity-50 shadow-md shadow-red-950/40 hover:scale-[1.02] active:scale-[0.98]"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>{isStartingBuild ? 'Submitting...' : 'Run Forge Build (CONFIG_ONLY)'}</span>
            </button>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-4 rounded-xl bg-[#121216] border border-surface-border text-xs font-mono">
            <div className="space-y-0.5">
              <span className="text-zinc-500 text-[11px]">Recipe:</span>
              <div className="text-zinc-200 font-semibold">{plan.recipe_id} v{plan.recipe_version}</div>
            </div>
            <div className="space-y-0.5">
              <span className="text-zinc-500 text-[11px]">Target GPU:</span>
              <div className="text-zinc-200 font-semibold">{plan.target_gpu || 'Host Default'}</div>
            </div>
            <div className="space-y-0.5">
              <span className="text-zinc-500 text-[11px]">Precision:</span>
              <div className="text-zinc-200 font-semibold uppercase">{plan.precision}</div>
            </div>
            <div className="space-y-0.5">
              <span className="text-zinc-500 text-[11px]">Estimated Disk:</span>
              <div className="text-zinc-200 font-semibold">
                {(plan.estimated_disk_space_bytes / 1e9).toFixed(2)} GB
              </div>
            </div>
          </div>

          {/* Ordered Build Steps */}
          <div className="space-y-2 pt-2">
            <span className="text-xs font-mono text-zinc-400 font-semibold uppercase">Step Execution Order</span>
            <div className="space-y-1.5">
              {plan.steps.map((step, idx) => (
                <div
                  key={idx}
                  className="px-3.5 py-2.5 rounded-lg bg-surface-elevated/80 border border-surface-border flex items-center justify-between text-xs font-mono"
                >
                  <div className="flex items-center space-x-2.5">
                    <span className="text-accent-red font-bold text-[11px]">{idx + 1}.</span>
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
        <div className="p-6 rounded-xl bg-surface border border-surface-border shadow-lg space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-2.5">
              <h3 className="text-sm font-semibold text-content-primary">Execution Progress</h3>
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
                <span>Cancel Job</span>
              </button>
            )}
          </div>

          {/* Monospace Event Log Stream */}
          <LogViewer events={jobEvents} />

          {/* Completed Job Result Artifacts */}
          {jobResult && (
            <div className="p-5 rounded-xl bg-[#121216] border border-surface-border space-y-3.5 text-xs font-mono">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="font-semibold text-emerald-400 flex items-center space-x-1.5 text-sm">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  <span>Build Finished — Domain Status: {jobResult.domain_status || 'CONFIG_ONLY'}</span>
                </span>
                <span className="text-zinc-500 text-[11px]">
                  {jobResult.completed_at ? new Date(jobResult.completed_at).toLocaleTimeString() : ''}
                </span>
              </div>

              {jobResult.result?.artifacts && (
                <div className="space-y-2 pt-2 border-t border-surface-border">
                  <span className="text-zinc-400 uppercase text-[11px] font-semibold flex items-center space-x-1">
                    <FileCode className="w-3.5 h-3.5 text-zinc-400" />
                    <span>Verified Artifact Digests:</span>
                  </span>
                  <div className="space-y-1 bg-surface p-3 rounded-lg border border-surface-border">
                    {Object.entries(jobResult.result.artifacts).map(([file, sha]) => (
                      <div key={file} className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 text-zinc-300 py-1 border-b border-surface-border/40 last:border-0">
                        <span className="text-zinc-200 font-semibold flex items-center space-x-1.5">
                          <Check className="w-3 h-3 text-emerald-400" />
                          <span>{file}</span>
                        </span>
                        <span className="text-zinc-400 font-mono text-[10px] truncate max-w-sm">
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
