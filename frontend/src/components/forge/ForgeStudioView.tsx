import React, { useState, useEffect } from 'react';
import type { ForgePlan, JobResponse, JobResultResponse, JobEvent } from '../../api/types';
import { createForgePlan, createJob, fetchJob, fetchJobResult, cancelJob } from '../../api/client';
import { subscribeToJobEvents } from '../../api/sse';
import { StatusBadge } from '../common/StatusBadge';
import { DomainStatusTag } from '../common/DomainStatusTag';
import { LogViewer } from '../common/LogViewer';
import { useToast } from '../common/Toast';
import {
  Play, XCircle, CheckCircle2, PackageCheck, FileCode, Check, AlertCircle,
  ChevronRight,
} from 'lucide-react';

import forgeStudioHeroImg from '../../assets/visuals/forge_studio_hero.jpg';

interface ForgeStudioViewProps {
  initialModelId?: string;
  initialRevision?: string;
  selectedJobId?: string | null;
  onJobCreated?: (jobId: string) => void;
}

const GPU_OPTIONS = [
  { value: 'gfx942', label: 'MI300X', sub: 'Instinct · 192 GB HBM3', tier: 'Enterprise' },
  { value: 'gfx90a', label: 'MI250X / MI210', sub: 'Instinct HPC · 64–128 GB', tier: 'HPC' },
  { value: 'gfx1100', label: 'RX 7900 XTX', sub: 'RDNA3 · 24 GB GDDR6', tier: 'Workstation' },
  { value: 'gfx1030', label: 'RX 6800 / 6900', sub: 'RDNA2 · 16 GB GDDR6', tier: 'Consumer' },
];

const PRECISION_OPTIONS = [
  { value: 'fp16', label: 'FP16', desc: 'Half precision · standard' },
  { value: 'bf16', label: 'BF16', desc: 'Brain float · CDNA native' },
  { value: 'fp32', label: 'FP32', desc: 'Single precision · reference' },
];

export const ForgeStudioView: React.FC<ForgeStudioViewProps> = ({
  initialModelId = 'Qwen/Qwen2.5-0.5B-Instruct',
  initialRevision = 'main',
  selectedJobId,
  onJobCreated,
}) => {
  const toast = useToast();

  const [modelId, setModelId] = useState(initialModelId);
  const [revision, setRevision] = useState(initialRevision);
  const [targetGpu, setTargetGpu] = useState('gfx90a');
  const [precision, setPrecision] = useState('fp16');
  const [plan, setPlan] = useState<ForgePlan | null>(null);
  const [isPlanning, setIsPlanning] = useState(false);
  const [planError, setPlanError] = useState<string | null>(null);

  const [activeJob, setActiveJob] = useState<JobResponse | null>(null);
  const [jobEvents, setJobEvents] = useState<JobEvent[]>([]);
  const [jobResult, setJobResult] = useState<JobResultResponse | null>(null);
  const [isStartingBuild, setIsStartingBuild] = useState(false);

  useEffect(() => {
    if (initialModelId) setModelId(initialModelId);
    if (initialRevision) setRevision(initialRevision);
  }, [initialModelId, initialRevision]);

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
      const msg = err.message || 'Failed to generate plan';
      setPlanError(msg);
      toast.error(msg);
      setPlan(null);
    } finally {
      setIsPlanning(false);
    }
  };

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
      toast.error(`Build failed: ${err.message}`);
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
      toast.error(`Cancel failed: ${e.message}`);
    }
  };

  const isRunning = activeJob?.status === 'RUNNING' || activeJob?.status === 'QUEUED';

  return (
    <div className="page-fade flex flex-col h-full">
      {/* ── PAGE HEADER ──────────────────────────────────────────────── */}
      <div className="relative overflow-hidden border-b border-surface-border shrink-0">
        <div
          className="absolute inset-0 bg-cover bg-center opacity-40"
          style={{ backgroundImage: `url(${forgeStudioHeroImg})` }}
        />
        <div className="absolute inset-0 hero-overlay" />
        <div className="relative z-10 px-8 sm:px-10 py-8">
          <p className="text-xs font-mono font-medium text-accent-red/90 uppercase tracking-wider mb-1.5">
            Deterministic Build Harness
          </p>
          <h1 className="text-2xl font-bold tracking-tight text-white mb-1">Forge Studio</h1>
          <p className="text-sm text-content-secondary">
            Package models for AMD GPUs with reproducible builds and verified artifact digests.
          </p>
        </div>
      </div>

      {/* ── SPLIT LAYOUT ─────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">
        {/* LEFT: Config panel */}
        <div className="w-80 shrink-0 border-r border-surface-border bg-surface-deep overflow-y-auto">
          <div className="p-5 space-y-5">
            {/* Model */}
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

            {/* Revision */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-content-muted">Revision / SHA</label>
              <input
                type="text"
                value={revision}
                onChange={(e) => setRevision(e.target.value)}
                className="input-field font-mono text-xs"
                placeholder="main"
              />
            </div>

            {/* Target GPU — visual option cards */}
            <div className="space-y-2">
              <label className="text-xs font-medium text-content-muted">Target GPU</label>
              <div className="space-y-1.5">
                {GPU_OPTIONS.map((gpu) => {
                  const isSelected = targetGpu === gpu.value;
                  return (
                    <button
                      key={gpu.value}
                      type="button"
                      onClick={() => setTargetGpu(gpu.value)}
                      className={`w-full flex items-center justify-between p-2.5 rounded-lg border text-left transition-all ${
                        isSelected
                          ? 'bg-accent-red/10 border-accent-red/50 text-content-primary'
                          : 'bg-surface-elevated border-surface-border text-content-secondary hover:border-zinc-600 hover:text-content-primary'
                      }`}
                    >
                      <div>
                        <div className="text-xs font-semibold font-mono">{gpu.label}</div>
                        <div className="text-[11px] text-content-muted mt-0.5">{gpu.sub}</div>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded font-mono ${
                          isSelected ? 'bg-accent-red/20 text-red-400' : 'bg-surface-border text-content-muted'
                        }`}>
                          {gpu.tier}
                        </span>
                        {isSelected && <Check className="w-3.5 h-3.5 text-accent-red" />}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Precision — toggle chips */}
            <div className="space-y-2">
              <label className="text-xs font-medium text-content-muted">Precision</label>
              <div className="grid grid-cols-3 gap-1.5">
                {PRECISION_OPTIONS.map((p) => {
                  const isSelected = precision === p.value;
                  return (
                    <button
                      key={p.value}
                      type="button"
                      onClick={() => setPrecision(p.value)}
                      title={p.desc}
                      className={`py-2 rounded-lg border text-xs font-mono font-semibold transition-all ${
                        isSelected
                          ? 'bg-accent-red/10 border-accent-red/50 text-content-primary'
                          : 'bg-surface-elevated border-surface-border text-content-secondary hover:border-zinc-600'
                      }`}
                    >
                      {p.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Recipe info */}
            <div className="pt-1 text-[11px] font-mono text-content-muted space-y-0.5">
              <div>Recipe: <span className="text-content-secondary">pytorch_transformers_hip</span></div>
              <div>Version: <span className="text-content-secondary">v1.0.0</span></div>
            </div>

            {/* Generate plan button */}
            <button
              onClick={handleGeneratePlan}
              disabled={isPlanning || isRunning}
              className="w-full btn-secondary justify-center"
            >
              {isPlanning ? (
                <><span className="w-3.5 h-3.5 border-2 border-content-secondary/30 border-t-content-secondary rounded-full animate-spin" /> Generating Plan...</>
              ) : (
                <>Generate Forge Plan<ChevronRight className="w-4 h-4" /></>
              )}
            </button>
          </div>
        </div>

        {/* RIGHT: Plan + Progress + Results */}
        <div className="flex-1 overflow-y-auto">
          {/* Plan error */}
          {planError && (
            <div className="m-6 p-4 rounded-xl bg-red-950/50 border border-red-500/25 flex items-start gap-3">
              <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
              <div>
                <div className="text-sm font-semibold text-red-300 mb-1">Plan Generation Failed</div>
                <p className="text-xs text-red-400/80 font-mono">{planError}</p>
              </div>
            </div>
          )}

          {/* No plan yet */}
          {!plan && !planError && (
            <div className="h-full flex flex-col items-center justify-center p-10 text-center">
              <div className="w-14 h-14 rounded-2xl bg-surface-elevated border border-surface-border flex items-center justify-center mb-4">
                <PackageCheck className="w-6 h-6 text-content-muted" />
              </div>
              <div className="text-sm font-semibold text-content-secondary mb-1">Configure & Generate Plan</div>
              <p className="text-xs text-content-muted max-w-xs leading-relaxed">
                Select a target GPU and precision on the left, then click <strong className="text-content-secondary">Generate Forge Plan</strong> to create a deterministic build plan.
              </p>
            </div>
          )}

          {/* Plan display */}
          {plan && (
            <div className="p-6 space-y-5">
              {/* Plan header */}
              <div className="card p-5 space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  <div className="space-y-1.5">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-medium text-content-muted">Build Plan</span>
                      <span className="text-xs font-mono text-content-primary bg-surface-elevated px-2 py-0.5 rounded border border-surface-border">
                        {plan.plan_id}
                      </span>
                    </div>
                    <div className="text-[11px] font-mono text-content-muted">
                      Commit SHA: <span className="text-content-secondary">{plan.revision}</span>
                    </div>
                  </div>
                  <button
                    onClick={handleStartBuild}
                    disabled={isStartingBuild || isRunning}
                    className="btn-primary shrink-0"
                  >
                    {isStartingBuild ? (
                      <><span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />Submitting...</>
                    ) : (
                      <><Play className="w-4 h-4 fill-current" />Run Forge Build (CONFIG_ONLY)</>
                    )}
                  </button>
                </div>

                {/* Plan specs */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-4 border-t border-surface-border">
                  <div>
                    <div className="text-[11px] text-content-muted mb-0.5">Recipe</div>
                    <div className="text-xs font-semibold font-mono text-content-primary">{plan.recipe_id}</div>
                  </div>
                  <div>
                    <div className="text-[11px] text-content-muted mb-0.5">Target GPU</div>
                    <div className="text-xs font-semibold font-mono text-content-primary">{plan.target_gpu || 'Host Default'}</div>
                  </div>
                  <div>
                    <div className="text-[11px] text-content-muted mb-0.5">Precision</div>
                    <div className="text-xs font-semibold font-mono text-content-primary uppercase">{plan.precision}</div>
                  </div>
                  <div>
                    <div className="text-[11px] text-content-muted mb-0.5">Estimated Size</div>
                    <div className="text-xs font-semibold font-mono text-content-primary">{(plan.estimated_disk_space_bytes / 1e9).toFixed(2)} GB</div>
                  </div>
                </div>

                {/* Steps */}
                <div className="pt-3 border-t border-surface-border space-y-1.5">
                  <div className="text-[11px] text-content-muted font-medium uppercase tracking-wide mb-2">Deterministic Plan — {plan.steps.length} steps</div>
                  {plan.steps.map((step, idx) => (
                    <div key={idx} className="flex items-center gap-3 p-2.5 rounded-lg bg-surface-elevated border border-surface-border text-xs">
                      <span className="w-5 h-5 rounded-full bg-accent-red/15 border border-accent-red/25 flex items-center justify-center text-[10px] font-mono font-bold text-accent-red shrink-0">
                        {idx + 1}
                      </span>
                      <span className="font-medium text-content-primary">{step.name}</span>
                      <span className="text-content-muted ml-auto text-[11px]">{step.description}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Active job progress */}
              {activeJob && (
                <div className="card p-5 space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <h3 className="text-sm font-semibold text-content-primary">Build Progress</h3>
                      <span className="text-[11px] font-mono text-content-muted bg-surface-elevated px-2 py-0.5 rounded border border-surface-border">
                        {activeJob.job_id}
                      </span>
                      <StatusBadge status={activeJob.status} />
                      <DomainStatusTag status={activeJob.domain_status} />
                    </div>

                    {isRunning && (
                      <button onClick={handleCancel} className="btn-danger">
                        <XCircle className="w-3.5 h-3.5" />
                        Cancel
                      </button>
                    )}
                  </div>

                  <LogViewer events={jobEvents} />

                  {/* Result artifacts */}
                  {jobResult && (
                    <div className="p-4 rounded-xl bg-surface-deep border border-surface-border space-y-3">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <span className="flex items-center gap-2 text-sm font-semibold text-emerald-400">
                          <CheckCircle2 className="w-4 h-4" />
                          Build Finished — Domain Status: {jobResult.domain_status || 'CONFIG_ONLY'}
                        </span>
                        <span className="text-[11px] text-content-muted font-mono">
                          {jobResult.completed_at ? new Date(jobResult.completed_at).toLocaleTimeString() : ''}
                        </span>
                      </div>

                      {jobResult.result?.artifacts && (
                        <div className="space-y-2 pt-2 border-t border-surface-border">
                          <div className="flex items-center gap-1.5 text-xs text-content-muted font-medium uppercase tracking-wide">
                            <FileCode className="w-3.5 h-3.5" />
                            Verified Artifact Digests:
                          </div>
                          <div className="space-y-1 bg-surface rounded-lg border border-surface-border p-3">
                            {Object.entries(jobResult.result.artifacts).map(([file, sha]) => (
                              <div key={file} className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 py-1 border-b border-surface-border/40 last:border-0">
                                <span className="flex items-center gap-1.5 text-xs text-content-primary font-semibold font-mono">
                                  <Check className="w-3 h-3 text-emerald-400 shrink-0" />
                                  {file}
                                </span>
                                <span className="text-[10px] font-mono text-content-muted truncate max-w-sm">
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
          )}
        </div>
      </div>
    </div>
  );
};
