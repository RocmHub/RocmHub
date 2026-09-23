import React from 'react';
import type { HealthResponse, JobListResponse } from '../../api/types';
import { StatusBadge } from '../common/StatusBadge';
import { DomainStatusTag } from '../common/DomainStatusTag';
import { Cpu, Server, Activity, ArrowRight, Clock, Sparkles, Terminal, ShieldCheck } from 'lucide-react';
import type { NavTab } from '../layout/Sidebar';

import dashboardHeroImg from '../../assets/visuals/dashboard_hero.jpg';
import noAcceleratorImg from '../../assets/visuals/no_accelerator.svg';
import emptyJobsImg from '../../assets/visuals/empty_jobs.svg';

interface DashboardViewProps {
  health: HealthResponse | null;
  jobsList: JobListResponse | null;
  isLoadingJobs: boolean;
  onNavigate: (tab: NavTab) => void;
  onSelectJob: (jobId: string) => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  health,
  jobsList,
  isLoadingJobs,
  onNavigate,
  onSelectJob,
}) => {
  const jobs = jobsList?.items || [];
  const activeJobs = jobs.filter((j) => j.status === 'RUNNING' || j.status === 'QUEUED');

  return (
    <div className="space-y-6">
      {/* Cinematic Hero Banner */}
      <div className="relative rounded-xl overflow-hidden border border-surface-border shadow-2xl bg-surface">
        {/* Background Hero Image with Dark Gradient Mask */}
        <div
          className="absolute inset-0 bg-cover bg-center opacity-40 mix-blend-luminosity"
          style={{ backgroundImage: `url(${dashboardHeroImg})` }}
        />
        <div className="absolute inset-0 hero-overlay" />

        {/* Hero Content */}
        <div className="relative p-6 sm:p-8 space-y-4 max-w-3xl">
          <div className="inline-flex items-center space-x-2 px-2.5 py-1 rounded-full bg-accent-red/15 border border-accent-red/30 text-[11px] font-mono font-medium text-red-400">
            <Sparkles className="w-3.5 h-3.5 text-accent-red" />
            <span>System Dashboard</span>
            <span className="text-zinc-500">•</span>
            <span className="text-zinc-400">ROCm Engineering Harness</span>
          </div>

          <div className="space-y-1.5">
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
              Autonomous AI Acceleration on AMD Hardware
            </h1>
            <p className="text-xs sm:text-sm text-zinc-300 leading-relaxed max-w-2xl font-sans">
              Deterministic preparation, hardware-truthful diagnostics, and autonomous model tuning for AMD ROCm &amp; Radeon / Instinct compute architectures.
            </p>
          </div>

          {/* Quick Action Pills */}
          <div className="flex flex-wrap items-center gap-2.5 pt-2">
            <button
              onClick={() => onNavigate('forge')}
              className="flex items-center space-x-2 px-3.5 py-2 rounded-lg bg-accent-red hover:bg-accent-red-hover text-white text-xs font-semibold shadow-lg shadow-red-950/40 transition-all hover:scale-[1.02] active:scale-[0.98]"
            >
              <span>Launch Forge Build</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>

            <button
              onClick={() => onNavigate('explorer')}
              className="flex items-center space-x-2 px-3.5 py-2 rounded-lg bg-surface-elevated/90 hover:bg-zinc-700 text-zinc-200 hover:text-white border border-surface-border text-xs font-medium transition-all"
            >
              <Terminal className="w-3.5 h-3.5 text-zinc-400" />
              <span>Inspect Model</span>
            </button>

            <button
              onClick={() => onNavigate('engineer')}
              className="flex items-center space-x-2 px-3.5 py-2 rounded-lg bg-surface-elevated/90 hover:bg-zinc-700 text-zinc-200 hover:text-white border border-surface-border text-xs font-medium transition-all"
            >
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
              <span>Autonomous Loop</span>
            </button>
          </div>
        </div>
      </div>

      {/* Host Hardware Diagnostic Notice */}
      {health && !health.rocm_available && (
        <div className="p-4 rounded-xl bg-gradient-to-r from-amber-500/10 via-surface to-surface border border-amber-500/25 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 text-xs">
          <div className="flex items-start space-x-3.5">
            <div className="w-10 h-10 rounded-lg bg-amber-500/15 border border-amber-500/30 flex items-center justify-center shrink-0 p-1">
              <img src={noAcceleratorImg} alt="Host Diagnostic" className="w-7 h-7" />
            </div>
            <div className="space-y-1">
              <div className="flex items-center space-x-2">
                <span className="font-semibold text-amber-300">Host Hardware Notice</span>
                <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-amber-500/20 text-amber-300 border border-amber-500/30">
                  NO_ACCELERATOR
                </span>
              </div>
              <p className="text-zinc-400 leading-relaxed max-w-3xl text-[11px]">
                {health.warnings[0] ||
                  'Running on host without native AMD ROCm GPU accelerator. Forge builds operate strictly in CONFIG_ONLY mode with zero weight downloads and zero synthetic GPU performance metrics.'}
              </p>
            </div>
          </div>
          <div className="shrink-0 font-mono text-[10px] text-zinc-500 bg-surface-elevated px-2.5 py-1 rounded border border-surface-border">
            Fail-Closed Enforced
          </div>
        </div>
      )}

      {/* High-Tech Metric Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1: Backend */}
        <div className="relative p-4 rounded-xl bg-surface hover:bg-surface-elevated/70 border border-surface-border hover:border-zinc-700 transition-all space-y-2 group overflow-hidden">
          <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 to-indigo-600 opacity-60" />
          <div className="flex items-center justify-between text-xs text-content-secondary">
            <span className="font-medium">FastAPI Backend</span>
            <Server className="w-4 h-4 text-zinc-400 group-hover:text-blue-400 transition-colors" />
          </div>
          <div className="text-xl font-mono font-bold text-content-primary tracking-tight">
            {health ? `v${health.version}` : 'Disconnected'}
          </div>
          <div className="text-[11px] font-mono text-zinc-500 truncate">
            {health ? `Python ${health.host_platform.python_version} (${health.host_platform.os})` : 'Offline'}
          </div>
        </div>

        {/* Card 2: ROCm Hardware */}
        <div className="relative p-4 rounded-xl bg-surface hover:bg-surface-elevated/70 border border-surface-border hover:border-zinc-700 transition-all space-y-2 group overflow-hidden">
          <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-accent-red to-amber-500 opacity-60" />
          <div className="flex items-center justify-between text-xs text-content-secondary">
            <span className="font-medium">AMD ROCm Platform</span>
            <Cpu className="w-4 h-4 text-zinc-400 group-hover:text-accent-red transition-colors" />
          </div>
          <div className="text-xl font-mono font-bold">
            {health?.rocm_available ? (
              <span className="text-emerald-400">{health.system.gpus[0]?.gfx_target || 'AMD GPU'}</span>
            ) : (
              <span className="text-amber-400">CONFIG_ONLY</span>
            )}
          </div>
          <div className="text-[11px] font-mono text-zinc-500">
            {health?.rocm_available
              ? `${health.system.gpus_detected} device(s) discovered`
              : 'Non-AMD Diagnostic Host'}
          </div>
        </div>

        {/* Card 3: Active Jobs */}
        <div className="relative p-4 rounded-xl bg-surface hover:bg-surface-elevated/70 border border-surface-border hover:border-zinc-700 transition-all space-y-2 group overflow-hidden">
          <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-emerald-500 to-teal-500 opacity-60" />
          <div className="flex items-center justify-between text-xs text-content-secondary">
            <span className="font-medium">Orchestrator Queue</span>
            <Activity className="w-4 h-4 text-zinc-400 group-hover:text-emerald-400 transition-colors" />
          </div>
          <div className="text-xl font-mono font-bold text-content-primary tracking-tight">
            {activeJobs.length} <span className="text-xs font-normal text-zinc-500">active</span>
          </div>
          <div className="text-[11px] font-mono text-zinc-500">
            Queue Depth: {health?.orchestrator.queue_size ?? 0}
          </div>
        </div>

        {/* Card 4: Total Orchestrated */}
        <div className="relative p-4 rounded-xl bg-surface hover:bg-surface-elevated/70 border border-surface-border hover:border-zinc-700 transition-all space-y-2 group overflow-hidden">
          <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-purple-500 to-pink-500 opacity-60" />
          <div className="flex items-center justify-between text-xs text-content-secondary">
            <span className="font-medium">Total Workflows</span>
            <Clock className="w-4 h-4 text-zinc-400 group-hover:text-purple-400 transition-colors" />
          </div>
          <div className="text-xl font-mono font-bold text-content-primary tracking-tight">
            {jobsList?.total ?? 0}
          </div>
          <div className="text-[11px] font-mono text-zinc-500">
            Persisted in SQLite DB
          </div>
        </div>
      </div>

      {/* Quick Action Navigation Panels */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div
          onClick={() => onNavigate('explorer')}
          className="p-5 rounded-xl bg-surface hover:bg-surface-elevated border border-surface-border hover:border-zinc-600 cursor-pointer transition-all space-y-2.5 group shadow-sm"
        >
          <div className="flex items-center justify-between">
            <span className="font-bold text-sm text-content-primary group-hover:text-white">Model Explorer</span>
            <ArrowRight className="w-4 h-4 text-zinc-500 group-hover:text-accent-red group-hover:translate-x-0.5 transition-all" />
          </div>
          <p className="text-xs text-content-secondary leading-relaxed">
            Inspect Hugging Face model metadata, resolve immutable commit SHAs, and verify architecture compatibility.
          </p>
        </div>

        <div
          onClick={() => onNavigate('forge')}
          className="p-5 rounded-xl bg-surface hover:bg-surface-elevated border border-surface-border hover:border-zinc-600 cursor-pointer transition-all space-y-2.5 group shadow-sm"
        >
          <div className="flex items-center justify-between">
            <span className="font-bold text-sm text-content-primary group-hover:text-white">Forge Studio</span>
            <ArrowRight className="w-4 h-4 text-zinc-500 group-hover:text-accent-red group-hover:translate-x-0.5 transition-all" />
          </div>
          <p className="text-xs text-content-secondary leading-relaxed">
            Generate deterministic ForgePlans and prepare causal language models for AMD GPUs with standalone launchers.
          </p>
        </div>

        <div
          onClick={() => onNavigate('engineer')}
          className="p-5 rounded-xl bg-surface hover:bg-surface-elevated border border-surface-border hover:border-zinc-600 cursor-pointer transition-all space-y-2.5 group shadow-sm"
        >
          <div className="flex items-center justify-between">
            <span className="font-bold text-sm text-content-primary group-hover:text-white">AI Engineer</span>
            <ArrowRight className="w-4 h-4 text-zinc-500 group-hover:text-accent-red group-hover:translate-x-0.5 transition-all" />
          </div>
          <p className="text-xs text-content-secondary leading-relaxed">
            Run autonomous model preparation loops with structured thought/action trajectories and error recovery.
          </p>
        </div>
      </div>

      {/* Recent Jobs Table */}
      <div className="rounded-xl bg-surface border border-surface-border overflow-hidden shadow-sm">
        <div className="px-5 py-3.5 border-b border-surface-border flex items-center justify-between bg-[#141418]">
          <div className="flex items-center space-x-2">
            <h2 className="text-sm font-bold text-content-primary">Recent Execution Jobs</h2>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-surface-elevated text-zinc-400 border border-surface-border">
              {jobs.length} latest
            </span>
          </div>
          <span className="text-xs text-zinc-500 font-mono">SQLite Orchestrator DB</span>
        </div>

        {isLoadingJobs ? (
          <div className="p-12 text-center text-xs text-content-secondary font-mono space-y-2">
            <div className="w-6 h-6 border-2 border-accent-red border-t-transparent rounded-full animate-spin mx-auto" />
            <div>Loading job history...</div>
          </div>
        ) : jobs.length === 0 ? (
          <div className="p-10 text-center space-y-3">
            <img src={emptyJobsImg} alt="No jobs" className="w-44 h-32 mx-auto opacity-70" />
            <div className="space-y-1">
              <div className="text-xs font-mono font-semibold text-zinc-300">No Execution History Yet</div>
              <p className="text-[11px] text-zinc-500 max-w-sm mx-auto">
                Launch a Forge build, inspect a model, or trigger an autonomous AI Engineer session to begin.
              </p>
            </div>
            <button
              onClick={() => onNavigate('forge')}
              className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-surface-elevated hover:bg-zinc-700 text-zinc-300 hover:text-white border border-surface-border text-xs font-mono transition-colors"
            >
              <span>Open Forge Studio</span>
              <ArrowRight className="w-3 h-3 text-accent-red" />
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-[#121216] text-content-secondary border-b border-surface-border">
                <tr>
                  <th className="px-4 py-3">Job ID</th>
                  <th className="px-4 py-3">Type</th>
                  <th className="px-4 py-3">Model</th>
                  <th className="px-4 py-3">Status</th>
                  <th className="px-4 py-3">Domain Status</th>
                  <th className="px-4 py-3">Created At</th>
                  <th className="px-4 py-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-border">
                {jobs.map((job) => (
                  <tr key={job.job_id} className="hover:bg-surface-elevated/40 transition-colors">
                    <td className="px-4 py-3 text-content-primary font-semibold">
                      {job.job_id}
                    </td>
                    <td className="px-4 py-3 text-zinc-400">
                      {job.job_type}
                    </td>
                    <td className="px-4 py-3 text-content-primary max-w-[200px] truncate" title={job.model_id}>
                      {job.model_id}
                    </td>
                    <td className="px-4 py-3">
                      <StatusBadge status={job.status} />
                    </td>
                    <td className="px-4 py-3">
                      <DomainStatusTag status={job.domain_status} />
                    </td>
                    <td className="px-4 py-3 text-zinc-500 whitespace-nowrap">
                      {job.created_at ? new Date(job.created_at).toLocaleString() : '-'}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button
                        onClick={() => onSelectJob(job.job_id)}
                        className="px-2.5 py-1 rounded bg-surface-elevated hover:bg-zinc-700 text-zinc-200 hover:text-white border border-surface-border transition-colors text-[11px]"
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
