import React from 'react';
import type { HealthResponse, JobListResponse } from '../../api/types';
import { StatusBadge } from '../common/StatusBadge';
import { DomainStatusTag } from '../common/DomainStatusTag';
import { Cpu, Server, Activity, ArrowRight, Clock, AlertCircle } from 'lucide-react';
import type { NavTab } from '../layout/Sidebar';

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
      {/* Top Header */}
      <div>
        <h1 className="text-xl font-bold text-content-primary">System Dashboard</h1>
        <p className="text-xs text-content-secondary mt-0.5">
          Real-time hardware status, task orchestration queue, and model preparation activity.
        </p>
      </div>

      {/* Warning if on non-AMD hardware */}
      {health && !health.rocm_available && (
        <div className="p-4 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-start space-x-3 text-xs">
          <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <span className="font-semibold text-amber-300">Host Hardware Notice</span>
            <p className="text-amber-400/90 leading-relaxed">
              {health.warnings[0] ||
                'Running on host without AMD ROCm GPU acceleration. Model preparation and builds operate strictly in CONFIG_ONLY mode without synthetic GPU performance metrics.'}
            </p>
          </div>
        </div>
      )}

      {/* Metric Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Card 1: Backend */}
        <div className="p-4 rounded-lg bg-surface border border-surface-border space-y-2">
          <div className="flex items-center justify-between text-xs text-content-secondary">
            <span>FastAPI Server</span>
            <Server className="w-4 h-4 text-zinc-400" />
          </div>
          <div className="text-lg font-mono font-bold text-content-primary">
            {health ? `v${health.version}` : 'Disconnected'}
          </div>
          <div className="text-[11px] font-mono text-zinc-500">
            {health ? `Python ${health.host_platform.python_version} (${health.host_platform.os})` : 'Offline'}
          </div>
        </div>

        {/* Card 2: ROCm Hardware */}
        <div className="p-4 rounded-lg bg-surface border border-surface-border space-y-2">
          <div className="flex items-center justify-between text-xs text-content-secondary">
            <span>AMD ROCm Hardware</span>
            <Cpu className="w-4 h-4 text-zinc-400" />
          </div>
          <div className="text-lg font-mono font-bold">
            {health?.rocm_available ? (
              <span className="text-emerald-400">{health.system.gpus[0]?.gfx_target || 'AMD GPU'}</span>
            ) : (
              <span className="text-amber-400">None</span>
            )}
          </div>
          <div className="text-[11px] font-mono text-zinc-500">
            {health?.rocm_available
              ? `${health.system.gpus_detected} device(s) discovered`
              : 'CONFIG_ONLY fallback'}
          </div>
        </div>

        {/* Card 3: Active Jobs */}
        <div className="p-4 rounded-lg bg-surface border border-surface-border space-y-2">
          <div className="flex items-center justify-between text-xs text-content-secondary">
            <span>Active Jobs</span>
            <Activity className="w-4 h-4 text-zinc-400" />
          </div>
          <div className="text-lg font-mono font-bold text-content-primary">
            {activeJobs.length}
          </div>
          <div className="text-[11px] font-mono text-zinc-500">
            Queue Depth: {health?.orchestrator.queue_size ?? 0}
          </div>
        </div>

        {/* Card 4: Total Orchestrated */}
        <div className="p-4 rounded-lg bg-surface border border-surface-border space-y-2">
          <div className="flex items-center justify-between text-xs text-content-secondary">
            <span>Total Jobs</span>
            <Clock className="w-4 h-4 text-zinc-400" />
          </div>
          <div className="text-lg font-mono font-bold text-content-primary">
            {jobsList?.total ?? 0}
          </div>
          <div className="text-[11px] font-mono text-zinc-500">
            Persisted in SQLite
          </div>
        </div>
      </div>

      {/* Quick Action Launchers */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div
          onClick={() => onNavigate('explorer')}
          className="p-4 rounded-lg bg-surface hover:bg-surface-elevated border border-surface-border cursor-pointer transition-all hover:border-zinc-700 space-y-2 group"
        >
          <div className="flex items-center justify-between">
            <span className="font-semibold text-sm text-content-primary group-hover:text-white">Model Explorer</span>
            <ArrowRight className="w-4 h-4 text-zinc-500 group-hover:text-accent-red transition-colors" />
          </div>
          <p className="text-xs text-content-secondary">
            Inspect Hugging Face model metadata, resolve immutable commit SHA, and verify architecture.
          </p>
        </div>

        <div
          onClick={() => onNavigate('forge')}
          className="p-4 rounded-lg bg-surface hover:bg-surface-elevated border border-surface-border cursor-pointer transition-all hover:border-zinc-700 space-y-2 group"
        >
          <div className="flex items-center justify-between">
            <span className="font-semibold text-sm text-content-primary group-hover:text-white">Forge Studio</span>
            <ArrowRight className="w-4 h-4 text-zinc-500 group-hover:text-accent-red transition-colors" />
          </div>
          <p className="text-xs text-content-secondary">
            Generate deterministic ForgePlans and prepare causal language models for AMD GPUs.
          </p>
        </div>

        <div
          onClick={() => onNavigate('engineer')}
          className="p-4 rounded-lg bg-surface hover:bg-surface-elevated border border-surface-border cursor-pointer transition-all hover:border-zinc-700 space-y-2 group"
        >
          <div className="flex items-center justify-between">
            <span className="font-semibold text-sm text-content-primary group-hover:text-white">AI Engineer</span>
            <ArrowRight className="w-4 h-4 text-zinc-500 group-hover:text-accent-red transition-colors" />
          </div>
          <p className="text-xs text-content-secondary">
            Run autonomous model preparation loops with structured actions and error recovery.
          </p>
        </div>
      </div>

      {/* Recent Jobs Table */}
      <div className="rounded-lg bg-surface border border-surface-border overflow-hidden">
        <div className="px-4 py-3 border-b border-surface-border flex items-center justify-between">
          <h2 className="text-sm font-semibold text-content-primary">Recent Execution Jobs</h2>
          <span className="text-xs text-zinc-500 font-mono">Showing {jobs.length} latest</span>
        </div>

        {isLoadingJobs ? (
          <div className="p-8 text-center text-xs text-content-secondary font-mono">
            Loading job history...
          </div>
        ) : jobs.length === 0 ? (
          <div className="p-8 text-center text-xs text-content-secondary font-mono">
            No jobs recorded yet. Launch a Forge build or AI Engineer task to begin.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-[#141418] text-content-secondary border-b border-surface-border">
                <tr>
                  <th className="px-4 py-2.5">Job ID</th>
                  <th className="px-4 py-2.5">Type</th>
                  <th className="px-4 py-2.5">Model</th>
                  <th className="px-4 py-2.5">Status</th>
                  <th className="px-4 py-2.5">Domain Status</th>
                  <th className="px-4 py-2.5">Created At</th>
                  <th className="px-4 py-2.5 text-right">Action</th>
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
                        className="px-2 py-1 rounded bg-surface-elevated hover:bg-zinc-700 text-zinc-300 transition-colors text-[11px]"
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
