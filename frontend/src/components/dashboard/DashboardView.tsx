import React from 'react';
import type { HealthResponse, JobListResponse } from '../../api/types';
import { StatusBadge } from '../common/StatusBadge';
import { DomainStatusTag } from '../common/DomainStatusTag';
import {
  Search,
  PackageCheck,
  BrainCircuit,
  SlidersHorizontal,
  ArrowRight,
  AlertTriangle,
  ExternalLink,
} from 'lucide-react';
import type { NavTab } from '../layout/Sidebar';

import dashboardHeroImg from '../../assets/visuals/dashboard_hero.jpg';

interface DashboardViewProps {
  health: HealthResponse | null;
  jobsList: JobListResponse | null;
  isLoadingJobs: boolean;
  onNavigate: (tab: NavTab) => void;
  onSelectJob: (jobId: string) => void;
}

const FEATURES = [
  {
    tab: 'explorer' as NavTab,
    icon: Search,
    title: 'Model Explorer',
    description: 'Inspect any Hugging Face model — resolve an immutable commit SHA and assess AMD hardware compatibility before you build.',
    cta: 'Explore Models',
    accent: 'from-blue-500/20 to-blue-500/5 border-blue-500/20 group-hover:border-blue-500/40',
    iconColor: 'text-blue-400',
  },
  {
    tab: 'forge' as NavTab,
    icon: PackageCheck,
    title: 'Forge Studio',
    description: 'Generate a deterministic build plan, package model configs, and produce standalone AMD GPU launchers with verified SHA256 digests.',
    cta: 'Open Studio',
    accent: 'from-accent-red/20 to-accent-red/5 border-accent-red/20 group-hover:border-accent-red/40',
    iconColor: 'text-accent-red',
  },
  {
    tab: 'engineer' as NavTab,
    icon: BrainCircuit,
    title: 'AI Engineer',
    description: 'Run an autonomous model preparation session that inspects, configures, and validates models through an iterative agent loop.',
    cta: 'Start Session',
    accent: 'from-violet-500/20 to-violet-500/5 border-violet-500/20 group-hover:border-violet-500/40',
    iconColor: 'text-violet-400',
  },
  {
    tab: 'optimization' as NavTab,
    icon: SlidersHorizontal,
    title: 'Optimization Lab',
    description: 'Compare precision strategies side by side. Results reflect truthful NOT_MEASURED on non-AMD hosts — no synthetic benchmarks.',
    cta: 'Compare Strategies',
    accent: 'from-emerald-500/20 to-emerald-500/5 border-emerald-500/20 group-hover:border-emerald-500/40',
    iconColor: 'text-emerald-400',
  },
];

const JOB_TYPE_LABELS: Record<string, string> = {
  FORGE_BUILD: 'Forge Build',
  ENGINEER: 'AI Engineer',
  OPTIMIZATION: 'Optimization',
};

const JOB_TYPE_COLORS: Record<string, string> = {
  FORGE_BUILD: 'text-accent-red bg-accent-red/10 border-accent-red/20',
  ENGINEER: 'text-violet-400 bg-violet-500/10 border-violet-500/20',
  OPTIMIZATION: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
};

export const DashboardView: React.FC<DashboardViewProps> = ({
  health,
  jobsList,
  isLoadingJobs,
  onNavigate,
  onSelectJob,
}) => {
  const jobs = jobsList?.items || [];

  return (
    <div className="page-fade">
      {/* ── HERO ─────────────────────────────────────────────────────── */}
      <div className="relative overflow-hidden min-h-[320px] flex items-end">
        {/* Background image — high visibility */}
        <div
          className="absolute inset-0 bg-cover bg-right-top opacity-55"
          style={{ backgroundImage: `url(${dashboardHeroImg})` }}
        />
        <div className="absolute inset-0 hero-overlay-subtle" />

        {/* Hardware warning banner — sits within hero area */}
        {health && !health.rocm_available && (
          <div className="absolute top-0 left-0 right-0 z-10">
            <div className="flex items-center gap-2 px-6 py-2.5 bg-amber-500/10 border-b border-amber-500/20 text-xs">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
              <span className="text-amber-300 font-medium">CONFIG ONLY mode —</span>
              <span className="text-amber-400/80">
                {health.warnings[0] || 'No AMD ROCm GPU detected. Builds generate configs without downloading model weights.'}
              </span>
            </div>
          </div>
        )}

        {/* Hero content */}
        <div className="relative z-10 px-8 sm:px-12 pb-10 pt-20 max-w-3xl">
          <p className="text-xs font-mono font-medium text-accent-red/90 uppercase tracking-wider mb-3">
            AMD ROCm Engineering Platform
          </p>
          <h1 className="text-3xl sm:text-4xl font-black tracking-tight text-white leading-tight mb-4">
            Prepare AI Models<br />
            <span className="text-content-secondary font-bold">for AMD Hardware</span>
          </h1>
          <p className="text-sm text-content-secondary leading-relaxed mb-7 max-w-xl">
            Inspect, package, and autonomously optimize language models for AMD ROCm GPUs —
            with deterministic builds, immutable SHAs, and zero synthetic benchmarks.
          </p>
          <div className="flex flex-wrap gap-3">
            <button
              onClick={() => onNavigate('forge')}
              className="btn-primary"
            >
              Open Forge Studio
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => onNavigate('explorer')}
              className="btn-secondary"
            >
              <Search className="w-4 h-4" />
              Inspect a Model
            </button>
          </div>
        </div>
      </div>

      {/* ── FEATURE CARDS ─────────────────────────────────────────────── */}
      <div className="px-6 sm:px-8 py-8">
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
          {FEATURES.map((f) => {
            const Icon = f.icon;
            return (
              <button
                key={f.tab}
                onClick={() => onNavigate(f.tab)}
                className={`group text-left p-5 rounded-xl border bg-gradient-to-br transition-all duration-200 hover:shadow-lg hover:shadow-black/20 hover:-translate-y-0.5 ${f.accent}`}
              >
                <div className={`w-8 h-8 rounded-lg bg-surface flex items-center justify-center mb-4 border border-surface-border`}>
                  <Icon className={`w-4 h-4 ${f.iconColor}`} />
                </div>
                <div className="text-sm font-semibold text-content-primary mb-1.5">{f.title}</div>
                <p className="text-xs text-content-secondary leading-relaxed">{f.description}</p>
                <div className={`mt-4 flex items-center gap-1 text-xs font-medium ${f.iconColor} opacity-70 group-hover:opacity-100 transition-opacity`}>
                  <span>{f.cta}</span>
                  <ArrowRight className="w-3 h-3 group-hover:translate-x-0.5 transition-transform" />
                </div>
              </button>
            );
          })}
        </div>

        {/* ── RECENT JOBS ─────────────────────────────────────────────── */}
        <div className="mt-8 rounded-xl border border-surface-border bg-surface overflow-hidden">
          <div className="px-6 py-4 border-b border-surface-border flex items-center justify-between bg-surface-deep">
            <h2 className="text-sm font-semibold text-content-primary">Recent Jobs</h2>
            {jobs.length > 0 && (
              <span className="text-xs text-content-muted font-mono">{jobs.length} records</span>
            )}
          </div>

          {isLoadingJobs ? (
            <div className="p-10 flex flex-col items-center gap-3">
              <div className="w-5 h-5 border-2 border-accent-red border-t-transparent rounded-full animate-spin" />
              <span className="text-xs text-content-muted">Loading...</span>
            </div>
          ) : jobs.length === 0 ? (
            <div className="p-12 text-center">
              <div className="w-12 h-12 rounded-full bg-surface-elevated border border-surface-border flex items-center justify-center mx-auto mb-4">
                <PackageCheck className="w-5 h-5 text-content-muted" />
              </div>
              <div className="text-sm font-medium text-content-secondary mb-1">No jobs yet</div>
              <p className="text-xs text-content-muted max-w-xs mx-auto leading-relaxed">
                Start with Model Explorer to inspect a model, then open Forge Studio to run your first build.
              </p>
              <button
                onClick={() => onNavigate('forge')}
                className="mt-5 btn-secondary text-xs"
              >
                Open Forge Studio
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-surface-deep border-b border-surface-border">
                  <tr>
                    <th className="px-5 py-3 text-content-muted font-medium">Job ID</th>
                    <th className="px-5 py-3 text-content-muted font-medium">Type</th>
                    <th className="px-5 py-3 text-content-muted font-medium">Model</th>
                    <th className="px-5 py-3 text-content-muted font-medium">Status</th>
                    <th className="px-5 py-3 text-content-muted font-medium">Result</th>
                    <th className="px-5 py-3 text-content-muted font-medium">Created</th>
                    <th className="px-5 py-3 text-right text-content-muted font-medium">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-surface-border">
                  {jobs.map((job) => (
                    <tr key={job.job_id} className="hover:bg-surface-elevated/40 transition-colors">
                      <td className="px-5 py-3 font-mono text-content-primary font-semibold">
                        {job.job_id}
                      </td>
                      <td className="px-5 py-3">
                        <span className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium border font-mono ${JOB_TYPE_COLORS[job.job_type] || 'text-content-secondary bg-surface-elevated border-surface-border'}`}>
                          {JOB_TYPE_LABELS[job.job_type] || job.job_type}
                        </span>
                      </td>
                      <td className="px-5 py-3 text-content-primary max-w-[220px] truncate font-mono" title={job.model_id}>
                        {job.model_id}
                      </td>
                      <td className="px-5 py-3">
                        <StatusBadge status={job.status} />
                      </td>
                      <td className="px-5 py-3">
                        <DomainStatusTag status={job.domain_status} />
                      </td>
                      <td className="px-5 py-3 text-content-muted whitespace-nowrap font-mono">
                        {job.created_at ? new Date(job.created_at).toLocaleString() : '—'}
                      </td>
                      <td className="px-5 py-3 text-right">
                        <button
                          onClick={() => onSelectJob(job.job_id)}
                          className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-surface-elevated hover:bg-zinc-700 text-content-secondary hover:text-content-primary border border-surface-border transition-colors text-[11px]"
                        >
                          <ExternalLink className="w-3 h-3" />
                          Open
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
    </div>
  );
};
