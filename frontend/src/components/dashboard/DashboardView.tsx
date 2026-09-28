import React, { useState } from 'react';
import type { AgentInfo, HealthResponse, JobListResponse } from '../../api/types';
import type { NavTab } from '../layout/Sidebar';
import { ArrowUpRight, Clock3, Search } from 'lucide-react';
import { DOMAIN_STATUS_COPY, JOB_STATUS_COPY, JOB_TYPE_COPY } from '../../ui/presentation';

interface Props {
  health: HealthResponse | null; isLoadingHealth?: boolean; isHealthError?: boolean;
  jobsList: JobListResponse | null; isLoadingJobs: boolean; isJobsError?: boolean; onRetryJobs?: () => void;
  onNavigate: (tab: NavTab) => void; onSelectJob: (jobId: string) => void; onSearchModel?: (modelId: string) => void;
  agents?: AgentInfo[]; isLoadingAgents?: boolean; isAgentsError?: boolean; onRetryAgents?: () => void;
}

export const DashboardView: React.FC<Props> = ({ jobsList, isLoadingJobs, isJobsError = false, onRetryJobs, onNavigate, onSelectJob, onSearchModel }) => {
  const [modelId, setModelId] = useState('');
  const jobs = jobsList?.items ?? [];
  return <div className="start-screen page-fade">
    <section className="start-main">
      <div className="start-copy">
        <div className="start-kicker"><span className="brand-mini"/> AMD model workspace</div>
        <h1>Find a model.<br/><span>Make it yours.</span></h1>
        <p>Inspect an open model, prepare it for an AMD system, and keep every outcome in one place.</p>
        <form className="start-search" onSubmit={e => { e.preventDefault(); if (modelId.trim()) onSearchModel?.(modelId.trim()); else onNavigate('explorer'); }}>
          <Search size={18}/><input aria-label="Search Hugging Face models or paste a model ID" placeholder="Search Hugging Face models or paste a model ID" value={modelId} onChange={e => setModelId(e.target.value)}/><button type="submit" aria-label="Inspect model"><span>Inspect</span><ArrowUpRight size={16}/></button>
        </form>
        <button className="text-action" onClick={() => onNavigate('explorer')}>Browse recent models <ArrowUpRight size={14}/></button>
      </div>
      <div className="start-art" aria-hidden="true"><div className="art-ring art-ring-one"/><div className="art-ring art-ring-two"/><div className="art-core"/><span className="art-caption">MODEL → AMD</span></div>
    </section>
    <section className="start-lower">
      <div className="recent-strip">
        <div className="strip-heading"><div><span className="eyebrow">Continue where you left off</span><h2>Recent activity</h2></div><button className="text-action" onClick={() => onNavigate('runs')}>All activity <ArrowUpRight size={14}/></button></div>
        {isLoadingJobs ? <div className="activity-skeleton" aria-label="Loading recent activity"><i/><i/><i/></div> : isJobsError ? <div className="activity-empty"><span>Recent activity couldn’t be loaded.</span><button onClick={onRetryJobs}>Retry <ArrowUpRight size={14}/></button></div> : jobs.length ? <div className="activity-list">{jobs.slice(0, 3).map(job => <button className="activity-row" key={job.job_id} onClick={() => onSelectJob(job.job_id)}><span className={`activity-symbol activity-${job.status.toLowerCase()}`}>{job.status === 'SUCCEEDED' ? '✓' : job.status === 'FAILED' ? '!' : job.status === 'CANCELLED' ? '−' : '·'}</span><span className="activity-model">{job.model_id.split('/').pop()}<small>{job.model_id}</small></span><span className="activity-action">{JOB_TYPE_COPY[job.job_type]}<small>{job.domain_status ? DOMAIN_STATUS_COPY[job.domain_status].label : JOB_STATUS_COPY[job.status]}</small></span><span className="activity-time"><Clock3 size={13}/>{new Date(job.created_at).toLocaleDateString()}</span><ArrowUpRight size={15}/></button>)}</div> : <div className="activity-empty"><span>No activity yet</span><button onClick={() => onNavigate('explorer')}>Inspect your first model <ArrowUpRight size={14}/></button></div>}
      </div>
      <div className="start-note"><span className="note-mark">/</span><div><strong>Ready when compute is.</strong><p>Model discovery and configuration preparation work without an AMD GPU.</p></div></div>
    </section>
  </div>;
};
