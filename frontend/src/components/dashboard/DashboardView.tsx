import React, { useState } from 'react';
import type { AgentInfo, HealthResponse, JobListResponse } from '../../api/types';
import type { NavTab } from '../layout/Sidebar';
import { ArrowUpRight, Clock3, Search, ChevronRight } from 'lucide-react';
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
        <p>Prepare, run and validate open models for AMD hardware — with every step and result recorded.</p>
        <form className="start-search" onSubmit={e => { e.preventDefault(); if (modelId.trim()) onSearchModel?.(modelId.trim()); else onNavigate('explorer'); }}>
          <Search size={18}/><input aria-label="Search Hugging Face models or paste a model ID" placeholder="Search Hugging Face models or paste a model ID" value={modelId} onChange={e => setModelId(e.target.value)}/><button type="submit" aria-label="Inspect model"><span>Inspect</span><ArrowUpRight size={16}/></button>
        </form>
        <button className="text-action" onClick={() => onNavigate('explorer')}>Browse models <ArrowUpRight size={14}/></button>
        <div className="quick-examples" aria-label="Example models">{['Qwen/Qwen2.5-0.5B-Instruct','Qwen/Qwen2.5-1.5B-Instruct','Qwen/Qwen2.5-7B-Instruct'].map((id, i) => <button key={id} onClick={() => onSearchModel?.(id)}>Qwen2.5 {['0.5B','1.5B','7B'][i]}</button>)}</div>
      </div>
      <div className="start-art" aria-hidden="true"><div className="art-ring art-ring-one"/><div className="art-ring art-ring-two"/><div className="art-core"/><span className="art-caption">MODEL → AMD</span></div>
    </section>
    <section className="start-lower">
      <div className="recent-strip">
        <div className="strip-heading"><div><span className="eyebrow">Continue where you left off</span><h2>Recent activity</h2></div><button className="text-action" onClick={() => onNavigate('runs')}>All activity <ArrowUpRight size={14}/></button></div>
        {isLoadingJobs ? <div className="activity-skeleton" aria-label="Loading recent activity"><i/><i/><i/></div> : isJobsError ? <div className="activity-empty"><span>Recent activity couldn’t be loaded.</span><button onClick={onRetryJobs}>Retry <ArrowUpRight size={14}/></button></div> : jobs.length ? <div className="activity-list">{jobs.slice(0, 3).map(job => <button className="activity-row" key={job.job_id} onClick={() => onSelectJob(job.job_id)}><span className={`activity-symbol activity-${job.status.toLowerCase()}`}>{job.status === 'SUCCEEDED' ? '✓' : job.status === 'FAILED' ? '!' : job.status === 'CANCELLED' ? '−' : '·'}</span><span className="activity-model">{job.model_id.split('/').pop()}<small>{job.model_id}</small></span><span className="activity-action">{JOB_TYPE_COPY[job.job_type]}<small>{job.domain_status ? DOMAIN_STATUS_COPY[job.domain_status].label : JOB_STATUS_COPY[job.status]}</small></span><span className="activity-time"><Clock3 size={13}/>{new Date(job.created_at).toLocaleDateString()}</span><ArrowUpRight size={15}/></button>)}</div> : <div className="activity-empty"><span>No activity yet</span><button onClick={() => onNavigate('explorer')}>Inspect your first model <ArrowUpRight size={14}/></button></div>}
      </div>
      <div className="how-it-works"><div className="eyebrow">How ROCmHub works</div><ol>{[
        ['Find','Inspect any public model.','Available'],
        ['Prepare','Create a reproducible AMD-ready setup.','Available'],
        ['Run','Execute on compatible AMD hardware.','Requires AMD compute'],
        ['Validate','Review execution and performance evidence.','Requires execution'],
      ].map(([title, copy, state], index) => <li key={title}><span className="journey-index">0{index + 1}</span><div><strong>{title}</strong><p>{copy}</p><small className={index > 1 ? 'journey-locked' : ''}>{state}</small></div>{index < 3 && <ChevronRight className="journey-arrow" size={15}/>}</li>)}</ol></div>
    </section>
  </div>;
};
