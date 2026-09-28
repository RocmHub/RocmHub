import React, { useState } from 'react';
import type { AgentInfo, HealthResponse, JobListResponse } from '../../api/types';
import type { NavTab } from '../layout/Sidebar';
import { ArrowUpRight, Clock3, Search, ChevronRight } from 'lucide-react';
import { activityOutcomeCopy, AGENT_STATUS_COPY, hasConnectedAmdCompute } from '../../ui/presentation';

interface Props {
  health: HealthResponse | null; isLoadingHealth?: boolean; isHealthError?: boolean;
  jobsList: JobListResponse | null; isLoadingJobs: boolean; isJobsError?: boolean; onRetryJobs?: () => void;
  onNavigate: (tab: NavTab) => void; onSelectJob: (jobId: string) => void; onSearchModel?: (modelId: string) => void;
  agents?: AgentInfo[]; isLoadingAgents?: boolean; isAgentsError?: boolean; onRetryAgents?: () => void;
}

export const DashboardView: React.FC<Props> = ({ health, isLoadingHealth = false, isHealthError = false, jobsList, isLoadingJobs, isJobsError = false, onRetryJobs, onNavigate, onSelectJob, onSearchModel, agents = [], isLoadingAgents = false, isAgentsError = false, onRetryAgents }) => {
  const [modelId, setModelId] = useState('');
  const jobs = jobsList?.items ?? [];
  const checkingCompute = isLoadingHealth || isLoadingAgents;
  const computeUnknown = !checkingCompute && (isHealthError || isAgentsError || !health);
  const computeConnected = hasConnectedAmdCompute(health, agents);
  const connectedAgent = agents.find(agent => (agent.status === 'ONLINE' || agent.status === 'BUSY') && agent.capabilities.rocm_detected && agent.capabilities.amd_gpu_count > 0);
  const localAmdGpu = health?.rocm_available ? health.system.gpus.find(gpu => gpu.vendor.toLowerCase() === 'amd' || gpu.gpu_present) : undefined;
  const computeName = connectedAgent?.capabilities.gpu_names.filter(Boolean).join(', ') || localAmdGpu?.device_name;
  return <div className="start-screen page-fade">
    <section className="start-main">
      <div className="start-copy">
        <div className="start-kicker"><span className="brand-mini"/> AMD model workspace</div>
        <h1>Find a model.<br/><span>Make it yours.</span></h1>
        <p>Inspect a model, prepare its setup, then run it when compatible AMD compute is connected.</p>
        <form className="start-search" onSubmit={e => { e.preventDefault(); if (modelId.trim()) onSearchModel?.(modelId.trim()); else onNavigate('explorer'); }}>
          <Search size={18} aria-hidden="true"/><input aria-label="Paste a public Hugging Face model ID" placeholder="Hugging Face model ID" value={modelId} onChange={e => setModelId(e.target.value)}/><button type="submit" className="btn-primary home-inspect-button" aria-label="Inspect model"><span>Inspect</span><ArrowUpRight size={16} aria-hidden="true"/></button>
        </form>
        <div className="quick-examples" aria-label="Example models">{['Qwen/Qwen2.5-0.5B-Instruct','Qwen/Qwen2.5-1.5B-Instruct','Qwen/Qwen2.5-7B-Instruct'].map((id, i) => <button className="quick-model-button" key={id} aria-label={`Inspect ${id}`} onClick={() => onSearchModel?.(id)}>Qwen2.5 {['0.5B','1.5B','7B'][i]}</button>)}</div>
      </div>
      <div className="start-art" aria-hidden="true"><div className="art-ring art-ring-one"/><div className="art-ring art-ring-two"/><div className="art-core"/><span className="art-caption">MODEL → AMD</span></div>
    </section>
    <section className="start-lower">
      <div className="recent-strip">
        <div className="strip-heading"><div><span className="eyebrow">Your recent work</span><h2>Recent activity</h2></div><button className="text-action" onClick={() => onNavigate('runs')}>All activity <ArrowUpRight size={14}/></button></div>
        {isLoadingJobs ? <div className="activity-skeleton" aria-label="Loading recent activity"><i/><i/><i/></div> : isJobsError ? <div className="activity-empty"><span>Recent activity couldn’t be loaded. Your work is unchanged.</span><button onClick={onRetryJobs}>Retry <ArrowUpRight size={14}/></button></div> : jobs.length ? <div className="activity-list">{jobs.slice(0, 3).map(job => { const outcome = activityOutcomeCopy(job); const configOnly = job.domain_status === 'CONFIG_ONLY'; return <button className="activity-row" key={job.job_id} aria-label={`${job.model_id.split('/').pop()}, ${outcome.title}, ${new Date(job.created_at).toLocaleDateString()}. Open activity details`} onClick={() => onSelectJob(job.job_id)}><span className={`activity-symbol activity-${job.status.toLowerCase()}${configOnly ? ' activity-config-only' : ''}`} aria-hidden="true">{job.status === 'SUCCEEDED' ? '✓' : job.status === 'FAILED' ? '!' : job.status === 'CANCELLED' ? '−' : '·'}</span><span className="activity-model">{job.model_id.split('/').pop()}<small>{job.model_id}</small></span><span className="activity-action">{outcome.title}<small>{outcome.description}</small></span><time className="activity-time" dateTime={job.created_at}><Clock3 size={13} aria-hidden="true"/>{new Date(job.created_at).toLocaleDateString()}</time><ArrowUpRight size={15} aria-hidden="true"/></button>; })}</div> : <div className="activity-empty"><span>No recent work yet</span><button onClick={() => onNavigate('explorer')}>Inspect a model to get started <ArrowUpRight size={14}/></button></div>}
      </div>
      <aside className="home-supporting">
        <section className="compute-summary" aria-label="AMD compute status">
          <div className="compute-summary-top"><div><span className="eyebrow">Compute</span><h2>{checkingCompute ? 'Checking compute' : computeUnknown ? 'Compute unavailable' : computeConnected ? 'AMD compute connected' : 'No AMD compute'}</h2>{computeConnected && computeName && <p>{computeName}</p>}</div><span className={`compute-mark ${computeConnected ? 'compute-mark-on' : checkingCompute ? 'compute-mark-wait' : ''}`}/></div>
          <p>{computeUnknown ? 'We could not confirm the connection. Retry the check or open compute details.' : computeConnected ? 'You can prepare models now. Actual execution still requires an action that records AMD run evidence.' : 'You can inspect models and prepare configuration without AMD hardware. Connect compute when you are ready to run or measure a model.'}</p>
          <details><summary>{computeUnknown ? 'Retry compute check' : 'Compute details'}</summary>{computeUnknown ? <button className="text-action" onClick={() => { onRetryAgents?.(); }}>Retry connection check <ArrowUpRight size={14}/></button> : <div className="compute-summary-details">{isLoadingAgents ? <span>Checking connected compute…</span> : agents.length ? agents.map(agent => <div key={agent.agent_id}><strong>{agent.name}</strong><span>{AGENT_STATUS_COPY[agent.status]}{agent.capabilities.gpu_names.length ? ` · ${agent.capabilities.gpu_names.join(', ')}` : ''}</span><details className="technical-details"><summary>Technical details</summary><span>{agent.hostname} · {agent.capabilities.capabilities.join(', ') || 'No capabilities reported'} · {new Date(agent.last_seen).toLocaleString()}</span></details></div>) : <span>No connected compute has reported to this workspace.</span>}{health?.rocm_available && localAmdGpu && <div><strong>{localAmdGpu.device_name}</strong><span>Local AMD compute</span></div>}{(isAgentsError || isHealthError) && <button onClick={() => { onRetryAgents?.(); }}>Retry connection check</button>}</div>}</details>
        </section>
        <div className="how-it-works"><div className="eyebrow">Your path</div><ol>{[
          ['Find','Inspect a public model.'],
          ['Prepare','Choose configuration only or download files.'],
          ['Run','Requires compatible AMD compute.'],
        ].map(([title, copy], index) => <li key={title}><span className="journey-index">0{index + 1}</span><div><strong>{title}</strong><p>{copy}</p></div>{index < 2 && <ChevronRight className="journey-arrow" size={15}/>}</li>)}</ol></div>
      </aside>
    </section>
  </div>;
};
