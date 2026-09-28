import React from 'react';
import type { AgentInfo, HealthResponse, JobListResponse } from '../../api/types';
import type { NavTab } from '../layout/Sidebar';
import { ArrowRight, Clock3, RotateCcw } from 'lucide-react';
import dashboardHeroImg from '../../assets/visuals/dashboard_hero.jpg';
import { AGENT_STATUS_COPY, agentSummary, DOMAIN_STATUS_COPY, JOB_STATUS_COPY, JOB_TYPE_COPY } from '../../ui/presentation';

interface Props {
  health: HealthResponse | null;
  jobsList: JobListResponse | null;
  isLoadingJobs: boolean;
  isJobsError?: boolean;
  onRetryJobs?: () => void;
  onNavigate: (tab: NavTab) => void;
  onSelectJob: (jobId: string) => void;
  agents?: AgentInfo[];
  isLoadingAgents?: boolean;
  isAgentsError?: boolean;
  onRetryAgents?: () => void;
}

export const DashboardView: React.FC<Props> = ({ health, jobsList, isLoadingJobs, isJobsError = false, onRetryJobs, onNavigate, onSelectJob, agents = [], isLoadingAgents = false, isAgentsError = false, onRetryAgents }) => {
  const jobs = jobsList?.items || [];
  const onlineCount = agents.filter((agent) => agent.status === 'ONLINE').length;

  return <div className="page-fade">
    <section className="relative min-h-[500px] overflow-hidden border-b hairline flex items-center">
      <img src={dashboardHeroImg} alt="Compute infrastructure" className="hero-image absolute inset-y-0 right-0 w-[67%] h-full object-cover opacity-70"/>
      <div className="absolute inset-0 bg-gradient-to-r from-[#09090b] via-[#09090b]/90 to-transparent"/><div className="absolute inset-0 grid-glow opacity-30"/>
      <div className="relative page-shell py-16 md:py-20"><div className="max-w-[760px]"><div className="eyebrow mb-6">Open model infrastructure for AMD</div><h1 className="display-title max-w-[720px]">Find a model.<br/><span className="text-zinc-500">Prepare it with confidence.</span></h1><p className="lede max-w-xl mt-7">Inspect an open model, choose what to prepare, and follow the result in one place. Hardware execution is reported only when it actually happens.</p><div className="flex flex-wrap gap-3 mt-9"><button onClick={()=>onNavigate('explorer')} className="btn-primary">Find a model <ArrowRight size={16}/></button></div><div className="mt-7 flex items-center gap-2 text-xs text-zinc-500"><span className={`w-2 h-2 rounded-full ${health?.rocm_available?'bg-emerald-400':'bg-amber-400'}`}/>{health?.rocm_available?'ROCm available on this host':'Preparation is available · hardware execution requires a ROCm host'}</div></div></div>
    </section>

    <section className="page-shell py-10"><div className="focus-panel p-5 md:p-7"><div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2"><div><div className="eyebrow mb-2">Agents</div><h2 className="text-xl font-semibold">Preparation capacity</h2></div>{!isLoadingAgents&&!isAgentsError&&agents.length>0&&<span className="text-xs text-zinc-400">{onlineCount} available</span>}</div>
      {isLoadingAgents?<div aria-label="Loading agents" className="mt-5 h-16 rounded-xl shimmer"/>:isAgentsError?<div role="status" className="mt-5 rounded-xl border border-red-500/15 bg-red-500/[.04] p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3"><div><div className="text-sm font-medium">Agent status couldn’t be loaded</div><p className="text-xs text-zinc-500 mt-1">The connection state is unknown. This doesn’t mean there are no agents.</p></div><button className="btn-ghost shrink-0" onClick={onRetryAgents}><RotateCcw size={13}/>Retry</button></div>:agents.length===0?<p className="mt-5 text-sm text-zinc-500">No agents are registered yet. You can still inspect models and prepare configuration-only outputs.</p>:<><p className="mt-4 text-xs text-zinc-500">{agentSummary(agents)}</p><div className="grid md:grid-cols-2 gap-3 mt-4">{agents.map(agent=><div key={agent.agent_id} className="rounded-xl border hairline p-4"><div className="flex justify-between gap-3"><div className="font-medium truncate">{agent.name}</div><span className={`shrink-0 text-xs ${agent.status==='ONLINE'||agent.status==='BUSY'?'text-emerald-400':agent.status==='DEGRADED'?'text-amber-300':'text-zinc-500'}`}>{AGENT_STATUS_COPY[agent.status]}</span></div><p className="text-sm text-zinc-500 mt-2">{agent.capabilities.rocm_detected&&agent.capabilities.amd_gpu_count?'ROCm hardware detected':'Configuration preparation only'}</p><details className="mt-3"><summary className="text-xs text-zinc-600 cursor-pointer">Technical details</summary><p className="text-xs text-zinc-500 mt-2">{agent.hostname} · {agent.capabilities.gpu_names.join(', ')||'No AMD GPU detected'} · Last seen {new Date(agent.last_seen).toLocaleString()} · <span className="font-mono">{agent.status}</span></p></details></div>)}</div></>}
    </div></section>

    <section className="page-shell pt-4 pb-20"><div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-6"><div><div className="eyebrow mb-2">Workspace</div><h2 className="text-2xl font-semibold tracking-[-.03em]">{jobs.length?'Recent runs':'Your runs will appear here'}</h2></div><button onClick={()=>onNavigate('runs')} className="btn-ghost shrink-0">All runs <ArrowRight size={14}/></button></div>
      {isLoadingJobs?<div aria-label="Loading recent runs" className="grid md:grid-cols-2 xl:grid-cols-3 gap-3">{[0,1,2].map(i=><div key={i} className="h-36 rounded-2xl shimmer"/>)}</div>:isJobsError?<div className="focus-panel p-6"><h3 className="font-semibold">Recent runs couldn’t be loaded</h3><p className="text-sm text-zinc-500 mt-1">Your work is unchanged. Retry or open all runs.</p><div className="flex flex-wrap gap-2 mt-4"><button className="btn-secondary" onClick={onRetryJobs}><RotateCcw size={13}/>Retry</button><button className="btn-ghost" onClick={()=>onNavigate('runs')}>All runs <ArrowRight size={14}/></button></div></div>:jobs.length===0?<div className="focus-panel p-7 md:p-9 flex flex-col md:flex-row md:items-center justify-between gap-7"><div><h3 className="text-lg font-semibold">Start with a model</h3><p className="text-sm text-zinc-500 mt-2 max-w-xl leading-relaxed">Inspect its architecture and revision first. You can decide later whether to prepare a configuration or download model files.</p></div><button onClick={()=>onNavigate('explorer')} className="btn-primary shrink-0">Find a model <ArrowRight size={15}/></button></div>:<div className="grid md:grid-cols-2 xl:grid-cols-3 gap-3">{jobs.slice(0,6).map(job=><button key={job.job_id} onClick={()=>onSelectJob(job.job_id)} className="card-interactive p-5 text-left"><div className="flex justify-between gap-4"><span className="text-xs text-zinc-400">{JOB_TYPE_COPY[job.job_type]}</span><span className="text-xs text-zinc-500">{JOB_STATUS_COPY[job.status]}</span></div><div className="mt-5 font-semibold truncate">{job.model_id.split('/').pop()}</div><div className="text-xs text-zinc-600 mt-1 truncate">{job.model_id}</div><div className="mt-5 pt-4 border-t hairline flex items-center justify-between text-[11px] text-zinc-500"><span className="flex gap-1.5 items-center"><Clock3 size={12}/>{new Date(job.created_at).toLocaleDateString()}</span><span>{job.domain_status?DOMAIN_STATUS_COPY[job.domain_status].label:'Details'}</span></div></button>)}</div>}
    </section>
  </div>;
};
