import React from 'react';
import type { AgentInfo, HealthResponse, JobListResponse } from '../../api/types';
import type { NavTab } from '../layout/Sidebar';
import { ArrowRight, Check, Clock3, Cpu, Layers3, RotateCcw, Search } from 'lucide-react';
import dashboardHeroImg from '../../assets/visuals/rocmhub-home-hero.jpg';
import { AGENT_STATUS_COPY, DOMAIN_STATUS_COPY, hasConnectedAmdCompute, JOB_STATUS_COPY, JOB_TYPE_COPY } from '../../ui/presentation';

interface Props {
  health: HealthResponse | null;
  isLoadingHealth?: boolean;
  isHealthError?: boolean;
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

export const DashboardView: React.FC<Props> = ({ health, isLoadingHealth = false, isHealthError = false, jobsList, isLoadingJobs, isJobsError = false, onRetryJobs, onNavigate, onSelectJob, agents = [], isLoadingAgents = false, isAgentsError = false, onRetryAgents }) => {
  const jobs = jobsList?.items || [];
  const computeConnected = hasConnectedAmdCompute(health, agents);
  const computeUnknown = !computeConnected && (isLoadingAgents || isAgentsError || isLoadingHealth || isHealthError || !health);
  const computeTitle = computeConnected ? 'AMD compute available' : computeUnknown ? 'Compute availability unknown' : 'No compute connected';
  const computeSummary = 'Find and inspect public models without compute. Configuration-only preparation can be created without AMD hardware; a connected computer may be needed to process it. Downloading model files or measuring performance requires compute with the right capabilities.';

  return <div className="page-fade">
    <section className="relative min-h-[490px] xl:min-h-[540px] overflow-hidden border-b hairline flex items-center">
      <img src={dashboardHeroImg} alt="Dark compute infrastructure with red-lit server systems" className="hero-image absolute inset-0 w-full h-full object-cover object-center"/>
      <div className="hero-overlay absolute inset-0"/><div className="grid-glow absolute inset-0 opacity-30 pointer-events-none"/>
      <div className="relative page-shell w-full py-16 md:py-20"><div className="max-w-[710px]">
        <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-black/25 px-3 py-1.5 text-[10px] uppercase tracking-[.16em] text-zinc-300 backdrop-blur"><span className="w-1.5 h-1.5 rounded-full bg-red-500"/>Open model infrastructure</div>
        <h1 className="display-title mt-7">Models, ready<br/><span className="text-zinc-400">for AMD.</span></h1>
        <p className="lede mt-6 max-w-[600px]">Discover open models, understand what’s needed, and prepare with a clear record of every step.</p>
        <div className="flex flex-wrap gap-3 mt-8"><button onClick={()=>onNavigate('explorer')} className="btn-primary">Explore models <ArrowRight size={16}/></button><button onClick={()=>onNavigate('runs')} className="btn-secondary">View runs</button></div>
        <div className="mt-8 flex items-center gap-2 text-xs text-zinc-400"><span className={`status-dot ${computeConnected?'status-dot-online':computeUnknown?'status-dot-pending':'status-dot-error'}`}/>{computeTitle}</div>
      </div></div>
    </section>

    <div className="page-shell py-9 md:py-12 space-y-12">
      <section aria-labelledby="workspace-flow-title">
        <div className="flex items-end justify-between gap-4 mb-5"><div><div className="eyebrow mb-2">A clear path forward</div><h2 id="workspace-flow-title" className="text-2xl font-semibold tracking-[-.04em]">From discovery to preparation</h2></div></div>
        <div className="grid md:grid-cols-3 gap-3">
          <FlowStep number="01" icon={<Search size={18}/>} title="Discover" copy="Find a public model and inspect its architecture and files."/>
          <FlowStep number="02" icon={<Layers3 size={18}/>} title="Prepare" copy="Create configuration files, or separately consent to download model weights."/>
          <FlowStep number="03" icon={<Check size={18}/>} title="Review" copy="Follow honest run states. AMD execution is shown only when it actually happens."/>
        </div>
      </section>

      <section className="focus-panel p-5 md:p-7" aria-labelledby="compute-title">
        <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3"><div><div className="eyebrow mb-2">Compute access</div><h2 id="compute-title" className="text-xl font-semibold">{computeTitle}</h2></div>{computeConnected&&<span className="inline-flex items-center gap-2 rounded-full border border-emerald-500/20 bg-emerald-500/[.07] px-3 py-1.5 text-[11px] text-emerald-300"><Cpu size={13}/>Hardware detected</span>}</div>
        <p className="mt-4 text-sm text-zinc-400 max-w-4xl leading-relaxed">{computeSummary}</p>
        {computeUnknown&&<p role="status" className="text-xs text-zinc-500 mt-2">Connection status could not be fully confirmed.</p>}
        <details className="mt-5 border-t hairline pt-4">
          <summary className="text-xs text-zinc-500 cursor-pointer">Connection details{agents.length ? ` · ${agents.length}` : ''}</summary>
          <div className="mt-4">
            {isLoadingAgents?<div aria-label="Loading connection details" className="h-16 rounded-xl shimmer"/>:isAgentsError?<div role="status" className="rounded-xl border border-red-500/15 bg-red-500/[.04] p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3"><div><div className="text-sm font-medium">Connection details couldn’t be loaded</div><p className="text-xs text-zinc-500 mt-1">The connection state is unknown. This doesn’t mean there are no connections.</p></div><button className="btn-ghost shrink-0" onClick={onRetryAgents}><RotateCcw size={13}/>Retry</button></div>:agents.length===0?<p className="text-sm text-zinc-500">No external compute connections are registered. You can still inspect models and prepare configuration-only outputs.</p>:<><p className="text-xs text-zinc-500 mb-3">{agents.filter((agent)=>agent.status==='ONLINE'||agent.status==='BUSY').length} online or working · {agents.filter((agent)=>agent.status==='OFFLINE').length} offline · {agents.filter((agent)=>agent.status==='DEGRADED').length} interrupted</p><div className="grid md:grid-cols-2 gap-3">{agents.map(agent=><div key={agent.agent_id} className="rounded-xl border hairline p-4"><div className="flex justify-between gap-3"><div className="font-medium truncate">{agent.name}</div><span className={`shrink-0 text-xs ${agent.status==='ONLINE'||agent.status==='BUSY'?'text-emerald-400':agent.status==='DEGRADED'?'text-amber-300':'text-zinc-500'}`}>{AGENT_STATUS_COPY[agent.status]}</span></div><p className="text-sm text-zinc-500 mt-2">{agent.capabilities.rocm_detected&&agent.capabilities.amd_gpu_count?'AMD hardware detected':'Configuration preparation only'}</p><details className="mt-3"><summary className="text-xs text-zinc-600 cursor-pointer">Technical details</summary><p className="text-xs text-zinc-500 mt-2">{agent.hostname} · {agent.capabilities.gpu_names.join(', ')||'No AMD GPU detected'} · Last seen {new Date(agent.last_seen).toLocaleString()} · <span className="font-mono">{agent.status}</span><br/>Capabilities: {agent.capabilities.capabilities.join(', ')||'Not reported'}</p></details></div>)}</div></>}
          </div>
        </details>
      </section>

      <section aria-labelledby="recent-runs-title">
        <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-5"><div><div className="eyebrow mb-2">Workspace</div><h2 id="recent-runs-title" className="text-2xl font-semibold tracking-[-.04em]">Recent runs</h2></div><button onClick={()=>onNavigate('runs')} className="btn-ghost shrink-0">All runs <ArrowRight size={14}/></button></div>
        {isLoadingJobs?<div aria-label="Loading recent runs" className="grid md:grid-cols-2 xl:grid-cols-3 gap-3">{[0,1,2].map(i=><div key={i} className="h-36 rounded-2xl shimmer"/>)}</div>:isJobsError?<div className="focus-panel p-6"><h3 className="font-semibold">Recent runs couldn’t be loaded</h3><p className="text-sm text-zinc-500 mt-1">Your work is unchanged. Retry or open all runs.</p><div className="flex flex-wrap gap-2 mt-4"><button className="btn-secondary" onClick={onRetryJobs}><RotateCcw size={13}/>Retry</button><button className="btn-ghost" onClick={()=>onNavigate('runs')}>All runs <ArrowRight size={14}/></button></div></div>:jobs.length===0?<div className="empty-panel flex flex-col md:flex-row md:items-center justify-between gap-6"><div><div className="empty-icon"><Clock3 size={19}/></div><h3 className="text-lg font-semibold mt-4">No runs yet</h3><p className="text-sm text-zinc-400 mt-2 max-w-xl leading-relaxed">Model preparation and optimization studies will appear here, alongside future AMD execution runs.</p></div><button onClick={()=>onNavigate('explorer')} className="btn-secondary shrink-0">Start with a model <ArrowRight size={15}/></button></div>:<div className="grid md:grid-cols-2 xl:grid-cols-3 gap-3">{jobs.slice(0,6).map(job=><button key={job.job_id} onClick={()=>onSelectJob(job.job_id)} className="card-interactive p-5 text-left"><div className="flex justify-between gap-4"><span className="text-xs text-zinc-400">{JOB_TYPE_COPY[job.job_type]}</span><span className="text-xs text-zinc-500">{JOB_STATUS_COPY[job.status]}</span></div><div className="mt-5 font-semibold truncate">{job.model_id.split('/').pop()}</div><div className="text-xs text-zinc-600 mt-1 truncate">{job.model_id}</div><div className="mt-5 pt-4 border-t hairline flex items-center justify-between text-[11px] text-zinc-500"><span className="flex gap-1.5 items-center"><Clock3 size={12}/>{new Date(job.created_at).toLocaleDateString()}</span><span>{job.domain_status?DOMAIN_STATUS_COPY[job.domain_status].label:'Details'}</span></div></button>)}</div>}
      </section>
    </div>
  </div>;
};

const FlowStep = ({number,icon,title,copy}:{number:string;icon:React.ReactNode;title:string;copy:string}) => <article className="flow-card"><div className="flex items-center justify-between"><span className="flow-number">{number}</span><span className="flow-icon">{icon}</span></div><h3 className="mt-5 text-base font-semibold">{title}</h3><p className="mt-2 text-sm text-zinc-500 leading-relaxed">{copy}</p></article>;
