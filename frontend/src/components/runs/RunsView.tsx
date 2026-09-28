import React, { useState } from 'react';
import { ArrowRight, Clock3, RotateCcw } from 'lucide-react';
import type { JobListResponse } from '../../api/types';
import { useQuery } from '@tanstack/react-query';
import { fetchJob, fetchJobResult } from '../../api/client';
import type { NavTab } from '../layout/Sidebar';
import { activityOutcomeCopy, JOB_STATUS_COPY, JOB_TYPE_COPY, tabForJobType } from '../../ui/presentation';

interface Props {
  jobsList: JobListResponse | null;
  isLoading: boolean;
  isError?: boolean;
  selectedJobId?: string | null;
  onRetry?: () => void;
  onOpenJob: (jobId: string, tab: NavTab) => void;
}

export const RunsView: React.FC<Props> = ({ jobsList, isLoading, isError = false, selectedJobId, onRetry, onOpenJob }) => {
  const jobs = jobsList?.items ?? [];
  const [filter, setFilter] = useState<'all' | 'active' | 'complete' | 'attention'>('all');
  const visibleJobs = jobs.filter((job) => filter === 'all' || (filter === 'active' ? job.status === 'QUEUED' || job.status === 'RUNNING' : filter === 'complete' ? job.status === 'SUCCEEDED' || job.status === 'CANCELLED' : job.status === 'FAILED'));
  const selectedFromList = selectedJobId ? jobs.find((job) => job.job_id === selectedJobId) : null;
  const selectedQuery = useQuery({ queryKey: ['job', selectedJobId], queryFn: () => fetchJob(selectedJobId as string), enabled: Boolean(selectedJobId && !selectedFromList), retry: false });
  const selectedJob = selectedFromList ?? selectedQuery.data ?? null;
  const resultQuery = useQuery({ queryKey: ['job-result', selectedJobId], queryFn: () => fetchJobResult(selectedJobId as string), enabled: Boolean(selectedJobId), retry: false });
  const selectedResult = resultQuery.data;
  const selectedOutcome = selectedJob ? activityOutcomeCopy({
    ...selectedJob,
    status: selectedResult?.job_status ?? selectedJob.status,
    domain_status: selectedResult?.domain_status ?? selectedJob.domain_status,
    error_message: selectedResult?.error_message ?? selectedJob.error_message,
  }) : null;

  return <div className="page-fade min-h-full page-shell py-10 md:py-14">
    <header className="max-w-3xl">
      <div className="eyebrow mb-3">Workspace activity</div>
      <h1 className="page-title">Activity</h1>
      <p className="lede mt-4">Preparation and comparison work, with the evidence and outcome for each run.</p>
    </header>

    {selectedJobId && <section className="focus-panel mt-8 p-5 md:p-7" aria-live="polite">
      {selectedJob ? <>
        <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-5">
          <div className="min-w-0">
            <div className="eyebrow">Activity · {JOB_TYPE_COPY[selectedJob.job_type]}</div>
            <h2 className="text-xl font-semibold mt-2 break-words">{selectedJob.model_id}</h2>
            <p className="text-sm text-content-secondary mt-2">{selectedOutcome?.title} · {new Date(selectedJob.created_at).toLocaleString()}</p>
            <details className="mt-2"><summary className="text-xs text-zinc-600 cursor-pointer">Technical details</summary><p className="text-xs text-zinc-600 font-mono mt-2 break-all">Run ID: {selectedJob.job_id}{selectedJob.domain_status ? ` · ${selectedJob.domain_status}` : ''}</p></details>
          </div>
          <button className="btn-primary shrink-0" onClick={() => onOpenJob(selectedJob.job_id, tabForJobType(selectedJob.job_type))}>Open result <ArrowRight size={15}/></button>
        </div>
        <div className="run-detail-layout">
          <div><div className="eyebrow">Run timeline</div><ol className="execution-timeline">
            <li className="timeline-done"><span>✓</span><div><strong>Request created</strong><small>{new Date(selectedJob.created_at).toLocaleString()}</small></div></li>
            <li className={selectedJob.status === 'RUNNING' || selectedJob.status === 'QUEUED' ? 'timeline-current' : selectedJob.status === 'SUCCEEDED' ? 'timeline-done' : 'timeline-stopped'}><span>{selectedJob.status === 'SUCCEEDED' ? '✓' : selectedJob.status === 'FAILED' ? '!' : selectedJob.status === 'CANCELLED' ? '−' : '·'}</span><div><strong>{selectedJob.status === 'QUEUED' ? 'Waiting for a worker' : selectedJob.status === 'RUNNING' ? 'Work in progress' : selectedJob.status === 'SUCCEEDED' ? 'Work completed' : JOB_STATUS_COPY[selectedJob.status]}</strong><small>{selectedJob.completed_at ? new Date(selectedJob.completed_at).toLocaleString() : selectedJob.started_at ? `Started ${new Date(selectedJob.started_at).toLocaleString()}` : 'Awaiting execution'}</small></div></li>
            <li className={selectedResult?.domain_status === 'EXECUTED' ? 'timeline-done' : 'timeline-neutral'}><span>{selectedResult?.domain_status === 'EXECUTED' ? '✓' : '—'}</span><div><strong>{selectedResult?.domain_status === 'EXECUTED' ? 'Ran on AMD' : 'AMD execution not performed'}</strong><small>{selectedResult?.domain_status === 'EXECUTED' ? 'Execution evidence recorded' : 'Preparation is not execution'}</small></div></li>
          </ol></div>
          <div className="run-result"><div className="eyebrow">What happened</div><h3>{selectedOutcome?.title}</h3><p>{selectedOutcome?.description}</p>{resultQuery.isError && <div className="result-recovery" role="status"><p>Result details could not be loaded. This connection issue does not mean the work failed.</p><button className="btn-secondary" onClick={() => { void resultQuery.refetch(); }}>Retry result details <RotateCcw size={14}/></button></div>}{resultQuery.isLoading && !selectedResult && <p role="status">Loading the recorded result…</p>}{selectedResult?.result && <details className="technical-details"><summary>Technical evidence</summary><pre>{JSON.stringify(selectedResult.result, null, 2)}</pre></details>}{selectedJob.output_dir && <details className="technical-details"><summary>Output location</summary><code>{selectedJob.output_dir}</code></details>}</div>
        </div>
      </> : <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div><h2 className="font-semibold">{selectedQuery.isLoading?'Loading run details':selectedQuery.isError?'Run details unavailable':'Run details unavailable'}</h2><p className="text-sm text-zinc-500 mt-1">{selectedQuery.isLoading?'Retrieving this run from the workspace.':selectedQuery.isError?'This run could not be loaded. Retry the request or return to all runs.':'This run is not in the recent list.'}</p></div>
        <button className="btn-secondary" onClick={() => selectedJobId && selectedQuery.refetch()} disabled={!selectedJobId || selectedQuery.isLoading}><RotateCcw size={14}/>Retry run</button>
      </div>}
    </section>}

    <section className="mt-9" aria-label="Recent runs">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-4"><div><div className="eyebrow mb-2">All work</div><h2 className="text-xl font-semibold">Recent activity</h2><p className="text-sm text-zinc-500 mt-1">The latest work in this workspace.</p></div>{jobsList && jobsList.total > 0 && <span className="text-xs text-zinc-600">{jobsList.total} runs</span>}</div>
      {jobsList && jobsList.total > jobs.length && <p className="text-xs text-zinc-600 -mt-2 mb-4">Showing the latest {jobs.length} of {jobsList.total} runs. Filters apply to this list.</p>}
      {!isLoading && !isError && jobs.length > 0 && <div className="run-filters" role="group" aria-label="Filter activity">{([['all','All'],['active','In progress'],['complete','Completed'],['attention','Needs attention']] as const).map(([id,label])=><button key={id} aria-pressed={filter===id} onClick={()=>setFilter(id)} className={`run-filter ${filter===id?'run-filter-active':''}`}>{label}<span>{jobs.filter((job)=>id==='all'?true:id==='active'?job.status==='QUEUED'||job.status==='RUNNING':id==='complete'?job.status==='SUCCEEDED'||job.status==='CANCELLED':job.status==='FAILED').length}</span></button>)}</div>}
      {isLoading ? <div aria-label="Loading runs" className="space-y-3">{[0,1,2].map((item) => <div key={item} className="h-24 rounded-2xl shimmer" />)}</div>
        : isError ? <div className="focus-panel p-6"><h3 className="font-semibold">Runs couldn’t be loaded</h3><p className="text-sm text-zinc-500 mt-1">Your work is unchanged. Try loading the list again.</p><button className="btn-secondary mt-4" onClick={onRetry}><RotateCcw size={14}/>Retry</button></div>
        : jobs.length === 0 ? <div className="empty-panel"><div className="empty-icon"><Clock3 size={19}/></div><h3 className="text-lg font-semibold mt-4">No runs yet</h3><p className="text-sm text-zinc-400 mt-2">Model preparation and optimization studies will appear here, followed by future AMD execution runs.</p><button className="btn-secondary mt-5" onClick={() => onOpenJob('', 'explorer')}>Explore models <ArrowRight size={14}/></button></div>
        : visibleJobs.length === 0 ? <div className="empty-panel"><h3 className="font-semibold">No {filter === 'active' ? 'work in progress' : filter === 'complete' ? 'completed runs' : 'runs need attention'}</h3><p className="text-sm text-zinc-500 mt-2">Choose another filter to see the rest of your workspace activity.</p></div>
        : <div className="space-y-0 activity-table">{visibleJobs.map((job) => <article key={job.job_id} className="activity-table-row">
          <button onClick={() => onOpenJob(job.job_id, 'runs')} className="w-full text-left flex flex-col sm:flex-row sm:items-center gap-4">
            <div className="min-w-0 flex-1"><div className="flex flex-wrap items-center gap-x-3 gap-y-1"><span className="text-xs text-zinc-400">{JOB_TYPE_COPY[job.job_type]}</span><span className="text-xs text-content-secondary">{activityOutcomeCopy(job).title}</span></div><div className="font-medium mt-2 truncate">{job.model_id}</div><p className="text-sm text-zinc-500 mt-1">{activityOutcomeCopy(job).description}</p></div>
            <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-500 sm:justify-end"><span className="flex items-center gap-1.5"><Clock3 size={12}/>{new Date(job.created_at).toLocaleDateString()}</span><ArrowRight size={14} className="text-zinc-600"/></div>
          </button>
          <details className="technical-details"><summary>Technical details</summary><p className="font-mono break-all">Run ID: {job.job_id}{job.domain_status ? ` · ${job.domain_status}` : ''}</p></details>
        </article>)}</div>}
    </section>
  </div>;
};
