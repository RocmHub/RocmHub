import React from 'react';
import { ArrowRight, Clock3, RotateCcw } from 'lucide-react';
import type { JobListResponse } from '../../api/types';
import { useQuery } from '@tanstack/react-query';
import { fetchJob } from '../../api/client';
import type { NavTab } from '../layout/Sidebar';
import { DOMAIN_STATUS_COPY, JOB_STATUS_COPY, JOB_TYPE_COPY, tabForJobType } from '../../ui/presentation';

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
  const selectedFromList = selectedJobId ? jobs.find((job) => job.job_id === selectedJobId) : null;
  const selectedQuery = useQuery({ queryKey: ['job', selectedJobId], queryFn: () => fetchJob(selectedJobId as string), enabled: Boolean(selectedJobId && !selectedFromList), retry: false });
  const selectedJob = selectedFromList ?? selectedQuery.data ?? null;

  return <div className="page-fade min-h-full page-shell py-10 md:py-14">
    <header className="max-w-3xl">
      <div className="eyebrow mb-3">Workspace</div>
      <h1 className="page-title">Runs</h1>
      <p className="lede mt-4">Follow preparation, builds, recommendations, and optimization studies from one place.</p>
    </header>

    {selectedJobId && <section className="focus-panel mt-8 p-5 md:p-7" aria-live="polite">
      {selectedJob ? <>
        <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-5">
          <div className="min-w-0">
            <div className="eyebrow">Selected run · {JOB_TYPE_COPY[selectedJob.job_type]}</div>
            <h2 className="text-xl font-semibold mt-2 break-words">{selectedJob.model_id}</h2>
            <p className="text-xs text-zinc-500 mt-2">{JOB_STATUS_COPY[selectedJob.status]} · {new Date(selectedJob.created_at).toLocaleString()}</p>
            <details className="mt-2"><summary className="text-xs text-zinc-600 cursor-pointer">Technical details</summary><p className="text-xs text-zinc-600 font-mono mt-2 break-all">Run ID: {selectedJob.job_id}{selectedJob.domain_status ? ` · ${selectedJob.domain_status}` : ''}</p></details>
          </div>
          <button className="btn-primary shrink-0" onClick={() => onOpenJob(selectedJob.job_id, tabForJobType(selectedJob.job_type))}>Open result <ArrowRight size={15}/></button>
        </div>
        {selectedJob.domain_status && <div className="mt-5 rounded-xl bg-white/[.025] p-4">
          <div className="text-sm font-medium">{DOMAIN_STATUS_COPY[selectedJob.domain_status].label}</div>
          <p className="text-sm text-zinc-500 mt-1">{DOMAIN_STATUS_COPY[selectedJob.domain_status].explanation}</p>
        </div>}
      </> : <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div><h2 className="font-semibold">{selectedQuery.isLoading?'Loading run details':selectedQuery.isError?'Run details unavailable':'Run details unavailable'}</h2><p className="text-sm text-zinc-500 mt-1">{selectedQuery.isLoading?'Retrieving this run from the workspace.':selectedQuery.isError?'This run could not be loaded. Retry the request or return to all runs.':'This run is not in the recent list.'}</p></div>
        <button className="btn-secondary" onClick={() => selectedJobId && selectedQuery.refetch()} disabled={!selectedJobId || selectedQuery.isLoading}><RotateCcw size={14}/>Retry run</button>
      </div>}
    </section>}

    <section className="mt-9" aria-label="Recent runs">
      <div className="flex items-end justify-between gap-4 mb-4"><div><h2 className="text-lg font-semibold">Recent runs</h2><p className="text-sm text-zinc-500 mt-1">The latest operations in this workspace.</p></div>{jobsList && <span className="text-xs text-zinc-600">{jobsList.total} total</span>}</div>
      {isLoading ? <div aria-label="Loading runs" className="space-y-3">{[0,1,2].map((item) => <div key={item} className="h-24 rounded-2xl shimmer" />)}</div>
        : isError ? <div className="focus-panel p-6"><h3 className="font-semibold">Runs couldn’t be loaded</h3><p className="text-sm text-zinc-500 mt-1">Your work is unchanged. Try loading the list again.</p><button className="btn-secondary mt-4" onClick={onRetry}><RotateCcw size={14}/>Retry</button></div>
        : jobs.length === 0 ? <div className="focus-panel p-7 md:p-9"><h3 className="text-lg font-semibold">No runs yet</h3><p className="text-sm text-zinc-500 mt-2">Start by finding a model. Your preparation and results will appear here.</p><button className="btn-primary mt-5" onClick={() => onOpenJob('', 'explorer')}>Find a model <ArrowRight size={15}/></button></div>
        : <div className="space-y-2">{jobs.map((job) => <article key={job.job_id} className="card-interactive p-4 md:p-5">
          <button onClick={() => onOpenJob(job.job_id, tabForJobType(job.job_type))} className="w-full text-left flex flex-col sm:flex-row sm:items-center gap-4">
            <div className="min-w-0 flex-1"><div className="flex flex-wrap items-center gap-x-3 gap-y-1"><span className="text-xs text-zinc-400">{JOB_TYPE_COPY[job.job_type]}</span><span className="text-xs text-zinc-500">{JOB_STATUS_COPY[job.status]}</span></div><div className="font-medium mt-2 truncate">{job.model_id}</div></div>
            <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-500 sm:justify-end">{job.domain_status && <span>{DOMAIN_STATUS_COPY[job.domain_status].label}</span>}<span className="flex items-center gap-1.5"><Clock3 size={12}/>{new Date(job.created_at).toLocaleDateString()}</span><ArrowRight size={14} className="text-zinc-600"/></div>
          </button>
          <details className="mt-2"><summary className="text-xs text-zinc-600 cursor-pointer">Technical details</summary><p className="text-xs text-zinc-600 font-mono mt-1 break-all">Run ID: {job.job_id}{job.domain_status ? ` · ${job.domain_status}` : ''}</p></details>
        </article>)}</div>}
    </section>
  </div>;
};
