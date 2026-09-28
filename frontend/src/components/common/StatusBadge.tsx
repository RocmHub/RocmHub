import React from 'react';
import type { JobStatus } from '../../api/types';
import { JOB_STATUS_COPY } from '../../ui/presentation';

interface StatusBadgeProps {
  status: JobStatus;
  className?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, className = '' }) => {
  const tone = status === 'QUEUED' ? 'bg-amber-500/10 text-amber-400 border-amber-500/20'
    : status === 'RUNNING' ? 'bg-blue-500/10 text-blue-400 border-blue-500/20'
    : status === 'SUCCEEDED' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
    : status === 'FAILED' ? 'bg-red-500/10 text-red-400 border-red-500/20'
    : 'bg-zinc-500/10 text-zinc-400 border-zinc-500/20';
  return <span aria-label={JOB_STATUS_COPY[status]} title={JOB_STATUS_COPY[status]} className={`inline-flex items-center px-2 py-1 rounded border text-xs font-medium ${tone} ${className}`}>
    <span className={`w-1.5 h-1.5 rounded-full mr-1.5 ${status==='QUEUED'?'bg-amber-400 animate-pulse':status==='RUNNING'?'bg-blue-400 animate-pulse':status==='SUCCEEDED'?'bg-emerald-400':status==='FAILED'?'bg-red-400':'bg-zinc-400'}`} />
    {JOB_STATUS_COPY[status]}
  </span>;
};
