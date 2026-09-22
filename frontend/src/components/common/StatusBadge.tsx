import React from 'react';
import type { JobStatus } from '../../api/types';

interface StatusBadgeProps {
  status: JobStatus;
  className?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, className = '' }) => {
  switch (status) {
    case 'QUEUED':
      return (
        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20 ${className}`}>
          <span className="w-1.5 h-1.5 rounded-full bg-amber-400 mr-1.5 animate-pulse" />
          QUEUED
        </span>
      );
    case 'RUNNING':
      return (
        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20 ${className}`}>
          <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mr-1.5 animate-spin" />
          RUNNING
        </span>
      );
    case 'SUCCEEDED':
      return (
        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 ${className}`}>
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5" />
          SUCCEEDED
        </span>
      );
    case 'FAILED':
      return (
        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-medium bg-red-500/10 text-red-400 border border-red-500/20 ${className}`}>
          <span className="w-1.5 h-1.5 rounded-full bg-red-400 mr-1.5" />
          FAILED
        </span>
      );
    case 'CANCELLED':
      return (
        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-medium bg-zinc-500/10 text-zinc-400 border border-zinc-500/20 ${className}`}>
          <span className="w-1.5 h-1.5 rounded-full bg-zinc-400 mr-1.5" />
          CANCELLED
        </span>
      );
    default:
      return (
        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-medium bg-zinc-800 text-zinc-300 ${className}`}>
          {status}
        </span>
      );
  }
};
