import React from 'react';
import type { DomainStatus } from '../../api/types';

interface DomainStatusTagProps {
  status: DomainStatus | string | null;
  className?: string;
}

export const DomainStatusTag: React.FC<DomainStatusTagProps> = ({ status, className = '' }) => {
  if (!status) return null;

  switch (status) {
    case 'CONFIG_ONLY':
      return (
        <span
          title="Build recipe and configurations generated. Weights not downloaded; no AMD GPU execution performed."
          className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-medium bg-amber-500/10 text-amber-300 border border-amber-500/20 cursor-help ${className}`}
        >
          CONFIG_ONLY
        </span>
      );
    case 'PREPARED':
      return (
        <span
          title="Full model weights and configuration verified in workspace; ready for runtime execution."
          className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-medium bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 cursor-help ${className}`}
        >
          PREPARED
        </span>
      );
    case 'EXECUTED':
      return (
        <span
          title="Actual model inference executed on physical AMD ROCm hardware."
          className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-medium bg-emerald-500/10 text-emerald-300 border border-emerald-500/20 cursor-help ${className}`}
        >
          EXECUTED
        </span>
      );
    case 'NOT_MEASURED':
      return (
        <span
          title="Performance benchmarks were omitted or run on non-AMD host. Zero synthetic metrics generated."
          className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-medium bg-purple-500/10 text-purple-300 border border-purple-500/20 cursor-help ${className}`}
        >
          NOT_MEASURED
        </span>
      );
    default:
      return (
        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-medium bg-zinc-800 text-zinc-400 ${className}`}>
          {status}
        </span>
      );
  }
};
