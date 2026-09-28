import React from 'react';
import type { DomainStatus } from '../../api/types';
import { DOMAIN_STATUS_COPY, domainStatusLabel } from '../../ui/presentation';

interface DomainStatusTagProps {
  status: DomainStatus | string | null;
  className?: string;
}

export const DomainStatusTag: React.FC<DomainStatusTagProps> = ({ status, className = '' }) => {
  if (!status) return null;

  const known = status in DOMAIN_STATUS_COPY ? status as DomainStatus : null;
  const label = domainStatusLabel(status) || status;
  const tone = status === 'CONFIG_ONLY' ? 'bg-amber-500/10 text-amber-300 border-amber-500/20'
    : status === 'PREPARED' ? 'bg-cyan-500/10 text-cyan-300 border-cyan-500/20'
    : status === 'EXECUTED' ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/20'
    : status === 'NOT_MEASURED' ? 'bg-purple-500/10 text-purple-300 border-purple-500/20'
    : status === 'FAILED' ? 'bg-red-500/10 text-red-300 border-red-500/20'
    : 'bg-zinc-800 text-zinc-400 border-transparent';
  return <span aria-label={label} title={known ? DOMAIN_STATUS_COPY[known].explanation : undefined} className={`inline-flex items-center px-2 py-1 rounded border text-xs font-medium ${tone} ${className}`}>
    {label}
  </span>;
};
