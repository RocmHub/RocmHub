import type { AgentInfo, DomainStatus, JobStatus, JobType } from '../api/types';

export const JOB_STATUS_COPY: Record<JobStatus, string> = {
  QUEUED: 'Waiting to start',
  RUNNING: 'In progress',
  SUCCEEDED: 'Completed',
  FAILED: 'Could not complete',
  CANCELLED: 'Cancelled',
};

export const DOMAIN_STATUS_COPY: Record<DomainStatus, { label: string; explanation: string }> = {
  CONFIG_ONLY: {
    label: 'Configuration ready',
    explanation: 'Weights were not downloaded. AMD execution was not performed.',
  },
  PREPARED: {
    label: 'Model files prepared',
    explanation: 'Model files were downloaded and verified. AMD execution was not performed.',
  },
  EXECUTED: {
    label: 'Inference completed on AMD',
    explanation: 'Inference was executed on physical AMD ROCm hardware.',
  },
  NOT_MEASURED: {
    label: 'Performance not measured',
    explanation: 'No performance measurements were collected; no synthetic metrics are shown.',
  },
  FAILED: {
    label: 'Did not complete',
    explanation: 'The requested operation did not complete.',
  },
};

export const JOB_TYPE_COPY: Record<JobType, string> = {
  FORGE_BUILD: 'Build',
  ENGINEER: 'Guided recommendation',
  OPTIMIZATION: 'Optimization study',
  PREPARE_MODEL_FOR_AMD: 'Model preparation',
};

export const AGENT_STATUS_COPY: Record<AgentInfo['status'], string> = {
  ONLINE: 'Connected',
  BUSY: 'Working',
  OFFLINE: 'Not connected',
  DEGRADED: 'Connection interrupted',
};

export function agentSummary(agents: AgentInfo[]): string {
  const counts = { ONLINE: 0, BUSY: 0, OFFLINE: 0, DEGRADED: 0 };
  agents.forEach((agent) => { counts[agent.status] += 1; });
  return `${counts.ONLINE} connected · ${counts.BUSY} working · ${counts.OFFLINE} not connected · ${counts.DEGRADED} interrupted`;
}

export function tabForJobType(type: JobType): 'forge' | 'engineer' | 'optimization' | 'explorer' {
  switch (type) {
    case 'FORGE_BUILD': return 'forge';
    case 'ENGINEER': return 'engineer';
    case 'OPTIMIZATION': return 'optimization';
    case 'PREPARE_MODEL_FOR_AMD': return 'explorer';
  }
}

export function domainStatusLabel(status: DomainStatus | string | null | undefined): string | null {
  if (!status) return null;
  return status in DOMAIN_STATUS_COPY
    ? DOMAIN_STATUS_COPY[status as DomainStatus].label
    : status.replace(/_/g, ' ').toLowerCase().replace(/^\w/, (letter) => letter.toUpperCase());
}

export function resultLabel(value: string | null | undefined): string | null {
  if (!value) return null;
  return domainStatusLabel(value) ?? value.replace(/_/g, ' ').toLowerCase().replace(/^\w/, (letter) => letter.toUpperCase());
}
