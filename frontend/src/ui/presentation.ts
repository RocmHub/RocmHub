import type { AgentInfo, DomainStatus, HealthResponse, JobResponse, JobStatus, JobType } from '../api/types';

export const JOB_STATUS_COPY: Record<JobStatus, string> = {
  QUEUED: 'Waiting',
  RUNNING: 'Running',
  SUCCEEDED: 'Completed',
  FAILED: 'Failed',
  CANCELLED: 'Cancelled',
};

/** Turns worker events into plain-language progress without changing their backend meaning. */
export function jobProgressCopy(status: JobStatus | string, phase?: string | null): string {
  if (status === 'FAILED' || status === 'CANCELLED' || status === 'SUCCEEDED') return JOB_STATUS_COPY[status];
  const normalized = (phase || '').toUpperCase();
  if (status === 'QUEUED' || normalized.includes('QUEUE') || normalized.includes('CLAIM')) return 'Waiting';
  if (normalized.includes('DOWNLOAD') || normalized.includes('MATERIALIZATION')) return 'Downloading';
  if (normalized.includes('VERIF')) return 'Verifying';
  if (normalized.includes('PREPAR') || normalized.includes('BUILD') || normalized.includes('FINAL')) return 'Preparing';
  return 'Running';
}

export const DOMAIN_STATUS_COPY: Record<DomainStatus, { label: string; explanation: string }> = {
  CONFIG_ONLY: {
    label: 'Configuration prepared',
    explanation: 'Weights were not downloaded. AMD execution was not performed.',
  },
  PREPARED: {
    label: 'Model prepared',
    explanation: 'Model files are available and verified on connected compute. AMD execution was not performed.',
  },
  EXECUTED: {
    label: 'Ran on AMD',
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
  ONLINE: 'Online',
  BUSY: 'Working',
};

export function agentSummary(agents: AgentInfo[]): string {
  const counts = { ONLINE: 0, BUSY: 0 };
  agents.forEach((agent) => { counts[agent.status] += 1; });
  return `${counts.ONLINE} online · ${counts.BUSY} working`;
}

export function hasConnectedAmdCompute(health: HealthResponse | null, agents: AgentInfo[]): boolean {
  const serviceCompute = health?.status === 'healthy' && health.rocm_available;
  const remoteCompute = agents.some((agent) =>
    (agent.status === 'ONLINE' || agent.status === 'BUSY') && agent.rocm_detected && agent.has_amd_gpu
  );
  return Boolean(serviceCompute || remoteCompute);
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

export function activityOutcomeCopy(
  job: Pick<JobResponse, 'status' | 'domain_status' | 'error_message' | 'job_type'>,
): { title: string; description: string } {
  if (job.domain_status && job.domain_status in DOMAIN_STATUS_COPY) {
    const outcome = DOMAIN_STATUS_COPY[job.domain_status];
    return { title: outcome.label, description: outcome.explanation };
  }
  if (job.status === 'QUEUED') {
    return { title: 'Waiting for compute', description: 'This work will start when compatible compute is available.' };
  }
  if (job.status === 'RUNNING') {
    return { title: job.job_type === 'PREPARE_MODEL_FOR_AMD' ? 'Preparing model' : 'In progress', description: 'Work is underway. No completion time is estimated.' };
  }
  if (job.status === 'FAILED') {
    return { title: 'Needs attention', description: 'This work did not complete. Open the result for technical details or retry.' };
  }
  if (job.status === 'CANCELLED') {
    return { title: 'Cancelled', description: 'This work stopped before completion. Its result can be reviewed in Activity.' };
  }
  if (job.job_type === 'OPTIMIZATION') {
    return { title: 'Comparison prepared', description: 'Performance measurements have not been collected.' };
  }
  return { title: 'Completed', description: 'The work finished. Open the result to see what was produced.' };
}

export function materializationCopy(cacheStatus?: string | null): { title: string; description: string } {
  if (cacheStatus === 'HIT_VERIFIED') {
    return { title: 'Model files already available', description: 'Verified from the connected compute cache. No repeat download was needed.' };
  }
  if (cacheStatus === 'MISS_DOWNLOADED') {
    return { title: 'Model files downloaded and verified', description: 'Stored on connected compute. AMD execution was not performed.' };
  }
  return { title: 'Verified model files are ready', description: 'Available on connected compute. AMD execution was not performed.' };
}
