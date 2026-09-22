import type {
  ForgePlan,
  HealthResponse,
  JobCreatePayload,
  JobListResponse,
  JobResponse,
  JobResultResponse,
  JobStatus,
  JobType,
  ModelMetadata,
} from './types';

const API_BASE = '';

export class ApiError extends Error {
  status: number;
  data: any;

  constructor(status: number, message: string, data?: any) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  };

  try {
    const res = await fetch(url, { ...options, headers });
    if (!res.ok) {
      let errData: any = null;
      try {
        errData = await res.json();
      } catch {
        errData = { detail: await res.text() };
      }
      const message = errData?.detail || `HTTP Error ${res.status}: ${res.statusText}`;
      throw new ApiError(res.status, message, errData);
    }
    return (await res.json()) as T;
  } catch (err: any) {
    if (err instanceof ApiError) {
      throw err;
    }
    throw new ApiError(0, err.message || 'Network error communicating with ROCmHub backend');
  }
}

export async function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health');
}

export async function fetchModel(modelId: string, revision: string = 'main'): Promise<ModelMetadata> {
  const encodedId = encodeURIComponent(modelId).replace(/%2F/g, '/');
  return request<ModelMetadata>(`/api/v1/models/${encodedId}?revision=${encodeURIComponent(revision)}`);
}

export async function createForgePlan(payload: {
  model_id: string;
  revision?: string;
  precision?: string;
  target_gpu?: string;
  recipe?: string;
}): Promise<ForgePlan> {
  return request<ForgePlan>('/api/v1/forge/plan', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function createJob(payload: JobCreatePayload): Promise<JobResponse> {
  return request<JobResponse>('/api/v1/jobs', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function fetchJob(jobId: string): Promise<JobResponse> {
  return request<JobResponse>(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
}

export async function fetchJobs(params?: {
  limit?: number;
  offset?: number;
  status?: JobStatus;
  job_type?: JobType;
}): Promise<JobListResponse> {
  const query = new URLSearchParams();
  if (params?.limit !== undefined) query.set('limit', params.limit.toString());
  if (params?.offset !== undefined) query.set('offset', params.offset.toString());
  if (params?.status) query.set('status', params.status);
  if (params?.job_type) query.set('job_type', params.job_type);

  const qs = query.toString();
  return request<JobListResponse>(`/api/v1/jobs${qs ? `?${qs}` : ''}`);
}

export async function fetchJobResult(jobId: string): Promise<JobResultResponse> {
  return request<JobResultResponse>(`/api/v1/jobs/${encodeURIComponent(jobId)}/result`);
}

export async function cancelJob(jobId: string): Promise<{ job_id: string; cancelled: boolean; status: string }> {
  return request<{ job_id: string; cancelled: boolean; status: string }>(
    `/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`,
    { method: 'POST' }
  );
}
