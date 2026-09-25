export type JobType = 'FORGE_BUILD' | 'ENGINEER' | 'OPTIMIZATION' | 'PREPARE_MODEL_FOR_AMD';

export type JobStatus = 'QUEUED' | 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'CANCELLED';

export type DomainStatus = 'CONFIG_ONLY' | 'PREPARED' | 'EXECUTED' | 'NOT_MEASURED' | 'FAILED';

export interface GpuInfo {
  device_name: string;
  vendor: string;
  gfx_target: string | null;
  vram_total_mb: number | null;
  gpu_present: boolean;
}

export interface HostPlatform {
  os: string;
  arch: string;
  python_version: string;
  is_apple_silicon: boolean;
}

export interface HealthResponse {
  status: string;
  version: string;
  rocm_available: boolean;
  host_platform: HostPlatform;
  system: {
    os: string;
    python_version: string;
    rocm_version: string | null;
    torch_version: string | null;
    gpus_detected: number;
    gpus: GpuInfo[];
  };
  orchestrator: {
    queue_size: number;
    active_directory_locks: string[];
  };
  warnings: string[];
}

export interface CompatibilityReason {
  code: string;
  message: string;
  severity: string;
}

export interface ModelMetadata {
  model_id: string;
  requested_revision: string;
  commit_sha: string;
  architecture: string | null;
  parameter_count: number | null;
  context_length: number | null;
  weights_format: string | null;
  license: string | null;
  pipeline_tag: string | null;
  tags: string[];
  files_count: number;
  safetensors_metadata: Record<string, any> | null;
  rocm_available: boolean;
  host_gpus_detected: number;
  host_gpu_summary: GpuInfo[];
  compatibility: {
    verdict: string;
    reasons: CompatibilityReason[];
    warnings: string[];
  };
}

export interface BuildStepSpec {
  name: string;
  description: string;
  required: boolean;
}

export interface ForgePlan {
  plan_id: string;
  model_id: string;
  revision: string;
  target_gpu: string | null;
  precision: string;
  recipe_id: string;
  recipe_version: string;
  output_dir: string;
  estimated_disk_space_bytes: number;
  steps: BuildStepSpec[];
  compatibility_confirmed: boolean;
}

export interface JobResponse {
  job_id: string;
  job_type: JobType;
  model_id: string;
  revision: string | null;
  status: JobStatus;
  domain_status: DomainStatus | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  timeout_seconds: number;
  output_dir: string | null;
  error_message: string | null;
  error_code: string | null;
}

export interface JobListResponse {
  items: JobResponse[];
  total: number;
  limit: number;
  offset: number;
}

export interface JobEvent {
  event_id: number;
  job_id: string;
  sequence: number;
  timestamp: string;
  phase: string;
  status: string;
  message: string;
  error_code: string | null;
  details: Record<string, any> | null;
}

export interface JobResultResponse {
  job_id: string;
  job_type: JobType;
  job_status: JobStatus;
  domain_status: DomainStatus | null;
  output_dir: string | null;
  completed_at: string | null;
  result: Record<string, any> | null;
  error_message: string | null;
}

export interface JobCreatePayload {
  job_type: JobType;
  model_id: string;
  revision?: string;
  target_gpu?: string;
  precision?: string;
  recipe?: string;
  objective?: string;
  strategies?: string[];
  allow_full_weights?: boolean;
  materialization_mode?: 'METADATA_ONLY' | 'FULL_WEIGHTS';
  weights_consent?: boolean;
  expected_capabilities?: string[];
  cache_policy?: 'REUSE';
  timeout_seconds?: number;
  max_candidates?: number;
  max_attempts?: number;
  max_disk_gb?: number;
  target_gfx?: string;
  runtime?: string;
}

export interface AgentInfo {agent_id:string;name:string;hostname:string;status:'ONLINE'|'BUSY'|'OFFLINE'|'DEGRADED';last_seen:string;capabilities:{rocm_detected:boolean;hip_detected:boolean;amd_gpu_count:number;gpu_names:string[];capabilities:string[]}}
