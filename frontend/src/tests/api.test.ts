import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  fetchHealth,
  fetchModel,
  createForgePlan,
  fetchJobs,
} from '../api/client';

describe('ROCmHub API Client', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('fetchHealth returns health payload', async () => {
    const mockHealth = {
      status: 'healthy',
      version: '0.1.0',
      rocm_available: false,
      host_platform: { os: 'darwin', arch: 'arm64', python_version: '3.9.6', is_apple_silicon: true },
      system: { os: 'darwin', python_version: '3.9.6', rocm_version: null, torch_version: '2.2.0', gpus_detected: 0, gpus: [] },
      orchestrator: { queue_size: 0, active_directory_locks: [] },
      warnings: ['No AMD ROCm GPU detected'],
    };

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockHealth,
    }));

    const result = await fetchHealth();
    expect(result.status).toBe('healthy');
    expect(result.rocm_available).toBe(false);
  });

  it('fetchModel queries /api/v1/models with model_id', async () => {
    const mockModel = {
      model_id: 'Qwen/Qwen2.5-0.5B-Instruct',
      commit_sha: '7ae557604adf67be50417f59c2c2f167def9a775',
      architecture: 'qwen2',
      parameter_count: 494032768,
      compatibility: { verdict: 'NO_ACCELERATOR', reasons: [] },
    };

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockModel,
    });
    vi.stubGlobal('fetch', fetchMock);

    const result = await fetchModel('Qwen/Qwen2.5-0.5B-Instruct', 'main');
    expect(result.model_id).toBe('Qwen/Qwen2.5-0.5B-Instruct');
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/models/Qwen/Qwen2.5-0.5B-Instruct?revision=main',
      expect.objectContaining({ headers: expect.any(Object) })
    );
  });

  it('createForgePlan posts to /api/v1/forge/plan', async () => {
    const mockPlan = {
      plan_id: 'plan_123',
      model_id: 'Qwen/Qwen2.5-0.5B-Instruct',
      revision: '7ae557604adf67be50417f59c2c2f167def9a775',
      precision: 'fp16',
      recipe_id: 'pytorch_transformers_hip',
      steps: [],
    };

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockPlan,
    });
    vi.stubGlobal('fetch', fetchMock);

    const plan = await createForgePlan({
      model_id: 'Qwen/Qwen2.5-0.5B-Instruct',
      precision: 'fp16',
    });
    expect(plan.plan_id).toBe('plan_123');
    expect(plan.recipe_id).toBe('pytorch_transformers_hip');
  });

  it('fetchJobs formats pagination and filter query params', async () => {
    const mockList = {
      items: [],
      total: 0,
      limit: 10,
      offset: 0,
    };

    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockList,
    });
    vi.stubGlobal('fetch', fetchMock);

    await fetchJobs({ limit: 10, offset: 0, status: 'SUCCEEDED', job_type: 'FORGE_BUILD' });
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/jobs?limit=10&offset=0&status=SUCCEEDED&job_type=FORGE_BUILD',
      expect.any(Object)
    );
  });

  it('throws ApiError with detail message on HTTP failure', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      statusText: 'Not Found',
      json: async () => ({ detail: "Model 'invalid' not found" }),
    }));

    await expect(fetchModel('invalid')).rejects.toThrow("Model 'invalid' not found");
  });
});
