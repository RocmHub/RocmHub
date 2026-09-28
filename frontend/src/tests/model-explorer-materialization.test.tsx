import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { ToastProvider } from '../components/common/Toast';
import { ModelExplorerView } from '../components/explorer/ModelExplorerView';

const api = vi.hoisted(() => ({
  fetchModel: vi.fn(),
  createJob: vi.fn(),
  fetchJob: vi.fn(),
  fetchJobResult: vi.fn(),
  cancelJob: vi.fn(),
  subscribeToJobEvents: vi.fn(() => () => undefined),
}));

vi.mock('../api/client', () => api);
vi.mock('../api/sse', () => ({ subscribeToJobEvents: api.subscribeToJobEvents }));

const metadata = {
  model_id: 'Qwen/Qwen2.5-0.5B-Instruct',
  requested_revision: 'main',
  commit_sha: '7ae557604adf67be50417f59c2c2f167def9a775',
  architecture: 'Qwen2ForCausalLM',
  parameter_count: 500_000_000,
  context_length: 32768,
  weights_format: 'safetensors',
  license: 'apache-2.0',
  pipeline_tag: 'text-generation',
  tags: [],
  files_count: 7,
  safetensors_metadata: null,
  rocm_available: false,
  host_gpus_detected: 0,
  host_gpu_summary: [],
  compatibility: { verdict: 'READY', reasons: [], warnings: [] },
} as any;

function renderView() {
  return render(
    <ToastProvider>
      <ModelExplorerView onSelectModelForForge={vi.fn()} onNavigate={vi.fn()} />
    </ToastProvider>
  );
}

async function inspectModel() {
  fireEvent.submit(screen.getByLabelText('Model ID').closest('form')!);
  await screen.findByText('Qwen/Qwen2.5-0.5B-Instruct');
}

describe('Model Explorer materialization consent', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    sessionStorage.clear();
    api.fetchModel.mockResolvedValue(metadata);
    api.createJob.mockResolvedValue({ job_id: 'job_prepare_01', status: 'QUEUED' });
    api.fetchJob.mockResolvedValue({ job_id: 'job_prepare_01', model_id: metadata.model_id, status: 'RUNNING' });
  });

  it('queues CONFIG_ONLY without weight consent by default', async () => {
    renderView();
    await inspectModel();
    fireEvent.click(screen.getByRole('button', { name: /Prepare setup/ }));
    await waitFor(() => expect(api.createJob).toHaveBeenCalledTimes(1));
    expect(api.createJob.mock.calls[0][0]).toMatchObject({
      materialization_mode: 'METADATA_ONLY',
      weights_consent: false,
      revision: metadata.commit_sha,
    });
  });

  it('opens a model workspace with files and AMD compatibility clearly separated', async () => {
    renderView();
    await inspectModel();
    expect(screen.getByRole('navigation', { name: 'Model workspace' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Overview' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Files' }));
    expect(screen.getByText('Repository contents')).toBeInTheDocument();
    expect(screen.getByText(/Inspection does not download or materialize model weights/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Compatibility' }));
    expect(screen.getByText('Not tested on AMD hardware yet.')).toBeInTheDocument();
  });

  it('requires a separate checked confirmation before FULL_WEIGHTS job creation', async () => {
    renderView();
    await inspectModel();
    fireEvent.click(screen.getByRole('button', { name: /Download & Prepare/ }));
    expect(await screen.findByRole('heading', { name: 'Download real model files?' })).toBeInTheDocument();
    expect(api.createJob).not.toHaveBeenCalled();
    const confirm = screen.getByRole('button', { name: /Confirm · Download & Prepare/ });
    expect(confirm).toBeDisabled();
    fireEvent.click(screen.getByRole('checkbox'));
    fireEvent.click(confirm);
    await waitFor(() => expect(api.createJob).toHaveBeenCalledTimes(1));
    expect(api.createJob.mock.calls[0][0]).toMatchObject({
      materialization_mode: 'FULL_WEIGHTS',
      weights_consent: true,
      revision: metadata.commit_sha,
      cache_policy: 'REUSE',
    });
  });

  it('retries a degraded observer for the same job instead of creating a duplicate', async () => {
    renderView();
    await inspectModel();
    fireEvent.click(screen.getByRole('button', { name: /Prepare setup/ }));
    await waitFor(() => expect(api.subscribeToJobEvents).toHaveBeenCalledTimes(1));

    const calls = api.subscribeToJobEvents.mock.calls as unknown as [string, { onError: () => void }][];
    const observer = calls[0][1];
    act(() => observer.onError());
    fireEvent.click(await screen.findByRole('button', { name: 'Retry' }));

    await waitFor(() => expect(api.subscribeToJobEvents).toHaveBeenCalledTimes(2));
    expect(api.createJob).toHaveBeenCalledTimes(1);
    expect(calls[1][0]).toBe('job_prepare_01');
  });

  it('resumes the stored job after reload without creating another job', async () => {
    sessionStorage.setItem('rocmhub.model-explorer.active-job', 'job_prepare_01');
    renderView();

    await waitFor(() => expect(api.fetchJob).toHaveBeenCalledWith('job_prepare_01'));
    expect(api.createJob).not.toHaveBeenCalled();
    expect(api.subscribeToJobEvents).toHaveBeenCalledTimes(1);
  });
});
