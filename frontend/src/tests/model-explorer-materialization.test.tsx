import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { ToastProvider } from '../components/common/Toast';
import { ModelExplorerView } from '../components/explorer/ModelExplorerView';

const api = vi.hoisted(() => ({
  fetchModel: vi.fn(),
  createJob: vi.fn(),
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
    api.fetchModel.mockResolvedValue(metadata);
    api.createJob.mockResolvedValue({ job_id: 'job_prepare_01', status: 'QUEUED' });
  });

  it('queues CONFIG_ONLY without weight consent by default', async () => {
    renderView();
    await inspectModel();
    fireEvent.click(screen.getByRole('button', { name: /Prepare configuration only/ }));
    await waitFor(() => expect(api.createJob).toHaveBeenCalledTimes(1));
    expect(api.createJob.mock.calls[0][0]).toMatchObject({
      materialization_mode: 'METADATA_ONLY',
      weights_consent: false,
      revision: metadata.commit_sha,
    });
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
});
