import { describe, it, expect, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { StatusBadge } from '../components/common/StatusBadge';
import { DomainStatusTag } from '../components/common/DomainStatusTag';
import { LogViewer } from '../components/common/LogViewer';
import { DashboardView } from '../components/dashboard/DashboardView';
import { Sidebar } from '../components/layout/Sidebar';
import { RunsView } from '../components/runs/RunsView';
import { Navbar } from '../components/layout/Navbar';
import { ModelExplorerView } from '../components/explorer/ModelExplorerView';
import { OptimizationLabView } from '../components/optimization/OptimizationLabView';
import { AIEngineerView } from '../components/engineer/AIEngineerView';
import { ForgeStudioView } from '../components/forge/ForgeStudioView';
import { ToastProvider } from '../components/common/Toast';
import { resolveInitialTab } from '../ui/navigation';
import { activityOutcomeCopy, agentSummary, DOMAIN_STATUS_COPY, JOB_STATUS_COPY, JOB_TYPE_COPY, jobProgressCopy, materializationCopy, tabForJobType } from '../ui/presentation';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as apiClient from '../api/client';
import type { JobEvent, HealthResponse, JobListResponse } from '../api/types';

describe('Common Components', () => {
  it('renders StatusBadge for all lifecycle statuses', () => {
    const { rerender } = render(<StatusBadge status="QUEUED" />);
    expect(screen.getByText('Waiting')).toBeInTheDocument();

    rerender(<StatusBadge status="RUNNING" />);
    expect(screen.getByText('Running')).toBeInTheDocument();

    rerender(<StatusBadge status="SUCCEEDED" />);
    expect(screen.getByText('Completed')).toBeInTheDocument();

    rerender(<StatusBadge status="FAILED" />);
    expect(screen.getByText('Failed')).toBeInTheDocument();

    rerender(<StatusBadge status="CANCELLED" />);
    expect(screen.getByText('Cancelled')).toBeInTheDocument();
  });

  it('renders DomainStatusTag with informative descriptions', () => {
    const { rerender } = render(<DomainStatusTag status="CONFIG_ONLY" />);
    expect(screen.getByText('Configuration prepared')).toBeInTheDocument();
    expect(screen.getByText('Configuration prepared')).toHaveAttribute('title', DOMAIN_STATUS_COPY.CONFIG_ONLY.explanation);

    rerender(<DomainStatusTag status="PREPARED" />);
    expect(screen.getByText('Model prepared')).toBeInTheDocument();

    rerender(<DomainStatusTag status="EXECUTED" />);
    expect(screen.getByText('Ran on AMD')).toBeInTheDocument();

    rerender(<DomainStatusTag status="NOT_MEASURED" />);
    expect(screen.getByText('Performance not measured')).toBeInTheDocument();
  });

  it('renders LogViewer with formatted events', () => {
    const events: JobEvent[] = [
      {
        event_id: 1,
        job_id: 'job_xyz',
        sequence: 1,
        timestamp: '2026-09-22T08:00:00Z',
        phase: 'PLANNING',
        status: 'RUNNING',
        message: 'Resolving model architecture',
        error_code: null,
        details: null,
      },
    ];

    render(<LogViewer events={events} />);
    expect(screen.getByText('Resolving model architecture')).toBeInTheDocument();
    expect(screen.getByText('PLANNING')).toBeInTheDocument();
  });

  it('renders the search-first start screen and recent activity', () => {
    const mockHealth: HealthResponse = {
      status: 'healthy',
      version: '0.1.0',
      rocm_available: false,
      host_platform: { os: 'darwin', arch: 'arm64', python_version: '3.9.6', is_apple_silicon: true },
      system: { os: 'darwin', python_version: '3.9.6', rocm_version: null, torch_version: '2.2.0', gpus_detected: 0, gpus: [] },
      orchestrator: { queue_size: 0, active_directory_locks: [] },
      warnings: ['Running on macOS without AMD ROCm GPU.'],
    };

    const mockJobs: JobListResponse = {
      items: [
        {
          job_id: 'job_abc',
          job_type: 'FORGE_BUILD',
          model_id: 'Qwen/Qwen2.5-0.5B-Instruct',
          revision: 'main',
          status: 'SUCCEEDED',
          domain_status: 'CONFIG_ONLY',
          created_at: '2026-09-22T08:00:00Z',
          started_at: '2026-09-22T08:00:01Z',
          completed_at: '2026-09-22T08:00:05Z',
          timeout_seconds: 600,
          output_dir: '/tmp/test',
          error_message: null,
          error_code: null,
        },
      ],
      total: 1,
      limit: 20,
      offset: 0,
    };

    const { container } = render(
      <DashboardView
        health={mockHealth}
        jobsList={mockJobs}
        isLoadingJobs={false}
        onNavigate={vi.fn()}
        onSelectJob={vi.fn()}
      />
    );

    expect(screen.getByLabelText('Paste a public Hugging Face model ID')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Inspect model' })).toBeInTheDocument();
    expect(screen.getByText('Configuration prepared')).toBeInTheDocument();
    expect(screen.getByText('Qwen2.5-0.5B-Instruct')).toBeInTheDocument();
    expect(container.querySelector('.activity-row .activity-symbol')).toHaveClass('activity-config-only');
  });

  it('keeps an empty Home state useful without inventing recent activity', () => {
    render(<DashboardView health={null} jobsList={{ items: [], total: 0, limit: 20, offset: 0 }} isLoadingJobs={false} onNavigate={vi.fn()} onSelectJob={vi.fn()} />);
    expect(screen.getByText('No recent work yet')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Inspect a model to get started' })).toBeInTheDocument();
  });

  it('shows compute state truthfully and keeps runtime facts in technical details', () => {
    const health: HealthResponse = {
      status: 'healthy', version: '0.1.0', rocm_available: false,
      host_platform: { os: 'darwin', arch: 'arm64', python_version: '3.9.6', is_apple_silicon: true },
      system: { os: 'darwin', python_version: '3.9.6', rocm_version: '6.2', torch_version: '2.2.0', gpus_detected: 0, gpus: [] },
      orchestrator: { queue_size: 0, active_directory_locks: [] }, warnings: [],
    };
    const onThemeChange = vi.fn();
    render(<Navbar health={health} isLoading={false} isError={false} theme="system" onThemeChange={onThemeChange} />);
    fireEvent.change(screen.getByRole('combobox', { name: 'Color theme' }), { target: { value: 'light' } });
    expect(onThemeChange).toHaveBeenCalledWith('light');
    expect(screen.getByRole('option', { name: 'System' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'No AMD compute' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'No AMD compute' }));
    expect(screen.getByText(/No AMD runtime detected here/)).toBeInTheDocument();
    fireEvent.click(screen.getByText('Service details'));
    expect(screen.getByText(/ROCm 6.2/)).toBeInTheDocument();
  });

  it('keeps model revisions in advanced search options and avoids a fake catalog', () => {
    sessionStorage.clear();
    render(<ToastProvider><ModelExplorerView onSelectModelForForge={vi.fn()} onNavigate={vi.fn()} /></ToastProvider>);
    const advanced = screen.getByText('Advanced options').closest('details');
    expect(advanced).not.toHaveAttribute('open');
    expect(advanced).toContainElement(screen.getAllByLabelText('Advanced revision')[0]);
    expect(screen.getAllByLabelText('Advanced revision')[0]).toHaveValue('main');
    expect(screen.queryByText('Production class')).not.toBeInTheDocument();
    expect(screen.getAllByText('Larger instruction model').length).toBeGreaterThan(0);
    expect(screen.getByText(/Select one, then choose Inspect/)).toBeInTheDocument();
  });

  it('explains that Optimize currently prepares configuration comparisons, even when AMD is detected', () => {
    const { rerender } = render(<ToastProvider><OptimizationLabView amdComputeAvailable={false} /></ToastProvider>);
    const note = screen.getByRole('note', { name: 'Hardware measurement availability' });
    expect(note).toHaveTextContent(/No AMD compute is available/);
    expect(note).toHaveTextContent(/Not measured/);
    expect(screen.getByRole('button', { name: 'Prepare comparison' })).toBeInTheDocument();
    rerender(<ToastProvider><OptimizationLabView amdComputeAvailable={true} /></ToastProvider>);
    expect(screen.getByRole('note', { name: 'Hardware measurement availability' })).toHaveTextContent(/does not download model weights or start inference/);
    expect(screen.getByRole('button', { name: 'Prepare comparison' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Run comparison on AMD' })).not.toBeInTheDocument();
  });

  it('makes AI Engineer goals understandable and marks execution goals unavailable without AMD compute', () => {
    render(<ToastProvider><AIEngineerView amdComputeAvailable={false} /></ToastProvider>);
    expect(screen.getByRole('button', { name: /Prepare this model/ })).toBeEnabled();
    expect(screen.getByRole('button', { name: /Improve throughput/ })).toBeDisabled();
    expect(screen.getByRole('button', { name: /Reduce latency/ })).toBeDisabled();
    expect(screen.getAllByText('Requires AMD compute')).toHaveLength(2);
    expect(screen.getByText(/AMD hardware validation is not performed/)).toBeInTheDocument();
  });

  it('uses the user-facing Forge stage labels', () => {
    render(<ToastProvider><ForgeStudioView /></ToastProvider>);
    for (const label of ['Model', 'Hardware', 'Profile', 'Review', 'Build', 'Result']) {
      expect(screen.getByText(label, { exact: true })).toBeInTheDocument();
    }
    expect(screen.getByText(/immutable revision before preparation/)).toBeInTheDocument();
  });

  it('keeps primary navigation focused on Models, Activity, and Optimize', () => {
    render(<Sidebar activeTab="runs" onTabChange={vi.fn()} />);
    expect(screen.getByRole('navigation', { name: 'Primary navigation' })).toHaveTextContent('Models');
    expect(screen.getByRole('navigation', { name: 'Primary navigation' })).toHaveTextContent('Activity');
    expect(screen.getByRole('navigation', { name: 'Primary navigation' })).toHaveTextContent('Optimize');
    expect(screen.queryByText('Forge')).not.toBeInTheDocument();
    expect(screen.queryByText('Engineer')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Tools/ }));
    expect(screen.getByRole('menuitem', { name: /AI Engineer/ })).toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: /Forge Studio/ })).toBeInTheDocument();
  });

  it('routes job deep links to Runs and preserves legacy workflow routes', () => {
    expect(resolveInitialTab('', '?job_id=prep-1')).toBe('runs');
    expect(resolveInitialTab('#explorer', '?job_id=prep-1')).toBe('explorer');
    expect(resolveInitialTab('#forge', '')).toBe('forge');
    expect(resolveInitialTab('#engineer', '')).toBe('engineer');
    expect(tabForJobType('PREPARE_MODEL_FOR_AMD')).toBe('explorer');
    expect(tabForJobType('FORGE_BUILD')).toBe('forge');
    expect(tabForJobType('ENGINEER')).toBe('engineer');
    expect(tabForJobType('OPTIMIZATION')).toBe('optimization');
  });

  it('keeps all status and type labels human-readable and truthful', () => {
    expect(JOB_STATUS_COPY).toEqual({ QUEUED: 'Waiting', RUNNING: 'Running', SUCCEEDED: 'Completed', FAILED: 'Failed', CANCELLED: 'Cancelled' });
    expect(jobProgressCopy('RUNNING', 'AGENT_CLAIMED')).toBe('Waiting');
    expect(jobProgressCopy('RUNNING', 'MODEL_DOWNLOADING')).toBe('Downloading');
    expect(jobProgressCopy('RUNNING', 'VERIFYING_DIGESTS')).toBe('Verifying');
    expect(jobProgressCopy('RUNNING', 'BUILD_ARTIFACT')).toBe('Preparing');
    expect(JOB_TYPE_COPY.PREPARE_MODEL_FOR_AMD).toBe('Model preparation');
    expect(DOMAIN_STATUS_COPY.CONFIG_ONLY.explanation).toMatch(/Weights were not downloaded/);
    expect(DOMAIN_STATUS_COPY.CONFIG_ONLY.explanation).toMatch(/AMD execution was not performed/);
    expect(DOMAIN_STATUS_COPY.PREPARED.explanation).toMatch(/AMD execution was not performed/);
    expect(DOMAIN_STATUS_COPY.EXECUTED.explanation).toMatch(/physical AMD ROCm hardware/);
  });

  it('summarizes Activity by actual outcome before exposing backend terms', () => {
    const configOnly = activityOutcomeCopy({ job_type: 'PREPARE_MODEL_FOR_AMD', status: 'SUCCEEDED', domain_status: 'CONFIG_ONLY', error_message: null });
    expect(configOnly).toEqual({ title: 'Configuration prepared', description: 'Weights were not downloaded. AMD execution was not performed.' });

    const prepared = activityOutcomeCopy({ job_type: 'PREPARE_MODEL_FOR_AMD', status: 'SUCCEEDED', domain_status: 'PREPARED', error_message: null });
    expect(prepared.title).toBe('Model prepared');
    expect(prepared.description).toMatch(/available and verified/);

    const optimization = activityOutcomeCopy({ job_type: 'OPTIMIZATION', status: 'SUCCEEDED', domain_status: null, error_message: null });
    expect(optimization).toEqual({ title: 'Comparison prepared', description: 'Performance measurements have not been collected.' });
    expect(optimization.title).not.toMatch(/best|optimized|benchmarked/i);
  });

  it('distinguishes a verified cache hit from a new model download', () => {
    expect(materializationCopy('HIT_VERIFIED')).toEqual({ title: 'Model files already available', description: 'Verified from the connected compute cache. No repeat download was needed.' });
    expect(materializationCopy('MISS_DOWNLOADED')).toEqual({ title: 'Model files downloaded and verified', description: 'Stored on connected compute. AMD execution was not performed.' });
    expect(materializationCopy(null).title).toBe('Verified model files are ready');
  });

  it('shows Runs empty/loading/error states distinctly and summarizes job types without raw enums', () => {
    const onRetry = vi.fn();
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { rerender } = render(<QueryClientProvider client={queryClient}><RunsView jobsList={null} isLoading onRetry={onRetry} onOpenJob={vi.fn()} /></QueryClientProvider>);
    expect(screen.getByLabelText('Loading runs')).toBeInTheDocument();
    rerender(<QueryClientProvider client={queryClient}><RunsView jobsList={{ items: [], total: 0, limit: 100, offset: 0 }} isLoading={false} onOpenJob={vi.fn()} /></QueryClientProvider>);
    expect(screen.getByText('No runs yet')).toBeInTheDocument();
    expect(screen.getAllByText(/Model preparation and optimization studies will appear here/).length).toBeGreaterThan(0);
    expect(screen.queryByText('0 total')).not.toBeInTheDocument();
    rerender(<QueryClientProvider client={queryClient}><RunsView jobsList={null} isLoading={false} isError onRetry={onRetry} onOpenJob={vi.fn()} /></QueryClientProvider>);
    expect(screen.getByText('Runs couldn’t be loaded')).toBeInTheDocument();
    expect(onRetry).not.toHaveBeenCalled();
  });

  it('filters Runs locally by activity outcome', () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const makeJob = (job_id: string, status: 'SUCCEEDED' | 'FAILED'): JobListResponse['items'][number] => ({
      job_id, job_type: 'PREPARE_MODEL_FOR_AMD', model_id: `org/${job_id}`, revision: 'main', status,
      domain_status: status === 'SUCCEEDED' ? 'CONFIG_ONLY' : null, created_at: '2026-09-22T08:00:00Z',
      started_at: null, completed_at: null, timeout_seconds: 600, output_dir: null, error_message: null, error_code: null,
    });
    const jobsList: JobListResponse = { items: [makeJob('success-run', 'SUCCEEDED'), makeJob('failed-run', 'FAILED')], total: 2, limit: 100, offset: 0 };
    render(<QueryClientProvider client={queryClient}><RunsView jobsList={jobsList} isLoading={false} onOpenJob={vi.fn()} /></QueryClientProvider>);
    expect(screen.getByText('org/success-run')).toBeInTheDocument();
    expect(screen.getByText('org/failed-run')).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole('button', { name: /Completed/ })[0]);
    expect(screen.getByText('org/success-run')).toBeInTheDocument();
    expect(screen.queryByText('org/failed-run')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Needs attention/ }));
    expect(screen.getByText('org/failed-run')).toBeInTheDocument();
    expect(screen.queryByText('org/success-run')).not.toBeInTheDocument();
  });

  it('shows an honest run timeline and result without implying AMD execution', async () => {
    const job: JobListResponse['items'][number] = {
      job_id: 'prepared-run', job_type: 'PREPARE_MODEL_FOR_AMD', model_id: 'Qwen/Qwen2.5-0.5B-Instruct', revision: 'main',
      status: 'SUCCEEDED', domain_status: 'PREPARED', created_at: '2026-09-22T08:00:00Z', started_at: '2026-09-22T08:01:00Z',
      completed_at: '2026-09-22T08:02:00Z', timeout_seconds: 600, output_dir: '/tmp/prepared-run', error_message: null, error_code: null,
    };
    const resultSpy = vi.spyOn(apiClient, 'fetchJobResult').mockResolvedValue({ job_id: job.job_id, job_type: job.job_type, job_status: 'SUCCEEDED', domain_status: 'PREPARED', output_dir: job.output_dir, completed_at: job.completed_at, result: { materialization: { cache_status: 'verified' } }, error_message: null });
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={queryClient}><RunsView jobsList={{ items: [job], total: 1, limit: 100, offset: 0 }} isLoading={false} selectedJobId={job.job_id} onOpenJob={vi.fn()} /></QueryClientProvider>);
    expect(await screen.findByText('Run timeline')).toBeInTheDocument();
    expect(screen.getAllByText('Model prepared').length).toBeGreaterThan(0);
    expect(screen.getAllByText('AMD execution not performed').length).toBeGreaterThan(0);
    expect(screen.queryByText('Ran on AMD')).not.toBeInTheDocument();
    expect(resultSpy).toHaveBeenCalledWith(job.job_id);
    resultSpy.mockRestore();
  });

  it('counts only accurately labeled Agent states', () => {
    expect(agentSummary([
      { agent_id: '1', name: 'Online', hostname: 'one', status: 'ONLINE', last_seen: '2026-09-01T00:00:00Z', capabilities: { rocm_detected: true, hip_detected: true, amd_gpu_count: 1, gpu_names: ['AMD'], capabilities: [] } },
      { agent_id: '2', name: 'Busy', hostname: 'two', status: 'BUSY', last_seen: '2026-09-01T00:00:00Z', capabilities: { rocm_detected: false, hip_detected: false, amd_gpu_count: 0, gpu_names: [], capabilities: [] } },
      { agent_id: '3', name: 'Offline', hostname: 'three', status: 'OFFLINE', last_seen: '2026-09-01T00:00:00Z', capabilities: { rocm_detected: false, hip_detected: false, amd_gpu_count: 0, gpu_names: [], capabilities: [] } },
    ])).toBe('1 connected · 1 working · 1 not connected · 0 interrupted');
  });

  it('never presents offline agents as connected in the compute panel', () => {
    const health: HealthResponse = { status: 'healthy', version: '0.1.0', rocm_available: false, host_platform: { os: 'darwin', arch: 'arm64', python_version: '3.9.6', is_apple_silicon: true }, system: { os: 'darwin', python_version: '3.9.6', rocm_version: null, torch_version: '2.2.0', gpus_detected: 0, gpus: [] }, orchestrator: { queue_size: 0, active_directory_locks: [] }, warnings: [] };
    const offline = [{ agent_id: 'offline', name: 'Offline', hostname: 'laptop', status: 'OFFLINE' as const, last_seen: '2026-09-01T00:00:00Z', capabilities: { rocm_detected: true, hip_detected: true, amd_gpu_count: 1, gpu_names: ['AMD GPU'], capabilities: ['PREPARE_MODEL_FOR_AMD'] } }];
    render(<Navbar health={health} isLoading={false} isError={false} agents={offline} theme="system" onThemeChange={vi.fn()} />);
    expect(screen.getByRole('button', { name: 'No AMD compute' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'No AMD compute' }));
    expect(screen.getByText('AMD GPU')).toBeInTheDocument();
    expect(screen.getByText('Not connected')).toBeInTheDocument();
    expect(screen.queryByText('AMD compute available')).not.toBeInTheDocument();
  });

  it('labels compute status as unknown when the service cannot be reached', () => {
    render(<Navbar health={null} isLoading={false} isError agents={[]} theme="system" onThemeChange={vi.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: 'Compute status unknown' }));
    expect(screen.getByText(/Connection status could not be confirmed/)).toBeInTheDocument();
  });

  it('does not turn a failed recent-runs request into a false empty state', () => {
    const retry = vi.fn();
    render(<DashboardView health={null} jobsList={null} isLoadingJobs={false} isJobsError onRetryJobs={retry} onNavigate={vi.fn()} onSelectJob={vi.fn()} />);
    expect(screen.getByText(/Recent activity couldn’t be loaded/)).toBeInTheDocument();
    expect(screen.getByText(/Inspect a model, prepare its setup/)).toBeInTheDocument();
    screen.getAllByRole('button', { name: /Retry/ })[0].click();
    expect(retry).toHaveBeenCalledOnce();
  });
});
