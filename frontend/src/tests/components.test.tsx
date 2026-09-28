import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StatusBadge } from '../components/common/StatusBadge';
import { DomainStatusTag } from '../components/common/DomainStatusTag';
import { LogViewer } from '../components/common/LogViewer';
import { DashboardView } from '../components/dashboard/DashboardView';
import { Sidebar } from '../components/layout/Sidebar';
import { RunsView } from '../components/runs/RunsView';
import { resolveInitialTab } from '../ui/navigation';
import { agentSummary, DOMAIN_STATUS_COPY, JOB_STATUS_COPY, JOB_TYPE_COPY, tabForJobType } from '../ui/presentation';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { JobEvent, HealthResponse, JobListResponse } from '../api/types';

describe('Common Components', () => {
  it('renders StatusBadge for all lifecycle statuses', () => {
    const { rerender } = render(<StatusBadge status="QUEUED" />);
    expect(screen.getByText('Waiting to start')).toBeInTheDocument();

    rerender(<StatusBadge status="RUNNING" />);
    expect(screen.getByText('In progress')).toBeInTheDocument();

    rerender(<StatusBadge status="SUCCEEDED" />);
    expect(screen.getByText('Completed')).toBeInTheDocument();

    rerender(<StatusBadge status="FAILED" />);
    expect(screen.getByText('Could not complete')).toBeInTheDocument();

    rerender(<StatusBadge status="CANCELLED" />);
    expect(screen.getByText('Cancelled')).toBeInTheDocument();
  });

  it('renders DomainStatusTag with informative descriptions', () => {
    const { rerender } = render(<DomainStatusTag status="CONFIG_ONLY" />);
    expect(screen.getByText('Configuration ready')).toBeInTheDocument();
    expect(screen.getByText('Configuration ready')).toHaveAttribute('title', DOMAIN_STATUS_COPY.CONFIG_ONLY.explanation);

    rerender(<DomainStatusTag status="PREPARED" />);
    expect(screen.getByText('Model files prepared')).toBeInTheDocument();

    rerender(<DomainStatusTag status="EXECUTED" />);
    expect(screen.getByText('Inference completed on AMD')).toBeInTheDocument();

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

  it('renders DashboardView with hardware notice and job history', () => {
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

    render(
      <DashboardView
        health={mockHealth}
        jobsList={mockJobs}
        isLoadingJobs={false}
        onNavigate={vi.fn()}
        onSelectJob={vi.fn()}
      />
    );

    // One obvious product action and truthful environment context
    expect(screen.getByRole('button', { name: /Find a model/ })).toBeInTheDocument();
    expect(screen.getByText(/hardware execution requires a ROCm host/)).toBeInTheDocument();

    // Recent work is presented as a continuation card, not an admin table
    expect(screen.getByText('Build')).toBeInTheDocument();
    expect(screen.getByText('Qwen2.5-0.5B-Instruct')).toBeInTheDocument();
  });

  it('keeps primary navigation focused on Home, Models, Runs, and Optimize', () => {
    render(<Sidebar activeTab="runs" onTabChange={vi.fn()} />);
    expect(screen.getByRole('navigation', { name: 'Primary navigation' })).toHaveTextContent('Home');
    expect(screen.getByRole('navigation', { name: 'Primary navigation' })).toHaveTextContent('Models');
    expect(screen.getByRole('navigation', { name: 'Primary navigation' })).toHaveTextContent('Runs');
    expect(screen.getByRole('navigation', { name: 'Primary navigation' })).toHaveTextContent('Optimize');
    expect(screen.queryByText('Forge')).not.toBeInTheDocument();
    expect(screen.queryByText('Engineer')).not.toBeInTheDocument();
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
    expect(JOB_STATUS_COPY).toEqual({ QUEUED: 'Waiting to start', RUNNING: 'In progress', SUCCEEDED: 'Completed', FAILED: 'Could not complete', CANCELLED: 'Cancelled' });
    expect(JOB_TYPE_COPY.PREPARE_MODEL_FOR_AMD).toBe('Model preparation');
    expect(DOMAIN_STATUS_COPY.CONFIG_ONLY.explanation).toMatch(/Weights were not downloaded/);
    expect(DOMAIN_STATUS_COPY.CONFIG_ONLY.explanation).toMatch(/AMD execution was not performed/);
    expect(DOMAIN_STATUS_COPY.PREPARED.explanation).toMatch(/AMD execution was not performed/);
    expect(DOMAIN_STATUS_COPY.EXECUTED.explanation).toMatch(/physical AMD ROCm hardware/);
  });

  it('shows Runs empty/loading/error states distinctly and summarizes job types without raw enums', () => {
    const onRetry = vi.fn();
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { rerender } = render(<QueryClientProvider client={queryClient}><RunsView jobsList={null} isLoading onRetry={onRetry} onOpenJob={vi.fn()} /></QueryClientProvider>);
    expect(screen.getByLabelText('Loading runs')).toBeInTheDocument();
    rerender(<QueryClientProvider client={queryClient}><RunsView jobsList={{ items: [], total: 0, limit: 100, offset: 0 }} isLoading={false} onOpenJob={vi.fn()} /></QueryClientProvider>);
    expect(screen.getByText('No runs yet')).toBeInTheDocument();
    rerender(<QueryClientProvider client={queryClient}><RunsView jobsList={null} isLoading={false} isError onRetry={onRetry} onOpenJob={vi.fn()} /></QueryClientProvider>);
    expect(screen.getByText('Runs couldn’t be loaded')).toBeInTheDocument();
    expect(onRetry).not.toHaveBeenCalled();
  });

  it('counts only accurately labeled Agent states', () => {
    expect(agentSummary([
      { agent_id: '1', name: 'Online', hostname: 'one', status: 'ONLINE', last_seen: '2026-09-01T00:00:00Z', capabilities: { rocm_detected: true, hip_detected: true, amd_gpu_count: 1, gpu_names: ['AMD'], capabilities: [] } },
      { agent_id: '2', name: 'Busy', hostname: 'two', status: 'BUSY', last_seen: '2026-09-01T00:00:00Z', capabilities: { rocm_detected: false, hip_detected: false, amd_gpu_count: 0, gpu_names: [], capabilities: [] } },
      { agent_id: '3', name: 'Offline', hostname: 'three', status: 'OFFLINE', last_seen: '2026-09-01T00:00:00Z', capabilities: { rocm_detected: false, hip_detected: false, amd_gpu_count: 0, gpu_names: [], capabilities: [] } },
    ])).toBe('1 connected · 1 working · 1 not connected · 0 interrupted');
  });

  it('distinguishes Agent loading, failure, and true zero-agent states', () => {
    const base = { health: null, jobsList: null, isLoadingJobs: false, onNavigate: vi.fn(), onSelectJob: vi.fn() };
    const { rerender } = render(<DashboardView {...base} isLoadingAgents />);
    expect(screen.getByLabelText('Loading agents')).toBeInTheDocument();
    expect(screen.queryByText('No agents are registered yet.')).not.toBeInTheDocument();
    rerender(<DashboardView {...base} isAgentsError onRetryAgents={vi.fn()} />);
    expect(screen.getByText('Agent status couldn’t be loaded')).toBeInTheDocument();
    expect(screen.getByText(/doesn’t mean there are no agents/)).toBeInTheDocument();
    rerender(<DashboardView {...base} agents={[]} />);
    expect(screen.getByText(/No agents are registered yet/)).toBeInTheDocument();
    rerender(<DashboardView {...base} agents={[{ agent_id: 'offline', name: 'Offline', hostname: 'laptop', status: 'OFFLINE', last_seen: '2026-09-01T00:00:00Z', capabilities: { rocm_detected: false, hip_detected: false, amd_gpu_count: 0, gpu_names: [], capabilities: [] } }]} />);
    expect(screen.getByText('0 available')).toBeInTheDocument();
    expect(screen.getByText('0 connected · 0 working · 1 not connected · 0 interrupted')).toBeInTheDocument();
    expect(screen.getByText('Not connected')).toBeInTheDocument();
  });

  it('does not turn a failed recent-runs request into a false empty state', () => {
    const retry = vi.fn();
    render(<DashboardView health={null} jobsList={null} isLoadingJobs={false} isJobsError onRetryJobs={retry} onNavigate={vi.fn()} onSelectJob={vi.fn()} />);
    expect(screen.getByText('Recent runs couldn’t be loaded')).toBeInTheDocument();
    expect(screen.queryByText('Start with a model')).not.toBeInTheDocument();
    screen.getByRole('button', { name: /Retry/ }).click();
    expect(retry).toHaveBeenCalledOnce();
  });
});
