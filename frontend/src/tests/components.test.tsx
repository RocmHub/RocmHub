import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StatusBadge } from '../components/common/StatusBadge';
import { DomainStatusTag } from '../components/common/DomainStatusTag';
import { LogViewer } from '../components/common/LogViewer';
import { DashboardView } from '../components/dashboard/DashboardView';
import type { JobEvent, HealthResponse, JobListResponse } from '../api/types';

describe('Common Components', () => {
  it('renders StatusBadge for all lifecycle statuses', () => {
    const { rerender } = render(<StatusBadge status="QUEUED" />);
    expect(screen.getByText('QUEUED')).toBeInTheDocument();

    rerender(<StatusBadge status="RUNNING" />);
    expect(screen.getByText('RUNNING')).toBeInTheDocument();

    rerender(<StatusBadge status="SUCCEEDED" />);
    expect(screen.getByText('SUCCEEDED')).toBeInTheDocument();

    rerender(<StatusBadge status="FAILED" />);
    expect(screen.getByText('FAILED')).toBeInTheDocument();

    rerender(<StatusBadge status="CANCELLED" />);
    expect(screen.getByText('CANCELLED')).toBeInTheDocument();
  });

  it('renders DomainStatusTag with informative descriptions', () => {
    const { rerender } = render(<DomainStatusTag status="CONFIG_ONLY" />);
    expect(screen.getByText('CONFIG_ONLY')).toBeInTheDocument();
    expect(screen.getByText('CONFIG_ONLY')).toHaveAttribute('title');

    rerender(<DomainStatusTag status="PREPARED" />);
    expect(screen.getByText('PREPARED')).toBeInTheDocument();

    rerender(<DomainStatusTag status="EXECUTED" />);
    expect(screen.getByText('EXECUTED')).toBeInTheDocument();

    rerender(<DomainStatusTag status="NOT_MEASURED" />);
    expect(screen.getByText('NOT_MEASURED')).toBeInTheDocument();
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

    // Product hero
    expect(screen.getByText('Prepare AI Models')).toBeInTheDocument();

    // Hardware notice (CONFIG ONLY mode shown)
    expect(screen.getByText(/CONFIG ONLY mode/)).toBeInTheDocument();

    // Jobs table
    expect(screen.getByText('job_abc')).toBeInTheDocument();
    expect(screen.getByText('Qwen/Qwen2.5-0.5B-Instruct')).toBeInTheDocument();
  });
});
