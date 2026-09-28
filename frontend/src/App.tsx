import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchAgents, fetchHealth, fetchJobs } from './api/client';
import { Navbar } from './components/layout/Navbar';
import { Sidebar, type NavTab } from './components/layout/Sidebar';
import { DashboardView } from './components/dashboard/DashboardView';
import { ModelExplorerView } from './components/explorer/ModelExplorerView';
import { ForgeStudioView } from './components/forge/ForgeStudioView';
import { AIEngineerView } from './components/engineer/AIEngineerView';
import { OptimizationLabView } from './components/optimization/OptimizationLabView';
import { RunsView } from './components/runs/RunsView';
import { ToastProvider } from './components/common/Toast';
import { resolveInitialTab } from './ui/navigation';
import { hasConnectedAmdCompute } from './ui/presentation';
import { readThemePreference, resolveTheme, persistThemePreference, type ThemePreference } from './ui/theme';

const VALID_TABS: NavTab[] = ['dashboard', 'explorer', 'runs', 'forge', 'engineer', 'optimization'];

function getInitialTab(): NavTab {
  if (typeof window === 'undefined') return 'dashboard';
  return resolveInitialTab(window.location.hash, window.location.search);
}

function getInitialJobId(): string | null {
  if (typeof window === 'undefined') return null;
  return new URLSearchParams(window.location.search).get('job_id');
}

export const App: React.FC = () => {
  const [theme, setTheme] = useState<ThemePreference>(readThemePreference);
  const [activeTab, setActiveTab] = useState<NavTab>(getInitialTab);
  const [selectedModelForForge, setSelectedModelForForge] = useState({
    modelId: 'Qwen/Qwen2.5-0.5B-Instruct',
    revision: 'main',
  });
  const [selectedModelId, setSelectedModelId] = useState('');
  const [selectedJobId, setSelectedJobId] = useState<string | null>(getInitialJobId);

  const navigateTo = (tab: NavTab) => {
    setActiveTab(tab);
    if (typeof window !== 'undefined' && window.location.hash !== `#${tab}`) {
      window.location.hash = tab;
    }
  };

  const switchTab = (tab: NavTab) => {
    setSelectedJobId(null);
    navigateTo(tab);
  };

  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace('#', '').toLowerCase();
      if (VALID_TABS.includes(hash as NavTab)) {
        setActiveTab(hash as NavTab);
      }
    };
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: light)');
    const applyTheme = () => {
      const resolved = resolveTheme(theme, media.matches);
      document.documentElement.dataset.theme = resolved;
      document.documentElement.style.colorScheme = resolved;
    };
    applyTheme();
    if (theme === 'system') media.addEventListener('change', applyTheme);
    persistThemePreference(theme);
    return () => media.removeEventListener('change', applyTheme);
  }, [theme]);

  const {
    data: health,
    isLoading: isLoadingHealth,
    isError: isHealthError,
  } = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: 10000,
  });
  const {data: agents, isLoading: isLoadingAgents, isError: isAgentsError, refetch: refetchAgents} = useQuery({queryKey:['agents'],queryFn:fetchAgents,refetchInterval:10000});

  const {
    data: jobsList,
    isLoading: isLoadingJobs,
    isError: isJobsError,
    refetch: refetchJobs,
  } = useQuery({
    queryKey: ['jobs'],
    queryFn: () => fetchJobs({ limit: 100 }),
    refetchInterval: 5000,
  });

  const handleSelectModelForForge = (modelId: string, revision?: string) => {
    setSelectedModelForForge({ modelId, revision: revision || 'main' });
    setSelectedJobId(null);
    switchTab('forge');
  };

  const handleSelectJob = (jobId: string) => {
    setSelectedJobId(jobId);
    navigateTo('runs');
  };

  const handleOpenJob = (jobId: string, tab: NavTab) => {
    if (!jobId) { setSelectedJobId(null); navigateTo(tab); return; }
    setSelectedJobId(jobId);
    navigateTo(tab);
  };

  const handleJobCreated = (jobId: string) => {
    refetchJobs();
    setSelectedJobId(jobId);
  };

  return (
    <ToastProvider>
      <div className="h-screen app-surface text-content-primary flex flex-col overflow-hidden">
        <Navbar
          health={health ?? null}
          isLoading={isLoadingHealth}
          isError={isHealthError}
          agents={agents ?? []}
          isLoadingAgents={isLoadingAgents}
          isAgentsError={isAgentsError}
          onRetryAgents={() => { void refetchAgents(); }}
          theme={theme}
          onThemeChange={setTheme}
        />
        <div className="nav-rail"><Sidebar activeTab={activeTab} onTabChange={switchTab}/></div>
        <main className="flex-1 overflow-y-auto">
            {activeTab === 'dashboard' && (
              <DashboardView
                health={health ?? null}
                isLoadingHealth={isLoadingHealth}
                isHealthError={isHealthError}
                jobsList={jobsList ?? null}
                isLoadingJobs={isLoadingJobs}
                isJobsError={isJobsError}
                onRetryJobs={() => { void refetchJobs(); }}
                onNavigate={switchTab}
                onSelectJob={handleSelectJob}
                onSearchModel={(modelId) => { setSelectedModelId(modelId); setSelectedJobId(null); switchTab('explorer'); }}
                agents={agents ?? []}
                isLoadingAgents={isLoadingAgents}
                isAgentsError={isAgentsError}
                onRetryAgents={() => { void refetchAgents(); }}
              />
            )}
            {activeTab === 'explorer' && (
              <ModelExplorerView
                onSelectModelForForge={handleSelectModelForForge}
                onNavigate={switchTab}
                selectedJobId={selectedJobId}
                initialModelId={selectedModelId || undefined}
              />
            )}
            {activeTab === 'runs' && <RunsView jobsList={jobsList ?? null} isLoading={isLoadingJobs} isError={isJobsError} selectedJobId={selectedJobId} onRetry={() => { void refetchJobs(); }} onOpenJob={handleOpenJob} />}
            {activeTab === 'forge' && (
              <ForgeStudioView
                initialModelId={selectedModelForForge.modelId}
                initialRevision={selectedModelForForge.revision}
                selectedJobId={selectedJobId}
                onJobCreated={handleJobCreated}
              />
            )}
            {activeTab === 'engineer' && (
              <AIEngineerView
                selectedJobId={selectedJobId}
                onJobCreated={handleJobCreated}
                amdComputeAvailable={isLoadingHealth || isHealthError || isLoadingAgents || isAgentsError || !health ? null : hasConnectedAmdCompute(health, agents ?? [])}
              />
            )}
            {activeTab === 'optimization' && (
              <OptimizationLabView
                selectedJobId={selectedJobId}
                onJobCreated={handleJobCreated}
                amdComputeAvailable={isLoadingHealth || isHealthError || isLoadingAgents || isAgentsError || !health ? null : hasConnectedAmdCompute(health, agents ?? [])}
              />
            )}
        </main>
      </div>
    </ToastProvider>
  );
};

export default App;
