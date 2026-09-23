import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchHealth, fetchJobs } from './api/client';
import { Navbar } from './components/layout/Navbar';
import { Sidebar, type NavTab } from './components/layout/Sidebar';
import { DashboardView } from './components/dashboard/DashboardView';
import { ModelExplorerView } from './components/explorer/ModelExplorerView';
import { ForgeStudioView } from './components/forge/ForgeStudioView';
import { AIEngineerView } from './components/engineer/AIEngineerView';
import { OptimizationLabView } from './components/optimization/OptimizationLabView';

const VALID_TABS: NavTab[] = ['dashboard', 'explorer', 'forge', 'engineer', 'optimization'];

function getInitialTab(): NavTab {
  if (typeof window === 'undefined') return 'dashboard';
  const hash = window.location.hash.replace('#', '').toLowerCase();
  return VALID_TABS.includes(hash as NavTab) ? (hash as NavTab) : 'dashboard';
}

function getInitialJobId(): string | null {
  if (typeof window === 'undefined') return null;
  return new URLSearchParams(window.location.search).get('job_id');
}

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavTab>(getInitialTab);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [selectedModelForForge, setSelectedModelForForge] = useState({
    modelId: 'Qwen/Qwen2.5-0.5B-Instruct',
    revision: 'main',
  });
  const [selectedJobId, setSelectedJobId] = useState<string | null>(getInitialJobId);

  // Sync activeTab with URL hash
  const switchTab = (tab: NavTab) => {
    setActiveTab(tab);
    if (typeof window !== 'undefined' && window.location.hash !== `#${tab}`) {
      window.location.hash = tab;
    }
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

  // Poll system health
  const {
    data: health,
    isLoading: isLoadingHealth,
    isError: isHealthError,
  } = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: 10000,
  });

  // Poll jobs history
  const {
    data: jobsList,
    isLoading: isLoadingJobs,
    refetch: refetchJobs,
  } = useQuery({
    queryKey: ['jobs'],
    queryFn: () => fetchJobs({ limit: 20 }),
    refetchInterval: 5000,
  });

  const handleSelectModelForForge = (modelId: string, revision?: string) => {
    setSelectedModelForForge({ modelId, revision: revision || 'main' });
    switchTab('forge');
  };

  const handleSelectJob = (jobId: string) => {
    setSelectedJobId(jobId);
    // Find job type to navigate to the respective studio tab
    const job = jobsList?.items.find((j) => j.job_id === jobId);
    if (job?.job_type === 'ENGINEER') {
      switchTab('engineer');
    } else if (job?.job_type === 'OPTIMIZATION') {
      switchTab('optimization');
    } else {
      switchTab('forge');
    }
  };

  const handleJobCreated = (jobId: string) => {
    refetchJobs();
    setSelectedJobId(jobId);
  };

  return (
    <div className="min-h-screen bg-background text-content-primary flex flex-col">
      {/* Top Navigation */}
      <Navbar
        health={health ?? null}
        isLoading={isLoadingHealth}
        isError={isHealthError}
        isMobileMenuOpen={isMobileMenuOpen}
        onToggleMobileMenu={() => setIsMobileMenuOpen((prev) => !prev)}
      />

      {/* Main Content Layout */}
      <div className="flex flex-1 overflow-hidden relative">
        {/* Left Sidebar */}
        <Sidebar
          activeTab={activeTab}
          onTabChange={switchTab}
          isOpenMobile={isMobileMenuOpen}
          onCloseMobile={() => setIsMobileMenuOpen(false)}
        />

        {/* View Container */}
        <main className="flex-1 overflow-y-auto p-4 sm:p-6 bg-background">
          {activeTab === 'dashboard' && (
            <DashboardView
              health={health ?? null}
              jobsList={jobsList ?? null}
              isLoadingJobs={isLoadingJobs}
              onNavigate={switchTab}
              onSelectJob={handleSelectJob}
            />
          )}

          {activeTab === 'explorer' && (
            <ModelExplorerView
              onSelectModelForForge={handleSelectModelForForge}
              onNavigate={switchTab}
            />
          )}

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
            />
          )}

          {activeTab === 'optimization' && (
            <OptimizationLabView
              selectedJobId={selectedJobId}
              onJobCreated={handleJobCreated}
            />
          )}
        </main>
      </div>
    </div>
  );
};
export default App;
