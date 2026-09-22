import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchHealth, fetchJobs } from './api/client';
import { Navbar } from './components/layout/Navbar';
import { Sidebar, type NavTab } from './components/layout/Sidebar';
import { DashboardView } from './components/dashboard/DashboardView';
import { ModelExplorerView } from './components/explorer/ModelExplorerView';
import { ForgeStudioView } from './components/forge/ForgeStudioView';
import { AIEngineerView } from './components/engineer/AIEngineerView';
import { OptimizationLabView } from './components/optimization/OptimizationLabView';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavTab>('dashboard');
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [selectedModelForForge, setSelectedModelForForge] = useState({
    modelId: 'Qwen/Qwen2.5-0.5B-Instruct',
    revision: 'main',
  });
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

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
    setActiveTab('forge');
  };

  const handleSelectJob = (jobId: string) => {
    setSelectedJobId(jobId);
    // Find job type to navigate to the respective studio tab
    const job = jobsList?.items.find((j) => j.job_id === jobId);
    if (job?.job_type === 'ENGINEER') {
      setActiveTab('engineer');
    } else if (job?.job_type === 'OPTIMIZATION') {
      setActiveTab('optimization');
    } else {
      setActiveTab('forge');
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
          onTabChange={setActiveTab}
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
              onNavigate={setActiveTab}
              onSelectJob={handleSelectJob}
            />
          )}

          {activeTab === 'explorer' && (
            <ModelExplorerView
              onSelectModelForForge={handleSelectModelForForge}
              onNavigate={setActiveTab}
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
