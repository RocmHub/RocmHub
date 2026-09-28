import React, { useState } from 'react';
import type { AgentInfo, HealthResponse } from '../../api/types';
import { Activity, ChevronDown } from 'lucide-react';
import { AGENT_STATUS_COPY, hasConnectedAmdCompute } from '../../ui/presentation';

interface NavbarProps {
  health: HealthResponse | null; isLoading: boolean; isError: boolean; agents?: AgentInfo[];
  isLoadingAgents?: boolean; isAgentsError?: boolean; onRetryAgents?: () => void;
  isMobileMenuOpen?: boolean; onToggleMobileMenu?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ health, isLoading, isError, agents = [], isLoadingAgents = false, isAgentsError = false, onRetryAgents }) => {
  const [open, setOpen] = useState(false);
  const connected = hasConnectedAmdCompute(health, agents);
  const checking = isLoading || isLoadingAgents;
  const unknown = !checking && (isError || isAgentsError || !health);
  const label = checking ? 'Checking compute' : connected ? 'AMD compute connected' : unknown ? 'Compute status unknown' : 'No AMD compute';
  return <header className="product-header">
    <a className="brand-lockup" href="#dashboard" aria-label="ROCmHub home" onClick={() => window.dispatchEvent(new HashChangeEvent('hashchange'))}>
      <span className="brand-symbol" aria-hidden="true"><i/><i/></span><span className="brand-name">ROCmHub</span>
    </a>
    <div className="header-end">
      <div className="relative">
        <button aria-expanded={open} aria-label={label} onClick={() => setOpen(v => !v)} className="compute-indicator">
          <span className={`compute-mark ${connected ? 'compute-mark-on' : isLoading || isLoadingAgents ? 'compute-mark-wait' : ''}`}/><span>{label}</span><ChevronDown size={13}/>
        </button>
        {open && <>
          <button className="drawer-scrim" aria-label="Close compute panel" onClick={() => setOpen(false)}/>
          <aside role="dialog" aria-label="Compute" className="compute-drawer">
            <div className="drawer-heading"><div><div className="drawer-kicker">ROCmHub</div><h2>Compute</h2></div><button aria-label="Close compute panel" className="icon-button" onClick={() => setOpen(false)}>×</button></div>
            <p className="drawer-intro">{unknown ? 'Connection status could not be confirmed. Retry or check the service before assuming no compute is available.' : connected ? 'Available compute targets for preparation and AMD execution.' : 'No AMD compute is available right now. Model inspection and configuration preparation still work.'}</p>
            {!isLoading && !isError && health && <div className="compute-target"><span className={`compute-mark ${health.rocm_available ? 'compute-mark-on' : ''}`}/><div className="target-copy"><strong>{health.rocm_available ? 'Local AMD compute' : 'This service host'}</strong><span>{health.rocm_available ? 'ROCm execution available' : 'AMD execution unavailable'}</span></div><span className="target-status">{health.rocm_available ? 'Online' : '—'}</span></div>}
            {isLoadingAgents ? <div className="drawer-loading" aria-label="Loading compute agents"/> : isAgentsError ? <div className="drawer-error" role="status"><span>Compute connections could not be loaded.</span><button onClick={onRetryAgents}>Retry</button></div> : agents.length === 0 ? <p className="drawer-empty">No remote compute agents registered.</p> : agents.map(agent => {
              const online = agent.status === 'ONLINE' || agent.status === 'BUSY';
              const amd = agent.capabilities.rocm_detected && agent.capabilities.amd_gpu_count > 0;
              return <div className="compute-target" key={agent.agent_id}><span className={`compute-mark ${online && amd ? 'compute-mark-on' : ''}`}/><div className="target-copy"><strong>{agent.name}</strong><span>{amd ? `${agent.capabilities.gpu_names.join(', ') || 'AMD GPU'} · ROCm execution` : 'Preparation only'}</span><span className="target-meta">{agent.hostname} · Last seen {new Date(agent.last_seen).toLocaleString()}</span></div><span className={`target-status ${online ? 'target-status-live' : ''}`}>{AGENT_STATUS_COPY[agent.status]}</span><details className="technical-details"><summary>Technical details</summary><p>{agent.capabilities.capabilities.join(', ') || 'No capabilities reported'} · <code>{agent.agent_id}</code></p></details></div>;
            })}
            <details className="technical-details service-technical"><summary><Activity size={13}/> Service details</summary><p>{isError ? 'Service status unavailable' : `Service ${health?.status || 'checking'} · ${health?.host_platform.os || 'unknown'} · ROCm ${health?.system.rocm_version || 'not detected'} · PyTorch ${health?.system.torch_version || 'not detected'}`}</p></details>
          </aside>
        </>}
      </div>
    </div>
  </header>;
};
