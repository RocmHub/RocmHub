import React, { useEffect, useRef, useState } from 'react';
import type { AgentInfo, HealthResponse } from '../../api/types';
import { Activity, ChevronDown } from 'lucide-react';
import { AGENT_STATUS_COPY, hasConnectedAmdCompute } from '../../ui/presentation';
import { Sidebar, type NavTab } from './Sidebar';

interface NavbarProps {
  activeTab: NavTab; onTabChange: (tab: NavTab) => void;
  health: HealthResponse | null; isLoading: boolean; isError: boolean; agents?: AgentInfo[];
  isLoadingAgents?: boolean; isAgentsError?: boolean; onRetryAgents?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, onTabChange, health, isLoading, isError, agents = [], isLoadingAgents = false, isAgentsError = false, onRetryAgents }) => {
  const [open, setOpen] = useState(false);
  const computeButtonRef = useRef<HTMLButtonElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const drawerRef = useRef<HTMLElement>(null);
  useEffect(() => {
    if (!open) return;
    closeButtonRef.current?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setOpen(false);
        computeButtonRef.current?.focus();
        return;
      }
      if (event.key !== 'Tab') return;
      const controls = [...(drawerRef.current?.querySelectorAll<HTMLElement>('button:not([disabled]),a[href],input:not([disabled]),select:not([disabled]),summary,[tabindex]:not([tabindex="-1"])') ?? [])];
      if (!controls.length) return;
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [open]);
  const connected = hasConnectedAmdCompute(health, agents);
  const checking = isLoading || isLoadingAgents;
  const unknown = !checking && (isError || isAgentsError || !health);
  const label = checking ? 'Checking compute' : connected ? 'AMD compute connected' : unknown ? 'Compute status unavailable' : 'No AMD compute';
  return <header className="product-header">
    <div className="product-header-inner">
      <a className="brand-lockup" href="#dashboard" aria-label="ROCmHub home" onClick={() => window.dispatchEvent(new HashChangeEvent('hashchange'))}>
        <span className="brand-symbol" aria-hidden="true"><i/><i/></span><span className="brand-name">ROCmHub</span>
      </a>
      <div className="header-nav"><Sidebar activeTab={activeTab} onTabChange={onTabChange}/></div>
      <div className="header-end">
      <div className="relative">
        <button ref={computeButtonRef} aria-controls="compute-panel" aria-haspopup="dialog" aria-expanded={open} aria-label={label} onClick={() => setOpen(v => !v)} className="compute-indicator">
          <span className={`compute-mark ${connected ? 'compute-mark-on' : isLoading || isLoadingAgents ? 'compute-mark-wait' : ''}`}/><span>{label}</span><ChevronDown size={13}/>
        </button>
        {open && <>
          <button className="drawer-scrim" aria-label="Close compute panel" onClick={() => setOpen(false)}/>
          <aside id="compute-panel" ref={drawerRef} role="dialog" aria-modal="true" aria-label="Compute" className="compute-drawer">
            <div className="drawer-heading"><div><div className="drawer-kicker">ROCmHub</div><h2>Compute</h2></div><button ref={closeButtonRef} aria-label="Close compute panel" className="icon-button" onClick={() => { setOpen(false); computeButtonRef.current?.focus(); }}>×</button></div>
            <p className="drawer-intro">{unknown ? 'Connection status could not be confirmed. Retry or check the service before assuming no compute is available.' : connected ? 'Available compute targets for preparation and AMD execution.' : 'No AMD compute is available right now. Model inspection and configuration preparation still work.'}</p>
            {!isLoading && !isError && health && <div className="compute-target"><span className={`compute-mark ${health.rocm_available ? 'compute-mark-on' : ''}`}/><div className="target-copy"><strong>{health.rocm_available ? health.system.gpus.find(gpu => gpu.gpu_present && gpu.vendor.toLowerCase() === 'amd')?.device_name || 'Local AMD compute' : 'Service host'}</strong><span>{health.rocm_available ? 'AMD runtime detected' : 'No AMD runtime detected here'}</span></div><span className="target-status">{health.rocm_available ? 'Available' : '—'}</span>{health.rocm_available && <details className="technical-details"><summary>Technical details</summary><p>{health.system.gpus.filter(gpu => gpu.gpu_present && gpu.vendor.toLowerCase() === 'amd').map(gpu => `${gpu.device_name}${gpu.vram_total_mb ? ` · ${Math.round(gpu.vram_total_mb / 1024)} GB` : ''}${gpu.gfx_target ? ` · ${gpu.gfx_target}` : ''}`).join(' · ') || 'GPU details were not reported'} · ROCm {health.system.rocm_version || 'unknown'} · PyTorch {health.system.torch_version || 'unknown'}</p></details>}</div>}
            {isLoadingAgents ? <div className="drawer-loading" aria-label="Loading compute connections"/> : isAgentsError ? <div className="drawer-error" role="status"><span>Compute connections could not be loaded.</span><button onClick={onRetryAgents}>Retry</button></div> : agents.length === 0 ? <p className="drawer-empty">No remote compute connections are available.</p> : agents.map(agent => {
              const online = agent.status === 'ONLINE' || agent.status === 'BUSY';
              const amd = agent.capabilities.rocm_detected && agent.capabilities.amd_gpu_count > 0;
              return <div className="compute-target" key={agent.agent_id}><span className={`compute-mark ${online && amd ? 'compute-mark-on' : ''}`}/><div className="target-copy"><strong>{amd ? agent.capabilities.gpu_names.join(', ') || agent.name : agent.name}</strong><span>{amd ? online ? 'AMD compute available' : 'AMD GPU reported' : 'AMD GPU not reported'}</span></div><span className={`target-status ${online ? 'target-status-live' : ''}`}>{AGENT_STATUS_COPY[agent.status]}</span><details className="technical-details"><summary>Technical details</summary><p>{agent.hostname} · Last seen {new Date(agent.last_seen).toLocaleString()} · ROCm detected: {agent.capabilities.rocm_detected ? 'yes' : 'no'} · {agent.capabilities.capabilities.join(', ') || 'No capabilities reported'} · <code>{agent.agent_id}</code></p></details></div>;
            })}
            <details className="technical-details service-technical"><summary><Activity size={13}/> Service details</summary><p>{isError ? 'Service status unavailable' : `Service ${health?.status || 'checking'} · ${health?.host_platform.os || 'unknown'} · ROCm ${health?.system.rocm_version || 'not detected'} · PyTorch ${health?.system.torch_version || 'not detected'}`}</p></details>
          </aside>
        </>}
      </div>
      </div>
    </div>
  </header>;
};
