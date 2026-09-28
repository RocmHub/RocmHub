import React, { useState } from 'react';
import type { HealthResponse } from '../../api/types';
import { Activity, ChevronDown, Menu, X } from 'lucide-react';

interface NavbarProps {
  health: HealthResponse | null;
  isLoading: boolean;
  isError: boolean;
  isMobileMenuOpen?: boolean;
  onToggleMobileMenu?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ health, isLoading, isError, isMobileMenuOpen, onToggleMobileMenu }) => {
  const [details, setDetails] = useState(false);
  const online = !isLoading && !isError && !!health;
  return <header className="product-header">
    <div className="flex items-center gap-3">
      <button onClick={onToggleMobileMenu} aria-label="Toggle navigation menu" className="mobile-menu-button md:hidden">{isMobileMenuOpen ? <X size={17}/> : <Menu size={17}/>}</button>
      <div className="brand-mark" aria-hidden="true"><span/><span/></div>
      <div><div className="font-semibold tracking-[-.035em] leading-none">ROCmHub</div><div className="hidden sm:block text-[9px] uppercase tracking-[.2em] text-zinc-600 mt-1.5">Open model infrastructure</div></div>
    </div>
    <div className="relative">
      <button onClick={() => setDetails(value => !value)} aria-label={online ? 'Service online' : isLoading ? 'Checking service' : 'Service unavailable'} className="service-pill">
        <span className={`status-dot ${online ? 'status-dot-online' : isLoading ? 'status-dot-pending' : 'status-dot-error'}`}/>
        <span className="hidden sm:inline">{online ? 'Service online' : isLoading ? 'Connecting' : 'Service unavailable'}</span><ChevronDown size={13}/>
      </button>
      {details && <div className="service-popover">
        <div className="flex gap-3"><div className="service-popover-icon"><Activity size={17}/></div><div><div className="text-sm font-semibold">Service status</div><p className="text-xs text-zinc-500 mt-1 leading-relaxed">{online ? 'ROCmHub is responding. Hardware measurement requires capable AMD compute and prepared model files.' : 'ROCmHub is unavailable. Try again shortly.'}</p></div></div>
        <details className="mt-4 pt-3 border-t hairline"><summary className="text-[11px] text-zinc-500 cursor-pointer">Technical details</summary><div className="mt-2 text-[10px] font-mono text-zinc-600 space-y-1"><div>Version {health?.version || '—'}</div><div>{health?.host_platform.os || 'unknown'} · {health?.host_platform.arch || 'unknown'}</div><div>{health?.system.gpus_detected || 0} accelerator(s)</div><div>ROCm {health?.system.rocm_version || 'not detected'}</div><div>HIP runtime / PyTorch {health?.system.torch_version || 'not detected'}</div></div></details>
      </div>}
    </div>
  </header>;
};
