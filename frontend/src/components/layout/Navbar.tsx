import React from 'react';
import type { HealthResponse } from '../../api/types';
import { Cpu, AlertTriangle, Layers, Menu, X } from 'lucide-react';

interface NavbarProps {
  health: HealthResponse | null;
  isLoading: boolean;
  isError: boolean;
  isMobileMenuOpen?: boolean;
  onToggleMobileMenu?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  health,
  isLoading,
  isError,
  isMobileMenuOpen,
  onToggleMobileMenu,
}) => {
  return (
    <header className="h-14 border-b border-surface-border bg-surface flex items-center justify-between px-4 sm:px-6 sticky top-0 z-40">
      {/* Brand & Mobile Toggle */}
      <div className="flex items-center space-x-3">
        {onToggleMobileMenu && (
          <button
            onClick={onToggleMobileMenu}
            aria-label="Toggle navigation menu"
            className="md:hidden p-1.5 rounded bg-surface-elevated text-zinc-300 hover:text-white border border-surface-border"
          >
            {isMobileMenuOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
          </button>
        )}
        <div className="w-7 h-7 rounded bg-accent-red flex items-center justify-center font-bold text-white text-sm tracking-wider font-mono shrink-0">
          RH
        </div>
        <div className="flex items-baseline space-x-2">
          <span className="font-semibold text-content-primary tracking-tight">ROCmHub</span>
          <span className="hidden sm:inline text-[10px] font-mono text-zinc-500 uppercase tracking-wider">Engineering Platform</span>
        </div>
      </div>

      {/* Hardware & Backend Status */}
      <div className="flex items-center space-x-4 text-xs font-mono">
        {/* Backend health pill */}
        <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-surface-elevated border border-surface-border">
          {isLoading ? (
            <>
              <span className="w-2 h-2 rounded-full bg-zinc-500 animate-pulse" />
              <span className="text-zinc-400">Connecting...</span>
            </>
          ) : isError || !health ? (
            <>
              <span className="w-2 h-2 rounded-full bg-red-500" />
              <span className="text-red-400">Backend Offline</span>
            </>
          ) : (
            <>
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <span className="text-zinc-300">API v{health.version}</span>
            </>
          )}
        </div>

        {/* Hardware Status Pill */}
        {health && (
          <div className="flex items-center space-x-2 px-2.5 py-1 rounded bg-surface-elevated border border-surface-border">
            <Cpu className="w-3.5 h-3.5 text-zinc-400" />
            {health.rocm_available ? (
              <span className="text-emerald-400">
                ROCm GPU: {health.system.gpus[0]?.gfx_target || health.system.gpus[0]?.device_name || 'Active'}
              </span>
            ) : (
              <span className="text-amber-400 flex items-center space-x-1" title={health.warnings[0] || 'Running in CONFIG_ONLY mode'}>
                <AlertTriangle className="w-3 h-3 text-amber-400 inline" />
                <span>No AMD ROCm GPU (CONFIG_ONLY mode)</span>
              </span>
            )}
          </div>
        )}

        {/* Queue telemetry */}
        {health?.orchestrator && (
          <div className="hidden md:flex items-center space-x-2 px-2.5 py-1 rounded bg-surface-elevated border border-surface-border text-zinc-400">
            <Layers className="w-3.5 h-3.5 text-zinc-400" />
            <span>Queue: {health.orchestrator.queue_size}</span>
          </div>
        )}
      </div>
    </header>
  );
};
