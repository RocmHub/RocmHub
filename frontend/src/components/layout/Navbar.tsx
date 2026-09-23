import React from 'react';
import type { HealthResponse } from '../../api/types';
import { Cpu, AlertTriangle } from 'lucide-react';

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
    <header className="relative h-12 border-b border-surface-border bg-surface/95 backdrop-blur-md flex items-center justify-between px-5 sm:px-7 sticky top-0 z-40 shrink-0">
      {/* Ambient accent line */}
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-accent-red/60 to-transparent" />

      {/* Brand */}
      <div className="flex items-center gap-3">
        {onToggleMobileMenu && (
          <button
            onClick={onToggleMobileMenu}
            aria-label="Toggle navigation menu"
            className="md:hidden p-1.5 rounded-lg bg-surface-elevated text-content-secondary hover:text-content-primary border border-surface-border transition-colors"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              {isMobileMenuOpen
                ? <><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></>
                : <><line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="12" x2="21" y2="12" /><line x1="3" y1="18" x2="21" y2="18" /></>
              }
            </svg>
          </button>
        )}

        {/* Logo mark */}
        <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-accent-red to-red-800 flex items-center justify-center font-black text-white text-[11px] tracking-wider shrink-0 shadow-md shadow-red-950/60 glow-red-sm">
          RH
        </div>

        <div className="flex items-center gap-2.5">
          <span className="font-bold text-content-primary tracking-tight text-[15px]">ROCmHub</span>
          <span className="hidden sm:inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider bg-accent-red/10 text-red-400 border border-red-500/20">
            v0.1.0
          </span>
        </div>
      </div>

      {/* Right: hardware status only */}
      <div className="flex items-center gap-3 text-xs">
        {/* Connection dot */}
        <div className="flex items-center gap-1.5">
          {isLoading ? (
            <span className="w-1.5 h-1.5 rounded-full bg-zinc-500 animate-pulse" />
          ) : isError || !health ? (
            <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
          ) : (
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
          )}
          <span className="hidden sm:inline text-content-muted font-mono text-[11px]">
            {isLoading ? 'Connecting' : isError || !health ? 'Offline' : 'Online'}
          </span>
        </div>

        {/* Hardware status pill */}
        {health && (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-surface-elevated border border-surface-border">
            {health.rocm_available ? (
              <>
                <Cpu className="w-3.5 h-3.5 text-emerald-400" />
                <span className="text-emerald-400 font-medium text-[11px] font-mono">
                  {health.system.gpus[0]?.gfx_target || 'AMD GPU'}
                </span>
              </>
            ) : (
              <>
                <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                <span className="text-amber-400 font-medium text-[11px] hidden sm:inline" title={health.warnings[0]}>
                  CONFIG ONLY
                </span>
              </>
            )}
          </div>
        )}
      </div>
    </header>
  );
};
