import React from 'react';
import {
  LayoutDashboard,
  Search,
  Hammer,
  Bot,
  Gauge,
} from 'lucide-react';

export type NavTab = 'dashboard' | 'explorer' | 'forge' | 'engineer' | 'optimization';

interface SidebarProps {
  activeTab: NavTab;
  onTabChange: (tab: NavTab) => void;
  isOpenMobile?: boolean;
  onCloseMobile?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onTabChange,
  isOpenMobile = false,
  onCloseMobile,
}) => {
  const navItems = [
    { id: 'dashboard' as NavTab, label: 'Dashboard', icon: LayoutDashboard },
    { id: 'explorer' as NavTab, label: 'Model Explorer', icon: Search },
    { id: 'forge' as NavTab, label: 'Forge Studio', icon: Hammer },
    { id: 'engineer' as NavTab, label: 'AI Engineer', icon: Bot },
    { id: 'optimization' as NavTab, label: 'Optimization Lab', icon: Gauge },
  ];

  const handleTabClick = (tab: NavTab) => {
    onTabChange(tab);
    onCloseMobile?.();
  };

  const content = (
    <div className="w-56 border-r border-surface-border bg-surface flex flex-col justify-between p-3 h-full shrink-0">
      <nav className="space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => handleTabClick(item.id)}
              className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-md text-sm font-medium transition-all ${
                isActive
                  ? 'bg-gradient-to-r from-accent-red/20 via-surface-elevated to-transparent text-white border-l-2 border-accent-red font-semibold shadow-sm'
                  : 'text-content-secondary hover:text-content-primary hover:bg-surface-elevated/60'
              }`}
            >
              <Icon className={`w-4 h-4 transition-colors ${isActive ? 'text-accent-red' : 'text-zinc-500'}`} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="p-3 rounded-lg bg-surface-elevated/50 border border-surface-border text-[11px] text-zinc-400 font-mono space-y-1.5 shadow-inner">
        <div className="flex items-center space-x-1.5 text-zinc-300 font-semibold text-[10px] tracking-wider uppercase">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          <span>Truthful Engine</span>
        </div>
        <div className="text-zinc-500 text-[10px] leading-tight">
          Fail-closed preflight active. Zero synthetic metrics.
        </div>
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop static sidebar */}
      <aside className="hidden md:flex h-full shrink-0">
        {content}
      </aside>

      {/* Mobile drawer with backdrop */}
      {isOpenMobile && (
        <div className="fixed inset-0 z-50 md:hidden flex">
          <div
            className="fixed inset-0 bg-black/60 backdrop-blur-sm transition-opacity"
            onClick={onCloseMobile}
          />
          <div className="relative z-10 h-full shadow-xl">
            {content}
          </div>
        </div>
      )}
    </>
  );
};
