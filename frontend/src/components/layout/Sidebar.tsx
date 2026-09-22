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
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, onTabChange }) => {
  const navItems = [
    { id: 'dashboard' as NavTab, label: 'Dashboard', icon: LayoutDashboard },
    { id: 'explorer' as NavTab, label: 'Model Explorer', icon: Search },
    { id: 'forge' as NavTab, label: 'Forge Studio', icon: Hammer },
    { id: 'engineer' as NavTab, label: 'AI Engineer', icon: Bot },
    { id: 'optimization' as NavTab, label: 'Optimization Lab', icon: Gauge },
  ];

  return (
    <aside className="w-56 border-r border-surface-border bg-surface flex flex-col justify-between p-3 shrink-0">
      <nav className="space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onTabChange(item.id)}
              className={`w-full flex items-center space-x-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-surface-elevated text-content-primary border-l-2 border-accent-red'
                  : 'text-content-secondary hover:text-content-primary hover:bg-surface-elevated/50'
              }`}
            >
              <Icon className={`w-4 h-4 ${isActive ? 'text-accent-red' : 'text-content-muted'}`} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="p-3 rounded-lg bg-surface-elevated/40 border border-surface-border text-[11px] text-zinc-500 font-mono space-y-1">
        <div className="text-zinc-400 font-semibold uppercase">Platform Mode</div>
        <div>Localhost Isolated</div>
        <div className="text-zinc-600">No synthetic GPU metrics</div>
      </div>
    </aside>
  );
};
