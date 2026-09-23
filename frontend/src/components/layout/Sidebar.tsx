import React from 'react';
import { LayoutGrid, Search, PackageCheck, BrainCircuit, SlidersHorizontal } from 'lucide-react';

export type NavTab = 'dashboard' | 'explorer' | 'forge' | 'engineer' | 'optimization';

interface SidebarProps {
  activeTab: NavTab;
  onTabChange: (tab: NavTab) => void;
  isOpenMobile?: boolean;
  onCloseMobile?: () => void;
}

const NAV_SECTIONS = [
  {
    label: 'Platform',
    items: [
      { id: 'dashboard' as NavTab, label: 'Home', icon: LayoutGrid },
      { id: 'explorer' as NavTab, label: 'Model Explorer', icon: Search },
    ],
  },
  {
    label: 'Workflows',
    items: [
      { id: 'forge' as NavTab, label: 'Forge Studio', icon: PackageCheck },
      { id: 'engineer' as NavTab, label: 'AI Engineer', icon: BrainCircuit },
      { id: 'optimization' as NavTab, label: 'Optimization Lab', icon: SlidersHorizontal },
    ],
  },
];

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onTabChange,
  isOpenMobile = false,
  onCloseMobile,
}) => {
  const handleTabClick = (tab: NavTab) => {
    onTabChange(tab);
    onCloseMobile?.();
  };

  const content = (
    <div className="w-[220px] h-full flex flex-col bg-surface border-r border-surface-border shrink-0">
      {/* Nav sections */}
      <nav className="flex-1 px-3 py-4 space-y-5 overflow-y-auto">
        {NAV_SECTIONS.map((section) => (
          <div key={section.label}>
            <div className="px-3 mb-1.5 section-label">{section.label}</div>
            <div className="space-y-0.5">
              {section.items.map((item) => {
                const Icon = item.icon;
                const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => handleTabClick(item.id)}
                    className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all relative ${
                      isActive
                        ? 'bg-accent-red/10 text-content-primary'
                        : 'text-content-secondary hover:text-content-primary hover:bg-surface-elevated'
                    }`}
                  >
                    {/* Left accent bar */}
                    {isActive && (
                      <span className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 bg-accent-red rounded-r-full" />
                    )}
                    <Icon
                      className={`w-[18px] h-[18px] shrink-0 transition-colors ${
                        isActive ? 'text-accent-red' : 'text-content-muted'
                      }`}
                    />
                    <span className={isActive ? 'font-semibold' : ''}>{item.label}</span>
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      {/* Bottom badge */}
      <div className="px-4 py-3 border-t border-surface-border">
        <div className="flex items-center gap-2 text-[11px] text-content-muted">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0" />
          <span>Truthful metrics only</span>
        </div>
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop */}
      <aside className="hidden md:flex h-full shrink-0">{content}</aside>

      {/* Mobile drawer */}
      {isOpenMobile && (
        <div className="fixed inset-0 z-50 md:hidden flex">
          <div
            className="fixed inset-0 bg-black/70 backdrop-blur-sm"
            onClick={onCloseMobile}
          />
          <div className="relative z-10 h-full shadow-2xl">
            {content}
          </div>
        </div>
      )}
    </>
  );
};
