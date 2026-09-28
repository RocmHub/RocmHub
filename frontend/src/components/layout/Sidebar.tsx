import React from 'react';
import { Activity, Gauge, Github, Home, Search } from 'lucide-react';

export type NavTab = 'dashboard' | 'explorer' | 'runs' | 'forge' | 'engineer' | 'optimization';
interface SidebarProps {
  activeTab: NavTab;
  onTabChange: (tab: NavTab) => void;
  isOpenMobile?: boolean;
  onCloseMobile?: () => void;
}

const ITEMS = [
  { id: 'dashboard' as NavTab, label: 'Home', icon: Home },
  { id: 'explorer' as NavTab, label: 'Models', icon: Search },
  { id: 'runs' as NavTab, label: 'Runs', icon: Activity },
  { id: 'optimization' as NavTab, label: 'Optimize', icon: Gauge },
];

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, onTabChange, isOpenMobile = false, onCloseMobile }) => {
  const selectedTab = activeTab === 'forge' || activeTab === 'engineer' ? 'runs' : activeTab;
  const nav = <div className="workspace-sidebar">
    <div className="sidebar-caption">Workspace</div>
    <nav aria-label="Primary navigation" className="flex-1 space-y-1">
      {ITEMS.map(({ id, label, icon: Icon }) => <button
        key={id}
        aria-current={selectedTab === id ? 'page' : undefined}
        onClick={() => { onTabChange(id); onCloseMobile?.(); }}
        className={`sidebar-link ${selectedTab === id ? 'sidebar-link-active' : ''}`}
      ><Icon size={17} strokeWidth={1.8}/><span>{label}</span>{selectedTab === id && <span className="sidebar-active-mark"/>}</button>)}
    </nav>
    <div className="sidebar-footer"><div className="sidebar-footer-mark"><Github size={15}/></div><div><div className="text-xs font-medium text-zinc-300">Open infrastructure</div><div className="text-[10px] text-zinc-600 mt-1">Built for AMD compute</div></div></div>
  </div>;

  return <><aside className="hidden md:flex h-full">{nav}</aside>{isOpenMobile && <div className="fixed inset-0 z-50 md:hidden flex"><button aria-label="Close navigation" className="absolute inset-0 bg-black/75 backdrop-blur-sm" onClick={onCloseMobile}/><div className="relative h-full">{nav}</div></div>}</>;
};
