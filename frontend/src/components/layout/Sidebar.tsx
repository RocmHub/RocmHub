import React, { useState } from 'react';
import { Activity, ChevronDown, Gauge, Search, Wrench } from 'lucide-react';

export type NavTab = 'dashboard' | 'explorer' | 'runs' | 'forge' | 'engineer' | 'optimization';
interface SidebarProps { activeTab: NavTab; onTabChange: (tab: NavTab) => void; isOpenMobile?: boolean; onCloseMobile?: () => void }
const ITEMS = [
  { id: 'explorer' as NavTab, label: 'Models', icon: Search },
  { id: 'runs' as NavTab, label: 'Activity', icon: Activity },
  { id: 'optimization' as NavTab, label: 'Optimize', icon: Gauge },
];

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, onTabChange }) => {
  const [toolsOpen, setToolsOpen] = useState(false);
  const selected = activeTab === 'forge' || activeTab === 'engineer' ? 'tools' : activeTab === 'dashboard' ? '' : activeTab;
  return <nav aria-label="Primary navigation" className="primary-nav">
    {ITEMS.map(({ id, label, icon: Icon }) => <button key={id} aria-current={selected === id ? 'page' : undefined} onClick={() => onTabChange(id)} className={`primary-nav-link ${selected === id ? 'primary-nav-link-active' : ''}`}>
      <Icon size={15} strokeWidth={1.8}/><span>{label}</span>
    </button>)}
    <div className="nav-tools-wrap"><button aria-expanded={toolsOpen} onClick={() => setToolsOpen(value => !value)} className={`primary-nav-link nav-tools ${activeTab === 'forge' || activeTab === 'engineer' ? 'primary-nav-link-active' : ''}`}><Wrench size={14}/><span>Tools</span><ChevronDown size={12}/></button>{toolsOpen&&<div role="menu" className="tools-menu"><button role="menuitem" onClick={()=>{onTabChange('engineer');setToolsOpen(false)}}>AI Engineer<small>Guided model goals</small></button><button role="menuitem" onClick={()=>{onTabChange('forge');setToolsOpen(false)}}>Forge Studio<small>Advanced model preparation</small></button></div>}</div>
  </nav>;
};
