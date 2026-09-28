import React, { useEffect, useRef, useState } from 'react';
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
  const toolsButtonRef = useRef<HTMLButtonElement>(null);
  const toolsMenuRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!toolsOpen) return;
    const firstItem = toolsMenuRef.current?.querySelector<HTMLButtonElement>('[role="menuitem"]');
    firstItem?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setToolsOpen(false);
        toolsButtonRef.current?.focus();
        return;
      }
      if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
      const items = [...(toolsMenuRef.current?.querySelectorAll<HTMLButtonElement>('[role="menuitem"]') ?? [])];
      if (!items.length) return;
      event.preventDefault();
      const current = items.indexOf(document.activeElement as HTMLButtonElement);
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1 : (current + (event.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
      items[next].focus();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [toolsOpen]);
  const selected = activeTab === 'forge' || activeTab === 'engineer' ? 'tools' : activeTab === 'dashboard' ? '' : activeTab;
  return <nav aria-label="Primary navigation" className="primary-nav">
    {ITEMS.map(({ id, label, icon: Icon }) => <button key={id} aria-current={selected === id ? 'page' : undefined} onClick={() => onTabChange(id)} className={`primary-nav-link ${selected === id ? 'primary-nav-link-active' : ''}`}>
      <Icon size={15} strokeWidth={1.8}/><span>{label}</span>
    </button>)}
    <div className="nav-tools-wrap" onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setToolsOpen(false); }}><button ref={toolsButtonRef} aria-haspopup="menu" aria-controls="tools-menu" aria-expanded={toolsOpen} onClick={() => setToolsOpen(value => !value)} className={`primary-nav-link nav-tools ${activeTab === 'forge' || activeTab === 'engineer' ? 'primary-nav-link-active' : ''}`}><Wrench size={14}/><span>Tools</span><ChevronDown size={12}/></button>{toolsOpen&&<div id="tools-menu" ref={toolsMenuRef} role="menu" aria-label="Tools" className="tools-menu"><button role="menuitem" onClick={()=>{onTabChange('engineer');setToolsOpen(false)}}>AI Engineer<small>Guided workflow</small></button><button role="menuitem" onClick={()=>{onTabChange('forge');setToolsOpen(false)}}>Forge Studio<small>Advanced build workflow</small></button></div>}</div>
  </nav>;
};
