import type { NavTab } from '../components/layout/Sidebar';

const VALID_TABS: NavTab[] = ['dashboard', 'explorer', 'runs', 'forge', 'engineer', 'optimization'];

export function resolveInitialTab(hash: string, search: string): NavTab {
  const tab = hash.replace(/^#/, '').toLowerCase();
  if (VALID_TABS.includes(tab as NavTab)) return tab as NavTab;
  return new URLSearchParams(search).has('job_id') ? 'runs' : 'dashboard';
}
