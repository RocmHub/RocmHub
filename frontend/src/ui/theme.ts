export type ThemePreference = 'system' | 'light' | 'dark';
export const THEME_STORAGE_KEY = 'rocmhub.theme';

export function readThemePreference(storage?: Pick<Storage, 'getItem'>): ThemePreference {
  try {
    const saved = (storage ?? window.localStorage).getItem(THEME_STORAGE_KEY);
    return saved === 'light' || saved === 'dark' ? saved : 'system';
  } catch {
    return 'system';
  }
}

export function resolveTheme(preference: ThemePreference, systemPrefersLight: boolean): 'light' | 'dark' {
  return preference === 'system' ? (systemPrefersLight ? 'light' : 'dark') : preference;
}

export function persistThemePreference(preference: ThemePreference, storage?: Pick<Storage, 'setItem'>): void {
  try { (storage ?? window.localStorage).setItem(THEME_STORAGE_KEY, preference); } catch { /* Session-only preference is still applied in memory. */ }
}
