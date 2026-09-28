import { describe, expect, it } from 'vitest';
import { persistThemePreference, readThemePreference, resolveTheme, THEME_STORAGE_KEY } from '../ui/theme';

describe('theme preference', () => {
  it('defaults to System and follows the operating system preference', () => {
    const storage = { getItem: () => null };
    expect(readThemePreference(storage)).toBe('system');
    expect(resolveTheme('system', true)).toBe('light');
    expect(resolveTheme('system', false)).toBe('dark');
  });

  it('uses an explicit Light or Dark selection and persists it', () => {
    const values = new Map<string, string>();
    const storage = {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => values.set(key, value),
    };
    persistThemePreference('light', storage);
    expect(values.get(THEME_STORAGE_KEY)).toBe('light');
    expect(readThemePreference(storage)).toBe('light');
    expect(resolveTheme('light', false)).toBe('light');
    persistThemePreference('dark', storage);
    expect(readThemePreference(storage)).toBe('dark');
    expect(resolveTheme('dark', true)).toBe('dark');
  });

  it('keeps explicit Light when the system prefers Dark', () => {
    expect(resolveTheme('light', false)).toBe('light');
  });

  it('keeps explicit Dark when the system prefers Light', () => {
    expect(resolveTheme('dark', true)).toBe('dark');
  });

  it('falls back to the system preference if browser storage is unavailable', () => {
    expect(readThemePreference({ getItem: () => { throw new Error('Storage blocked'); } })).toBe('system');
    expect(() => persistThemePreference('light', { setItem: () => { throw new Error('Storage blocked'); } })).not.toThrow();
  });
});
