import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';

const stylesheet = readFileSync('src/index.css', 'utf8');
const entrypoint = readFileSync('index.html', 'utf8');

describe('fixed light appearance', () => {
  it('sets a light color scheme without OS preference or theme-switching branches', () => {
    expect(stylesheet).toMatch(/color-scheme\s*:\s*light/);
    expect(stylesheet).not.toMatch(/prefers-color-scheme|data-theme|theme-control/i);
    expect(entrypoint).toMatch(/<meta\s+name="color-scheme"\s+content="light"\s*\/>/i);
    expect(entrypoint).not.toMatch(/rocmhub\.theme|prefers-color-scheme|data-theme|colorScheme|class="dark"/i);
  });
});
