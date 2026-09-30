import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';

const stylesheet = readFileSync('src/index.css', 'utf8');

describe('fixed light appearance', () => {
  it('sets a light color scheme without OS preference or theme-switching branches', () => {
    expect(stylesheet).toMatch(/color-scheme\s*:\s*light/);
    expect(stylesheet).not.toMatch(/prefers-color-scheme|data-theme|theme-control/i);
  });
});
