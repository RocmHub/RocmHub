/**
 * Quick visual screenshot capture of all 5 ROCmHub views
 * against the already-running dev server on port 5173
 */
import puppeteer from 'puppeteer-core';
import path from 'path';
import fs from 'fs';

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const BASE_URL = 'http://127.0.0.1:5173';
const OUT_DIR = path.resolve('screenshots/phase22');

fs.mkdirSync(OUT_DIR, { recursive: true });

const VIEWS = [
  { hash: '',            label: 'dashboard',       file: '01_dashboard.png' },
  { hash: '#explorer',   label: 'explorer',        file: '02_explorer.png' },
  { hash: '#forge',      label: 'forge',           file: '03_forge.png' },
  { hash: '#engineer',   label: 'engineer',        file: '04_engineer.png' },
  { hash: '#optimization',label:'optimization',    file: '05_optimization.png' },
];

async function main() {
  console.log('Launching Chrome for Phase 22 screenshot capture...');
  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: 'new',
    args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900 });

    for (const view of VIEWS) {
      console.log(`  Capturing ${view.label}...`);
      await page.goto(`${BASE_URL}/${view.hash}`, { waitUntil: 'networkidle2', timeout: 15000 });
      await new Promise(r => setTimeout(r, 1200));
      const fp = path.join(OUT_DIR, view.file);
      await page.screenshot({ path: fp, fullPage: false });
      console.log(`  -> ${fp}`);
    }

    // Also capture Explorer with a model pre-filled
    console.log('  Capturing explorer with model result...');
    await page.goto(`${BASE_URL}/#explorer`, { waitUntil: 'networkidle2', timeout: 15000 });
    await new Promise(r => setTimeout(r, 800));
    // The default model is pre-filled, just click Inspect
    const buttons = await page.$$('button');
    for (const btn of buttons) {
      const text = await page.evaluate(el => el.textContent, btn);
      if (text && text.toLowerCase().includes('inspect')) {
        await btn.click();
        break;
      }
    }
    await new Promise(r => setTimeout(r, 4000));
    const inspectFp = path.join(OUT_DIR, '02b_explorer_result.png');
    await page.screenshot({ path: inspectFp, fullPage: false });
    console.log(`  -> ${inspectFp}`);

    console.log('\nAll screenshots captured successfully!');
  } finally {
    await browser.close();
  }
}

main().catch(err => { console.error(err); process.exit(1); });
