/**
 * Full product-state UX acceptance for ROCmHub.
 * Exercises real backend flows and captures idle, loading, running, result,
 * failure, recovery, and mobile states without mocking product data.
 */
import puppeteer from 'puppeteer-core';
import { spawn } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';

const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const FRONTEND_DIR = path.join(PROJECT_ROOT, 'frontend');
const BACKEND_PORT = Number(process.env.ROCMHUB_BACKEND_PORT || 8780);
const FRONTEND_PORT = Number(process.env.ROCMHUB_FRONTEND_PORT || 5180);
const BASE = `http://127.0.0.1:${FRONTEND_PORT}`;
const OUT = path.resolve(process.env.ROCMHUB_SCREENSHOT_DIR || path.join(PROJECT_ROOT, 'screenshots', 'acceptance'));
const VALID_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct';
const INVALID_MODEL = 'rocmhub-ux/no-such-model-acceptance';

function findPython() {
  const configured = process.env.ROCMHUB_PYTHON || process.env.PYTHON;
  if (configured) return configured;
  const venv = process.env.VIRTUAL_ENV;
  const localVenv = path.join(PROJECT_ROOT, '.venv');
  const candidates = [
    venv && path.join(venv, process.platform === 'win32' ? 'Scripts' : 'bin', process.platform === 'win32' ? 'python.exe' : 'python3'),
    path.join(localVenv, process.platform === 'win32' ? 'Scripts' : 'bin', process.platform === 'win32' ? 'python.exe' : 'python3'),
  ].filter(Boolean);
  return candidates.find((candidate) => fs.existsSync(candidate)) || (process.platform === 'win32' ? 'python' : 'python3');
}

function requireChrome() {
  const configured = process.env.CHROME_PATH || process.env.PUPPETEER_EXECUTABLE_PATH;
  if (configured && fs.existsSync(configured)) return configured;
  throw new Error('Set CHROME_PATH or PUPPETEER_EXECUTABLE_PATH to a Chrome/Chromium executable.');
}

const PYTHON = findPython();
const CHROME = requireChrome();
const TEMP_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'rocmhub-ux-acceptance-'));
const DB = path.join(TEMP_DIR, 'jobs.db');
const WORKSPACE = path.join(TEMP_DIR, 'workspace');

fs.mkdirSync(OUT, { recursive: true });
fs.mkdirSync(WORKSPACE, { recursive: true });

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function waitForHttp(url, timeout = 25000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    try { if ((await fetch(url)).ok) return; } catch { /* keep waiting */ }
    await sleep(250);
  }
  throw new Error(`Timed out waiting for ${url}`);
}

async function waitForText(page, text, timeout = 45000) {
  await page.waitForFunction(
    value => (document.body.innerText || '').toLowerCase().includes(value.toLowerCase()),
    { timeout },
    text,
  );
}

async function clickText(page, text) {
  const clicked = await page.evaluate(value => {
    const lower = value.toLowerCase();
    const candidates = [...document.querySelectorAll('button')];
    const target = candidates.find(el => (el.textContent || '').toLowerCase().includes(lower) && !el.disabled);
    if (!target) return false;
    target.click();
    return true;
  }, text);
  if (!clicked) throw new Error(`Button not found: ${text}`);
}

async function fill(page, selector, value) {
  await page.$eval(selector, (input, nextValue) => {
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
    setter.call(input, nextValue);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
}

async function scrollToText(page, text) {
  await page.evaluate(value => {
    const lower = value.toLowerCase();
    const node = [...document.querySelectorAll('h1,h2,h3,h4,div,span')]
      .find(el => (el.textContent || '').trim().toLowerCase() === lower);
    node?.scrollIntoView({ block: 'center', inline: 'start', behavior: 'instant' });
    const main = document.querySelector('main');
    if (main) main.scrollLeft = 0;
    document.documentElement.scrollLeft = 0;
  }, text);
  await sleep(350);
}

async function shot(page, name, fullPage = false) {
  const target = path.join(OUT, name);
  await page.screenshot({ path: target, fullPage });
  console.log(`  ✓ ${name}`);
}

async function main() {
  const backend = spawn(PYTHON, [
    '-m', 'uvicorn', 'rocmhub.server.app:create_app', '--factory',
    '--host', '127.0.0.1', '--port', String(BACKEND_PORT),
  ], {
    cwd: PROJECT_ROOT,
    env: { ...process.env, ROCMHUB_DB_PATH: DB, ROCMHUB_ALLOWED_WORKSPACES: `${WORKSPACE},${TEMP_DIR}` },
    stdio: 'pipe',
  });
  const frontend = spawn(process.execPath, [
    'node_modules/vite/bin/vite.js', '--port', String(FRONTEND_PORT), '--host', '127.0.0.1',
  ], {
    cwd: FRONTEND_DIR,
    env: { ...process.env, VITE_BACKEND_PORT: String(BACKEND_PORT) },
    stdio: 'pipe',
  });

  let browser;
  const jobIds = {};
  try {
    await waitForHttp(`http://127.0.0.1:${BACKEND_PORT}/health`);
    await waitForHttp(BASE);
    browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new', args: ['--no-sandbox', '--disable-dev-shm-usage'] });
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900 });
    await page.setRequestInterception(true);
    page.on('request', request => {
      const url = request.url();
      const needsVisibleLoadingState = url.includes('/api/v1/models/') || url.includes('/api/v1/forge/plan');
      const needsVisibleSubmitState = url.endsWith('/api/v1/jobs') && request.method() === 'POST';
      if (needsVisibleLoadingState || needsVisibleSubmitState) {
        setTimeout(() => request.continue(), needsVisibleLoadingState ? 1200 : 650);
      } else {
        request.continue();
      }
    });
    page.on('response', async response => {
      if (!response.url().endsWith('/api/v1/jobs') || response.request().method() !== 'POST') return;
      try {
        const data = await response.json();
        if (data.job_type && data.job_id) jobIds[data.job_type] = data.job_id;
      } catch { /* ignore non-JSON */ }
    });

    console.log('Model Explorer: idle → input → loading → failure → retry → success');
    await page.goto(`${BASE}/#explorer`, { waitUntil: 'networkidle2' });
    await waitForText(page, 'Curated starting points');
    await shot(page, '01_explorer_idle.png');
    await fill(page, 'input[aria-label="Model ID"]', INVALID_MODEL);
    await shot(page, '02_explorer_input.png');
    await clickText(page, 'Inspect model');
    await waitForText(page, 'Resolving model identity');
    await shot(page, '03_explorer_loading.png');
    await waitForText(page, 'We couldn’t inspect this model');
    await shot(page, '04_explorer_failure_toast_retry.png');
    await fill(page, 'input[aria-label="Model ID"]', VALID_MODEL);
    await clickText(page, 'Inspect model');
    await waitForText(page, 'Model profile');
    await shot(page, '05_explorer_success.png');

    console.log('Forge: every wizard step → planning → running → output → failure → recovery');
    await clickText(page, 'Review in Forge');
    await waitForText(page, 'Choose the model');
    await shot(page, '06_forge_step_model.png');
    await clickText(page, 'Choose target');
    await waitForText(page, 'Choose the hardware target');
    await shot(page, '07_forge_step_target.png');
    await clickText(page, 'MI300X');
    await clickText(page, 'Choose profile');
    await waitForText(page, 'Choose a preparation profile');
    await shot(page, '08_forge_step_profile.png');
    await clickText(page, 'Native CDNA');
    await clickText(page, 'Review build plan');
    await waitForText(page, 'Resolving recipe');
    await shot(page, '09_forge_loading_plan.png');
    await waitForText(page, 'Your build plan is ready');
    await shot(page, '10_forge_review.png');
    await clickText(page, 'Build artifact');
    await waitForText(page, 'Building your artifact');
    await shot(page, '11_forge_running.png');
    await waitForText(page, 'Artifact ready');
    await shot(page, '12_forge_success_artifacts.png');
    await clickText(page, 'Build another');
    await fill(page, 'input[placeholder="org/model-name"]', INVALID_MODEL);
    await clickText(page, 'Choose target');
    await clickText(page, 'Choose profile');
    await clickText(page, 'Review build plan');
    await waitForText(page, 'Build needs attention');
    await shot(page, '13_forge_failure_retry.png');
    await clickText(page, 'Edit model');
    await fill(page, 'input[placeholder="org/model-name"]', VALID_MODEL);
    await clickText(page, 'Choose target');
    await clickText(page, 'Choose profile');
    await clickText(page, 'Review build plan');
    await waitForText(page, 'Your build plan is ready');
    await shot(page, '14_forge_retry_recovered.png');

    console.log('AI Engineer: idle → running activity → failure → retry → executive result');
    await clickText(page, 'Engineer');
    await waitForText(page, 'What do you want to accomplish?');
    await shot(page, '15_engineer_idle.png');
    await fill(page, 'input[placeholder="org/model-name"]', INVALID_MODEL);
    await clickText(page, 'Start Session');
    await waitForText(page, 'Engineer at work');
    await shot(page, '16_engineer_running_activity.png');
    await waitForText(page, 'We couldn’t complete this session');
    await scrollToText(page, 'We couldn’t complete this session');
    await shot(page, '17_engineer_failure_retry.png');
    await fill(page, 'input[placeholder="org/model-name"]', VALID_MODEL);
    await clickText(page, 'Start Session');
    await waitForText(page, 'Building a recommendation');
    await shot(page, '18_engineer_retry_running.png');
    await waitForText(page, 'Recommendation ready');
    await scrollToText(page, 'Recommendation ready');
    await shot(page, '19_engineer_success_executive.png');

    console.log('Optimization: idle comparison → running lanes → failure → retry → result');
    await clickText(page, 'Optimize');
    await waitForText(page, 'Baseline versus candidates');
    await shot(page, '20_optimization_idle_comparison.png');
    await fill(page, 'input[placeholder="org/model-name"]', INVALID_MODEL);
    await clickText(page, 'Run Optimization');
    await waitForText(page, 'Experiment running');
    await shot(page, '21_optimization_running_lanes.png');
    await waitForText(page, 'No comparison was produced');
    await scrollToText(page, 'No comparison was produced');
    await shot(page, '22_optimization_failure_retry.png');
    await fill(page, 'input[placeholder="org/model-name"]', VALID_MODEL);
    await clickText(page, 'Run Optimization');
    await waitForText(page, 'Experiment running');
    await shot(page, '23_optimization_retry_running.png');
    await waitForText(page, 'Candidate Comparison Table');
    await scrollToText(page, 'Candidate Comparison Table');
    await shot(page, '24_optimization_success_comparison.png');

    console.log('Mobile result acceptance');
    await page.setViewport({ width: 390, height: 844, isMobile: true, hasTouch: true });
    await page.goto(`${BASE}/?job_id=${jobIds.OPTIMIZATION}#optimization`, { waitUntil: 'networkidle2' });
    await waitForText(page, 'Candidate Comparison Table');
    await scrollToText(page, 'Candidate Comparison Table');
    await shot(page, '25_mobile_optimization_result.png');
    for (const [kind, hash, file, expected] of [
      ['FORGE_BUILD', 'forge', '26_mobile_forge_output.png', 'Artifact ready'],
      ['ENGINEER', 'engineer', '27_mobile_engineer_result.png', 'Recommendation ready'],
    ]) {
      if (!jobIds[kind]) continue;
      await page.goto(`${BASE}/?job_id=${jobIds[kind]}#${hash}`, { waitUntil: 'networkidle2' });
      await waitForText(page, expected);
      await scrollToText(page, expected);
      await shot(page, file);
    }
    await page.goto(`${BASE}/#explorer`, { waitUntil: 'networkidle2' });
    await clickText(page, 'Inspect model');
    await waitForText(page, 'Model profile');
    await scrollToText(page, 'Model profile');
    await shot(page, '28_mobile_explorer_result.png');

    console.log(`UX acceptance complete: ${Object.keys(jobIds).length} workflow job types, ${fs.readdirSync(OUT).length} screenshots.`);
  } finally {
    if (browser) await browser.close();
    backend.kill();
    frontend.kill();
    await sleep(400);
    fs.rmSync(TEMP_DIR, { recursive: true, force: true });
  }
}

main().catch(error => { console.error(error); process.exit(1); });
