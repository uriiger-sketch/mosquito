'use strict';
/*
 * The production configuration of https://mosquito.charity, end to end: the
 * minified build from tools/build-site.sh, served by the real backend
 * (federated_server.py with SITE_DIR) and opened at the server's own origin in
 * headless Chromium, with a WAV file as the microphone. The page must run under
 * its Content-Security-Policy without a single violation or error, reach its
 * API on the same origin, detect a mosquito, and its service worker must cache
 * the app but never answer an API request. An admin session (the server's
 * ADMIN_PASSWORD typed into About → Advanced) must open the connected-devices
 * sheet from the top bar and the Server tab, and leave no trace once ended.
 *
 *   needs `npm ci` (the minifiers), Playwright + Chromium, and the server's
 *   Python dependencies (pip install -r requirements.txt)
 */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const net = require('net');
const http = require('http');
const path = require('path');
const { spawn, execFileSync } = require('child_process');
const { loadPlaywright, isolateNetwork } = require('./run-app');
const { FIXTURES, writeWav, SR, rng } = require('./make-wav');

const ROOT = path.join(__dirname, '..', '..');
const PY = process.env.PYTHON || 'python3';
const API_PATH = /^\/(?:(?:heartbeat|detection|hotspots|health|log|log\.txt)$|federated\/|admin\/)/;
const ADMIN_PW = 'e2e admin password, long enough';

function missingDeps() {
  try { execFileSync(PY, ['-c', 'import flask, numpy'], { stdio: 'ignore' }); } catch (e) { return 'server deps missing: pip install -r requirements.txt'; }
  try { require.resolve('terser', { paths: [ROOT] }); } catch (e) { return 'minifiers missing: npm ci'; }
  return false;
}
// Locally a missing dependency skips with a reason; in CI (CI=true) it must fail.
const SKIP = !process.env.CI && missingDeps();

function freePort() {
  return new Promise((resolve, reject) => {
    const s = net.createServer().once('error', reject).listen(0, '127.0.0.1', () => {
      const { port } = s.address();
      s.close(() => resolve(port));
    });
  });
}

function getJSON(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let body = '';
      res.on('data', (c) => { body += c; });
      res.on('end', () => { try { resolve(JSON.parse(body)); } catch (e) { reject(e); } });
    }).on('error', reject);
  });
}

async function until(fn, ms, what) {
  const t0 = Date.now();
  let last;
  while (Date.now() - t0 < ms) {
    try { last = await fn(); if (last) return last; } catch (e) { last = e; }
    await new Promise(r => setTimeout(r, 250));
  }
  throw new Error(`timed out after ${ms} ms waiting for ${what} (last: ${last && (last.message || JSON.stringify(last))})`);
}

test('built site served by the backend: CSP-clean, same-origin API, detection, service worker',
     { skip: SKIP, timeout: 180000 }, async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'mn-site-'));
  const port = await freePort();
  const origin = `http://127.0.0.1:${port}`;
  const site = path.join(tmp, 'site');
  execFileSync('bash', [path.join(ROOT, 'tools', 'build-site.sh'), site], {
    stdio: 'pipe',
    env: { ...process.env, SITE_RUN: '7', SITE_SHA: '0123456789abcdef0123456789abcdef01234567', SERVER_URLS: origin },
  });
  const base = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8').match(/^const APP_VERSION = '(\d+\.\d+)\.\d+';$/m)[1];
  const wav = path.join(tmp, 'aedes.wav');
  writeWav(wav, FIXTURES['aedes-628'].build(30 * SR, rng(628)));

  const log = [];
  const proc = spawn(PY, ['federated_server.py'], {
    cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, PORT: String(port), DATA_DIR: path.join(tmp, 'data'), SITE_DIR: site, ADMIN_PASSWORD: ADMIN_PW },
  });
  proc.stdout.on('data', (d) => log.push(String(d)));
  proc.stderr.on('data', (d) => log.push(String(d)));
  const { chromium } = loadPlaywright();
  let browser;
  try {
    await until(() => getJSON(`${origin}/health`).then(h => h.status === 'ok'), 20000, 'server');
    browser = await chromium.launch({ args: [
      '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
      `--use-file-for-fake-audio-capture=${wav}`, '--autoplay-policy=no-user-gesture-required',
    ] });
    const ctx = await browser.newContext({                   // service workers allowed here
      viewport: { width: 390, height: 844 }, permissions: ['microphone', 'geolocation'],
      geolocation: { latitude: FIXTURES['aedes-628'].loc.lat, longitude: FIXTURES['aedes-628'].loc.lng },
    });
    await isolateNetwork(ctx);
    await ctx.addInitScript(() => {
      window.__csp = [];
      document.addEventListener('securitypolicyviolation', (e) => window.__csp.push(`${e.violatedDirective} ${e.blockedURI}`));
      try { localStorage.setItem('mosquitonet_onboarded', '1'); } catch (e) {}
    });
    const page = await ctx.newPage();
    const errors = [], apiFromSw = [], appFromSw = [];
    page.on('pageerror', e => errors.push(`pageerror: ${e.message}`));
    page.on('console', m => {
      if (m.type() === 'error' && !/Failed to load resource|ERR_FAILED|net::/.test(m.text())) errors.push(`console: ${m.text()}`);
    });
    page.on('response', (r) => {
      const u = new URL(r.url());
      if (u.origin !== origin || !r.fromServiceWorker()) return;
      (API_PATH.test(u.pathname) ? apiFromSw : appFromSw).push(u.pathname);
    });

    await page.goto(`${origin}/`, { waitUntil: 'load' });
    await page.waitForFunction(() => window._app, null, { timeout: 15000 });
    assert.equal(await page.textContent('.about-foot'), `MosquitoNet v${base}.7`, 'stamped version');
    await page.waitForFunction(() => document.querySelector('#net-pill').dataset.net === 'online', null, { timeout: 15000 });

    // Second load: under the service worker, which serves the app but not the API.
    await page.evaluate(() => navigator.serviceWorker.ready.then(() => true));
    await page.reload({ waitUntil: 'load' });
    await page.waitForFunction(() => window._app && navigator.serviceWorker.controller, null, { timeout: 15000 });
    await page.waitForFunction(() => document.querySelector('#net-pill').dataset.net === 'online', null, { timeout: 15000 });

    await page.evaluate(() => (window._app.startFromGesture || window._app.startListening).call(window._app));
    const genus = await until(() => page.evaluate(() => {
      const s = window._app.classifier && window._app.classifier.getState();
      const d = s && s.detections.find(x => x.genus === 'Aedes');
      return d && d.genus;
    }), 40000, 'an Aedes detection');
    assert.equal(genus, 'Aedes');

    // ── Admin session ──────────────────────────────────────────────────────
    await page.click('#tab-about');
    await page.click('#about-advanced > summary');
    await page.fill('#adv-pw', 'not the password');
    await page.click('#adv-form button[type="submit"]');
    await page.waitForSelector('#adv-error:not([hidden])');
    assert.equal(await page.$('#net-pill[data-admin]'), null, 'a wrong password grants nothing');
    await page.fill('#adv-pw', ADMIN_PW);
    await page.click('#adv-form button[type="submit"]');
    await page.waitForSelector('#net-pill[data-admin][role="button"]', { timeout: 10000 });
    assert.ok(await page.evaluate(() => document.querySelector('#about-advanced').open), 'Advanced unlocked too');
    assert.equal(await page.evaluate(() => Object.keys(localStorage).filter(k => /admin|token/i.test(k)).length +
                                           Object.keys(sessionStorage).length), 0, 'the admin token is never stored');

    await page.click('#net-pill');
    await page.waitForSelector('#dlg-admin-devices[open]');
    await until(() => page.evaluate(() => [...document.querySelectorAll('#adm-dev-list .adm-badge')]
      .some(b => b.textContent === 'Listening')), 20000, 'this device listed as listening');
    assert.match(await page.textContent('#adm-dev-summary'), /connected · \d+ listening/);

    await page.click('#adm-dev-server');
    await page.waitForSelector('#panel-server:not([hidden])');
    await until(() => page.evaluate(() => /\[Admin\] login from/.test(document.querySelector('#adm-log').textContent)),
                10000, 'the server log in the Server tab');
    assert.match(await page.textContent('#adm-health'), /Status\s*ok/);
    assert.match(await page.textContent('#adm-files'), /mosquitonet_state\.json/);

    await page.click('#adm-logout');
    await page.waitForSelector('#tab-server[hidden]', { state: 'attached' });
    assert.equal(await page.$('#net-pill[data-admin]'), null);
    assert.equal(await page.evaluate(() => document.querySelector('#about-advanced').open), false, 'Advanced locked again');
    assert.equal(await page.evaluate(() => document.querySelector('#adm-log').innerHTML +
                                           document.querySelector('#adm-dev-list').innerHTML), '', 'admin views wiped');

    assert.deepEqual(await page.evaluate(() => window.__csp), [], 'Content-Security-Policy violations');
    assert.deepEqual(errors, [], 'errors in the page');
    assert.ok(appFromSw.length > 0, 'the service worker serves the app on its second load');
    assert.deepEqual(apiFromSw, [], 'API requests answered by the service worker');
  } catch (e) {
    console.log(log.join('').slice(-3000));
    throw e;
  } finally {
    if (browser) await browser.close();
    proc.kill('SIGTERM');
  }
});
