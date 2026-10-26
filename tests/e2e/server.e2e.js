'use strict';
/*
 * End-to-end connectivity: the REAL app in headless Chromium talking to the
 * REAL backend (federated_server.py). The detection suite blocks the
 * production API, so nothing else tests that the app can reach a server at
 * all — production sat "Offline" for months with every test green.
 *
 * The page is pointed, by tools/set-server-urls.js exactly as the Pages deploy
 * does it, at three endpoints in order: one that refuses connections, one that
 * answers but is not MosquitoNet (like a domain not attached to the backend),
 * and the real server — so failover is exercised on the way to "Online".
 *
 *   npm run test:e2e     (Playwright + Chromium, and the server's Python deps:
 *                         pip install -r requirements.txt)
 */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const net = require('net');
const http = require('http');
const path = require('path');
const { spawn, execFileSync } = require('child_process');
const { serve, loadPlaywright } = require('./run-app');
const { setServerUrls } = require('../../tools/set-server-urls');

const ROOT = path.join(__dirname, '..', '..');
const PY = process.env.PYTHON || 'python3';

function serverDepsInstalled() {
  try { execFileSync(PY, ['-c', 'import flask, numpy'], { stdio: 'ignore' }); return true; } catch (e) { return false; }
}
// Locally a missing Flask skips with a reason; in CI (CI=true) it must fail.
const SKIP = !process.env.CI && !serverDepsInstalled() && 'server deps missing: pip install -r requirements.txt';

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

function startServer(port, dataDir, log) {
  const proc = spawn(PY, ['federated_server.py'], {
    cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, PORT: String(port), DATA_DIR: dataDir },
  });
  proc.stdout.on('data', (d) => log.push(String(d)));
  proc.stderr.on('data', (d) => log.push(String(d)));
  const health = `http://127.0.0.1:${port}/health`;
  return until(() => getJSON(health).then(h => h.status === 'ok'), 20000, `server on ${port}`).then(() => proc);
}

function stopServer(proc) {
  if (!proc || proc.exitCode !== null) return Promise.resolve();
  return new Promise((resolve) => { proc.once('exit', resolve); proc.kill('SIGTERM'); });
}

// Fill `dir` with the site, its index.html pointed at `urls`; everything else
// is a symlink to the working tree.
function writeSite(dir, urls) {
  for (const name of fs.readdirSync(ROOT)) {
    if (name === 'index.html' || name === '.git') continue;
    fs.symlinkSync(path.join(ROOT, name), path.join(dir, name));
  }
  const { html } = setServerUrls(fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8'), urls.join(','));
  fs.writeFileSync(path.join(dir, 'index.html'), html);
}

test('app reaches the real server: failover, heartbeat, upload, diagnostics, outage recovery',
     { skip: SKIP, timeout: 150000 }, async () => {
  const log = [];
  const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'mn-data-'));
  const apiPort = await freePort();
  const live = `http://127.0.0.1:${apiPort}`;
  const refused = `http://127.0.0.1:${await freePort()}`;   // nothing listens here
  let api = await startServer(apiPort, dataDir, log);

  // The static server doubles as the "answers, but isn't MosquitoNet" endpoint:
  // POST /heartbeat gets its 404, like a platform's "application not found".
  // (It reads from disk per request, so the site can be written once its port is known.)
  const siteRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'mn-site-'));
  const srv = await serve(siteRoot);
  const wrongApp = `http://127.0.0.1:${srv.address().port}`;
  writeSite(siteRoot, [refused, wrongApp, live]);

  const { chromium } = loadPlaywright();
  const browser = await chromium.launch();
  const pageLog = [];
  try {
    const ctx = await browser.newContext({ serviceWorkers: 'block' });
    await ctx.route(/tile\.openstreetmap\.org|fonts\.(googleapis|gstatic)\.com/, r => r.abort());
    const page = await ctx.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(`pageerror: ${e.message}`));
    page.on('console', m => pageLog.push(m.text()));
    await page.goto(`${wrongApp}/index.html`);
    await page.waitForFunction(() => window._app, null, { timeout: 15000 });
    const pill = () => page.evaluate(() => document.querySelector('#net-pill').dataset.net);

    // 1. Fails over past both dead endpoints to the real one, and remembers it.
    await until(async () => (await pill()) === 'online', 30000, 'pill → online');
    assert.equal(await page.evaluate(() => window.SERVER_URL), live);
    assert.equal(await page.evaluate(() => localStorage.getItem('mosquitonet_server')), live);
    const h1 = await getJSON(`${live}/health`);
    assert.ok(h1.unique_devices >= 1, `server never saw the device: ${JSON.stringify(h1)}`);

    // 2. A confirmed find travels the real upload path and is stored.
    await page.evaluate(() => {
      const sp = SPECIES_DB.find(s => s.id === 'culex');
      window._app.fedClient.reportDetection({ species: sp, confidence: 0.9, frequency: 455, f0: 455 });
    });
    await until(() => getJSON(`${live}/health`).then(h => h.total_detections === 1), 15000, 'detection stored');
    // The server stores before the browser handles the ack, so wait for the dequeue.
    await until(() => page.evaluate(() => window._app.fedClient._loadQueue().length === 0), 5000, 'acked find dequeued');

    // 3. Diagnostics name each endpoint's failure mode.
    await page.evaluate(() => window.testServerConnection());
    const report = await until(() => page.evaluate(() => {
      const el = document.querySelector('#dbg-server-result');
      return !el.hidden && el.textContent.split('\n').length === 3 && el.textContent;
    }), 15000, 'diagnostics');
    const [r1, r2, r3] = report.split('\n');
    assert.match(r1, /unreachable/, r1);
    assert.match(r2, /HTTP 404 .*not the MosquitoNet server/, r2);
    assert.match(r3, /OK \(MosquitoNet v\d+\)/, r3);

    // 4. A find sent to a host that answers 404 but isn't MosquitoNet (an
    //    unattached domain) stays queued, and is delivered once the real server is back.
    await page.evaluate((u) => { window.SERVER_URL = u; }, wrongApp);
    const bounced = page.waitForResponse(r => r.url() === `${wrongApp}/detection`);
    // (Another species: the server folds a repeat of the same device + species
    // within RECENT_WINDOW into the earlier record instead of counting it.)
    await page.evaluate(() => {
      const sp = SPECIES_DB.find(s => s.id === 'aedes_aegypti');
      window._app.fedClient.reportDetection({ species: sp, confidence: 0.9, frequency: 610, f0: 610 });
    });
    assert.equal((await bounced).status(), 404);
    await until(() => page.evaluate(() => !window._app.fedClient._flushing), 5000, 'flush to settle');
    assert.equal(await page.evaluate(() => window._app.fedClient._loadQueue().length), 1, 'find dropped on a platform 404');
    await page.evaluate((u) => { window.SERVER_URL = u; window._app.fedClient.flushDetectionQueue(); }, live);
    await until(() => getJSON(`${live}/health`).then(h => h.total_detections === 2), 15000, 'queued find delivered');
    await until(() => page.evaluate(() => window._app.fedClient._loadQueue().length === 0), 5000, 'delivered find dequeued');

    // 5. Backend outage → "Offline"; back up → "Online" again within one beat.
    await stopServer(api);
    await until(async () => (await pill()) === 'offline', 25000, 'pill → offline');
    api = await startServer(apiPort, dataDir, log);
    await until(async () => (await pill()) === 'online', 25000, 'pill → online after restart');
    assert.equal(await page.evaluate(() => window.SERVER_URL), live);
    assert.deepEqual(errors, [], 'JavaScript errors in the page');
  } catch (e) {
    console.log(`# server log:\n${log.join('').split('\n').filter(l => !l.includes('GET /health')).slice(-30).join('\n')}`);
    console.log(`# page console:\n${pageLog.slice(-30).join('\n')}`);
    throw e;
  } finally {
    await browser.close();
    srv.close();
    await stopServer(api);
  }
});
