'use strict';
/*
 * Client hardening on the BUILT site (tools/build-site.sh), in headless Chromium:
 *   • the service worker's offline shell can only ever be the app page, and
 *     offline the page runs the app code it was last served with (sw.js);
 *   • the GitHub Pages copy — static files, no response headers — runs under
 *     the page's own CSP <meta> (tools/csp-meta.js): the app violates nothing,
 *     admin login works cross-origin, and injected markup runs no code.
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
const { serve, loadPlaywright, isolateNetwork } = require('./run-app');

const ROOT = path.join(__dirname, '..', '..');
const PY = process.env.PYTHON || 'python3';
const SHA = '0123456789abcdef0123456789abcdef01234567';

function missingDeps() {
  try { execFileSync(PY, ['-c', 'import flask, numpy'], { stdio: 'ignore' }); } catch (e) { return 'server deps missing: pip install -r requirements.txt'; }
  try { require.resolve('terser', { paths: [ROOT] }); } catch (e) { return 'minifiers missing: npm ci'; }
  return false;
}
// Locally a missing dependency skips with a reason; in CI (CI=true) it must fail.
const SKIP = !process.env.CI && missingDeps();

const build = (out, urls) => execFileSync('bash', [path.join(ROOT, 'tools', 'build-site.sh'), out],
  { stdio: 'pipe', env: { ...process.env, SITE_RUN: '7', SITE_SHA: SHA, SERVER_URLS: urls } });
const listen = (root, port = 0) => serve(root, port).then(srv => ({ srv, port: srv.address().port }));
// Close a test server now: drop its sockets first. close() alone waits for every
// open connection, and one Chromium kept open made a CI run hang until the
// job's timeout (the next listen() on the same port, and the browser, waited).
const close = (srv) => new Promise(r => { srv.closeAllConnections(); srv.close(() => r()); });

function freePort() {
  return new Promise((resolve, reject) => {
    const s = net.createServer().once('error', reject).listen(0, '127.0.0.1', () => {
      const { port } = s.address();
      s.close(() => resolve(port));
    });
  });
}

async function until(fn, ms, what) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    try { if (await fn()) return; } catch (e) { /* not yet */ }
    await new Promise(r => setTimeout(r, 250));
  }
  throw new Error(`timed out after ${ms} ms waiting for ${what}`);
}
const healthy = (port) => new Promise((resolve) => {
  http.get(`http://127.0.0.1:${port}/health`, (res) => { res.resume(); resolve(res.statusCode === 200); })
      .on('error', () => resolve(false));
});

async function browserContext() {
  const { chromium } = loadPlaywright();
  const browser = await chromium.launch();
  const ctx = await browser.newContext();               // service workers allowed
  await isolateNetwork(ctx);
  await ctx.addInitScript(() => {
    window.__csp = [];
    document.addEventListener('securitypolicyviolation', (e) => window.__csp.push(`${e.violatedDirective} ${e.blockedURI}`));
    try { localStorage.setItem('mosquitonet_onboarded', '1'); } catch (e) {}
  });
  return { browser, ctx, page: await ctx.newPage() };
}
// Load, wait until the service worker is ACTIVE, then reload under it (as
// site.e2e.js does). Reloading before it activated could leave the page
// uncontrolled on a slow CI runner. Bounded: never a silent hang.
async function openControlled(page, url) {
  await page.goto(url, { waitUntil: 'load' });
  await page.evaluate(() => Promise.race([
    navigator.serviceWorker.ready.then(() => true),
    new Promise((_, reject) => setTimeout(() => reject(new Error('no active service worker after 30 s')), 30000)),
  ]));
  await page.reload({ waitUntil: 'load' });
  await page.waitForFunction(() => window._app && navigator.serviceWorker.controller, null, { timeout: 30000 });
}

test('a navigation to another file in scope never becomes the offline app', { skip: SKIP, timeout: 120000 }, async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'mn-sw-'));
  build(path.join(tmp, 'site'), 'http://127.0.0.1:9');
  const { srv, port } = await listen(path.join(tmp, 'site'));
  const { browser, ctx, page } = await browserContext();
  try {
    const origin = `http://127.0.0.1:${port}`;
    await openControlled(page, `${origin}/`);
    // What a link on any other web page can do: open a different file of the site.
    await page.goto(`${origin}/map-data.json`).catch(() => {});
    await ctx.setOffline(true);
    await page.goto(`${origin}/`, { waitUntil: 'load' }).catch(() => {});
    assert.ok(await page.evaluate(() => !!window._app).catch(() => false), 'the app still opens offline');
  } finally {
    await browser.close();
    await close(srv);
  }
});

test('offline, the page runs the app code it was last served with', { skip: SKIP, timeout: 180000 }, async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'mn-sw-'));
  const a = path.join(tmp, 'a'), b = path.join(tmp, 'b');
  build(a, 'http://127.0.0.1:9');
  fs.cpSync(a, b, { recursive: true });                        // release b: same build, new ?v= stamp
  fs.writeFileSync(path.join(b, 'index.html'),
    fs.readFileSync(path.join(a, 'index.html'), 'utf8').replace(/\?v=[0-9a-f]{12}"/g, '?v=bbbbbbbbbbbb"'));
  fs.appendFileSync(path.join(a, 'js', 'detector.js'), ';self.__release="a";');
  fs.appendFileSync(path.join(b, 'js', 'detector.js'), ';self.__release="b";');
  const port = await freePort();                               // one origin: one service worker
  const { browser, ctx, page } = await browserContext();
  try {
    for (const root of [a, b]) {                               // the user opens release a, later b
      const { srv } = await listen(root, port);
      await openControlled(page, `http://127.0.0.1:${port}/`);
      await close(srv);
    }
    await ctx.setOffline(true);
    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' });
    await page.waitForFunction(() => window._app);
    assert.equal(await page.evaluate(() => self.__release), 'b', 'release b page runs release b code');
  } finally {
    await browser.close();
  }
});

test('the GitHub Pages copy (no headers) runs under its own CSP <meta>', { skip: SKIP, timeout: 120000 }, async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'mn-meta-'));
  const apiPort = await freePort();
  const PW = 'placeholder admin password for tests';
  build(path.join(tmp, 'site'), `http://127.0.0.1:${apiPort}`);
  const log = [];
  const proc = spawn(PY, ['federated_server.py'], {
    cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, PORT: String(apiPort), DATA_DIR: path.join(tmp, 'data'), ADMIN_PASSWORD: PW },
  });
  proc.stdout.on('data', (d) => log.push(String(d)));
  proc.stderr.on('data', (d) => log.push(String(d)));
  const { srv, port } = await listen(path.join(tmp, 'site'));  // like Pages: files, no CSP header
  const { browser, page } = await browserContext();
  const errors = [];
  page.on('pageerror', e => errors.push(`pageerror: ${e.message}`));
  page.on('console', m => {
    if (m.type() === 'error' && !/Failed to load resource|ERR_FAILED|net::/.test(m.text())) errors.push(`console: ${m.text()}`);
  });
  try {
    await until(() => healthy(apiPort), 20000, 'the backend');
    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' });
    assert.ok(await page.$('meta[http-equiv="Content-Security-Policy"]'), 'the built page carries its CSP');
    await page.waitForFunction(() => document.querySelector('#net-pill').dataset.net === 'online', null, { timeout: 15000 });
    await page.evaluate(() => window._app.selectTab('map'));
    await page.waitForSelector('.leaflet-container', { timeout: 15000 });
    await page.evaluate(() => window._app.selectTab('about'));
    await page.click('#about-advanced > summary');
    await page.fill('#adv-pw', PW);
    await page.click('#adv-form button[type="submit"]');
    await page.waitForSelector('#tab-server:not([hidden])', { timeout: 10000 });   // cross-origin admin login
    await page.evaluate(() => window._app.selectTab('server'));
    await page.waitForSelector('#adm-log .adm-log-row', { timeout: 10000 });
    assert.deepEqual(await page.evaluate(() => window.__csp), [], 'the app itself violates nothing');
    assert.deepEqual(errors, [], 'errors in the page');
    // A missed esc() somewhere: injected markup must not run code next to an admin session.
    await page.evaluate(() => document.body.insertAdjacentHTML('beforeend', '<img src=x onerror="window.__xss=1">'));
    await page.waitForTimeout(400);
    assert.equal(await page.evaluate(() => window.__xss), undefined, 'injected markup ran code');
  } catch (e) {
    console.log(log.join('').slice(-3000));
    throw e;
  } finally {
    await browser.close();
    await close(srv);
    proc.kill('SIGTERM');
  }
});
