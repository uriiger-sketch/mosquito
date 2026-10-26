'use strict';
/*
 * Drive the real app in headless Chromium with a WAV file as the microphone
 * and record what the engine and the UI report. Used by the E2E suite and
 * runnable on its own for ad-hoc checks:
 *
 *   node tests/e2e/run-app.js <file.wav> [--seconds 20] [--lat 13.75 --lng 100.5] [--port 8765]
 */
const path = require('path');
const http = require('http');
const fs = require('fs');

function loadPlaywright() {
  try { return require('playwright'); } catch (e) { /* fall through */ }
  return require('/opt/node22/lib/node_modules/playwright');
}

// Tests may reach only the local servers they start. An allowlist, never a list
// of production hosts: the app's endpoints (SERVER_URLS) change, and a missed
// one sends test detections to the live map — in October 2026 adding the
// Railway domain to SERVER_URLS did exactly that.
const LOCAL_HOSTS = new Set(['127.0.0.1', 'localhost', '[::1]']);
const leavesTestMachine = (url) => !LOCAL_HOSTS.has(url.hostname);

function isolateNetwork(ctx) {
  return ctx.route(leavesTestMachine, r => r.abort());
}

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.webmanifest': 'application/manifest+json',
  '.woff2': 'font/woff2', '.woff': 'font/woff', '.wav': 'audio/wav' };

// Minimal static server (no dependency on http-server in CI).
function serve(root, port = 0) {
  return new Promise((resolve) => {
    const srv = http.createServer((req, res) => {
      let p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
      if (p.endsWith('/')) p += 'index.html';
      const file = path.join(root, path.normalize(p).replace(/^(\.\.[/\\])+/, ''));
      if (!file.startsWith(root)) { res.writeHead(403); return res.end(); }
      fs.readFile(file, (err, data) => {
        if (err) { res.writeHead(404); return res.end('not found'); }
        res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream',
                             'Cache-Control': 'no-store' });
        res.end(data);
      });
    });
    srv.listen(port, '127.0.0.1', () => resolve(srv));
  });
}

async function runApp(wav, { seconds = 20, lat = null, lng = null, root = path.join(__dirname, '..', '..'),
                             viewport = { width: 390, height: 844 }, onPage } = {}) {
  const { chromium } = loadPlaywright();
  const srv = await serve(path.resolve(root));
  const port = srv.address().port;
  const browser = await chromium.launch({ args: [
    '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
    `--use-file-for-fake-audio-capture=${path.resolve(wav)}`,
    '--autoplay-policy=no-user-gesture-required',
  ] });
  const perms = ['microphone'];
  if (lat !== null) perms.push('geolocation');
  const ctx = await browser.newContext({ viewport, permissions: perms,
    geolocation: lat !== null ? { latitude: lat, longitude: lng } : undefined,
    serviceWorkers: 'block' });
  await isolateNetwork(ctx);     // never the production API (nor tiles, fonts, CDNs)
  const page = await ctx.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(`pageerror: ${e.message}`));
  page.on('console', m => {
    if (m.type() === 'error' && !/Failed to load resource|ERR_FAILED|net::/.test(m.text())) errors.push(`console: ${m.text()}`);
  });
  try {
    await page.goto(`http://127.0.0.1:${port}/index.html`, { waitUntil: 'load' });
    await page.waitForFunction(() => window._app, null, { timeout: 15000 });
    await page.waitForTimeout(800);
    // The same path the Start button takes (GPS request + start), not the bare
    // startListening(): location-dependent behaviour (native species, local
    // names) is only exercised when the app actually acquires a fix.
    await page.evaluate(() => (window._app.startFromGesture || window._app.startListening).call(window._app));
    const samples = [];
    const t0 = Date.now();
    while (Date.now() - t0 < seconds * 1000) {
      await page.waitForTimeout(500);
      samples.push(await page.evaluate(() => {
        const c = window._app.classifier;
        if (!c) return null;
        const s = c.getState();
        return { t: performance.now(), phase: s.phase, ambientDb: s.ambientDb,
                 sr: window._app.engine && window._app.engine.ctx ? window._app.engine.ctx.sampleRate : null,
                 dets: s.detections.map(d => ({ id: d.species.id, genus: d.genus, f0: d.f0,
                                                conf: d.confidence, range: d.range })) };
      }));
    }
    if (onPage) await onPage(page);
    return { samples: samples.filter(Boolean), errors };
  } finally {
    await browser.close();
    srv.close();
  }
}

module.exports = { runApp, serve, loadPlaywright, isolateNetwork, leavesTestMachine };

if (require.main === module) {
  const a = process.argv.slice(2);
  const opt = (k, d) => { const i = a.indexOf(k); return i >= 0 ? +a[i + 1] : d; };
  runApp(a[0], { seconds: opt('--seconds', 20), lat: opt('--lat', null), lng: opt('--lng', null) })
    .then(({ samples, errors }) => {
      const dets = samples.flatMap(s => s.dets);
      const byGenus = {};
      for (const d of dets) byGenus[d.genus] = (byGenus[d.genus] || 0) + 1;
      const first = samples.findIndex(s => s.dets.length);
      console.log(JSON.stringify({ sr: samples[0] && samples[0].sr, snapshots: samples.length,
        phases: [...new Set(samples.map(s => s.phase))], detectedSnapshots: samples.filter(s => s.dets.length).length,
        firstDetectionAt_s: first >= 0 ? (first + 1) * 0.5 : null, byGenus,
        f0: dets.length ? +(dets.reduce((s, d) => s + d.f0, 0) / dets.length).toFixed(1) : null,
        errors }, null, 1));
    })
    .catch(e => { console.error(e); process.exit(1); });
}
