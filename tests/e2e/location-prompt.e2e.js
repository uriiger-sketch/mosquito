'use strict';
/*
 * The user is asked for location ONCE. Safari shows a separate permission
 * prompt for every geolocation request still pending, so the app may only
 * start its continuous watch after the first request has been answered with
 * access granted (App.acquireGeolocation). A fake navigator.geolocation records
 * how many requests were outstanding whenever a new one starts.
 */
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');
const { serve, loadPlaywright, isolateNetwork } = require('./run-app');

const ROOT = process.env.SITE_ROOT ? path.resolve(process.env.SITE_ROOT) : path.join(__dirname, '..', '..');

function fakeGeolocation(answer) {
  // Runs in the page before the app: one shared "permission prompt" that
  // answers after 400 ms, like a user tapping Allow / Don't Allow.
  const g = window.__geo = { calls: [], outstandingAtCall: [] };
  let outstanding = 0;
  const reply = (ok, err) => setTimeout(() => {
    outstanding--;
    if (answer === 'allow') ok({ coords: { latitude: 32.08, longitude: 34.78, accuracy: 25 }, timestamp: Date.now() });
    else err({ code: 1, message: 'User denied Geolocation' });
  }, 400);
  const geo = {
    getCurrentPosition(ok, err) { g.calls.push('get'); g.outstandingAtCall.push(outstanding++); reply(ok, err); },
    watchPosition(ok, err) { g.calls.push('watch'); g.outstandingAtCall.push(outstanding++); reply(ok, err); return 1; },
    clearWatch() {},
  };
  Object.defineProperty(Navigator.prototype, 'geolocation', { get: () => geo, configurable: true });
  try { localStorage.setItem('mosquitonet_onboarded', '1'); } catch (e) {}
}

async function run(answer) {
  const srv = await serve(ROOT);
  const { chromium } = loadPlaywright();
  const browser = await chromium.launch();
  try {
    const ctx = await browser.newContext({ serviceWorkers: 'block' });
    await isolateNetwork(ctx);
    await ctx.addInitScript(fakeGeolocation, answer);
    const page = await ctx.newPage();
    await page.goto(`http://127.0.0.1:${srv.address().port}/`, { waitUntil: 'load' });
    await page.waitForFunction(() => window._app);
    await page.evaluate(() => window._app.acquireGeolocation());       // what the Start gesture does
    await page.waitForTimeout(1500);
    return await page.evaluate(() => ({ ...window.__geo, geo: window._app.geoState }));
  } finally {
    await browser.close();
    srv.close();
  }
}

test('location is requested once; the watch starts only after access was granted', { timeout: 60000 }, async () => {
  const r = await run('allow');
  assert.deepEqual(r.calls, ['get', 'watch']);
  assert.deepEqual(r.outstandingAtCall, [0, 0], 'a second request started while the first was still asking');
  assert.equal(r.geo, 'ok');
});

test('when location is refused, nothing asks again', { timeout: 60000 }, async () => {
  const r = await run('deny');
  assert.deepEqual(r.calls, ['get'], 'asked again after the user said no');
  assert.equal(r.geo, 'denied');
});
