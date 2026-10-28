'use strict';
/*
 * What the App makes of the engine's detections (App.handleDetections), in the
 * real page: a candidate below CONF_MIN is not a detection at all; from
 * CONF_MIN it is counted and kept in History; from CONF_UPLOAD it is confident —
 * shared with the map and one training step for the shared model, which is sent
 * after every such step. One insect (track) is one event, whatever its updates.
 */
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');
const { serve, loadPlaywright, isolateNetwork } = require('./run-app');

const ROOT = process.env.SITE_ROOT ? path.resolve(process.env.SITE_ROOT) : path.join(__dirname, '..', '..');

test('confidence rules: ignore < 50 %, count ≥ 50 %, share + train once ≥ 70 %', { timeout: 60000 }, async () => {
  const srv = await serve(ROOT);
  const { chromium } = loadPlaywright();
  const browser = await chromium.launch();
  try {
    const ctx = await browser.newContext({ serviceWorkers: 'block' });
    await isolateNetwork(ctx);                                   // offline backend: uploads queue
    await ctx.addInitScript(() => { try { localStorage.setItem('mosquitonet_onboarded', '1'); } catch (e) {} });
    const page = await ctx.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(`http://127.0.0.1:${srv.address().port}/`, { waitUntil: 'load' });
    await page.waitForFunction(() => window._app && window._app.fedClient);

    const feed = (conf, trackId) => page.evaluate(([c, t]) => {
      const app = window._app;
      const det = { species: SPECIES_DB.find(s => s.id === 'culex'), genus: 'Culex', confidence: c, frequency: 455, f0: 455,
                    posterior: [{ id: 'culex', p: 0.9 }], taxonLevel: 'species', trackId: t, snr: 14, range: 'near' };
      app.handleDetections([det], Date.now());
      return { events: app.history.events().length, steps: app.fedClient.localSteps,
               queued: app.fedClient._loadQueue().length, locked: !!app._lastLock };
    }, [conf, trackId]);

    const before = await feed(0.01, 1);                          // the "1 %" candidate
    assert.deepEqual(before, { events: 0, steps: 0, queued: 0, locked: false }, 'a 1 % candidate is not a detection');
    assert.deepEqual(await feed(0.49, 1), before, 'nor is 49 %');

    const counted = await feed(0.6, 2);                          // a detection, not yet confident
    assert.deepEqual(counted, { events: 1, steps: 0, queued: 0, locked: true });

    const confident = await feed(0.8, 2);                        // same insect becomes confident
    assert.deepEqual(confident, { events: 1, steps: 1, queued: 1, locked: true }, 'shared and trained once');
    assert.deepEqual(await feed(0.95, 2), confident, 'still once per insect');
    assert.deepEqual(await feed(0.3, 2), confident, 'a dip on a counted insect changes nothing');

    assert.equal(await page.evaluate(() => window._app.fedClient.UPLOAD_EVERY), 1, 'an update after every confident step');
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
    srv.close();
  }
});
