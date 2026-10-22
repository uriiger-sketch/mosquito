'use strict';
/*
 * End-to-end detection tests: the REAL app in headless Chromium, with a WAV
 * file as the microphone. Exercises Web Audio capture, the AnalyserNode path,
 * the rAF analysis loop, the classifier, and the rendered UI together.
 *
 *   npm run test:e2e            (needs Playwright + Chromium)
 */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { FIXTURES, writeWav, SR, rng } = require('./make-wav');
const { runApp } = require('./run-app');

const SECONDS = +(process.env.E2E_SECONDS || 18);
const SETTLE_S = 5;                    // ignore calibration + vote warm-up
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'mn-e2e-'));

const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[s.length >> 1]; };

for (const [name, fx] of Object.entries(FIXTURES)) {
  test(`${name}: ${fx.expect === 'none' ? 'no detection' : `detects ${fx.expect}`}`, { timeout: (SECONDS + 60) * 1000 }, async () => {
    const wav = path.join(dir, `${name}.wav`);
    if (!fs.existsSync(wav)) writeWav(wav, fx.build(30 * SR, rng(name.length * 7919)));
    let uiText = '';
    const { samples, errors } = await runApp(wav, {
      seconds: SECONDS, lat: fx.loc ? fx.loc.lat : null, lng: fx.loc ? fx.loc.lng : null,
      onPage: async (page) => { uiText = await page.evaluate(() => document.body.innerText); },
    });
    assert.deepEqual(errors, [], 'JavaScript errors in the page');
    assert.ok(samples.length >= SECONDS, 'engine produced no state snapshots');
    const settled = samples.slice(SETTLE_S * 2);
    const dets = settled.flatMap(s => s.dets);

    if (fx.expect === 'none') {
      assert.equal(dets.length, 0, `false alarm(s): ${JSON.stringify(dets.slice(0, 3))}`);
      return;
    }
    const hitFrac = settled.filter(s => s.dets.some(d => d.genus === fx.expect)).length / settled.length;
    assert.ok(hitFrac >= 0.4, `${fx.expect} locked in only ${(hitFrac * 100).toFixed(0)} % of snapshots`);
    const byGenus = {};
    for (const d of dets) byGenus[d.genus] = (byGenus[d.genus] || 0) + 1;
    const top = Object.entries(byGenus).sort((a, b) => b[1] - a[1])[0][0];
    assert.equal(top, fx.expect, `majority genus ${top} (${JSON.stringify(byGenus)})`);
    const f0 = median(dets.filter(d => d.genus === fx.expect).map(d => d.f0));
    assert.ok(Math.abs(f0 - fx.f0) <= 8, `measured f0 ${f0.toFixed(1)} Hz vs true ${fx.f0} Hz`);
    assert.ok(uiText.includes(fx.expect), `UI never shows "${fx.expect}"`);
  });
}
