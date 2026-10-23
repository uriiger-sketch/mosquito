'use strict';
// Regression tests for the P1 bugs fixed in the v3 engine, plus measured-f0,
// sample-rate and identification behaviour.
const test = require('node:test');
const assert = require('node:assert/strict');
const H = require('./helpers');

const eng = H.loadEngine();
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };

function tone(f0, { sr = H.SR, seconds = 10, onset = 3, amp = 0.02, noise = 0.003, seed = 7 } = {}) {
  const R = H.rng(seed), len = Math.round(seconds * sr);
  return H.mix(H.whiteNoise(len, noise, R), H.mosquitoTone(len, f0, amp, R, Math.round(onset * sr), sr));
}

test('Ae. caspius (548 Hz = 11 × 50 Hz) is detectable — no static mains-hum blacklist', () => {
  const { frames } = H.runClassifier(tone(548));
  const dets = frames.flat();
  assert.ok(dets.length > 0, 'no detection of a 548 Hz flight tone');
  const p = dets[dets.length - 1].posterior.find(x => x.id === 'ochlerotatus_caspius');
  assert.ok(p && p.p > 0.1, 'caspius missing from the posterior');
});

test('Detection.frequency is the MEASURED f0 (sub-bin), not the catalogue value', () => {
  // E2E fixtures: Aedes 628 Hz, Anopheles 418 Hz, Culex 464 Hz must report within ±8 Hz.
  for (const f0 of [628, 418, 464, 410, 552]) {
    const { frames } = H.runClassifier(tone(f0));
    const fs = frames.flat().map(d => d.frequency);
    assert.ok(fs.length > 0, `no detection at ${f0} Hz`);
    const m = median(fs);
    assert.ok(Math.abs(m - f0) < 2, `f0 ${f0} reported ${m.toFixed(2)}`);
    for (const d of frames.flat()) assert.equal(d.frequency, d.f0);
  }
});

test('the engine runs at 48 kHz (all DSP derived from the real sample rate)', () => {
  const sr = 48000;
  const { frames, clf } = H.runClassifier(tone(617, { sr }), { sr });
  assert.equal(clf.sampleRate, 48000);
  assert.ok(Math.abs(clf.binHz - 48000 / 8192) < 1e-9);
  const fs = frames.flat().map(d => d.frequency);
  assert.ok(fs.length > 0, 'no detection at 48 kHz');
  assert.ok(Math.abs(median(fs) - 617) < 2, `48 kHz f0 ${median(fs)}`);
  const R = H.rng(5);
  assert.equal(H.runClassifier(H.whiteNoise(20 * sr, 0.01, R), { sr }).frames.flat().length, 0);
});

test('no detections are emitted during the startup calibration', () => {
  // Loud tone present from t = 0: nothing may be reported before endCalibration (2 s).
  const sig = tone(617, { onset: 0, seconds: 8 });
  const seen = [];
  H.runClassifier(sig, { onFrame: (i, t, dets, clf) => seen.push({ t, n: dets.length, phase: clf.getState().phase, cal: clf.getState().calibration }) });
  const during = seen.filter(s => s.phase === 'calibrating');
  assert.ok(during.length > 10);
  assert.equal(during.reduce((n, s) => n + s.n, 0), 0, 'detection during calibration');
  for (let i = 1; i < during.length; i++) assert.ok(during[i].cal >= during[i - 1].cal && during[i].cal <= 1);
  assert.ok(seen.some(s => s.n > 0), 'tone never detected after calibration');
});

test('steady machine tone, constant-level tone and voiced speech produce no detection', () => {
  const R = H.rng(4242), len = 60 * H.SR;
  const noise = H.whiteNoise(len, 0.01, R);
  // legacy bench "fan tone 420 Hz": rock-steady frequency and level, harmonics
  const fan = new Float32Array(len);
  for (let i = 30 * H.SR; i < len; i++) { const ph = 2 * Math.PI * 420 * i / H.SR; fan[i] = 0.005 * (Math.sin(ph) + 0.6 * Math.sin(2 * ph) + 0.4 * Math.sin(3 * ph) + 0.3 * Math.sin(4 * ph)); }
  assert.equal(H.runClassifier(H.mix(noise, fan)).frames.flat().length, 0, 'machine tone detected');
  // a constant-level (non-flying) tone with only slow pitch wander
  const held = H.mosquitoTone(len, 500, 0.02, H.rng(9), 5 * H.SR, H.SR, { am: false });
  assert.equal(H.runClassifier(H.mix(noise, held)).frames.flat().length, 0, 'constant-level tone detected');
  // voiced speech: f0 150-280 Hz glides in syllables, 12 harmonics
  const sp = new Float32Array(len), R2 = H.rng(77);
  let ph = 0, t0 = 0, on = false, f0 = 200, target = 200;
  for (let i = 0; i < len; i++) {
    if (i >= t0) { on = !on || R2.u() < 0.2; t0 = i + Math.floor((on ? 0.15 + R2.u() * 0.25 : 0.05 + R2.u() * 0.2) * H.SR); target = 150 + R2.u() * 130; }
    f0 += (target - f0) * 0.0004; ph += 2 * Math.PI * f0 / H.SR;
    if (on) { let s = 0; for (let h = 1; h <= 12; h++) s += Math.sin(h * ph) / h; sp[i] = 0.015 * s; }
  }
  assert.equal(H.runClassifier(H.mix(noise, sp)).frames.flat().length, 0, 'speech detected');
});

test('posterior: location prior, disabled species, genus fallback', () => {
  const m = new eng.SpeciesModel();
  // Clean 617 Hz with all species enabled: honest overlap (aegypti on top, not certain)
  const all = m.posterior(617, { disabled: new Set(), lat: null, lng: null, tod: null });
  assert.equal(all.post[0].id, 'aedes_aegypti');
  assert.ok(all.post[0].p < 0.6, 'overlapping species cannot be certain without priors');
  const sum = all.post.reduce((s, o) => s + o.p, 0);
  assert.ok(Math.abs(sum - 1) < 1e-9);
  // Rome: Cx. pipiens and Ae. albopictus native, Ae. aegypti / Mansonia not (and disabled by the app)
  const nat = eng.getNativeSpecies(41.9, 12.5);
  const disabled = new Set(eng.SPECIES_DB.filter(s => !nat.has(s.id)).map(s => s.id));
  const rome = m.posterior(617, { disabled, lat: 41.9, lng: 12.5, tod: null });
  assert.ok(rome.post.every(o => !disabled.has(o.id)));
  assert.ok(rome.post[0].p > all.post[0].p, 'location prior should sharpen the posterior');
  // Non-native but enabled species get a much lower prior than natives
  const romeAll = m.posterior(617, { disabled: new Set(), lat: 41.9, lng: 12.5, tod: null });
  const pAeg = romeAll.post.find(o => o.id === 'aedes_aegypti').p, pPip = romeAll.post.find(o => o.id === 'culex_pipiens').p;
  assert.ok(pPip > 5 * pAeg);
  // Activity prior is mild
  const night = m.posterior(617, { disabled: new Set(), lat: null, lng: null, tod: 'night' });
  assert.ok(night.post.find(o => o.id === 'aedes_aegypti').p > 0.5 * all.post[0].p);
});

test('f0 estimator: parabolic interpolation on the Blackman dB spectrum is unbiased to < 0.02 bin', () => {
  const N = H.N, sr = H.SR, binHz = sr / N;
  let worst = 0;
  for (let i = 0; i < 9; i++) {
    const f = (90 + i / 8) * binHz;                     // fractional offsets 0 … 1 bin
    const x = new Float32Array(N);
    for (let n = 0; n < N; n++) x[n] = 0.1 * Math.sin(2 * Math.PI * f * n / sr);
    const tr = new eng.FarFieldTracker(binHz, 4096 / sr);
    const db = H.analyserDb(x);
    for (let k = 0; k < tr.nb; k++) { tr.P[k] = Math.pow(10, db[k] / 10); tr.L[k] = db[k]; }
    worst = Math.max(worst, Math.abs(tr._interp(90 + Math.round(i / 8)) * binHz - f) / binHz);
  }
  assert.ok(worst < 0.02, `bias ${worst.toFixed(4)} bin`);
});
