'use strict';
// Engine ⇄ UI contract (see header of js/detector.js). These guard the API the
// UI depends on; detector internals may change freely as long as these hold.
const test = require('node:test');
const assert = require('node:assert/strict');
const H = require('./helpers');

const eng = H.loadEngine();

test('SPECIES_DB entries carry every field the UI/server rely on', () => {
  const ids = new Set();
  for (const sp of eng.SPECIES_DB) {
    for (const k of ['id', 'name', 'common', 'disease', 'risk', 'freq', 'freqRange', 'color',
                     'asymptomatic', 'activity', 'regions', 'genus']) {
      assert.ok(sp[k] !== undefined, `${sp.id} missing ${k}`);
    }
    assert.ok(!ids.has(sp.id), `duplicate id ${sp.id}`); ids.add(sp.id);
    assert.ok(sp.freqRange[0] < sp.freq && sp.freq < sp.freqRange[1], `${sp.id} freq outside range`);
    assert.match(sp.color, /^#[0-9a-f]{6}$/i);
    assert.ok(['day', 'night', 'dusk'].includes(sp.activity));
  }
});

test('classifier exposes the contract API', () => {
  const clf = new eng.MosquitoClassifier({ binHz: H.BIN_HZ, freqToBin: hz => Math.round(hz / H.BIN_HZ) });
  for (const m of ['setSensitivity', 'setLocation', 'setDisabledSpecies', 'beginCalibration',
                   'endCalibration', 'analyze', 'getState', 'getStatus', 'getRawConfidences']) {
    assert.equal(typeof clf[m], 'function', m);
  }
  const st = clf.getState();
  assert.ok(['idle', 'calibrating', 'gated', 'listening', 'candidate', 'detected'].includes(st.phase));
  assert.equal(Object.keys(st.perSpecies).length, eng.SPECIES_DB.length);
  for (const v of Object.values(st.perSpecies)) {
    assert.ok(['off', 'idle', 'building', 'locked'].includes(v.status));
    assert.ok(v.level >= 0 && v.level <= 1);
    assert.equal(typeof v.reason, 'string');
  }
});

test('a loud, clean mosquito tone is detected with a well-formed Detection', () => {
  const R = H.rng(7);
  const len = 9 * H.SR;
  const sig = H.mix(H.whiteNoise(len, 0.003, R), H.mosquitoTone(len, 617, 0.05, R, 3 * H.SR));
  const { frames, clf } = H.runClassifier(sig);
  const hits = frames.flat().filter(d => d.genus === 'Aedes');
  assert.ok(hits.length > 0, 'no Aedes detection on a clean 617 Hz tone');
  const d = hits[hits.length - 1];
  assert.ok(Math.abs(d.f0 - 617) < 40, `f0 ${d.f0}`);
  assert.equal(d.frequency, d.f0);
  assert.ok(d.confidence > 0 && d.confidence <= 1);
  assert.ok(['near', 'far'].includes(d.range));
  assert.ok(Array.isArray(d.posterior) && d.posterior.length > 0);
  const sum = d.posterior.reduce((s, p) => s + p.p, 0);
  assert.ok(sum <= 1 + 1e-9);
  for (let i = 1; i < d.posterior.length; i++) assert.ok(d.posterior[i - 1].p >= d.posterior[i].p);
  assert.equal(clf.getState().phase, 'detected');
});

test('noise alone produces no detection', () => {
  const R = H.rng(11);
  const { frames } = H.runClassifier(H.whiteNoise(20 * H.SR, 0.01, R));
  assert.equal(frames.flat().length, 0);
});

test('disabled species are never reported', () => {
  const R = H.rng(7);
  const len = 9 * H.SR;
  const sig = H.mix(H.whiteNoise(len, 0.003, R), H.mosquitoTone(len, 617, 0.05, R, 3 * H.SR));
  const disabled = new Set(['aedes_aegypti']);
  const { frames } = H.runClassifier(sig, { disabled });
  assert.equal(frames.flat().filter(d => d.species.id === 'aedes_aegypti').length, 0);
});

test('contract extensions: taxonLevel / genusP / trackId on every Detection', () => {
  const R = H.rng(3);
  const len = 10 * H.SR;
  const sig = H.mix(H.whiteNoise(len, 0.003, R), H.mosquitoTone(len, 560, 0.03, R, 3 * H.SR));
  const { frames } = H.runClassifier(sig);
  const dets = frames.flat();
  assert.ok(dets.length > 0);
  for (const d of dets) {
    assert.ok(['species', 'genus'].includes(d.taxonLevel));
    assert.equal(d.genus, d.species.genus, 'species.genus must equal genus');
    assert.ok(d.genusP > 0 && d.genusP <= 1);
    assert.ok(Number.isInteger(d.trackId));
    const top = d.posterior[0];
    if (d.taxonLevel === 'species') { assert.equal(d.species.id, top.id); assert.ok(top.p >= 0.6); }
    else assert.ok(top.p < 0.6);
  }
});

test('EngineState extensions: calibration progress, noisy, ambientGateDb', () => {
  const mk = () => new eng.MosquitoClassifier({ binHz: H.BIN_HZ, sampleRate: H.SR, fftSize: H.N, freqToBin: hz => Math.round(hz / H.BIN_HZ) });
  const run = (rms) => {
    const clf = mk(); clf.setTimeOfDay('none'); clf.beginCalibration();
    const R = H.rng(1), x = H.whiteNoise(6 * H.SR, rms, R);
    const cal = [];
    for (let end = H.N, i = 0; end <= x.length; end += H.HOP, i++) {
      if (i === 22) clf.endCalibration();
      const fr = x.subarray(end - H.N, end);
      clf.analyze(H.analyserDb(fr), Float32Array.from(fr));
      const st = clf.getState();
      if (st.phase === 'calibrating') cal.push(st.calibration);
    }
    return { st: clf.getState(), cal };
  };
  // Absolute level matters here: 1.6e-5 RMS white noise ≈ a quiet bedroom on a
  // typical phone mic; 0.01 RMS (−40 dBFS) is a loud room.
  const quiet = run(1.6e-5), loud = run(0.01);
  assert.equal(typeof quiet.st.ambientGateDb, 'number');
  assert.equal(quiet.st.noisy, false, `quiet room flagged noisy (margin ${quiet.st.noiseMarginDb})`);
  assert.equal(loud.st.noisy, true, `loud room not flagged (margin ${loud.st.noiseMarginDb})`);
  assert.equal(loud.st.phase, 'listening', 'noisy is a hint, not the hard gate');
  assert.ok(quiet.cal.length > 10 && quiet.cal.every(c => c >= 0 && c <= 1));
  assert.ok(quiet.cal[quiet.cal.length - 1] > 0.9, 'calibration progress should approach 1 by 2 s');
  assert.equal(quiet.st.calibration, 1);
});
