'use strict';
// Regression tests for defects found by the independent review of the v3
// engine: microphone silence, species-set changes under a lock, and
// sensitivity changes mid-session.
const test = require('node:test');
const assert = require('node:assert/strict');
const H = require('./helpers');

const eng = H.loadEngine();
const SILENT = new Float32Array(H.N / 2).fill(-Infinity);

// Drive a classifier frame by frame; `silentAt(t)` replaces the frame with the
// all −∞ spectrum a muted microphone produces. Returns per-frame detections.
function drive(signal, { silentAt = () => false, onFrame = () => {}, level = 3 } = {}) {
  const clf = new eng.MosquitoClassifier({ sampleRate: H.SR, fftSize: H.N, binHz: H.BIN_HZ });
  clf.setTimeOfDay('none');
  clf.setSensitivity(level);
  clf.beginCalibration();
  const calib = Math.round(2 * H.SR / H.HOP), out = [];
  let i = 0;
  for (let end = H.N; end <= signal.length; end += H.HOP, i++) {
    if (i === calib) clf.endCalibration();
    const t = end / H.SR;
    onFrame(clf, t, i);
    const fr = signal.subarray(end - H.N, end);
    out.push({ t, dets: silentAt(t) ? clf.analyze(SILENT, new Float32Array(H.N))
                                     : clf.analyze(H.analyserDb(fr), Float32Array.from(fr)) });
  }
  return out;
}

const firstDetection = (frames, after = 0) => {
  const f = frames.find(x => x.t >= after && x.dets.length);
  return f ? f.t : null;
};

function scene(seconds, toneFrom, f0 = 610) {
  const R = H.rng(11), len = Math.round(seconds * H.SR);
  return H.mix(H.whiteNoise(len, 0.003, R), H.mosquitoTone(len, f0, 0.02, R, Math.round(toneFrom * H.SR)));
}

test('digital silence during calibration does not deafen the detector', () => {
  const sig = scene(24, 10);
  const ctl = firstDetection(drive(sig), 10);
  const muted = firstDetection(drive(sig, { silentAt: t => t < 3 }), 10);
  assert.ok(ctl !== null, 'control never detected');
  assert.ok(muted !== null, 'never detected after a muted start (was deaf for ~19 min)');
  assert.ok(muted - ctl < 2, `detection delayed ${(muted - ctl).toFixed(1)} s by the muted start`);
});

test('a mid-session mute does not deafen the detector', () => {
  const sig = scene(60, 42);
  const t = firstDetection(drive(sig, { silentAt: t => t >= 10 && t < 40 }), 42);
  assert.ok(t !== null && t - 42 < 6, `first detection ${t === null ? 'never' : (t - 42).toFixed(1) + ' s'} after the tone began`);
});

test('a locked track is never re-labelled as a species that cannot produce its tone', () => {
  const sig = scene(20, 3, 617);
  let disabledAt = null;
  const frames = drive(sig, {
    onFrame: (clf, t) => {
      if (disabledAt === null && clf.getState().phase === 'detected') {
        disabledAt = t;                                   // leave only a 290 Hz species enabled
        clf.setDisabledSpecies(new Set(eng.SPECIES_DB.filter(s => s.id !== 'toxorhynchites').map(s => s.id)));
      }
    },
  });
  assert.ok(disabledAt !== null, 'never locked');
  const wrong = frames.filter(f => f.t > disabledAt).flatMap(f => f.dets).filter(d => d.species.id === 'toxorhynchites');
  assert.equal(wrong.length, 0, `${wrong.length} frames reported a 617 Hz tone as Toxorhynchites (290 Hz)`);
});

test('changing sensitivity mid-session keeps the noise floor and the lock', () => {
  const sig = scene(30, 2);
  let changedAt = null, trackBefore = null;
  const frames = drive(sig, {
    onFrame: (clf, t) => {
      if (changedAt === null && t > 15) {
        const d = clf.getState().detections[0];
        trackBefore = d ? d.trackId : null;
        changedAt = t;
        clf.setSensitivity(2);
      }
    },
  });
  assert.ok(trackBefore !== null, 'not locked before the change');
  const after = frames.filter(f => f.t > changedAt && f.t < changedAt + 3);
  const dropped = after.filter(f => !f.dets.length).length;
  assert.ok(dropped <= 1, `lock dropped for ${dropped} frames after setSensitivity`);
  assert.ok(after.some(f => f.dets.some(d => d.trackId === trackBefore)), 'the same track did not survive');
});
