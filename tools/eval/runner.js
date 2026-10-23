'use strict';
/*
 * Drive a MosquitoClassifier over a synthetic scene exactly the way the app
 * does: an 8192-sample analysis window advanced by a 4096-sample hop, the
 * AnalyserNode Blackman dB spectrum + the raw time samples handed to
 * analyze(), REUSED buffers between calls (the app passes the same
 * Float32Arrays every frame), the 2-second startup calibration, and the
 * ADC's ±1 full-scale clipping.
 */
const { Analyser } = require('../lib/dsp');

// Construct + configure a classifier like App.startListening() does.
//  o: { sr, N, level, lat, lng, disabled: [ids] }
function makeClassifier(eng, o) {
  const N = o.N, binHz = o.sr / N;
  const engine = {
    binHz, sampleRate: o.sr, fftSize: N,
    freqToBin: (hz) => Math.min(Math.round(hz / binHz), N / 2 - 1),
  };
  const clf = new eng.MosquitoClassifier(engine);
  // The corpus has no time of day: neutralise any activity/time prior.
  if (typeof clf.setTimeOfDay === 'function') clf.setTimeOfDay('none');
  if (typeof clf.getActivityFactor === 'function') clf.getActivityFactor = () => 1.0;
  clf.setLocation(o.lat ?? null, o.lng ?? null);
  clf.setDisabledSpecies(new Set(o.disabled || []));
  clf.setSensitivity(o.level || 3);
  // Tuning sweeps (dev split only): MN_TUNE='{"MIN_TRACK_S":1.6}' overrides engine constants.
  if (process.env.MN_TUNE && typeof clf._tune === 'function') clf._tune(JSON.parse(process.env.MN_TUNE));
  return clf;
}

//  o: { sr, N, hop, seconds, calibSec, timing, onFrame(frameIdx, tEndSec, dets, clf) }
function drive(clf, scene, o) {
  const sr = o.sr, N = o.N, hop = o.hop || N / 2;
  const an = new Analyser(N), frame = new Float32Array(N);
  const nHops = Math.floor(o.seconds * sr / hop);
  const calibFrames = Math.round((o.calibSec ?? 2) * sr / hop);
  const times = o.timing ? [] : null;
  clf.beginCalibration();
  let filled = 0, fi = 0;
  for (let h = 0; h < nHops; h++) {
    const blk = scene.next(hop);
    frame.copyWithin(0, hop);
    const base = N - hop;
    for (let i = 0; i < hop; i++) { const v = blk[i]; frame[base + i] = v > 1 ? 1 : v < -1 ? -1 : v; }
    filled += hop;
    if (filled < N) continue;
    if (fi === calibFrames) clf.endCalibration();
    const db = an.db(frame);
    let dets;
    if (times) {
      const t0 = process.hrtime.bigint();
      dets = clf.analyze(db, frame);
      times.push(Number(process.hrtime.bigint() - t0) / 1e6);
    } else dets = clf.analyze(db, frame);
    o.onFrame(fi, (h + 1) * hop / sr, dets || [], clf);
    fi++;
  }
  return { frames: fi, times, hopSec: hop / sr };
}

// Compact, engine-agnostic view of a Detection.
function summarize(d) {
  const post = (d.posterior || []).map(p => [p.id, +p.p.toFixed(5)]);
  let gpost = null;
  if (Array.isArray(d.genusPosterior)) gpost = d.genusPosterior.map(g => [g.genus, +g.p.toFixed(5)]);
  return {
    sp: d.species.id, genus: d.genus || d.species.genus,
    f0: +(d.f0 ?? d.frequency).toFixed(2), conf: +(+d.confidence).toFixed(4),
    snr: +(+d.snr).toFixed(2), range: d.range, post, gpost,
  };
}

// App event logic (index.html analysis loop): a species' detection is a NEW
// event when its last event was ≥ 60 s ago (continuous locks re-fire every 60 s).
class EventCounter {
  constructor(cooldownSec = 60) { this.cd = cooldownSec; this.last = {}; this.events = []; }
  push(tSec, dets) {
    for (const d of dets) {
      const id = d.species.id;
      if (this.last[id] === undefined || tSec - this.last[id] >= this.cd) {
        this.last[id] = tSec;
        this.events.push({ t: +tSec.toFixed(2), ...summarize(d) });
      }
    }
  }
}

module.exports = { makeClassifier, drive, summarize, EventCounter };
