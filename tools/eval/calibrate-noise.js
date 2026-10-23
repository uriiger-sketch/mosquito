#!/usr/bin/env node
/*
 * Measures the estimator noise of the tracker's per-frame f0 and level for a
 * perfectly STEADY tone in white noise, as a function of the per-frame SNR
 * s1/n1 (3-bin main-lobe signal power over per-bin noise power). The fitted
 * constants go into MosquitoClassifier._steadiness:
 *   var(f)/binHz² ≈ A_f / snr        var(level, dB²) ≈ A_l / snr + B_l / snr²
 *
 *   node tools/eval/calibrate-noise.js [--sr 44100]
 */
'use strict';
const { rng, Analyser } = require('../lib/dsp');
const loadEngine = require('../lib/load-engine');

const a = process.argv.slice(2), get = (k, d) => { const i = a.indexOf(k); return i >= 0 ? +a[i + 1] : d; };
const sr = get('--sr', 44100), N = 8192, hop = N / 2, binHz = sr / N;
const eng = loadEngine();
const rows = [];
for (const inbin of [3, 6, 9, 12, 15, 18, 21, 25, 30, 35]) {
  const fs = [], ls = [], snrs = [];
  for (let trial = 0; trial < 12; trial++) {
    const R = rng(1000 + trial * 31 + inbin);
    const f0 = 300 + 500 * R.u();
    // white noise σ: per-bin noise power (Blackman, |X|/N) = σ²·Σw²/N² ; tone peak power ≈ (A/2)²·(Σw/N)²
    const sigma = 0.01, Sw2 = 0.3046 * N, Sw = 0.42 * N;
    const nBin = sigma * sigma * Sw2 / (N * N);
    const A = 2 * Math.sqrt(nBin * Math.pow(10, inbin / 10)) * N / Sw;
    const tr = new eng.FarFieldTracker(binHz, hop / sr);
    tr.setSensitivity(1);
    const an = new Analyser(N), frame = new Float32Array(N);
    let ph = R.u() * 6.28, n = 0;
    const hist = [];
    for (let h = 0; h < 120; h++) {
      frame.copyWithin(0, hop);
      const on = h >= 25 ? 1 : 0;   // tone starts after the noise-floor calibration
      for (let i = N - hop; i < N; i++) { ph += 2 * Math.PI * f0 / sr; frame[i] = on * (A * Math.sin(ph) + 0.25 * A * Math.sin(2 * ph + 1)) + sigma * R.g(); }
      if (++n < 2) continue;
      tr.process(an.db(frame));
      const t = tr.tracks.find(x => Math.abs(x.k * binHz - f0) < 3 * binHz);
      if (h > 45 && t && t.hist.length && t.hist[t.hist.length - 1]) hist.push(t.hist[t.hist.length - 1]);
    }
    if (hist.length < 30) continue;
    const mf = hist.reduce((s, x) => s + x.f, 0) / hist.length, ml = hist.reduce((s, x) => s + x.lv, 0) / hist.length;
    fs.push(hist.reduce((s, x) => s + (x.f - mf) ** 2, 0) / (hist.length - 1));
    ls.push(hist.reduce((s, x) => s + (x.lv - ml) ** 2, 0) / (hist.length - 1));
    snrs.push(hist.reduce((s, x) => s + x.s1 / x.n1, 0) / hist.length);
  }
  if (!fs.length) continue;
  const m = (v) => v.reduce((s, x) => s + x, 0) / v.length;
  rows.push({ inbin, snr: m(snrs), vf: m(fs) / binHz ** 2, vl: m(ls) });
}
console.log(' in-bin dB   snr(s1/n1)   var(f)/bin²   ×snr     var(lv) dB²   ×snr');
for (const r of rows) console.log(`${String(r.inbin).padStart(8)}   ${r.snr.toFixed(1).padStart(10)}   ${r.vf.toExponential(3).padStart(11)}  ${(r.vf * r.snr).toFixed(3).padStart(6)}   ${r.vl.toExponential(3).padStart(11)}  ${(r.vl * r.snr).toFixed(2).padStart(6)}`);
