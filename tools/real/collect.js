#!/usr/bin/env node
'use strict';
/*
 * Collect track-level training rows for the learned flight-plausibility model.
 *
 *   node tools/real/collect.js <out.jsonl> [--split dev|test] [--synth-minutes 3]
 *
 * One row per (vote-eligible track, frame): the plausibility features the
 * engine computes for that track (MosquitoClassifier._plausibility), the hard
 * physics vetoes it tripped, a label and a weight 1/(frames of that track), so
 * every TRACK contributes equally regardless of how long it lived.
 *
 *   label 1  tracks ≥ 300 Hz inside an independently "audible" 10-s window of a
 *            real mosquito recording (tools/real/tune.js audibleFrames)
 *   label 0  tracks in real non-mosquito recordings, and in synthetic adversary
 *            scenes (tools/eval/corpus.js) generated with the split's seeds —
 *            except midges and male mosquitoes, which are genuinely mosquito-like
 *
 * Features deliberately exclude f0 and absolute level (see train.js).
 */
const fs = require('fs');
const path = require('path');
const T = require('./tune');
const H = require('../../tests/helpers');
const C = require('../eval/corpus');
const S = require('../eval/synth');
const { rng, seedFrom, Analyser } = require('../lib/dsp');

const SR = 44100, N = H.N, HOP = H.HOP, BIN = SR / N, BINS = 1120;
const eng = H.loadEngine();
// Modulation measurable on quiet insects; tracks eligible from MIN_TRACK_S
// (override with --min-age) so sub-second flight bursts are represented.
let COLLECT_CFG = { STEADY_MIN_SNR: 4, STEADY_MIN_N: 5 };

function makeClf(loc) {
  const clf = new eng.MosquitoClassifier({ sampleRate: SR, fftSize: N, binHz: BIN });
  clf.setSensitivity(3);
  clf._tune(COLLECT_CFG);
  clf.setTimeOfDay('none');
  if (loc) clf.setLocation(loc[0], loc[1]);
  clf.beginCalibration();
  return clf;
}

const VETO = /^(sub|harmonic-comb|machine|pure)/;
function rowsFromFrame(clf, meta, label, rows, counts, inWindow, toneF0 = 0, tSec = 0) {
  for (const t of clf.ff.tracks) {
    if (!t.feat || t.age < clf.MIN_TRACK || t.T < clf.ff.TAU) continue;
    const f0 = clf._trackF0(t);
    if (!(f0 >= 300 && f0 <= 1000)) continue;
    void inWindow;
    const key = `${meta.src}#${t.id}`;
    counts.set(key, (counts.get(key) || 0) + 1);
    const f = t.feat;
    // Instantaneous track frequency (latest live frame): real insects glide at
    // ~65 Hz/s, so the track's multi-second mean f0 lags the tone.
    let fNow = f0;
    for (let j = t.hist.length - 1; j >= 0; j--) if (t.hist[j]) { fNow = t.hist[j].f; break; }
    rows.push({ key, y: label, src: meta.src, kind: meta.kind, t: +tSec.toFixed(3), fNow: +fNow.toFixed(1), win: inWindow ? 1 : 0, match: toneF0 > 0 && Math.abs(toneF0 - fNow) <= 20 ? 1 : 0,
                veto: (t.fails || []).some(q => VETO.test(q)),
                x: { pres: f.pres, dfRate: f.dfRate, subFrac: f.subFrac, combFrac: f.combFrac, dom: f.dom,
                     spread: f.spread, hRatio: f.hRatio, zf: f.zf, zl: f.zl, ubf: f.ubf, ubl: f.ubl,
                     slope: f.slope, stN: f.n, hasSt: f.zf !== undefined ? 1 : 0 } });
  }
}

function realRows(split, rows, counts) {
  const labels = T.labelsFor();
  for (const f of T.FILES) {
    const data = T.load(f), mid = Math.floor(data.frames / 2);
    const [from, to] = split === 'dev' ? [0, mid] : [mid, data.frames];
    const isMos = f.kind !== 'background';
    // Labelled culture recordings use tune.js's cached labels; field recordings
    // (all mosquito clips, species unlabelled) use the same independent estimator.
    // Per-frame f0 of the independently detected tone (0 = none). A track is a
    // POSITIVE only if on ≥ 30 % of its frames that tone sits within ±20 Hz of
    // the track: co-existing noise tracks in a mosquito recording are not.
    const toneF0 = new Float32Array(to - from);
    const audible = isMos ? T.audibleFrames(data.spec, from, to, toneF0) : null;
    void labels;
    const W = Math.round(10 * SR / HOP), win = new Uint8Array(to - from);
    if (isMos) for (let w = 0; w < win.length; w += W) {
      let a = 0; for (let i = w; i < Math.min(win.length, w + W); i++) a += audible[i];
      if (a >= 3) win.fill(1, w, Math.min(win.length, w + W));
    }
    const clf = makeClf(f.loc), full = new Float32Array(N / 2), calib = Math.round(2 * SR / HOP);
    for (let i = from; i < to; i++) {
      if (i - from === calib) clf.endCalibration();
      full.fill(-160); full.set(data.spec.subarray(i * BINS, i * BINS + BINS));
      clf.analyze(full, null);
      if (i - from < calib || data.excl[i]) continue;
      rowsFromFrame(clf, { src: f.name, kind: f.kind }, isMos ? 1 : 0, rows, counts, isMos && win[i - from], isMos ? toneF0[i - from] : 0, (i - from) * HOP / SR);
    }
    console.error(`  ${f.name}: rows so far ${rows.length}`);
  }
}

function synthRows(split, minutes, rows, counts) {
  const skip = new Set(['insect_midge', 'mosquito_male']);
  for (const [cls, def] of Object.entries(C.CLASSES)) {
    if (skip.has(cls)) continue;
    const seed = seedFrom(split, 'plaus', cls, 0);
    const scene = new S.Scene(SR, [new C.RoomNoise(SR, rng(seedFrom(seed, 'room'))), def.make(SR, rng(seed))]);
    const clf = makeClf(null), an = new Analyser(N), frame = new Float32Array(N);
    const hops = Math.floor(minutes * 60 * SR / HOP), calib = Math.round(2 * SR / HOP);
    let filled = 0, fi = 0;
    for (let h = 0; h < hops; h++) {
      const blk = scene.next(HOP);
      frame.copyWithin(0, HOP);
      for (let i = 0; i < HOP; i++) { const v = blk[i]; frame[N - HOP + i] = v > 1 ? 1 : v < -1 ? -1 : v; }
      filled += HOP; if (filled < N) continue;
      if (fi === calib) clf.endCalibration();
      clf.analyze(an.db(frame), frame);
      if (fi++ >= calib) rowsFromFrame(clf, { src: `synth:${cls}`, kind: 'synthetic' }, 0, rows, counts, false, 0, fi * HOP / SR);
    }
  }
  console.error(`  synthetic adversaries: rows so far ${rows.length}`);
}

function main() {
  const a = process.argv.slice(2);
  const out = a[0], opt = (k, d) => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : d; };
  const split = opt('--split', 'dev'), minutes = +opt('--synth-minutes', 3);
  if (opt('--min-age', null)) COLLECT_CFG = { ...COLLECT_CFG, MIN_TRACK_S: +opt('--min-age') };
  const rows = [], counts = new Map();
  realRows(split, rows, counts);
  if (minutes > 0) synthRows(split, minutes, rows, counts);
  const match = new Map();
  for (const r of rows) if (r.y) match.set(r.key, (match.get(r.key) || 0) + r.match);
  const fd = fs.openSync(out, 'w');
  for (const r of rows) {
    r.w = 1 / counts.get(r.key);
    if (r.y) r.trackMatch = +(match.get(r.key) / counts.get(r.key)).toFixed(3);
    fs.writeSync(fd, JSON.stringify(r) + '\n');
  }
  fs.closeSync(fd);
  const by = {};
  for (const r of rows) { const k = `${r.y}:${r.kind}`; by[k] = by[k] || { rows: 0, tracks: new Set() }; by[k].rows++; by[k].tracks.add(r.key); }
  console.error(Object.entries(by).map(([k, v]) => `${k} rows=${v.rows} tracks=${v.tracks.size}`).join('\n'));
}

if (require.main === module) main();
