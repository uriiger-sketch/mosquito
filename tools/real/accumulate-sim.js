#!/usr/bin/env node
'use strict';
/*
 * Offline simulation of burst accumulation on collected rows (collect.js).
 *
 * A track is a BURST once its plausibility-model score reaches θ on any frame
 * (instantaneous f0 recorded then). A DETECTION fires when ≥ K bursts within
 * W seconds agree in f0 within ±tol (relative). Detections of one f0 cluster
 * within 60 s are one event (the app's event rule).
 *
 *   node tools/real/accumulate-sim.js <rows.jsonl> <model.json> [--split dev|test]
 */
const fs = require('fs');
const T = require('./tune');
const H = require('../../tests/helpers');
const Tr = require('./train');

const HOP = H.HOP, SR = 44100;
const eng = H.loadEngine();
const vec = eng.MosquitoClassifier.plausVector;

function bursts(rows, model, theta) {
  const seen = new Set(), out = {};
  for (const r of rows) {
    if (r.veto || seen.has(r.key)) continue;
    if (Tr.predict(model, vec(r.x)) < theta) continue;
    seen.add(r.key);
    (out[r.src] = out[r.src] || []).push({ t: r.t, f: r.fNow });
  }
  for (const k of Object.keys(out)) out[k].sort((a, b) => a.t - b.t);
  return out;
}

function detections(bs, K, W, tol) {
  const evs = [];
  for (let i = 0; i < bs.length; i++) {
    const b = bs[i];
    let n = 0;
    for (let j = i; j >= 0 && b.t - bs[j].t <= W; j--) if (Math.abs(bs[j].f - b.f) <= tol * b.f) n++;
    if (n < K) continue;
    const ev = evs.find(e => b.t - e.last < 60 && Math.abs(e.f - b.f) <= 0.1 * b.f);
    if (ev) { ev.last = b.t; continue; }
    evs.push({ t: b.t, last: b.t, f: b.f });
  }
  return evs;
}

function main() {
  const [rowsFile, modelFile, ...rest] = process.argv.slice(2);
  const split = rest.includes('--split') ? rest[rest.indexOf('--split') + 1] : 'dev';
  const model = JSON.parse(fs.readFileSync(modelFile, 'utf8')).model;
  const rows = fs.readFileSync(rowsFile, 'utf8').trim().split('\n').map(l => JSON.parse(l));
  // audible-window denominators (independent estimator) per mosquito file
  const W10 = Math.round(10 * SR / HOP), wins = {}, hours = {};
  for (const f of T.FILES) {
    const d = T.load(f), mid = Math.floor(d.frames / 2), [a, b] = split === 'dev' ? [0, mid] : [mid, d.frames];
    hours[f.name] = (b - a) * HOP / SR / 3600;
    if (f.kind === 'background') continue;
    const lab = T.audibleFrames(d.spec, a, b), w = [];
    for (let s = 0; s + W10 <= lab.length; s += W10) { let c = 0; for (let i = s; i < s + W10; i++) c += lab[i]; w.push(c >= 3); }
    wins[f.name] = w;
  }
  const synthHours = new Set(rows.filter(r => r.kind === 'synthetic').map(r => r.src)).size * 3 / 60;
  const grid = [];
  for (const theta of [0.5, 0.6, 0.7, 0.8]) for (const K of [1, 2, 3, 4]) for (const W of [15, 30]) for (const tol of [0.03, 0.06]) grid.push({ theta, K, W, tol });
  const res = [];
  for (const g of grid) {
    if (g.K === 1 && (g.W !== 15 || g.tol !== 0.03)) continue;
    const bs = bursts(rows, model, g.theta);
    let pos = 0, hit = 0, bgEv = 0, bgH = 0, syEv = 0, fieldPos = 0, fieldHit = 0;
    for (const f of T.FILES) {
      const ev = detections(bs[f.name] || [], g.K, g.W, g.tol);
      if (f.kind === 'background') { bgEv += ev.length; bgH += hours[f.name]; continue; }
      const w = wins[f.name];
      // a window is detected if a detection fired (or stayed live) inside it
      const det = new Array(w.length).fill(false);
      for (const e of ev) for (let t = e.t; t <= e.last + 1e-9; t += 10) { const k = Math.floor(t / 10); if (k < det.length) det[k] = true; }
      const p = w.filter(Boolean).length, h = w.filter((x, k) => x && det[k]).length;
      if (f.kind === 'mosquito') { pos += p; hit += h; } else { fieldPos += p; fieldHit += h; }
    }
    for (const src of Object.keys(bs)) if (src.startsWith('synth:')) syEv += detections(bs[src], g.K, g.W, g.tol).length;
    res.push({ ...g, recallCDC: +(hit / pos).toFixed(3), recallField: +(fieldHit / fieldPos).toFixed(3),
               bgFAh: +(bgEv / bgH).toFixed(1), synthFAh: +(syEv / synthHours).toFixed(1) });
  }
  res.sort((a, b) => a.bgFAh - b.bgFAh || b.recallCDC - a.recallCDC);
  for (const r of res) console.log(JSON.stringify(r));
}

if (require.main === module) main();
