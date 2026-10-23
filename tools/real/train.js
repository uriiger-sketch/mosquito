#!/usr/bin/env node
'use strict';
/*
 * Fit the learned flight-plausibility model on rows from tools/real/collect.js.
 *
 *   node tools/real/train.js <rows-dev.jsonl> [--out model.json]
 *
 * L2-regularised logistic regression on MosquitoClassifier.plausVector(feat)
 * (the exact transform the app uses), standardised. Only rows that pass the
 * hard physics vetoes are used — the model decides only what the vetoes leave.
 * Weights: every track counts once (row weight 1/frames); the positive class,
 * the real-background negatives and the synthetic-adversary negatives are
 * rebalanced to 1/2, 1/4, 1/4 of the total mass. λ is chosen by 5-fold
 * cross-validation GROUPED BY TRACK (frames of one insect never straddle folds),
 * scoring the weighted log-loss.
 */
const fs = require('fs');
const H = require('../../tests/helpers');

const eng = H.loadEngine();
const vec = eng.MosquitoClassifier.plausVector;

function load(file) {
  // Positives: tracks in real mosquito recordings whose instantaneous f0 matched
  // the independently detected tone on ≥ 30 % of their frames.
  const rows = fs.readFileSync(file, 'utf8').trim().split('\n').map(l => JSON.parse(l))
    .filter(r => !r.veto && (!r.y || r.trackMatch === undefined || r.trackMatch >= 0.3));
  const mass = { pos: 0, bg: 0, syn: 0 };
  const grp = (r) => (r.y ? 'pos' : r.kind === 'synthetic' ? 'syn' : 'bg');
  for (const r of rows) mass[grp(r)] += r.w;
  const target = { pos: 0.5, bg: 0.25, syn: 0.25 };
  return rows.map(r => ({ x: vec(r.x), y: r.y, w: r.w * target[grp(r)] / mass[grp(r)] * rows.length, key: r.key, kind: r.kind }));
}

function standardise(rows) {
  const d = rows[0].x.length, mu = new Array(d).fill(0), sd = new Array(d).fill(0);
  let W = 0;
  for (const r of rows) { W += r.w; for (let j = 0; j < d; j++) mu[j] += r.w * r.x[j]; }
  for (let j = 0; j < d; j++) mu[j] /= W;
  for (const r of rows) for (let j = 0; j < d; j++) sd[j] += r.w * (r.x[j] - mu[j]) ** 2;
  for (let j = 0; j < d; j++) sd[j] = Math.sqrt(sd[j] / W) || 1;
  return { mu, sd };
}

const sig = (z) => 1 / (1 + Math.exp(-z));

// Newton–Raphson (IRLS) with an L2 penalty on the non-bias weights.
function fit(rows, mu, sd, lambda) {
  const d = mu.length + 1;
  const X = rows.map(r => [1, ...r.x.map((v, j) => (v - mu[j]) / sd[j])]);
  let w = new Array(d).fill(0);
  const Wsum = rows.reduce((s, r) => s + r.w, 0);
  for (let it = 0; it < 50; it++) {
    const g = new Array(d).fill(0), Hm = Array.from({ length: d }, () => new Array(d).fill(0));
    for (let i = 0; i < rows.length; i++) {
      const xi = X[i], p = sig(xi.reduce((s, v, j) => s + v * w[j], 0)), wi = rows[i].w / Wsum;
      for (let a = 0; a < d; a++) {
        g[a] += wi * (p - rows[i].y) * xi[a];
        for (let b = 0; b <= a; b++) Hm[a][b] += wi * p * (1 - p) * xi[a] * xi[b];
      }
    }
    for (let a = 1; a < d; a++) { g[a] += lambda * w[a]; Hm[a][a] += lambda; }
    for (let a = 0; a < d; a++) for (let b = a + 1; b < d; b++) Hm[a][b] = Hm[b][a];
    const step = solve(Hm, g);
    let mx = 0;
    w = w.map((v, j) => { mx = Math.max(mx, Math.abs(step[j])); return v - step[j]; });
    if (mx < 1e-8) break;
  }
  return w;
}

function solve(A, b) {                       // Gaussian elimination with partial pivoting
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const v = M[c][c] || 1e-12;
    for (let r = c + 1; r < n; r++) { const f = M[r][c] / v; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / (M[r][r] || 1e-12); }
  return x;
}

const predict = (m, x) => sig(m.w[0] + x.reduce((s, v, j) => s + m.w[j + 1] * (v - m.mu[j]) / m.sd[j], 0));

function logloss(m, rows) {
  let L = 0, W = 0;
  for (const r of rows) { const p = Math.min(1 - 1e-9, Math.max(1e-9, predict(m, r.x))); L -= r.w * (r.y ? Math.log(p) : Math.log(1 - p)); W += r.w; }
  return L / W;
}

// Track-level AUC (mean score over each track's frames) per negative family.
function auc(m, rows, negKind) {
  const tr = new Map();
  for (const r of rows) { const t = tr.get(r.key) || { y: r.y, kind: r.kind, s: 0, n: 0 }; t.s += predict(m, r.x); t.n++; tr.set(r.key, t); }
  const pos = [], neg = [];
  for (const t of tr.values()) { if (t.y) pos.push(t.s / t.n); else if (!negKind || t.kind === negKind) neg.push(t.s / t.n); }
  let c = 0; for (const p of pos) for (const q of neg) c += p > q ? 1 : p === q ? 0.5 : 0;
  return +(c / (pos.length * neg.length || 1)).toFixed(3);
}

function hash(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }

function main() {
  const a = process.argv.slice(2), opt = (k, d) => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : d; };
  const rows = load(a[0]);
  console.error(`${rows.length} rows (vetoed rows excluded), ${new Set(rows.map(r => r.key)).size} tracks`);
  const K = 5, fold = (r) => hash(r.key) % K;
  let best = null;
  for (const lambda of [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]) {
    let L = 0;
    for (let k = 0; k < K; k++) {
      const tr = rows.filter(r => fold(r) !== k), te = rows.filter(r => fold(r) === k);
      const { mu, sd } = standardise(tr);
      L += logloss({ mu, sd, w: fit(tr, mu, sd, lambda) }, te) / K;
    }
    console.error(`λ=${lambda}: CV log-loss ${L.toFixed(4)}`);
    if (!best || L < best.L) best = { lambda, L };
  }
  const { mu, sd } = standardise(rows);
  const m = { mu, sd, w: fit(rows, mu, sd, best.lambda), lambda: best.lambda };
  // Out-of-fold AUCs at the chosen λ (honest discrimination estimate on dev).
  const oof = rows.map(r => ({ ...r }));
  for (let k = 0; k < K; k++) {
    const tr = rows.filter(r => fold(r) !== k); const s = standardise(tr);
    const mk = { ...s, w: fit(tr, s.mu, s.sd, best.lambda) };
    for (const r of oof) if (fold(r) === k) r.oofP = predict(mk, r.x);
  }
  const tAUC = (negKind) => {
    const tr = new Map();
    for (const r of oof) { const t = tr.get(r.key) || { y: r.y, kind: r.kind, s: 0, n: 0 }; t.s += r.oofP; t.n++; tr.set(r.key, t); }
    const pos = [], neg = [];
    for (const t of tr.values()) { if (t.y) pos.push(t.s / t.n); else if (!negKind || t.kind === negKind) neg.push(t.s / t.n); }
    let c = 0; for (const p of pos) for (const q of neg) c += p > q ? 1 : p === q ? 0.5 : 0;
    return +(c / (pos.length * neg.length || 1)).toFixed(3);
  };
  const names = ['pres', 'log pitch rate', 'subFrac', 'combFrac', 'log dom', 'spread', 'log hRatio', 'hasMod', 'z pitch-mod', 'z level-mod', 'log ubf', 'ubl', 'slope'];
  const report = { lambda: best.lambda, cvLogLoss: +best.L.toFixed(4),
                   oofTrackAUC: { all: tAUC(), realBackground: tAUC('background'), synthetic: tAUC('synthetic') },
                   inSampleTrackAUC: { all: auc(m, rows), realBackground: auc(m, rows, 'background'), synthetic: auc(m, rows, 'synthetic') },
                   coefficients: Object.fromEntries(names.map((n, j) => [n, +(m.w[j + 1] / m.sd[j]).toFixed(4)])) };
  console.error(JSON.stringify(report, null, 1));
  const model = { mu: m.mu.map(v => +v.toFixed(6)), sd: m.sd.map(v => +v.toFixed(6)), w: m.w.map(v => +v.toFixed(6)) };
  const out = opt('--out', null);
  if (out) fs.writeFileSync(out, JSON.stringify({ model, report }, null, 1));
  else console.log(JSON.stringify(model));
}

if (require.main === module) main();
module.exports = { load, fit, standardise, predict };
