#!/usr/bin/env node
/*
 * Fit the Detection.confidence model — P(real flying mosquito | track
 * evidence) — by L2-regularised logistic regression (Platt-style) on the DEV
 * split, and print the weights to paste into MN_PL.CONF in js/detector.js.
 *
 *   node tools/eval/fit-confidence.js [--raw <dev raw.json>] [--workers N]
 *
 * Data (one row per app-level event, the unit the app counts / uploads):
 *   positives — the first lock of every mosquito trial (detection sweep,
 *               identification and mixture trials) and male-mosquito events;
 *   negatives — every false-alarm event in the adversary runs.
 * The class balance is the harness's natural mix (stated in EVALUATION.md):
 * P(real) is conditional on that exposure; the reported reliability is
 * measured on the TEST split with the same construction.
 */
'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');
const C = require('./corpus');
const St = require('./stats');

const a = process.argv.slice(2), get = (k, d) => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : d; };
let raw = get('--raw', null);
const keepRaw = a.includes('--keep');
if (!raw) {
  raw = path.join(os.tmpdir(), `mn-fitconf-${process.pid}.raw.json`);
  execFileSync(process.execPath, [path.join(__dirname, 'eval.js'), '--dev', '--split', 'dev', '--no-compute', '--quiet',
    '--workers', get('--workers', '4'), '--minutes', get('--minutes', '8'), '--out', raw + '.m.json', '--raw', raw], { stdio: 'inherit' });
}
const res = JSON.parse(fs.readFileSync(raw, 'utf8'));
const X = [], Y = [];
for (const r of res) {
  if (r.exp === 'fa') {
    const real = C.CLASSES[r.cls].countsAsFA === false ? 1 : 0;
    for (const e of r.events) if (e.x) { X.push(e.x); Y.push(real); }
  } else if (r.detected && r.first && r.first.x && (r.exp !== 'mix' || r.detectedMatch)) { X.push(r.first.x); Y.push(1); }
}
// --mask 1,0,0,1,0 keeps only the selected features (others fixed at weight 0).
const mask = get('--mask', null) ? get('--mask').split(',').map(Number) : null;
if (mask) for (const x of X) for (let j = 0; j < x.length; j++) if (!mask[j]) x[j] = 0;
const d = X[0].length, lam = +get('--l2', '1.0');
console.log(`rows ${X.length} (+${Y.filter(y => y).length} / −${Y.filter(y => !y).length}), features ${d}`);
// Newton–Raphson for L2-regularised logistic regression (intercept unpenalised).
let w = new Array(d + 1).fill(0);
for (let it = 0; it < 50; it++) {
  const g = new Array(d + 1).fill(0), H = Array.from({ length: d + 1 }, () => new Array(d + 1).fill(0));
  for (let i = 0; i < X.length; i++) {
    const x = [1, ...X[i]];
    const z = x.reduce((s, v, j) => s + v * w[j], 0), p = 1 / (1 + Math.exp(-z)), q = p * (1 - p);
    for (let j = 0; j <= d; j++) { g[j] += (Y[i] - p) * x[j]; for (let k = 0; k <= d; k++) H[j][k] += q * x[j] * x[k]; }
  }
  for (let j = 1; j <= d; j++) { g[j] -= lam * w[j]; H[j][j] += lam; }
  // solve H·δ = g (Gaussian elimination)
  const A = H.map((row, j) => [...row, g[j]]);
  for (let c = 0; c <= d; c++) {
    let piv = c; for (let r = c + 1; r <= d; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    [A[c], A[piv]] = [A[piv], A[c]];
    for (let r = 0; r <= d; r++) if (r !== c) { const f = A[r][c] / A[c][c]; for (let k = c; k <= d + 1; k++) A[r][k] -= f * A[c][k]; }
  }
  const delta = A.map((row, j) => row[d + 1] / row[j]);
  w = w.map((v, j) => v + delta[j]);
  // Monotonicity: more evidence above the CUSUM threshold must never LOWER
  // P(real). If the corpus says otherwise (loud adversaries), pin it to 0.
  if (w[1] < 0) { w[1] = 0; for (const x of X) x[0] = 0; }
  if (delta.reduce((s, v) => s + Math.abs(v), 0) < 1e-9) break;
}
const ps = X.map(x => 1 / (1 + Math.exp(-(w[0] + x.reduce((s, v, j) => s + v * w[j + 1], 0)))));
const rel = St.reliability(ps, Y, 10);
console.log('weights (MN_PL.CONF):', JSON.stringify(w.map(v => +v.toFixed(3))));
console.log(`in-sample Brier ${St.brierBinary(ps, Y).toFixed(4)}  ECE ${rel.ece.toFixed(4)}`);
for (const b of rel.bins) if (b.n) console.log(`  [${b.lo.toFixed(1)}, ${b.hi.toFixed(1)})  n=${String(b.n).padStart(5)}  mean p ${b.meanP.toFixed(3)}  freq ${b.freq.toFixed(3)}`);
