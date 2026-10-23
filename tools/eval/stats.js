'use strict';
/*
 * Statistics for the evaluation harness (no dependencies).
 *
 *  wilson(k, n)            Wilson score 95 % interval for a binomial proportion.
 *  poissonRateCI(k, T)     exact (Garwood) 95 % interval for a Poisson rate k/T.
 *  logisticFit(x, y)       MLE of P(y=1|x) = σ(a·(x − x50)) by Newton/IRLS with a
 *                          weak ridge (keeps separable data finite); returns x50
 *                          and its delta-method standard error.
 *  brier / ece / reliability   calibration of probabilistic predictions.
 */

const Z95 = 1.959964;

function wilson(k, n, z = Z95) {
  if (!n) return [0, 1];
  const p = k / n, d = 1 + z * z / n;
  const c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}

// ── regularized lower incomplete gamma P(a, x) (Numerical Recipes §6.2) ──────
function lgamma(x) {
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x, t = x + 5.5; t -= (x + 0.5) * Math.log(t);
  let s = 1.000000000190015;
  for (const ci of c) s += ci / ++y;
  return -t + Math.log(2.5066282746310005 * s / x);
}
function gammaP(a, x) {
  if (x <= 0) return 0;
  if (x < a + 1) {
    let ap = a, sum = 1 / a, del = sum;
    for (let n = 0; n < 500; n++) { ap++; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-14) break; }
    return sum * Math.exp(-x + a * Math.log(x) - lgamma(a));
  }
  let b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
  for (let i = 1; i < 500; i++) {
    const an = -i * (i - a); b += 2;
    d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300;
    c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300;
    d = 1 / d; const del = d * c; h *= del;
    if (Math.abs(del - 1) < 1e-14) break;
  }
  return 1 - Math.exp(-x + a * Math.log(x) - lgamma(a)) * h;
}
// Quantile of Gamma(a, 1) by bisection.
function gammaQuantile(a, p) {
  let lo = 0, hi = Math.max(10, a * 10 + 50);
  for (let i = 0; i < 200; i++) { const m = 0.5 * (lo + hi); if (gammaP(a, m) < p) lo = m; else hi = m; }
  return 0.5 * (lo + hi);
}
// Exact Poisson CI for the mean count, scaled to a rate per unit exposure T.
function poissonRateCI(k, T, alpha = 0.05) {
  const lo = k === 0 ? 0 : gammaQuantile(k, alpha / 2);
  const hi = gammaQuantile(k + 1, 1 - alpha / 2);
  return [lo / T, hi / T];
}

// ── logistic regression in one variable ──────────────────────────────────────
// Parameterised as logit p = b0 + b1·x. Ridge λ on b1 (weak prior) only.
function logisticFit(xs, ys, { ridge = 1e-3 } = {}) {
  const n = xs.length;
  if (!n) return null;
  const k = ys.reduce((s, y) => s + y, 0);
  const xm = xs.reduce((s, x) => s + x, 0) / n;
  let b0 = 0, b1 = 0.3;
  for (let it = 0; it < 100; it++) {
    let g0 = 0, g1 = -ridge * b1, h00 = 0, h01 = 0, h11 = ridge;
    for (let i = 0; i < n; i++) {
      const x = xs[i] - xm, p = 1 / (1 + Math.exp(-(b0 + b1 * x))), w = p * (1 - p);
      g0 += ys[i] - p; g1 += (ys[i] - p) * x;
      h00 += w; h01 += w * x; h11 += w * x * x;
    }
    h00 += 1e-9;
    const det = h00 * h11 - h01 * h01;
    if (!(det > 0)) break;
    const d0 = (h11 * g0 - h01 * g1) / det, d1 = (h00 * g1 - h01 * g0) / det;
    b0 += d0; b1 += d1;
    if (Math.abs(d0) + Math.abs(d1) < 1e-10) break;
  }
  // x50 = xm − b0/b1; delta-method SE from the inverse Hessian.
  let h00 = 0, h01 = 0, h11 = ridge;
  for (let i = 0; i < n; i++) {
    const x = xs[i] - xm, p = 1 / (1 + Math.exp(-(b0 + b1 * x))), w = p * (1 - p);
    h00 += w; h01 += w * x; h11 += w * x * x;
  }
  const det = h00 * h11 - h01 * h01;
  const v00 = h11 / det, v01 = -h01 / det, v11 = h00 / det;
  const x50 = xm - b0 / b1;
  const gA = -1 / b1, gB = b0 / (b1 * b1);
  const se = Math.sqrt(Math.max(0, gA * gA * v00 + 2 * gA * gB * v01 + gB * gB * v11));
  return { x50, se, slope: b1, k, n, all: k === n, none: k === 0 };
}

// ── calibration ──────────────────────────────────────────────────────────────
// Binary: probs p_i, outcomes y_i ∈ {0,1}.
function brierBinary(ps, ys) {
  if (!ps.length) return null;
  let s = 0; for (let i = 0; i < ps.length; i++) s += (ps[i] - ys[i]) ** 2;
  return s / ps.length;
}
// Multiclass Brier: Σ_c (p_c − 1[c = y])², averaged (range 0…2).
function brierMulti(dists, trueIds) {
  if (!dists.length) return null;
  let s = 0;
  for (let i = 0; i < dists.length; i++) {
    const d = dists[i] || {}; let tot = 0, hit = 0;
    for (const [id, p] of Object.entries(d)) { tot += p * p; if (id === trueIds[i]) hit = p; }
    s += tot - 2 * hit + 1;
  }
  return s / dists.length;
}
// Reliability bins on [0,1] (equal width) + expected calibration error.
function reliability(ps, ys, nBins = 10) {
  const bins = Array.from({ length: nBins }, (_, i) => ({ lo: i / nBins, hi: (i + 1) / nBins, n: 0, sumP: 0, sumY: 0 }));
  for (let i = 0; i < ps.length; i++) {
    const b = bins[Math.min(nBins - 1, Math.max(0, Math.floor(ps[i] * nBins)))];
    b.n++; b.sumP += ps[i]; b.sumY += ys[i];
  }
  let ece = 0;
  const out = bins.map(b => {
    const meanP = b.n ? b.sumP / b.n : null, freq = b.n ? b.sumY / b.n : null;
    if (b.n) ece += (b.n / ps.length) * Math.abs(meanP - freq);
    return { lo: b.lo, hi: b.hi, n: b.n, meanP, freq };
  });
  return { ece: ps.length ? ece : null, n: ps.length, bins: out };
}

// Weighted isotonic (non-decreasing) regression — pool-adjacent-violators.
function isotonic(ys, ws) {
  const blocks = [];
  for (let i = 0; i < ys.length; i++) {
    blocks.push({ v: ys[i], w: ws ? ws[i] : 1, n: 1 });
    while (blocks.length > 1 && blocks[blocks.length - 2].v > blocks[blocks.length - 1].v) {
      const b = blocks.pop(), a = blocks.pop(), w = a.w + b.w;
      blocks.push({ v: (a.v * a.w + b.v * b.w) / w, w, n: a.n + b.n });
    }
  }
  const out = [];
  for (const b of blocks) for (let i = 0; i < b.n; i++) out.push(b.v);
  return out;
}

const mean = (a) => a.length ? a.reduce((s, x) => s + x, 0) / a.length : null;
function quantile(a, q) {
  if (!a.length) return null;
  const s = [...a].sort((x, y) => x - y), i = (s.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
  return s[lo] + (s[hi] - s[lo]) * (i - lo);
}

module.exports = { wilson, poissonRateCI, gammaP, gammaQuantile, logisticFit, isotonic, brierBinary, brierMulti, reliability, mean, quantile };
