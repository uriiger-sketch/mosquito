'use strict';
/*
 * Shared DSP utilities for tests/ and tools/ (Node only, no dependencies).
 *
 *  • rng(seed)            — deterministic mulberry32 uniform + Gaussian generator
 *  • Analyser(N)          — bit-faithful emulation of the Web Audio AnalyserNode
 *                           float spectrum with smoothingTimeConstant = 0:
 *                           Blackman window (α = 0.16), |X[k]|/N, 20·log10.
 *                           (W3C Web Audio API §1.8.6 "FFT windowing and smoothing")
 *  • fftComplex / realPower — radix-2 FFT helpers
 *  • seedFrom(...parts)   — stable 32-bit seed from strings / numbers (FNV-1a)
 */

// ── RNG ──────────────────────────────────────────────────────────────────────
// mulberry32 (same generator as the legacy bench, so seeds stay comparable).
// g() is Box–Muller with the second variate cached (2× fewer log/sqrt calls).
function rng(seed) {
  let s = seed >>> 0;
  const u = () => {
    s = (s + 0x6D2B79F5) >>> 0; let t = s;
    t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
  let spare = 0, has = false;
  const g = () => {
    if (has) { has = false; return spare; }
    const r = Math.sqrt(-2 * Math.log(u() || 1e-12)), th = 2 * Math.PI * u();
    spare = r * Math.sin(th); has = true;
    return r * Math.cos(th);
  };
  const range = (a, b) => a + (b - a) * u();
  const logRange = (a, b) => Math.exp(Math.log(a) + (Math.log(b) - Math.log(a)) * u());
  const int = (a, b) => a + Math.floor(u() * (b - a + 1));         // inclusive
  const pick = (arr) => arr[Math.floor(u() * arr.length)];
  return { u, g, range, logRange, int, pick };
}

// FNV-1a over the string form of the parts → 32-bit seed.
function seedFrom(...parts) {
  let h = 0x811c9dc5;
  const s = parts.join('|');
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}

// ── FFT ──────────────────────────────────────────────────────────────────────
// In-place iterative radix-2 complex FFT (forward, e^{-jωn}).
function fftComplex(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang), h = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < h; k++) {
        const a = i + k, b = a + h;
        const br = re[b] * cr - im[b] * ci, bi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - br; im[b] = im[a] - bi; re[a] += br; im[a] += bi;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}

// Fast real FFT of length N (power of 2) via one complex FFT of length N/2 with
// precomputed bit-reversal and twiddle tables. Writes |X[k]|² for k < N/2.
class RealFFT {
  constructor(N) {
    if (N & (N - 1)) throw new Error('N must be a power of two');
    this.N = N; const M = this.M = N >> 1;
    this.re = new Float64Array(M); this.im = new Float64Array(M);
    this.rev = new Uint32Array(M);
    for (let i = 1, j = 0; i < M; i++) {
      let bit = M >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit; this.rev[i] = j;
    }
    this.twr = new Float64Array(M >> 1); this.twi = new Float64Array(M >> 1);
    for (let k = 0; k < (M >> 1); k++) { this.twr[k] = Math.cos(-2 * Math.PI * k / M); this.twi[k] = Math.sin(-2 * Math.PI * k / M); }
    this.splr = new Float64Array(M); this.spli = new Float64Array(M);
    for (let k = 0; k < M; k++) { this.splr[k] = Math.cos(-2 * Math.PI * k / N); this.spli[k] = Math.sin(-2 * Math.PI * k / N); }
  }
  // x: windowed real input (length N). pow: output |X[k]|², length N/2.
  power(x, pow) {
    const { M, re, im, rev, twr, twi, splr, spli } = this;
    for (let i = 0; i < M; i++) { const r = rev[i]; re[r] = x[2 * i]; im[r] = x[2 * i + 1]; }
    for (let len = 2, step = M >> 1; len <= M; len <<= 1, step >>= 1) {
      const h = len >> 1;
      for (let i = 0; i < M; i += len) {
        for (let k = 0, t = 0; k < h; k++, t += step) {
          const a = i + k, b = a + h, cr = twr[t], ci = twi[t];
          const br = re[b] * cr - im[b] * ci, bi = re[b] * ci + im[b] * cr;
          re[b] = re[a] - br; im[b] = im[a] - bi; re[a] += br; im[a] += bi;
        }
      }
    }
    for (let k = 0; k < M; k++) {
      const m = k === 0 ? 0 : M - k;
      const rk = re[k], ik = im[k], rm = re[m], imm = im[m];
      const fer = 0.5 * (rk + rm), fei = 0.5 * (ik - imm);
      const for_ = 0.5 * (ik + imm), foi = -0.5 * (rk - rm);
      const wr = splr[k], wi = spli[k];
      const xr = fer + wr * for_ - wi * foi, xi = fei + wr * foi + wi * for_;
      pow[k] = xr * xr + xi * xi;
    }
    return pow;
  }
}

// ── AnalyserNode emulation ───────────────────────────────────────────────────
const _blackman = new Map();
function blackman(N) {
  if (!_blackman.has(N)) {
    _blackman.set(N, Float64Array.from({ length: N }, (_, i) =>
      0.42 - 0.5 * Math.cos(2 * Math.PI * i / N) + 0.08 * Math.cos(4 * Math.PI * i / N)));
  }
  return _blackman.get(N);
}

class Analyser {
  constructor(N = 8192) {
    this.N = N; this.fft = new RealFFT(N); this.win = blackman(N);
    this.x = new Float64Array(N); this.pow = new Float64Array(N / 2);
    this.out = new Float32Array(N / 2);                 // reused, like the app's rawFreqBuf
    this.dbOff = -20 * Math.log10(N);
  }
  // frame: N time samples → Float32Array(N/2) of dBFS (REUSED between calls).
  db(frame, out = this.out) {
    const { N, x, win, pow } = this;
    for (let i = 0; i < N; i++) x[i] = frame[i] * win[i];
    this.fft.power(x, pow);
    const off = this.dbOff;
    for (let k = 0; k < N / 2; k++) out[k] = pow[k] > 0 ? 10 * Math.log10(pow[k]) + off : -Infinity;
    return out;
  }
}

// One-shot convenience (allocates): same result as new Analyser(N).db(frame).
function analyserDb(frame, N = frame.length) {
  return Float32Array.from(new Analyser(N).db(frame));
}

module.exports = { rng, seedFrom, fftComplex, RealFFT, Analyser, analyserDb, blackman };
