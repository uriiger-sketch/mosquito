'use strict';
// Shared test helpers: synthetic audio framed exactly like the app
// (8192-sample window, hop 4096, Blackman dB spectrum like an AnalyserNode).
const loadEngine = require('../tools/lib/load-engine');

const SR = 44100, N = 8192, HOP = 4096, BIN_HZ = SR / N;

function rng(seed) {
  let s = seed >>> 0;
  const u = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s;
    t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const g = () => Math.sqrt(-2 * Math.log(u() || 1e-12)) * Math.cos(2 * Math.PI * u());
  return { u, g };
}

function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const br = re[b] * cr - im[b] * ci, bi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - br; im[b] = im[a] - bi; re[a] += br; im[a] += bi;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}

const BLACKMAN = Float64Array.from({ length: N }, (_, i) =>
  0.42 - 0.5 * Math.cos(2 * Math.PI * i / N) + 0.08 * Math.cos(4 * Math.PI * i / N));

function analyserDb(frame) {
  const re = new Float64Array(N), im = new Float64Array(N);
  for (let i = 0; i < N; i++) re[i] = frame[i] * BLACKMAN[i];
  fft(re, im);
  const out = new Float32Array(N / 2);
  for (let k = 0; k < N / 2; k++) {
    const mag = Math.hypot(re[k], im[k]) / N;
    out[k] = mag > 0 ? 20 * Math.log10(mag) : -Infinity;
  }
  return out;
}

function whiteNoise(len, rms, R) {
  const x = new Float32Array(len);
  for (let i = 0; i < len; i++) x[i] = rms * R.g();
  return x;
}

// Flying-mosquito tone: f0 with slow wander, harmonics at -6/-12 dB.
function mosquitoTone(len, f0, amp, R, start = 0) {
  const x = new Float32Array(len);
  let ph = 0, walk = 0;
  for (let i = start; i < len; i++) {
    walk += R.g() * 0.02; walk *= 0.9999;
    const f = f0 + 4 * Math.sin(2 * Math.PI * 0.6 * i / SR) + walk;
    ph += 2 * Math.PI * f / SR;
    x[i] = amp * (Math.sin(ph) + 0.5 * Math.sin(2 * ph + 0.3) + 0.25 * Math.sin(3 * ph + 1.1));
  }
  return x;
}

function mix(...xs) {
  const out = new Float32Array(xs[0].length);
  for (const x of xs) for (let i = 0; i < out.length; i++) out[i] += x[i];
  return out;
}

// Drive a classifier over a signal the way the app does (2 s calibration).
function runClassifier(signal, { level = 3, lat = null, lng = null, disabled } = {}) {
  const eng = loadEngine();
  const clf = new eng.MosquitoClassifier({
    binHz: BIN_HZ, sampleRate: SR,
    freqToBin: hz => Math.min(Math.round(hz / BIN_HZ), N / 2 - 1),
  });
  clf.getActivityFactor = () => 1.0;
  clf.setSensitivity(level);
  clf.setLocation(lat, lng);
  if (disabled) clf.setDisabledSpecies(disabled);
  clf.beginCalibration();
  const calibFrames = Math.round(2 * SR / HOP);
  const frames = [];
  let i = 0;
  for (let end = N; end <= signal.length; end += HOP, i++) {
    if (i === calibFrames) clf.endCalibration();
    const fr = signal.subarray(end - N, end);
    frames.push(clf.analyze(analyserDb(fr), Float32Array.from(fr)));
  }
  return { clf, frames, eng };
}

module.exports = { SR, N, HOP, BIN_HZ, rng, analyserDb, whiteNoise, mosquitoTone, mix, runClassifier, loadEngine };
