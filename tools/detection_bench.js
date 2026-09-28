#!/usr/bin/env node
/*
 * Detection-range benchmark for the MosquitoNet classifier.
 *
 * Runs the REAL MosquitoClassifier (extracted from index.html) on synthetic
 * audio framed exactly like the app (8192-sample window, hop 4096, Blackman
 * dB spectrum like an AnalyserNode) and measures:
 *
 *   • detection probability vs signal-to-noise ratio (SNR sweep)
 *   • false alarms on noise-only, a machine tone, and voiced-speech-like input
 *
 * Physics: beyond the acoustic near field (~λ/2π ≈ 8-14 cm at 400-700 Hz) the
 * flight tone's pressure falls as 1/r, i.e. -6 dB per doubling of distance, and
 * air absorption at these frequencies is negligible. So a detector that works
 * at an SNR X dB lower has a detection range 10^(X/20) times longer.
 *
 * Usage:  node tools/detection_bench.js [--trials N] [--quick]
 */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const SR = 44100, N = 8192, HOP = 4096;
const BIN_HZ = SR / N;

// ── Extract the classifier (and what it depends on) from index.html ─────────
const HTML = (() => { const i = process.argv.indexOf('--html');
  return i >= 0 ? path.resolve(process.argv[i + 1]) : path.join(__dirname, '..', 'index.html'); })();

function loadClassifier() {
  const src = fs.readFileSync(HTML, 'utf8');
  const grab = (startMark, endMark) => {
    const a = src.indexOf(startMark);
    const b = src.indexOf(endMark, a + startMark.length);
    if (a < 0 || b < 0) throw new Error('cannot extract ' + startMark);
    return src.slice(a, b);
  };
  const code = [
    grab('const SPECIES_DB = [', '\nfunction getNativeSpecies'),
    src.includes('class FarFieldTracker') ? grab('class FarFieldTracker', '\nclass MosquitoClassifier') : '',
    grab('class MosquitoClassifier', '\nclass FederatedLearningClient'),
    'this.SPECIES_DB = SPECIES_DB; this.MosquitoClassifier = MosquitoClassifier;',
  ].join('\n');
  const quiet = { log: (...a) => { if (!String(a[0]).startsWith('[Sensitivity]')) console.log(...a); },
                  warn: console.warn, error: console.error };
  const ctx = {
    window: { _app: null }, console: quiet, Math, Float32Array, Float64Array, Uint8Array,
    Int32Array, Array, Object, Number, Set, Map, isFinite, Infinity, NaN,
    timeOfDayCategory: () => 'day',
  };
  vm.createContext(ctx);
  vm.runInContext(code, ctx);
  return ctx;
}

// ── DSP helpers ──────────────────────────────────────────────────────────────
function fftPow(re, im) {                       // in-place radix-2, returns nothing
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
        const ar = re[i + k], ai = im[i + k];
        const br = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const bi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ar + br; im[i + k] = ai + bi;
        re[i + k + len / 2] = ar - br; im[i + k + len / 2] = ai - bi;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}
const BLACKMAN = new Float64Array(N).map((_, i) =>
  0.42 - 0.5 * Math.cos(2 * Math.PI * i / N) + 0.08 * Math.cos(4 * Math.PI * i / N));

function analyserDb(frame) {                    // mimics AnalyserNode.getFloatFrequencyData
  const re = new Float64Array(N), im = new Float64Array(N);
  for (let i = 0; i < N; i++) re[i] = frame[i] * BLACKMAN[i];
  fftPow(re, im);
  const out = new Float32Array(N / 2);
  for (let k = 0; k < N / 2; k++) {
    const mag = Math.hypot(re[k], im[k]) / N;
    out[k] = mag > 0 ? 20 * Math.log10(mag) : -Infinity;
  }
  return out;
}

// Seeded RNG so every run is reproducible
function rng(seed) {
  let s = seed >>> 0;
  const u = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s;
    t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const g = () => Math.sqrt(-2 * Math.log(u() || 1e-12)) * Math.cos(2 * Math.PI * u());
  return { u, g };
}

// Room noise: pink-ish (1/f) plus a little white, RMS normalised to `rms`.
function roomNoise(len, rms, R) {
  const x = new Float32Array(len);
  let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
  for (let i = 0; i < len; i++) {              // Paul Kellet pink filter
    const w = R.g();
    b0 = 0.99886 * b0 + w * 0.0555179; b1 = 0.99332 * b1 + w * 0.0750759;
    b2 = 0.96900 * b2 + w * 0.1538520; b3 = 0.86650 * b3 + w * 0.3104856;
    b4 = 0.55000 * b4 + w * 0.5329522; b5 = -0.7616 * b5 - w * 0.0168980;
    x[i] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362 + 0.3 * R.g();
    b6 = w * 0.115926;
  }
  let e = 0; for (const v of x) e += v * v;
  const k = rms / Math.sqrt(e / len);
  for (let i = 0; i < len; i++) x[i] *= k;
  return x;
}

// Free-flying mosquito: f0 wanders (slow vibrato + random walk), amplitude is
// modulated by the changing flight distance, harmonics at -6 / -12 dB.
function mosquito(len, f0c, amp, R, startSample) {
  const x = new Float32Array(len);
  let ph = 0, walk = 0, amWalk = 0;
  const vib = 0.4 + R.u() * 0.6, vibA = 3 + R.u() * 5, ph0 = R.u() * 6.28;
  for (let i = startSample; i < len; i++) {
    const t = i / SR;
    walk += R.g() * 0.02; walk *= 0.9999;          // slow random frequency drift
    amWalk += R.g() * 0.0005; amWalk *= 0.9995;    // flight-distance fluctuation
    const f = f0c + vibA * Math.sin(2 * Math.PI * vib * t + ph0) + walk;
    ph += 2 * Math.PI * f / SR;
    const a = amp * Math.max(0.2, 1 + 0.25 * Math.sin(2 * Math.PI * 0.9 * t) + amWalk);
    x[i] = a * (Math.sin(ph) + 0.5 * Math.sin(2 * ph + 0.3) + 0.25 * Math.sin(3 * ph + 1.1));
  }
  return x;
}

// Machine tone (fan/motor): rock-steady frequency and amplitude, harmonics.
function machine(len, f0, amp, startSample) {
  const x = new Float32Array(len);
  for (let i = startSample; i < len; i++) {
    const ph = 2 * Math.PI * f0 * i / SR;
    x[i] = amp * (Math.sin(ph) + 0.6 * Math.sin(2 * ph) + 0.4 * Math.sin(3 * ph) + 0.3 * Math.sin(4 * ph));
  }
  return x;
}

// Voiced-speech-like source: f0 glides 150-280 Hz in syllables with pauses,
// rich harmonic series (many harmonics fall inside the mosquito band).
function speech(len, amp, R) {
  const x = new Float32Array(len);
  let ph = 0, t0 = 0, on = false, f0 = 200, target = 200;
  for (let i = 0; i < len; i++) {
    if (i >= t0) {
      on = !on || R.u() < 0.2;
      t0 = i + Math.floor((on ? 0.15 + R.u() * 0.25 : 0.05 + R.u() * 0.2) * SR);
      target = 150 + R.u() * 130;
    }
    f0 += (target - f0) * 0.0004;
    ph += 2 * Math.PI * f0 / SR;
    if (!on) continue;
    let s = 0;
    for (let h = 1; h <= 12; h++) s += Math.sin(h * ph) / h;
    x[i] = amp * s;
  }
  return x;
}

// Species whose ranges overlap are acoustically ambiguous (e.g. An. gambiae 375-445
// Hz vs An. stephensi 360-412 Hz); a detection within the same genus counts as a hit.
const speciesGroup = id => id.startsWith('anopheles') ? 'anopheles' : id.startsWith('culex') ? 'culex' : id;

// ── Run the classifier over a signal the way the app does ────────────────────
function run(ctx, signal, speciesId, level) {
  const clf = new ctx.MosquitoClassifier({
    freqToBin: hz => Math.min(Math.round(hz / BIN_HZ), N / 2 - 1), binHz: BIN_HZ,
  });
  clf.getActivityFactor = () => 1.0;
  clf.setSensitivity(level);
  clf._fastCalib = true;
  const calibFrames = Math.round(2.0 * SR / HOP);
  let frames = 0, hits = 0, anyHits = 0, firstHit = -1;
  const perSpecies = {};
  for (let end = N; end <= signal.length; end += HOP) {
    const frame = signal.subarray(end - N, end);
    if (frames === calibFrames) {                 // what startListening() does at 2 s
      clf._fastCalib = false;
      for (const sp of ctx.SPECIES_DB) {
        clf.voteBuffer[sp.id].fill(0); clf.votePtr[sp.id] = 0; clf.voteCount[sp.id] = 0;
      }
    }
    const dets = clf.analyze(analyserDb(frame), Float32Array.from(frame));
    frames++;
    for (const d of dets) perSpecies[d.species.id] = (perSpecies[d.species.id] || 0) + 1;
    if (dets.length) anyHits++;
    if (speciesId && dets.some(d => d.species.id === speciesId || (speciesGroup(d.species.id) === speciesGroup(speciesId)))) {
      hits++; if (firstHit < 0) firstHit = frames;
    }
  }
  return { frames, hits, anyHits, firstHit, perSpecies };
}

// ── Experiments ──────────────────────────────────────────────────────────────
function main() {
  const argv = process.argv.slice(2);
  const quick = argv.includes('--quick');
  const tIdx = argv.indexOf('--trials');
  const TRIALS = tIdx >= 0 ? +argv[tIdx + 1] : (quick ? 3 : 6);
  const level = 3;
  const ctx = loadClassifier();
  const hasFF = /class FarFieldTracker/.test(fs.readFileSync(HTML, 'utf8'));
  console.log(`Classifier: ${path.basename(HTML)} (${hasFF ? 'with far-field tracker' : 'baseline'})  | sensitivity ${level} | ${TRIALS} trials/point`);

  const NOISE_RMS = 0.01;                         // -40 dBFS room noise
  const DUR = 14, SIG_START = 3;                  // 3 s noise-only warm-up, then 11 s of flight
  const len = DUR * SR;
  // SNR is defined as fundamental power / total noise power (broadband), in dB.
  const snrs = quick ? [-10, -20, -30] : [0, -5, -10, -15, -20, -25, -30, -35];
  const species = [['anopheles', 406], ['aedes_aegypti', 617]];

  const curve = {};
  for (const [spId, f0] of species) {
    console.log(`\n${spId} (f0≈${f0} Hz)`);
    console.log('  SNR(dB)  P(detect)  frames-locked  mean-latency(s)');
    curve[spId] = [];
    for (const snr of snrs) {
      let det = 0, fracSum = 0, latSum = 0, latN = 0;
      for (let t = 0; t < TRIALS; t++) {
        const R = rng(1000 * t + Math.round(-snr * 7) + f0);
        const noise = roomNoise(len, NOISE_RMS, R);
        const amp = NOISE_RMS * Math.sqrt(2) * Math.pow(10, snr / 20);   // sine RMS = A/√2
        const tone = mosquito(len, f0 + (R.u() - 0.5) * 16, amp, R, SIG_START * SR);
        const sig = new Float32Array(len);
        for (let i = 0; i < len; i++) sig[i] = noise[i] + tone[i];
        const r = run(ctx, sig, spId, level);
        const sigFrames = Math.round((DUR - SIG_START) * SR / HOP);
        if (r.hits > 0) { det++; if (r.firstHit > 0) { latSum += (r.firstHit * HOP / SR) - SIG_START; latN++; } }
        fracSum += r.hits / sigFrames;
      }
      const pd = det / TRIALS;
      curve[spId].push([snr, pd]);
      console.log(`  ${String(snr).padStart(6)}   ${pd.toFixed(2).padStart(8)}   ${(fracSum / TRIALS * 100).toFixed(0).padStart(10)}%   ${latN ? (latSum / latN).toFixed(1) : '   -'}`);
    }
  }

  // SNR at which P(detect) crosses 0.5 (linear interpolation)
  const threshold = pts => {
    for (let i = 1; i < pts.length; i++) {
      const [s0, p0] = pts[i - 1], [s1, p1] = pts[i];
      if (p0 >= 0.5 && p1 < 0.5) return s0 + (0.5 - p0) * (s1 - s0) / (p1 - p0);
    }
    return pts[pts.length - 1][1] >= 0.5 ? pts[pts.length - 1][0] : null;
  };
  console.log('\nSNR needed for 50% detection:');
  for (const k of Object.keys(curve)) {
    const t = threshold(curve[k]);
    console.log(`  ${k.padEnd(16)} ${t === null ? 'n/a (never reached 50%)' : t.toFixed(1) + ' dB'}`);
  }

  // False alarms (long runs)
  const FA_DUR = quick ? 60 : 180;
  const faLen = FA_DUR * SR;
  const faCases = {
    'noise only':         R => roomNoise(faLen, NOISE_RMS, R),
    'fan tone 420 Hz':    R => { const n = roomNoise(faLen, NOISE_RMS, R), m = machine(faLen, 420, NOISE_RMS * 0.5, 30 * SR);
                                 for (let i = 0; i < faLen; i++) n[i] += m[i]; return n; },
    'voiced speech':      R => { const n = roomNoise(faLen, NOISE_RMS, R), s = speech(faLen, NOISE_RMS * 1.5, R);
                                 for (let i = 0; i < faLen; i++) n[i] += s[i]; return n; },
  };
  console.log(`\nFalse alarms (${FA_DUR} s each, frames with ANY species locked):`);
  for (const [name, gen] of Object.entries(faCases)) {
    const r = run(ctx, gen(rng(4242)), null, level);
    const which = Object.entries(r.perSpecies).map(([k, v]) => `${k}:${v}`).join(' ') || '-';
    console.log(`  ${name.padEnd(18)} ${String(r.anyHits).padStart(5)} / ${r.frames} frames   ${which}`);
  }
}

main();
