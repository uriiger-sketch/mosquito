'use strict';
/*
 * Generate 16-bit PCM mono WAV fixtures for end-to-end tests. Chrome plays the
 * file in a loop as the microphone (--use-file-for-fake-audio-capture), so the
 * REAL app — Web Audio capture, AnalyserNode, rAF loop, classifier, UI — is
 * exercised exactly as on a phone.
 *
 *   node tests/e2e/make-wav.js <out-dir>
 */
const fs = require('fs');
const path = require('path');

const SR = 44100;

function rng(seed) {
  let s = seed >>> 0;
  const u = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s;
    t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const g = () => Math.sqrt(-2 * Math.log(u() || 1e-12)) * Math.cos(2 * Math.PI * u());
  return { u, g };
}

function pinkNoise(len, rms, R) {
  const x = new Float32Array(len);
  let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
  for (let i = 0; i < len; i++) {
    const w = R.g();
    b0 = 0.99886 * b0 + w * 0.0555179; b1 = 0.99332 * b1 + w * 0.0750759;
    b2 = 0.96900 * b2 + w * 0.1538520; b3 = 0.86650 * b3 + w * 0.3104856;
    b4 = 0.55000 * b4 + w * 0.5329522; b5 = -0.7616 * b5 - w * 0.0168980;
    x[i] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362;
    b6 = w * 0.115926;
  }
  let e = 0; for (const v of x) e += v * v;
  const k = rms / Math.sqrt(e / len);
  for (let i = 0; i < len; i++) x[i] *= k;
  return x;
}

// Free-flying mosquito: wandering f0, flight-distance AM, decaying harmonics.
function mosquito(len, f0, amp, R) {
  const x = new Float32Array(len);
  let ph = 0, walk = 0;
  for (let i = 0; i < len; i++) {
    const t = i / SR;
    walk += R.g() * 0.02; walk *= 0.9999;
    const f = f0 + 5 * Math.sin(2 * Math.PI * 0.5 * t) + walk;
    ph += 2 * Math.PI * f / SR;
    const a = amp * (1 + 0.25 * Math.sin(2 * Math.PI * 0.9 * t));
    x[i] = a * (Math.sin(ph) + 0.5 * Math.sin(2 * ph + 0.3) + 0.25 * Math.sin(3 * ph + 1.1));
  }
  return x;
}

// Steady machine (fan/motor) line with harmonics.
function machine(len, f0, amp) {
  const x = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    const ph = 2 * Math.PI * f0 * i / SR;
    x[i] = amp * (Math.sin(ph) + 0.6 * Math.sin(2 * ph) + 0.4 * Math.sin(3 * ph) + 0.3 * Math.sin(4 * ph));
  }
  return x;
}

// Voiced speech: gliding 150-280 Hz f0 in syllables, 12 harmonics.
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

function writeWav(file, x) {
  const n = x.length, buf = Buffer.alloc(44 + 2 * n);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + 2 * n, 4); buf.write('WAVE', 8);
  buf.write('fmt ', 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22); buf.writeUInt32LE(SR, 24); buf.writeUInt32LE(2 * SR, 28);
  buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write('data', 36); buf.writeUInt32LE(2 * n, 40);
  for (let i = 0; i < n; i++) {
    const v = Math.max(-1, Math.min(1, x[i]));
    buf.writeInt16LE(Math.round(v * 32767), 44 + 2 * i);
  }
  fs.writeFileSync(file, buf);
}

const add = (a, b) => { for (let i = 0; i < a.length; i++) a[i] += b[i]; return a; };

// Fixtures: name → { build(len, R), expect, f0?, loc? }. 30 s loops.
// Mosquito individuals are deliberately OFF the catalogue centre frequency
// (real wingbeat f0 varies by individual, size and temperature), so a detector
// that reports the catalogue value instead of measuring fails the f0 check.
// Adversaries run with NO location, i.e. every species enabled — the strictest
// false-alarm setting.
const BANGKOK = { lat: 13.75, lng: 100.5 }, ACCRA = { lat: 5.6, lng: -0.19 };
const FIXTURES = {
  'quiet-room':     { expect: 'none', build: (n, R) => pinkNoise(n, 0.003, R) },
  'aedes-628':      { expect: 'Aedes', f0: 628, loc: BANGKOK,
                      build: (n, R) => add(pinkNoise(n, 0.003, R), mosquito(n, 628, 0.02, R)) },
  'anopheles-418':  { expect: 'Anopheles', f0: 418, loc: ACCRA,
                      build: (n, R) => add(pinkNoise(n, 0.003, R), mosquito(n, 418, 0.02, R)) },
  'culex-464':      { expect: 'Culex', f0: 464, loc: BANGKOK,
                      build: (n, R) => add(pinkNoise(n, 0.003, R), mosquito(n, 464, 0.02, R)) },
  'fan-420':        { expect: 'none', build: (n, R) => add(pinkNoise(n, 0.003, R), machine(n, 420, 0.01)) },
  'speech':         { expect: 'none', build: (n, R) => add(pinkNoise(n, 0.003, R), speech(n, 0.02, R)) },
};

if (require.main === module) {
  const out = path.resolve(process.argv[2] || '.');
  fs.mkdirSync(out, { recursive: true });
  for (const [name, f] of Object.entries(FIXTURES)) {
    writeWav(path.join(out, `${name}.wav`), f.build(30 * SR, rng(name.length * 7919)));
  }
  console.log(`wrote ${Object.keys(FIXTURES).length} fixtures to ${out}`);
}
module.exports = { FIXTURES, writeWav, SR, rng };
