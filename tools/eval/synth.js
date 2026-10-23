'use strict';
/*
 * Streaming synthesis primitives for the evaluation corpus.
 *
 * Every source implements  render(out, off, s, n)  which ADDS samples for the
 * absolute sample range [s, s+n) into out[off … off+n). Sources keep their own
 * state, so a 20-minute scene is rendered hop by hop in constant memory and the
 * result is bit-identical however the hops are sized.
 *
 * Primitives
 *   OU            Ornstein–Uhlenbeck process sampled on a fixed grid (random
 *                 wander of pitch / level / directivity with a correlation time).
 *   Tone          harmonic oscillator: time-varying f0, envelope and per-harmonic
 *                 amplitudes (optionally recomputed per control block, e.g.
 *                 formants). Harmonics via the angle-addition recurrence, so cost
 *                 is ~6 flops per harmonic per sample.
 *   Partials      inharmonic partials with individual exponential decays
 *                 (piano, bells, glass, cricket carrier).
 *   NoiseSrc      white / pink / brown noise through optional biquads, with a
 *                 gain envelope.
 *   EventStream   lazily-scheduled sequence of events (syllables, notes, flight
 *                 bouts, drops…).
 *   Scene         sum of sources → hop-sized Float64Array blocks.
 */

const TWO_PI = 2 * Math.PI;
const CB = 32;                         // control block (samples) for parameter updates

const dbToAmp = (db) => Math.pow(10, db / 20);
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

// ── Ornstein–Uhlenbeck process (stationary: x ~ N(0, σ²)) ─────────────────────
class OU {
  constructor(R, sigma, tau, dt = 0.005) {
    this.R = R; this.sigma = sigma; this.dt = dt;
    this.a = Math.exp(-dt / Math.max(tau, 1e-6));
    this.b = sigma * Math.sqrt(1 - this.a * this.a);
    this.x = sigma * R.g(); this.t = 0;
  }
  at(t) {                              // monotone in t (sources render forward)
    while (this.t + this.dt <= t) { this.x = this.a * this.x + this.b * this.R.g(); this.t += this.dt; }
    return this.x;
  }
}

// ── Biquad (RBJ cookbook) ────────────────────────────────────────────────────
class Biquad {
  constructor(type, sr, f, q = 0.707) {
    const w = TWO_PI * Math.min(f, 0.49 * sr) / sr, cs = Math.cos(w), sn = Math.sin(w), al = sn / (2 * q);
    let b0, b1, b2, a0, a1, a2;
    if (type === 'lp')      { b0 = (1 - cs) / 2; b1 = 1 - cs; b2 = (1 - cs) / 2; }
    else if (type === 'hp') { b0 = (1 + cs) / 2; b1 = -(1 + cs); b2 = (1 + cs) / 2; }
    else                    { b0 = al; b1 = 0; b2 = -al; }                  // 'bp', 0 dB peak
    a0 = 1 + al; a1 = -2 * cs; a2 = 1 - al;
    this.b0 = b0 / a0; this.b1 = b1 / a0; this.b2 = b2 / a0; this.a1 = a1 / a0; this.a2 = a2 / a0;
    this.x1 = this.x2 = this.y1 = this.y2 = 0;
  }
  step(x) {
    const y = this.b0 * x + this.b1 * this.x1 + this.b2 * this.x2 - this.a1 * this.y1 - this.a2 * this.y2;
    this.x2 = this.x1; this.x1 = x; this.y2 = this.y1; this.y1 = y;
    return y;
  }
}

// ── Coloured noise generators (unit-ish variance) ────────────────────────────
// Pink: Paul Kellet's refined filter (±0.05 dB 1/f from 9 Hz to Nyquist at 44.1 kHz).
class Pink {
  constructor(R) { this.R = R; this.b = new Float64Array(7); }
  next() {
    const w = this.R.g(), b = this.b;
    b[0] = 0.99886 * b[0] + w * 0.0555179; b[1] = 0.99332 * b[1] + w * 0.0750759;
    b[2] = 0.96900 * b[2] + w * 0.1538520; b[3] = 0.86650 * b[3] + w * 0.3104856;
    b[4] = 0.55000 * b[4] + w * 0.5329522; b[5] = -0.7616 * b[5] - w * 0.0168980;
    const y = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + w * 0.5362;
    b[6] = w * 0.115926;
    return y * 0.3277;                  // unit RMS (measured: raw Kellet output RMS = 3.05 × 0.11)
  }
}
// Brown (1/f²): leaky integrator of white noise, DC-blocked at ~15 Hz.
class Brown {
  constructor(R, sr) { this.R = R; this.y = 0; this.k = Math.exp(-TWO_PI * 15 / sr); this.g = Math.sqrt(1 - this.k * this.k); }
  next() { this.y = this.k * this.y + this.g * this.R.g(); return this.y; }
}

// ── Harmonic tone ────────────────────────────────────────────────────────────
// o: { start (s), dur (s), f0: t→Hz, env: t→amplitude, amps: [a1, a2, …] (H1 = amps[0]),
//      phases?: [rad], harmFn?: (t, f0, ampsOut) → void, R }
class Tone {
  constructor(sr, o) {
    this.sr = sr;
    this.s0 = Math.round(o.start * sr);
    this.s1 = o.dur === Infinity ? Infinity : this.s0 + Math.round(o.dur * sr);
    this.f0 = o.f0; this.env = o.env; this.harmFn = o.harmFn || null;
    const H = this.H = o.amps.length;
    this.amp = Float64Array.from(o.amps);
    this.phs = new Float64Array(H);
    for (let h = 0; h < H; h++) this.phs[h] = o.phases ? o.phases[h] : (o.R ? o.R.u() * TWO_PI : 0);
    this.c = new Float64Array(H); this.d = new Float64Array(H);
    this._coef();
    this.phase = o.R ? o.R.u() * TWO_PI : 0;
  }
  _coef() { for (let h = 0; h < this.H; h++) { this.c[h] = this.amp[h] * Math.cos(this.phs[h]); this.d[h] = this.amp[h] * Math.sin(this.phs[h]); } }
  render(out, off, s, n) {
    const a = Math.max(s, this.s0), b = Math.min(s + n, this.s1);
    if (a >= b) return;
    const sr = this.sr, c = this.c, d = this.d;
    for (let blk = a; blk < b;) {
      const e = Math.min(b, blk + CB - ((blk - this.s0) % CB));
      const t0 = (blk - this.s0) / sr, t1 = (e - this.s0) / sr;
      const fa = this.f0(t0), fb = this.f0(t1), ea = this.env(t0), eb = this.env(t1);
      if (this.harmFn) { this.harmFn(t0, fa, this.amp); this._coef(); }
      const Hmax = Math.min(this.H, Math.floor(0.45 * sr / Math.max(fa, fb, 1)));
      const len = e - blk, df = (fb - fa) / len, de = (eb - ea) / len;
      let f = fa, en = ea, ph = this.phase, o = off + (blk - s);
      if (ea === 0 && eb === 0) {                      // silent block: advance phase only
        this.phase = (ph + TWO_PI * (fa + fb) * 0.5 * len / sr) % TWO_PI; blk = e; continue;
      }
      for (let i = 0; i < len; i++) {
        ph += TWO_PI * f / sr; f += df;
        const s1 = Math.sin(ph), c1 = Math.cos(ph);
        let sh = s1, ch = c1, acc = c[0] * s1 + d[0] * c1;
        for (let h = 1; h < Hmax; h++) {
          const ns = sh * c1 + ch * s1; ch = ch * c1 - sh * s1; sh = ns;
          acc += c[h] * sh + d[h] * ch;
        }
        out[o + i] += en * acc; en += de;
      }
      this.phase = ph % TWO_PI;
      blk = e;
    }
  }
}

// ── Inharmonic decaying partials (piano, bells, glass, cricket carrier) ──────
// o: { start, dur, freqs: [Hz], amps: [a], decays: [1/s] (amplitude e-folding rate),
//      env?: t→gain (attack / damping), fm?: t→ratio (pitch multiplier), R }
class Partials {
  constructor(sr, o) {
    this.sr = sr; this.s0 = Math.round(o.start * sr); this.s1 = this.s0 + Math.round(o.dur * sr);
    this.f = Float64Array.from(o.freqs); this.a = Float64Array.from(o.amps);
    this.k = Float64Array.from(o.decays || o.freqs.map(() => 0));
    this.ph = Float64Array.from(o.freqs, () => (o.R ? o.R.u() * TWO_PI : 0));
    this.env = o.env || (() => 1); this.fm = o.fm || null;
  }
  render(out, off, s, n) {
    const a = Math.max(s, this.s0), b = Math.min(s + n, this.s1);
    if (a >= b) return;
    const sr = this.sr, P = this.f.length;
    for (let blk = a; blk < b;) {
      const e = Math.min(b, blk + CB - ((blk - this.s0) % CB));
      const t0 = (blk - this.s0) / sr, t1 = (e - this.s0) / sr;
      const ga = this.env(t0), gb = this.env(t1), len = e - blk;
      const m = this.fm ? this.fm(t0) : 1;
      for (let p = 0; p < P; p++) {
        const fp = this.f[p] * m;
        if (fp >= 0.45 * sr) continue;
        const A0 = this.a[p] * Math.exp(-this.k[p] * t0), A1 = this.a[p] * Math.exp(-this.k[p] * t1);
        const w = TWO_PI * fp / sr;
        let ph = this.ph[p], A = A0 * ga; const dA = (A1 * gb - A) / len;
        if (Math.abs(A) + Math.abs(A1 * gb) < 1e-12) { this.ph[p] = (ph + w * len) % TWO_PI; continue; }
        let o = off + (blk - s);
        for (let i = 0; i < len; i++) { ph += w; out[o + i] += A * Math.sin(ph); A += dA; }
        this.ph[p] = ph % TWO_PI;
      }
      blk = e;
    }
  }
}

// ── Noise source with filters and envelope ───────────────────────────────────
// o: { start=0, dur=Infinity, color: 'white'|'pink'|'brown', filters: [Biquad],
//      gain: number | t→gain, R }
class NoiseSrc {
  constructor(sr, o) {
    this.sr = sr; this.s0 = Math.round((o.start || 0) * sr);
    this.s1 = (o.dur === undefined || o.dur === Infinity) ? Infinity : this.s0 + Math.round(o.dur * sr);
    this.R = o.R;
    this.gen = o.color === 'pink' ? new Pink(o.R) : o.color === 'brown' ? new Brown(o.R, sr) : null;
    this.filters = o.filters || [];
    this.gain = typeof o.gain === 'function' ? o.gain : (() => (o.gain === undefined ? 1 : o.gain));
  }
  render(out, off, s, n) {
    const a = Math.max(s, this.s0), b = Math.min(s + n, this.s1);
    if (a >= b) return;
    const sr = this.sr, F = this.filters, nf = F.length, gen = this.gen, R = this.R;
    for (let blk = a; blk < b;) {
      const e = Math.min(b, blk + CB - ((blk - this.s0) % CB));
      const ga = this.gain((blk - this.s0) / sr), gb = this.gain((e - this.s0) / sr);
      const len = e - blk, dg = (gb - ga) / len;
      let g = ga, o = off + (blk - s);
      for (let i = 0; i < len; i++) {
        let x = gen ? gen.next() : R.g();
        for (let j = 0; j < nf; j++) x = F[j].step(x);
        out[o + i] += g * x; g += dg;
      }
      blk = e;
    }
  }
}

// ── Event scheduling ─────────────────────────────────────────────────────────
// scheduler(stream, horizonSec) must add ≥1 event / gap and return the new
// horizon (seconds) up to which the timeline is now planned.
class EventStream {
  constructor(sr, scheduler) { this.sr = sr; this.sched = scheduler; this.events = []; this.horizon = 0; }
  add(ev) { this.events.push(ev); return ev; }
  render(out, off, s, n) {
    const tEnd = (s + n) / this.sr;
    let guard = 0;
    while (this.horizon < tEnd + 0.5 && guard++ < 10000) {
      const h = this.sched(this, this.horizon);
      this.horizon = h > this.horizon ? h : this.horizon + 0.05;
    }
    const keep = [];
    for (const ev of this.events) {
      if (ev.s1 <= s) continue;                        // finished: drop
      keep.push(ev);
      if (ev.s0 < s + n) ev.render(out, off, s, n);
    }
    this.events = keep;
  }
}

// Plain sum of sources. next(n) returns a REUSED Float64Array(n).
class Scene {
  constructor(sr, parts) { this.sr = sr; this.parts = parts.filter(Boolean); this.s = 0; this.buf = null; }
  add(src) { this.parts.push(src); return this; }
  next(n) {
    if (!this.buf || this.buf.length !== n) this.buf = new Float64Array(n);
    const out = this.buf; out.fill(0);
    for (const p of this.parts) p.render(out, 0, this.s, n);
    this.s += n;
    return out;
  }
}

// Wrap a source with a constant gain.
class Gain {
  constructor(src, g) { this.src = src; this.g = g; this.tmp = null; }
  render(out, off, s, n) {
    if (!this.tmp || this.tmp.length < n) this.tmp = new Float64Array(n);
    const t = this.tmp; t.fill(0, 0, n);
    this.src.render(t, 0, s, n);
    for (let i = 0; i < n; i++) out[off + i] += this.g * t[i];
  }
}

// ── Envelope helpers (t in seconds from event start) ─────────────────────────
// Attack-sustain-release with raised-cosine edges.
function asr(dur, att, rel, level = 1) {
  return (t) => {
    if (t < 0 || t > dur) return 0;
    let g = level;
    if (t < att) g *= 0.5 - 0.5 * Math.cos(Math.PI * t / att);
    if (t > dur - rel) g *= 0.5 - 0.5 * Math.cos(Math.PI * Math.max(0, dur - t) / rel);
    return g;
  };
}

// Linear interpolation into a grid array (grid step dt).
function gridFn(arr, dt) {
  const n = arr.length;
  return (t) => {
    const x = t / dt, i = Math.floor(x);
    if (i <= 0) return arr[0];
    if (i >= n - 1) return arr[n - 1];
    const f = x - i;
    return arr[i] * (1 - f) + arr[i + 1] * f;
  };
}

// Formant (2nd-order resonance) magnitude, unity at DC.
function resonance(f, F, B) {
  const F2 = F * F;
  return F2 / Math.sqrt((F2 - f * f) ** 2 + (B * f) ** 2);
}

module.exports = {
  TWO_PI, CB, dbToAmp, clamp, OU, Biquad, Pink, Brown, Tone, Partials, NoiseSrc,
  EventStream, Scene, Gain, asr, gridFn, resonance,
};
