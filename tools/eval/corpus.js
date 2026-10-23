'use strict';
/*
 * Adversary corpus: everything a phone on a bedside table hears that is NOT a
 * flying female mosquito. Each class is a streaming generator of unbounded
 * length whose episodes (who speaks, which note, which fan speed, …) are drawn
 * from seeded RNGs, so a 20-minute run covers many parameter combinations.
 *
 * Levels are given relative to the reference room noise (ROOM_RMS = 0.01,
 * −40 dBFS, the legacy bench's pink+white room noise, which is ALWAYS present
 * underneath). "+20 dB" = source RMS 20 dB above the room-noise RMS. A quiet
 * bedroom is ≈ 25–35 dBA (WHO Night Noise Guidelines 2009), so +20…+30 dB is
 * ordinary conversation / TV level at 1–3 m.
 *
 * Parameter sources (ranges, not fits):
 *  speech     f0 men ≈ 85–155, women ≈ 165–255, children ≈ 250–400 Hz; vowel
 *             formants Hillenbrand et al. 1995 (JASA 97:3099), scaled ×1.17
 *             (women) / ×1.30 (children); syllabic rate ≈ 4–5 Hz (Ding et al.
 *             2017, Neurosci Biobehav Rev 81:181).
 *  singing    vibrato 4.5–7 Hz, extent ±0.2–1 semitone (Sundberg 1987, "The
 *             Science of the Singing Voice"; Prame 1994, JASA 96:1979).
 *  violin     vibrato 5–7 Hz, ±0.15–0.45 semitone (Mellody & Wakefield 2000,
 *             JASA 107:598); body resonances A0 ≈ 280 Hz, B1 ≈ 450–550 Hz,
 *             bridge hill 2–3 kHz.
 *  whistling  near-sinusoidal 500–1500 Hz (here; real range up to ~4 kHz),
 *             H2 ≥ 25 dB down (Shadle 1983, Phys Teach 21:148).
 *  piano      inharmonicity B ≈ 1e-4…5e-4, f_h = h f0 √(1+Bh²) (Fletcher &
 *             Rossing 1998, "The Physics of Musical Instruments").
 *  fans       blade-pass frequency = blades × rpm/60 with harmonics, shaft-rate
 *             sidebands, broadband turbulence AM at BPF (Neise 1992, Noise
 *             Control Eng J 38:113); cross-flow AC blowers ≈ 30–40 blades.
 *  mains      50/60 Hz, magnetostriction at 100/120 Hz + harmonics; induction
 *             motors slip 1–4 % below synchronous speed.
 *  telephony  ringback EU 425 Hz 1 s on / 4 s off, US 440+480 Hz 2 s / 4 s
 *             (ITU-T E.180 Suppl. 2).
 *  insects    see population.js.
 *  crickets   Gryllus carrier 3.5–5.5 kHz, 3–5 pulse chirps; Oecanthus trills
 *             1.8–3.2 kHz (Bennet-Clark 1989, in "Cricket Behavior and
 *             Neurobiology").
 *  traffic    4-stroke firing frequency = cylinders·rpm/120 ≈ 25–120 Hz;
 *             tyre/road noise peak ≈ 700–1500 Hz (Sandberg & Ejsmont 2002).
 *  birds      low tonal calls (doves, cuckoo, owls) ≈ 300–750 Hz; passerine
 *             song 2–8 kHz FM sweeps (Catchpole & Slater 2008).
 */
const { rng, seedFrom } = require('../lib/dsp');
const S = require('./synth');
const P = require('./population');
const { SPECIES_DB } = require('../../js/species.js');

const ROOM_RMS = 0.01;
const lvl = (dbReRoom) => ROOM_RMS * S.dbToAmp(dbReRoom);    // RMS amplitude
const rmsOfAmps = (amps) => Math.sqrt(amps.reduce((s, a) => s + a * a, 0) / 2);
const semis = (n) => Math.pow(2, n / 12);

// ── Reference room noise (legacy bench model: Kellet pink + 0.3·white) ───────
const _roomGain = new Map();
class RoomNoise {
  constructor(sr, R, rms = ROOM_RMS) {
    if (!_roomGain.has(sr)) {                      // measure the unnormalised RMS once
      const r = rng(12345), g = new RoomNoise._Raw(r); let e = 0; const n = sr * 20;
      for (let i = 0; i < n; i++) { const v = g.next(); e += v * v; }
      _roomGain.set(sr, 1 / Math.sqrt(e / n));
    }
    this.raw = new RoomNoise._Raw(R); this.k = rms * _roomGain.get(sr);
  }
  render(out, off, s, n) { for (let i = 0; i < n; i++) out[off + i] += this.k * this.raw.next(); }
}
RoomNoise._Raw = class {
  constructor(R) { this.R = R; this.b = new Float64Array(7); }
  next() {
    const R = this.R, b = this.b, w = R.g();
    b[0] = 0.99886 * b[0] + w * 0.0555179; b[1] = 0.99332 * b[1] + w * 0.0750759;
    b[2] = 0.96900 * b[2] + w * 0.1538520; b[3] = 0.86650 * b[3] + w * 0.3104856;
    b[4] = 0.55000 * b[4] + w * 0.5329522; b[5] = -0.7616 * b[5] - w * 0.0168980;
    const x = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + w * 0.5362 + 0.3 * R.g();
    b[6] = w * 0.115926;
    return x;
  }
};

// ── Composition helpers ──────────────────────────────────────────────────────
// A set of sources active only in [start, start+dur) (absolute seconds).
class Group {
  constructor(sr, start, dur, parts) { this.s0 = Math.round(start * sr); this.s1 = this.s0 + Math.round(dur * sr); this.parts = parts; }
  render(out, off, s, n) {
    const a = Math.max(s, this.s0), b = Math.min(s + n, this.s1);
    if (a >= b) return;
    for (const p of this.parts) p.render(out, off + (a - s), a, b - a);
  }
}
// Run a generator in LOCAL time (its t=0 at `start`) inside [start, start+dur).
class Segment {
  constructor(sr, start, dur, src) { this.s0 = Math.round(start * sr); this.s1 = this.s0 + Math.round(dur * sr); this.src = src; }
  render(out, off, s, n) {
    const a = Math.max(s, this.s0), b = Math.min(s + n, this.s1);
    if (a >= b) return;
    this.src.render(out, off + (a - s), a - this.s0, b - a);
  }
}

// Piecewise level schedule (dB re room), lazily extended. Abrupt or ramped steps.
function levelSchedule(R, levels, segDur, rampProb = 0.5) {
  const pts = [{ t: 0, db: R.pick(levels) }];
  return (t) => {
    while (pts[pts.length - 1].t < t + 1) {
      const last = pts[pts.length - 1];
      const tEnd = last.t + R.range(segDur[0], segDur[1]);
      const ramp = R.u() < rampProb ? R.range(2, 10) : 0.02;
      pts.push({ t: tEnd, db: last.db }, { t: tEnd + ramp, db: R.pick(levels) });
    }
    let i = 0; while (i < pts.length - 1 && pts[i + 1].t <= t) i++;
    const p0 = pts[i], p1 = pts[i + 1];
    const db = p1 ? p0.db + (p1.db - p0.db) * S.clamp((t - p0.t) / Math.max(1e-9, p1.t - p0.t), 0, 1) : p0.db;
    return S.dbToAmp(db);
  };
}

// Monotone-time melodic contour: notes [{t0, dur, f, vib:{rate, ext, delay}|null, lvlDb}]
// f(t) with portamento + vibrato + jitter; env(t) with phrase edges + articulation.
function contour(R, notes, o) {
  const port = o.port ?? 0.06, art = o.art ?? 0.2, att = o.att ?? 0.04, rel = o.rel ?? 0.08;
  const total = notes[notes.length - 1].t0 + notes[notes.length - 1].dur;
  const jit = new S.OU(R, o.jitter ?? 0.004, o.jitterTau ?? 0.08, 0.005);
  let iF = 0, iE = 0;
  const vibPh = notes.map(() => R.u() * S.TWO_PI);
  const f = (t) => {
    while (iF > 0 && t < notes[iF].t0) iF--;
    while (iF < notes.length - 1 && t >= notes[iF + 1].t0) iF++;
    const nt = notes[iF]; let fr = nt.f;
    const dtn = t - nt.t0;
    if (iF > 0 && dtn < port) { const pf = notes[iF - 1].f, x = dtn / port; fr = pf + (nt.f - pf) * (0.5 - 0.5 * Math.cos(Math.PI * x)); }
    if (nt.vib && dtn > nt.vib.delay) {
      const ramp = Math.min(1, (dtn - nt.vib.delay) / 0.25);
      fr *= 1 + nt.vib.ext * ramp * Math.sin(S.TWO_PI * nt.vib.rate * dtn + vibPh[iF]);
    }
    return fr * (1 + jit.at(t));
  };
  const env = (t) => {
    if (t < 0 || t > total) return 0;
    while (iE > 0 && t < notes[iE].t0) iE--;
    while (iE < notes.length - 1 && t >= notes[iE + 1].t0) iE++;
    const nt = notes[iE];
    let g = S.dbToAmp(nt.lvlDb || 0);
    if (t < att) g *= 0.5 - 0.5 * Math.cos(Math.PI * t / att);
    if (t > total - rel) g *= 0.5 - 0.5 * Math.cos(Math.PI * Math.max(0, total - t) / rel);
    if (art > 0) {                                  // articulation dip around note boundaries
      const d0 = t - nt.t0, d1 = nt.t0 + nt.dur - t, w = 0.03;
      if (iE > 0 && d0 < w) g *= 1 - art * (1 - d0 / w);
      if (iE < notes.length - 1 && d1 < w) g *= 1 - art * (1 - d1 / w);
    }
    return g;
  };
  return { f, env, total };
}

const MAJOR = [0, 2, 4, 5, 7, 9, 11], MINOR = [0, 2, 3, 5, 7, 8, 10];
// Random-walk melody on a diatonic scale.
function makeNotes(R, baseHz, nNotes, durRange, vibProb, vibRate, vibExtSemis, lo, hi) {
  const scale = R.u() < 0.6 ? MAJOR : MINOR;
  let deg = R.int(0, 7), t = 0;
  const notes = [];
  for (let i = 0; i < nNotes; i++) {
    deg += R.pick([-3, -2, -1, -1, 0, 1, 1, 2, 3, 4, -4]);
    deg = S.clamp(deg, -7, 14);
    const oct = Math.floor(deg / 7), st = scale[((deg % 7) + 7) % 7] + 12 * oct;
    let f = baseHz * semis(st);
    while (f > hi) f /= 2; while (f < lo) f *= 2;
    const dur = R.logRange(durRange[0], durRange[1]);
    const vib = (dur > 0.35 && R.u() < vibProb)
      ? { rate: R.range(vibRate[0], vibRate[1]), ext: semis(R.range(vibExtSemis[0], vibExtSemis[1])) - 1, delay: R.range(0.1, 0.35) }
      : null;
    notes.push({ t0: t, dur, f, vib, lvlDb: R.range(-3, 3) });
    t += dur;
  }
  return notes;
}

// ── Voice (speech / singing / humming) spectra ───────────────────────────────
const VOWELS = [ // Hillenbrand et al. 1995, adult male means (F1, F2, F3 Hz)
  [342, 2322, 3000], [427, 2034, 2684], [580, 1799, 2605], [588, 1952, 2601], [768, 1333, 2522],
  [652, 997, 2538], [469, 1122, 2434], [378, 997, 2343], [623, 1200, 2550], [474, 1379, 1710],
];
const FORMANT_SCALE = { male: 1.0, female: 1.17, child: 1.30 };
function formantFn(vowel, scale, extra) {
  const F = vowel.map(x => x * scale), B = [80, 100, 150];
  return (fr) => {
    let g = 1;
    for (let i = 0; i < 3; i++) g *= S.resonance(fr, F[i], B[i]);
    if (extra) g *= extra(fr);
    return g;
  };
}
// Harmonic amplitudes of a voiced source through a vocal tract: glottal −12 dB/oct
// + radiation +6 dB/oct → a_h ∝ 1/h, times the formant gain at h·f0.
function voiceHarm(gainAt, H, tilt = 1) {
  return (t, f0, amps) => { for (let h = 0; h < H; h++) amps[h] = gainAt((h + 1) * f0) / Math.pow(h + 1, tilt); };
}

// ── Speech ───────────────────────────────────────────────────────────────────
const TALKER = {
  male:   { f0: R => S.clamp(115 + 15 * R.g(), 85, 155),  scale: 'male' },
  female: { f0: R => S.clamp(210 + 20 * R.g(), 165, 255), scale: 'female' },
  child:  { f0: R => S.clamp(300 + 35 * R.g(), 240, 400), scale: 'child' },
};
function speechStream(sr, R, talkers, levelDb) {
  const tk = talkers.map(k => ({ ...TALKER[k], kind: k, base: TALKER[k].f0(R) }));
  return new S.EventStream(sr, (st, h) => {
    const who = R.pick(tk), spurt = R.range(1.5, 8), L = lvl(levelDb(R));
    let t = h, end = h + spurt;
    const decl0 = R.range(1.0, 1.12), decl1 = R.range(0.85, 0.97);
    const pitchVar = who.kind === 'child' ? 0.3 : 0.2;
    while (t < end) {
      const dur = R.range(0.1, 0.28);
      const x = (t - h) / spurt, decl = decl0 + (decl1 - decl0) * x;
      const fS = who.base * decl * (1 + pitchVar * (R.u() - 0.5)), fP = fS * (1 + R.range(-0.05, pitchVar)), fE = fS * (1 + R.range(-pitchVar, 0.05));
      if (R.u() < 0.25) {                                  // unvoiced consonant before the vowel
        const cd = R.range(0.04, 0.09);
        st.add(new S.NoiseSrc(sr, { start: t, dur: cd, R, filters: [new S.Biquad('bp', sr, R.range(2500, 6000), 1.5)],
          gain: S.asr(cd, 0.01, 0.02, L * R.range(0.3, 1.0)) }));
        t += cd;
      }
      const H = Math.max(4, Math.floor(4000 / fS));
      const gainAt = formantFn(R.pick(VOWELS), FORMANT_SCALE[who.scale]);
      const harm = voiceHarm(gainAt, H);
      const tmp = new Float64Array(H); harm(0, fS, tmp);
      const norm = L * S.dbToAmp(R.range(-4, 4)) / rmsOfAmps(tmp);
      const jit = new S.OU(R, 0.006, 0.01, 0.002);
      st.add(new S.Tone(sr, { start: t, dur, R, amps: tmp,
        f0: (u) => { const x = u / dur; const base = x < 0.4 ? fS + (fP - fS) * (x / 0.4) : fP + (fE - fP) * ((x - 0.4) / 0.6); return base * (1 + jit.at(u)); },
        env: S.asr(dur, 0.025, 0.04, norm), harmFn: harm }));
      t += dur + (R.u() < 0.3 ? 0 : R.range(0.02, 0.12));
    }
    return end + R.range(0.5, 5);                          // turn-taking pause
  });
}

// ── Melodic voices: humming / singing / whistling ────────────────────────────
function melodicStream(sr, R, o) {
  return new S.EventStream(sr, (st, h) => {
    const base = R.range(o.base[0], o.base[1]);
    const notes = makeNotes(R, base, R.int(o.nNotes[0], o.nNotes[1]), o.dur, o.vibProb, o.vibRate, o.vibExt, o.lo, o.hi);
    const c = contour(R, notes, { port: R.range(0.03, 0.12), art: o.art ?? 0.25, jitter: o.jitter, jitterTau: o.jitterTau });
    const L = lvl(R.range(o.level[0], o.level[1]));
    const amps = o.amps(R), H = amps.length;
    let harmFn = null;
    if (o.formants) {
      // Trained singers add a "singer's formant" cluster near 2.5–3 kHz (Sundberg 1987).
      const gainAt = formantFn(R.pick(VOWELS), FORMANT_SCALE[R.pick(['male', 'female', 'child'])],
        o.singerFormant ? (fr) => 1 + 2 * Math.exp(-(((fr - 2900) / 500) ** 2)) : null);
      harmFn = voiceHarm(gainAt, H);
      harmFn(0, notes[0].f, amps);
    }
    const norm = L / rmsOfAmps(amps);
    st.add(new S.Tone(sr, { start: h, dur: c.total, R, amps, f0: c.f, env: (t) => norm * c.env(t), harmFn }));
    if (o.breath) {                                         // breath noise around the tone (whistle / flute)
      const fc = notes.reduce((s, n) => s + n.f, 0) / notes.length;
      st.add(new S.NoiseSrc(sr, { start: h, dur: c.total, R, filters: [new S.Biquad('bp', sr, fc * o.breath.fMul, o.breath.q)],
        gain: (t) => L * S.dbToAmp(o.breath.db) * c.env(t) * 3 }));
    }
    return h + c.total + R.range(o.pause[0], o.pause[1]);
  });
}

const HUMMING = { base: [180, 520], nNotes: [4, 10], dur: [0.25, 2.5], vibProb: 0.5, vibRate: [4.5, 6.5], vibExt: [0.2, 0.8],
  lo: 150, hi: 700, level: [0, 25], pause: [0.3, 3], jitter: 0.004, jitterTau: 0.1,
  amps: R => { const k = R.range(1.6, 2.4); return Array.from({ length: 10 }, (_, h) => Math.pow(h + 1, -k) * S.dbToAmp(R.range(-2, 2)) * (h === 0 ? 1.5 : 1)); } };
const SINGING = { base: [150, 700], nNotes: [4, 12], dur: [0.2, 2.5], vibProb: 0.7, vibRate: [5, 7], vibExt: [0.3, 1.0],
  lo: 130, hi: 1000, level: [5, 30], pause: [0.3, 3], formants: true, singerFormant: true,
  amps: R => new Array(20).fill(0) };
const WHISTLING = { base: [500, 1500], nNotes: [3, 12], dur: [0.12, 1.2], vibProb: 0.4, vibRate: [4, 7], vibExt: [0.17, 0.5],
  lo: 500, hi: 1500, level: [5, 30], pause: [0.3, 2], jitter: 0.003, jitterTau: 0.03, art: 0.4,
  amps: R => { const h2 = R.range(-40, -25); return [1, S.dbToAmp(h2), S.dbToAmp(h2 + R.range(-15, -5))]; },
  breath: { fMul: 1.0, q: 4, db: -25 } };

// ── Instruments ──────────────────────────────────────────────────────────────
function violinStream(sr, R) {
  return new S.EventStream(sr, (st, h) => {
    const notes = makeNotes(R, R.range(196, 600), R.int(4, 12), [0.2, 2.0], 0.8, [5, 7], [0.15, 0.45], 196, 1000);
    const c = contour(R, notes, { port: R.range(0.02, 0.08), art: 0.3, jitter: 0.002 });
    const A0 = R.range(270, 300), B1 = R.range(450, 550), H = 24;
    const body = (fr) => 0.6 + 1.6 * Math.exp(-(((fr - A0) / 40) ** 2)) + 1.4 * Math.exp(-(((fr - B1) / 80) ** 2)) + 1.2 * Math.exp(-(((fr - 2500) / 600) ** 2));
    const rnd = Array.from({ length: H }, () => S.dbToAmp(R.range(-4, 4)));
    const harm = (t, f0, amps) => { for (let k = 0; k < H; k++) amps[k] = rnd[k] * body((k + 1) * f0) / (k + 1); };
    const amps = new Float64Array(H); harm(0, notes[0].f, amps);
    const L = lvl(R.range(5, 30)), norm = L / rmsOfAmps(Array.from(amps));
    st.add(new S.Tone(sr, { start: h, dur: c.total, R, amps, f0: c.f, env: (t) => norm * c.env(t), harmFn: harm }));
    if (R.u() < 0.15) {                                     // double stop a fifth above
      st.add(new S.Tone(sr, { start: h, dur: c.total, R, amps: Float64Array.from(amps), f0: (t) => c.f(t) * 1.4983, env: (t) => 0.7 * norm * c.env(t), harmFn: harm }));
    }
    st.add(new S.NoiseSrc(sr, { start: h, dur: c.total, R, filters: [new S.Biquad('hp', sr, 2000)], gain: (t) => L * 0.03 * c.env(t) }));
    return h + c.total + R.range(0.2, 2);
  });
}

function fluteStream(sr, R) {
  return melodicStream(sr, R, { base: [262, 900], nNotes: [4, 12], dur: [0.2, 2.0], vibProb: 0.8, vibRate: [4, 6], vibExt: [0.17, 0.35],
    lo: 262, hi: 1500, level: [5, 30], pause: [0.3, 2], jitter: 0.002, art: 0.35,
    amps: R => [1, S.dbToAmp(R.range(-15, -6)), S.dbToAmp(R.range(-25, -12)), S.dbToAmp(R.range(-35, -20))],
    breath: { fMul: 3, q: 0.7, db: -28 } });
}

function pianoNote(sr, R, t0, f0, L, hold) {
  const B = R.logRange(1e-4, 5e-4), Hn = 12;
  const T60 = S.clamp(9 - 6 * Math.log2(f0 / 130) / 3, 1.5, 9);
  const freqs = [], amps = [], decays = [];
  for (let h = 1; h <= Hn; h++) {
    freqs.push(h * f0 * Math.sqrt(1 + B * h * h));
    amps.push(L * Math.pow(h, -1.3) * S.dbToAmp(R.range(-3, 3)));
    decays.push(6.91 / T60 * (1 + 0.5 * (h - 1)));
  }
  const dur = hold + 0.4;
  return [
    new S.Partials(sr, { start: t0, dur, freqs, amps, decays, R,
      env: (t) => (t < 0.003 ? t / 0.003 : 1) * (t > hold ? Math.exp(-15 * (t - hold)) : 1) }),
    new S.NoiseSrc(sr, { start: t0, dur: 0.012, R, filters: [new S.Biquad('lp', sr, 4000)], gain: S.asr(0.012, 0.001, 0.01, L * 0.5) }),
  ];
}
function pianoStream(sr, R) {
  return new S.EventStream(sr, (st, h) => {
    const L = lvl(R.range(5, 30)), n = R.int(1, 4), root = 130.8 * semis(R.int(0, 30));
    const hold = R.range(0.2, 2.0);
    const chord = [0, R.pick([3, 4]), 7, 12].slice(0, n);
    for (const iv of chord) for (const p of pianoNote(sr, R, h, root * semis(iv), L / Math.sqrt(n), hold)) st.add(p);
    return h + R.range(0.2, 1.0) + (R.u() < 0.1 ? R.range(1, 4) : 0);
  });
}

function popStream(sr, R) {
  const bpm = R.range(90, 130), beat = 60 / bpm, L = lvl(R.range(5, 25));
  let bar = 0;
  return new S.EventStream(sr, (st, h) => {
    const chordRoot = 196 * semis(R.pick([0, 2, 5, 7, 9]));
    for (let b = 0; b < 4; b++) {
      const t = h + b * beat;
      if (b % 2 === 0) st.add(new S.Partials(sr, { start: t, dur: 0.3, freqs: [55], amps: [L * 1.5], decays: [12], R, fm: (u) => 1 + 1.5 * Math.exp(-u * 40) }));
      else st.add(new S.NoiseSrc(sr, { start: t, dur: 0.18, R, filters: [new S.Biquad('bp', sr, 2500, 0.8)], gain: (u) => L * 1.2 * Math.exp(-u * 20) }));
      for (let e = 0; e < 2; e++) st.add(new S.NoiseSrc(sr, { start: t + e * beat / 2, dur: 0.05, R, filters: [new S.Biquad('hp', sr, 7000)], gain: (u) => L * 0.5 * Math.exp(-u * 60) }));
      const bf = chordRoot / 4;
      st.add(new S.Tone(sr, { start: t, dur: beat * 0.9, R, amps: Array.from({ length: 20 }, (_, k) => 1 / (k + 1)), f0: () => bf, env: S.asr(beat * 0.9, 0.01, 0.05, L * 0.6) }));
    }
    if (bar % 2 === 0) {                                    // sustained pad chord: steady, no vibrato
      const pad = [0, 4, 7].map(iv => chordRoot * semis(iv) * (1 + 0.003 * R.g()));
      for (const pf of pad) st.add(new S.Tone(sr, { start: h, dur: 8 * beat, R, amps: Array.from({ length: 10 }, (_, k) => 1 / (k + 1)), f0: () => pf, env: S.asr(8 * beat, 0.3, 0.3, L * 0.25) }));
    }
    bar++;
    return h + 4 * beat;
  });
}

// ── Machines ─────────────────────────────────────────────────────────────────
// Speed process: OU drift + occasional set-point changes with glides, × spin envelope.
function speedProcess(R, o) {
  const ou = new S.OU(R, o.sigma, o.tau, 0.05);
  let sp = 1, target = 1, tNext = R.range(o.change[0], o.change[1]), glide = 1, tStart = 0, from = 1;
  return (t) => {
    if (t >= tNext) { from = sp; target = S.clamp(target * R.range(o.step[0], o.step[1]), o.min, o.max); tStart = t; glide = R.range(o.glide[0], o.glide[1]); tNext = t + R.range(o.change[0], o.change[1]); }
    const x = S.clamp((t - tStart) / glide, 0, 1);
    sp = from + (target - from) * (0.5 - 0.5 * Math.cos(Math.PI * x));
    return sp * (1 + ou.at(t));
  };
}
// On/off machine with spin-up/down; build(start, dur, spin) returns the parts for one run.
function cyclingMachine(sr, R, onRange, offRange, build, startOff = [5, 40]) {
  return new S.EventStream(sr, (st, h) => {
    if (h === 0) return R.range(startOff[0], startOff[1]);
    const dur = R.range(onRange[0], onRange[1]), up = R.range(2, 6), dn = R.range(2, 5);
    const spin = (t) => t < up ? 0.5 - 0.5 * Math.cos(Math.PI * t / up) : t > dur - dn ? 0.5 - 0.5 * Math.cos(Math.PI * Math.max(0, dur - t) / dn) : 1;
    st.add(new Group(sr, h, dur, build(h, dur, spin)));
    return h + dur + R.range(offRange[0], offRange[1]);
  });
}

function fanDesk(sr, R) {
  const mains = R.pick([50, 60]);
  return cyclingMachine(sr, R, [120, 480], [20, 90], (h, dur, spin) => {
    const fs = R.range(12, 25), B = R.int(3, 5), bpf = fs * B;
    const speed = speedProcess(R, { sigma: R.range(0.003, 0.012), tau: R.range(10, 40), change: [60, 240], step: [0.85, 1.15], glide: [2, 4], min: 0.7, max: 1.3 });
    const M = Math.min(40, Math.floor(3000 / bpf)), slope = R.range(1.5, 4);
    const amps = Array.from({ length: M }, (_, m) => S.dbToAmp(-m * slope + R.range(-4, 4)));
    const Lt = lvl(R.range(0, 15)), Lb = lvl(R.range(5, 20)), tn = Lt / rmsOfAmps(amps);
    const f = (t) => bpf * speed(h + t) * Math.max(0.02, spin(t));
    const hum = Array.from({ length: 10 }, (_, k) => S.dbToAmp(-6 - 3 * k + R.range(-3, 3)));
    const depth = R.range(0.1, 0.3);
    return [
      new S.Tone(sr, { start: h, dur, R, amps, f0: f, env: (t) => tn * spin(t) }),
      new S.Tone(sr, { start: h, dur, R, amps: [1, 0.3, 0.2, 0.1, 0.05, 0.03], f0: (t) => fs * speed(h + t) * Math.max(0.02, spin(t)), env: (t) => Lt * 0.1 * spin(t) }),
      new S.Tone(sr, { start: h, dur, R, amps: hum, f0: () => 2 * mains, env: (t) => Lt * 0.3 * (spin(t) > 0.05 ? 1 : 0) }),
      new S.NoiseSrc(sr, { start: h, dur, color: 'pink', R, filters: [new S.Biquad('bp', sr, R.range(300, 900), 0.5)],
        gain: (t) => Lb * spin(t) * (1 + depth * Math.sin(S.TWO_PI * bpf * t)) }),
    ];
  });
}

function fanCeiling(sr, R) {
  const mains = R.pick([50, 60]);
  return cyclingMachine(sr, R, [180, 600], [20, 120], (h, dur, spin) => {
    const bpf = R.range(1.5, 4) * R.int(3, 5), depth = R.range(0.2, 0.5), Lb = lvl(R.range(5, 15));
    const hum = Array.from({ length: 8 }, (_, k) => S.dbToAmp(-6 - 4 * k + R.range(-3, 3)));
    return [
      new S.NoiseSrc(sr, { start: h, dur, color: 'pink', R, filters: [new S.Biquad('bp', sr, R.range(200, 600), 0.4)],
        gain: (t) => Lb * spin(t) * (1 + depth * Math.sin(S.TWO_PI * bpf * t)) }),
      new S.Tone(sr, { start: h, dur, R, amps: hum, f0: () => 2 * mains, env: (t) => Lb * 0.15 * spin(t) }),
    ];
  });
}

// Small fan (laptop / PC / USB): BPF 150–700 Hz lands IN the mosquito band with
// nothing below it but a weak shaft line — the hardest machine adversary.
function fanSmall(sr, R) {
  return cyclingMachine(sr, R, [120, 420], [10, 60], (h, dur, spin) => {
    let fs, B;
    do { fs = R.range(25, 75); B = R.pick([5, 7, 9, 11]); } while (fs * B < 150 || fs * B > 700);
    const H = 6 * B + 2, amps = new Array(H).fill(0);
    const decay = R.range(4, 10);
    for (let m = 1; m <= 6; m++) {
      const a = S.dbToAmp(-(m - 1) * decay + R.range(-3, 3));
      amps[m * B - 1] = a;
      amps[m * B - 2] = a * S.dbToAmp(R.range(-25, -15));
      if (m * B < H) amps[m * B] = a * S.dbToAmp(R.range(-25, -15));
    }
    amps[0] = S.dbToAmp(R.range(-30, -20));                 // weak shaft-rate line
    const speed = speedProcess(R, { sigma: R.range(0.005, 0.02), tau: R.range(5, 30), change: [20, 120], step: [0.8, 1.25], glide: [1, 6], min: 0.6, max: 1.6 });
    const Lt = lvl(R.range(0, 20)), tn = Lt / rmsOfAmps(amps);
    return [
      new S.Tone(sr, { start: h, dur, R, amps, f0: (t) => fs * speed(h + t) * Math.max(0.02, spin(t)), env: (t) => tn * spin(t) }),
      new S.NoiseSrc(sr, { start: h, dur, R, filters: [new S.Biquad('hp', sr, 500)], gain: (t) => lvl(R.range(-5, 5)) * spin(t) }),
    ];
  });
}

function compressor(sr, R, mains, h, dur, spin, L) {
  const frot = mains * (1 - R.range(0.01, 0.04));
  const amps = Array.from({ length: 20 }, (_, k) => S.dbToAmp(-k * R.range(0.5, 2) + R.range(-8, 4)));
  const mag = Array.from({ length: 8 }, (_, k) => S.dbToAmp(-3 * k + R.range(-6, 3)));
  return [
    new S.Tone(sr, { start: h, dur, R, amps, f0: (t) => frot * Math.max(0.05, Math.min(1, t / 0.5)), env: (t) => L / rmsOfAmps(amps) * spin(t) }),
    new S.Tone(sr, { start: h, dur, R, amps: mag, f0: () => 2 * mains, env: (t) => L * 0.5 / rmsOfAmps(mag) * spin(t) }),
    new S.NoiseSrc(sr, { start: h, dur, color: 'brown', R, filters: [new S.Biquad('lp', sr, 300)], gain: (t) => L * 0.7 * spin(t) }),
    new S.NoiseSrc(sr, { start: h, dur: 0.25, R, filters: [new S.Biquad('lp', sr, 400)], gain: S.asr(0.25, 0.005, 0.2, L * 3) }),
  ];
}
function humFridge(sr, R) {
  const mains = R.pick([50, 60]);
  return cyclingMachine(sr, R, [180, 600], [60, 300], (h, dur, spin) => compressor(sr, R, mains, h, dur, spin, lvl(R.range(0, 15))));
}
// Split air-conditioner: cross-flow blower (≈30–40 blades → BPF 300–700 Hz,
// weak), airflow broadband, inverter compressor with slowly ramping speed.
function humAC(sr, R) {
  const mains = R.pick([50, 60]);
  return cyclingMachine(sr, R, [240, 900], [30, 180], (h, dur, spin) => {
    const Lb = lvl(R.range(5, 20));
    const bpf = R.range(300, 700), bl = [1, S.dbToAmp(R.range(-12, -4)), S.dbToAmp(R.range(-20, -8))];
    const sp = speedProcess(R, { sigma: 0.004, tau: 20, change: [30, 120], step: [0.9, 1.1], glide: [3, 10], min: 0.8, max: 1.2 });
    const inv = speedProcess(R, { sigma: 0.01, tau: 30, change: [20, 90], step: [0.7, 1.4], glide: [10, 60], min: 0.3, max: 1.0 });
    const fc = R.range(60, 110), camps = Array.from({ length: 16 }, (_, k) => S.dbToAmp(-2 * k + R.range(-6, 3)));
    return [
      new S.NoiseSrc(sr, { start: h, dur, color: 'pink', R, filters: [new S.Biquad('bp', sr, R.range(400, 1200), 0.35)], gain: (t) => Lb * spin(t) }),
      new S.Tone(sr, { start: h, dur, R, amps: bl, f0: (t) => bpf * sp(h + t) * Math.max(0.05, spin(t)), env: (t) => Lb * S.dbToAmp(R.range(-20, -10)) * spin(t) }),
      new S.Tone(sr, { start: h, dur, R, amps: camps, f0: (t) => fc * inv(h + t) * Math.max(0.05, spin(t)), env: (t) => Lb * 0.4 / rmsOfAmps(camps) * spin(t) }),
      new S.Tone(sr, { start: h, dur, R, amps: [1, 0.5, 0.3, 0.2], f0: () => 2 * mains, env: (t) => Lb * 0.1 * spin(t) }),
    ];
  });
}

// Electrical: transformer / LED-driver buzz (100/120 Hz combs) and coil whine
// (steady tones at arbitrary frequencies, jumping with load).
function whineElectrical(sr, R) {
  const mains = R.pick([50, 60]);
  return new S.EventStream(sr, (st, h) => {
    const dur = R.range(30, 240), k = R.u();
    if (k < 0.35) {
      const n = R.int(6, 20), amps = Array.from({ length: n }, (_, j) => S.dbToAmp(-j * R.range(1, 4) + R.range(-6, 3)));
      st.add(new S.Tone(sr, { start: h, dur, R, amps, f0: () => 2 * mains * (1 + 0.0002 * R.g()), env: S.asr(dur, 0.05, 0.05, lvl(R.range(-5, 12)) / rmsOfAmps(amps)) }));
    } else {
      let t = h; const L = lvl(R.range(-5, 15));
      const drift = new S.OU(R, R.range(0.0005, 0.002), 30, 0.05);
      while (t < h + dur) {
        const seg = Math.min(h + dur - t, R.range(10, 60)), f = R.logRange(250, 4000), t0 = t;
        const amps = [1, S.dbToAmp(R.range(-35, -20)), S.dbToAmp(R.range(-45, -30))];
        st.add(new S.Tone(sr, { start: t, dur: seg, R, amps, f0: (u) => f * (1 + drift.at(t0 + u)), env: S.asr(seg, 0.02, 0.02, L) }));
        t += seg;
      }
    }
    return h + dur + R.range(5, 60);
  });
}

// ── Flying insects (bouts of flight) ─────────────────────────────────────────
function insectStream(sr, R, drawF0, kind, boutRange, gapRange, levelRange) {
  return new S.EventStream(sr, (st, h) => {
    if (h === 0) return R.range(2, 20);
    const dur = R.range(boutRange[0], boutRange[1]), f0 = drawF0(R);
    const p1 = (ROOM_RMS ** 2) * Math.pow(10, R.range(levelRange[0], levelRange[1]) / 10);
    st.add(P.flightTone(sr, { f0, start: h, dur, p1, flight: P.drawFlight(R, kind), R, fadeIn: R.range(0.3, 1.5), fadeOut: R.range(0.3, 1.5) }));
    return h + dur + R.range(gapRange[0], gapRange[1]);
  });
}

// ── Crickets, birds ──────────────────────────────────────────────────────────
function crickets(sr, R) {
  const parts = [];
  const n = R.int(1, 4);
  for (let i = 0; i < n; i++) {
    const L = lvl(R.range(-5, 20));
    if (R.u() < 0.6) {                                      // field cricket chirps
      const fc = R.range(3500, 5500);
      parts.push(new S.EventStream(sr, (st, h) => {
        const np = R.int(3, 5), pw = R.range(0.012, 0.025), gap = R.range(0.015, 0.025);
        for (let p = 0; p < np; p++) st.add(new S.Partials(sr, { start: h + p * (pw + gap), dur: pw, freqs: [fc, 2 * fc], amps: [L * 1.4, L * 0.25], R,
          env: S.asr(pw, 0.003, 0.005) }));
        return h + R.range(0.3, 1.0) + (R.u() < 0.05 ? R.range(2, 20) : 0);
      }));
    } else {                                                // tree cricket trill
      const fc = R.range(1800, 3200), pr = R.range(30, 60);
      parts.push(new S.EventStream(sr, (st, h) => {
        const dur = R.range(2, 20), edge = S.asr(dur, 0.05, 0.05);
        st.add(new S.Tone(sr, { start: h, dur, R, amps: [1, 0.15], f0: () => fc, env: (t) => { const g = 0.5 + 0.5 * Math.sin(S.TWO_PI * pr * t); return L * 1.4 * g * g * edge(t); } }));
        return h + dur + R.range(1, 10);
      }));
    }
  }
  return { render: (o, off, s, n) => { for (const p of parts) p.render(o, off, s, n); } };
}

function birdsSong(sr, R) {
  const birds = Array.from({ length: R.int(1, 3) }, () => ({ L: lvl(R.range(-5, 20)), lo: R.range(2000, 4000), hi: R.range(4500, 8000) }));
  const parts = birds.map(b => new S.EventStream(sr, (st, h) => {
    let t = h; const end = h + R.range(1, 3);
    while (t < end) {
      const d = R.range(0.03, 0.2), f1 = R.range(b.lo, b.hi), f2 = R.range(b.lo, b.hi);
      if (R.u() < 0.3) {                                    // trill: repeated short elements
        const n = R.int(4, 15), per = 1 / R.range(10, 30);
        for (let k = 0; k < n; k++) st.add(new S.Tone(sr, { start: t + k * per, dur: per * 0.6, R, amps: [1, 0.1], f0: (u) => f1 + (f2 - f1) * u / (per * 0.6), env: S.asr(per * 0.6, 0.004, 0.006, b.L * 1.4) }));
        t += n * per;
      } else {
        st.add(new S.Tone(sr, { start: t, dur: d, R, amps: [1, 0.1], f0: (u) => f1 + (f2 - f1) * u / d, env: S.asr(d, 0.008, 0.01, b.L * 1.4) }));
        t += d;
      }
      t += R.range(0.01, 0.12);
    }
    return end + R.range(2, 10);
  }));
  return { render: (o, off, s, n) => { for (const p of parts) p.render(o, off, s, n); } };
}

// Low tonal bird calls in the mosquito band: doves, cuckoo, owls.
function birdsLow(sr, R) {
  return new S.EventStream(sr, (st, h) => {
    const kind = R.pick(['dove', 'cuckoo', 'owl']), L = lvl(R.range(0, 25));
    const h2 = S.dbToAmp(R.range(-30, -15));
    let t = h;
    if (kind === 'dove') {
      const f = R.range(350, 650), n = R.int(3, 5), reps = R.int(2, 6);
      for (let r = 0; r < reps; r++) {
        for (let k = 0; k < n; k++) {
          const d = R.range(0.25, 0.8), fk = f * R.range(0.92, 1.08), arch = R.range(0.03, 0.08);
          st.add(new S.Tone(sr, { start: t, dur: d, R, amps: [1, h2], f0: (u) => fk * (1 + arch * Math.sin(Math.PI * u / d)), env: S.asr(d, 0.05, 0.1, L * 1.4) }));
          t += d + R.range(0.1, 0.3);
        }
        t += R.range(1, 4);
      }
    } else if (kind === 'cuckoo') {
      const f1 = R.range(600, 720), f2 = f1 * R.range(0.78, 0.86), reps = R.int(8, 20), per = R.range(0.8, 1.2);
      for (let r = 0; r < reps; r++) {
        const d1 = R.range(0.18, 0.3), d2 = R.range(0.2, 0.35);
        st.add(new S.Tone(sr, { start: t, dur: d1, R, amps: [1, h2], f0: () => f1, env: S.asr(d1, 0.02, 0.05, L * 1.4) }));
        st.add(new S.Tone(sr, { start: t + d1 + 0.12, dur: d2, R, amps: [1, h2], f0: (u) => f2 * (1 - 0.03 * u / d2), env: S.asr(d2, 0.02, 0.08, L * 1.4) }));
        t += per;
      }
    } else {
      const f = R.range(300, 750), reps = R.int(2, 6);
      for (let r = 0; r < reps; r++) {
        const d = R.range(0.3, 1.2), trem = R.u() < 0.4 ? R.range(10, 20) : 0, edge = S.asr(d, 0.04, 0.15, L * 1.4);
        st.add(new S.Tone(sr, { start: t, dur: d, R, amps: [1, h2, h2 * 0.3], f0: (u) => f * (1 - 0.06 * u / d), env: (u) => edge(u) * (trem ? 0.7 + 0.3 * Math.sin(S.TWO_PI * trem * u) : 1) }));
        t += d + R.range(0.3, 2);
      }
    }
    return t + R.range(10, 60);
  });
}

// ── Traffic, rain, impulsive ─────────────────────────────────────────────────
function traffic(sr, R) {
  const rumbleLvl = levelSchedule(R, [-5, 0, 5, 10], [30, 120]);
  const rumble = new S.NoiseSrc(sr, { color: 'brown', R, filters: [new S.Biquad('lp', sr, 200)], gain: (t) => lvl(0) * rumbleLvl(t) });
  const cars = new S.EventStream(sr, (st, h) => {
    const w = R.range(2, 6), dur = 4 * w, tc = h + dur / 2, moto = R.u() < 0.1;
    const L = lvl(R.range(0, 20) + (moto ? 5 : 0));
    const prof = (t) => Math.exp(-0.5 * ((t - dur / 2) / (w / 2)) ** 2);
    const fe = moto ? R.range(40, 150) : R.range(25, 120), v = R.range(8, 20), d0 = R.range(5, 30);
    const fD = (t) => { const x = v * (t - dur / 2), r = Math.hypot(x, d0); return fe / (1 + v * (x / r) / 343); };
    const amps = Array.from({ length: moto ? 25 : 15 }, (_, k) => S.dbToAmp(-k * R.range(1, 3) + R.range(-5, 3)));
    st.add(new S.Tone(sr, { start: h, dur, R, amps, f0: fD, env: (t) => L * 0.5 / rmsOfAmps(amps) * prof(t) }));
    st.add(new S.NoiseSrc(sr, { start: h, dur, color: 'pink', R, filters: [new S.Biquad('bp', sr, R.range(700, 1500), 0.7)], gain: (t) => L * prof(t) }));
    return h + R.range(2, 12);
  });
  return { render: (o, off, s, n) => { rumble.render(o, off, s, n); cars.render(o, off, s, n); } };
}

function rain(sr, R) {
  const hissLvl = levelSchedule(R, [-5, 0, 5, 10, 15], [30, 120], 0.8);
  const hiss = new S.NoiseSrc(sr, { color: 'pink', R, filters: [new S.Biquad('hp', sr, 500)], gain: (t) => lvl(0) * hissLvl(t) });
  let rate = R.range(20, 300), tRate = 0;
  const drops = new S.EventStream(sr, (st, h) => {
    if (h > tRate) { rate = R.logRange(20, 300); tRate = h + R.range(20, 90); }
    let t = h; const end = h + 0.25;
    while (t < end) {
      const A = lvl(R.range(-10, 15)) * Math.pow(R.u(), 2), f = R.logRange(800, 8000);
      st.add(new S.Partials(sr, { start: t, dur: 0.04, freqs: [f], amps: [A], decays: [R.range(100, 400)], R }));
      st.add(new S.NoiseSrc(sr, { start: t, dur: 0.002, R, gain: A * 0.7 }));
      t += -Math.log(R.u() || 1e-9) / rate;
    }
    return end;
  });
  const thunder = new S.EventStream(sr, (st, h) => {
    if (h === 0) return R.range(30, 180);
    const d = R.range(3, 10);
    st.add(new S.NoiseSrc(sr, { start: h, dur: d, color: 'brown', R, filters: [new S.Biquad('lp', sr, 120)], gain: S.asr(d, 0.3, d * 0.6, lvl(R.range(5, 25))) }));
    return h + d + R.range(60, 300);
  });
  return { render: (o, off, s, n) => { hiss.render(o, off, s, n); drops.render(o, off, s, n); thunder.render(o, off, s, n); } };
}

function impulsive(sr, R) {
  return new S.EventStream(sr, (st, h) => {
    const kind = R.pick(['door', 'steps', 'typing', 'clink', 'glass', 'drop']);
    let t = h;
    const L = lvl(R.range(5, 35));
    if (kind === 'door') {
      st.add(new S.Partials(sr, { start: t, dur: 0.5, freqs: [R.range(60, 150), R.range(150, 300)], amps: [L * 2, L], decays: [20, 30], R }));
      st.add(new S.NoiseSrc(sr, { start: t, dur: 0.5, R, gain: (u) => L * 1.5 * Math.exp(-u * 30) }));
    } else if (kind === 'steps') {
      const n = R.int(5, 20), per = 1 / R.range(1.6, 2.2);
      for (let k = 0; k < n; k++, t += per) {
        st.add(new S.Partials(sr, { start: t, dur: 0.15, freqs: [R.range(80, 200)], amps: [L], decays: [40], R }));
        st.add(new S.NoiseSrc(sr, { start: t, dur: 0.03, R, filters: [new S.Biquad('lp', sr, 1500)], gain: (u) => L * Math.exp(-u * 100) }));
      }
    } else if (kind === 'typing') {
      const dur = R.range(3, 20);
      for (; t < h + dur; t += -Math.log(R.u() || 1e-9) / R.range(4, 10)) {
        st.add(new S.NoiseSrc(sr, { start: t, dur: 0.006, R, filters: [new S.Biquad('hp', sr, 1000)], gain: (u) => L * 0.5 * Math.exp(-u * 600) }));
        st.add(new S.Partials(sr, { start: t, dur: 0.02, freqs: [R.range(2000, 5000)], amps: [L * 0.2], decays: [300], R }));
      }
    } else if (kind === 'clink') {
      const n = R.int(2, 5);
      st.add(new S.Partials(sr, { start: t, dur: 0.6, freqs: Array.from({ length: n }, () => R.logRange(1000, 6000)), amps: Array.from({ length: n }, () => L * R.range(0.3, 1)), decays: Array.from({ length: n }, () => R.range(5, 30)), R }));
    } else if (kind === 'glass') {                           // a ringing glass / bowl: decaying, steady pitch
      const f = R.logRange(500, 2000), d = R.range(1, 3);
      st.add(new S.Partials(sr, { start: t, dur: d, freqs: [f, f * R.range(2.2, 2.9)], amps: [L * 1.4, L * 0.4], decays: [R.range(2, 6), R.range(5, 12)], R }));
    } else {
      st.add(new S.NoiseSrc(sr, { start: t, dur: 0.2, R, filters: [new S.Biquad('lp', sr, 3000)], gain: (u) => L * 2 * Math.exp(-u * 40) }));
    }
    return Math.max(t, h) + R.range(1, 20);
  });
}

function phoneTones(sr, R) {
  return new S.EventStream(sr, (st, h) => {
    if (h === 0) return R.range(2, 15);
    const kind = R.pick(['chime', 'chime', 'ringtone', 'ringback_eu', 'ringback_us', 'alarm']);
    const L = lvl(R.range(5, 30));
    let t = h;
    const beep = (t0, d, f, amps = [1, S.dbToAmp(-30)]) => st.add(new S.Tone(sr, { start: t0, dur: d, R, amps, f0: () => f, env: S.asr(d, 0.008, 0.015, L * 1.4) }));
    if (kind === 'chime') {
      const n = R.int(2, 4);
      for (let k = 0; k < n; k++) {
        const d = R.range(0.08, 0.3), f = R.logRange(500, 2500);
        if (R.u() < 0.5) beep(t, d, f);
        else st.add(new S.Partials(sr, { start: t, dur: d + 0.5, freqs: [f, f * 4], amps: [L * 1.4, L * 0.3], decays: [R.range(3, 8), 20], R }));
        t += d + R.range(0, 0.1);
      }
    } else if (kind === 'ringtone') {
      const pat = Array.from({ length: R.int(4, 8) }, () => ({ d: R.range(0.1, 0.3), f: R.logRange(400, 1600) }));
      const reps = R.int(5, 15);
      for (let r = 0; r < reps; r++) { for (const p of pat) { beep(t, p.d, p.f); t += p.d; } t += R.range(0.5, 2); }
    } else if (kind === 'ringback_eu') {
      for (let r = 0, n = R.int(4, 8); r < n; r++, t += 5) beep(t, 1, 425, [1]);
    } else if (kind === 'ringback_us') {
      for (let r = 0, n = R.int(3, 6); r < n; r++, t += 6) { beep(t, 2, 440, [1]); beep(t, 2, 480, [1]); }
    } else {
      const f = R.range(1000, 3000), dur = R.range(20, 60);
      for (; t < h + dur; t += 0.2) beep(t, 0.1, f);
    }
    return t + R.range(10, 90);
  });
}

// ── TV: dialogue (with/without music bed) and music segments ─────────────────
function tv(sr, R) {
  return new S.EventStream(sr, (st, h) => {
    const dur = R.range(5, 40), type = R.u();
    const levelDb = R.range(5, 25), lv = (r) => levelDb + r.range(-4, 2);
    if (type < 0.6) {
      st.add(new Segment(sr, h, dur, speechStream(sr, R, ['male', 'female', R.u() < 0.3 ? 'child' : 'female'], lv)));
      if (R.u() < 0.4) st.add(new Segment(sr, h, dur, new Gain(R.u() < 0.5 ? popStream(sr, R) : pianoStream(sr, R), 0.25)));
    } else {
      const m = R.pick([popStream, pianoStream, violinStream, (s, r) => melodicStream(s, r, SINGING)]);
      st.add(new Segment(sr, h, dur, m(sr, R)));
    }
    return h + dur;
  });
}
class Gain {
  constructor(src, g) { this.src = src; this.g = g; this.tmp = null; }
  render(out, off, s, n) {
    if (!this.tmp || this.tmp.length < n) this.tmp = new Float64Array(n);
    const t = this.tmp; t.fill(0, 0, n); this.src.render(t, 0, s, n);
    for (let i = 0; i < n; i++) out[off + i] += this.g * t[i];
  }
}

// ── Coloured-noise classes with level steps ──────────────────────────────────
function noiseClass(color) {
  return (sr, R) => new S.NoiseSrc(sr, { color, R, gain: (() => { const f = levelSchedule(R, [-10, 0, 10, 20], [45, 120]); return (t) => lvl(0) * f(t); })() });
}

// ── Registry ─────────────────────────────────────────────────────────────────
// group: 'noise' | 'machine' | 'voice' | 'music' | 'insect' | 'environment' | 'mosquito'
// countsAsFA: false for male mosquitoes (see EVALUATION.md: they ARE mosquitoes)
const CLASSES = {
  quiet_room:       { group: 'noise',   make: (sr, R) => ({ render() {} }), desc: 'reference room noise only (pink+white, −40 dBFS)' },
  noise_white:      { group: 'noise',   make: noiseClass('white'), desc: 'white noise, level steps −10…+20 dB re room (abrupt/ramped)' },
  noise_pink:       { group: 'noise',   make: noiseClass('pink'),  desc: 'pink noise, level steps' },
  noise_brown:      { group: 'noise',   make: noiseClass('brown'), desc: 'brown noise, level steps' },
  fan_desk:         { group: 'machine', make: fanDesk,    desc: 'desk fan: BPF 36–125 Hz + harmonics, 2×mains hum, broadband AM, speed drift, on/off' },
  fan_ceiling:      { group: 'machine', make: fanCeiling, desc: 'ceiling fan: infrasonic BPF AM on broadband, motor hum' },
  fan_small:        { group: 'machine', make: fanSmall,   desc: 'laptop/PC/USB fan: BPF 150–700 Hz in band, shaft sidebands, thermal speed ramps' },
  hum_fridge:       { group: 'machine', make: humFridge,  desc: 'fridge compressor: slip-speed harmonics, 100/120 Hz magnetostriction, cycling' },
  hum_ac:           { group: 'machine', make: humAC,      desc: 'split AC: cross-flow blower BPF 300–700 Hz, airflow noise, inverter compressor ramps' },
  whine_electrical: { group: 'machine', make: whineElectrical, desc: 'transformer/LED buzz combs + coil whine tones 250–4000 Hz with load jumps' },
  speech_male:      { group: 'voice',   make: (sr, R) => speechStream(sr, R, ['male', 'male'], r => r.range(5, 30)),   desc: 'adult male conversation, f0 85–155 Hz' },
  speech_female:    { group: 'voice',   make: (sr, R) => speechStream(sr, R, ['female', 'female'], r => r.range(5, 30)), desc: 'adult female conversation, f0 165–255 Hz' },
  speech_child:     { group: 'voice',   make: (sr, R) => speechStream(sr, R, ['child', 'child'], r => r.range(5, 30)), desc: 'child speech, f0 240–400 Hz (inside the band)' },
  humming:          { group: 'voice',   make: (sr, R) => melodicStream(sr, R, HUMMING),   desc: 'humming 150–700 Hz, H1-dominant, vibrato on ~50 % of notes' },
  singing:          { group: 'voice',   make: (sr, R) => melodicStream(sr, R, SINGING),   desc: 'singing 130–1000 Hz, vowels, vibrato 5–7 Hz' },
  whistling:        { group: 'voice',   make: (sr, R) => melodicStream(sr, R, WHISTLING), desc: 'whistling 500–1500 Hz near-pure tone, glides, vibrato' },
  music_violin:     { group: 'music',   make: violinStream, desc: 'solo violin 196–1000 Hz, vibrato, double stops' },
  music_flute:      { group: 'music',   make: fluteStream,  desc: 'solo flute 262–1500 Hz, near-pure, vibrato' },
  music_piano:      { group: 'music',   make: pianoStream,  desc: 'piano notes/chords, inharmonic decaying partials' },
  music_pop:        { group: 'music',   make: popStream,    desc: 'band: drums, bass, steady pad chords' },
  tv:               { group: 'music',   make: tv,           desc: 'TV: dialogue (± music bed) and music segments' },
  insect_housefly:  { group: 'insect',  make: (sr, R) => insectStream(sr, R, P.INSECTS.housefly.f0, 'fly', [1, 15], [3, 60], [-20, 10]),  desc: P.INSECTS.housefly.ref },
  insect_honeybee:  { group: 'insect',  make: (sr, R) => insectStream(sr, R, P.INSECTS.honeybee.f0, 'fly', [2, 20], [3, 60], [-20, 10]),  desc: P.INSECTS.honeybee.ref },
  insect_bumblebee: { group: 'insect',  make: (sr, R) => insectStream(sr, R, P.INSECTS.bumblebee.f0, 'fly', [2, 20], [3, 60], [-15, 15]), desc: P.INSECTS.bumblebee.ref },
  insect_fruitfly:  { group: 'insect',  make: (sr, R) => insectStream(sr, R, P.INSECTS.fruitfly.f0, 'fly', [1, 15], [3, 60], [-25, 0]),   desc: P.INSECTS.fruitfly.ref },
  insect_midge:     { group: 'insect',  make: (sr, R) => insectStream(sr, R, P.INSECTS.midge.f0, 'mosquito', [2, 20], [3, 60], [-20, 0]), desc: P.INSECTS.midge.ref + ' — genuinely mosquito-like' },
  crickets:         { group: 'environment', make: crickets, desc: 'field-cricket chirps 3.5–5.5 kHz, tree-cricket trills 1.8–3.2 kHz' },
  birds_song:       { group: 'environment', make: birdsSong, desc: 'passerine song: FM sweeps and trills 2–8 kHz' },
  birds_low:        { group: 'environment', make: birdsLow, desc: 'doves / cuckoo / owls: tonal calls 300–750 Hz' },
  traffic:          { group: 'environment', make: traffic, desc: 'rumble + vehicle pass-bys (engine harmonics with Doppler, tyre noise)' },
  rain:             { group: 'environment', make: rain, desc: 'rain hiss + drop impacts + thunder' },
  impulsive:        { group: 'environment', make: impulsive, desc: 'door slams, footsteps, typing, clinks, ringing glass, drops' },
  phone_tones:      { group: 'environment', make: phoneTones, desc: 'notification chimes, ringtones, ringback 425 Hz (EU) / 440+480 Hz (US), alarms' },
  mosquito_male:    { group: 'mosquito', countsAsFA: false,
    make: (sr, R) => insectStream(sr, R, (r) => { const sp = r.pick(SPECIES_DB); return P.femaleF0(sp, r, P.drawTemperature(r)) * r.range(P.POP.MALE_RATIO[0], P.POP.MALE_RATIO[1]); },
      'mosquito', [2, 20], [3, 60], [-20, 0]), desc: 'male mosquitoes, f0 = 1.3–1.6 × female (non-biting; counted as mosquito, not FA)' },
};

module.exports = { ROOM_RMS, lvl, RoomNoise, CLASSES, Group, Segment, Gain, speechStream, melodicStream, HUMMING, SINGING, WHISTLING,
  violinStream, fluteStream, pianoStream, popStream, insectStream };
