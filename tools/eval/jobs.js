'use strict';
/*
 * Experiment definitions and per-job execution for the evaluation harness.
 *
 * A job is a plain JSON object (so it can be posted to a worker thread) and is
 * fully determined by its seed: the same job gives the same result on any
 * machine, in any order, with any number of workers.
 *
 * Seeds are derived from (split, experiment, …) — the "dev" split is what the
 * detector was tuned on, the "test" split (default) is what is reported.
 *
 * Paired designs (variance reduction, fair comparisons):
 *  • detection sweep: the SAME individual mosquito + room-noise realisation is
 *    replayed at every SNR, only the mosquito's amplitude changes;
 *  • mixtures: each trial is run with and without the adversary (same mosquito,
 *    same room noise).
 */
const { rng, seedFrom } = require('../lib/dsp');
const S = require('./synth');
const P = require('./population');
const C = require('./corpus');
const { makeClassifier, drive, summarize, EventCounter } = require('./runner');
const { SPECIES_DB, getNativeSpecies } = require('../../js/species.js');

const LOCATIONS = {
  none:   null,
  accra:  { lat: 5.6037, lng: -0.1870, label: 'Accra, Ghana (West Africa)' },
  rome:   { lat: 41.9028, lng: 12.4964, label: 'Rome, Italy (Southern Europe)' },
  mumbai: { lat: 19.0760, lng: 72.8777, label: 'Mumbai, India (South Asia)' },
};
function locConfig(name) {
  const L = LOCATIONS[name];
  if (!L) return { name, lat: null, lng: null, disabled: [], natives: SPECIES_DB.map(s => s.id) };
  const nat = getNativeSpecies(L.lat, L.lng);
  // App behaviour (index.html, first GPS fix): every non-native species is disabled.
  return { name, lat: L.lat, lng: L.lng, label: L.label, natives: SPECIES_DB.filter(s => nat.has(s.id)).map(s => s.id),
           disabled: SPECIES_DB.filter(s => !nat.has(s.id)).map(s => s.id) };
}

// Hard, in-band adversaries used for the quick FA check and the sensitivity-level ROC.
const HARD_FA = ['quiet_room', 'fan_small', 'whine_electrical', 'speech_child', 'humming', 'whistling', 'singing',
  'music_violin', 'music_flute', 'birds_low', 'insect_midge', 'insect_housefly', 'phone_tones', 'tv'];
const MIX_CLASSES = ['fan_desk', 'fan_small', 'hum_ac', 'hum_fridge', 'tv', 'speech_female', 'speech_child',
  'music_pop', 'traffic', 'rain', 'whine_electrical'];

const MODES = {
  quick: {
    detect: { snrs: [-28, -18, -8], trials: 2 },
    id:     { locs: ['none'], trials: 2, snr: [-15, -5], nonNative: 0 },
    fa:     { classes: HARD_FA, runs: 1, minutes: 1 },
    mix:    { classes: ['fan_small', 'tv', 'speech_child', 'fan_desk'], trials: 2, snr: -12 },
    levels: null,
    compute: { seconds: 30 },
  },
  // Tuning mode (use with --split dev): every adversary class, moderate trials.
  dev: {
    detect: { snrs: [-32, -26, -20, -14, -8, -2], trials: 3 },
    id:     { locs: ['none'], trials: 3, snr: [-15, -5], nonNative: 0 },
    fa:     { classes: Object.keys(C.CLASSES), runs: 1, minutes: 2 },
    mix:    { classes: ['fan_small', 'tv', 'speech_child', 'fan_desk', 'music_pop', 'hum_ac'], trials: 4, snr: -12 },
    levels: null,
    compute: { seconds: 30 },
  },
  full: {
    detect: { snrs: [-36, -32, -28, -24, -20, -16, -12, -8, -4, 0, 4], trials: 12 },
    id:     { locs: ['none', 'accra', 'rome', 'mumbai'], trials: 12, snr: [-15, -5], nonNative: 3 },
    fa:     { classes: Object.keys(C.CLASSES), runs: 4, minutes: 5 },
    mix:    { classes: MIX_CLASSES, trials: 20, snr: -12 },
    levels: { levels: [1, 2, 4, 5], detect: { snrs: [-32, -24, -16, -8, 0], trials: 4 }, fa: { classes: HARD_FA, runs: 1, minutes: 5 } },
    compute: { seconds: 60 },
  },
};

const SR = 44100, N = 8192;

function buildJobs(mode, split, opts = {}) {
  const M = MODES[mode], jobs = [];
  const sr = opts.sr || SR, n = opts.N || N;
  const want = (e) => !opts.only || opts.only.includes(e);
  if (want('detect')) {
    for (const sp of SPECIES_DB) for (let t = 0; t < M.detect.trials; t++) for (const snr of M.detect.snrs) {
      jobs.push({ kind: 'trial', exp: 'detect', sp: sp.id, snr, trial: t, level: 3, loc: 'none',
        seed: seedFrom(split, 'detect', sp.id, t), onset: 3, bout: 10, tail: 1, sr, N: n, cost: 14 });
    }
  }
  if (want('id')) {
    for (const ln of M.id.locs) {
      const L = locConfig(ln);
      for (const id of L.natives) for (let t = 0; t < M.id.trials; t++) {
        jobs.push({ kind: 'trial', exp: 'id', sp: id, trial: t, level: 3, loc: ln, native: true, snrRange: M.id.snr,
          seed: seedFrom(split, 'id', ln, id, t), onset: 3, bout: 8, tail: 0.5, sr, N: n, cost: 11.5 });
      }
      if (ln !== 'none') for (const sp of SPECIES_DB) if (!L.natives.includes(sp.id)) for (let t = 0; t < M.id.nonNative; t++) {
        jobs.push({ kind: 'trial', exp: 'id', sp: sp.id, trial: t, level: 3, loc: ln, native: false, snrRange: M.id.snr,
          seed: seedFrom(split, 'id-nonnative', ln, sp.id, t), onset: 3, bout: 8, tail: 0.5, sr, N: n, cost: 11.5 });
      }
    }
  }
  if (want('mix')) {
    for (const cls of ['none', ...M.mix.classes]) for (let t = 0; t < M.mix.trials; t++) {
      jobs.push({ kind: 'trial', exp: 'mix', adversary: cls === 'none' ? null : cls, mixClass: cls, trial: t, level: 3, loc: 'none',
        sp: SPECIES_DB[seedFrom(split, 'mix-sp', t) % SPECIES_DB.length].id, snr: M.mix.snr,
        seed: seedFrom(split, 'mix', t), advSeed: seedFrom(split, 'mix-adv', cls, t), onset: 15, bout: 10, tail: 1, sr, N: n, cost: 26 });
    }
  }
  if (want('fa')) {
    for (const cls of M.fa.classes) for (let r = 0; r < M.fa.runs; r++) {
      jobs.push({ kind: 'fa', exp: 'fa', cls, run: r, level: 3, minutes: opts.minutes || M.fa.minutes,
        seed: seedFrom(split, 'fa', cls, r), sr, N: n, cost: 60 * (opts.minutes || M.fa.minutes) * (cls.startsWith('music_piano') ? 2 : 1) });
    }
  }
  if (want('levels') && M.levels) {
    for (const level of M.levels.levels) {
      for (const sp of SPECIES_DB) for (let t = 0; t < M.levels.detect.trials; t++) for (const snr of M.levels.detect.snrs) {
        jobs.push({ kind: 'trial', exp: 'levels', sp: sp.id, snr, trial: t, level, loc: 'none',
          seed: seedFrom(split, 'detect', sp.id, t), onset: 3, bout: 10, tail: 1, sr, N: n, cost: 14 });
      }
      for (const cls of M.levels.fa.classes) for (let r = 0; r < M.levels.fa.runs; r++) {
        jobs.push({ kind: 'fa', exp: 'levels-fa', cls, run: r, level, minutes: M.levels.fa.minutes,
          seed: seedFrom(split, 'fa', cls, r), sr, N: n, cost: 60 * M.levels.fa.minutes });
      }
    }
  }
  return jobs;
}

// ── scene construction ───────────────────────────────────────────────────────
function mosquitoTrialScene(job) {
  const sr = job.sr;
  const Ri = rng(seedFrom(job.seed, 'individual'));
  const sp = SPECIES_DB.find(s => s.id === job.sp);
  const T = P.drawTemperature(Ri), f0 = P.femaleF0(sp, Ri, T), flight = P.drawFlight(Ri);
  const snr = job.snrRange ? job.snrRange[0] + (job.snrRange[1] - job.snrRange[0]) * Ri.u() : job.snr;
  const p1 = (C.ROOM_RMS ** 2) * Math.pow(10, snr / 10);
  const parts = [
    new C.RoomNoise(sr, rng(seedFrom(job.seed, 'room'))),
    P.flightTone(sr, { f0, start: job.onset, dur: job.bout, p1, flight, R: rng(seedFrom(job.seed, 'tone')) }),
  ];
  if (job.adversary) parts.push(C.CLASSES[job.adversary].make(sr, rng(job.advSeed)));
  return { scene: new S.Scene(sr, parts), f0, T, snr };
}

function runTrial(job, eng) {
  const L = locConfig(job.loc);
  const clf = makeClassifier(eng, { sr: job.sr, N: job.N, level: job.level, lat: L.lat, lng: L.lng, disabled: L.disabled });
  const { scene, f0, T, snr } = mosquitoTrialScene(job);
  const t1 = job.onset, t2 = job.onset + job.bout + 0.5;
  let first = null, last = null, firstT = null, nDet = 0, nBout = 0, preFA = 0, multi = 0, peakConf = 0;
  let matchT = null, otherFrames = 0;      // f0-matched (±5 %) detection of THIS mosquito; other locks in the bout
  const confs = [];
  drive(clf, scene, {
    sr: job.sr, N: job.N, seconds: job.onset + job.bout + job.tail,
    onFrame: (i, t, dets) => {
      const inBout = t > t1 && t <= t2;
      if (inBout) nBout++;
      if (!dets.length) return;
      if (!inBout) { if (t <= t1) preFA++; return; }
      nDet++;
      const top = dets[0];
      if (dets.some(d => Math.abs((d.f0 ?? d.frequency) / f0 - 1) < 0.05)) { if (matchT === null) matchT = t; }
      else otherFrames++;
      multi = Math.max(multi, new Set(dets.map(d => d.species.id)).size);
      if (!first) { first = summarize(top); firstT = t; }
      last = summarize(top);
      peakConf = Math.max(peakConf, top.confidence);
      if (confs.length < 200) confs.push(+(+top.confidence).toFixed(4));
    },
  });
  return {
    exp: job.exp, sp: job.sp, snr: +snr.toFixed(2), trial: job.trial, level: job.level, loc: job.loc, native: job.native,
    mixClass: job.mixClass, f0True: +f0.toFixed(2), T: +T.toFixed(2),
    detected: !!first, latency: first ? +(firstT - t1).toFixed(3) : null,
    detectedMatch: matchT !== null, latencyMatch: matchT !== null ? +(matchT - t1).toFixed(3) : null, otherFrames,
    lockedFrac: nBout ? +(nDet / nBout).toFixed(4) : 0, preFA, multi, peakConf: +peakConf.toFixed(4),
    first, last, confs,
  };
}

function runFA(job, eng) {
  const cls = C.CLASSES[job.cls];
  const clf = makeClassifier(eng, { sr: job.sr, N: job.N, level: job.level });
  const sr = job.sr;
  const scene = new S.Scene(sr, [new C.RoomNoise(sr, rng(seedFrom(job.seed, 'room'))), cls.make(sr, rng(job.seed))]);
  const ev = new EventCounter(60);
  let locked = 0, frames = 0, lvl30 = 0, maxLvl = 0, calLocked = 0;
  const perSpecies = {};
  drive(clf, scene, {
    sr, N: job.N, seconds: job.minutes * 60,
    onFrame: (i, t, dets, c) => {
      frames++;
      const raw = c.getRawConfidences ? c.getRawConfidences() : {};
      let m = 0; for (const k in raw) if (raw[k] > m) m = raw[k];
      if (m > maxLvl) maxLvl = m;
      if (m >= 0.3) lvl30++;
      if (!dets.length) return;
      locked++;
      if (t <= 2.25) calLocked++;
      for (const d of dets) perSpecies[d.species.id] = (perSpecies[d.species.id] || 0) + 1;
      ev.push(t, dets);
    },
  });
  return { exp: job.exp, cls: job.cls, run: job.run, level: job.level, seconds: job.minutes * 60, frames, locked, calLocked,
           events: ev.events, perSpecies, maxLevel: +maxLvl.toFixed(3), fracLevel30: +(lvl30 / Math.max(1, frames)).toFixed(5) };
}

function execute(job, eng) {
  if (job.kind === 'trial') return runTrial(job, eng);
  if (job.kind === 'fa') return runFA(job, eng);
  throw new Error('unknown job kind ' + job.kind);
}

// Compute cost: a fixed 3-part scene (quiet → mosquito at −15 dB → speech), timed per analyze().
function computeBench(eng, seconds = 60, sr = SR, n = N) {
  const R = rng(seedFrom('compute'));
  const sp = SPECIES_DB.find(s => s.id === 'aedes_aegypti');
  const parts = [
    new C.RoomNoise(sr, rng(1)),
    P.flightTone(sr, { f0: sp.freq, start: seconds / 3, dur: seconds / 3, p1: (C.ROOM_RMS ** 2) * Math.pow(10, -1.5), flight: P.drawFlight(R), R }),
    new C.Segment(sr, 2 * seconds / 3, seconds / 3, C.speechStream(sr, R, ['female', 'male'], r => r.range(10, 25))),
  ];
  const clf = makeClassifier(eng, { sr, N: n, level: 3 });
  const r = drive(clf, new S.Scene(sr, parts), { sr, N: n, seconds, timing: true, onFrame: () => {} });
  const ts = r.times.slice(5).sort((a, b) => a - b);
  const q = (p) => ts[Math.min(ts.length - 1, Math.floor(p * ts.length))];
  return { calls: ts.length, meanMs: +(ts.reduce((s, x) => s + x, 0) / ts.length).toFixed(3), p50Ms: +q(0.5).toFixed(3),
           p95Ms: +q(0.95).toFixed(3), maxMs: +ts[ts.length - 1].toFixed(3) };
}

module.exports = { MODES, LOCATIONS, HARD_FA, MIX_CLASSES, locConfig, buildJobs, execute, computeBench, SR, N, _scene: (job) => mosquitoTrialScene(job).scene, _sceneFull: mosquitoTrialScene };
