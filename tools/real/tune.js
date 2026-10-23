#!/usr/bin/env node
'use strict';
/*
 * Tune the detector's plausibility gates on REAL recordings (tools/real/SOURCES.md).
 *
 *   node tools/real/tune.js cache                 # spectra of every recording, once (~2 min)
 *   node tools/real/tune.js sweep configs.json [--split dev|test] [--jobs 4] > results.json
 *
 * Protocol
 *   • Every recording is split in time: the FIRST half is the dev split used
 *     to choose parameters, the SECOND half is the test split, touched only to
 *     report the chosen configuration. Each half is streamed separately with
 *     the app's 2 s start-up calibration.
 *   • Ground truth "a mosquito is audible" comes from a detector-INDEPENDENT
 *     estimator: per-bin whitening by the half's median (removes stationary
 *     lines, e.g. the 284.5 Hz line present throughout the CDC culture-room
 *     recordings), harmonic sum H1+0.7H2+0.5H3 over 380–900 Hz, tone ≥ 12 dB
 *     above its ±150 Hz neighbourhood and ≥ 12 dB above its own bin's median,
 *     nothing at f0/2. A 10-s window with ≥ 3 such frames is "audible".
 *     Recall = audible windows with ≥ 1 detection / audible windows.
 *   • False alarms: detection events (counted like the app) per hour on the
 *     real non-mosquito recordings.
 *   • ESC-50 is NOT used here — it is the held-out false-alarm test
 *     (tools/real/realdata.js).
 */
const fs = require('fs');
const path = require('path');
const os = require('os');
const { execFileSync, spawn } = require('child_process');
const H = require('../../tests/helpers');
const R = require('./realdata');

const SR = 44100, N = H.N, HOP = H.HOP, BINS = 1120;     // keep 0–6 kHz (sources are ≤ 4 kHz)
const CACHE = process.env.REAL_CACHE || path.join(os.tmpdir(), 'mn-real-cache');
const MOZZ = process.env.MOZZBNN_DIR || '/home/user/humbug-mosquito/mozzbnn/Code/lib/VAD-Humbug/input';

const FILES = [
  { name: 'cdc-ae-aegypti', file: 'CDC_Ae-aegypti.wav', kind: 'mosquito', genus: 'Aedes', sp: 'aedes_aegypti', loc: [33.75, -84.39] },
  { name: 'cdc-ae-albopictus', file: 'CDC_Ae-albopictus.wav', kind: 'mosquito', genus: 'Aedes', sp: 'aedes_albopictus', loc: [33.75, -84.39] },
  { name: 'cdc-cx-quinquefasciatus', file: 'CDC_Cx-quinquefasciatus.wav', kind: 'mosquito', genus: 'Culex', sp: 'culex', loc: [33.75, -84.39] },
  { name: 'ifakara-field', file: 'IHI_tascam_1-2_concatenated.wav', kind: 'field', loc: [-8.13, 36.68] },
  { name: 'drc-field', file: 'Humbug_DRC_cleaned_concatenated.wav', kind: 'field', loc: [-4.32, 15.31] },
  { name: 'bus-stop', file: 'bus_stop.wav', kind: 'background' },
  { name: 'construction-site', file: 'construction_site.wav', kind: 'background' },
  { name: 'speech-background', file: 'background_with_speech.wav', kind: 'background' },
];

// ── 1. cache spectra ──────────────────────────────────────────────────────
function cache() {
  fs.mkdirSync(CACHE, { recursive: true });
  for (const f of FILES) {
    const out = path.join(CACHE, `${f.name}.f32`);
    if (fs.existsSync(out)) continue;
    const { x, sr } = R.readWav(path.join(MOZZ, f.file));
    const y = R.resample(x, sr, SR);
    const excl = f.kind === 'mosquito'
      ? fs.readFileSync(path.join(MOZZ, 'speech_labels', f.file.replace(/\.wav$/, '.txt')), 'utf8')
          .trim().split(/\n/).map(l => l.trim().split(/\s+/).map(Number)).filter(a => a.length >= 2) : [];
    const nF = Math.floor((y.length - N) / HOP) + 1, buf = new Float32Array(nF * BINS), t = [], ex = [];
    for (let i = 0; i < nF; i++) {
      const fr = y.subarray(i * HOP, i * HOP + N), db = H.analyserDb(fr);
      for (let k = 0; k < BINS; k++) buf[i * BINS + k] = Number.isFinite(db[k]) ? db[k] : -300;
      const tt = (i * HOP + N) / SR;
      t.push(tt); ex.push(excl.some(([a, b]) => tt >= a && tt - N / SR < b) ? 1 : 0);
    }
    fs.writeFileSync(out, Buffer.from(buf.buffer));
    fs.writeFileSync(path.join(CACHE, `${f.name}.json`), JSON.stringify({ frames: nF, bins: BINS, excl: ex }));
    console.error(`cached ${f.name}: ${nF} frames`);
  }
}

function load(f) {
  const meta = JSON.parse(fs.readFileSync(path.join(CACHE, `${f.name}.json`), 'utf8'));
  const b = fs.readFileSync(path.join(CACHE, `${f.name}.f32`));
  return { ...meta, spec: new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4) };
}

// ── 2. independent "audible mosquito" labels ──────────────────────────────
const BIN = SR / N;
function audibleFrames(spec, from, to, f0s = null) {   // f0s: optional Float32Array(to-from) of tone f0s
  const kLo = Math.round(380 / BIN), kHi = Math.round(900 / BIN), kMax = 3 * kHi + 2, nb = Math.round(150 / BIN);
  const n = to - from, med = new Float64Array(kMax + 1), col = new Float64Array(n);
  for (let k = 1; k <= kMax; k++) {                       // per-bin median power over the half
    for (let i = 0; i < n; i++) col[i] = Math.pow(10, spec[(from + i) * BINS + k] / 10);
    med[k] = col.slice().sort()[n >> 1] + 1e-30;
  }
  const out = new Uint8Array(n), W = new Float64Array(kMax + 2);
  const m3 = (k) => Math.max(W[k - 1], W[k], W[k + 1]);
  for (let i = 0; i < n; i++) {
    for (let k = 1; k <= kMax; k++) W[k] = Math.pow(10, spec[(from + i) * BINS + k] / 10) / med[k];
    let best = -1, bs = 0;
    for (let k = kLo; k <= kHi; k++) { const s = W[k] + 0.7 * m3(2 * k) + 0.5 * m3(3 * k); if (s > bs) { bs = s; best = k; } }
    if (best < 0 || W[best] < 16) continue;
    const loc = []; for (let k = best - nb; k <= best + nb; k++) loc.push(W[k]);
    loc.sort((a, b) => a - b);
    if (W[best] < 15.85 * loc[loc.length >> 1]) continue;
    if (m3(Math.round(best / 2)) > 0.3 * W[best]) continue;
    out[i] = 1;
    if (f0s) f0s[i] = best * BIN;
  }
  return out;
}

// ── 3. run one configuration over one half ────────────────────────────────
const eng = H.loadEngine();
const FULL = new Float32Array(N / 2);
function runHalf(data, from, to, cfg, loc, level = 3) {
  const clf = new eng.MosquitoClassifier({ sampleRate: SR, fftSize: N, binHz: BIN });
  // cfg.level selects a shipped sensitivity level; other keys override constants.
  const { level: lv, ...over } = cfg || {};
  clf.setSensitivity(lv || level);
  if (Object.keys(over).length) clf._tune(over);
  clf.setTimeOfDay('none');
  if (loc) clf.setLocation(loc[0], loc[1]);
  clf.beginCalibration();
  const calib = Math.round(2 * SR / HOP), dets = [];
  for (let i = from; i < to; i++) {
    if (i - from === calib) clf.endCalibration();
    FULL.fill(-160);
    FULL.set(data.spec.subarray(i * BINS, i * BINS + BINS));
    const d = clf.analyze(FULL, null);
    dets.push(i - from < calib || data.excl[i] ? [] : d.map(q => ({ track: q.trackId, genus: q.genus, sp: q.species.id, level: q.taxonLevel, conf: q.confidence })));
  }
  return dets;
}

function countEvents(dets) {
  const byTrack = new Map(), byGenus = {}, evs = [];
  dets.forEach((ds, i) => {
    const t = i * HOP / SR;
    for (const d of ds) {
      let ev = byTrack.get(d.track);
      if (!ev) { const g = byGenus[d.genus]; if (g && t - g.last < 60) ev = g; }
      if (!ev) { ev = { t, last: t, genus: d.genus, sp: d.sp, level: d.level }; evs.push(ev); }
      ev.last = t; if (d.level === 'species') { ev.sp = d.sp; ev.level = 'species'; }
      byTrack.set(d.track, ev); byGenus[d.genus] = ev;
    }
  });
  return evs;
}

function evaluate(cfg, split, labels) {
  const res = { cfg, split, files: {} };
  let posWin = 0, hitWin = 0, bgEv = 0, bgH = 0, genusOk = 0, genusN = 0, spOk = 0, spN = 0, fieldEv = 0, fieldH = 0;
  for (const f of FILES) {
    const data = load(f), mid = Math.floor(data.frames / 2);
    const [from, to] = split === 'dev' ? [0, mid] : [mid, data.frames];
    const dets = runHalf(data, from, to, cfg, f.loc);
    const evs = countEvents(dets), hours = (to - from) * HOP / SR / 3600;
    const r = { events: evs.length, hours: +hours.toFixed(3) };
    if (f.kind === 'mosquito') {
      const lab = labels[f.name][split], W = Math.round(10 * SR / HOP);
      let p = 0, h = 0;
      for (let w = 0; w + W <= lab.length; w += W) {
        let a = 0, d = false;
        for (let i = w; i < w + W; i++) { a += lab[i]; if (dets[i] && dets[i].length) d = true; }
        if (a >= 3) { p++; if (d) h++; }
      }
      r.audibleWindows = p; r.detectedWindows = h; posWin += p; hitWin += h;
      for (const e of evs) { genusN++; if (e.genus === f.genus) genusOk++; if (e.level === 'species') { spN++; if (e.sp === f.sp) spOk++; } }
    } else if (f.kind === 'background') { bgEv += evs.length; bgH += hours; }
    else { fieldEv += evs.length; fieldH += hours; }
    res.files[f.name] = r;
  }
  res.recall = posWin ? +(hitWin / posWin).toFixed(3) : null;
  res.bgFAperHour = +(bgEv / bgH).toFixed(2);
  res.bgEvents = bgEv;
  res.fieldEventsPerHour = +(fieldEv / fieldH).toFixed(1);
  res.genusAcc = genusN ? +(genusOk / genusN).toFixed(3) : null;
  res.speciesAcc = spN ? +(spOk / spN).toFixed(3) : null;
  res.speciesN = spN;
  return res;
}

function labelsFor(names) {
  const L = {};
  for (const f of FILES.filter(q => q.kind === 'mosquito' && (!names || names.includes(q.name)))) {
    const d = load(f), mid = Math.floor(d.frames / 2);
    L[f.name] = { dev: audibleFrames(d.spec, 0, mid), test: audibleFrames(d.spec, mid, d.frames) };
  }
  return L;
}

// ── CLI ───────────────────────────────────────────────────────────────────
function main() {
  const [cmd, arg, ...rest] = process.argv.slice(2);
  const opt = (k, d) => { const i = rest.indexOf(k); return i >= 0 ? rest[i + 1] : d; };
  if (cmd === 'cache') return cache();
  if (cmd === 'worker') {                       // internal: evaluate configs[i] for i ≡ part (mod jobs)
    const configs = JSON.parse(fs.readFileSync(arg, 'utf8')), split = opt('--split', 'dev');
    const part = +opt('--part', 0), jobs = +opt('--jobs', 1);
    const labels = labelsFor();
    for (let i = part; i < configs.length; i += jobs) process.stdout.write(JSON.stringify({ i, ...evaluate(configs[i], split, labels) }) + '\n');
    return;
  }
  if (cmd === 'sweep') {
    const jobs = +opt('--jobs', Math.min(4, os.cpus().length)), split = opt('--split', 'dev');
    const procs = Array.from({ length: jobs }, (_, p) => new Promise((res, rej) => {
      const c = spawn(process.execPath, [__filename, 'worker', arg, '--split', split, '--part', String(p), '--jobs', String(jobs)]);
      let out = ''; c.stdout.on('data', d => { out += d; process.stderr.write('.'); });
      c.stderr.on('data', d => process.stderr.write(d));
      c.on('close', code => code ? rej(new Error(`worker ${p} exited ${code}`)) : res(out));
    }));
    Promise.all(procs).then(outs => {
      const rows = outs.join('').trim().split('\n').filter(Boolean).map(l => JSON.parse(l)).sort((a, b) => a.i - b.i);
      process.stdout.write(JSON.stringify(rows, null, 1) + '\n');
    });
    return;
  }
  console.error('usage: tune.js cache | sweep <configs.json> [--split dev|test] [--jobs N]');
  process.exit(2);
}

if (require.main === module) main();
module.exports = { FILES, evaluate, labelsFor, load, cache, audibleFrames };
