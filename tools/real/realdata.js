#!/usr/bin/env node
'use strict';
/*
 * Real-audio evaluation of the detection engine.
 *
 * Runs the UNMODIFIED engine (js/species.js + js/detector.js, loaded as the
 * page loads them) over published, real recordings and reports what a user of
 * the app would have seen: time locked, detection EVENTS (counted like the app:
 * one per tracked insect, a re-acquired track of the same genus within 60 s is
 * the same insect), genus/species of each event, and the measured wingbeat f0.
 *
 * Sources (see tools/real/SOURCES.md; audio is NOT committed — run
 * tools/real/fetch.sh to clone it):
 *   • HumBug project (Univ. of Oxford) — MozzBNN repository: phone recordings
 *     (8 kHz) of CDC insect-culture Ae. aegypti, Ae. albopictus and
 *     Cx. quinquefasciatus, field recordings from Ifakara (Tanzania) and the DRC,
 *     and real non-mosquito backgrounds (bus stop, construction site, speech).
 *   • ESC-50 (Piczak 2015): 2,000 labelled 5-s real recordings, 50 classes.
 *
 *   node tools/real/realdata.js [--only name,name] [--json out.json] [--level 3]
 */
const fs = require('fs');
const path = require('path');
const H = require('../../tests/helpers');

const SR = 44100, N = H.N, HOP = H.HOP;
const MOZZ = process.env.MOZZBNN_DIR || '/home/user/humbug-mosquito/mozzbnn/Code/lib/VAD-Humbug/input';
const ESC = process.env.ESC50_DIR || '/home/user/karolpiczak/esc-50';

// ── WAV I/O (PCM 16-bit, any rate; channels averaged) ─────────────────────
function readWav(file) {
  const b = fs.readFileSync(file);
  if (b.toString('ascii', 0, 4) !== 'RIFF' || b.toString('ascii', 8, 12) !== 'WAVE') throw new Error(`not a WAV: ${file}`);
  let off = 12, fmt = null;
  while (off + 8 <= b.length) {
    const id = b.toString('ascii', off, off + 4), sz = b.readUInt32LE(off + 4), body = off + 8;
    if (id === 'fmt ') fmt = { format: b.readUInt16LE(body), ch: b.readUInt16LE(body + 2), sr: b.readUInt32LE(body + 4), bits: b.readUInt16LE(body + 14) };
    if (id === 'data') {
      // PCM 16- or 24-bit (WAVE_FORMAT_EXTENSIBLE 0xFFFE carries PCM too).
      if (!fmt || (fmt.format !== 1 && fmt.format !== 0xFFFE) || (fmt.bits !== 16 && fmt.bits !== 24)) {
        throw new Error(`unsupported WAV format in ${file}`);
      }
      const bps = fmt.bits / 8, full = fmt.bits === 16 ? 32768 : 8388608;
      const rd = fmt.bits === 16 ? (o) => b.readInt16LE(o) : (o) => b.readIntLE(o, 3);
      const n = Math.floor(Math.min(sz, b.length - body) / (bps * fmt.ch)), x = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let c = 0; c < fmt.ch; c++) s += rd(body + bps * (i * fmt.ch + c));
        x[i] = s / fmt.ch / full;
      }
      return { x, sr: fmt.sr };
    }
    off = body + sz + (sz & 1);
  }
  throw new Error(`no data chunk in ${file}`);
}

// Band-limited resampling: polyphase windowed-sinc (Blackman), ±24 zero
// crossings, cut-off at 0.95 × the lower Nyquist. The rate ratio is reduced to
// L/M (44100/8000 = 441/80), so every output sample uses one of L precomputed
// phases exactly. The app hears the room at 44.1 kHz; 8 kHz phone audio simply
// has nothing above 4 kHz, as on a cheap handset.
function resample(x, from, to) {
  if (from === to) return x;
  const g = (a, b) => (b ? g(b, a % b) : a), d = g(from, to), L = to / d, M = from / d;
  const fc = 0.95 * Math.min(from, to) / 2 / from;        // cycles per input sample
  const half = Math.ceil(24 / (2 * fc)), taps = 2 * half;
  const bank = Array.from({ length: L }, (_, ph) => {      // phase ph: output at input time c + ph/L
    const frac = ph / L, h = new Float64Array(taps);
    let w = 0;
    for (let j = 0; j < taps; j++) {
      const dd = frac - (j - half + 1), u = dd / half;
      if (u <= -1 || u >= 1) continue;
      const win = 0.42 + 0.5 * Math.cos(Math.PI * u) + 0.08 * Math.cos(2 * Math.PI * u);
      h[j] = (dd === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * dd) / (Math.PI * dd)) * win;
      w += h[j];
    }
    for (let j = 0; j < taps; j++) h[j] /= (w || 1);       // unity DC gain
    return h;
  });
  const n = Math.floor(x.length * L / M), y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const num = i * M, c = Math.floor(num / L), h = bank[num - c * L];
    let s = 0;
    for (let j = 0, k = c - half + 1; j < taps; j++, k++) if (k >= 0 && k < x.length) s += x[k] * h[j];
    y[i] = s;
  }
  return y;
}

// Seconds excluded from a file (e.g. narrated speech before a culture recording).
function readSegments(file) {
  if (!fs.existsSync(file)) return [];
  return fs.readFileSync(file, 'utf8').split(/\r?\n/).map(l => l.trim().split(/\s+/).map(Number))
    .filter(a => a.length >= 2 && Number.isFinite(a[0]) && Number.isFinite(a[1])).map(a => [a[0], a[1]]);
}

// ── Drive the engine like the app (2 s calibration, then hop-by-hop) ──────
function analyse(x, { level = 3, lat = null, lng = null, exclude = [], eng }) {
  const clf = new eng.MosquitoClassifier({ sampleRate: SR, fftSize: N, binHz: SR / N,
    freqToBin: (hz) => Math.min(Math.round(hz / (SR / N)), N / 2 - 1) });   // legacy engine shape too
  clf.setSensitivity(level);
  clf.setLocation(lat, lng);
  if (typeof clf.setTimeOfDay === 'function') clf.setTimeOfDay('none');
  if (typeof clf.getActivityFactor === 'function') clf.getActivityFactor = () => 1.0;   // pre-v3 engines
  clf.beginCalibration();
  const calib = Math.round(2 * SR / HOP);
  const frames = [];
  const inExcl = (t) => exclude.some(([a, b]) => t >= a && t < b);
  let i = 0;
  for (let end = N; end <= x.length; end += HOP, i++) {
    if (i === calib) clf.endCalibration();
    const fr = x.subarray(end - N, end);
    const dets = clf.analyze(H.analyserDb(fr), Float32Array.from(fr));
    const t = end / SR;
    frames.push({ t, excl: inExcl(t) || inExcl(t - N / SR), cal: i < calib,
                  dets: dets.map(d => ({ sp: d.species.id, genus: d.genus, level: d.taxonLevel, f0: d.f0,
                                         conf: d.confidence, track: d.trackId,
                                         pSp: ((d.posterior || []).find(p => p.id === d.species.id) || {}).p || 0,
                                         maleP: d.maleP, genusP: Object.fromEntries((d.genusPosterior || []).map(g => [g.genus, g.p])) })) });
  }
  return frames;
}

// App-equivalent event counting (see App.handleDetections in index.html).
function events(frames) {
  const evs = [], byTrack = new Map(), byGenus = {};
  for (const f of frames) {
    if (f.excl || f.cal) continue;
    for (const d of f.dets) {
      const tk = d.track ?? d.sp;                          // engines without track ids: per species
      let ev = byTrack.get(tk);
      if (!ev) { const g = byGenus[d.genus]; if (g && f.t - g.last < 60) ev = g; }
      if (!ev) { ev = { t: f.t, last: f.t, genus: d.genus, sp: d.sp, level: d.level, pSp: d.pSp, peak: d.conf, f0s: [] }; evs.push(ev); }
      ev.last = f.t; ev.peak = Math.max(ev.peak, d.conf); ev.f0s.push(d.f0);
      if (d.level === 'species' && (ev.level !== 'species' || d.pSp > ev.pSp)) { ev.sp = d.sp; ev.level = 'species'; ev.pSp = d.pSp; }
      byTrack.set(tk, ev); byGenus[d.genus] = ev;
    }
  }
  return evs;
}

const median = (a) => { if (!a.length) return null; const s = [...a].sort((p, q) => p - q); return s[s.length >> 1]; };
const pct = (a, q) => { if (!a.length) return null; const s = [...a].sort((p, r) => p - r); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };

function summarise(name, frames, truth) {
  const live = frames.filter(f => !f.excl && !f.cal);
  const hours = live.length * HOP / SR / 3600;
  const locked = live.filter(f => f.dets.length).length / (live.length || 1);
  const evs = events(frames);
  const f0 = live.flatMap(f => f.dets.map(d => d.f0)).filter(Number.isFinite);
  const genus = {}, species = {};
  for (const e of evs) { genus[e.genus] = (genus[e.genus] || 0) + 1; if (e.level === 'species') species[e.sp] = (species[e.sp] || 0) + 1; }
  const out = { name, truth, minutes: +(hours * 60).toFixed(1), lockedFraction: +locked.toFixed(3),
                events: evs.length, eventsPerHour: +(evs.length / (hours || 1)).toFixed(1),
                uploadable: evs.filter(e => e.level === 'species' && e.peak >= 0.7).length,
                genus, species,
                f0: f0.length ? { p10: +pct(f0, 0.1).toFixed(1), median: +median(f0).toFixed(1), p90: +pct(f0, 0.9).toFixed(1), n: f0.length } : null };
  if (truth && truth.genus) {
    const ok = evs.filter(e => e.genus === truth.genus).length;
    out.genusAccuracy = evs.length ? +(ok / evs.length).toFixed(3) : null;
    if (truth.species) {
      const sl = evs.filter(e => e.level === 'species');
      out.speciesLevelEvents = sl.length;
      out.speciesAccuracy = sl.length ? +(sl.filter(e => e.sp === truth.species).length / sl.length).toFixed(3) : null;
    }
  }
  return out;
}

// ── Corpus ────────────────────────────────────────────────────────────────
const ATLANTA = { lat: 33.75, lng: -84.39 }, IFAKARA = { lat: -8.13, lng: 36.68 }, KINSHASA = { lat: -4.32, lng: 15.31 };
const MOZZ_FILES = [
  { name: 'cdc-ae-aegypti',   file: 'CDC_Ae-aegypti.wav',   kind: 'mosquito', truth: { genus: 'Aedes', species: 'aedes_aegypti' }, loc: ATLANTA },
  { name: 'cdc-ae-albopictus', file: 'CDC_Ae-albopictus.wav', kind: 'mosquito', truth: { genus: 'Aedes', species: 'aedes_albopictus' }, loc: ATLANTA },
  { name: 'cdc-cx-quinquefasciatus', file: 'CDC_Cx-quinquefasciatus.wav', kind: 'mosquito', truth: { genus: 'Culex', species: 'culex' }, loc: ATLANTA },
  { name: 'ifakara-field',    file: 'IHI_tascam_1-2_concatenated.wav', kind: 'mosquito', truth: null, loc: IFAKARA, noExclude: true },
  { name: 'drc-field',        file: 'Humbug_DRC_cleaned_concatenated.wav', kind: 'mosquito', truth: null, loc: KINSHASA, noExclude: true },
  { name: 'bus-stop',         file: 'bus_stop.wav', kind: 'background' },
  { name: 'construction-site', file: 'construction_site.wav', kind: 'background' },
  { name: 'speech-background', file: 'background_with_speech.wav', kind: 'background', noExclude: true },
];

function escFiles() {
  const meta = path.join(ESC, 'meta', 'esc50.csv');
  if (!fs.existsSync(meta)) return [];
  const rows = fs.readFileSync(meta, 'utf8').trim().split('\n').slice(1).map(l => l.split(','));
  const byCat = {};
  for (const r of rows) (byCat[r[3]] = byCat[r[3]] || []).push(path.join(ESC, 'audio', r[0]));
  return Object.entries(byCat).map(([cat, files]) => ({ name: `esc50-${cat}`, files, kind: cat === 'insects' ? 'insects' : 'background' }));
}

function main() {
  const a = process.argv.slice(2);
  const opt = (k) => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : null; };
  const only = opt('--only') ? new Set(opt('--only').split(',')) : null;
  const level = +(opt('--level') || 3);
  const eng = H.loadEngine(opt('--root') ? { root: path.resolve(opt('--root')) } : {});
  const results = [];
  const t0 = Date.now();

  for (const f of MOZZ_FILES) {
    if (only && !only.has(f.name)) continue;
    const file = path.join(MOZZ, f.file);
    if (!fs.existsSync(file)) { console.warn(`skip ${f.name}: ${file} missing (run tools/real/fetch.sh)`); continue; }
    const { x, sr } = readWav(file);
    const y = resample(x, sr, SR);
    const exclude = f.noExclude ? [] : readSegments(path.join(MOZZ, 'speech_labels', f.file.replace(/\.wav$/, '.txt')));
    const frames = analyse(y, { level, lat: f.loc ? f.loc.lat : null, lng: f.loc ? f.loc.lng : null, exclude, eng });
    const s = { ...summarise(f.name, frames, f.truth), kind: f.kind, sourceRate: sr };
    // Same audio with NO location prior: how much does GPS carry identification?
    if (f.truth && f.truth.species) {
      const s0 = summarise(f.name, analyse(y, { level, exclude, eng }), f.truth);
      s.noLocation = { genusAccuracy: s0.genusAccuracy, speciesAccuracy: s0.speciesAccuracy, speciesLevelEvents: s0.speciesLevelEvents };
    }
    results.push(s);
    console.log(JSON.stringify(s));
  }

  for (const c of escFiles()) {
    if (only && !only.has(c.name) && !only.has('esc50')) continue;
    // ESC-50 clips are independent 5-s recordings: run each on its own, after
    // 2 s of its own quietest-start calibration (prepend 2 s of the clip's
    // first 0.5 s, looped — the app always calibrates on the room first).
    let evs = 0, minutes = 0, locked = 0, frames = 0;
    const genus = {}, f0 = [];
    for (const file of c.files) {
      const { x, sr } = readWav(file);
      const y0 = resample(x, sr, SR);
      const lead = y0.subarray(0, Math.round(0.5 * SR)), pre = new Float32Array(Math.round(2.2 * SR));
      for (let i = 0; i < pre.length; i++) pre[i] = lead[i % lead.length] * 0.25;
      const y = new Float32Array(pre.length + y0.length); y.set(pre); y.set(y0, pre.length);
      const fr = analyse(y, { level, eng });
      const s = summarise(c.name, fr, null);
      evs += s.events; minutes += s.minutes; frames += fr.length;
      locked += fr.filter(q => !q.cal && q.dets.length).length;
      for (const [g, n] of Object.entries(s.genus)) genus[g] = (genus[g] || 0) + n;
      for (const q of fr) for (const d of q.dets) f0.push(d.f0);
    }
    const s = { name: c.name, kind: c.kind, clips: c.files.length, minutes: +minutes.toFixed(1), events: evs,
                eventsPerHour: +(evs / (minutes / 60 || 1)).toFixed(1), clipsWithEvent: null, genus,
                f0: f0.length ? { median: +median(f0).toFixed(1), n: f0.length } : null };
    results.push(s);
    if (evs || c.kind === 'insects') console.log(JSON.stringify(s));
  }

  const out = opt('--json');
  if (out) fs.writeFileSync(out, JSON.stringify({ level, generated: new Date().toISOString(), seconds: (Date.now() - t0) / 1000, results }, null, 1));
  // Roll-up
  const bg = results.filter(r => r.kind === 'background');
  const bgMin = bg.reduce((s, r) => s + r.minutes, 0), bgEv = bg.reduce((s, r) => s + r.events, 0);
  console.log(`\nREAL BACKGROUND: ${bgEv} detection events in ${(bgMin / 60).toFixed(2)} h → ${(bgEv / (bgMin / 60 || 1)).toFixed(2)} false alarms/hour`);
  for (const r of results.filter(r => r.kind === 'mosquito')) {
    console.log(`MOSQUITO ${r.name.padEnd(24)} locked ${(r.lockedFraction * 100).toFixed(0)}% · ${r.events} events · genus ok ${r.genusAccuracy ?? '–'} · species ok ${r.speciesAccuracy ?? '–'} (n=${r.speciesLevelEvents ?? '–'}) · f0 median ${r.f0 ? r.f0.median : '–'} Hz [p10 ${r.f0 ? r.f0.p10 : '–'}, p90 ${r.f0 ? r.f0.p90 : '–'}]`);
  }
}

if (require.main === module) main();
module.exports = { readWav, resample, analyse, events, summarise };
