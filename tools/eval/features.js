#!/usr/bin/env node
/*
 * Developer tool (dev split only): per-test rejection rates of the
 * plausibility gates on TRUE mosquito tracks vs. adversary tracks, plus
 * feature quantiles. Used to set MN_PL thresholds from data.
 *
 *   node tools/eval/features.js [--trials N] [--minutes M] [--classes a,b] [--snrs -20,-14,-8,-2]
 *
 * Mosquito frames: frames inside the flight bout, track age ≥ MIN_TRACK, track
 * f0 within 4 % of the individual's true f0. Adversary frames: any track of
 * age ≥ MIN_TRACK whose statistic T exceeds the CFAR threshold.
 */
'use strict';
const os = require('os');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const { rng, seedFrom } = require('../lib/dsp');

const TESTS = ['gaps', 'pitch', 'sub', 'harmonic-comb', 'h2>h1', 'vibrato', 'pure', 'machine:', 'machine-decay', 'machine-level', 'steady', 'range'];
const FEATS = ['pres', 'dfRate', 'subFrac', 'combFrac', 'dom', 'spread', 'hRatio', 'pitchEx', 'levelEx', 'zf', 'zl', 'ubf', 'ubl', 'slope', 'snr1'];

function runJob(job) {
  const S = require('./synth'), C = require('./corpus'), J = require('./jobs');
  const { makeClassifier, drive } = require('./runner');
  const eng = require('../lib/load-engine')();
  const sr = 44100, N = 8192;
  const clf = makeClassifier(eng, { sr, N, level: 3 });
  const rows = [];
  let scene, f0True = null, t1 = 0, t2 = Infinity, seconds;
  if (job.kind === 'mosq') {
    const j = { sp: job.sp, snr: job.snr, trial: job.trial, seed: seedFrom('dev', 'detect', job.sp, job.trial), onset: 3, bout: 10, tail: 1, sr, N };
    const ms = (() => { const r = J._sceneFull(j); f0True = r.f0; return r.scene; })();
    scene = ms; t1 = 3 + 0.3; t2 = 13; seconds = 14;
  } else {
    const seed = seedFrom('dev', 'feat', job.cls, job.run);
    scene = new S.Scene(sr, [new C.RoomNoise(sr, rng(seedFrom(seed, 'room'))), C.CLASSES[job.cls].make(sr, rng(seed))]);
    seconds = job.minutes * 60;
  }
  drive(clf, scene, { sr, N, seconds, onFrame: (i, t, dets, c) => {
    if (t < t1 || t > t2) return;
    for (const tr of c.ff.tracks) {
      if (tr.age < c.MIN_TRACK || !tr.feat) continue;
      const f0 = c._trackF0(tr);
      if (job.kind === 'mosq') { if (Math.abs(f0 / f0True - 1) > 0.04) continue; }
      else if (tr.T < c.ff.TAU) continue;
      const fails = (tr.fails || []).map(s => TESTS.find(x => s.startsWith(x)) || s.split(':')[0]);
      if (tr.reason && tr.reason.startsWith('range')) fails.push('range');
      const fe = {}; for (const k of FEATS) if (tr.feat[k] !== undefined) fe[k] = tr.feat[k];
      rows.push({ fails, fe, locked: tr.locked ? 1 : 0 });
    }
  } });
  return { job, rows };
}

if (!isMainThread) {
  parentPort.on('message', (job) => parentPort.postMessage(runJob(job)));
} else {
  const a = process.argv.slice(2), get = (k, d) => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : d; };
  const trials = +get('--trials', 4), minutes = +get('--minutes', 3);
  const snrs = get('--snrs', '-20,-14,-8,-2').split(',').map(Number);
  const C = require('./corpus');
  const classes = get('--classes', Object.keys(C.CLASSES).filter(k => k !== 'mosquito_male').join(',')).split(',');
  const { SPECIES_DB } = require('../../js/species.js');
  const jobs = [];
  for (const sp of SPECIES_DB) for (let t = 0; t < trials; t++) for (const snr of snrs) jobs.push({ kind: 'mosq', sp: sp.id, trial: t, snr });
  for (const cls of classes) jobs.push({ kind: 'adv', cls, run: 0, minutes });
  const results = [];
  let next = 0;
  const nW = Math.max(1, +get('--workers', Math.min(4, os.cpus().length)));
  const done = new Promise((resolve) => {
    let fin = 0;
    for (let w = 0; w < nW; w++) {
      const wk = new Worker(__filename);
      const feed = () => { if (next < jobs.length) wk.postMessage(jobs[next++]); else { wk.terminate(); if (++fin === nW) resolve(); } };
      wk.on('message', (r) => { results.push(r); feed(); });
      feed();
    }
  });
  done.then(() => {
    const groups = { mosquito: [] };
    for (const r of results) { const g = r.job.kind === 'mosq' ? `mosquito@${r.job.snr}` : r.job.cls; (groups[g] = groups[g] || []).push(...r.rows); if (r.job.kind === 'mosq') groups.mosquito.push(...r.rows); }
    const pct = (x) => (100 * x).toFixed(1).padStart(5);
    console.log('per-test FAIL rate (% of eligible track-frames); "all-ok" = passes every test\n');
    console.log('group'.padEnd(20) + '  frames ' + TESTS.map(t => t.replace(':', '').slice(0, 9).padStart(10)).join('') + '    all-ok');
    const order = Object.keys(groups).sort((x, y) => (x.startsWith('mosq') ? 0 : 1) - (y.startsWith('mosq') ? 0 : 1) || x.localeCompare(y));
    for (const g of order) {
      const rows = groups[g]; if (!rows.length) { console.log(g.padEnd(20) + '       0'); continue; }
      const line = TESTS.map(t => pct(rows.filter(r => r.fails.includes(t)).length / rows.length).padStart(10)).join('');
      console.log(g.padEnd(20) + String(rows.length).padStart(8) + line + pct(rows.filter(r => !r.fails.length).length / rows.length).padStart(10));
    }
    const dump = get('--dump', null);
    if (dump) for (const cls of dump.split(',')) {
      const ok = (groups[cls] || []).filter(r => !r.fails.length);
      console.log(`\n${cls}: ${ok.length} all-ok frames; sample:`);
      for (let i = 0; i < ok.length; i += Math.max(1, Math.floor(ok.length / 12))) console.log('  ' + Object.entries(ok[i].fe).map(([k, v]) => `${k}=${v.toPrecision(3)}`).join(' '));
    }
    console.log('\nfeature quantiles (p01 / p10 / p50 / p90 / p99) on mosquito frames:');
    const m = groups.mosquito;
    for (const k of FEATS) {
      const v = m.map(r => r.fe[k]).filter(x => x !== undefined && Number.isFinite(x)).sort((x, y) => x - y);
      if (!v.length) continue;
      const q = (p) => v[Math.min(v.length - 1, Math.floor(p * v.length))];
      console.log(`  ${k.padEnd(9)} n=${String(v.length).padStart(6)}  ${[0.01, 0.1, 0.5, 0.9, 0.99].map(p => q(p).toPrecision(3).padStart(9)).join(' ')}`);
    }
  });
}
