#!/usr/bin/env node
/*
 * MosquitoNet detection/identification evaluation harness.
 *
 *   node tools/eval/eval.js [--quick | --full] [--root <checkout>] [--split test|dev]
 *                           [--only detect,id,fa,mix,levels] [--workers N] [--minutes M]
 *                           [--out <file.json>] [--raw <file.json>] [--check] [--no-compute]
 *
 *   --quick     CI mode (< ~90 s on 4 cores): all 12 species at 3 SNRs, 14 hard
 *               adversary classes × 1 min, 4 mixtures; with --check, fails (exit 1)
 *               if any metric regresses past tools/eval/thresholds.json.
 *   --full      the reported evaluation (default): see jobs.js MODES.full.
 *   --root      evaluate another checkout (e.g. a baseline `git worktree`).
 *   --split     'test' (default, reported) or 'dev' (the seeds used for tuning).
 *
 * Writes a machine-readable report (default tools/eval/results/<mode>-<split>.json)
 * and prints a summary. Everything is seeded: same code ⇒ same numbers.
 */
'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execSync } = require('child_process');
const { Worker } = require('worker_threads');
const J = require('./jobs');
const M = require('./metrics');
const loadEngine = require('../lib/load-engine');

function args() {
  const a = process.argv.slice(2), get = (k, d) => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : d; };
  return {
    mode: a.includes('--quick') ? 'quick' : a.includes('--dev') ? 'dev' : 'full',
    root: path.resolve(get('--root', path.join(__dirname, '..', '..'))),
    split: get('--split', 'test'),
    only: get('--only', null)?.split(','),
    workers: +get('--workers', Math.max(1, Math.min(os.cpus().length, 8))),
    minutes: get('--minutes', null) ? +get('--minutes') : null,
    out: get('--out', null), raw: get('--raw', null),
    check: a.includes('--check'), compute: !a.includes('--no-compute'),
    quiet: a.includes('--quiet'), fromRaw: get('--from-raw', null),
  };
}

function runPool(jobs, root, nWorkers, log) {
  return new Promise((resolve, reject) => {
    const order = jobs.map((j, i) => i).sort((x, y) => (jobs[y].cost || 1) - (jobs[x].cost || 1));
    const results = new Array(jobs.length);
    let next = 0, done = 0, failed = null;
    const total = jobs.length, t0 = Date.now();
    let lastLog = 0;
    const workers = Array.from({ length: Math.min(nWorkers, jobs.length) }, () => new Worker(path.join(__dirname, 'worker.js'), { workerData: { root } }));
    const feed = (w) => {
      if (failed) return;
      if (next >= order.length) { w.terminate(); return; }
      const idx = order[next++];
      w.postMessage({ idx, job: jobs[idx] });
    };
    for (const w of workers) {
      w.on('message', ({ idx, r, err }) => {
        if (err) { failed = new Error(`job ${idx} ${JSON.stringify(jobs[idx])}\n${err}`); workers.forEach(x => x.terminate()); reject(failed); return; }
        results[idx] = r; done++;
        if (log && Date.now() - lastLog > 5000) { lastLog = Date.now(); log(`  ${done}/${total} jobs, ${((Date.now() - t0) / 1000).toFixed(0)} s`); }
        if (done === total) { workers.forEach(x => x.terminate()); resolve(results); } else feed(w);
      });
      w.on('error', (e) => { if (!failed) { failed = e; reject(e); } });
      feed(w);
    }
    if (!total) resolve(results);
  });
}

function gitRev(root) { try { return execSync('git rev-parse --short HEAD', { cwd: root }).toString().trim() + (execSync('git status --porcelain js', { cwd: root }).toString().trim() ? '+dirty' : ''); } catch (e) { return 'unknown'; } }

const f1 = (x, d = 1) => (x === null || x === undefined) ? '  –  ' : (typeof x === 'number' ? x.toFixed(d) : String(x));
function printSummary(m, log) {
  if (m.detection) {
    const d = m.detection;
    log('\nDETECTION  (SNR = fundamental power / broadband room-noise power; SNR50 from logistic fit)');
    log('  species                   f0cat   SNR50 (dB)   r50/r0   P(det) per SNR');
    const { SPECIES_DB } = require('../../js/species.js');
    for (const sp of SPECIES_DB) {
      const s = d.perSpecies[sp.id]; if (!s) continue;
      log(`  ${sp.id.padEnd(24)} ${String(sp.freq).padStart(5)}   ${(s.bound + f1(s.snr50)).padStart(7)} ${s.monotone ? ' ' : '~'}     ${f1(s.rangeFactor).padStart(6)}   ${s.curve.map(c => `${c.snr}:${f1(c.p, 2)}`).join(' ')}`);
    }
    log(`  mean SNR50 ${f1(d.meanSNR50)} dB (worst ${f1(d.worstSNR50)}), mean range factor ${f1(d.meanRangeFactor, 2)}, P(det | SNR ≥ −8) ${f1(d.pDetectLoud, 3)}, median latency (SNR ≥ −12) ${f1(d.latencyMedianAtSNRge12, 2)} s, pre-onset false frames ${d.preOnsetFalseFrames}`);
  }
  if (m.identification) {
    log('\nIDENTIFICATION (first lock)   n  det  sp-top1  sp-top3  genus-top1  sp-Brier  sp-ECE  genus-ECE');
    for (const [loc, v] of Object.entries(m.identification)) {
      const b = v.first;
      log(`  ${loc.padEnd(26)} ${String(b.n).padStart(4)} ${String(b.detected).padStart(4)}   ${f1(b.speciesTop1, 3)}    ${f1(b.speciesTop3, 3)}     ${f1(b.genusTop1, 3)}      ${f1(b.speciesBrier, 3)}   ${f1(b.speciesECE, 3)}    ${f1(b.genusECE, 3)}` +
        (v.nonNative ? `   non-native: ${v.nonNative.detected}/${v.nonNative.n} detected, genus ok ${v.nonNative.genusCorrect}` : ''));
    }
  }
  if (m.confidence) {
    const c = m.confidence;
    log(`\nCONFIDENCE vs real-mosquito: n=${c.n} (+${c.positives}/−${c.negatives}) Brier ${f1(c.brier, 3)} ECE ${f1(c.ece, 3)} mean conf(+) ${f1(c.meanConfPositive, 3)} (−) ${f1(c.meanConfNegative, 3)}`);
  }
  if (m.falseAlarms) {
    log('\nFALSE ALARMS (level 3)       hours  events  per-hour [95% CI]          locked%   max-level');
    for (const [cls, v] of Object.entries(m.falseAlarms.perClass)) {
      log(`  ${cls.padEnd(18)}${v.countsAsFA ? ' ' : '*'}  ${f1(v.hours, 2).padStart(6)}  ${String(v.events).padStart(6)}  ${f1(v.perHour, 2).padStart(7)} [${f1(v.ci[0], 2)}, ${f1(v.ci[1], 2)}]`.padEnd(66) +
        `${(100 * v.lockedFrac).toFixed(2).padStart(7)}   ${f1(v.maxLevel, 2)}  ${Object.entries(v.eventSpecies).map(([k, n]) => k + ':' + n).join(' ')}`);
    }
    const p = m.falseAlarms.pooled;
    log(`  pooled (* excluded): ${p.events} events / ${f1(p.hours, 2)} h = ${f1(p.perHour, 2)}/h [${f1(p.ci[0], 2)}, ${f1(p.ci[1], 2)}]`);
  }
  if (m.mixtures) {
    log('\nMIXTURES (mosquito at fixed SNR + adversary; f0-matched)   P(detect) [CI]   retained   genus-ok   latency   other-lock-frames');
    for (const [cls, v] of Object.entries(m.mixtures)) log(`  ${cls.padEnd(18)} ${String(v.k).padStart(3)}/${String(v.n).padEnd(3)} ${f1(v.p, 2)} [${f1(v.ci[0], 2)}, ${f1(v.ci[1], 2)}]   ${f1(v.retainedOfRef, 2)}   ${f1(v.genusCorrect, 2)}   ${f1(v.latencyMedian, 2)}   ${v.otherLockFrames ?? ''}`);
  }
  if (m.levels) {
    log('\nSENSITIVITY LEVELS   mean SNR50   worst   P(det loud)   FA/h pooled hard set [CI]');
    for (const [lv, v] of Object.entries(m.levels)) log(`  level ${lv}            ${f1(v.meanSNR50)}      ${f1(v.worstSNR50)}    ${f1(v.pDetectLoud, 3)}        ${f1(v.faPooled.perHour, 2)} [${f1(v.faPooled.ci[0], 2)}, ${f1(v.faPooled.ci[1], 2)}]`);
  }
  if (m.compute) log(`\nCOMPUTE  analyze(): mean ${m.compute.meanMs} ms, p50 ${m.compute.p50Ms}, p95 ${m.compute.p95Ms}, max ${m.compute.maxMs} (${m.compute.calls} calls, single thread)`);
}

// Regression check against tools/eval/thresholds.json (quick mode).
function check(m, mode, log) {
  const file = path.join(__dirname, 'thresholds.json');
  if (!fs.existsSync(file)) { log('no thresholds.json — skipping check'); return true; }
  const T = JSON.parse(fs.readFileSync(file, 'utf8'))[mode];
  if (!T) { log(`no thresholds for mode ${mode}`); return true; }
  const val = (p) => p.split('.').reduce((o, k) => (o == null ? o : o[k]), m);
  let ok = true;
  log('\nREGRESSION CHECK (' + file + ')');
  for (const [p, lim] of Object.entries(T)) {
    if (p.startsWith('_')) continue;
    const v = val(p);
    const pass = v !== null && v !== undefined && (lim.max !== undefined ? v <= lim.max : true) && (lim.min !== undefined ? v >= lim.min : true);
    log(`  ${pass ? 'ok  ' : 'FAIL'} ${p} = ${v}  (${lim.min !== undefined ? '≥ ' + lim.min : ''}${lim.max !== undefined ? '≤ ' + lim.max : ''})`);
    if (!pass) ok = false;
  }
  return ok;
}

async function main() {
  const A = args();
  const log = A.quiet ? () => {} : (s) => console.log(s);
  const t0 = Date.now();
  const jobs = J.buildJobs(A.mode, A.split, { only: A.only, minutes: A.minutes });
  let results;
  if (A.fromRaw) {                       // re-aggregate a saved run (metric changes need no re-simulation)
    results = JSON.parse(fs.readFileSync(A.fromRaw, 'utf8'));
    log(`re-aggregating ${results.length} raw results from ${A.fromRaw}`);
    A.compute = false;
  } else {
    log(`MosquitoNet eval — mode ${A.mode}, split ${A.split}, engine ${A.root} @ ${gitRev(A.root)}, ${jobs.length} jobs on ${A.workers} workers`);
    results = await runPool(jobs, A.root, A.workers, log);
  }
  const meta = { mode: A.mode, split: A.split, root: A.root, commit: gitRev(A.root), date: new Date().toISOString(),
                 node: process.version, jobs: jobs.length, sampleRate: J.SR, fftSize: J.N };
  const m = M.aggregate(results, meta);
  if (A.fromRaw && A.out && fs.existsSync(A.out)) { const prev = JSON.parse(fs.readFileSync(A.out, 'utf8')); if (prev.compute) m.compute = prev.compute; }
  if (A.compute && (!A.only || A.only.includes('compute'))) {
    const eng = loadEngine({ root: A.root });
    m.compute = J.computeBench(eng, J.MODES[A.mode].compute.seconds);
  }
  meta.wallSeconds = +((Date.now() - t0) / 1000).toFixed(1);
  printSummary(m, log);
  const out = A.out || path.join(__dirname, 'results', `${A.mode}-${A.split}.json`);
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, JSON.stringify(m, null, 1));
  if (A.raw) fs.writeFileSync(A.raw, JSON.stringify(results));
  log(`\nwrote ${path.relative(process.cwd(), out)}  (${meta.wallSeconds} s)`);
  if (A.check && !check(m, A.mode, log)) process.exit(1);
}

main().catch(e => { console.error(e); process.exit(2); });
