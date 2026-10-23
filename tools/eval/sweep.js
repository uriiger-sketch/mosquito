#!/usr/bin/env node
/*
 * Developer tool: sweep engine constants on the DEV split and print the
 * trade-off table (detection vs false alarms vs latency vs mixtures).
 *
 *   node tools/eval/sweep.js '{"MIN_TRACK_S":1.1}' '{"MIN_TRACK_S":1.6}' … [--workers N] [--mode dev]
 *
 * Each argument is a JSON object of MN_DSP / MN_PL overrides ({} = current).
 */
'use strict';
const { execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const args = process.argv.slice(2);
const wi = args.indexOf('--workers'), workers = wi >= 0 ? args.splice(wi, 2)[1] : '4';
const mi = args.indexOf('--mode'), mode = mi >= 0 ? args.splice(mi, 2)[1] : 'dev';
const configs = args.length ? args : ['{}'];
const rows = [];
for (const cfg of configs) {
  const out = path.join(os.tmpdir(), `mn-sweep-${process.pid}-${rows.length}.json`);
  execFileSync(process.execPath, [path.join(__dirname, 'eval.js'), '--' + mode, '--split', 'dev', '--workers', workers, '--no-compute', '--quiet', '--out', out],
    { env: { ...process.env, MN_TUNE: cfg }, stdio: 'inherit' });
  const m = JSON.parse(fs.readFileSync(out, 'utf8'));
  fs.unlinkSync(out);
  const fa = m.falseAlarms, d = m.detection, mx = m.mixtures || {};
  const worst = Object.entries(fa.perClass).filter(([, v]) => v.countsAsFA).sort((a, b) => b[1].events - a[1].events).slice(0, 4).map(([k, v]) => `${k}:${v.events}`).join(' ');
  const mixP = Object.entries(mx).filter(([k]) => k !== 'none').map(([, v]) => v.p);
  rows.push({ cfg, snr50: d.meanSNR50, worst: d.worstSNR50, loud: d.pDetectLoud, lat: d.latencyMedianAtSNRge12,
    fa: fa.pooled.perHour, ev: fa.pooled.events, mix: mixP.length ? mixP.reduce((s, x) => s + x, 0) / mixP.length : null,
    id: m.identification?.none?.first?.genusTop1, worstFA: worst });
  const r = rows[rows.length - 1];
  console.log(`${cfg.padEnd(48)} SNR50 ${r.snr50} (worst ${r.worst})  loud ${r.loud}  lat ${r.lat}s  FA ${r.fa}/h (${r.ev} ev)  mix ${r.mix?.toFixed(2)}  | ${r.worstFA}`);
}
