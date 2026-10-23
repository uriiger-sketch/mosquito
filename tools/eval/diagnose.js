#!/usr/bin/env node
/*
 * Developer tool: which tracks lock, and what do their plausibility features
 * look like? Runs adversary classes and mosquito trials (dev split) through
 * the CURRENT engine and prints, per scenario, the feature vector of every
 * track at the moment it first locks (and optionally per-frame dumps).
 *
 *   node tools/eval/diagnose.js [--classes a,b,c] [--minutes M] [--mosq N] [--snr S] [--frames]
 */
'use strict';
const { rng, seedFrom } = require('../lib/dsp');
const S = require('./synth');
const C = require('./corpus');
const J = require('./jobs');
const { makeClassifier, drive } = require('./runner');
const loadEngine = require('../lib/load-engine');

const a = process.argv.slice(2), get = (k, d) => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : d; };
const classes = get('--classes', 'music_pop,music_piano,traffic,fan_desk,hum_ac,humming,insect_bumblebee').split(',').filter(Boolean);
const minutes = +get('--minutes', 2), nMosq = +get('--mosq', 0), snr = +get('--snr', -14);
const eng = loadEngine();
const sr = 44100, N = 8192;
const fmt = (o) => Object.entries(o || {}).map(([k, v]) => `${k}=${typeof v === 'number' ? (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3)) : v}`).join(' ');

function watch(clf, label) {
  const seen = new Set();
  return (i, t, dets, c) => {
    for (const tr of c.ff.tracks) {
      if (tr.locked && !seen.has(tr.id)) {
        seen.add(tr.id);
        const f0 = c._trackF0(tr);
        console.log(`${label.padEnd(18)} t=${t.toFixed(1).padStart(6)} track ${tr.id} f0=${f0.toFixed(1)} T=${tr.T.toFixed(0)} age=${tr.age} ${fmt(tr.feat)}`);
      }
    }
    if (a.includes('--frames') && c.lastEvidence) console.log(`   ${t.toFixed(2)} ${c.lastEvidence.reason} vf=${c.lastEvidence.voteFraction.toFixed(2)} ${fmt(c.lastEvidence.feat)}`);
  };
}

for (const cls of classes) {
  const clf = makeClassifier(eng, { sr, N, level: 3 });
  const seed = seedFrom('dev', 'fa', cls, 0);
  const scene = new S.Scene(sr, [new C.RoomNoise(sr, rng(seedFrom(seed, 'room'))), C.CLASSES[cls].make(sr, rng(seed))]);
  drive(clf, scene, { sr, N, seconds: minutes * 60, onFrame: watch(clf, cls) });
}
for (let t = 0; t < nMosq; t++) {
  const sp = eng.SPECIES_DB[t % 12].id;
  const job = { sp, snr, trial: t, seed: seedFrom('dev', 'detect', sp, t), onset: 3, bout: 10, tail: 1, sr, N };
  const clf = makeClassifier(eng, { sr, N, level: 3 });
  const { scene } = (() => { const r = require('./jobs'); return { scene: r._scene ? r._scene(job) : null }; })();
  if (!scene) { console.log('mosquito scenes need jobs._scene'); break; }
  drive(clf, scene, { sr, N, seconds: 14, onFrame: watch(clf, 'mosq:' + sp) });
}
