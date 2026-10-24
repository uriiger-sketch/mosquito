#!/usr/bin/env node
'use strict';
/*
 * Held-out evaluation against HUMAN labels: the HumBug coarse-labelled Culex
 * recordings in HumBug-Mosquito/MFCC_features ("mfcc_weak - Culex": 57 phone
 * recordings, 8 kHz, 49.9 min; 5,824 one-second mosquito-presence labels from
 * the HumBug Zooniverse crowd-labelling, Kiskin et al. 2020). Verified
 * independent of the MozzBNN CDC Culex recording used for tuning (max
 * normalised cross-correlation 0.24). NEVER used for any tuning choice.
 *
 *   node tools/real/labelled.js [--root <engine checkout>] [--levels 1,2,3,4,5] [--json out.json]
 *
 * Per file, the engine runs exactly as in the app (2 s calibration). Scores:
 *   episodes  labelled seconds merged across gaps ≤ 2 s (= one insect pass)
 *   recall    episodes with a detection frame within ±1 s
 *   unlabelled detection events  app-style events (tools/real/realdata.js
 *             events()) with no labelled second within ±2 s — an UPPER bound
 *             on false alarms (crowd labels miss faint mosquitoes)
 *   genus     genus of events that overlap a labelled episode (truth: Culex)
 */
const fs = require('fs');
const path = require('path');
const H = require('../../tests/helpers');
const R = require('./realdata');

const DIR = process.env.HUMBUG_CULEX_DIR ||
  '/home/user/datasets/HumBug-Mosquito_MFCC_features/mfcc_weak - Culex/inputs';
const SR = 44100;

function labels() {
  const by = new Map();
  for (const line of fs.readFileSync(path.join(DIR, 'label.csv'), 'utf8').trim().split('\n')) {
    const [, id, a, b] = line.split(',').map(Number);
    if (!Number.isFinite(id)) continue;
    if (!by.has(id)) by.set(id, []);
    by.get(id).push([a, b]);
  }
  return by;
}

function episodes(segs) {
  const s = [...segs].sort((p, q) => p[0] - q[0]), out = [];
  for (const [a, b] of s) {
    const last = out[out.length - 1];
    if (last && a - last[1] <= 2) last[1] = Math.max(last[1], b); else out.push([a, b]);
  }
  return out;
}

function main() {
  const a = process.argv.slice(2), opt = (k, d) => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : d; };
  const root = opt('--root', null);
  const eng = H.loadEngine(root ? { root: path.resolve(root) } : {});
  const levels = opt('--levels', '1,2,3,4,5').split(',').map(Number);
  const lab = labels();
  const files = fs.readdirSync(path.join(DIR, 'audio')).filter(f => f.endsWith('.wav')).sort();
  const audio = files.map(f => {
    const { x, sr } = R.readWav(path.join(DIR, 'audio', f));
    return { id: +f.replace('.wav', ''), y: R.resample(x, sr, SR), seconds: x.length / sr };
  });
  const out = [];
  for (const level of levels) {
    let eps = 0, hit = 0, evN = 0, evUnl = 0, minutes = 0, genusOk = 0, genusN = 0;
    let inFrames = 0, culexCand = 0, maleSum = 0;
    const genus = {};
    for (const f of audio) {
      const ep = episodes(lab.get(f.id) || []);
      const frames = R.analyse(f.y, { level, eng });
      minutes += f.seconds / 60;
      const det = frames.filter(q => !q.cal && q.dets.length).map(q => q.t);
      for (const [s, e] of ep) { eps++; if (det.some(t => t >= s - 1 && t <= e + 1)) hit++; }
      // Identification inside labelled episodes, frame by frame.
      for (const q of frames) {
        if (q.cal || !q.dets.length || !ep.some(([s, e]) => q.t >= s - 1 && q.t <= e + 1)) continue;
        const d = q.dets[0]; inFrames++;
        if ((d.genusP && d.genusP.Culex || 0) >= 0.2) culexCand++;
        maleSum += Number.isFinite(d.maleP) ? d.maleP : 0;
      }
      for (const ev of R.events(frames)) {
        evN++;
        const ov = ep.some(([s, e]) => ev.last >= s - 2 && ev.t <= e + 2);
        if (!ov) evUnl++;
        else { genusN++; if (ev.genus === 'Culex') genusOk++; genus[ev.genus] = (genus[ev.genus] || 0) + 1; }
      }
    }
    const r = { level, files: audio.length, minutes: +minutes.toFixed(1), episodes: eps, detectedEpisodes: hit,
                recall: +(hit / (eps || 1)).toFixed(3), events: evN, unlabelledEvents: evUnl,
                unlabelledPerHour: +(evUnl / (minutes / 60)).toFixed(1),
                genusAccuracy: genusN ? +(genusOk / genusN).toFixed(3) : null, genusOfLabelledEvents: genus,
                culexCandidateFrames: inFrames ? +(culexCand / inFrames).toFixed(3) : null,
                meanMaleP: inFrames ? +(maleSum / inFrames).toFixed(3) : null };
    out.push(r);
    console.log(JSON.stringify(r));
  }
  const j = opt('--json', null);
  if (j) fs.writeFileSync(j, JSON.stringify({ root: root || '.', generated: new Date().toISOString(), results: out }, null, 1));
}

if (require.main === module) main();
