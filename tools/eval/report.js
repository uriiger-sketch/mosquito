#!/usr/bin/env node
/*
 * Render baseline-vs-final tables from two eval JSON reports into
 * docs/EVALUATION.md, between the markers
 *   <!-- GENERATED:BEGIN -->  …  <!-- GENERATED:END -->
 * so the numbers in the document are never transcribed by hand.
 *
 *   node tools/eval/report.js [--baseline tools/eval/results/baseline-full-test.json]
 *                             [--final tools/eval/results/final-full-test.json] [--doc docs/EVALUATION.md]
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { SPECIES_DB } = require('../../js/species.js');
const C = require('./corpus');

const a = process.argv.slice(2), get = (k, d) => { const i = a.indexOf(k); return i >= 0 ? a[i + 1] : d; };
const root = path.join(__dirname, '..', '..');
const B = JSON.parse(fs.readFileSync(get('--baseline', path.join(__dirname, 'results', 'baseline-full-test.json')), 'utf8'));
const F = JSON.parse(fs.readFileSync(get('--final', path.join(__dirname, 'results', 'final-full-test.json')), 'utf8'));
const docPath = get('--doc', path.join(root, 'docs', 'EVALUATION.md'));

const n = (x, d = 1) => (x === null || x === undefined || !Number.isFinite(+x)) ? '–' : (+x).toFixed(d);
const pct = (x) => (x === null || x === undefined) ? '–' : (100 * x).toFixed(0) + ' %';
const ci = (v, d = 2) => `${n(v.perHour, d)} [${n(v.ci[0], d)}, ${n(v.ci[1], d)}]`;
const L = [];
const row = (...c) => L.push('| ' + c.join(' | ') + ' |');
const head = (...c) => { row(...c); L.push('|' + c.map(() => '---').join('|') + '|'); };

// ── headline ────────────────────────────────────────────────────────────────
const idb = (M, loc, k) => M.identification?.[loc]?.first?.[k];
L.push('### Headline (test split, sensitivity 3)', '');
head('Metric', 'Baseline (e46de8d)', 'Final (this branch)');
row('Mean SNR50 over 12 species (dB, lower = better)', n(B.detection.meanSNR50), n(F.detection.meanSNR50));
row('Worst-species SNR50 (dB)', n(B.detection.worstSNR50), n(F.detection.worstSNR50));
row('Mean range factor r50/r0 (= 10^(−SNR50/20))', n(B.detection.meanRangeFactor, 2), n(F.detection.meanRangeFactor, 2));
row('P(detect) for loud mosquitoes (SNR ≥ −8 dB)', n(B.detection.pDetectLoud, 3), n(F.detection.pDetectLoud, 3));
row('Median lock latency (SNR ≥ −12 dB, s)', n(B.detection.latencyMedianAtSNRge12, 2), n(F.detection.latencyMedianAtSNRge12, 2));
row('Trials with ≥2 species locked at once', B.detection.multiSpeciesLockTrials, F.detection.multiSpeciesLockTrials);
row('False alarms / hour, pooled over 33 adversary classes [95 % CI]', ci(B.falseAlarms.pooled), ci(F.falseAlarms.pooled));
row('Adversary classes with any false alarm (of 33)', B.falseAlarms.classesWithAnyEvent.length, F.falseAlarms.classesWithAnyEvent.length);
for (const loc of ['none', 'accra', 'rome', 'mumbai']) {
  row(`Species top-1 / genus top-1 — ${loc === 'none' ? 'no location' : loc}`,
    `${pct(idb(B, loc, 'speciesTop1'))} / ${pct(idb(B, loc, 'genusTop1'))}`, `${pct(idb(F, loc, 'speciesTop1'))} / ${pct(idb(F, loc, 'genusTop1'))}`);
}
row('Species posterior Brier / ECE (no location)', `${n(idb(B, 'none', 'speciesBrier'), 3)} / ${n(idb(B, 'none', 'speciesECE'), 3)}`, `${n(idb(F, 'none', 'speciesBrier'), 3)} / ${n(idb(F, 'none', 'speciesECE'), 3)}`);
row('Genus posterior ECE (no location)', n(idb(B, 'none', 'genusECE'), 3), n(idb(F, 'none', 'genusECE'), 3));
row('Detection.confidence vs real-mosquito: Brier / ECE', `${n(B.confidence.brier, 3)} / ${n(B.confidence.ece, 3)}`, `${n(F.confidence.brier, 3)} / ${n(F.confidence.ece, 3)}`);
row('Mean confidence on real / false locks', `${n(B.confidence.meanConfPositive, 2)} / ${n(B.confidence.meanConfNegative, 2)}`, `${n(F.confidence.meanConfPositive, 2)} / ${n(F.confidence.meanConfNegative, 2)}`);
row('analyze() mean / p95 (ms, Node, single thread)', `${n(B.compute.meanMs, 2)} / ${n(B.compute.p95Ms, 2)}`, `${n(F.compute.meanMs, 2)} / ${n(F.compute.p95Ms, 2)}`);
L.push('');

// ── detection per species ───────────────────────────────────────────────────
L.push('### Detection: SNR50 per species', '',
  'SNR50 from isotonic regression of P(detect) on SNR (12 individuals × 11 SNRs per species); `<`/`>` = censored at the grid edge; `~` = non-monotone curve (louder trials missed more often than quieter ones). P(loud) = P(detect) pooled over SNR ≥ −8 dB.', '');
head('Species', 'f0 (Hz)', 'Baseline SNR50', 'Final SNR50', 'Δ (dB)', 'Range ×', 'Baseline P(loud)', 'Final P(loud)');
const loudP = (M, id) => { const c = M.detection.perSpecies[id].curve.filter(x => x.snr >= -8); const k = c.reduce((s, x) => s + x.k, 0), m = c.reduce((s, x) => s + x.n, 0); return m ? k / m : null; };
for (const sp of SPECIES_DB) {
  const b = B.detection.perSpecies[sp.id], f = F.detection.perSpecies[sp.id];
  const d = f.snr50 - b.snr50;
  row(sp.name, sp.freq, `${b.bound}${n(b.snr50)}${b.monotone ? '' : ' ~'}`, `${f.bound}${n(f.snr50)}${f.monotone ? '' : ' ~'}`, n(d), n(Math.pow(10, -d / 20), 2), n(loudP(B, sp.id), 2), n(loudP(F, sp.id), 2));
}
L.push('');

// ── false alarms ────────────────────────────────────────────────────────────
L.push('### False alarms per adversary class (sensitivity 3)', '',
  `Each class: ${n(F.falseAlarms.perClass.quiet_room.hours * 60, 0)} simulated minutes (4 independent 5-min runs, each starting with the 2 s calibration). Events counted with the app's logic (a species' lock is a new event if its previous event was ≥ 60 s ago). 95 % CI: exact Poisson. Target ≤ 0.2 /h. * = not counted as a false alarm (male mosquitoes are mosquitoes).`, '');
head('Class', 'Group', 'Baseline events', 'Baseline FA/h [95 % CI]', 'Final events', 'Final FA/h [95 % CI]');
for (const [cls, v] of Object.entries(F.falseAlarms.perClass)) {
  const b = B.falseAlarms.perClass[cls];
  row(cls + (v.countsAsFA ? '' : ' *'), v.group, b ? b.events : '–', b ? ci(b) : '–', v.events, ci(v));
}
L.push('');

// ── mixtures ────────────────────────────────────────────────────────────────
L.push('### Mixtures: mosquito (SNR −12 dB) with an adversary already playing', '',
  'Same 20 individuals per row (paired). Detected = a Detection whose f0 is within 5 % of the individual\'s true f0 inside the 10 s flight bout (onset at 15 s). Other-lock frames = frames in the bout with a lock on something else.', '');
head('Adversary', 'Baseline P(detect) [CI]', 'Final P(detect) [CI]', 'Baseline other-lock frames', 'Final other-lock frames');
for (const [cls, v] of Object.entries(F.mixtures)) {
  const b = B.mixtures[cls];
  row(cls, `${b.k}/${b.n} = ${n(b.p, 2)} [${n(b.ci[0], 2)}, ${n(b.ci[1], 2)}]`, `${v.k}/${v.n} = ${n(v.p, 2)} [${n(v.ci[0], 2)}, ${n(v.ci[1], 2)}]`, b.otherLockFrames ?? '–', v.otherLockFrames ?? '–');
}
L.push('');

// ── identification ──────────────────────────────────────────────────────────
L.push('### Identification (first lock; trials drawn from the location\'s native species; SNR uniform in [−15, −5] dB)', '');
head('Location', 'Engine', 'Trials', 'Detected', 'Species top-1', 'Species top-3', 'Genus top-1', 'Species Brier', 'Species ECE', 'Genus ECE', 'Non-native: detected / genus ok');
for (const loc of ['none', 'accra', 'rome', 'mumbai']) for (const [name, M] of [['baseline', B], ['final', F]]) {
  const v = M.identification[loc]; if (!v) continue; const b = v.first, nn = v.nonNative;
  row(loc, name, b.n, b.detected, pct(b.speciesTop1), pct(b.speciesTop3), pct(b.genusTop1), n(b.speciesBrier, 3), n(b.speciesECE, 3), n(b.genusECE, 3), nn ? `${nn.detected}/${nn.n} / ${nn.genusCorrect}` : '–');
}
L.push('');
L.push('Final-engine genus confusion (no location prior; rows = true genus, columns = reported):', '');
const gc = F.identification.none.first.genusConfusion, cols = [...new Set(Object.values(gc).flatMap(o => Object.keys(o)))];
head('true \\ reported', ...cols);
for (const [g, o] of Object.entries(gc)) row(g, ...cols.map(c => o[c] || 0));
L.push('');

// ── calibration ─────────────────────────────────────────────────────────────
const rel = (title, Bb, Fb) => {
  L.push(title, '');
  head('bin', 'baseline n', 'baseline mean p', 'baseline freq', 'final n', 'final mean p', 'final freq');
  for (let i = 0; i < Fb.length; i++) {
    const b = Bb[i], f = Fb[i];
    if (!b.n && !f.n) continue;
    row(`[${b.lo.toFixed(1)}, ${b.hi.toFixed(1)})`, b.n, n(b.meanP, 2), n(b.freq, 2), f.n, n(f.meanP, 2), n(f.freq, 2));
  }
  L.push('');
};
rel('### Reliability: Detection.confidence vs "real mosquito" (events: first locks of mosquito trials + adversary false-alarm events)', B.confidence.reliability, F.confidence.reliability);
rel('### Reliability: top-1 species probability vs species correct (no location)', B.identification.none.first.speciesReliability, F.identification.none.first.speciesReliability);

// ── sensitivity levels ──────────────────────────────────────────────────────
if (F.levels) {
  L.push('### Sensitivity levels (operating points)', '',
    'Detection on 4 individuals × 5 SNRs per species; false alarms pooled over the 14 hardest in-band adversary classes (5 min each).', '');
  head('Level', 'Baseline mean SNR50', 'Baseline FA/h [CI]', 'Final mean SNR50', 'Final FA/h [CI]', 'Final P(loud)', 'Final latency (s)');
  for (const lv of Object.keys(F.levels)) {
    const b = B.levels?.[lv], f = F.levels[lv];
    row(lv, b ? n(b.meanSNR50) : '–', b ? ci(b.faPooled) : '–', n(f.meanSNR50), ci(f.faPooled), n(f.pDetectLoud, 3), n(f.latencyMedianAtSNRge12, 2));
  }
  L.push('');
}
L.push(`_Generated by tools/eval/report.js from ${path.relative(root, get('--baseline', 'tools/eval/results/baseline-full-test.json'))} (${B.meta.commit}) and ${path.relative(root, get('--final', 'tools/eval/results/final-full-test.json'))} (${F.meta.commit})._`);

const doc = fs.existsSync(docPath) ? fs.readFileSync(docPath, 'utf8') : '<!-- GENERATED:BEGIN -->\n<!-- GENERATED:END -->\n';
const out = doc.replace(/<!-- GENERATED:BEGIN -->[\s\S]*<!-- GENERATED:END -->/, `<!-- GENERATED:BEGIN -->\n${L.join('\n')}\n<!-- GENERATED:END -->`);
fs.mkdirSync(path.dirname(docPath), { recursive: true });
fs.writeFileSync(docPath, out);
console.log(`wrote ${path.relative(process.cwd(), docPath)} (${L.length} generated lines)`);
