'use strict';
/*
 * Turn raw job results into the reported metrics. Definitions (also in
 * docs/EVALUATION.md):
 *
 *  SNR       10·log10(P_H1 / P_noise): mean power of the flight tone's
 *            FUNDAMENTAL over the flight bout vs. the TOTAL (0–Nyquist) power
 *            of the reference room noise (legacy bench definition).
 *  detected  ≥ 1 Detection (any species) in a frame ending inside
 *            (onset, onset + bout + 0.5 s].
 *  SNR50     SNR where P(detect) = 0.5, from a per-species logistic MLE over
 *            all trials (± delta-method s.e.).
 *  range     r50 / r0 = 10^(−SNR50/20): distance at 50 % detection relative to
 *            the distance r0 at which the same mosquito gives SNR = 0 dB
 *            (far-field 1/r spreading, −6 dB per doubling; Arthur et al. 2014).
 *  first     the top Detection in the first detected frame (what the app counts
 *            as the event and draws on the map); "last" = last detected frame.
 *  genus     MAP genus: argmax of genusPosterior when the engine provides it,
 *            otherwise the genus of Detection.species.
 *  FA event  app logic: a species' detection is a new event if its previous
 *            event was ≥ 60 s earlier (a continuous lock re-fires each minute).
 */
const St = require('./stats');
const { SPECIES_DB } = require('../../js/species.js');
const C = require('./corpus');

const GENUS = Object.fromEntries(SPECIES_DB.map(s => [s.id, s.genus || s.name.split(' ')[0]]));
const IDS = SPECIES_DB.map(s => s.id);
const GENERA = [...new Set(IDS.map(id => GENUS[id]))];

const r3 = (x) => (x === null || x === undefined || !Number.isFinite(x)) ? x : +x.toFixed(3);

function genusDist(sum) {
  if (!sum) return {};
  if (sum.gpost && sum.gpost.length) return Object.fromEntries(sum.gpost);
  const g = {};
  for (const [id, p] of sum.post || []) g[GENUS[id]] = (g[GENUS[id]] || 0) + p;
  return g;
}
const argmax = (d) => { let k = null, v = -1; for (const [a, b] of Object.entries(d)) if (b > v) { v = b; k = a; } return k; };
const mapGenus = (sum) => { const g = genusDist(sum); return Object.keys(g).length ? argmax(g) : (sum ? sum.genus : null); };

// ── detection ────────────────────────────────────────────────────────────────
function detection(trials) {
  const bySp = {};
  for (const r of trials) (bySp[r.sp] = bySp[r.sp] || []).push(r);
  const perSpecies = {};
  const snr50s = [];
  for (const id of IDS) {
    const rs = bySp[id] || [];
    if (!rs.length) continue;
    const snrs = [...new Set(rs.map(r => r.snr))].sort((a, b) => a - b);
    const curve = snrs.map(s => {
      const g = rs.filter(r => r.snr === s), k = g.filter(r => r.detected).length;
      const lat = g.filter(r => r.detected).map(r => r.latency);
      const gen = g.filter(r => r.detected);
      return { snr: s, n: g.length, k, p: r3(k / g.length), ci: St.wilson(k, g.length).map(r3),
               latencyMedian: r3(St.quantile(lat, 0.5)), lockedFrac: r3(St.mean(g.map(r => r.lockedFrac))),
               genusCorrect: gen.length ? r3(gen.filter(r => mapGenus(r.first) === GENUS[id]).length / gen.length) : null };
    });
    // Primary SNR50: isotonic (monotone non-decreasing) regression of P(detect)
    // on SNR, then linear interpolation of the 0.5 crossing — robust to sparse
    // grids and to non-monotone curves (loud mosquitoes missed), which are
    // reported separately via pDetectLoud. Censored at the grid edges ('<', '>').
    const iso = St.isotonic(curve.map(c => c.p), curve.map(c => c.n));
    let snr50 = null, note = '';
    if (iso[0] >= 0.5) { snr50 = snrs[0]; note = '<'; }
    else if (iso[iso.length - 1] < 0.5) { snr50 = snrs[snrs.length - 1]; note = '>'; }
    else for (let i = 1; i < iso.length; i++) if (iso[i] >= 0.5) {
      snr50 = iso[i] === iso[i - 1] ? snrs[i] : snrs[i - 1] + (0.5 - iso[i - 1]) * (snrs[i] - snrs[i - 1]) / (iso[i] - iso[i - 1]); break;
    }
    const fit = St.logisticFit(rs.map(r => r.snr), rs.map(r => r.detected ? 1 : 0));
    const logit = fit && !fit.all && !fit.none && fit.slope > 0 && fit.x50 > snrs[0] - 4 && fit.x50 < snrs[snrs.length - 1] + 4 ? fit : null;
    perSpecies[id] = { snr50: r3(snr50), bound: note, logisticSNR50: logit ? r3(logit.x50) : null, logisticSE: logit ? r3(logit.se) : null,
                       rangeFactor: snr50 !== null ? r3(Math.pow(10, -snr50 / 20)) : null, monotone: curve.every((c, i) => !i || c.p >= curve[i - 1].p - 1e-9), curve };
    if (snr50 !== null) snr50s.push(snr50);
  }
  const loud = trials.filter(r => r.snr >= -8);
  const mid = trials.filter(r => r.snr >= -12 && r.detected);
  const pre = trials.reduce((s, r) => s + r.preFA, 0);
  return {
    perSpecies,
    meanSNR50: r3(St.mean(snr50s)), worstSNR50: r3(Math.max(...snr50s)), bestSNR50: r3(Math.min(...snr50s)),
    meanRangeFactor: r3(Math.pow(10, -St.mean(snr50s) / 20)),
    pDetectLoud: r3(loud.filter(r => r.detected).length / Math.max(1, loud.length)), nLoud: loud.length,
    latencyMedianAtSNRge12: r3(St.quantile(mid.map(r => r.latency), 0.5)),
    lockedFracAtSNRge12: r3(St.mean(trials.filter(r => r.snr >= -12).map(r => r.lockedFrac))),
    preOnsetFalseFrames: pre, trialsWithPreOnsetFA: trials.filter(r => r.preFA > 0).length, nTrials: trials.length,
    multiSpeciesLockTrials: trials.filter(r => r.multi > 1).length,
  };
}

// ── identification ───────────────────────────────────────────────────────────
function idBlock(rs, which) {
  const det = rs.filter(r => r.detected && r[which]);
  const spConf = {}, gConf = {};
  for (const r of rs) {
    const s = r.detected ? r[which].sp : 'none', g = r.detected ? mapGenus(r[which]) : 'none';
    (spConf[r.sp] = spConf[r.sp] || {})[s] = (spConf[r.sp][s] || 0) + 1;
    const tg = GENUS[r.sp]; (gConf[tg] = gConf[tg] || {})[g] = (gConf[tg][g] || 0) + 1;
  }
  const top1 = det.filter(r => r[which].sp === r.sp).length;
  const top3 = det.filter(r => (r[which].post || []).slice(0, 3).some(p => p[0] === r.sp) || r[which].sp === r.sp).length;
  const gTop1 = det.filter(r => mapGenus(r[which]) === GENUS[r.sp]).length;
  const gOfSp = det.filter(r => GENUS[r[which].sp] === GENUS[r.sp]).length;
  // posterior calibration (species): top-1 probability = p assigned to the reported species
  const pTop = [], yTop = [], dists = [], truths = [], gp = [], gy = [], gd = [], gt = [];
  let ll = 0;
  for (const r of det) {
    const post = Object.fromEntries(r[which].post || []);
    pTop.push(post[r[which].sp] || 0); yTop.push(r[which].sp === r.sp ? 1 : 0);
    dists.push(post); truths.push(r.sp);
    ll += -Math.log(Math.max(1e-4, post[r.sp] || 0));
    const g = genusDist(r[which]), mg = mapGenus(r[which]);
    gp.push(g[mg] || 0); gy.push(mg === GENUS[r.sp] ? 1 : 0); gd.push(g); gt.push(GENUS[r.sp]);
  }
  const rel = St.reliability(pTop, yTop, 10), grel = St.reliability(gp, gy, 10);
  return {
    n: rs.length, detected: det.length,
    speciesTop1: r3(top1 / Math.max(1, det.length)), speciesTop3: r3(top3 / Math.max(1, det.length)),
    genusTop1: r3(gTop1 / Math.max(1, det.length)), genusOfSpeciesTop1: r3(gOfSp / Math.max(1, det.length)),
    speciesBrier: r3(St.brierMulti(dists, truths)), speciesECE: r3(rel.ece), speciesLogLoss: r3(ll / Math.max(1, det.length)),
    genusBrier: r3(St.brierMulti(gd, gt)), genusECE: r3(grel.ece),
    speciesReliability: rel.bins.map(b => ({ ...b, meanP: r3(b.meanP), freq: r3(b.freq) })),
    genusReliability: grel.bins.map(b => ({ ...b, meanP: r3(b.meanP), freq: r3(b.freq) })),
    speciesConfusion: spConf, genusConfusion: gConf,
  };
}
function identification(trials) {
  const out = {};
  for (const loc of [...new Set(trials.map(r => r.loc))]) {
    const nat = trials.filter(r => r.loc === loc && r.native !== false);
    const non = trials.filter(r => r.loc === loc && r.native === false);
    out[loc] = { first: idBlock(nat, 'first'), last: idBlock(nat, 'last') };
    if (non.length) {
      const d = non.filter(r => r.detected);
      out[loc].nonNative = { n: non.length, detected: d.length,
        speciesCorrect: d.filter(r => r.first.sp === r.sp).length,
        genusCorrect: d.filter(r => mapGenus(r.first) === GENUS[r.sp]).length };
    }
  }
  return out;
}

// ── detection confidence calibration: P(real mosquito | confidence) ─────────
function confidenceCalibration(trials, faRuns) {
  const ps = [], ys = [], pk = [], yk = [];
  for (const r of trials) if (r.detected) { ps.push(r.first.conf); ys.push(1); pk.push(r.peakConf); yk.push(1); }
  for (const f of faRuns) {
    const real = C.CLASSES[f.cls].countsAsFA === false ? 1 : 0;
    for (const e of f.events) { ps.push(e.conf); ys.push(real); pk.push(e.conf); yk.push(real); }
  }
  const rel = St.reliability(ps, ys, 10);
  return { n: ps.length, positives: ys.filter(y => y).length, negatives: ys.filter(y => !y).length,
           brier: r3(St.brierBinary(ps, ys)), ece: r3(rel.ece), meanConfPositive: r3(St.mean(ps.filter((_, i) => ys[i]))),
           meanConfNegative: r3(St.mean(ps.filter((_, i) => !ys[i]))),
           reliability: rel.bins.map(b => ({ ...b, meanP: r3(b.meanP), freq: r3(b.freq) })) };
}

// ── false alarms ─────────────────────────────────────────────────────────────
function falseAlarms(runs) {
  const by = {};
  for (const r of runs) (by[r.cls] = by[r.cls] || []).push(r);
  const perClass = {};
  let totEv = 0, totH = 0;
  for (const [cls, rs] of Object.entries(by)) {
    const sec = rs.reduce((s, r) => s + r.seconds, 0), ev = rs.reduce((s, r) => s + r.events.length, 0);
    const frames = rs.reduce((s, r) => s + r.frames, 0), locked = rs.reduce((s, r) => s + r.locked, 0);
    const sp = {};
    for (const r of rs) for (const e of r.events) sp[e.sp] = (sp[e.sp] || 0) + 1;
    const h = sec / 3600, ci = St.poissonRateCI(ev, h);
    const counts = C.CLASSES[cls].countsAsFA !== false;
    perClass[cls] = { group: C.CLASSES[cls].group, countsAsFA: counts, hours: r3(h), events: ev, perHour: r3(ev / h), ci: ci.map(r3),
      lockedFrac: +(locked / Math.max(1, frames)).toFixed(5), calLockedFrames: rs.reduce((s, r) => s + r.calLocked, 0),
      eventSpecies: sp, maxLevel: r3(Math.max(...rs.map(r => r.maxLevel))), fracLevel30: +(St.mean(rs.map(r => r.fracLevel30))).toFixed(5) };
    if (counts) { totEv += ev; totH += h; }
  }
  const counted = Object.entries(perClass).filter(([, v]) => v.countsAsFA);
  return { perClass, pooled: { events: totEv, hours: r3(totH), perHour: r3(totEv / Math.max(1e-9, totH)), ci: St.poissonRateCI(totEv, totH).map(r3) },
           classesOverTarget: counted.filter(([, v]) => v.perHour > 0.2).map(([k]) => k),
           classesWithAnyEvent: counted.filter(([, v]) => v.events > 0).map(([k]) => k) };
}

// ── mixtures (paired with the adversary-free run of the same trial) ─────────
function mixtures(trials) {
  const by = {};
  for (const r of trials) (by[r.mixClass] = by[r.mixClass] || []).push(r);
  const ref = by.none || [];
  const out = {};
  // A mixture trial counts as detected only if a Detection's f0 is within 5 %
  // of the individual's true f0 (otherwise the adversary itself locked).
  // Older engines without measured f0 on every path are still matched on f0.
  const hit = (r) => r.detectedMatch !== undefined ? r.detectedMatch : r.detected;
  for (const [cls, rs] of Object.entries(by)) {
    const k = rs.filter(hit).length;
    const both = rs.filter(r => hit(r) && hit(ref.find(x => x.trial === r.trial) || {})).length;
    const refK = rs.filter(r => hit(ref.find(x => x.trial === r.trial) || {})).length;
    const gen = rs.filter(r => r.detected);
    out[cls] = { n: rs.length, k, p: r3(k / rs.length), ci: St.wilson(k, rs.length).map(r3), retainedOfRef: refK ? r3(both / refK) : null,
                 anyLock: rs.filter(r => r.detected).length, otherLockFrames: rs.reduce((s, r) => s + (r.otherFrames || 0), 0),
                 genusCorrect: gen.length ? r3(gen.filter(r => mapGenus(r.first) === GENUS[r.sp]).length / gen.length) : null,
                 latencyMedian: r3(St.quantile(rs.filter(hit).map(r => r.latencyMatch ?? r.latency), 0.5)), preOnsetFalseFrames: rs.reduce((s, r) => s + r.preFA, 0) };
  }
  return out;
}

function levelsSummary(trials, faRuns) {
  const out = {};
  for (const level of [...new Set(trials.map(r => r.level))].sort()) {
    const d = detection(trials.filter(r => r.level === level));
    const fa = falseAlarms(faRuns.filter(r => r.level === level));
    out[level] = { meanSNR50: d.meanSNR50, worstSNR50: d.worstSNR50, pDetectLoud: d.pDetectLoud, faPooled: fa.pooled,
                   faClassesWithEvents: fa.classesWithAnyEvent, latencyMedianAtSNRge12: d.latencyMedianAtSNRge12 };
  }
  return out;
}

function aggregate(results, meta) {
  const T = (e) => results.filter(r => r.exp === e);
  const det = T('detect'), id = T('id'), mix = T('mix'), fa = T('fa');
  const m = { meta };
  if (det.length) m.detection = detection(det);
  if (id.length) m.identification = identification(id);
  if (fa.length) m.falseAlarms = falseAlarms(fa);
  if (det.length || id.length) m.confidence = confidenceCalibration([...det, ...id, ...mix], fa);
  if (mix.length) m.mixtures = mixtures(mix);
  const lv = T('levels'), lvfa = T('levels-fa');
  if (lv.length) {
    // level 3 rows come from the main experiments, restricted to the same SNR grid / classes
    const snrs = new Set(lv.map(r => r.snr)), trialsN = Math.max(...lv.map(r => r.trial)) + 1;
    const l3 = det.filter(r => snrs.has(r.snr) && r.trial < trialsN).map(r => ({ ...r, level: 3 }));
    const cls = new Set(lvfa.map(r => r.cls));
    const l3fa = fa.filter(r => cls.has(r.cls) && r.run === 0).map(r => ({ ...r, level: 3, seconds: r.seconds }));
    m.levels = levelsSummary([...lv, ...l3], [...lvfa.map(r => ({ ...r })), ...l3fa]);
  }
  return m;
}

module.exports = { aggregate, detection, identification, falseAlarms, mixtures, confidenceCalibration, GENUS, IDS, GENERA, mapGenus };
