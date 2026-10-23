'use strict';
/*
 * Flight-tone populations: female mosquitoes of the 12 catalogued species,
 * male mosquitoes, and other flying insects — plus the streaming flight-tone
 * source used for all of them.
 *
 * ── Ground truth used by the corpus (deliberately NOT identical to the model
 *    the detector assumes, so identification is tested under model mismatch) ──
 *
 *  • Individual female f0 at the reference temperature is log-normal around the
 *    catalogue value SPECIES_DB.freq with a per-species coefficient of variation
 *    drawn once in [4 %, 8 %] (body size, age, nutrition). Literature: CV of
 *    female wingbeat ≈ 4–8 % within a species/population — Brogdon 1994 (J Med
 *    Entomol 31:700), Villarreal et al. 2017 (J Med Entomol 54:1116), Kiskin
 *    et al. 2021 (HumBugDB, NeurIPS D&B).
 *  • Temperature: wingbeat frequency of ectotherms rises ≈ 1–2 %/°C (Unwin &
 *    Corbet 1984, Physiol Entomol 9:115; Villarreal et al. 2017 for Ae.
 *    aegypti). Per-species coefficient drawn once in [1.0, 2.0] %/°C; room
 *    temperature per trial ~ N(26, 3²) °C clipped to [18, 34]; the catalogue
 *    values are taken to hold at 26 °C.
 *  • Flight tone = harmonic series. H2 is typically 2–16 dB below H1 and the
 *    harmonics radiate with different directivity, so their ratios fluctuate as
 *    the insect turns (Arthur et al. 2014, JASA 135:933). Far-field pressure
 *    falls as 1/r (ibid.), so flight-path distance changes modulate the level.
 *  • Within a flight f0 wanders slowly (manoeuvres, Doppler v/c ≈ 0.1–0.3 %):
 *    OU process with σ = 0.25–1.5 % and correlation time 0.5–3 s, plus a fast
 *    cycle-to-cycle jitter of 0.1–0.4 % (τ = 50 ms). The lower end of σ models
 *    near-hovering flight and stresses any "too steady = machine" veto.
 *  • Males fly at ≈ 1.3–1.6 × the conspecific female f0 (Gibson & Russell 2006,
 *    Curr Biol 16:1311; Cator et al. 2009, Science 323:1077; Pennetier et al.
 *    2010, Curr Biol 20:131).
 *  • Other insects (Sotavalta 1947, Acta Entomol Fenn 4; Unwin & Corbet 1984):
 *    house fly ≈ 190–200 Hz, honeybee ≈ 230–250 Hz, bumblebee ≈ 130–200 Hz,
 *    fruit fly ≈ 200–250 Hz, non-biting midges (Chironomidae) ≈ 400–1000 Hz.
 *    Flies and bees have richer harmonic series than mosquitoes.
 */
const { rng, seedFrom } = require('../lib/dsp');
const S = require('./synth');

const POP = {
  T_REF: 26, T_MEAN: 26, T_SD: 3, T_MIN: 18, T_MAX: 34,
  CV: [0.04, 0.08],
  TC: [0.010, 0.020],
  MALE_RATIO: [1.3, 1.6],
};

// Per-species ground-truth variability (fixed per species, unknown to the detector).
function speciesTruth(sp) {
  const R = rng(seedFrom('species-truth-v1', sp.id));
  return { cv: R.range(POP.CV[0], POP.CV[1]), tc: R.range(POP.TC[0], POP.TC[1]) };
}

function drawTemperature(R) {
  return S.clamp(POP.T_MEAN + POP.T_SD * R.g(), POP.T_MIN, POP.T_MAX);
}

// Female individual's mean wingbeat f0 at temperature T.
function femaleF0(sp, R, T) {
  const { cv, tc } = speciesTruth(sp);
  return sp.freq * (1 + tc * (T - POP.T_REF)) * Math.exp(cv * R.g());
}

// Flight dynamics + harmonic structure of one individual.
//   kind: 'mosquito' | 'fly' (richer harmonics)
function drawFlight(R, kind = 'mosquito') {
  const hdb = [0];
  if (kind === 'fly') {
    hdb.push(R.range(-8, -1));
    for (let h = 3; h <= 8; h++) hdb.push(hdb[h - 2] + R.range(-5, -1));
  } else {
    hdb.push(R.range(-16, -2));
    for (let h = 3; h <= 6; h++) hdb.push(hdb[h - 2] + R.range(-10, 0));
  }
  return {
    harmonicsDb: hdb,
    wanderSigma: R.logRange(0.0025, 0.015), wanderTau: R.logRange(0.5, 3),
    jitterSigma: R.logRange(0.001, 0.004), jitterTau: 0.05,
    distSigma: R.range(0.15, 0.35), distTau: R.logRange(0.5, 2),     // log-distance OU (1/r level)
    dirSigmaDb: 2.0, dirTau: 0.7,                                     // per-harmonic directivity
  };
}

// Streaming flight-tone source.
//   o: { f0, start (s), dur (s), p1 (mean power of the fundamental over the bout,
//        i.e. A1²/2 averaged), flight, fadeIn, fadeOut, R }
// The envelope is precomputed on a 5 ms grid and normalised so that the
// fundamental's mean power over the bout (excluding fades) is exactly p1.
function flightTone(sr, o) {
  const R = o.R, fl = o.flight, dt = 0.005;
  const n = Math.ceil(o.dur / dt) + 2;
  const fArr = new Float64Array(n), eArr = new Float64Array(n);
  const wander = new S.OU(R, fl.wanderSigma, fl.wanderTau, dt);
  const jitter = new S.OU(R, fl.jitterSigma, fl.jitterTau, dt);
  const dist = new S.OU(R, fl.distSigma, fl.distTau, dt);
  for (let i = 0; i < n; i++) {
    const t = i * dt;
    fArr[i] = o.f0 * (1 + wander.at(t) + jitter.at(t));
    eArr[i] = Math.exp(-dist.at(t));                 // amplitude ∝ 1/r
  }
  const fi = o.fadeIn ?? 0.6, fo = o.fadeOut ?? 0.6;
  let ms = 0, cnt = 0;
  for (let i = 0; i < n; i++) { const t = i * dt; if (t >= fi && t <= o.dur - fo) { ms += eArr[i] * eArr[i]; cnt++; } }
  const norm = cnt ? Math.sqrt(ms / cnt) : 1;
  const a1 = Math.sqrt(2 * o.p1) / norm;
  for (let i = 0; i < n; i++) {
    const t = i * dt;
    let g = a1;
    if (t < fi) g *= 0.5 - 0.5 * Math.cos(Math.PI * t / fi);
    if (t > o.dur - fo) g *= 0.5 - 0.5 * Math.cos(Math.PI * Math.max(0, o.dur - t) / fo);
    eArr[i] *= g;
  }
  const base = fl.harmonicsDb.map(S.dbToAmp);
  const dir = base.map(() => new S.OU(R, fl.dirSigmaDb, fl.dirTau, 0.02));
  return new S.Tone(sr, {
    start: o.start, dur: o.dur, R,
    f0: S.gridFn(fArr, dt), env: S.gridFn(eArr, dt), amps: base,
    harmFn: (t, f, amps) => {
      for (let h = 1; h < amps.length; h++) amps[h] = base[h] * Math.pow(10, dir[h].at(t) / 20);
    },
  });
}

// Other insects: f0 draw per individual.
const INSECTS = {
  housefly:  { kind: 'fly', f0: R => S.clamp(195 + 8 * R.g(), 165, 225), ref: 'Musca domestica ≈ 190–200 Hz (Sotavalta 1947)' },
  honeybee:  { kind: 'fly', f0: R => S.clamp(240 + 8 * R.g(), 220, 260), ref: 'Apis mellifera ≈ 230–250 Hz (Sotavalta 1947)' },
  bumblebee: { kind: 'fly', f0: R => R.range(130, 200),                  ref: 'Bombus spp. ≈ 130–200 Hz (Unwin & Corbet 1984)' },
  fruitfly:  { kind: 'fly', f0: R => R.range(200, 250),                  ref: 'Drosophila ≈ 200–250 Hz' },
  midge:     { kind: 'mosquito', f0: R => R.logRange(400, 1000),         ref: 'Chironomidae ≈ 400–1000 Hz (Sotavalta 1947)' },
};

module.exports = { POP, speciesTruth, drawTemperature, femaleF0, drawFlight, flightTone, INSECTS };
