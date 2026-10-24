/* MosquitoNet — acoustic detection engine (pure DSP, no DOM).
 *
 * Runs unchanged in the browser (classic script) and in Node (tests/, tools/)
 * via tools/lib/load-engine.js, which evaluates js/species.js + this file in
 * one vm context — the same shared-global environment the page provides.
 *
 * ══ ENGINE ⇄ UI CONTRACT (stable; extend, never break) ══════════════════
 *
 *  const clf = new MosquitoClassifier(engine)   // engine: { binHz, freqToBin(hz), sampleRate?, fftSize? }
 *  clf.setSensitivity(level)                    // 1 (most sensitive) … 5 (strictest)
 *  clf.setLocation(lat, lng)                    // null/null when unknown
 *  clf.setDisabledSpecies(set)                  // Set of species ids to ignore
 *  clf.beginCalibration() / clf.endCalibration()// fast noise learning, then clean vote buffers
 *  clf.analyze(rawFreqDb, timeSamples) → Detection[]   // call once per hop (~4096 samples)
 *  clf.getState() → EngineState                 // cheap snapshot for the UI, any time
 *  clf.getStatus(), clf.getRawConfidences()     // legacy, kept for compatibility
 *
 *  Detection = {
 *    species:     SPECIES_DB entry (most probable species, MAP of `posterior`),
 *    genus:       'Anopheles' | 'Aedes' | 'Culex' | … (from species.genus),
 *    frequency:   measured fundamental f0 in Hz (alias: f0),
 *    f0:          same as frequency,
 *    snr:         dB, snrLinear: linear power ratio,
 *    confidence:  0…1 calibrated P(real flying mosquito | track evidence) — NOT species
 *                 certainty (that is `posterior` / `taxonLevel`),
 *    voteFraction:0…1,
 *    range:       'near' | 'far',
 *    posterior:   [{ id, p }]  species probabilities, sorted desc, Σp ≤ 1,
 *    maleP:       P(male | f0, location) — males (≈ 1.42× the female wingbeat)
 *                 do not bite; species/posterior sum both sexes,
 *    mode:        'burst' when accumulated from repeated short passes (see
 *                 _burstDetection; confidence then NOT calibrated), else absent,
 *    // extensions (additive):
 *    genusPosterior: [{ genus, p }] sorted desc,
 *    taxonLevel:  'species' | 'genus'. 'species' when the top species posterior
 *                 ≥ 0.6; otherwise 'genus', and then `species` is the most probable
 *                 species WITHIN the most probable genus (species.genus === genus),
 *                 so the UI can render "Aedes sp. — species uncertain",
 *    genusP:      posterior probability of `genus`,
 *    trackId:     integer, stable while the same flight track stays locked,
 *  }
 *
 *  EngineState = {
 *    phase:   'idle' | 'calibrating' | 'gated' | 'listening' | 'candidate' | 'detected',
 *    noiseFloorDb, ambientDb,
 *    candidate: { speciesId, level 0…1, f0 } | null,  // strongest evidence (even pre-lock)
 *    detections: Detection[] (last analyze() result),
 *    perSpecies: { [id]: { level 0…1, status: 'off'|'idle'|'building'|'locked',
 *                          reason: short human-readable text, debug: raw gate string } },
 *    // extensions (additive):
 *    calibration:   0…1 progress of the startup calibration (1 when not calibrating),
 *    ambientGateDb: the hard "loud room" gate (phase 'gated') in AnalyserNode dB,
 *    bandNoiseDb:   per-bin noise floor in 300-800 Hz (AnalyserNode dB),
 *    noiseMarginDb: how far (dB) a typical mosquito at 30 cm would sit above the
 *                   lock threshold in this noise (MN_DSP.REF_MOSQ_PEAK_DB),
 *    noisy:         noiseMarginDb < 0 — "too noisy to hear a mosquito at ~30 cm"
 *                   (may be true while phase is 'listening'),
 *  }
 *
 *  Additive API: clf.setTimeOfDay('day'|'dusk'|'night'|'none'|null)  (null = device clock)
 *                clf.lastEvidence  — per-frame internals of the best track (debug / tools)
 *  Legacy fields still maintained for the current UI: lastRejection, voteCount, VOTE_FRAMES.
 * ═══════════════════════════════════════════════════════════════════════
 */
'use strict';
/* global SPECIES_DB, WINGBEAT_MODEL, timeOfDayCategory, getNativeSpecies */

// ═══════════════════════════════════════════════════════════════════
//  Architecture (v3)
//
//   DETECTION — "is a flying mosquito audible?" — is decided per flight
//   TRACK, independently of species:
//     FarFieldTracker: per-bin whitening → harmonic sum (H1..H3) with a
//     sub-harmonic veto → Viterbi/CUSUM track-before-detect → up to four
//     simultaneous tracks (so a loud fan or voice cannot mask a mosquito).
//     Each track then has to look PHYSICALLY like a flying insect for most of
//     the last ~2 s before it may lock (see _plausibility).
//
//   IDENTIFICATION — "which species?" — is a Bayesian posterior over the
//   enabled species given the track's mean f0 (SpeciesModel): log-normal
//   wingbeat populations (between-individual + unknown-temperature spread),
//   native-range prior, time-of-day prior.
//
//  Everything is derived from the engine's actual sample rate, FFT size and
//  hop, so the same code is correct at 44.1 kHz and 48 kHz.
// ═══════════════════════════════════════════════════════════════════

// Physical / timing constants. Times in seconds, frequencies in Hz; converted
// to frames with the actual hop at configure time.
const MN_DSP = {
  F_LO: 262,              // f0 search band: Toxorhynchites (290 Hz) −1.4σ …
  F_HI: 900,              // … Ae. japonicus (730 Hz) +2.6σ (σ ≈ 8 % incl. temperature)
  TRACK_MEMORY_S: 1.28,   // CUSUM forgetting time (λ = 0.93 per 92.9 ms frame at 44.1 kHz)
  NOISE_TAU_S: 9.3,       // per-bin noise floor time constant for noise-like bins
  STRONG_RISE_PER_S: 0.04,// max fractional floor rise per second under a strong line (absorbs a
                          // machine in minutes, a mosquito visit barely at all)
  NOTCH_TAU_S: 2.3,       // fast absorption of lines identified as machines
  CAL_S: 1.86,            // initial noise-floor learning before any decision
  BLANK_NORM: 2,          // mean local normaliser above which a frame is blanked (dev: traffic FA −56 %)
  SUB_VETO: 5,            // whitened level that marks a sub-harmonic as present
  MAX_TRACKS: 4,
  TRACK_MISS_S: 0.4,      // a track survives this long without a supporting peak
  WIN_S: 2.0,             // plausibility window
  MIN_TRACK_S: 1.1,       // a track must exist this long before it may vote
  VOTE_S: 1.5,            // lock vote window
  // "Room too noisy" reference (EngineState.noisy): a typical female mosquito's
  // fundamental at 30 cm, as it appears in the AnalyserNode peak bin, versus
  // the per-bin noise floor in 300-800 Hz plus the in-bin SNR the tracker needs
  // to lock (≈ +7 dB at SNR50 in tools/eval). Assumptions (docs/EVALUATION.md):
  // 30 dB SPL at 10 cm (audible to people at ~0.5-1 m in a quiet room) → 20.5 dB
  // SPL at 30 cm by 1/r; phone MEMS mic −38 dBFS at 94 dB SPL; Blackman peak
  // bin of a sinusoid of RMS a is 20·log10(a) − 10.5 dB  ⇒  ≈ −122 dB.
  REF_MOSQ_PEAK_DB: -122,
  LOCK_INBIN_SNR_DB: 7,
  CAL_EXPECTED_S: 2.0,    // the app's startup calibration length (progress indicator)
  VOTE_ON: 0.6,           // fraction of the vote window that must pass to lock …
  VOTE_OFF: 0.3,          // … and below which a lock is released (hysteresis)
};

// Sensitivity levels 1 (most sensitive) … 5 (strictest).
// BURST_*: the burst-accumulation path (see _burstDetection), operating points
// measured on REAL recordings (tools/real; docs/REAL_DATA.md), held-out test
// split. At the default level the strict intermittency rule (0.3 s of silence
// at the pitch before a burst) keeps synthetic held notes/tones out and leaves
// false alarms unchanged; levels 1–2 relax it to 0.1 s, which doubled
// detections in real field recordings (29.8 → 64.6 events/h) for +1.8 false
// alarms/h on real street/construction/speech audio.
const MN_LEVELS = [
  { VOTE_S: 1.2, VOTE_ON: 0.55, Z_MOD: 1.5, BURST_THETA: 0.6, BURST_K: 3, BURST_W_S: 30, BURST_TOL: 0.06, BURST_GAP_S: 0.1 },
  { VOTE_S: 1.5, VOTE_ON: 0.6,  Z_MOD: 2.0, BURST_THETA: 0.6, BURST_K: 3, BURST_W_S: 15, BURST_TOL: 0.03, BURST_GAP_S: 0.1 },
  { VOTE_S: 2.0, VOTE_ON: 0.7,  Z_MOD: 2.0, BURST_THETA: 0.7, BURST_K: 4, BURST_W_S: 30, BURST_TOL: 0.06, BURST_GAP_S: 0.3 },
  { VOTE_S: 2.5, VOTE_ON: 0.75, Z_MOD: 3.0, BURST_THETA: 0.8, BURST_K: 4, BURST_W_S: 30, BURST_TOL: 0.06, BURST_GAP_S: 0.3 },
  { VOTE_S: 3.0, VOTE_ON: 0.8,  Z_MOD: 4.0, BURST_K: 0 },      // strictest: lock path only
];

// Plausibility thresholds (see MosquitoClassifier._plausibility; measured on
// the dev split of tools/eval — never on the reported test split).
const MN_PL = {
  PRES: 0.75,                // fraction of window frames with the fundamental ≥ 2× noise
  DF_RATE: 43,               // Hz/s, median frame-to-frame |Δf0|
  SUB_FRAC: 0.3,             // fraction of frames with a sub-harmonic (f/2 & 3f/2, or f/3 & 2f/3)
  DOM: 0.5,                  // P(H1)/P(H2) over the window
  SPREAD: 0.12,              // median line-width excess (vibrato)
  PURE_SNR: 600, PURE_RATIO: 0.003,   // loud (≥ 28 dB) yet H2+H3 ≤ −25 dB → pure tone
  COMB_FRAC: 0.2,            // fraction of frames with ≥2 comparable neighbouring lines (mosquito p99 ≈ 0.14)
  SPECIES_P: 0.6,            // posterior needed to name a species (else genus level)
  // Confidence logistic weights [bias, margin, snr, modulation, busyness, harmonicity],
  // fitted on the dev split by tools/eval/fit-confidence.js --mask 1,0,0,1,0
  // (margin + busyness; SNR/modulation/harmonicity added nothing on dev, and the
  // margin weight is constrained ≥ 0 so more evidence never lowers P(real)).
  CONF: [3.054, 0.374, 0, 0, -2.603, 0],
  Z_MOD: 2.0,                // modulation significance needed (pitch OR level) before voting
  STEADY_MIN_SNR: 8,         // in-bin SNR a frame needs to enter the modulation/steadiness test
  STEADY_MIN_N: 8,           // … and how many such frames the test needs
  // ── Burst accumulation (real recordings: flight is audible in short bursts) ──
  // Real phone recordings of mosquitoes (HumBug/MozzBNN: CDC cultures, Ifakara,
  // DRC) show audible flight in EPISODES of median 0.1–0.4 s (90th pct ≈ 0.4–1.5 s),
  // far shorter than the lock path's ≥ 1.1 s track + 2 s vote. A short track
  // that the learned plausibility model scores ≥ BURST_THETA is a BURST; K
  // bursts within BURST_W_S whose f0 agree within ±BURST_TOL make a detection
  // (a mosquito makes repeated passes at nearly the same wingbeat; tonal
  // transients of streets, voices and machines come at scattered pitches).
  BURST_MIN_AGE_S: 0.35, BURST_MIN_SNR: 4, BURST_MIN_N: 5, BURST_HOLD_S: 3,
  BURST_GAP_S: 0.3,          // silence at the pitch required before a burst (0 = off)
  BURST_THETA: 0.7, BURST_K: 4, BURST_W_S: 30, BURST_TOL: 0.06,
  // Logistic flight-plausibility model on MosquitoClassifier.plausVector(feat),
  // fitted by tools/real/train.js on the DEV split of real recordings (HumBug
  // mosquito tracks vs. real street/construction/speech tracks and synthetic
  // adversaries); out-of-fold track AUC 0.78 (0.80 vs real backgrounds).
  PLAUS_MODEL: {"mu": [0.429614, 1.876362, 0.006685, 0.027241, 1.224724, 0.105342, -0.623028, 0.341956, 1.213729, 0.630242, -0.700905, 1.195243, 0.028581], "sd": [0.286537, 0.683323, 0.031958, 0.057512, 2.063817, 0.188532, 1.244465, 0.474365, 1.865457, 1.172836, 0.978731, 1.903854, 0.507013], "w": [0.063466, 0.964352, 0.177164, 0.063274, -0.16207, -0.806463, 0.191022, -0.744899, -0.134505, 0.652881, 0.481277, 0.165588, -0.33902, 0.419324]},
  STEADY_PITCH: 6e-4,        // detrended relative pitch s.d. below which a tone is "exact"
  STEADY_LEVEL_MACHINE: 0.5, // dB: … and level this steady → machine / electronic tone
  DECAY_DB_S: -5,            // exact pitch + steady decay → struck/plucked resonator
  STEADY_LEVEL: 0.3,         // dB: level steadier than any flying insect → held note / tone
};

// ═══════════════════════════════════════════════════════════════════
//  FarFieldTracker — whitened harmonic track-before-detect (multi-track)
//
//  Physics: beyond the acoustic near field (λ/2π ≈ 8-14 cm at 400-700 Hz)
//  a flight tone's pressure falls as 1/r (-6 dB per doubling of distance)
//  and air absorption is negligible (Arthur et al. 2014, JASA 135:933), so
//  range is set purely by SNR: every 6 dB of processing gain doubles the
//  distance at which a mosquito can be heard. The narrowband FFT already
//  buys ~25-28 dB over the full audio band in a typical room; this tracker
//  keeps that gain and adds more:
//   1. Whitening — each bin is divided by its own slowly-adapting noise
//      floor, so evidence is "times above local noise" even in coloured
//      (fan / traffic / pink) noise.
//   2. Harmonic summation — evidence from f0, 2f0 and 3f0 is combined, with
//      each upper harmonic capped relative to H1 so a single strong line at
//      2f or 3f (cricket, whistle, machine) cannot fake a track at f.
//   3. Track-before-detect — a Viterbi/CUSUM recursion accumulates evidence
//      along a frequency path allowed to drift ±1 bin per frame (Doppler,
//      manoeuvring), so ~1.3 s of weak frames add up (sonar LOFAR line
//      tracking) rather than each frame having to win alone.
// ═══════════════════════════════════════════════════════════════════
class FarFieldTracker {
  constructor(binHz, hopSec, cfg) {
    this.TAU = 30;                                    // CUSUM threshold (set by setSensitivity)
    this.cfg = cfg || MN_DSP;
    this.configure(binHz || 44100 / 8192, hopSec || 4096 / 44100);
  }

  configure(binHz, hopSec) {
    const D = this.cfg;
    const geo = [this.binHz, this.hopSec, this.kLo, this.kHi].join();
    this.binHz = binHz; this.hopSec = hopSec;
    this.kLo  = Math.max(6, Math.round(D.F_LO / binHz));
    this.kHi  = Math.round(D.F_HI / binHz);
    this.kMin = Math.max(2, Math.floor(this.kLo / 3) - 2);    // room for the f0/3 veto
    this.kMax = 3 * this.kHi + 10;                             // room for 3f0 (+ spread bins)
    this.WH   = [1.0, 0.7, 0.5];                               // harmonic weights (H1 dominates)
    // Expected harmonic sum on pure noise: W ~ Exp(1), max of 3 ~ mean 1.833.
    this.BIAS = (this.WH[0] + (this.WH[1] + this.WH[2]) * 1.833) * 1.10;
    this.LAMBDA = Math.exp(-hopSec / D.TRACK_MEMORY_S);
    this.aNoise = 1 - Math.exp(-hopSec / D.NOISE_TAU_S);
    this.aNotch = 1 - Math.exp(-hopSec / D.NOTCH_TAU_S);
    this.strongRise = D.STRONG_RISE_PER_S * hopSec;
    this.calFrames = Math.max(5, Math.round(D.CAL_S / hopSec));
    this.maxMiss = Math.max(1, Math.round(D.TRACK_MISS_S / hopSec));
    this.histLen = Math.round(4.0 / hopSec);
    // Only a change of geometry (bin width, hop, band edges) invalidates the
    // learned noise floor and the tracks. A sensitivity change mid-session
    // re-derives the parameters but keeps both — it used to wipe the floor and
    // drop a locked mosquito for ~4 s.
    if (!this.noise || geo !== [binHz, hopSec, this.kLo, this.kHi].join()) this.reset();
  }

  reset() {
    const nb = this.nb = this.kMax + 12;
    this.noise = new Float64Array(nb);
    this.W     = new Float64Array(nb);
    this.P     = new Float64Array(nb);
    this.L     = new Float64Array(nb);                         // 10·log10(P), for interpolation
    this.S     = new Float64Array(this.kHi + 2);
    this.S2    = new Float64Array(this.kHi + 2);
    this.veto  = new Uint8Array(this.kHi + 2);
    this.Z     = new Float64Array(nb).fill(1);                // local normaliser per bin
    this.BLK   = Math.max(16, Math.round(210 / this.binHz));   // ≈ 210 Hz blocks
    this.nBlk  = Math.ceil((this.kMax - this.kMin + 3) / this.BLK);
    this._zb   = new Float64Array(this.nBlk);
    this._med  = new Float64Array(this.BLK);
    this.frames = 0; this.calCount = 0; this.forceCal = false;
    this.tracks = []; this.nextId = 1; this.notches = [];
    this.blanked = false;
  }

  setSensitivity(level) {
    const scale = [0.70, 0.85, 1.0, 1.3, 1.7][Math.max(0, Math.min(4, (level | 0) - 1))];
    this.TAU = (this.cfg.TAU_BASE || 30) * scale;
  }

  _max3(k) { const W = this.W; return Math.max(W[k - 1], W[k], W[k + 1]); }
  _notched(k) {
    for (const n of this.notches) {
      const h = Math.round(k / n.k);
      if (h >= 1 && h <= 3 && Math.abs(k - h * n.k) <= h + 1) return true;
    }
    return false;
  }
  notch(k, seconds) {
    const frames = Math.round(seconds / this.hopSec);
    const ex = this.notches.find(n => Math.abs(n.k - k) <= 2);
    if (ex) { ex.k = k; ex.frames = Math.max(ex.frames, frames); } else this.notches.push({ k, frames });
  }

  // Fractional peak position by parabolic interpolation of the dB spectrum.
  // For the Blackman window the residual bias is < 0.01 bin (tests/dsp-f0.test.js).
  _interp(k) {
    const a = this.L[k - 1], b = this.L[k], c = this.L[k + 1], den = a - 2 * b + c;
    return den < 0 ? k + Math.max(-0.5, Math.min(0.5, 0.5 * (a - c) / den)) : k;
  }
  _peakNear(kc, r) {
    let kb = kc;
    for (let d = -r; d <= r; d++) { const k = kc + d; if (k > 1 && k < this.nb - 1 && this.P[k] > this.P[kb]) kb = k; }
    return kb;
  }

  // Per-frame measurements of a candidate source whose H1 is near bin k.
  measure(k) {
    const { W, P, binHz, Z } = this;
    const k1 = this._peakNear(k, 1);
    const f1 = this._interp(k1) * binHz;
    let num = 0, den = 0;
    const w = [0, 0, 0], sig = [0, 0, 0], nse = [0, 0, 0];
    for (let h = 1; h <= 3; h++) {
      const kh = h === 1 ? k1 : this._peakNear(Math.round(h * f1 / binHz), 1);
      if (kh + 2 >= this.nb) break;
      let s = 0;
      for (let d = -1; d <= 1; d++) s += Math.max(0, P[kh + d] - this.noise[kh + d] * Z[kh + d]);
      w[h - 1] = W[kh]; sig[h - 1] = s; nse[h - 1] = this.noise[kh] * Z[kh];
      // Harmonic-consistent f0: weight each harmonic's estimate f_h/h by its
      // inverse variance ∝ h²·SNR_h (a harmonic's frequency error divides by h).
      if (W[kh] >= 4) {
        const fh = this._interp(kh) * binHz / h;
        if (Math.abs(fh - f1) <= 0.02 * f1) { const wt = h * h * (W[kh] - 1); num += wt * fh; den += wt; }
      }
    }
    // Line width: whitened excess power 4..8 bins from the H1 peak vs within ±3
    // (the Blackman main lobe). A steady tone or a slowly wandering wingbeat
    // keeps its power inside the main lobe over the 186 ms window; 4-7 Hz
    // vibrato (singing, violin, flute, whistling) smears it over ±(Δf + f_m).
    let main = 0, side = 0;
    for (let d = -8; d <= 8; d++) {
      const v = W[k1 + d] - 1;
      if (v > 0) { if (d >= -3 && d <= 3) main += v; else side += v; }
    }
    const spread = Math.max(0, side - 3.68) / (main + side + 1e-9);   // 3.68 = E[side] on noise (10·e⁻¹)
    // Comb: is k1 the n-th harmonic of a lower-pitched source (engine, fan
    // blade-pass, bee, fly, voice; f_low = f/n)? Then its NEIGHBOURING
    // harmonics f·(n−1)/n and f·(n+1)/n are present too — including the odd
    // multiples (5th, 7th, …) that the f/2 and f/3 vetoes cannot see. A
    // wingbeat has nothing between f/2 and 3f/2, and unrelated sources
    // (a voice or a chord next to a mosquito) match this pattern only by
    // coincidence. Score = best over n of min(W(f−Δ), W(f+Δ)), Δ ≥ 5 bins.
    let combScore = 0;
    for (let nH = 2, nMax = Math.floor(k1 / 5); nH <= nMax; nH++) {
      const m = Math.min(this._max3(Math.round(k1 * (nH - 1) / nH)), this._max3(Math.round(k1 * (nH + 1) / nH)));
      if (m > combScore) combScore = m;
    }
    const comb = combScore > Math.max(5, 0.1 * W[k1]);
    return { k: k1, f: den > 0 ? num / den : f1, f1, w1: w[0], w2: w[1], w3: w[2],
             s1: sig[0], s2: sig[1], s3: sig[2], n1: nse[0], n2: nse[1], n3: nse[2], spread, comb, combScore,
             lv: 10 * Math.log10(P[k1 - 1] + P[k1] + P[k1 + 1] + 1e-30),          // robust level (signal + noise)
             sub2: Math.min(this._max3(Math.round(k1 / 2)), this._max3(Math.round(1.5 * k1))),
             sub3: Math.min(this._max3(Math.round(k1 / 3)), this._max3(Math.round(2 * k1 / 3))) };
  }

  process(db) {
    const { kMin, kMax, kLo, kHi, W, P, L, noise } = this;
    if (!db || db.length <= kMax + 2) return null;

    // ── 0. no data ≠ silence ──────────────────────────────────────────
    // A muted/stolen microphone (OS mic toggle, a phone call, another app,
    // the first frames of a stream) delivers digital zeros: the AnalyserNode
    // returns −∞ in every bin. Learning that as the noise floor collapses it to
    // ~1e-30, and because a strong floor may only rise slowly afterwards, every
    // real frame then reads as an extreme broadband burst and is blanked — the
    // detector went deaf for minutes (19 min after 3 s of zeros during
    // calibration). Such frames carry no information: skip them entirely.
    let live = 0;
    for (let k = kLo; k <= kHi; k += 4) if (db[k] > -200) live++;
    if (live * 4 < (kHi - kLo + 1) / 2) {
      this.blanked = true;
      return { blanked: true, noData: true, tracks: this.tracks };
    }

    this.frames++;
    // Calibrate until at least half a calibration's worth of REAL frames has
    // been seen, even if the app's calibration window ended during a mute.
    const cal = this.forceCal || this.frames <= this.calFrames || this.calCount < (this.calFrames >> 1);
    if (cal) this.calCount++;

    // ── 1. per-bin noise floor + whitening ────────────────────────────
    for (let k = kMin - 1; k <= kMax + 1; k++) {
      const v = db[k];
      const p = Number.isFinite(v) ? Math.pow(10, v / 10) : 1e-30;
      if (cal || noise[k] <= 0) {
        noise[k] += (p - noise[k]) / Math.min(Math.max(1, this.calCount), this.calFrames);
      } else if (this.notches.length && this._notched(k)) {
        noise[k] += this.aNotch * (p - noise[k]);
      } else if (p > 4 * noise[k]) {
        // Strong component: the floor may rise only slowly, so a passing
        // mosquito is not learned as noise while a machine running for
        // minutes eventually is.
        noise[k] *= 1 + this.strongRise;
      } else {
        noise[k] += this.aNoise * (p - noise[k]);
      }
      if (noise[k] < 1e-16) noise[k] = 1e-16;          // −160 dB: below any real mic's floor
      P[k] = p; L[k] = Number.isFinite(v) ? v : -300;
      W[k] = p / (noise[k] + 1e-30);
    }
    if (cal) this._capFloor();
    for (const n of this.notches) n.frames--;
    if (this.notches.length) this.notches = this.notches.filter(n => n.frames > 0);
    if (cal) { this.blanked = false; return { calibrating: true, tracks: this.tracks }; }

    // ── 2. local level normalisation (order-statistic CFAR) ───────────
    // The per-bin floors know the noise COLOUR but adapt in ~10 s. Anything
    // faster — a level step, a click, rain, a fan's turbulence, loud speech
    // formants — moves whole regions of the spectrum. Divide each bin by the
    // median whitened power of its ~210 Hz neighbourhood (≈ ln 2 on stationary
    // noise; sparse tonal lines barely move a median), interpolated between
    // block centres. A tone survives a broadband transient as long as it stands
    // out locally, instead of the whole frame being thrown away.
    const S = this.S, S2 = this.S2, lam = this.LAMBDA, Z = this.Z, B = this.BLK;
    const nBlk = this.nBlk, zb = this._zb, md = this._med;
    for (let b = 0; b < nBlk; b++) {
      const k0 = kMin - 1 + b * B, k1 = Math.min(kMax + 1, k0 + B - 1);
      let n = 0;
      for (let k = k0; k <= k1; k++) md[n++] = W[k];
      zb[b] = Math.max(0.25, md.subarray(0, n).sort()[n >> 1] / 0.693);
    }
    for (let k = kMin - 1; k <= kMax + 1; k++) {
      const x = (k - (kMin - 1)) / B - 0.5, b0 = Math.max(0, Math.min(nBlk - 1, Math.floor(x)));
      const b1 = Math.min(nBlk - 1, b0 + 1), fr = Math.max(0, Math.min(1, x - b0));
      Z[k] = zb[b0] * (1 - fr) + zb[b1] * fr;
      W[k] /= Z[k];
    }
    // Extreme broadband frames (median whitened power over 260-2700 Hz above
    // BLANK_NORM × its noise value: bangs, a passing vehicle's tyre roar) carry
    // no usable tonal evidence: let the track statistics decay instead.
    this.blanked = false;
    if (this.cfg.BLANK_NORM > 0) {
      let m = 0; for (let b = 0; b < nBlk; b++) m += zb[b];
      if (m / nBlk > this.cfg.BLANK_NORM) {
        for (let k = kLo; k <= kHi; k++) S[k] *= lam;
        this.blanked = true;
        for (const t of this.tracks) { t.miss++; t.hist.push(null); if (t.hist.length > this.histLen) t.hist.shift(); }
        this._prune();
        return { blanked: true, tracks: this.tracks };
      }
    }

    // ── 3. harmonic sum + sub-harmonic veto + Viterbi/CUSUM update ────
    const V = this.cfg.SUB_VETO, WH = this.WH, bias = this.BIAS;
    for (let k = kLo; k <= kHi; k++) {
      const w1 = W[k], cap = 2 * w1 + 2;              // H2, H3 count at most ≈ 3 dB above H1
      let hs = WH[0] * w1 + WH[1] * Math.min(this._max3(2 * k), cap) + WH[2] * Math.min(this._max3(3 * k), cap);
      // A lower-pitched source (voice, fly, bee, motor) whose harmonics land
      // on k, 2k, 3k also has energy at k/2 & 3k/2, or k/3 & 2k/3. A mosquito
      // has nothing below its fundamental.
      const v = (this._max3(Math.round(k / 2)) > V && this._max3(Math.round(1.5 * k)) > V) ||
                (this._max3(Math.round(k / 3)) > V && this._max3(Math.round(2 * k / 3)) > V);
      this.veto[k] = v ? 1 : 0;
      if (v) hs = 0;
      const prev = Math.max(S[k - 1] || 0, S[k], S[k + 1] || 0);
      S2[k] = Math.max(0, lam * prev + hs - bias);
    }
    for (let k = kLo; k <= kHi; k++) S[k] = S2[k];

    // ── 4. peaks of the track statistic → multi-track association ─────
    const pre = 0.2 * this.TAU, peaks = [];
    for (let k = kLo + 1; k < kHi; k++) {
      const s = S[k];
      if (s < pre || s < S[k - 1] || s <= S[k + 1]) continue;
      peaks.push({ k, s });
    }
    peaks.sort((a, b) => b.s - a.s);
    const kept = [];
    for (const p of peaks) { if (kept.every(q => Math.abs(q.k - p.k) > 3)) kept.push(p); if (kept.length >= 6) break; }

    const used = new Set();
    const order = this.tracks.slice().sort((a, b) => b.T - a.T);
    for (const t of order) {
      let best = -1, bd = 3;
      for (let i = 0; i < kept.length; i++) {
        if (used.has(i)) continue;
        const d = Math.abs(kept[i].k - t.k);
        if (d < bd) { bd = d; best = i; }
      }
      if (best >= 0) { used.add(best); this._update(t, kept[best]); }
      else { t.miss++; t.T = S[t.k] || 0; t.hist.push(null); if (t.hist.length > this.histLen) t.hist.shift(); }
    }
    for (let i = 0; i < kept.length && this.tracks.length < this.cfg.MAX_TRACKS; i++) {
      if (used.has(i)) continue;
      const t = { id: this.nextId++, k: kept[i].k, T: 0, age: 0, miss: 0, hist: [], votes: [], locked: false, lockedFrames: 0 };
      this.tracks.push(t);
      this._update(t, kept[i]);
    }
    this._prune();
    return { tracks: this.tracks };
  }

  // During calibration the floor is a plain mean, so a tone that is already
  // sounding (a mosquito buzzing at start-up, a fridge) would be learned as
  // "noise" and become invisible. Cap each bin at 4× (6 dB) the local median
  // floor (~210 Hz blocks, interpolated): the noise COLOUR is still learned,
  // narrow lines are not. Steady machines kept this way are rejected by the
  // plausibility tests and absorbed/notched later.
  _capFloor() {
    const { kMin, kMax, noise, BLK: B, nBlk, _zb: zb, _med: md } = this;
    for (let b = 0; b < nBlk; b++) {
      const k0 = kMin - 1 + b * B, k1 = Math.min(kMax + 1, k0 + B - 1);
      let n = 0;
      for (let k = k0; k <= k1; k++) md[n++] = noise[k];
      zb[b] = md.subarray(0, n).sort()[n >> 1];
    }
    for (let k = kMin - 1; k <= kMax + 1; k++) {
      const x = (k - (kMin - 1)) / B - 0.5, b0 = Math.max(0, Math.min(nBlk - 1, Math.floor(x)));
      const b1 = Math.min(nBlk - 1, b0 + 1), fr = Math.max(0, Math.min(1, x - b0));
      const cap = 4 * (zb[b0] * (1 - fr) + zb[b1] * fr);
      if (noise[k] > cap) noise[k] = cap;
    }
  }

  _update(t, pk) {
    const m = this.measure(pk.k);
    m.T = pk.s; m.veto = this.veto[pk.k];
    t.k = pk.k; t.T = pk.s; t.miss = 0; t.age++;
    t.hist.push(m);
    if (t.hist.length > this.histLen) t.hist.shift();
  }

  _prune() {
    this.tracks = this.tracks.filter(t => t.miss <= this.maxMiss);
  }
}

// ═══════════════════════════════════════════════════════════════════
//  SpeciesModel — Bayesian identification from the measured wingbeat f0
//
//  p(s | f0, location, time) ∝ L(f0 | s) · π_loc(s) · π_time(s)
//
//  L: ln f0 ~ N(ln μ_s, σ_s²) with σ_s² = (freqSigma/freq)² + (β·σ_T)² + σ_m²
//     — between-individual spread at the reference temperature, the unknown
//     room temperature (β ≈ 1.5 %/°C, σ_T = 3.5 °C) and measurement error.
//  π_loc: native species (getNativeSpecies) 1, non-native 0.03, disabled 0.
//  π_time: activity period matches 1, dusk-adjacent 0.85, opposite 0.7
//     (judgement; no field data in this repo to fit it — neutral when unknown).
//  Species and genus probabilities are conditional on "this is a catalogued
//  female mosquito"; `zMin` (distance in σ to the nearest enabled species)
//  lets the caller reject tones no enabled species could produce.
// ═══════════════════════════════════════════════════════════════════
class SpeciesModel {
  constructor() {
    const M = (typeof WINGBEAT_MODEL !== 'undefined') ? WINGBEAT_MODEL : { T_REF: 26, T_MEAN: 26, T_SD: 3.5, CV: 0.06, TEMP_COEF: 0.015 };
    this.M = M;
    this.MEAS = 0.004;                       // relative f0 measurement error of a track mean
    this.NON_NATIVE = 0.03;
    // Males. Catalogue frequencies are FEMALE (the biting sex). Males beat
    // ≈ 1.3–1.6× faster (Cx. quinquefasciatus: ♀ 415–475 Hz, ♂ 534–717 Hz;
    // Sci Rep 2025 doi:10.1038/s41598-025-89608-7, J R Soc Interface 2022
    // doi:10.1098/rsif.2022.0285). Real HumBug culture/cup recordings are
    // male-dominated: their Culex flight sits at ≈ 640 Hz (docs/REAL_DATA.md),
    // where a male Culex is indistinguishable by f0 from a female Aedes.
    // Each species therefore also has a male component; its prior is low
    // because males do not seek hosts (a tone beside a sleeper is far more
    // likely female) — an assumption, not fitted (no sexed field data).
    this.MALE_RATIO = 1.42; this.MALE_RATIO_SD = 0.07; this.MALE_PRIOR = 0.25;
    this.species = SPECIES_DB.map(sp => {
      const beta = sp.freqTempCoef ?? M.TEMP_COEF;
      const cv = (sp.freqSigma ?? sp.freq * M.CV) / sp.freq;
      const sd = Math.sqrt(cv * cv + (beta * M.T_SD) ** 2 + this.MEAS ** 2);
      const mu = Math.log(sp.freq * (1 + beta * (M.T_MEAN - M.T_REF)));
      return { sp, mu, sd, muM: mu + Math.log(this.MALE_RATIO), sdM: Math.sqrt(sd * sd + this.MALE_RATIO_SD ** 2) };
    });
    this._natKey = null; this._nat = null;
  }

  _natives(lat, lng) {
    if (lat === null || lng === null || typeof getNativeSpecies !== 'function') return null;
    const key = lat.toFixed(2) + ',' + lng.toFixed(2);
    if (key !== this._natKey) { this._natKey = key; this._nat = getNativeSpecies(lat, lng); }
    return this._nat;
  }

  // ctx: { disabled: Set, lat, lng, tod: 'day'|'dusk'|'night'|null }
  posterior(f0, ctx) {
    if (!Number.isFinite(f0) || f0 <= 0) return null;
    const lf = Math.log(f0), nat = this._natives(ctx.lat, ctx.lng);
    let z = 0, zMin = Infinity, zMale = 0;
    const out = [];
    for (const m of this.species) {
      const id = m.sp.id;
      if (ctx.disabled && ctx.disabled.has(id)) continue;
      const d = (lf - m.mu) / m.sd, dM = (lf - m.muM) / m.sdM;
      zMin = Math.min(zMin, Math.abs(d), Math.abs(dM));
      let prior = 1;
      if (nat && nat.size && !nat.has(id)) prior *= this.NON_NATIVE;
      if (ctx.tod) prior *= SpeciesModel.activityPrior(m.sp.activity, ctx.tod);
      const lF = prior * Math.exp(-0.5 * d * d) / m.sd;
      const lM = prior * this.MALE_PRIOR * Math.exp(-0.5 * dM * dM) / m.sdM;
      // Species probability sums both sexes; pMale is P(male | f0, species).
      out.push({ id, p: lF + lM, sp: m.sp, pMale: lF + lM > 0 ? lM / (lF + lM) : 0 }); z += lF + lM; zMale += lM;
    }
    if (!out.length || z <= 0) return { post: [], genus: [], zMin, maleP: 0 };
    const g = {};
    for (const o of out) { o.p /= z; g[o.sp.genus] = (g[o.sp.genus] || 0) + o.p; }
    out.sort((a, b) => b.p - a.p);
    const genus = Object.entries(g).map(([genus, p]) => ({ genus, p })).sort((a, b) => b.p - a.p);
    return { post: out, genus, zMin, maleP: zMale / z };
  }

  // Deliberately mild (×1.4 at most): day-biters also bite indoors at night
  // under lights, and there is no field data here to fit stronger values.
  static activityPrior(activity, tod) {
    if (!tod || tod === 'none') return 1;
    if (activity === tod) return 1;
    if (activity === 'dusk' || tod === 'dusk') return 0.85;
    return 0.7;
  }
}

// ═══════════════════════════════════════════════════════════════════
//  MosquitoClassifier — contract wrapper: plausibility, locking, ID, state
// ═══════════════════════════════════════════════════════════════════
class MosquitoClassifier {
  constructor(engine) {
    this.engine = engine || {};
    this._lat = null; this._lng = null;
    this._disabled = new Set();
    this._fastCalib = false;
    this._tod = null; this._todCache = { t: -1, v: null };
    this._lastDetections = [];
    this.level = 3;

    // Legacy gates kept for getStatus().gated (UI "loud room" indicator).
    this.AMBIENT_GATE_DB = -18;
    this.lastNoiseFloor = -70;
    this.ambientRMS = -80;

    // Legacy per-species fields read by the current UI.
    this.VOTE_FRAMES = 16;
    this.voteCount = {}; this.lastRejection = {};
    SPECIES_DB.forEach(sp => { this.voteCount[sp.id] = 0; this.lastRejection[sp.id] = ''; });

    this.model = new SpeciesModel();
    this.cfg = Object.assign({}, MN_DSP, MN_PL);      // per-instance copy (tools may _tune it)
    this._burstList = []; this._burstLock = null; this._burstSeq = 0;   // burst-accumulation state
    this._toneLog = [];
    this.ff = null; this._cfgKey = '';
    this._configure(null);
    this.setSensitivity(3);
  }

  // ── configuration from the engine's ACTUAL sample rate / FFT size ──
  _configure(nBins) {
    const e = this.engine;
    let N = e.fftSize || (nBins ? 2 * nBins : 8192);
    if (nBins && 2 * nBins !== N) N = 2 * nBins;
    const sr = e.sampleRate || (e.binHz ? e.binHz * N : 44100);
    const binHz = sr / N;
    const hop = this._hopSamples || N / 2;          // the app analyses every N/2 new samples
    const key = `${sr}|${N}|${hop}`;
    if (key === this._cfgKey) return;
    this._cfgKey = key;
    this.sampleRate = sr; this.fftSize = N; this.binHz = binHz; this.hopSec = hop / sr;
    if (!this.ff) this.ff = new FarFieldTracker(binHz, this.hopSec, this.cfg);
    else this.ff.configure(binHz, this.hopSec);
    const fr = (s) => Math.max(1, Math.round(s / this.hopSec));
    this.WIN = fr(this.cfg.WIN_S);
    this.MIN_TRACK = fr(this.cfg.MIN_TRACK_S);
    this.BURST_MIN_AGE = fr(this.cfg.BURST_MIN_AGE_S);
    this.VOTE_FRAMES = fr(this.cfg.VOTE_S);
    if (this.level) this.ff.setSensitivity(this.level);
  }
  // Tools only (tools/eval): override configuration constants, then rebuild.
  _tune(o) { this._tuned = true; Object.assign(this.cfg, o || {}); this._cfgKey = ''; this.ff = null; this._configure(this.fftSize / 2); }

  // Optional: tell the engine the real analysis hop (samples) if it is not fftSize/2.
  setHop(samples) { this._hopSamples = samples > 0 ? samples : null; this._cfgKey = ''; this._configure(this.fftSize / 2); }

  // ── Contract API ─────────────────────────────────────────────────
  setLocation(lat, lng) {
    this._lat = Number.isFinite(lat) ? lat : null;
    this._lng = Number.isFinite(lng) ? lng : null;
  }
  // The UI passes its live Set; mutations made later are seen immediately.
  setDisabledSpecies(set) { this._disabled = set instanceof Set ? set : new Set(set || []); }
  setTimeOfDay(tod) { this._tod = tod === undefined ? null : tod; }

  // Startup calibration: learn the noise floor fast; no detections until done.
  beginCalibration() { this._fastCalib = true; this._calFrames = 0; if (this.ff) { this.ff.forceCal = true; this.ff.calCount = 0; } }
  endCalibration() {
    this._fastCalib = false;
    this._burstList = []; this._burstLock = null; this._toneLog = [];
    if (this.ff) { this.ff.forceCal = false; this.ff.tracks = []; }
    this.resetVotes();
  }
  resetVotes() {
    SPECIES_DB.forEach(sp => { this.voteCount[sp.id] = 0; });
    if (this.ff) for (const t of this.ff.tracks) { t.votes = []; t.locked = false; }
  }

  // Sensitivity = operating point on the detection/false-alarm trade-off.
  // Each level scales the CUSUM threshold τ, the vote window/quorum and the
  // flight-modulation evidence required (MN_LEVELS; measured operating points
  // in docs/EVALUATION.md).
  setSensitivity(level) {
    this.level = Math.max(1, Math.min(5, Math.round(level) || 3));
    if (!this._tuned) Object.assign(this.cfg, MN_LEVELS[this.level - 1]);
    this._cfgKey = '';
    if (this.fftSize) this._configure(this.fftSize / 2);
    if (this.ff) {
      this.ff.setSensitivity(this.level);
      // Vote windows may have shrunk: keep each track's most recent votes.
      for (const t of this.ff.tracks) if (t.votes && t.votes.length > this.VOTE_FRAMES) t.votes.splice(0, t.votes.length - this.VOTE_FRAMES);
    }
  }

  // Feature vector of the learned flight-plausibility model (tools/real/train.js
  // fits it on real recordings; the SAME function is used for training and in
  // the app, so the two can never drift apart). Shape and dynamics only — no f0
  // and no absolute level, so the model cannot learn "this lab's mosquitoes sit
  // at 700 Hz" or "loud means mosquito".
  static plausVector(f) {
    // Missing values (a track with no live frame in the window omits them) take a
    // neutral default INSIDE the clamp, so no log() ever sees 0.
    const cl = (v, a, b, d = 0) => Math.max(a, Math.min(b, Number.isFinite(v) ? v : d));
    const sl = (z) => (Number.isFinite(z) ? Math.sign(z) * Math.log1p(Math.abs(z)) : 0);
    const st = Number.isFinite(f.zf) ? 1 : 0;
    return [
      cl(f.pres, 0, 1),                                   // continuity of the fundamental
      Math.log1p(cl(f.dfRate, 0, 2000) / 10),             // pitch rate
      cl(f.subFrac, 0, 1),                                // sub-harmonic evidence
      cl(f.combFrac, 0, 1),                               // comb of a lower source
      Math.log(cl(f.dom, 0.01, 1000, 1)),                 // H1 dominance
      cl(f.spread, 0, 2),                                 // line broadening / vibrato
      Math.log10(cl(f.hRatio, 0, 100) + 1e-4),            // harmonic richness
      st,                                                 // modulation measurable?
      st ? cl(sl(f.zf), -3, 8) : 0,                       // pitch modulation significance
      st ? cl(sl(f.zl), -3, 8) : 0,                       // level modulation significance
      st ? Math.log10(cl(f.ubf, 0, 1) + 1e-5) : 0,        // pitch steadiness bound
      st ? cl(f.ubl, 0, 12) : 0,                          // level steadiness bound
      st ? cl(f.slope, -60, 60) / 10 : 0,                 // level trend (decay = struck note)
    ];
  }

  // Human-readable text for a raw gate string (raw text stays in .debug).
  static humanReason(raw) {
    if (!raw) return '';
    if (raw.startsWith('weak'))     return 'no tone';
    if (raw.startsWith('short'))    return 'too brief';
    if (raw.startsWith('gaps'))     return 'not continuous';
    if (raw.startsWith('pitch'))    return 'pitch jumps';
    if (raw.startsWith('sub'))      return 'lower-pitched source';
    if (raw.startsWith('h2'))       return 'harmonics too strong';
    if (raw.startsWith('vibrato'))  return 'vibrato (voice/music)';
    if (raw.startsWith('pure'))     return 'pure tone';
    if (raw.startsWith('machine'))  return 'steady machine tone';
    if (raw.startsWith('harmonic')) return 'harmonic of another tone';
    if (raw.startsWith('range'))    return 'not a listed species';
    if (raw.startsWith('loud'))     return 'room too loud';
    if (raw.startsWith('calib'))    return 'calibrating';
    return 'no match';
  }

  _timeOfDay() {
    if (this._tod) return this._tod === 'none' ? null : this._tod;
    const now = Date.now();
    if (now - this._todCache.t > 60000 && typeof timeOfDayCategory === 'function') {
      this._todCache = { t: now, v: timeOfDayCategory(this._lat, this._lng) };
    }
    return this._todCache.v;
  }

  // ── Plausibility: does this track look like a free-flying mosquito? ──
  // Returns { ok, reason, feat }. Each test is a physical property of the
  // wingbeat tone (thresholds measured on the dev split of tools/eval):
  //  continuity  — a flight tone is continuous; speech comes in syllables
  //  smooth pitch— manoeuvres/Doppler move f0 by a few Hz/s; intonation and
  //                melodies jump
  //  H1 dominance— the fundamental is (nearly always) the strongest partial
  //  no vibrato  — singing, violin, flute: 4-7 Hz FM smears the line
  //  harmonics   — wing motion is far from sinusoidal: a loud tone with no
  //                H2/H3 at all is a whistle / phone beep, not a wingbeat
  //  not machine — a flying insect's level and harmonic balance fluctuate
  //                as it moves/turns; a fan or transformer line is steady
  _plausibility(t, o = {}) {
    const H = t.hist, n = H.length, w = Math.min(n, this.WIN);
    const recent = [];
    for (let i = n - w; i < n; i++) recent.push(H[i]);
    const live = recent.filter(Boolean);
    const P = this.cfg, f = { len: t.age, n: live.length, T: t.T };
    const fails = [];                       // every failing test (diagnostics); reason = first
    if (t.age < this.MIN_TRACK) fails.push(`short:${t.age}`);
    if (t.T < this.ff.TAU) fails.push(`weak:T=${t.T.toFixed(0)}<${this.ff.TAU.toFixed(0)}`);
    if (!live.length) return { ok: false, reason: fails[0] || 'weak', feat: f, fails };
    // continuity: fundamental present in most frames
    f.pres = recent.filter(h => h && h.w1 >= 2).length / recent.length;
    if (f.pres < P.PRES) fails.push(`gaps:${f.pres.toFixed(2)}`);
    // smooth pitch (median |Δf| per frame, scaled to Hz/s)
    const dfs = [];
    for (let i = 1; i < recent.length; i++) if (recent[i] && recent[i - 1]) dfs.push(Math.abs(recent[i].f - recent[i - 1].f));
    dfs.sort((a, b) => a - b);
    f.dfRate = dfs.length ? dfs[dfs.length >> 1] / this.hopSec : 0;
    // allowance for estimator noise: median |Δf| of pure noise = 0.674·√2·σ_f
    const snrs = live.map(h => h.s1 / (h.n1 + 1e-30)).sort((a, b) => a - b), snrMed = Math.max(1, snrs[snrs.length >> 1]);
    const dfNoise = 0.954 * this.binHz * Math.sqrt(0.5 / snrMed + 3 / (snrMed * snrMed)) / this.hopSec;
    if (f.dfRate > P.DF_RATE + 2 * dfNoise) fails.push(`pitch:${f.dfRate.toFixed(0)}Hz/s`);
    // sub-harmonic evidence on many frames → harmonic of a lower source
    f.subFrac = live.filter(h => h.veto).length / live.length;
    if (f.subFrac > P.SUB_FRAC) fails.push(`sub:${f.subFrac.toFixed(2)}`);
    // comb: neighbouring harmonics of a lower-pitched source on many frames
    f.combFrac = live.filter(h => h.comb).length / live.length;
    if (f.combFrac > P.COMB_FRAC) fails.push(`harmonic-comb:${f.combFrac.toFixed(2)}`);
    // H1 dominance (raw, noise-corrected powers)
    let p1 = 0, p2 = 0, p3 = 0, nn1 = 0;
    for (const h of live) { p1 += h.s1; p2 += h.s2; p3 += h.s3; nn1 += h.n1; }
    f.dom = p1 / (p2 + 1e-30);
    if (f.dom < P.DOM) fails.push(`h2>h1:${f.dom.toFixed(2)}`);
    // vibrato / line broadening (median over frames)
    const spreads = live.map(h => h.spread).sort((a, b) => a - b);
    f.spread = spreads[spreads.length >> 1];
    if (f.spread > P.SPREAD) fails.push(`vibrato:${f.spread.toFixed(2)}`);
    // pure tone: loud H1 but H2+H3 > 25 dB down
    f.snr1 = p1 / (nn1 + 1e-30);
    f.hRatio = (p2 + p3) / (p1 + 1e-30);
    if (f.snr1 > P.PURE_SNR && f.hRatio < P.PURE_RATIO) fails.push(`pure:${(10 * Math.log10(f.hRatio + 1e-12)).toFixed(0)}dB`);
    // Steadiness — decidable only where the estimator noise is small, i.e. on
    // frames ≥ 20 dB above the local noise. Residuals are taken about a straight
    // line (slow drift / glide / decay is not "unsteadiness") and the expected
    // estimator variance is subtracted:
    //   pitch  σ_f ≈ 0.47·binHz/√SNR  (measured, parabolic interpolation)
    //   level  σ_L ≈ 4.34·√(2/SNR) dB
    // o.minSnr/o.minN: steadiness gating for the burst path's model features;
    // o.dry: evaluate only — no sticky machine flag, no notch (side effects).
    const st = this._steadiness(t.hist, o.minSnr, o.minN);
    if (st) Object.assign(f, st);
    if (t.machine) fails.push('machine:sticky');
    else if (st) {
      const steadyPitch = st.ubf < P.STEADY_PITCH;
      if (steadyPitch && st.ubl < P.STEADY_LEVEL_MACHINE) {
        if (!o.dry) {
          t.machine = true;                       // confidently steady: a machine for good
          this.ff.notch(t.k, 10);
        }
        fails.push(`machine:${(st.ubf * 1e4).toFixed(1)}bp,${st.ubl.toFixed(2)}dB`);
      }
      if (steadyPitch && st.slope < P.DECAY_DB_S) fails.push(`machine-decay:${st.slope.toFixed(0)}dB/s`);
      if (st.ubl < P.STEADY_LEVEL) fails.push(`machine-level:${st.ubl.toFixed(2)}dB`);
    }
    // Positive evidence of flight modulation is required before voting.
    if (!st || (st.zf < P.Z_MOD && st.zl < P.Z_MOD)) fails.push(`steady:${st ? Math.max(st.zf, st.zl).toFixed(1) : 'n/a'}`);
    return { ok: fails.length === 0, reason: fails[0] || '', feat: f, fails };
  }

  // Flight modulation vs estimator noise over the track's history (≤ 4 s).
  // Pitch (ln f) and level (dB) are detrended by a straight line (slow drift,
  // glides and decays are not "modulation") and compared with the estimator
  // variance a perfectly steady tone would show at the same SNR, measured by
  // tools/eval/calibrate-noise.js (snr = s1/n1, 3-bin signal over per-bin noise):
  //   var(f)  ≈ binHz²·(0.5/snr + 3/snr²)        var(level) ≈ 62/(snr + 3) dB²
  // z  = (Σr² − E₀[Σr²]) / √(2Σσ⁴): significance of modulation (H0: steady)
  // ub = one-sided upper bound (≈97.5 %) on the modulation s.d.
  _steadiness(hist, minSnr = this.cfg.STEADY_MIN_SNR, minN = this.cfg.STEADY_MIN_N) {
    const xs = [], lf = [], lv = [], vf = [], vl = [];
    for (let i = 0; i < hist.length; i++) {
      const h = hist[i];
      if (!h) continue;
      const snr = h.s1 / (h.n1 + 1e-30);
      if (snr < minSnr) continue;                  // below ≈ 8 the f0 estimate has outliers (not Gaussian)
      xs.push(i); lf.push(Math.log(h.f)); lv.push(h.lv);
      vf.push((this.binHz / h.f) ** 2 * (0.5 / snr + 3 / (snr * snr))); vl.push(62 / (snr + 3));
    }
    const n = xs.length;
    if (n < minN) return null;
    const one = (ys, vs) => {
      let sx = 0, sy = 0, sxx = 0, sxy = 0;
      for (let i = 0; i < n; i++) { sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i]; }
      const den = n * sxx - sx * sx, b = den > 0 ? (n * sxy - sx * sy) / den : 0, a = (sy - b * sx) / n;
      let r2 = 0, v = 0, v2 = 0;
      for (let i = 0; i < n; i++) { r2 += (ys[i] - a - b * xs[i]) ** 2; v += vs[i]; v2 += vs[i] * vs[i]; }
      const e0 = v * (n - 2) / n, sd0 = Math.sqrt(2 * v2);
      return { slope: b, ex: Math.sqrt(Math.max(0, (r2 - e0) / n)), z: (r2 - e0) / sd0,
               ub: Math.sqrt(Math.max(0, (r2 - e0 + 2 * sd0) / n)) };
    };
    const pf = one(lf, vf), pl = one(lv, vl);
    return { n, pitchEx: pf.ex, levelEx: pl.ex, zf: pf.z, zl: pl.z, ubf: pf.ub, ubl: pl.ub, slope: pl.slope / this.hopSec };
  }

  // Mean f0 of the track, SNR-weighted over its live history.
  _trackF0(t) {
    let num = 0, den = 0;
    for (const h of t.hist) if (h) { const wt = Math.max(0.1, h.w1 - 1); num += wt * h.f; den += wt; }
    return den > 0 ? num / den : NaN;
  }

  analyze(rawFreqData, timeData) {
    if (!rawFreqData || !rawFreqData.length) return [];
    this._configure(rawFreqData.length);
    this.lastNoiseFloor = this.estimateNoiseFloor(rawFreqData);
    this.ambientRMS     = this.computeAmbientRMS(rawFreqData);

    const fr = this.ff.process(rawFreqData);
    if (!this._busy) this._busy = new Map();
    const busyHorizon = this.ff.frames - Math.round(30 / this.hopSec);
    for (const [id, f] of this._busy) if (f < busyHorizon) this._busy.delete(id);
    if (this._fastCalib) this._calFrames = (this._calFrames || 0) + 1;
    const gated = this.ambientRMS > this.AMBIENT_GATE_DB;
    const calibrating = this._fastCalib || !fr || fr.calibrating;
    const ctx = { disabled: this._disabled, lat: this._lat, lng: this._lng, tod: this._timeOfDay() };
    const tracks = this.ff.tracks;
    const V = this.VOTE_FRAMES;
    const out = [];
    let best = null;

    // Harmonic relations between simultaneous tracks: a track at ≈2× or 3× a
    // stronger track's f0 is that source's harmonic, not a second insect.
    const f0s = tracks.map(t => { const h = t.hist[t.hist.length - 1]; return h ? h.f : t.k * this.binHz; });

    for (let i = 0; i < tracks.length; i++) {
      const t = tracks[i];
      let res;
      if (calibrating) res = { ok: false, reason: 'calibrating', feat: {} };
      else if (gated) res = { ok: false, reason: 'loud', feat: {} };
      else {
        res = { ok: false, reason: '', feat: {} };
        let harm = false;
        for (let j = 0; j < tracks.length; j++) {
          if (j === i || tracks[j].T < 0.3 * t.T) continue;
          const r = f0s[i] / f0s[j];
          if (Math.abs(r - 2) < 0.05 || Math.abs(r - 3) < 0.07) harm = true;
        }
        res = harm ? { ok: false, reason: 'harmonic', feat: {} } : this._plausibility(t);
      }
      // species plausibility (only for otherwise valid tracks: costs a posterior)
      let post = null;
      const f0 = this._trackF0(t);
      if (res.ok) {
        post = this.model.posterior(f0, ctx);
        if (!post || !post.post.length || post.zMin > 3.5) res = { ok: false, reason: `range:${f0.toFixed(0)}Hz`, feat: res.feat };
      }
      // Burst path: score each sufficiently strong short track once.
      if (this.cfg.BURST_K > 0 && !calibrating && !gated && !t.burst && res.reason !== 'harmonic' &&
          t.age >= this.BURST_MIN_AGE && t.T >= this.ff.TAU) {
        const pb = this._plausibility(t, { minSnr: this.cfg.BURST_MIN_SNR, minN: this.cfg.BURST_MIN_N, dry: true });
        const h = t.hist[t.hist.length - 1], fNow = h ? h.f : f0;
        // Intermittency (decided once, when the track first becomes eligible): a
        // burst is a NEW episode only if no other strong track held this pitch in
        // the 0.3 s before this track was born. A continuous tone that the
        // tracker fragments into successive tracks is not a series of passes.
        if (t.burstCont === undefined) {
          const birth = this.ff.frames - t.age, gap = Math.round(this.cfg.BURST_GAP_S / this.hopSec);
          t.burstCont = gap > 0 && this._toneLog.some(e => e.fr >= birth - gap && e.fr <= birth &&
            e.items.some(q => q.id !== t.id && Math.abs(q.f - fNow) <= this.cfg.BURST_TOL * fNow));
        }
        if (t.burstCont) t.burst = true;
        else if (!(pb.fails || []).some(q => /^(sub|harmonic-comb|machine|pure)/.test(q))) {
          // Scored on every frame until it qualifies (evidence grows as it lives).
          const pr = this._plausP(pb.feat);
          if (pr >= this.cfg.BURST_THETA) { t.burst = true; this._burstList.push({ fr: this.ff.frames, f: fNow, p: pr }); }
        }
      }
      t.votes.push(res.ok ? 1 : 0);
      if (t.votes.length > V) t.votes.shift();
      const vf = t.votes.reduce((s, x) => s + x, 0) / V;
      t.voteFraction = vf; t.reason = res.reason; t.feat = res.feat; t.fails = res.fails;
      // Acoustic "busyness": strong tonal tracks judged NOT to be a mosquito
      // (voices, music, machines) seen in the last 30 s — context for confidence.
      if (!calibrating && !res.ok && !t.locked && t.T >= this.ff.TAU && t.age >= this.MIN_TRACK) this._busy.set(t.id, this.ff.frames);
      if (!t.locked && vf >= this.cfg.VOTE_ON && res.ok) t.locked = true;
      else if (t.locked && vf < this.cfg.VOTE_OFF) t.locked = false;
      if (calibrating) t.locked = false;
      if (t.locked) {
        t.lockedFrames++;
        t.busy = 0; for (const id of this._busy.keys()) if (id !== t.id) t.busy++;
        if (!post) post = this.model.posterior(f0, ctx);
        // A lock must still pass the species-range gate: if the enabled set
        // changes under a locked track (user toggle, first GPS fix), a 617 Hz
        // tone must not be reported as the only species left enabled (290 Hz).
        if (post && post.post.length && post.zMin <= 3.5) out.push(this._detection(t, f0, post));
        else t.locked = false;
      }
      if (!best || vf > best.voteFraction || (vf === best.voteFraction && t.T > best.T)) best = t;
    }

    if (this.cfg.BURST_K > 0) {
      // Pitch occupancy of strong tracks over the last ~1 s (intermittency test).
      this._toneLog.push({ fr: this.ff.frames, items: tracks.filter(q => q.T >= 0.5 * this.ff.TAU)
        .map(q => { const h = q.hist[q.hist.length - 1]; return { id: q.id, f: h ? h.f : q.k * this.binHz }; }) });
      const keep = Math.round(1 / this.hopSec);
      if (this._toneLog.length > keep) this._toneLog.splice(0, this._toneLog.length - keep);
      if (!calibrating) {
        const bd = this._burstDetection(ctx, out);
        if (bd) out.push(bd);
      }
    }

    // Legacy per-species levels: mosquito vote level × species probability.
    this._best = best;
    let bestPost = null;
    if (best && best.voteFraction > 0) bestPost = this.model.posterior(this._trackF0(best), ctx);
    const pmap = {};
    if (bestPost) for (const o of bestPost.post) pmap[o.id] = o.p;
    for (const sp of SPECIES_DB) {
      const lvl = best ? (best.voteFraction || 0) * (pmap[sp.id] || 0) : 0;
      this.voteCount[sp.id] = Math.round(lvl * V);
      this.lastRejection[sp.id] = calibrating ? 'calibrating' : gated ? `loud:${this.ambientRMS.toFixed(0)}dB`
        : (best && pmap[sp.id] > 0.05 && best.reason) ? best.reason : (best ? '' : 'weak');
    }
    this._bestPost = bestPost;
    this.lastEvidence = best ? { trackId: best.id, T: best.T, tau: this.ff.TAU, voteFraction: best.voteFraction,
      reason: best.reason, feat: best.feat, f0: this._trackF0(best), locked: best.locked } : null;

    out.sort((a, b) => b.confidence - a.confidence);
    this._lastDetections = out;
    return out;
  }

  _detection(t, f0, post) {
    const top = post.post[0], sp = top.sp;
    const recent = t.hist.slice(-this.WIN).filter(Boolean);
    let s1 = 0, n1 = 0;
    for (const h of recent) { s1 += h.s1; n1 += h.n1; }
    const snrLinear = s1 / (n1 + 1e-30);
    const confidence = this._confidence(t);
    // Taxonomic level: 'species' when one species is confident enough,
    // otherwise 'genus' — and then Detection.species is the most probable
    // species WITHIN the most probable genus, so species.genus === genus and
    // the UI can show "Aedes sp." with the posterior candidates.
    const gTop = post.genus[0];
    const taxonLevel = top.p >= this.cfg.SPECIES_P ? 'species' : 'genus';
    let spOut = sp;
    if (taxonLevel === 'genus' && gTop && gTop.genus !== sp.genus) spOut = post.post.find(o => o.sp.genus === gTop.genus).sp;
    const genusP = (post.genus.find(g => g.genus === spOut.genus) || { p: 0 }).p;
    return {
      species: spOut, genus: spOut.genus, frequency: f0, f0,
      snr: 10 * Math.log10(Math.max(1, snrLinear)), snrLinear,
      confidence, voteFraction: t.voteFraction,
      range: snrLinear >= 100 ? 'near' : 'far',
      posterior: post.post.filter(o => o.p >= 0.01).map(o => ({ id: o.id, p: o.p })),
      genusPosterior: post.genus.filter(g => g.p >= 0.01).map(g => ({ genus: g.genus, p: g.p })),
      taxonLevel, genusP, trackId: t.id,
      // P(the insect is male | f0): males do not bite; the UI says so.
      maleP: Number.isFinite(post.maleP) ? post.maleP : 0,
    };
  }

  // Learned flight-plausibility probability (PLAUS_MODEL) of a feature set.
  _plausP(feat) {
    const m = this.cfg.PLAUS_MODEL, x = MosquitoClassifier.plausVector(feat || {});
    let z = m.w[0];
    for (let j = 0; j < x.length; j++) z += m.w[j + 1] * (x[j] - m.mu[j]) / m.sd[j];
    return 1 / (1 + Math.exp(-z));
  }

  // Burst accumulation: K bursts within W whose f0 agree within ±TOL fire a
  // detection that is held for BURST_HOLD_S after the last agreeing burst.
  // Not emitted when the lock path already reports the same insect.
  _burstDetection(ctx, out) {
    const now = this.ff.frames, W = Math.round(this.cfg.BURST_W_S / this.hopSec);
    const L = this._burstList;
    while (L.length && now - L[0].fr > W) L.shift();
    if (L.length > 64) L.splice(0, L.length - 64);
    const last = L[L.length - 1];
    if (last && last.fr === now) {
      const agree = L.filter(b => Math.abs(b.f - last.f) <= this.cfg.BURST_TOL * last.f);
      if (agree.length >= this.cfg.BURST_K) {
        const fs = agree.map(b => b.f).sort((a, b) => a - b), f0 = fs[fs.length >> 1];
        const pMean = agree.reduce((s, b) => s + b.p, 0) / agree.length;
        const bl = this._burstLock;
        if (bl && Math.abs(bl.f - f0) <= 0.1 * f0 && now <= bl.until + W) Object.assign(bl, { f: f0, p: Math.max(bl.p, pMean), n: agree.length });
        else this._burstLock = { id: `b${++this._burstSeq}`, f: f0, p: pMean, n: agree.length, first: now };
        this._burstLock.until = now + Math.round(this.cfg.BURST_HOLD_S / this.hopSec);
      }
    }
    const bl = this._burstLock;
    if (!bl || now > bl.until) return null;
    if (out.some(d => Math.abs(d.f0 - bl.f) <= 0.1 * bl.f)) return null;   // the lock path has it
    const post = this.model.posterior(bl.f, ctx);
    if (!post || !post.post.length || post.zMin > 3.5) return null;
    // Pseudo-track for _detection(): f0 history = the cluster's f0.
    const pseudo = { id: bl.id, hist: [], voteFraction: 1, T: this.ff.TAU, feat: {}, lockedFrames: now - bl.first, busy: 0 };
    const d = this._detection(pseudo, bl.f, post);
    // Burst confidence: mean model score of the agreeing bursts, discounted.
    // NOT calibrated like the lock path's (too little labelled real data yet).
    d.confidence = Math.max(0.01, Math.min(0.9, 0.9 * bl.p));
    d.range = 'far'; d.mode = 'burst'; d.snr = 0; d.snrLinear = 1; d.voteFraction = Math.min(1, bl.n / this.cfg.BURST_K);
    return d;
  }

  // Calibrated P(real mosquito | track evidence): logistic model on the
  // track's evidence at this frame, fitted on the DEV split of tools/eval
  // (tools/eval/fit-confidence.js; reliability in docs/EVALUATION.md).
  _confidence(t) {
    const x = MosquitoClassifier.confFeatures(t, this.ff.TAU), w = this.cfg.CONF;
    let z = w[0];
    for (let i = 0; i < x.length; i++) z += w[i + 1] * x[i];
    return Math.max(0.01, Math.min(0.99, 1 / (1 + Math.exp(-z))));
  }
  static confFeatures(t, tau) {
    const f = t.feat || {};
    return [
      Math.log(Math.max(1e-3, t.T / tau)),                       // CUSUM margin over threshold
      10 * Math.log10(Math.max(1, f.snr1 || 1)) / 10,              // in-bin SNR (dB/10)
      Math.log1p(Math.max(0, f.zf || 0, f.zl || 0)) / 3,          // flight-modulation evidence
      Math.log1p(t.busy || 0),                                     // rejected strong tones in the last 30 s
      Math.log10(Math.max(1e-3, f.hRatio || 1e-3)),                // harmonic richness (H2+H3)/H1
    ];
  }

  getState() {
    const dets = this._lastDetections || [];
    const st = this.getStatus();
    const perSpecies = {};
    const raw = this.getRawConfidences();
    for (const sp of SPECIES_DB) {
      const off = this._disabled.has(sp.id);
      const det = dets.find(d => d.species.id === sp.id);
      const level = off ? 0 : raw[sp.id];
      const status = off ? 'off' : det ? 'locked' : level > 0.05 ? 'building' : 'idle';
      const r = this.lastRejection[sp.id] || '';
      perSpecies[sp.id] = { level, status, reason: status === 'idle' ? MosquitoClassifier.humanReason(r) : '', debug: r };
    }
    let candidate = null;
    const b = this._best;
    if (b && b.voteFraction > 0 && this._bestPost && this._bestPost.post.length) {
      candidate = { speciesId: this._bestPost.post[0].id, level: b.voteFraction, f0: this._trackF0(b) };
    }
    const phase = this._fastCalib ? 'calibrating'
                : st.gated        ? 'gated'
                : dets.length     ? 'detected'
                : candidate && candidate.level >= 0.25 ? 'candidate'
                : 'listening';
    const band = this._bandNoiseDb();
    const noiseMarginDb = band === null ? null : this.cfg.REF_MOSQ_PEAK_DB - (band + this.cfg.LOCK_INBIN_SNR_DB);
    return { phase, noiseFloorDb: st.noiseFloor, ambientDb: st.ambientRMS, candidate, detections: dets, perSpecies,
             // extensions
             calibration: this._fastCalib ? Math.min(1, (this._calFrames || 0) * this.hopSec / this.cfg.CAL_EXPECTED_S) : 1,
             ambientGateDb: this.AMBIENT_GATE_DB,
             bandNoiseDb: band, noiseMarginDb,
             noisy: noiseMarginDb !== null && noiseMarginDb < 0 };
  }

  // Median per-bin noise floor over 300-800 Hz in AnalyserNode dB (incl. the
  // fast local normaliser), or null before the first frame.
  _bandNoiseDb() {
    const ff = this.ff;
    if (!ff || !ff.frames) return null;
    const lo = Math.max(ff.kMin, Math.round(300 / this.binHz)), hi = Math.min(ff.kMax, Math.round(800 / this.binHz));
    const v = [];
    for (let k = lo; k <= hi; k++) v.push(ff.noise[k] * (ff.Z[k] || 1));
    v.sort((a, b) => a - b);
    return 10 * Math.log10(v[v.length >> 1] + 1e-30);
  }

  estimateNoiseFloor(data) {
    const lo = Math.max(1, Math.round(80 / this.binHz));
    const hi = Math.min(Math.round(6000 / this.binHz), data.length - 1);
    const slice = [];
    for (let i = lo; i <= hi; i += 2) if (Number.isFinite(data[i]) && data[i] > -130) slice.push(data[i]);
    if (!slice.length) return -70;
    slice.sort((a, b) => a - b);
    return slice[Math.floor(slice.length * 0.15)];
  }

  computeAmbientRMS(freqData) {
    const lo = Math.round(200 / this.binHz);
    const hi = Math.min(Math.round(4000 / this.binHz), freqData.length - 1);
    let sumLin = 0, count = 0;
    for (let i = lo; i <= hi; i++) if (Number.isFinite(freqData[i])) { sumLin += Math.pow(10, freqData[i] / 10); count++; }
    return count > 0 ? 10 * Math.log10(sumLin / count) : -80;
  }

  getStatus() {
    return { noiseFloor: this.lastNoiseFloor, ambientRMS: this.ambientRMS,
             gated: this.ambientRMS > this.AMBIENT_GATE_DB, snrMin: this.ff ? this.ff.TAU : 0 };
  }

  getRawConfidences() {
    const result = {};
    SPECIES_DB.forEach(sp => { result[sp.id] = this._disabled.has(sp.id) ? 0 : Math.min(1, this.voteCount[sp.id] / this.VOTE_FRAMES); });
    return result;
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { FarFieldTracker, SpeciesModel, MosquitoClassifier, MN_DSP };
}
