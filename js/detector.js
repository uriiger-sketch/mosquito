/* MosquitoNet — acoustic detection engine (pure DSP, no DOM).
 *
 * Runs unchanged in the browser (classic script) and in Node (tests/, tools/)
 * via tools/lib/load-engine.js, which evaluates js/species.js + this file in
 * one vm context — the same shared-global environment the page provides.
 *
 * ══ ENGINE ⇄ UI CONTRACT (stable; extend, never break) ══════════════════
 *
 *  const clf = new MosquitoClassifier(engine)   // engine: { binHz, freqToBin(hz), sampleRate? }
 *  clf.setSensitivity(level)                    // 1 (most sensitive) … 5 (strictest)
 *  clf.setLocation(lat, lng)                    // null/null when unknown
 *  clf.setDisabledSpecies(set)                  // Set of species ids to ignore
 *  clf.beginCalibration() / clf.endCalibration()// fast noise learning, then clean vote buffers
 *  clf.analyze(rawFreqDb, timeSamples) → Detection[]   // call once per hop (~4096 samples)
 *  clf.getState() → EngineState                 // cheap snapshot for the UI, any time
 *  clf.getStatus(), clf.getRawConfidences()     // legacy, kept for compatibility
 *
 *  Detection = {
 *    species:     SPECIES_DB entry (most probable species),
 *    genus:       'Anopheles' | 'Aedes' | 'Culex' | … (from species.genus),
 *    frequency:   measured fundamental f0 in Hz (alias: f0),
 *    f0:          same as frequency,
 *    snr:         dB, snrLinear: linear power ratio,
 *    confidence:  0…1 (calibrated: P(real mosquito | evidence) as well as we can),
 *    voteFraction:0…1,
 *    range:       'near' | 'far',
 *    posterior:   [{ id, p }]  species probabilities, sorted desc, Σp ≤ 1,
 *  }
 *
 *  EngineState = {
 *    phase:   'idle' | 'calibrating' | 'gated' | 'listening' | 'candidate' | 'detected',
 *    noiseFloorDb, ambientDb,
 *    candidate: { speciesId, level 0…1, f0 } | null,  // strongest evidence (even pre-lock)
 *    detections: Detection[] (last analyze() result),
 *    perSpecies: { [id]: { level 0…1, status: 'off'|'idle'|'building'|'locked',
 *                          reason: short human-readable text, debug: raw gate string } },
 *  }
 * ═══════════════════════════════════════════════════════════════════════
 */
'use strict';
/* global SPECIES_DB, timeOfDayCategory */
// ═══════════════════════════════════════════════════════════════════
//  MosquitoClassifier — research-grade, false-positive hardened
//
//  Key science:
//  • Uses RAW (unsmoothed) FFT frames so no smoothing artifact
//  • Frequency stability gate: peak must stay within ±25 Hz
//    across 60% of temporal frames — eliminates all transient tones
//  • Ambient gate raised to -52 dBFS average (eliminates most rooms)
//  • Q-factor gate 18 dB (narrower than before)
//  • Harmonic ratio 6–20 dB below fundamental (biology-based)
//  • 50 frames (~2 s) temporal averaging before reporting
//  • Detection only reported if BOTH harmonic structure AND
//    frequency stability pass simultaneously
//
//  Sources: Pantula et al. 2021 (Sci Rep), Brogdon & McAllister 1998,
//           Rompotis et al. 2015, Moore et al. 2018 (PLOS CB)
// ═══════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════
//  FarFieldTracker — long-range harmonic track-before-detect
//
//  Physics: beyond the acoustic near field (λ/2π ≈ 8-14 cm at 400-700 Hz)
//  a flight tone's pressure falls as 1/r (-6 dB per doubling of distance)
//  and air absorption is negligible, so range is set purely by SNR: every
//  6 dB of processing gain doubles the distance at which a mosquito can be
//  heard. The narrowband FFT already buys ~36 dB over the full audio band;
//  this tracker keeps that gain instead of discarding it, and adds more:
//
//   1. Whitening — each bin is divided by its own slowly-adapting noise
//      floor, so evidence is "times above local noise" even in coloured
//      (fan / traffic / pink) noise.
//   2. Harmonic summation — evidence from f0, 2f0 and 3f0 is combined.
//   3. Track-before-detect — a Viterbi/CUSUM recursion accumulates evidence
//      along a frequency path allowed to drift ±1 bin per frame (Doppler,
//      manoeuvring), so ~1.5 s of weak frames add up (the sonar LOFAR
//      line-tracking technique) rather than each frame having to win alone.
//
//  False-alarm control (the whole point — range is worthless if it cries
//  wolf): a sub-harmonic veto (voices and motors at f0/2 or f0/3 also put
//  energy on a candidate's harmonics; a mosquito has nothing below f0), a
//  stationarity test (motors are rock-steady in pitch and level; a flying
//  mosquito wanders), broadband blanking (clicks, bangs, speech onsets), a
//  minimum track length, and a CFAR threshold calibrated on noise-only
//  audio by tools/detection_bench.js.
// ═══════════════════════════════════════════════════════════════════
class FarFieldTracker {
  constructor(binHz) {
    this.binHz  = binHz;
    // f0 search band 280-800 Hz. Adult voice fundamentals sit below ~260 Hz;
    // above 280 Hz a voice only appears through harmonics, which the
    // sub-harmonic veto removes. (Lowest target species: ~290 Hz.)
    this.kLo    = Math.round(280 / binHz);
    this.kHi    = Math.round(800 / binHz);
    this.kMin   = Math.max(2, Math.floor(this.kLo / 3) - 2);  // room for f0/3 veto
    this.kMax   = 3 * this.kHi + 2;                  // room for 3f0
    this.WH     = [1.0, 0.7, 0.5];                   // harmonic weights (H1 dominates)
    // Expected harmonic sum on pure noise: W[k] ~ Exp(1), max of 3 ~ mean 1.833.
    this.BIAS   = (this.WH[0] * 1 + (this.WH[1] + this.WH[2]) * 1.833) * 1.10;
    this.LAMBDA = 0.93;                              // track memory ≈ 14 frames ≈ 1.3 s
    this.TAU    = 30;                                // CFAR threshold (calibrated; see bench)
    this.MIN_TRACK = 8;                              // frames (~0.75 s) a track must persist
    this.CAL_FRAMES = 20;
    this.reset();
  }

  reset() {
    const nb = this.kMax + 2;
    this.noise  = new Float64Array(nb);
    this.W      = new Float64Array(nb);
    this.P      = new Float64Array(nb);             // raw power (for physical ratios)
    this.S      = new Float64Array(this.kHi + 2);
    this.S2     = new Float64Array(this.kHi + 2);
    this.hist   = [];                                // {k, f, a} per frame of the best track
    this.frames = 0;
    this._notch = null;
    this._med   = new Float64Array(3 * this.kHi - this.kLo + 1);
  }

  setSensitivity(level) {
    const scale = [0.70, 0.85, 1.0, 1.3, 1.7][Math.max(0, Math.min(4, level - 1))];
    this.TAU = 30 * scale;
  }

  _max3(k) {
    const W = this.W;
    return Math.max(W[k - 1], W[k], W[k + 1]);
  }

  process(db) {
    const { kMin, kMax, kLo, kHi, W, noise } = this;
    if (!db || db.length <= kMax + 1) return { detected: false, T: 0 };
    this.frames++;

    // ── 1. per-bin noise floor + whitening ────────────────────────────
    const cal = this.frames <= this.CAL_FRAMES;
    for (let k = kMin - 1; k <= kMax + 1; k++) {
      const v = db[k];
      const P = Number.isFinite(v) ? Math.pow(10, v / 10) : 1e-30;
      if (cal || noise[k] <= 0) {
        noise[k] += (P - noise[k]) / Math.min(this.frames, this.CAL_FRAMES);
      } else {
        // Fast when the bin looks like noise, very slow when it holds a strong
        // component — so a passing mosquito is not learned as "noise", while a
        // machine that runs for minutes is eventually absorbed.
        // Strong components move the floor by at most ×(1+3/800) per frame:
        // a mosquito visit (seconds to a minute) is barely absorbed, however loud,
        // while a machine running for minutes still is. (An uncapped slow EMA
        // absorbed a fixed FRACTION per frame, collapsing any loud tone to W≈8
        // within ~10 s.) Lines identified as machines are notched out quickly.
        const nt = this._notch;
        let notched = false;
        if (nt && nt.frames > 0) {
          const h = Math.round(k / nt.k);
          notched = h >= 1 && h <= 3 && Math.abs(k - h * nt.k) <= h;
        }
        if (notched)                 noise[k] += (P - noise[k]) / 25;
        else if (P > 4 * noise[k])   noise[k] += (4 * noise[k] - noise[k]) / 800;
        else                         noise[k] += (P - noise[k]) / 100;
      }
      this.P[k] = P;
      W[k] = P / (noise[k] + 1e-30);
    }
    if (this._notch && this._notch.frames > 0) this._notch.frames--;
    if (cal) return { detected: false, T: 0, calibrating: true };

    // ── 2. broadband blanking / level normalisation ───────────────────
    const md = this._med; let n = 0;
    for (let k = kLo; k <= 3 * kHi; k++) md[n++] = W[k];
    const sorted = md.subarray(0, n).sort();
    const norm = sorted[n >> 1] / 0.693;             // median of Exp(1) is ln 2
    const S = this.S, S2 = this.S2, L = this.LAMBDA;
    if (norm > 4) {                                  // click / bang / broadband burst
      for (let k = kLo; k <= kHi; k++) S[k] *= L;
      this.hist.push(null); if (this.hist.length > 24) this.hist.shift();
      return { detected: false, T: 0, blanked: true };
    }
    if (norm > 1) for (let k = kMin - 1; k <= kMax + 1; k++) W[k] /= norm;

    // ── 3. harmonic sum + sub-harmonic veto + Viterbi/CUSUM update ────
    let T = 0, kBest = -1;
    for (let k = kLo; k <= kHi; k++) {
      let hs = this.WH[0] * W[k] + this.WH[1] * this._max3(2 * k) + this.WH[2] * this._max3(3 * k);
      // A lower-pitched source (voice, motor) whose harmonics land on k, 2k, 3k
      // also has energy at k/2 & 3k/2, or k/3 & 2k/3. A mosquito does not.
      const s2 = this._max3(Math.round(k / 2)), s32 = this._max3(Math.round(1.5 * k));
      const s3 = this._max3(Math.round(k / 3)), s23 = this._max3(Math.round(2 * k / 3));
      if ((s2 > 8 && s32 > 8) || (s3 > 8 && s23 > 8)) hs = 0;
      const prev = Math.max(S[k - 1] || 0, S[k], S[k + 1] || 0);
      const v = Math.max(0, L * prev + hs - this.BIAS);
      S2[k] = v;
      if (v > T) { T = v; kBest = k; }
    }
    for (let k = kLo; k <= kHi; k++) S[k] = S2[k];

    // ── 4. track bookkeeping: sub-bin f0 (parabolic, in dB) + amplitude ──
    let f = kBest * this.binHz;
    if (kBest > 0) {
      const a = db[kBest - 1], b = db[kBest], c = db[kBest + 1];
      if (Number.isFinite(a) && Number.isFinite(b) && Number.isFinite(c)) {
        const den = a - 2 * b + c;
        if (den < 0) f = (kBest + Math.max(-0.5, Math.min(0.5, 0.5 * (a - c) / den))) * this.binHz;
      }
    }
    // Per-frame features of the candidate: fundamental energy (mainlobe spans
    // ±1 bin) and 2nd-harmonic energy.
    const Pm = (k) => Math.max(this.P[k - 1], this.P[k], this.P[k + 1]);
    this.hist.push(kBest > 0 ? { k: kBest, f, a: this._max3(kBest),
                                 p1: Pm(kBest), p2: Pm(2 * kBest) } : null);
    if (this.hist.length > 24) this.hist.shift();

    // contiguous track: recent frames whose best bin stayed within ±2 bins
    const tr = [];
    for (let i = this.hist.length - 1; i >= 0 && tr.length < 16; i--) {
      const h = this.hist[i];
      if (!h || Math.abs(h.k - kBest) > 2) break;
      tr.push(h);
    }
    tr.reverse();

    // ── 5. is this track physically a free-flying mosquito? ───────────
    // Measured on synthetic adversaries (tools/detection_bench.js):
    //  • continuity — a wingbeat tone is continuous; voices come in syllables,
    //    so the track only survives their gaps through memory, with the
    //    fundamental bin empty. Require the fundamental in most frames.
    //  • smooth pitch — manoeuvres and Doppler move a mosquito's pitch by a few
    //    Hz per second; speech intonation glides several Hz per frame.
    //  • H1 dominance — the flight tone's fundamental is its strongest partial.
    //  • not steady — even a constant wingbeat carries ±~1 Hz of Doppler as the
    //    insect circles (v/c·f0 ≈ 1/343 × 400 Hz); a motor or fan is steady to
    //    a fraction of a Hz. Only testable once the pitch estimate is precise.
    let machine = false, fStd = 0, pres = 0, dfMed = 0, dom = 0, ma = 0;
    if (tr.length >= this.MIN_TRACK) {
      const fs = tr.map(h => h.f);
      const mf = fs.reduce((s, x) => s + x, 0) / fs.length;
      fStd = Math.sqrt(fs.reduce((s, x) => s + (x - mf) ** 2, 0) / fs.length);
      ma   = tr.reduce((s, h) => s + h.a, 0) / tr.length;
      // Dominance is a property of the SOURCE, so compare raw powers.
      const p1 = tr.reduce((s, h) => s + h.p1, 0), p2 = tr.reduce((s, h) => s + h.p2, 0);
      dom  = p1 / (p2 || 1e-30);
      pres = tr.filter(h => h.a >= 2.0).length / tr.length;
      const dfs = [];
      for (let i = 1; i < fs.length; i++) dfs.push(Math.abs(fs[i] - fs[i - 1]));
      dfs.sort((a, b) => a - b);
      dfMed = dfs.length ? dfs[dfs.length >> 1] : 0;
      // A steady tone still shows pitch-estimate jitter that grows as SNR falls
      // (~2.5/sqrt(SNR) Hz measured: 0.5 Hz at 14 dB per bin). Free flight
      // measured 2.3-3 Hz. Call it a machine if it is no less steady than the
      // estimator alone would make a pure tone look (with margin).
      machine = fStd < Math.max(1.0, 4.0 / Math.sqrt(Math.max(ma, 1)));
      // Machine lines are notched out: absorb them into the noise floor fast.
      if (machine) this._notch = { k: kBest, frames: 60 };
    }
    const plausible = pres >= 0.75 && dfMed <= 4.0 && dom >= 0.8 && !machine;

    const detected = kBest > 0 && T > this.TAU && tr.length >= this.MIN_TRACK && plausible;
    // Confidence grows with evidence above threshold; floor 0.55 so a lock
    // still needs a sustained track (the vote window averages frame confidences).
    const conf = detected ? Math.min(0.92, 0.55 + 0.35 * Math.log2(T / this.TAU) / 3) : 0;
    return {
      detected, T, f0: f, k: kBest, conf, machine, fStd, pres, dfMed, dom,
      trackLen: tr.length, snrLinear: kBest > 0 ? W[kBest] : 0,
    };
  }
}

class MosquitoClassifier {
  // ── DETECTION ENGINE v2 ─────────────────────────────────────────
  // Improvements over v1 (from mosquito-acoustics skill):
  //   1. Adaptive per-species noise floor (EMA per-band, not global)
  //   2. Spectral flatness / tonality gate (rejects broadband noise)
  //   3. Species disambiguation for close-frequency pairs
  //   4. User-adjustable sensitivity slider (SNR_MIN: 3–15)
  //   5. Harmonic presence BONUS: strong 2f → 30% conf boost
  //   6. Log-domain SNR → confidence with per-species SNR_MAX tuning
  //   7. Improved vote buffer: exponential decay for fast response

  constructor(engine) {
    this.engine = engine;

    // ── Tunable thresholds (user can adjust via sensitivity slider) ─
    // GOERTZEL_SNR_MIN: signal power must exceed per-band noise by this ratio
    // Higher = fewer false positives (harder to trigger)
    // Lower  = more sensitive (easier to trigger, more false positives)
    this.GOERTZEL_SNR_MIN     = 5.0;   // Tightened: SNR 5× above background (was 3.5 — too many FP)
    this.CONFIDENCE_THRESHOLD = 0.45;  // min avg conf in vote window (was 0.38)
    this.AMBIENT_GATE_DB      = -18;   // broadband gate (was -15 — stricter room noise floor)
    this.TONALITY_MIN         = 2.2;   // tonality ratio gate (was 1.8 — too permissive for fan noise)
    this.HARMONIC_REQUIRED    = false; // hard gate for 2f/3f presence (only enforced at levels 4–5)
    this.HARMONIC_SNR_RATIO   = 0.10;  // min H2/H1 power ratio (HumBugDB: Anopheles 10–65%)
    this.FLATNESS_GATE        = 0.88;  // reject flatter spectra (was 0.92 — too lenient)
    this.AUTOCORR_MIN         = 0.10;  // min normalized autocorrelation at lag=SR/f₀ (Mukundarajan 2017)

    // ── Vote buffer (rolling window for sustained signal) ──────────
    // VOTE_FRAMES: how many frames to look back (~3s at 25fps)
    // Increase → more robust, slower to lock
    // Decrease → faster to lock, more false positives
    this.VOTE_FRAMES    = 22;    // ~2s at 10.75 analyses/s (decimated, was 64 at 60fps)
    this.VOTE_THRESHOLD = 0.62;  // 62% of frames must pass all gates (was 0.55)

    this.voteBuffer  = {};
    this.votePtr     = {};
    this.voteCount   = {};
    SPECIES_DB.forEach(sp => {
      this.voteBuffer[sp.id] = new Uint8Array(this.VOTE_FRAMES);
      this.votePtr[sp.id]    = 0;
      this.voteCount[sp.id]  = 0;
    });

    // ── Confidence accumulator (for live charge-bar display) ───────
    // CONF_FRAMES MUST equal VOTE_FRAMES so avgConf averages exactly the
    // same window the vote gate evaluates — otherwise the two windows wrap
    // at different points and the acceptance decision is inconsistent.
    this.CONF_FRAMES = this.VOTE_FRAMES;
    this.confBuffer  = {};
    SPECIES_DB.forEach(sp => {
      this.confBuffer[sp.id] = new Float32Array(this.CONF_FRAMES);
    });
    this.confIdx = 0;

    // ── Adaptive per-species noise floor (KEY IMPROVEMENT) ─────────
    // Instead of a global noise floor, each species band has its own
    // noise estimate. Updated via EMA during frames where SNR is LOW
    // (signal absent). This adapts to: fan noise, AC hum, speech, etc.
    // Source: Boll 1979 spectral subtraction, adapted for Goertzel.
    this.NOISE_EMA_ALPHA   = 0.97;   // slow EMA (adapts over ~30s)
    this.OVERSUBTRACTION   = 1.3;    // Boll oversubtraction factor
    this.adaptiveNoise     = {};     // per-species noise estimate
    SPECIES_DB.forEach(sp => {
      this.adaptiveNoise[sp.id] = 1e-6;  // initialise low
    });

    // ── Species disambiguation table (close-frequency pairs) ───────
    // When two species both pass the SNR gate simultaneously,
    // suppress the weaker one unless they're within 2 dB of each other.
    // Pairs separated by < 55 Hz can bleed into each other's band.
    this.DISAMBIGUATION_PAIRS = [
      ['anopheles', 'anopheles_stephensi'],        // 406 vs 388 Hz — 18 Hz
      ['aedes_aegypti', 'culex_pipiens'],          // 617 vs 594 Hz — 23 Hz
      ['aedes_vexans', 'ochlerotatus_caspius'],    // 515 vs 548 Hz — 33 Hz
      ['ochlerotatus_caspius', 'mansonia_uniformis'], // 548 vs 570 Hz — 22 Hz
      ['culex_pipiens', 'mansonia_uniformis'],     // 594 vs 570 Hz — 24 Hz
      ['aedes_aegypti', 'aedes_albopictus'],       // 617 vs 668 Hz — 51 Hz
      ['anopheles', 'culex'],                      // 406 vs 453 Hz — 47 Hz
    ];

    // ── GPS for activity-period calculations ─────────────────────
    this._lat = null;
    this._lng = null;
    this._disabled = new Set();      // species ids the user/location excluded
    this._fastCalib = false;
    this._lastDetections = [];

    // ── Goertzel filter bank ──────────────────────────────────────
    this.goertzelParams = {};
    this._buildGoertzelParams();

    // ── Hum blacklist ─────────────────────────────────────────────
    this.humSet = new Set();
    for (let h = 1; h <= 20; h++) {
      this.humSet.add(50 * h);
      this.humSet.add(60 * h);
    }

    // ── Diagnostics ──────────────────────────────────────────────
    this.lastNoiseFloor  = -70;
    this.ambientRMS      = -80;
    this.lastRejection   = {};
    this.lastGoertzelSNR = {};
    this.lastTonality    = {};
    this.lastVia         = {};
    this.ff = new FarFieldTracker(engine && engine.binHz ? engine.binHz : 44100 / 8192);
    SPECIES_DB.forEach(sp => {
      this.lastRejection[sp.id]   = '';
      this.lastGoertzelSNR[sp.id] = 0;
      this.lastTonality[sp.id]    = 0;
    });
  }

  // ── Contract API (see header) ─────────────────────────────────────
  setLocation(lat, lng) {
    this._lat = Number.isFinite(lat) ? lat : null;
    this._lng = Number.isFinite(lng) ? lng : null;
  }

  // The UI passes its live Set; mutations made later are seen immediately.
  setDisabledSpecies(set) { this._disabled = set instanceof Set ? set : new Set(set || []); }

  // Startup calibration: learn the noise floor fast, then discard the votes
  // cast while the floor was still settling.
  beginCalibration() { this._fastCalib = true; }
  endCalibration()   { this._fastCalib = false; this.resetVotes(); }

  resetVotes() {
    SPECIES_DB.forEach(sp => {
      this.voteBuffer[sp.id].fill(0);
      this.votePtr[sp.id]   = 0;
      this.voteCount[sp.id] = 0;
      this.confBuffer[sp.id].fill(0);
    });
  }

  // Species probabilities given a measured f0: Gaussian likelihood around each
  // enabled species' centre (σ = half its range), flat prior. Σp = 1 over the
  // returned list; only species with p ≥ 0.01 are listed.
  speciesPosterior(f0) {
    if (!Number.isFinite(f0) || f0 <= 0) return [];
    const out = [];
    let z = 0;
    for (const sp of SPECIES_DB) {
      if (this._disabled.has(sp.id)) continue;
      const sigma = Math.max(8, (sp.freqRange[1] - sp.freqRange[0]) / 2);
      const l = Math.exp(-0.5 * ((f0 - sp.freq) / sigma) ** 2) / sigma;
      out.push({ id: sp.id, p: l }); z += l;
    }
    if (z <= 0) return [];
    for (const o of out) o.p /= z;
    return out.filter(o => o.p >= 0.01).sort((a, b) => b.p - a.p);
  }

  // Human-readable text for a raw gate string (raw text stays in .debug).
  static humanReason(raw) {
    if (!raw) return '';
    if (raw.startsWith('snr'))        return 'no tone';
    if (raw.startsWith('tonal'))      return 'not tonal';
    if (raw.startsWith('acorr'))      return 'not periodic';
    if (raw.startsWith('transient'))  return 'too brief';
    if (raw.startsWith('no-harmonic'))return 'no harmonics';
    if (raw.startsWith('loud'))       return 'room too loud';
    if (raw.startsWith('flat'))       return 'broadband noise';
    if (raw.startsWith('hum'))        return 'mains-hum band';
    return 'no match';
  }

  getState() {
    const dets = this._lastDetections || [];
    const st = this.getStatus();
    const perSpecies = {};
    let candidate = null;
    for (const sp of SPECIES_DB) {
      const level = this.voteCount[sp.id] / this.VOTE_FRAMES;
      const det = dets.find(d => d.species.id === sp.id);
      const off = this._disabled.has(sp.id);
      const status = off ? 'off' : det ? 'locked' : level > 0.05 ? 'building' : 'idle';
      const raw = this.lastRejection[sp.id] || '';
      perSpecies[sp.id] = { level: off ? 0 : level, status,
                            reason: status === 'idle' ? MosquitoClassifier.humanReason(raw) : '',
                            debug: raw };
      if (!off && level > 0 && (!candidate || level > candidate.level)) {
        const ff = this.lastFarField;
        const f0 = det ? det.f0
                 : (ff && ff.k > 0 && ff.f0 >= sp.freqRange[0] && ff.f0 <= sp.freqRange[1]) ? ff.f0 : null;
        candidate = { speciesId: sp.id, level, f0 };
      }
    }
    const phase = this._fastCalib ? 'calibrating'
                : st.gated        ? 'gated'
                : dets.length     ? 'detected'
                : candidate && candidate.level >= 0.25 ? 'candidate'
                : 'listening';
    return { phase, noiseFloorDb: st.noiseFloor, ambientDb: st.ambientRMS,
             candidate, detections: dets, perSpecies };
  }

  // ── Sensitivity API (called by UI slider) ─────────────────────────
  setSensitivity(level) {
    // level: 1 (most sensitive) → 5 (least sensitive / most strict)
    // Rebalanced for real-world environments: level 3 default must detect
    // a real mosquito at ~1–2 m in a typical room with HVAC background.
    const snrTable      = [2.5,  4.0,  5.0,  8.0,  14.0];
    const voteTable     = [10,   15,   22,   32,   48];   // recalibrated for ~10.75 analyses/s
    const confTable     = [0.25, 0.35, 0.45, 0.58, 0.72];
    const tonalTable    = [1.2,  1.7,  2.2,  3.0,  4.0];
    const harmonicTable = [false, false, false, true, true];
    const ambientTable  = [-8,   -14,  -18,  -25,  -35];
    const flatnessTable = [0.97, 0.93, 0.88, 0.84, 0.80];
    const autocorrTable = [0.04, 0.07, 0.10, 0.15, 0.22];
    const idx = Math.max(0, Math.min(4, level - 1));
    this.GOERTZEL_SNR_MIN     = snrTable[idx];
    this.VOTE_FRAMES          = voteTable[idx];
    this.CONFIDENCE_THRESHOLD = confTable[idx];
    this.TONALITY_MIN         = tonalTable[idx];
    this.HARMONIC_REQUIRED    = harmonicTable[idx];
    this.AMBIENT_GATE_DB      = ambientTable[idx];
    this.FLATNESS_GATE        = flatnessTable[idx];
    this.AUTOCORR_MIN         = autocorrTable[idx];
    if (this.ff) this.ff.setSensitivity(level);
    // Keep the confidence window aligned to the (new) vote window length.
    this.CONF_FRAMES = this.VOTE_FRAMES;
    this.confIdx     = 0;
    SPECIES_DB.forEach(sp => {
      this.voteBuffer[sp.id] = new Uint8Array(this.VOTE_FRAMES);
      this.votePtr[sp.id]    = 0;
      this.voteCount[sp.id]  = 0;
      this.confBuffer[sp.id] = new Float32Array(this.CONF_FRAMES);
    });
    console.log(`[Sensitivity] level=${level} SNR_MIN=${this.GOERTZEL_SNR_MIN} VOTE_FRAMES=${this.VOTE_FRAMES} tonality>=${this.TONALITY_MIN} harmonic=${this.HARMONIC_REQUIRED}`);
  }

  _buildGoertzelParams() {
    // ── Optimal Gaussian Goertzel Filter Bank ────────────────────────
    // Block = 8192 → resolution = 44100/8192 = 5.38 Hz/bin, 186ms window
    // 9 taps at exact bin multiples → zero inter-tap leakage
    // Gaussian σ = 15 Hz → response at ±5.4 Hz: 93.8%
    const BLOCK  = 8192;
    const SR     = 44100;
    const BW     = SR / BLOCK;
    const SIGMA  = 15;

    const TAP_BINS = [-4,-3,-2,-1,0,1,2,3,4];
    const rawW     = TAP_BINS.map(b => Math.exp(-((b*BW)**2)/(2*SIGMA*SIGMA)));
    const wSum     = rawW.reduce((a,b)=>a+b,0);
    const gaussW   = rawW.map(w=>w/wSum);

    const H2_BINS  = [-2,-1,0,1,2];
    const rawWh2   = H2_BINS.map(b => Math.exp(-((b*BW)**2)/(2*SIGMA*SIGMA)));
    const wSumH2   = rawWh2.reduce((a,b)=>a+b,0);
    const gaussWh2 = rawWh2.map(w=>w/wSumH2);

    const allCenters = SPECIES_DB.map(s => s.freq);
    const GUARD = 65;  // wider guard: cleaner per-species background for pairs < 70 Hz apart
    const isOccupied = hz => allCenters.some(c => Math.abs(hz-c) < GUARD);

    const makeCoeff = hz => {
      const k = Math.round(BLOCK * hz / SR);
      const w = (2 * Math.PI * k) / BLOCK;
      return { k, coeff: 2*Math.cos(w), targetHz: hz, block: BLOCK };
    };

    SPECIES_DB.forEach(sp => {
      const freq = sp.freq;
      const f0Bank = TAP_BINS.map((b,i)=>({...makeCoeff(freq+b*BW),   gaussW: gaussW[i]}));
      const f2Bank = H2_BINS.map((b,i) =>({...makeCoeff(freq*2+b*BW), gaussW: gaussWh2[i]}));
      const f3Bank = H2_BINS.map((b,i) =>({...makeCoeff(freq*3+b*BW), gaussW: gaussWh2[i]}));

      // Safe reference taps: walk outward from ±55 Hz, skip occupied frequencies
      const ref = [];
      let lo = freq-55, hi = freq+55;
      while (lo >= 200 && ref.filter(r=>r.targetHz<freq).length < 3) {
        if (!isOccupied(lo)) ref.push(makeCoeff(lo));
        lo -= BW;
      }
      while (hi <= 1200 && ref.filter(r=>r.targetHz>freq).length < 3) {
        if (!isOccupied(hi)) ref.push(makeCoeff(hi));
        hi += BW;
      }
      if (ref.length === 0) {
        ref.push(makeCoeff(200), makeCoeff(220), makeCoeff(1000), makeCoeff(1020));
      }
      this.goertzelParams[sp.id] = { f0Bank, f2Bank, f3Bank, ref };
    });
    this._goertzelBlock = BLOCK;
  }

  goertzelPower(timeSamples, param) {
    const { coeff, k, block } = param;
    let s0=0, s1=0, s2=0;
    const N = Math.min(timeSamples.length, block);
    for (let n=0; n<N; n++) {
      s0 = timeSamples[n] + coeff*s1 - s2;
      s2 = s1; s1 = s0;
    }
    return s1*s1 + s2*s2 - s1*s2*coeff;
  }

  gaussianBankPower(timeSamples, bank) {
    let pow = 0;
    for (const tap of bank) pow += tap.gaussW * this.goertzelPower(timeSamples, tap);
    return pow;
  }

  // ── Pre-emphasis filter (first-order high-pass) ──────────────────
  // x[n] - α·x[n-1] → boosts 300-1000 Hz (mosquito range) ~4-6× relative to
  // 50 Hz HVAC/wind noise. α=0.97 is standard in speech/bio-acoustic processing.
  preEmphasis(samples, alpha = 0.97) {
    const N   = samples.length;
    const out = new Float32Array(N);
    out[0] = samples[0];
    for (let i = 1; i < N; i++) out[i] = samples[i] - alpha * samples[i - 1];
    return out;
  }

  refPower(timeSamples, refTaps) {
    if (!refTaps.length) return 0;
    // Median rather than mean — one hot reference tap can't skew the background estimate
    const powers = refTaps.map(r => this.goertzelPower(timeSamples, r));
    powers.sort((a, b) => a - b);
    const m = powers.length;
    return m & 1 ? powers[m >> 1] : (powers[m / 2 - 1] + powers[m / 2]) * 0.5;
  }

  // ── Autocorrelation periodicity (Mukundarajan et al. STM 2017) ───
  // A periodic mosquito wingbeat at f₀ Hz creates R[τ₀]/R[0] ≈ cos(2π)≈1.
  // Aperiodic fan/voice/HVAC: R[τ₀]/R[0] ≈ 0.
  // Only first 2048 samples used — fast and sufficient for 20 ms periodicity.
  autocorrPeriodicity(timeSamples, freq) {
    const SR   = 44100;
    const tau0 = Math.round(SR / freq);
    const M    = Math.min(2048, timeSamples.length - tau0);
    if (M <= 0) return 0;
    let r0 = 0, rTau = 0;
    for (let n = 0; n < M; n++) {
      r0   += timeSamples[n] * timeSamples[n];
      rTau += timeSamples[n] * timeSamples[n + tau0];
    }
    return r0 > 1e-20 ? rTau / r0 : 0;
  }

  // ── Sub-frame temporal consistency ───────────────────────────────
  // Split 8192-sample block into 4×2048 sub-frames; count how many
  // show signal above threshold. Real mosquito (sustained): 3–4/4.
  // Transients (taps, brief tones, coughs): 0–1/4.
  subframeConsistency(timeSamples, freq) {
    // Split 8192-sample block into 4×2048 sub-frames.
    // Two requirements: (1) signal present in ≥3 sub-frames; (2) peak frequency
    // centroid stays within ±2 bins (~40 Hz) across sub-frames — kills swept tones,
    // voice, and other frequency-drifting interference.
    const SR = 44100, SUB = 2048;
    const BW_SUB    = SR / SUB;                         // ~21.5 Hz/bin
    const WIN_BINS  = Math.round(75 / BW_SUB);          // ±75 Hz search window
    const threshold = this.GOERTZEL_SNR_MIN * 0.5;
    const kCenter   = Math.round(SUB * freq / SR);
    const kRef1     = Math.round(SUB * (freq + 130) / SR);
    const kRef2     = Math.round(SUB * (freq - 130) / SR);
    const cR1       = 2 * Math.cos(2 * Math.PI * kRef1 / SUB);
    const cR2       = 2 * Math.cos(2 * Math.PI * kRef2 / SUB);

    let passing = 0;
    const peakBins = [];

    for (let i = 0; i < 4; i++) {
      const sub = timeSamples.subarray ? timeSamples.subarray(i * SUB, (i + 1) * SUB)
                                       : timeSamples.slice(i * SUB, (i + 1) * SUB);

      // Find highest-power bin within ±75 Hz of target (frequency centroid)
      let bestPow = 0, bestBin = kCenter;
      for (let dk = -WIN_BINS; dk <= WIN_BINS; dk++) {
        const k = kCenter + dk;
        if (k < 1 || k >= SUB / 2) continue;
        const c = 2 * Math.cos(2 * Math.PI * k / SUB);
        const p = this.goertzelPower(sub, { coeff: c, block: SUB, k });
        if (p > bestPow) { bestPow = p; bestBin = k; }
      }

      const refP = (this.goertzelPower(sub, { coeff: cR1, block: SUB, k: kRef1 }) +
                    this.goertzelPower(sub, { coeff: cR2, block: SUB, k: kRef2 })) * 0.5;

      if (bestPow / (refP + 1e-12) >= threshold) {
        passing++;
        peakBins.push(bestBin);
      }
    }

    // Centroid stability: passing sub-frames must agree on frequency within ±2 bins
    if (peakBins.length >= 2) {
      const spread = Math.max(...peakBins) - Math.min(...peakBins);
      if (spread > 2) return Math.max(0, passing - 2); // heavy penalty for frequency drift
    }

    return passing;
  }

  // ── Spectral tonality (narrow-band vs broadband) ─────────────────
  // A real mosquito produces a narrow tonal peak — high tonality.
  // Fan noise / hiss / speech are broadband — low tonality.
  // Method: ratio of peak Goertzel power to mean Goertzel power across
  // a ±100Hz window. Values > 3 indicate a tonal signal.
  tonalityScore(timeSamples, freq) {
    // Ratio of peak bin power to local background estimated over 8 bins
    // spanning ±40…±150 Hz. More background samples → more robust estimate.
    // Skip the ±20 Hz bins to avoid contaminating background with peak sidelobes.
    const SR=44100, BLOCK=8192;
    const k0 = Math.round(BLOCK * freq / SR);
    const peakPow = this.goertzelPower(timeSamples, {
      coeff: 2*Math.cos(2*Math.PI*k0/BLOCK), block: BLOCK, k: k0
    });
    const offsets = [freq-150, freq-100, freq-60, freq-40, freq+40, freq+60, freq+100, freq+150];
    let bgPow = 0, bgCount = 0;
    for (const f of offsets) {
      if (f < 80 || f > SR / 2 - 80) continue;
      const k = Math.round(BLOCK * f / SR);
      bgPow += this.goertzelPower(timeSamples, { coeff: 2*Math.cos(2*Math.PI*k/BLOCK), block: BLOCK, k });
      bgCount++;
    }
    return peakPow / (bgPow / (bgCount || 1) + 1e-12);
  }

  // ── Adaptive spectral subtraction SNR ────────────────────────────
  // Returns SNR after subtracting per-species adaptive noise estimate.
  // Updates noise estimate when signal is weak (noise-only frames).
  adaptiveSNR(timeSamples, sp) {
    const params = this.goertzelParams[sp.id];
    const sigPow = this.gaussianBankPower(timeSamples, params.f0Bank);
    const rawRef = this.refPower(timeSamples, params.ref);
    const rawSNR = sigPow / (rawRef + 1e-12);

    // Noise freeze: when signal is strong, hold the noise estimate for N frames
    // so we never "learn" the mosquito itself as the noise floor (Hirsch & Ehrlicher 1995).
    if (!this._noiseFreeze) this._noiseFreeze = {};
    if (rawSNR >= this.GOERTZEL_SNR_MIN * 2.0) {
      this._noiseFreeze[sp.id] = 12;           // freeze for ~480ms at 25fps
    } else if ((this._noiseFreeze[sp.id] || 0) > 0) {
      this._noiseFreeze[sp.id]--;
    } else if (rawSNR < this.GOERTZEL_SNR_MIN * 0.7) {
      // True noise-only frame: update noise estimate.
      // During startup calibration use α=0.5 (converges in ~3 frames vs ~30s normally).
      const ema = this._fastCalib ? 0.50 : this.NOISE_EMA_ALPHA;
      this.adaptiveNoise[sp.id] = ema * this.adaptiveNoise[sp.id] + (1 - ema) * sigPow;
    }

    const noisePow = this.adaptiveNoise[sp.id];
    const cleanSig = Math.max(0, sigPow - this.OVERSUBTRACTION * noisePow);
    return cleanSig / (rawRef + 1e-12);
  }

  // ── Activity-period sensitivity boost ───────────────────────────
  // Returns SNR threshold multiplier: < 1 means easier to detect (more sensitive).
  // Active species at their peak time: 0.80 × threshold (20% easier to trigger).
  // All other times: 1.0 (no change — we never penalise, only reward).
  getActivityFactor(sp) {
    const period = timeOfDayCategory(this._lat, this._lng);
    const isActive = sp.activity === 'day'   && period === 'day'   ||
                     sp.activity === 'night' && period === 'night' ||
                     sp.activity === 'dusk'  && period === 'dusk'  ||
                     // Night species also boosted at dusk (starting to emerge)
                     sp.activity === 'night' && period === 'dusk';
    return isActive ? 0.80 : 1.0;
  }

  // ── Spectral flatness (Wiener entropy) pre-gate ─────────────────
  // Geometric/arithmetic mean ratio of linear spectrum 200–2000 Hz.
  // ~1.0 = perfectly flat broadband noise; ~0 = strong tonal signal.
  // Gate: reject all detection when flatness > 0.80 (pure noise env).
  spectralFlatness(freqData) {
    const lo = this.engine.freqToBin(200);
    const hi = Math.min(this.engine.freqToBin(2000), freqData.length - 1);
    let logSum = 0, linSum = 0, count = 0;
    for (let i = lo; i <= hi; i++) {
      if (!isFinite(freqData[i]) || freqData[i] < -130) continue;
      const lin = Math.pow(10, freqData[i] / 10);
      logSum += Math.log(lin + 1e-20);
      linSum += lin;
      count++;
    }
    if (count === 0) return 1;
    return Math.exp(logSum / count) / (linSum / count + 1e-20);
  }

  // ── Harmonic gate + bonus (2f and 3f) ───────────────────────────
  // Returns 0 if HARMONIC_REQUIRED and no harmonics detected (hard gate).
  // Otherwise returns confidence multiplier 1.0–1.35 based on harmonic strength.
  harmonicBonus(timeSamples, sp) {
    const params = this.goertzelParams[sp.id];
    // timeSamples are pre-emphasised (x[n] − 0.97·x[n−1]), whose power gain
    // |1 − αe^{−jω}|² rises with frequency: ~4× more at 2f0 than at f0 for
    // these species. Undo it so the harmonic RATIOS below are physical —
    // otherwise real mosquitoes read H2 ≈ H1, trip the "inverted harmonics"
    // rejection and the HPS penalty, and rarely lock even when loud.
    const SR = 44100, A = 0.97;
    const g = (hz) => 1 + A * A - 2 * A * Math.cos(2 * Math.PI * hz / SR);
    const pow1f  = this.gaussianBankPower(timeSamples, params.f0Bank) / g(sp.freq);
    const pow2f  = this.gaussianBankPower(timeSamples, params.f2Bank) / g(2 * sp.freq);
    const pow3f  = params.f3Bank ? this.gaussianBankPower(timeSamples, params.f3Bank) / g(3 * sp.freq) : 0;

    const h2Present = pow2f >= this.HARMONIC_SNR_RATIO * pow1f;
    const h3Present = pow3f >= this.HARMONIC_SNR_RATIO * 0.5 * pow1f;

    if (this.HARMONIC_REQUIRED && !h2Present && !h3Present) return 0;

    // Harmonic decay check (HumBugDB 2020): mosquito series decays — P(f₀) ≥ P(2f₀).
    // If 2nd harmonic exceeds fundamental by >20%, it's likely mechanical interference.
    if (pow2f > pow1f * 1.2) return 0;

    // Cache harmonic powers for HPS confidence shaping in analyze()
    if (!this._harms) this._harms = {};
    this._harms[sp.id] = { p1f: pow1f, p2f: pow2f, p3f: pow3f };

    const ref   = this.refPower(timeSamples, params.ref) / g(sp.freq);   // same compensation
    const snr2f = pow2f / (ref + 1e-12);
    const snr3f = pow3f / (ref + 1e-12);
    const bonus2 = Math.min(0.25, Math.max(0,
      (snr2f - this.GOERTZEL_SNR_MIN) / (3 * this.GOERTZEL_SNR_MIN) * 0.25
    ));
    const bonus3 = Math.min(0.10, Math.max(0,
      (snr3f - this.GOERTZEL_SNR_MIN) / (3 * this.GOERTZEL_SNR_MIN) * 0.10
    ));
    return 1.0 + bonus2 + bonus3;
  }

  // ── Species disambiguation ───────────────────────────────────────
  // For close-frequency pairs: suppress weaker if SNR difference > 2 dB
  disambiguate(candidates) {
    if (candidates.length <= 1) return candidates;
    const suppressed = new Set();
    for (const [idA, idB] of this.DISAMBIGUATION_PAIRS) {
      const a = candidates.find(c => c.species.id === idA);
      const b = candidates.find(c => c.species.id === idB);
      if (!a || !b) continue;
      const snrDiff = 10*Math.log10((a.snrLinear+1e-9)/(b.snrLinear+1e-9));
      if (Math.abs(snrDiff) > 2.0) {
        suppressed.add(snrDiff > 0 ? idB : idA);  // suppress weaker
      }
      // If within 2 dB: ambiguous → keep both (real overlap scenario)
    }
    return candidates.filter(c => !suppressed.has(c.species.id));
  }

  // Species the far-field track belongs to: the nearest species centre whose
  // frequency range contains the measured f0 (skipping disabled species).
  _farFieldSpecies(f0) {
    let best = null, bestD = Infinity;
    for (const sp of SPECIES_DB) {
      if (this._disabled.has(sp.id)) continue;
      if (f0 < sp.freqRange[0] || f0 > sp.freqRange[1]) continue;
      const d = Math.abs(f0 - sp.freq);
      if (d < bestD) { bestD = d; best = sp; }
    }
    return best;
  }

  // Near-field evidence for one species (the original gate chain, unchanged in
  // intent): returns { pass, conf, snrLinear, reason }.
  _nearField(sp, td, rawFreqData, useGoertzel) {
    let snrLinear;
    if (useGoertzel) {
      snrLinear = this.adaptiveSNR(td, sp);
      this.lastGoertzelSNR[sp.id] = snrLinear;
    } else {
      const loBin = this.engine.freqToBin(sp.freqRange[0]);
      const hiBin = Math.min(this.engine.freqToBin(sp.freqRange[1]), rawFreqData.length-1);
      let peak = -130;
      for (let i=loBin; i<=hiBin; i++) {
        if (isFinite(rawFreqData[i]) && rawFreqData[i] > peak) peak = rawFreqData[i];
      }
      snrLinear = Math.pow(10, (peak - this.lastNoiseFloor) / 10);
    }
    const fail = (reason) => ({ pass: false, conf: 0, snrLinear, reason });

    const effectiveSNR = this.GOERTZEL_SNR_MIN * this.getActivityFactor(sp);
    if (snrLinear < effectiveSNR) return fail(`snr×${snrLinear.toFixed(1)}<${effectiveSNR.toFixed(1)}`);

    if (useGoertzel) {
      const tonality = this.tonalityScore(td, sp.freq);
      this.lastTonality[sp.id] = tonality;
      if (tonality < this.TONALITY_MIN) return fail(`tonal:${tonality.toFixed(1)}<${this.TONALITY_MIN}`);

      // Autocorrelation periodicity (Mukundarajan STM 2017). NB: R[τ0]/R[0] on
      // the raw waveform is ~ tone power / TOTAL broadband power, so this gate
      // only passes close, loud mosquitoes — the far-field tracker covers range.
      const acorr = this.autocorrPeriodicity(td, sp.freq);
      if (acorr < this.AUTOCORR_MIN) return fail(`acorr:${acorr.toFixed(3)}<${this.AUTOCORR_MIN}`);

      const frames = this.subframeConsistency(td, sp.freq);
      if (frames < 3) return fail(`transient:${frames}/4`);
    }

    let hBonus = 1.0;
    if (useGoertzel) {
      hBonus = this.harmonicBonus(td, sp);
      if (hBonus === 0) return fail('no-harmonic');
    }

    const SNR_MAX = 80;
    let conf = Math.min(1, (Math.log10(snrLinear) - Math.log10(this.GOERTZEL_SNR_MIN)) /
                            (Math.log10(SNR_MAX)  - Math.log10(this.GOERTZEL_SNR_MIN)));
    conf = Math.min(1, conf * hBonus);

    // Harmonic-structure check (HumBugDB 2020: H2 = 10-75% of H1 in power).
    // _harms holds pre-emphasis-compensated powers (see harmonicBonus).
    if (useGoertzel) {
      const hp = this._harms?.[sp.id];
      if (hp && hp.p1f > 1e-20) {
        const h2r = hp.p2f / hp.p1f, h3r = hp.p3f / hp.p1f;
        if (!(h2r >= 0.05 && h2r <= 0.80) || !(h3r <= 0.60)) conf *= 0.70;
      }
    }
    return { pass: true, conf, snrLinear, reason: '' };
  }

  analyze(rawFreqData, timeData) {
    if (!rawFreqData || !rawFreqData.length) return [];

    this.lastNoiseFloor = this.estimateNoiseFloor(rawFreqData);
    this.ambientRMS     = this.computeAmbientRMS(rawFreqData);

    // Far-field tracker runs on EVERY frame (even gated ones) so its per-bin
    // noise floor stays current.
    const ff = this.ff ? this.ff.process(rawFreqData) : { detected: false };
    this.lastFarField = ff;

    const confIdx = this.confIdx % this.CONF_FRAMES;
    this.confIdx++;

    // Ambient gate
    if (this.ambientRMS > this.AMBIENT_GATE_DB) {
      SPECIES_DB.forEach(sp => {
        this.confBuffer[sp.id][confIdx] = 0;
        this.lastRejection[sp.id] = `loud:${this.ambientRMS.toFixed(0)}dB`;
        this._addVote(sp.id, false);
      });
      this._lastDetections = [];
      return [];
    }

    // Spectral flatness pre-gate: reject completely flat (broadband-only) environments
    const flatness = this.spectralFlatness(rawFreqData);
    if (flatness > this.FLATNESS_GATE) {
      SPECIES_DB.forEach(sp => {
        this.confBuffer[sp.id][confIdx] = 0;
        this.lastRejection[sp.id] = `flat:${flatness.toFixed(2)}`;
        this._addVote(sp.id, false);
      });
      this._lastDetections = [];
      return [];
    }

    const useGoertzel = timeData && timeData.length >= this._goertzelBlock;
    // Pre-emphasis computed once per analysis frame — boosts mosquito freq range vs low-freq noise
    const td = useGoertzel ? this.preEmphasis(timeData) : null;
    const ffSp = ff.detected ? this._farFieldSpecies(ff.f0) : null;
    const candidates  = [];

    for (const sp of SPECIES_DB) {
      if (this._disabled.has(sp.id)) {
        this.confBuffer[sp.id][confIdx] = 0;
        this._addVote(sp.id, false);
        continue;
      }
      if (this.isHum(sp.freq)) {
        this.confBuffer[sp.id][confIdx] = 0;
        this.lastRejection[sp.id] = `hum@${sp.freq}Hz`;
        this._addVote(sp.id, false);
        continue;
      }

      // Two independent evidence paths; a frame passes if either does.
      const nf = this._nearField(sp, td, rawFreqData, useGoertzel);
      let pass = nf.pass, conf = nf.conf, snrLinear = nf.snrLinear, via = 'near';
      if (ffSp && ffSp.id === sp.id && ff.conf > (pass ? conf : 0)) {
        pass = true; conf = ff.conf; via = 'far';
        snrLinear = Math.max(snrLinear, ff.snrLinear);
      }
      this.lastVia[sp.id] = pass ? via : '';

      if (!pass) {
        this.confBuffer[sp.id][confIdx] = 0;
        this.lastRejection[sp.id] = nf.reason;
        this._addVote(sp.id, false);
        continue;
      }

      this.confBuffer[sp.id][confIdx] = conf;
      this.lastRejection[sp.id] = '';
      this._addVote(sp.id, true);

      const voteFraction = this.voteCount[sp.id] / this.VOTE_FRAMES;
      if (voteFraction >= this.VOTE_THRESHOLD) {
        const buf     = this.confBuffer[sp.id];
        const avgConf = buf.reduce((s,v)=>s+v, 0) / this.CONF_FRAMES;
        if (avgConf >= this.CONFIDENCE_THRESHOLD) {
          candidates.push({
            species:      sp,
            frequency:    via === 'far' && ff.f0 ? ff.f0 : sp.freq,
            snr:          10 * Math.log10(Math.max(1, snrLinear)),
            snrLinear,
            voteFraction,
            confidence:   avgConf,
            range:        via,          // 'near' (gate chain) or 'far' (tracker)
          });
        }
      }
    }

    const disambiguated = this.disambiguate(candidates);
    const out = disambiguated.sort((a,b) => b.confidence - a.confidence);
    for (const d of out) {
      d.f0 = d.frequency;
      d.genus = d.species.genus;
      d.posterior = this.speciesPosterior(d.f0);
    }
    this._lastDetections = out;
    return out;
  }

  estimateNoiseFloor(data) {
    const lo = this.engine.freqToBin(80);
    const hi = Math.min(this.engine.freqToBin(6000), data.length-1);
    const slice = [];
    for (let i=lo; i<=hi; i++) {
      if (isFinite(data[i]) && data[i] > -130) slice.push(data[i]);
    }
    if (!slice.length) return -70;
    slice.sort((a,b)=>a-b);
    return slice[Math.floor(slice.length * 0.15)];
  }

  computeAmbientRMS(freqData) {
    const lo = this.engine.freqToBin(200);
    const hi = Math.min(this.engine.freqToBin(4000), freqData.length-1);
    let sumLin=0, count=0;
    for (let i=lo; i<=hi; i++) {
      if (isFinite(freqData[i])) { sumLin += Math.pow(10, freqData[i]/10); count++; }
    }
    return count > 0 ? 10*Math.log10(sumLin/count) : -80;
  }

  isHum(hz) {
    const r = Math.round(hz);
    for (let d=-2; d<=2; d++) { if (this.humSet.has(r+d)) return true; }
    return false;
  }

  _addVote(spId, positive) {
    const ptr = this.votePtr[spId];
    const old = this.voteBuffer[spId][ptr];
    const now = positive ? 1 : 0;
    this.voteBuffer[spId][ptr] = now;
    this.voteCount[spId]      += (now - old);
    this.votePtr[spId]         = (ptr+1) % this.VOTE_FRAMES;
  }

  getStatus() {
    return {
      noiseFloor:  this.lastNoiseFloor,
      ambientRMS:  this.ambientRMS,
      gated:       this.ambientRMS > this.AMBIENT_GATE_DB,
      snrMin:      this.GOERTZEL_SNR_MIN,
    };
  }

  getRawConfidences() {
    const result = {};
    SPECIES_DB.forEach(sp => { result[sp.id] = this.voteCount[sp.id] / this.VOTE_FRAMES; });
    return result;
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { FarFieldTracker, MosquitoClassifier };
}
