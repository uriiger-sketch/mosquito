# MosquitoNet detection engine — evaluation

This document describes how the acoustic detector (`js/detector.js`) is
evaluated, what changed between the baseline (`e46de8d`) and this version, and
what the numbers do and do not mean. All tables between the GENERATED markers
are produced by `tools/eval/report.js` from the machine-readable reports in
`tools/eval/results/` — nothing is transcribed by hand.

> **The single most important limitation.** Every number here comes from a
> **synthetic, physically-parameterised corpus**. Real recordings (HumBugDB,
> Abuzz) were not reachable from the build environment. The corpus is built
> from literature parameters, but the detector was designed by someone who
> also wrote the corpus, so the results are an upper bound on real-world
> performance and must be re-measured on real phone recordings before any
> claim is made externally (see *Validation on real data* below).

## How to run

```
npm test                 # unit + contract tests (< 30 s)
npm run eval:quick       # CI: ~1-2 min on 4 cores; exits 1 on regression vs tools/eval/thresholds.json
npm run eval             # full test-split evaluation (~30 min on 4 cores)
npm run eval:report      # regenerate the tables below from tools/eval/results/*.json
node tools/eval/eval.js --full --root <other checkout>   # evaluate another engine (e.g. a baseline worktree)
```

Developer tools (dev split only): `tools/eval/sweep.js` (constant sweeps),
`tools/eval/features.js` (per-test rejection rates on true mosquito vs
adversary tracks), `tools/eval/fit-confidence.js` (confidence model),
`tools/eval/calibrate-noise.js` (estimator-noise model), `tools/eval/diagnose.js`.

## Methodology

**Pipeline emulation.** The harness drives the real `MosquitoClassifier`
(loaded exactly as the page loads it) the way the app does: an 8192-sample
window advanced by 4096 samples at 44.1 kHz, the AnalyserNode spectrum
reproduced bit-for-bit (Blackman α = 0.16, |X|/N, dB — W3C Web Audio §1.8.6),
the raw time samples, *reused* buffers between calls, ±1 full-scale clipping,
and the app's 2 s start-up calibration (`beginCalibration` … `endCalibration`).
Every job is seeded; the same code gives the same numbers on any machine.

**Splits.** Every random draw is seeded from (split, experiment, …). All tuning
(thresholds, the confidence model, the noise model) used the **dev** split;
every number in this document is from the **test** split, which was not used
for any design decision.

**SNR and range.** SNR = 10·log10(P_H1 / P_noise): mean power of the flight
tone's *fundamental* over the flight bout versus the *total* (0–22 kHz) power
of the reference room noise (pink + white, the legacy bench's model). In this
noise, SNR 0 dB corresponds to an in-bin SNR of +25 dB at 290 Hz … +28.5 dB at
700 Hz (Blackman peak bin vs mean noise bin). Beyond the near field
(λ/2π ≈ 8–14 cm) the tone falls as 1/r (Arthur et al. 2014), −6 dB per
doubling of distance, so an SNR50 of S dB means 50 % detection at
r50 = r0·10^(−S/20), where r0 is the distance at which that mosquito gives
SNR = 0 dB in that room. "Range factor" in the tables is r50/r0; the ratio of
two engines' range factors is their range ratio for the same mosquito and room.

**Detection trials.** For each species, 12 individuals; each is replayed at 11
SNRs (−36 … +4 dB; paired design — same individual, same noise, only the
amplitude changes). 3 s of room noise (incl. calibration), a 10 s flight bout,
1 s tail. *Detected* = any Detection in a frame ending inside the bout (+0.5 s).
SNR50 = 0.5-crossing of the isotonic (monotone) regression of P(detect) on SNR.

**Identification trials.** Individuals of each species native to the location
(app behaviour: every non-native species is disabled on the first GPS fix),
SNR uniform in [−15, −5] dB, plus 3 non-native individuals per species per
location (to measure what the location prior costs when an invasive species
turns up). Locations: no GPS; Accra (West Africa); Rome (Southern Europe);
Mumbai (South Asia). Top-1 is `Detection.species` at the first lock (what the
app counts and maps); top-3 uses `posterior`; genus = MAP genus.

**False alarms.** 33 adversary classes × 20 simulated minutes (4 independent
5-minute runs, each starting with calibration) on top of the room noise, plus
male mosquitoes. Events use the app's own logic (a species' lock is a new event
if that species' previous event was ≥ 60 s ago — so a stuck lock costs 60/h).
CIs are exact (Garwood) Poisson intervals. 20 minutes per class bounds a
zero-event rate only to < 11/h per class; the pooled 11 hours bound it to
< 0.34/h. A per-class proof of ≤ 0.2/h would need ≈ 15 h per class.

**Mixtures.** A mosquito (random species, SNR −12 dB) enters 15 s after an
adversary started (so stationary machines are already partly learned). Paired:
the same 20 individuals with and without the adversary. Only a Detection whose
f0 is within 5 % of the individual's true f0 counts (otherwise the adversary
itself locked). *Caveat for the baseline:* its close-range path reported the
catalogue frequency instead of a measurement (bug P1-2), so a few of its real
detections fail the 5 % match.

**Compute.** Mean and p95 wall time of `analyze()` alone (not the FFT the
browser does natively), single thread, Node 22, 2.1 GHz Xeon, on a 60 s scene
(quiet → mosquito → speech). Phones are ≈ 2–5× slower.

### Corpus parameters (literature-grounded ranges)

*Mosquitoes (tools/eval/population.js).* Individual female f0 at 26 °C is
log-normal around the catalogue value with a per-species CV drawn once in
[4 %, 8 %] (Brogdon 1994; Villarreal et al. 2017; Kiskin et al. 2021, HumBugDB).
Room temperature per trial ~ N(26, 3²) °C clipped to [18, 34]; f0 rises
1–2 %/°C per species (Unwin & Corbet 1984; Villarreal et al. 2017). Harmonics:
H2 2–16 dB below H1, H3–H6 decaying further, each with independent ±2 dB
directivity wander as the insect turns (Arthur et al. 2014). Within a flight f0
wanders (OU, σ 0.25–1.5 %, τ 0.5–3 s) with 0.1–0.4 % cycle jitter; level follows
1/r of an OU flight distance (±1.3–3 dB). Males: 1.3–1.6 × female f0 (Gibson &
Russell 2006; Cator et al. 2009; Pennetier et al. 2010).

*Adversaries (tools/eval/corpus.js).* White/pink/brown noise with level steps;
desk, ceiling and small (laptop/PC) fans with blade-pass tones, shaft sidebands,
speed drift and on/off cycles (Neise 1992); fridge compressor and split-AC
(inverter ramps, cross-flow blower BPF 300–700 Hz); transformer/LED buzz and
coil whine; male/female/child conversation (f0 85–155 / 165–255 / 240–400 Hz,
Hillenbrand et al. 1995 formants, syllabic rate ≈ 4–5 Hz); humming and singing
with 4.5–7 Hz vibrato of ±0.2–1 semitone (Sundberg 1987; Prame 1994);
whistling 500–1500 Hz (Shadle 1983); violin with vibrato and body resonances
(Mellody & Wakefield 2000), flute, piano (inharmonic decaying partials, Fletcher
& Rossing 1998), a pop band with steady pad chords, TV; house fly, honeybee,
bumblebee, fruit fly, non-biting midges (Sotavalta 1947; Unwin & Corbet 1984);
field and tree crickets (Bennet-Clark 1989); passerine song and low tonal calls
(doves, cuckoo, owls); traffic with Doppler engine harmonics and tyre noise
(Sandberg & Ejsmont 2002); rain; door slams, footsteps, typing, clinks, ringing
glasses; phone chimes, ringtones and ITU-T E.180 ringback tones (425 Hz EU,
440+480 Hz US).

**Male mosquitoes are counted as mosquitoes, not false alarms.** They are
mosquitoes (a surveillance signal of nearby breeding), and no single-microphone
acoustic feature separates a male from a higher-pitched female of another
species without species-specific harmonic templates that cannot be validated
here. Their *species labels* will be wrong (shifted towards Ae. albopictus /
japonicus); their event rate is reported separately (marked *).

## What changed, and why

**Detection is decided per flight track, independently of species.** The
baseline ran twelve per-species gate chains (Goertzel banks at catalogue
frequencies) plus a far-field tracker whose f0 was then forced into hard,
gapped `freqRange` windows; votes were split across species when f0 sat near a
window boundary. The new engine keeps and extends the far-field
track-before-detect front end and votes once per *track*:

1. *Whitening + local order-statistic CFAR.* Per-bin noise floors learn the
   noise colour; a median over ~210 Hz neighbourhoods tracks fast broadband
   changes (level steps, clicks, rain, speech formants). Whole-frame blanking
   (which threw away every frame in which someone spoke) is kept only for
   extreme broadband frames (mean local normaliser > 2×: bangs, a vehicle's
   tyre roar) — without it, traffic false alarms rose to ≈ 190/h on dev.
2. *Harmonic sum with H1-capped harmonics* (H2/H3 count at most ≈ 3 dB above
   H1) — a single strong line at 2f or 3f can no longer fake a track at f.
3. *Viterbi/CUSUM over up to four simultaneous tracks*, so a loud fan or voice
   cannot mask a mosquito by owning the single argmax.
4. *Physics plausibility per track* (every threshold set from dev-split
   feature distributions; `tools/eval/features.js`): fundamental present in
   ≥ 75 % of frames (speech has syllables); median pitch rate ≤ 43 Hz/s plus an
   estimator-noise allowance; no sub-harmonics and no *harmonic comb* — the
   neighbouring harmonics f·(n±1)/n of a lower-pitched source (engines, fan
   blade-pass, bees, flies, voices), which also catches odd multiples that the
   old f/2, f/3 veto missed; H1 dominance; line width (4–7 Hz vibrato smears
   the line); no "pure" tone (a loud tone with H2+H3 < −25 dB is a whistle or a
   beep — wing motion is far from sinusoidal); and **positive evidence of flight
   modulation**: detrended pitch or level variance significantly above what
   the estimator alone produces for a steady tone (z ≥ 2). The estimator-noise
   model (var f ≈ bin²·(0.5/snr + 3/snr²), var L ≈ 62/(snr+3) dB²) was
   *measured* (`tools/eval/calibrate-noise.js`); the first guess was 2.3× too
   optimistic. Tracks that are *confidently* steady (upper bound < 6 bp pitch
   and < 0.5 dB level) are machines for the rest of their life and are notched
   out of the noise floor; confidently steady level (< 0.3 dB) rejects held
   notes. Tones that are already sounding during calibration are no longer
   learned as noise (floors capped at 6 dB above the local median).
5. *A track-level vote* (≥ 70 % of 2 s at level 3) instead of 12 vote buffers.

**Identification is a Bayesian posterior** (`SpeciesModel`):
ln f0 ~ N(ln μ_s, σ²) with σ² = (freqSigma/freq)² + (β·σ_T)² + σ_m² — between-
individual spread (6 %, new `freqSigma` field), unknown room temperature
(β = 1.5 %/°C, σ_T = 3.5 °C, new `freqTempCoef`) and measurement error; native
species prior 1 vs 0.03 for non-natives; disabled species 0; a deliberately
mild time-of-day prior (×0.7 for the opposite activity period; not validated —
neutral in all evaluations). The f0 is the SNR-weighted track mean of a
harmonic-consistent estimate (H1–H3 weighted by h²·SNR_h; parabolic
interpolation on the Blackman dB spectrum, bias < 0.02 bin, tested).
`Detection.taxonLevel` is `'species'` when the top posterior ≥ 0.6, otherwise
`'genus'`, and then `Detection.species` is the most probable species *within
the most probable genus* so that `species.genus === genus`.

Honest consequence: with ≈ 8 % total f0 spread per species, the catalogue's
dense 515–668 Hz cluster (Ae. vexans, Ae. caspius, Ma. uniformis, Cx. pipiens,
Ae. aegypti, Ae. albopictus — 4–6 % apart) **cannot be resolved by f0 alone**.
Without a location prior, species top-1 is poor for any engine; the location
prior is what makes identification useful, and the posterior now *says* how
uncertain it is (calibration tables).

**`Detection.confidence` = calibrated P(real flying mosquito | evidence)**, a
logistic model of (a) the CUSUM margin over threshold (weight constrained ≥ 0:
more evidence never lowers it) and (b) *acoustic busyness* — the number of
strong tonal tracks rejected as non-mosquito in the last 30 s. A lock in a quiet
room is trustworthy; a lock while TV or music is playing much less so. Fitted on
dev events (`tools/eval/fit-confidence.js`); SNR, modulation strength and
harmonicity added nothing on dev, and a free fit gave the margin a *negative*
weight (the adversaries are loud) — not physically defensible, so it is
constrained. The absolute values are conditional on the harness's exposure mix
(every detection trial contains a mosquito; adversary classes run 20 min each);
in a real deployment the prior differs. Species certainty lives in
`posterior` / `taxonLevel`, not in `confidence`.

**Sensitivity levels** scale the CUSUM threshold τ (×0.70, 0.85, 1, 1.3, 1.7),
the vote window/quorum (1.2 s/55 %, 1.5 s/60 %, 2 s/70 %, 2.5 s/75 %,
3 s/80 %) and the modulation evidence required (z ≥ 1.5, 2, 2, 3, 4). Their
measured operating points are in the levels table. Sweeps on dev showed that
raising τ alone buys almost no false-alarm reduction (the remaining false
alarms are strong, structured tones), so the levels act mostly on evidence
duration and modulation strictness.

**P1 bugs fixed (each with a regression test in `tests/detector-p1.test.js`).**
(1) *Ae. caspius* (548 Hz = 11 × 50 Hz) was blocked from *both* paths by the
static mains-hum blacklist, killing the 532–559 Hz band; stationary lines are
now handled adaptively (whitening, steadiness, notches). (2) The close-range
path reported the catalogue frequency; every Detection now carries the measured
sub-bin f0 (the E2E fixtures at 418, 464, 628 Hz are reproduced within 2 Hz).
(3) 44100/8192 were hard-coded throughout; all DSP now derives from the
engine's `sampleRate`, `fftSize`/bin width and hop, a 48 kHz test passes, and
`MosquitoAudioEngine.init()` falls back to the device's native rate when a
44.1 kHz context cannot be created or connected (Firefox rate mismatch), with
the FFT size chosen to keep a ≈ 186 ms window. (4) Detections were emitted
during the start-up calibration; they are now suppressed until
`endCalibration()`. (5) The baseline could lock two species at once on one tone
(two app events, two map markers); one track now gives one Detection.

**Tried and dropped (no measurable benefit on dev).** A melodic note-change
detector (tracks born as a previous tone ended a semitone–octave away): no
effect once sub-harmonic "ghost" tracks were excluded. A pitch-only
"exact-pitch" veto: no effect. Raising τ: −1.5 dB SNR50 per small FA gain.
Longer minimum track age (2.1 s): −0.5 dB and +1 s latency for a modest FA
reduction — the 2 s/70 % vote was the better trade.

## Key results (test split)

* **False alarms −68 %** pooled over 33 adversary classes (55 → 17 /h). The
  in-band voice/music/whistle/phone-tone classes that dominated the baseline
  fall 5–25× (humming 282 → 45 /h, violin 255 → 15, singing 225 → 9, flute
  222 → 18, whistling 39 → 0, phone tones 9 → 0); small fans 111 → 42, traffic
  117 → 87. **Regressions:** pop music 84 → 108 /h and non-biting midges 54 →
  69 /h (the more sensitive engine hears more mosquito-like midges). No class
  meets the 0.2 /h target except the 15 classes with zero events (whose 20-min
  upper bound is 11 /h).
* **Detection range unchanged on average** (mean SNR50 −19.7 vs −19.3 dB), but
  **loud mosquitoes are no longer missed** (P(detect | SNR ≥ −8 dB) 0.99 vs
  0.86); *Ae. caspius* +3 dB (1.4× range) after removing the hum blacklist;
  one Detection per insect (the baseline locked two species at once in 13
  trials). At **equal false-alarm rate** the new engine has ≈ 4 dB (1.6×) more
  range: baseline level 4 = −15.3 dB at 18.9 FA/h vs new level 3 = −19.1 dB at
  18.0 FA/h on the same hard adversary set (levels table).
* **Confidence is now informative and calibrated**: Brier 0.225 → 0.044,
  ECE 0.19 → 0.06; mean confidence 0.91 on real locks vs 0.24 on false ones
  (baseline 0.56 vs 0.53 — no discrimination).
* **Identification**: more trials correctly identified at every location
  (species correct/trials: Rome 31 → 49 of 72, Accra 38 → 50 of 84, Mumbai
  28 → 48 of 96, no GPS 53 → 73 of 144), posteriors better calibrated.
  Conditional top-1 in Rome looks lower only because the baseline detected
  just the easy half of the Rome trials.
* **Cost**: `analyze()` 5.6 → 2.5 ms; latency 2.3 → 2.9 s.

<!-- GENERATED:BEGIN -->
### Headline (test split, sensitivity 3)

| Metric | Baseline (e46de8d) | Final (this branch) |
|---|---|---|
| Mean SNR50 over 12 species (dB, lower = better) | -19.3 | -19.7 |
| Worst-species SNR50 (dB) | -14.0 | -15.3 |
| Mean range factor r50/r0 (= 10^(−SNR50/20)) | 9.23 | 9.70 |
| P(detect) for loud mosquitoes (SNR ≥ −8 dB) | 0.863 | 0.988 |
| Median lock latency (SNR ≥ −12 dB, s) | 2.29 | 2.94 |
| Trials with ≥2 species locked at once | 13 | 0 |
| False alarms / hour, pooled over 33 adversary classes [95 % CI] | 55.18 [50.88, 59.75] | 17.45 [15.07, 20.11] |
| Adversary classes with any false alarm (of 33) | 21 | 18 |
| Species top-1 / genus top-1 — no location | 48 % / 66 % | 51 % / 70 % |
| Species top-1 / genus top-1 — accra | 57 % / 67 % | 62 % / 70 % |
| Species top-1 / genus top-1 — rome | 86 % / 89 % | 70 % / 79 % |
| Species top-1 / genus top-1 — mumbai | 42 % / 64 % | 53 % / 72 % |
| Species posterior Brier / ECE (no location) | 0.689 / 0.143 | 0.610 / 0.110 |
| Genus posterior ECE (no location) | 0.149 | 0.091 |
| Detection.confidence vs real-mosquito: Brier / ECE | 0.225 / 0.190 | 0.044 / 0.058 |
| Mean confidence on real / false locks | 0.56 / 0.53 | 0.91 / 0.24 |
| analyze() mean / p95 (ms, Node, single thread) | 5.63 / 7.58 | 2.46 / 4.19 |

### Detection: SNR50 per species

SNR50 from isotonic regression of P(detect) on SNR (12 individuals × 11 SNRs per species); `<`/`>` = censored at the grid edge; `~` = non-monotone curve (louder trials missed more often than quieter ones). P(loud) = P(detect) pooled over SNR ≥ −8 dB.

| Species | f0 (Hz) | Baseline SNR50 | Final SNR50 | Δ (dB) | Range × | Baseline P(loud) | Final P(loud) |
|---|---|---|---|---|---|---|---|
| Anopheles gambiae | 406 | -19.0 ~ | -19.3 | -0.3 | 1.04 | 0.79 | 1.00 |
| Anopheles stephensi | 388 | -17.0 ~ | -15.3 | 1.7 | 0.83 | 0.83 | 1.00 |
| Aedes aegypti | 617 | -20.0 ~ | -21.0 | -1.0 | 1.12 | 0.98 | 1.00 |
| Aedes albopictus | 668 | -21.5 ~ | -21.1 | 0.4 | 0.96 | 0.94 | 1.00 |
| Culex quinquefasciatus | 453 | -19.0 | -19.0 ~ | 0.0 | 1.00 | 0.85 | 0.94 |
| Aedes japonicus | 730 | -22.9 | -22.4 | 0.5 | 0.95 | 0.92 | 1.00 |
| Aedes vexans | 515 | -20.6 ~ | -20.6 | 0.0 | 1.00 | 0.88 | 1.00 |
| Mansonia uniformis | 570 | -21.7 ~ | -21.3 | 0.4 | 0.96 | 0.90 | 1.00 |
| Culex pipiens | 594 | -21.0 | -20.6 | 0.4 | 0.95 | 0.92 | 1.00 |
| Culiseta annulata | 344 | -17.0 ~ | -18.3 | -1.3 | 1.16 | 0.92 | 1.00 |
| Aedes (Ochlerotatus) caspius | 548 | -18.0 | -21.0 | -3.0 | 1.41 | 0.69 | 1.00 |
| Toxorhynchites brevipalpis | 290 | -14.0 ~ | -16.8 | -2.8 | 1.38 | 0.75 | 0.92 |

### False alarms per adversary class (sensitivity 3)

Each class: 20 simulated minutes (4 independent 5-min runs, each starting with the 2 s calibration). Events counted with the app's logic (a species' lock is a new event if its previous event was ≥ 60 s ago). 95 % CI: exact Poisson. Target ≤ 0.2 /h. * = not counted as a false alarm (male mosquitoes are mosquitoes).

| Class | Group | Baseline events | Baseline FA/h [95 % CI] | Final events | Final FA/h [95 % CI] |
|---|---|---|---|---|---|
| quiet_room | noise | 0 | 0.00 [0.00, 11.07] | 0 | 0.00 [0.00, 11.07] |
| noise_white | noise | 0 | 0.00 [0.00, 11.07] | 0 | 0.00 [0.00, 11.07] |
| noise_pink | noise | 0 | 0.00 [0.00, 11.07] | 0 | 0.00 [0.00, 11.07] |
| noise_brown | noise | 0 | 0.00 [0.00, 11.07] | 0 | 0.00 [0.00, 11.07] |
| fan_desk | machine | 5 | 15.00 [4.87, 35.01] | 3 | 9.00 [1.86, 26.30] |
| fan_ceiling | machine | 0 | 0.00 [0.00, 11.07] | 0 | 0.00 [0.00, 11.07] |
| fan_small | machine | 37 | 111.00 [78.15, 153.00] | 14 | 42.00 [22.96, 70.47] |
| hum_fridge | machine | 9 | 27.00 [12.35, 51.25] | 5 | 15.00 [4.87, 35.01] |
| hum_ac | machine | 20 | 60.00 [36.65, 92.67] | 16 | 48.00 [27.44, 77.95] |
| whine_electrical | machine | 8 | 24.00 [10.36, 47.29] | 2 | 6.00 [0.73, 21.67] |
| speech_male | voice | 0 | 0.00 [0.00, 11.07] | 0 | 0.00 [0.00, 11.07] |
| speech_female | voice | 0 | 0.00 [0.00, 11.07] | 0 | 0.00 [0.00, 11.07] |
| speech_child | voice | 0 | 0.00 [0.00, 11.07] | 0 | 0.00 [0.00, 11.07] |
| humming | voice | 94 | 282.00 [227.88, 345.10] | 15 | 45.00 [25.19, 74.22] |
| singing | voice | 75 | 225.00 [176.98, 282.04] | 3 | 9.00 [1.86, 26.30] |
| whistling | voice | 13 | 39.00 [20.77, 66.69] | 0 | 0.00 [0.00, 11.07] |
| music_violin | music | 85 | 255.00 [203.69, 315.31] | 5 | 15.00 [4.87, 35.01] |
| music_flute | music | 74 | 222.00 [174.32, 278.70] | 6 | 18.00 [6.61, 39.18] |
| music_piano | music | 30 | 90.00 [60.72, 128.48] | 15 | 45.00 [25.19, 74.22] |
| music_pop | music | 28 | 84.00 [55.82, 121.40] | 36 | 108.00 [75.64, 149.52] |
| tv | music | 16 | 48.00 [27.44, 77.95] | 6 | 18.00 [6.61, 39.18] |
| insect_housefly | insect | 10 | 30.00 [14.39, 55.17] | 2 | 6.00 [0.73, 21.67] |
| insect_honeybee | insect | 17 | 51.00 [29.71, 81.66] | 4 | 12.00 [3.27, 30.73] |
| insect_bumblebee | insect | 12 | 36.00 [18.60, 62.88] | 4 | 12.00 [3.27, 30.73] |
| insect_fruitfly | insect | 11 | 33.00 [16.47, 59.05] | 4 | 12.00 [3.27, 30.73] |
| insect_midge | insect | 18 | 54.00 [32.00, 85.34] | 23 | 69.00 [43.74, 103.53] |
| crickets | environment | 0 | 0.00 [0.00, 11.07] | 0 | 0.00 [0.00, 11.07] |
| birds_song | environment | 0 | 0.00 [0.00, 11.07] | 0 | 0.00 [0.00, 11.07] |
| birds_low | environment | 3 | 9.00 [1.86, 26.30] | 0 | 0.00 [0.00, 11.07] |
| traffic | environment | 39 | 117.00 [83.20, 159.94] | 29 | 87.00 [58.27, 124.95] |
| rain | environment | 0 | 0.00 [0.00, 11.07] | 0 | 0.00 [0.00, 11.07] |
| impulsive | environment | 0 | 0.00 [0.00, 11.07] | 0 | 0.00 [0.00, 11.07] |
| phone_tones | environment | 3 | 9.00 [1.86, 26.30] | 0 | 0.00 [0.00, 11.07] |
| mosquito_male * | mosquito | 13 | 39.00 [20.77, 66.69] | 22 | 66.00 [41.36, 99.92] |

### Mixtures: mosquito (SNR −12 dB) with an adversary already playing

Same 20 individuals per row (paired). Detected = a Detection whose f0 is within 5 % of the individual's true f0 inside the 10 s flight bout (onset at 15 s). Other-lock frames = frames in the bout with a lock on something else.

| Adversary | Baseline P(detect) [CI] | Final P(detect) [CI] | Baseline other-lock frames | Final other-lock frames |
|---|---|---|---|---|
| none | 15/20 = 0.75 [0.53, 0.89] | 20/20 = 1.00 [0.84, 1.00] | 0 | 0 |
| fan_desk | 13/20 = 0.65 [0.43, 0.82] | 15/20 = 0.75 [0.53, 0.89] | 37 | 0 |
| fan_small | 12/20 = 0.60 [0.39, 0.78] | 16/20 = 0.80 [0.58, 0.92] | 180 | 0 |
| hum_ac | 10/20 = 0.50 [0.30, 0.70] | 11/20 = 0.55 [0.34, 0.74] | 16 | 0 |
| hum_fridge | 10/20 = 0.50 [0.30, 0.70] | 12/20 = 0.60 [0.39, 0.78] | 125 | 25 |
| tv | 3/20 = 0.15 [0.05, 0.36] | 4/20 = 0.20 [0.08, 0.42] | 56 | 134 |
| speech_female | 2/20 = 0.10 [0.03, 0.30] | 5/20 = 0.25 [0.11, 0.47] | 0 | 0 |
| speech_child | 1/20 = 0.05 [0.01, 0.24] | 2/20 = 0.10 [0.03, 0.30] | 0 | 0 |
| music_pop | 4/20 = 0.20 [0.08, 0.42] | 1/20 = 0.05 [0.01, 0.24] | 12 | 296 |
| traffic | 10/20 = 0.50 [0.30, 0.70] | 12/20 = 0.60 [0.39, 0.78] | 20 | 23 |
| rain | 12/20 = 0.60 [0.39, 0.78] | 16/20 = 0.80 [0.58, 0.92] | 0 | 0 |
| whine_electrical | 15/20 = 0.75 [0.53, 0.89] | 17/20 = 0.85 [0.64, 0.95] | 113 | 0 |

### Identification (first lock; trials drawn from the location's native species; SNR uniform in [−15, −5] dB)

Top-1/top-3 are conditional on detection; "correct / trials" counts every trial (an undetected mosquito is not identified).

| Location | Engine | Trials | Detected | Species top-1 | Species correct / trials | Species top-3 | Genus top-1 | Genus correct / trials | Species Brier | Species ECE | Genus ECE | Non-native: detected / genus ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | baseline | 144 | 111 | 48 % | 53/144 | 87 % | 66 % | 73/144 | 0.689 | 0.143 | 0.149 | – |
| none | final | 144 | 142 | 51 % | 73/144 | 91 % | 70 % | 100/144 | 0.610 | 0.110 | 0.091 | – |
| accra | baseline | 84 | 67 | 57 % | 38/84 | 93 % | 67 % | 45/84 | 0.513 | 0.123 | 0.134 | 4/15 / 2 |
| accra | final | 84 | 81 | 62 % | 50/84 | 96 % | 70 % | 57/84 | 0.453 | 0.086 | 0.115 | 13/15 / 6 |
| rome | baseline | 72 | 36 | 86 % | 31/72 | 100 % | 89 % | 32/72 | 0.251 | 0.107 | 0.062 | 9/18 / 6 |
| rome | final | 72 | 70 | 70 % | 49/72 | 100 % | 79 % | 55/72 | 0.395 | 0.118 | 0.097 | 17/18 / 9 |
| mumbai | baseline | 96 | 66 | 42 % | 28/96 | 88 % | 64 % | 42/96 | 0.674 | 0.221 | 0.208 | 4/12 / 2 |
| mumbai | final | 96 | 90 | 53 % | 48/96 | 92 % | 72 % | 65/96 | 0.568 | 0.080 | 0.105 | 12/12 / 5 |

Final-engine genus confusion (no location prior; rows = true genus, columns = reported):

| true \ reported | Anopheles | Culiseta | Culex | Aedes | Toxorhynchites | none |
|---|---|---|---|---|---|---|
| Anopheles | 21 | 1 | 2 | 0 | 0 | 0 |
| Aedes | 0 | 0 | 4 | 56 | 0 | 0 |
| Culex | 3 | 0 | 6 | 15 | 0 | 0 |
| Mansonia | 0 | 0 | 0 | 12 | 0 | 0 |
| Culiseta | 1 | 7 | 0 | 0 | 3 | 1 |
| Toxorhynchites | 0 | 1 | 0 | 0 | 10 | 1 |

### Reliability: Detection.confidence vs "real mosquito" (events: first locks of mosquito trials + adversary false-alarm events)

| bin | baseline n | baseline mean p | baseline freq | final n | final mean p | final freq |
|---|---|---|---|---|---|---|
| [0.0, 0.1) | 0 | – | – | 107 | 0.04 | 0.11 |
| [0.1, 0.2) | 0 | – | – | 44 | 0.14 | 0.27 |
| [0.2, 0.3) | 0 | – | – | 22 | 0.24 | 0.41 |
| [0.3, 0.4) | 0 | – | – | 10 | 0.35 | 0.40 |
| [0.4, 0.5) | 436 | 0.47 | 0.41 | 20 | 0.46 | 0.40 |
| [0.5, 0.6) | 1298 | 0.57 | 0.79 | 17 | 0.56 | 0.65 |
| [0.6, 0.7) | 104 | 0.63 | 0.30 | 42 | 0.64 | 0.91 |
| [0.7, 0.8) | 2 | 0.70 | 0.50 | 67 | 0.76 | 0.97 |
| [0.8, 0.9) | 1 | 0.84 | 1.00 | 237 | 0.86 | 0.96 |
| [0.9, 1.0) | 4 | 0.94 | 0.25 | 1132 | 0.96 | 0.99 |

### Reliability: top-1 species probability vs species correct (no location)

| bin | baseline n | baseline mean p | baseline freq | final n | final mean p | final freq |
|---|---|---|---|---|---|---|
| [0.1, 0.2) | 0 | – | – | 1 | 0.19 | 0.00 |
| [0.2, 0.3) | 1 | 0.30 | 1.00 | 37 | 0.25 | 0.32 |
| [0.3, 0.4) | 12 | 0.38 | 0.42 | 40 | 0.37 | 0.47 |
| [0.4, 0.5) | 32 | 0.45 | 0.38 | 27 | 0.43 | 0.59 |
| [0.5, 0.6) | 19 | 0.53 | 0.58 | 14 | 0.54 | 0.43 |
| [0.6, 0.7) | 21 | 0.65 | 0.43 | 10 | 0.67 | 0.90 |
| [0.7, 0.8) | 13 | 0.73 | 0.46 | 4 | 0.74 | 0.75 |
| [0.8, 0.9) | 7 | 0.85 | 0.43 | 6 | 0.84 | 0.83 |
| [0.9, 1.0) | 6 | 0.94 | 1.00 | 3 | 0.95 | 1.00 |

### Sensitivity levels (operating points)

Detection on 4 individuals × 5 SNRs per species; false alarms pooled over the 14 hardest in-band adversary classes (5 min each).

| Level | Baseline mean SNR50 | Baseline FA/h [CI] | Final mean SNR50 | Final FA/h [CI] | Final P(loud) | Final latency (s) |
|---|---|---|---|---|---|---|
| 1 | -20.2 | 182.57 [158.88, 208.81] | -19.7 | 46.29 [34.77, 60.39] | 0.979 | 2.20 |
| 2 | -18.6 | 140.57 [119.88, 163.81] | -19.3 | 36.86 [26.67, 49.65] | 0.979 | 2.39 |
| 3 | -17.8 | 96.00 [79.05, 115.51] | -19.1 | 18.00 [11.14, 27.52] | 0.979 | 2.85 |
| 4 | -15.3 | 18.86 [11.82, 28.55] | -18.0 | 11.14 [5.93, 19.05] | 0.979 | 3.32 |
| 5 | -4.2 | 4.29 [1.39, 10.00] | -17.0 | 5.14 [1.89, 11.19] | 0.979 | 3.69 |

_Generated by tools/eval/report.js from tools/eval/results/baseline-full-test.json (e46de8d) and tools/eval/results/final-full-test.json (d79f1de)._
<!-- GENERATED:END -->

## Known limitations

* **Synthetic corpus only** (see the top). Model mismatch is built in on
  purpose for identification (per-species CVs and temperature coefficients
  differ from what the identifier assumes), but the *detector features* —
  especially the requirement of flight modulation (pitch wander, 1/r level
  fluctuation) — rest on physical assumptions that must be checked on real
  free-flight recordings. If real mosquitoes near a phone are steadier than
  modelled, sensitivity will drop.
* **Music, humming and some machines still produce false alarms** (tables
  above), well above the 0.2 /h product target for those classes. Steady synth
  pads doubled by bass harmonics, held hummed notes, inverter AC units and
  traffic engine harmonics are the main residual leaks.
* **Masking.** A mosquito at −12 dB under loud speech or TV (20–40 dB louder in
  the same band) is mostly not detected. That is physics without source
  separation; the busyness feature at least keeps the confidence of any lock
  under such conditions low.
* **Non-biting midges and male mosquitoes are detected as mosquitoes.**
* **Identification within the 515–668 Hz cluster is near chance without a
  location prior**; the posterior is calibrated against the corpus model, whose
  spread was itself drawn from the same literature ranges as the identifier's
  prior (so calibration is optimistic).
* **Absolute levels.** `EngineState.noisy` assumes a mosquito fundamental of
  30 dB SPL at 10 cm (≈ audible at 0.5–1 m in a quiet room) and a phone MEMS
  microphone of −38 dBFS at 94 dB SPL; both vary by ±10 dB.
* **Latency** at level 3 is ≈ 3 s (baseline ≈ 2.3 s): the price of requiring
  positive modulation evidence and a longer vote.

## Validation on real data (recommended next steps)

1. Record a calibrated set on the target phones: free-flying caged mosquitoes
   of the main local species at measured distances (10 cm – 2 m) in quiet and
   noisy rooms, with a reference SPL meter; plus 50–100 h of real household
   audio (bedrooms with fans/AC, TV, music, conversation, children) without
   mosquitoes, from many homes.
2. Re-run the same metrics on HumBugDB (Kiskin et al. 2021) and Abuzz
   (Mukundarajan et al. 2017) recordings — they are not phone-on-bedside-table
   audio, but they test the flight-modulation assumption and the f0
   distributions per species.
3. Refit `freqSigma`/`freqTempCoef` and the confidence model on real data;
   replace the synthetic adversary FA table with real-household FA/hour,
   which is what the 0.2 /h product target is about.
4. Field-check `noisy` against measured detection range.

## References

Arthur BJ et al. 2014, JASA 135:933 · Bennet-Clark HC 1989, in *Cricket Behavior
and Neurobiology* · Brogdon WG 1994, J Med Entomol 31:700 · Cator LJ et al. 2009,
Science 323:1077 · Ding N et al. 2017, Neurosci Biobehav Rev 81:181 · Fletcher NH &
Rossing TD 1998, *The Physics of Musical Instruments* · Gibson G & Russell I 2006,
Curr Biol 16:1311 · Hillenbrand J et al. 1995, JASA 97:3099 · ITU-T E.180 Suppl. 2 ·
Kiskin I et al. 2021, HumBugDB (NeurIPS Datasets & Benchmarks) · Mellody M &
Wakefield GH 2000, JASA 107:598 · Mukundarajan H et al. 2017, eLife 6:e27854 ·
Neise W 1992, Noise Control Eng J 38:113 · Pennetier C et al. 2010, Curr Biol
20:131 · Prame E 1994, JASA 96:1979 · Sandberg U & Ejsmont JA 2002, *Tyre/Road
Noise Reference Book* · Shadle CH 1983, Phys Teach 21:148 · Sotavalta O 1947,
Acta Entomol Fenn 4 · Sundberg J 1987, *The Science of the Singing Voice* ·
Unwin DM & Corbet SA 1984, Physiol Entomol 9:115 · Villarreal SM et al. 2017,
J Med Entomol 54:1116 · WHO 2009, Night Noise Guidelines for Europe.
