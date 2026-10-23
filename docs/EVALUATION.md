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

<!-- GENERATED:BEGIN -->
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
