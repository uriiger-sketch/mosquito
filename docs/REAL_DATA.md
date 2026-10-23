# Validation and training on real recordings

`docs/EVALUATION.md` measures the detector on a physically parameterised
*synthetic* corpus. This document measures it on **real, published recordings**
and records what was learned from them. Sources, licences and pinned versions:
[`tools/real/SOURCES.md`](../tools/real/SOURCES.md). Everything here is
reproducible with the scripts in `tools/real/`.

## Data

| Recording (HumBug / MozzBNN, phone, 8 kHz) | Length | Used as |
|---|---|---|
| CDC culture *Ae. aegypti*, *Ae. albopictus*, *Cx. quinquefasciatus* | 53 min | mosquito, species-labelled |
| Ifakara (Tanzania) and DR Congo field recordings | 24 min | mosquito, species unlabelled (mostly *Anopheles*) |
| Bus stop, construction site, indoor speech | 66 min | real non-mosquito background |
| ESC-50: 49 everyday classes (2,000 clips; Piczak 2015) | 3.3 h | **held-out** false-alarm test only |

The 8 kHz audio is resampled to the app's 44.1 kHz with a polyphase
windowed-sinc filter (exact 441/80 ratio) and fed to the **unmodified** engine
exactly as the app does (2 s calibration, 8192-point frames, half-frame hop).
Every MozzBNN recording is split in time: **first half = dev** (all training and
every choice), **second half = test** (reported only). ESC-50 is never used for
any choice.

"Audible mosquito" ground truth comes from a detector-independent estimator
(`tools/real/wingbeat_f0.py`, `tune.js:audibleFrames`): each frequency bin is
whitened by its own median over the recording (removing stationary lines),
then a harmonic sum H1 + 0.7 H2 + 0.5 H3 is maximised over 380–900 Hz; a frame
is tonal if the fundamental is ≥ 12 dB above its neighbourhood and its own
median, with nothing at f0/2. A 10-s window with ≥ 3 such frames is "audible".

## Findings

### 1. Real mosquitoes are audible in short bursts — and both engines missed them

| | Production engine (before this work) | v3 engine (synthetic-tuned) |
|---|---|---|
| CDC culture recordings (53 min, 3 species) | **0 detections** | **0 detections** |
| Field recordings (24 min) | 1 event | 7 events |
| Real backgrounds (45 min after speech exclusion) | 5 events (6.7/h) | 9 events (12/h) |

Instrumenting the tracker on the real recordings explained why. On
vote-eligible tracks of real *Aedes*/*Culex*:

| Gate | Requirement (synthetic-tuned) | Real mosquitoes (median) |
|---|---|---|
| Continuity | fundamental in ≥ 75 % of frames | **36–39 %** |
| Flight-modulation proof | computed only on frames ≥ 9 dB in-bin SNR, needs 8 | in-bin SNR **≈ 5 dB** → test never runs |
| Pitch rate | ≤ 43 Hz/s (+ noise allowance) | **64–67 Hz/s** |

And measuring the audible episodes directly (independent estimator):

| Recording | Median episode | 90th pct | Longest | Flight time in episodes ≥ 1.1 s |
|---|---|---|---|---|
| CDC *Ae. aegypti* | 0.19 s | 0.74 s | 2.8 s | 26 % |
| CDC *Ae. albopictus* | 0.09 s | 0.65 s | 2.5 s | 18 % |
| CDC *Cx. quinquefasciatus* | 0.09 s | 0.37 s | 0.74 s | 0 % |
| Ifakara field | 0.37 s | 1.49 s | 7.8 s | 53 % |
| DRC field | 0.37 s | 1.39 s | 4.2 s | 46 % |

The lock path needs a ≥ 1.1 s track and a 2-s vote: structurally it cannot fire
on sub-second episodes. The synthetic corpus modelled flight as continuous; real
recordings show it is not.

### 2. A stationary 284.5 Hz line in the culture-room recordings

Both CDC *Ae. aegypti* and *Cx. quinquefasciatus* recordings carry a
284.5 Hz ± 1 Hz line in essentially every frame — a room/recorder artefact, the
kind of interference real deployments face. It must be whitened away before any
wingbeat measurement (the engine's per-bin noise floor does this).

### 3. Real wingbeat frequencies (after whitening)

| CDC culture | Measured f0, median [IQR] | Catalogue value |
|---|---|---|
| *Ae. aegypti* | **697 Hz** [653–724] (halves 698 / 694) | 617 [580–655] |
| *Ae. albopictus* | **658 Hz** [598–685], second mode ≈ 425 Hz | 668 [640–705] |
| *Cx. quinquefasciatus* | **638 Hz** [622–692], n = 81 frames | 453 [428–482] |
| Ifakara field (mostly *Anopheles*) | modes ≈ 375–425 and 750–800 Hz | *An.* ≈ 390–450 |

The culture recordings do not record the sex of each insect (males beat
≈ 1.3–1.6× faster; the Ifakara 750–800 Hz mode looks like males) nor the room
temperature (f0 rises ≈ 1.5 %/°C). The species centres were therefore **not**
moved on this evidence; the *Ae. aegypti* and *Cx. quinquefasciatus* gaps are
open questions for sexed, temperature-logged recordings.

## What was trained on the real data

1. **A learned flight-plausibility model** (`tools/real/collect.js`, `train.js`):
   L2-regularised logistic regression on 13 shape/dynamics features
   (`MosquitoClassifier.plausVector` — the same function in training and in the
   app; f0 and loudness deliberately excluded). Positives: 1,401 dev-split tracks
   from the real mosquito recordings whose instantaneous f0 matched the
   independent tone on ≥ 30 % of frames. Negatives: real background tracks and
   synthetic-adversary tracks (speech, singing, whistling, music, machines,
   other insects; midges and male mosquitoes excluded). Class-balanced, λ by
   5-fold cross-validation grouped by track. **Out-of-fold track AUC 0.78
   (0.80 against real backgrounds, 0.77 against synthetic adversaries).**
   Strongest weights: continuity +3.4, harmonic comb −2.8, sub-harmonic,
   pitch- and level-modulation positive, harmonic richness negative.
2. **Burst accumulation** (`MosquitoClassifier._burstDetection`): a short track
   (≥ 0.35 s) the model scores ≥ θ is a *burst*; K bursts within W seconds whose
   f0 agree within ±tol make a detection (`Detection.mode = 'burst'`). A burst
   must follow a silence at its pitch (`BURST_GAP_S`), so a continuous tone the
   tracker fragments is not mistaken for repeated passes. Burst confidence is
   **not** calibrated (too little labelled real data) and is capped at 0.9.

### Results (held-out test half of the MozzBNN recordings)

| Sensitivity | Field-recording events/h | CDC audible-window recall | Real background FA/h |
|---|---|---|---|
| 1 | **84.5** | 9.7 % | 27.2 |
| 2 | **69.6** | 3.2 % | 12.7 |
| 3 (default) | 29.8 | 0 % | 10.9 |
| 4 | 9.9 | 0 % | 9.1 |
| 5 | 5.0 | 0 % | 3.6 |

(Real backgrounds here are a bus stop and a construction site — far noisier
than a bedroom.)

### Held-out household false alarms (ESC-50, 49 classes, 3.27 h, never used for any choice)

| Engine | False alarms/h | Classes that fired |
|---|---|---|
| Production engine (before) | 7.35 | siren 10, wind 4, church bells 4, 6 others |
| **This engine, sensitivity 3** | **1.84** | church bells 2, snoring, siren, glass, mouse click |
| This engine, sensitivity 2 | 4.59 | mouse click 3, church bells 3, snoring 2, 6 others |

The synthetic regression gate (`npm run eval:quick`) is unchanged at the default
level.

### What did not work

- Loosening the hand-set gates (continuity, modulation SNR, pitch rate, vote)
  traces a poor curve on real data: 28 % recall cost 22 false alarms/h, 81 %
  recall cost 51/h and destroyed genus accuracy (13 %).
- Burst accumulation with no intermittency rule matched the simulation on real
  data (dev CDC recall 17.5 %) but tripled synthetic false alarms (held notes
  and tones fragment into many "bursts"); kept only at sensitivities 1–2 with a
  0.1 s gap.
- The dev-split gains on the CDC *cup* recordings did not survive the test
  split (48 % → 6.5 % at the most permissive simulated point). The small,
  intermittent cup data does not support a confident CDC-recall claim.

## Bottom line

On real recordings the engine still detects only a minority of mosquito flight
episodes, and almost none of the very brief ones in the CDC cup recordings; at
the default sensitivity its main real-data gain is **75 % fewer false alarms**
on everyday sounds than the engine it replaces. Sensitivities 1–2 trade false
alarms for 2–3× more detections in real field recordings. Closing the gap needs
what these public sources lack: **phone recordings of free-flying, sexed,
temperature-logged mosquitoes at measured distances in real bedrooms**, plus
tens of hours of real night-time household audio — and access to the full
HumBugDB audio on Zenodo.

## Reproduce

```bash
eval "$(tools/real/fetch.sh ~/mosquito-real-data | tail -1)"     # clones pinned sources, exports paths
REAL_CACHE=/tmp/mn-real node tools/real/tune.js cache
REAL_CACHE=/tmp/mn-real node tools/real/collect.js /tmp/rows-dev.jsonl --split dev --min-age 0.35
node tools/real/train.js /tmp/rows-dev.jsonl --out /tmp/model.json
echo '[{"level":1},{"level":2},{"level":3},{"level":4},{"level":5}]' > /tmp/levels.json
REAL_CACHE=/tmp/mn-real node tools/real/tune.js sweep /tmp/levels.json --split test
node tools/real/realdata.js --only esc50 --level 3          # held-out household false alarms
```
