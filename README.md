# MosquitoNet — acoustic mosquito surveillance

**Turn a phone into a mosquito detector.** MosquitoNet listens through the
microphone for the wingbeat tone of mosquitoes (≈ 290–730 Hz plus harmonics),
estimates which genus and species is in the room, and adds an anonymised
detection to a global vector map (with a coarsened location, if the user allows
location).

- Installable PWA, no app store, no account. Works offline once loaded.
- All audio analysis runs on the device. **No audio is ever recorded or uploaded.**
- 12 catalogued species, including the main malaria (*Anopheles*), dengue/Zika
  (*Aedes*) and West Nile (*Culex*) vectors; location-aware species priors and
  local names.

Live app: https://mosquito.charity (also https://uriiger-sketch.github.io/mosquito/)

---

## How detection works

1. **Capture** (`js/audio-engine.js`) — Web Audio at the device's real sample
   rate (44.1 kHz preferred, native-rate fallback), 8192-point spectra every
   4096 samples (~10.8 analyses/s).
2. **Detect** (`js/detector.js`) — a harmonic track-before-detect tracker
   follows up to four tonal tracks at once on a whitened spectrum. A track may
   vote only if it looks like a *flying insect*: continuous fundamental, smooth
   pitch, nothing below f0 and no harmonic comb from a lower-pitched source
   (voices, bees, flies, fans), fundamental-dominant harmonic series, no musical
   vibrato, and positive evidence of flight modulation (pitch/level wander
   beyond estimator noise). Steady lines are classified as machines and
   notched out.
3. **Identify** — a Bayesian posterior over the enabled species from the
   measured f0 (H1–H3, sub-bin precision), per-species wingbeat spread and
   temperature uncertainty, location priors and a mild time-of-day prior.
   When no single species is confident the result is reported at genus level
   ("*Aedes* sp. — species uncertain") with the candidates shown.
4. **Confidence** — `confidence` is a calibrated probability that the track
   is a real mosquito (logistic model; Brier 0.044, ECE 0.058 on the
   synthetic test split). The app reports a detection only from 50 %
   (below that it is a candidate, shown as "checking"); finds ≥ 70 % are
   confident — shared to the map, and one training step for the shared model,
   whose update is sent after each one.

The engine ⇄ UI contract is documented at the top of `js/detector.js`.

## Accuracy — current status

Measured with the seeded evaluation harness in `tools/eval/` (full method,
tables and limitations: [`docs/EVALUATION.md`](docs/EVALUATION.md)).

| Metric (synthetic test split, sensitivity 3) | Previous engine | This engine |
|---|---|---|
| False alarms / hour, pooled over 33 adversary classes | 55.2 | **17.5** |
| P(detect), loud mosquito | 0.86 | **0.99** |
| Range at equal false-alarm rate | — | **≈ 1.6×** |
| Confidence calibration (Brier / ECE) | 0.225 / 0.190 | **0.044 / 0.058** |
| Reported frequency | catalogue value | **measured f0** |
| `analyze()` cost (mean, Node) | 5.6 ms | **2.5 ms** |

**On real recordings** ([`docs/REAL_DATA.md`](docs/REAL_DATA.md); HumBug/MozzBNN
phone recordings from CDC cultures, Tanzania and the DRC, plus the ESC-50
everyday-sound benchmark as a held-out test):

- Everyday sounds (ESC-50, 49 classes, never used for tuning): **1.84 false
  alarms/h** at the default sensitivity, versus 7.35/h for the previous engine.
- Real mosquito flight is mostly audible in **sub-second bursts**; the previous
  engine detected none of 53 min of CDC culture recordings. A burst-accumulation
  path trained on the real recordings (learned plausibility model, out-of-fold
  AUC 0.78) gives sensitivities 1–2 **2–3× more detections in real field
  recordings**; brief culture-cup bursts remain largely missed.
- **Held-out, human-labelled test** (HumBug crowd-labelled Culex phone
  recordings, 94 flight episodes, never used for tuning): the default
  sensitivity detects **3× more** labelled episodes than the previous engine
  (12.8 % vs 4.3 %), sensitivity 2 detects **8.7× more** (37.2 %).
- Those cage recordings are dominated by **male** Culex (≈ 640 Hz, which a
  wingbeat alone cannot tell from a female *Aedes*). The identifier now models
  male wingbeats for every species and tells the user when a detection is
  probably a male (males don't bite).
- The synthetic figures above are an upper bound; real-world performance must be
  re-measured on phone recordings of free-flying mosquitoes in bedrooms.
- Known weak spots: hummed notes, pop-music pads, AC units, traffic and small
  fans still produce false alarms (church bells and snoring occasionally, in
  ESC-50); non-biting midges are detected as mosquitoes; male mosquitoes are
  flagged but still counted;
  loud TV or speech masks a mosquito.

## Privacy

- No audio leaves the device; there is no account.
- A confirmed detection carries species/genus, confidence, wingbeat frequency
  and a location **snapped to a fixed ~2 km grid and moved by a secret per-device
  offset of up to ~1.5 km**, so many reports from home never average back to the
  home — not even across wiped browser storage, except that a home within a few
  hundred metres of a cell edge can then alternate between two cells. Before anything is published the
  server adds a second secret offset, snaps to its ~5 km grid and rotates device
  pseudonyms daily. A phone's public point stays the same each day, but the grid
  makes many phones share it.
- While the app is open it sends a heartbeat (no location: just that it is open),
  and it shares small noise-added updates to the detector's model.
- Detection history is stored only on the device and can be exported or cleared.
- The operator's admin view (server `ADMIN_PASSWORD`, see `DEPLOY.md` → *Admin
  view*) shows the stored detections and the locations shared with them, as
  sent; there is no list of connected or listening devices. Heartbeats carry
  no location.

## Repository layout

```
index.html               UI (design system, App, map, history, spectrogram)
js/species.js            species catalogue, regional priors, local names/disease
js/audio-engine.js       microphone capture
js/detector.js           detection + identification engine (pure DSP, runs in Node too)
sw.js                    service worker (network-first app code, cached tiles)
vendor/, assets/fonts/   self-hosted Leaflet and fonts (no third-party CDNs)
federated_server.py      Flask backend: detections, map hotspots, federated model
tools/eval/              accuracy evaluation harness + regression thresholds
tests/                   unit/contract tests, tests/server (pytest), tests/e2e (Playwright: detection + app↔server)
docs/EVALUATION.md       evaluation method and results
docs/REAL_DATA.md        real recordings: sources, held-out results, findings
tools/real/              real-data fetch, tuning, training and held-out evaluators
DEPLOY.md                backend and Pages deployment guide
OUTREACH.md              launch and outreach copy
```

## Development

No build step. Serve the folder over HTTP (the mic needs a secure context;
`localhost` counts):

```bash
npm run serve                      # http://localhost:8080
npm test                           # unit + engine contract tests (~15 s)
npm run eval:quick                 # accuracy regression gate (~30 s on 4 cores)
npm run test:e2e                   # real app in headless Chromium, WAV files as the mic
pip install -r requirements.txt -r requirements-dev.txt && pytest   # server suite
```

CI (`.github/workflows/ci.yml`) runs all of the above plus a Docker smoke
test; the GitHub Pages deploy only publishes after CI passes, and publishes
only the static site.

## Deployment

See [`DEPLOY.md`](DEPLOY.md). In short: the PWA deploys to GitHub Pages from
`main`; the backend runs anywhere Docker or a Python 3.12 buildpack runs, with
`DATA_DIR` on a persistent volume. The app calls the backend endpoints in
`SERVER_URLS` in order (overridable with the repository variable `SERVER_URLS`,
no code change); if it shows "Offline", see *The app shows "Offline"* in
`DEPLOY.md`.
