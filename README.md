# 🦟 MosquitoNet — Acoustic Mosquito Surveillance

**Turn your phone into a disease-vector detector. Free. No install required.**

MosquitoNet uses your phone's microphone and a Goertzel filter bank to acoustically detect mosquito species in real time. Identified detections are crowdsourced anonymously to build a live global vector-density map.

→ **Open the app:** `index.html` — single-file PWA, no server required.

---

## Features

- 12 species including *Ae. aegypti*, *An. gambiae*, *Cx. quinquefasciatus*
- Harmonic filter bank (f₀ + 2f + 3f) with adaptive noise subtraction
- Spectral flatness pre-gate rejects fans / AC / speech
- GPS-native species filtering (shows only species present at your location)
- Live crowdsourced map (Leaflet + OpenStreetMap)
- Federated learning with differential privacy (ε=1.0)
- Works offline — full PWA, installable to home screen
- No account, no ads, no data sold

---

## Deployment

Drop `index.html` on any static host (GitHub Pages, Netlify, Cloudflare Pages):

```bash
# GitHub Pages — free, zero config
git checkout -b gh-pages
git push origin gh-pages
# → live at https://<user>.github.io/<repo>/
```

No build step. No dependencies to install. Single file.

---

## Ready-to-post copy

### Product Hunt

**Tagline:** Detect disease-carrying mosquitoes with your phone's microphone

**Description:**
MosquitoNet is a free, single-file PWA that turns any smartphone into an acoustic mosquito surveillance station. It uses a precision Goertzel filter bank (not FFT peak picking) to identify 12 mosquito species by their wingbeat frequency — including *Aedes aegypti* (Dengue/Zika), *Anopheles gambiae* (Malaria), and *Culex quinquefasciatus* (West Nile).

All detections are anonymously crowdsourced to a live global map. No account. No ads. No data sold. Works offline as a PWA.

**Topics:** Health, Open Source, Mobile, Citizen Science, PWA

---

### Hacker News — Show HN

```
Show HN: MosquitoNet – detect mosquito species acoustically with your phone (single-file PWA)

I built a browser-only mosquito detector that runs entirely in a single HTML file.
It uses a Gaussian-weighted Goertzel filter bank (not FFT) tuned to each species'
wingbeat frequency, with adaptive spectral subtraction to handle background noise.
Currently tracks 12 species including the main Dengue/Malaria/West Nile vectors.

Detections are crowdsourced to a live global map. No install, no account, no server
required — drop the file on any static host.

Frequencies calibrated against the Stanford Abuzz dataset and Arthur et al. 2014 (JASA).
Happy to discuss the signal processing approach.

[link]
```

---

### Reddit posts

**r/selfhosted / r/PWA:**
```
I made a single-file PWA mosquito detector that runs entirely in your browser

No server, no install — drop the HTML file anywhere. Uses a Goertzel filter bank
to detect 12 mosquito species by wingbeat frequency (including Dengue and Malaria
vectors). Crowdsources detections to a live map. Free and open.

[link]
```

**r/publichealth / r/citizenscience:**
```
Free tool to help map mosquito disease vectors in your area — runs on any smartphone

MosquitoNet uses your phone's microphone to acoustically detect mosquitoes and
crowdsources detections anonymously to a global map. Useful for field researchers,
health departments, and citizen scientists tracking Dengue/Malaria/West Nile spread.
No app store, no account — just open the link.

[link]
```

**r/MachineLearning / r/DSP:**
```
Mosquito species classifier using Gaussian-weighted Goertzel filter bank + federated learning

Built a real-time acoustic classifier for 12 mosquito species running in a browser.
Signal chain: 9-tap Gaussian-weighted Goertzel (f₀) + 5-tap banks (2f, 3f) +
adaptive spectral subtraction (Boll 1979) + tonality gate + mandatory harmonic presence.
Federated learning (FedAvg + DP, ε=1.0) aggregates updates from the network.

Happy to discuss the Goertzel vs STFT tradeoffs for narrow-band detection.

[link]
```

---

### Twitter/X thread starter

```
🦟 I built a free mosquito detector that runs in your browser.

Open the link → grant mic → it listens for wingbeat frequencies (290–730 Hz
depending on species) and tells you if you have Dengue/Malaria vectors nearby.

Detections are crowdsourced to a live global map.

No install. No account. Single HTML file.

[link]

🧵 How it works: a Goertzel filter bank tuned to each species' wingbeat frequency,
with adaptive noise subtraction so fans/AC don't trigger false positives.
Requires 2nd or 3rd harmonic to confirm it's biological (not a tone generator).

Data: Stanford Abuzz dataset + HumBugDB. 12 species. Works offline as a PWA.
```

---

### WhatsApp / Telegram forward

```
🦟 This is wild — free app that uses your phone mic to detect disease-carrying mosquitoes nearby. Open the link, tap Start, done.

[link]

Works for Dengue, Malaria, West Nile species. No install needed, runs in your browser.
```

---

## PWA directories to submit to

- https://appsco.pe
- https://pwa-directory.appspot.com
- https://findpwa.com
- https://progressiveapp.store
- https://web.dev/showcase (Google showcase — requires Lighthouse ≥90)

## Research / health org outreach

- WHO Vector Control team: vectorcontrol@who.int
- CDC Arboviral Disease Branch: dvbid@cdc.gov
- Abuzz project (Stanford): https://abuzz.stanford.edu
- HumBugDB authors (Imperial College): contact via paper DOI 10.1371/journal.pcbi.1009755
- iNaturalist citizen-science integration (contact via forum)
