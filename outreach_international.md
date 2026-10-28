# MosquitoNet — international outreach (6 researchers)

One personal email each, sent separately. Figures quoted are from docs/REAL_DATA.md on main (2026-10-28).

## Who

| # | Researcher | Current post | Email | Where the address is published | Confidence |
|---|---|---|---|---|---|
| 12 | Prof. Stephen Roberts | RAEng/Man Group Professor of Machine Learning, Engineering Science, Oxford (HumBug) | sjrob@robots.ox.ac.uk | Oxford Open Data vacancy contact record (data.ox.ac.uk/doc/vacancy/176519/contact) | medium-high |
| 13 | Prof. Manu Prakash | Associate Professor of Bioengineering, Stanford (Abuzz) | manup@stanford.edu | prakashlab.stanford.edu/contact; biology.stanford.edu profile | high |
| 14 | Dr. Lauren Cator | Associate Professor, Life Sciences, Imperial College London (Silwood Park) | l.cator@imperial.ac.uk | profiles.imperial.ac.uk/l.cator; Imperial directory | high |
| 15 | Prof. Jörg Albert | Professor of Sensory Physiology and Behaviour, Univ. of Oldenburg (moved from UCL Ear Institute, 2023) | joerg.albert@uni-oldenburg.de | UCL profile (profiles.ucl.ac.uk/2226), "Work" email | medium |
| 16 | Prof. Eamonn Keogh | Distinguished Professor of CSE, UC Riverside | eamonn.keogh@ucr.edu | UCR Profiles directory; UCR News expert listing 2024 (alt: eamonn@cs.ucr.edu) | high |
| 17 | Dr. Pie Müller | Head, Vector Biology Unit, Swiss TPH, Basel | pie.mueller@swisstph.ch | swisstph.ch Vector Biology unit page | high |

Addresses come from search-engine extracts of these pages; the pages could not be opened from the sandbox.
Backup: Dr. Ivan Kiskin (Univ. of Surrey; HumBugDB first author) — no published email found.

---

## Email 12 — Prof. Stephen Roberts

**To:** sjrob@robots.ox.ac.uk
**Subject:** A browser-based mosquito detector built on HumBug recordings: a request for advice

Dear Prof. Roberts,

I'm developing MosquitoNet, a free, open-source web app that detects mosquitoes from their wingbeat tone using an ordinary phone microphone. It runs entirely in the browser, processes audio on the device and stores none of it; confirmed detections go to a public map at a coarse, privacy-blurred location.

The HumBug project's open data made this work possible. I tuned the detector on the MozzBNN phone recordings (CDC cultures, Ifakara and the DRC), using the first half of each recording for development and the second half only for testing, and kept the Zooniverse-labelled Culex recordings as a held-out test that played no part in tuning.

The results are modest, and I've tried to report them plainly. Against the 94 human-labelled flight episodes, the default setting detects 12.8% and a more sensitive setting 37%, with about 1.8 false alarms per hour on the ESC-50 everyday-sound set. Very brief culture-cup bursts are still mostly missed, and a male Culex at around 640 Hz can't be told from a female Aedes by wingbeat alone, so the app now models males explicitly and tells the user when a detection is probably male.

Your group's experience would help a great deal on two questions. Would you be willing to look over the method, or point me to whoever in the group would be best placed to? And do you know of phone recordings of free-flying mosquitoes in real homes, ideally at measured distances? That is the gap that matters most now, and I haven't found it in the public datasets. HumBug is credited in the app, and will be in anything I write up.

App: https://uriiger-sketch.github.io/mosquito/
Code: https://github.com/uriiger-sketch/mosquito
Real-data method and results: https://github.com/uriiger-sketch/mosquito/blob/main/docs/REAL_DATA.md

With thanks and best regards,
[Your name]
[Phone]

---

## Email 13 — Prof. Manu Prakash

**To:** manup@stanford.edu
**Subject:** Phone-microphone mosquito detection, building on Abuzz: a short request

Dear Prof. Prakash,

Your group's Abuzz study (Mukundarajan et al., eLife 2017) showed that ordinary mobile phones can record mosquito wingbeats well enough for surveillance. I've been trying to take that idea a step further. MosquitoNet is a free, open-source web app that runs in any phone browser, listens for wingbeat tones in real time, processes audio on the device without storing it, and reports confirmed detections to a public map at a coarse, privacy-blurred location.

Part of the real-data tuning uses the CDC culture recordings made for the Abuzz study, as redistributed in Oxford's HumBug MozzBNN set. The results so far are modest: on an independent, human-labelled set of Culex phone recordings, the default setting detects 12.8% of flight episodes and a more sensitive setting 37%, and very brief bursts in cup recordings are mostly missed.

If you or someone in your lab has a few minutes, I'd value your view on two things: whether the approach looks sound to you, and whether you know of phone recordings of free-flying mosquitoes in homes at known distances. That is the gap that matters most now. Abuzz is credited in the app and will be in anything I write up.

App: https://uriiger-sketch.github.io/mosquito/
Code: https://github.com/uriiger-sketch/mosquito
Real-data method and results: https://github.com/uriiger-sketch/mosquito/blob/main/docs/REAL_DATA.md

With thanks and best regards,
[Your name]
[Phone]

---

## Email 14 — Dr. Lauren Cator

**To:** l.cator@imperial.ac.uk
**Subject:** Male versus female wingbeats in a phone-based mosquito detector

Dear Dr. Cator,

I'm developing MosquitoNet, a free, open-source web app that detects mosquitoes from their wingbeat tone using an ordinary phone microphone, with all processing on the device. Your work on harmonic convergence (Cator et al., Science 2009) has turned out to bear directly on a problem I didn't expect.

When I tested the detector on real phone recordings of caged Culex from Oxford's HumBug data, most detections came from flying males at around 640 Hz, which a wingbeat alone cannot distinguish from a female Aedes. The app now models a male component for each species and tells the user when a detection is probably male, since males don't bite, but species identification in male-dominated settings remains poor.

I'd value your view on two things: whether there are acoustic cues beyond the fundamental, such as harmonic structure or frequency modulation, that could help separate the sexes in short, noisy phone recordings; and whether sexed recordings from your insectary might ever be available for testing. I would share all results and credit your group.

App: https://uriiger-sketch.github.io/mosquito/
Code: https://github.com/uriiger-sketch/mosquito
Real-data method and results: https://github.com/uriiger-sketch/mosquito/blob/main/docs/REAL_DATA.md

Best regards,
[Your name]
[Phone]

---

## Email 15 — Prof. Jörg Albert

**To:** joerg.albert@uni-oldenburg.de
**Subject:** A phone-based mosquito detector, and what the mosquito ear might teach it

Dear Prof. Albert,

I'm developing MosquitoNet, a free, open-source web app that detects mosquitoes from their wingbeat tone using an ordinary phone microphone, with all processing on the device. Your group's recent PNAS paper on a mosquito-inspired framework for acoustic signal detection (Faber et al., 2025) caught my attention, because my detector faces the same problem from the outside: picking out a faint, frequency-modulated flight tone in a noisy room.

On real phone recordings from Oxford's HumBug data, mosquito flight is audible mostly in sub-second bursts at only about 5 dB above the local noise, and at its default setting the detector finds 12.8% of human-labelled Culex flight episodes. Sex is a further complication: a male Culex near 640 Hz is indistinguishable by wingbeat alone from a female Aedes.

I'd be grateful for your view on whether principles from the mosquito ear could inform a better detector, and whether your lab might have flight-tone recordings of known sex at known distances. The code is fully open, and any contribution would be credited.

App: https://uriiger-sketch.github.io/mosquito/
Code: https://github.com/uriiger-sketch/mosquito
Real-data method and results: https://github.com/uriiger-sketch/mosquito/blob/main/docs/REAL_DATA.md

Kind regards,
[Your name]
[Phone]

---

## Email 16 — Prof. Eamonn Keogh

**To:** eamonn.keogh@ucr.edu
**Subject:** Phone-microphone wingbeat detection: would you critique it?

Dear Prof. Keogh,

I'm developing MosquitoNet, a free, open-source web app that detects mosquitoes from their wingbeat tone using an ordinary phone microphone, with all processing on the device. I know that your 2014 paper with Chen and colleagues argued that acoustic, wingbeat-frequency approaches had made little lasting impact and that inexpensive optical sensors do better. My results on real data are consistent with much of that critique, which is exactly why I'm writing.

The detector tracks the wingbeat fundamental and its harmonics, adds a learned plausibility model, and reports calibrated uncertainty rather than a single confident species. On an independent, human-labelled set of Culex phone recordings it detects 12.8% of flight episodes at the default setting and 37% at a more sensitive one, with about 1.8 false alarms per hour on the ESC-50 everyday-sound set. Species identification from frequency alone is weak, particularly between male Culex and female Aedes.

Would you be willing to look at the method and tell me where it is most likely to be fooling itself? I'd value a frank critique more than encouragement. The code, evaluation harness and real-data results are all open.

App: https://uriiger-sketch.github.io/mosquito/
Code: https://github.com/uriiger-sketch/mosquito
Real-data method and results: https://github.com/uriiger-sketch/mosquito/blob/main/docs/REAL_DATA.md

Best regards,
[Your name]
[Phone]

---

## Email 17 — Dr. Pie Müller

**To:** pie.mueller@swisstph.ch
**Subject:** Phone-based mosquito detection and Aedes surveillance in Switzerland

Dear Dr. Müller,

I'm developing MosquitoNet, a free, open-source web app that detects mosquitoes from their wingbeat tone using an ordinary phone microphone, with all processing on the device and detections reported to a map at a coarse, privacy-blurred location. Your surveillance of invasive Aedes along Swiss traffic axes (Müller et al., PLOS NTD 2020) made me wonder whether many phones could add a cheap, dense layer of reports alongside ovitraps, as Ae. albopictus continues to spread north of the Alps.

The app includes Ae. albopictus and Ae. japonicus with regional settings for Switzerland, but I should be clear about its limits. On real phone recordings it detects only a minority of mosquito flight episodes, it cannot reliably separate species by wingbeat alone, and it has not been tested in Swiss conditions. I'd value your view on whether reports like these could be useful to surveillance at all, and if so, what quality they would need. If the Vector Biology Unit ever runs field or semi-field trials, I would gladly provide phones and do the recording work.

App: https://uriiger-sketch.github.io/mosquito/
Code: https://github.com/uriiger-sketch/mosquito
Real-data method and results: https://github.com/uriiger-sketch/mosquito/blob/main/docs/REAL_DATA.md

Kind regards,
[Your name]
[Phone]
