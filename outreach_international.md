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

The HumBug project's open data made this work possible, and I'm grateful for it. I tuned the detector on the MozzBNN phone recordings (CDC cultures, Ifakara and the DRC), using the first half of each recording for development and the second half only for testing, and kept the Zooniverse-labelled Culex recordings as a held-out test that played no part in tuning.

The results are modest, and I've tried to report them plainly. Against the 94 human-labelled flight episodes, the default setting detects 12.8% and a more sensitive setting 37%, with about 1.8 false alarms per hour on the ESC-50 everyday-sound set. Very brief culture-cup bursts are still mostly missed, and a male Culex at around 640 Hz can't be told from a female Aedes by wingbeat alone, so the app now models males explicitly and tells the user when a detection is probably male.

I would be very grateful for your help in two ways. First, a critical look at the method, from you or from whoever in your group is best placed, would be extremely valuable to me. Second, I'm looking for recordings to run preliminary tests beyond the public sets, ideally phone recordings of free-flying mosquitoes in real homes at measured distances. If those tests are successful, I would warmly welcome a proper collaboration, with full credit to HumBug, which is already acknowledged in the app.

App: https://uriiger-sketch.github.io/mosquito/
Code: https://github.com/uriiger-sketch/mosquito
Real-data method and results: https://github.com/uriiger-sketch/mosquito/blob/main/docs/REAL_DATA.md

You can reach me by return email, or by phone or WhatsApp on [Phone]. Thank you very much for your time.

With best regards,
[Your name]
[Phone] (phone and WhatsApp)

---

## Email 13 — Prof. Manu Prakash

**To:** manup@stanford.edu
**Subject:** Phone-microphone mosquito detection, building on Abuzz: a short request

Dear Prof. Prakash,

Your group's Abuzz study (Mukundarajan et al., eLife 2017) showed that ordinary mobile phones can record mosquito wingbeats well enough for surveillance, and it has been a real inspiration for my own work. MosquitoNet is a free, open-source web app that tries to take that idea a step further: it runs in any phone browser, listens for wingbeat tones in real time, processes audio on the device without storing it, and reports confirmed detections to a public map at a coarse, privacy-blurred location.

Part of the real-data tuning uses the CDC culture recordings made for the Abuzz study, as redistributed in Oxford's HumBug MozzBNN set. The results so far are modest: on an independent, human-labelled set of Culex phone recordings, the default setting detects 12.8% of flight episodes and a more sensitive setting 37%, and very brief bursts in cup recordings are mostly missed.

I would be very grateful for your critique of the approach, or that of someone in your lab, on whether it is sound and where it is weakest. I'm also looking for recordings to run preliminary tests, ideally phone recordings of free-flying mosquitoes at known distances; if any from the Abuzz work could be shared, they would help a great deal. If the preliminary tests are successful, I would warmly welcome the chance to collaborate. Abuzz is credited in the app and will be in anything I write up.

App: https://uriiger-sketch.github.io/mosquito/
Code: https://github.com/uriiger-sketch/mosquito
Real-data method and results: https://github.com/uriiger-sketch/mosquito/blob/main/docs/REAL_DATA.md

You can reach me by return email, or by phone or WhatsApp on [Phone]. Thank you very much for your time.

With best regards,
[Your name]
[Phone] (phone and WhatsApp)

---

## Email 14 — Dr. Lauren Cator

**To:** l.cator@imperial.ac.uk
**Subject:** Male versus female wingbeats in a phone-based mosquito detector

Dear Dr. Cator,

I'm developing MosquitoNet, a free, open-source web app that detects mosquitoes from their wingbeat tone using an ordinary phone microphone, with all processing on the device. Your work on harmonic convergence (Cator et al., Science 2009) has turned out to bear directly on a problem I didn't expect.

When I tested the detector on real phone recordings of caged Culex from Oxford's HumBug data, most detections came from flying males at around 640 Hz, which a wingbeat alone cannot distinguish from a female Aedes. The app now models a male component for each species and tells the user when a detection is probably male, since males don't bite, but species identification in male-dominated settings remains poor.

I would be very grateful for your critique of this reasoning, and for your view on whether acoustic cues beyond the fundamental, such as harmonic structure or frequency modulation, could help separate the sexes in short, noisy phone recordings. I'm also looking for recordings to run preliminary tests; sexed recordings from your insectary, if they could ever be shared, would be ideal. If those tests are successful, I would warmly welcome a collaboration, with all results shared and full credit to your group.

App: https://uriiger-sketch.github.io/mosquito/
Code: https://github.com/uriiger-sketch/mosquito
Real-data method and results: https://github.com/uriiger-sketch/mosquito/blob/main/docs/REAL_DATA.md

You can reach me by return email, or by phone or WhatsApp on [Phone]. Thank you very much for your time.

With best regards,
[Your name]
[Phone] (phone and WhatsApp)

---

## Email 15 — Prof. Jörg Albert

**To:** joerg.albert@uni-oldenburg.de
**Subject:** A phone-based mosquito detector, and what the mosquito ear might teach it

Dear Prof. Albert,

I'm developing MosquitoNet, a free, open-source web app that detects mosquitoes from their wingbeat tone using an ordinary phone microphone, with all processing on the device. Your group's recent PNAS paper on a mosquito-inspired framework for acoustic signal detection (Faber et al., 2025) caught my attention, because my detector faces the same problem from the outside: picking out a faint, frequency-modulated flight tone in a noisy room.

On real phone recordings from Oxford's HumBug data, mosquito flight is audible mostly in sub-second bursts at only about 5 dB above the local noise, and at its default setting the detector finds 12.8% of human-labelled Culex flight episodes. Sex is a further complication: a male Culex near 640 Hz is indistinguishable by wingbeat alone from a female Aedes.

I would be very grateful for your critique of the approach, and for your view on whether principles from the mosquito ear could inform a better detector. I'm also looking for recordings to run preliminary tests; flight-tone recordings of known sex at known distances, if your lab could share any, would be especially valuable. If those tests are successful, I would warmly welcome a collaboration. The code is fully open, and any contribution would be fully credited.

App: https://uriiger-sketch.github.io/mosquito/
Code: https://github.com/uriiger-sketch/mosquito
Real-data method and results: https://github.com/uriiger-sketch/mosquito/blob/main/docs/REAL_DATA.md

You can reach me by return email, or by phone or WhatsApp on [Phone]. Thank you very much for your time.

With best regards,
[Your name]
[Phone] (phone and WhatsApp)

---

## Email 16 — Prof. Eamonn Keogh

**To:** eamonn.keogh@ucr.edu
**Subject:** Phone-microphone wingbeat detection: would you critique it?

Dear Prof. Keogh,

I'm developing MosquitoNet, a free, open-source web app that detects mosquitoes from their wingbeat tone using an ordinary phone microphone, with all processing on the device. I know that your 2014 paper with Chen and colleagues argued that acoustic, wingbeat-frequency approaches had made little lasting impact and that inexpensive optical sensors do better. My results on real data are consistent with much of that critique, which is exactly why I'm writing to you.

The detector tracks the wingbeat fundamental and its harmonics, adds a learned plausibility model, and reports calibrated uncertainty rather than a single confident species. On an independent, human-labelled set of Culex phone recordings it detects 12.8% of flight episodes at the default setting and 37% at a more sensitive one, with about 1.8 false alarms per hour on the ESC-50 everyday-sound set. Species identification from frequency alone is weak, particularly between male Culex and female Aedes.

Would you be willing to look at the method and tell me where it is most likely to be fooling itself? I would value a frank critique far more than encouragement. I'm also looking for recordings to run preliminary tests, and if any of your group's wingbeat data would be suitable, I would be glad to use it with full credit. If those tests are successful, I would warmly welcome a collaboration. The code, evaluation harness and real-data results are all open.

App: https://uriiger-sketch.github.io/mosquito/
Code: https://github.com/uriiger-sketch/mosquito
Real-data method and results: https://github.com/uriiger-sketch/mosquito/blob/main/docs/REAL_DATA.md

You can reach me by return email, or by phone or WhatsApp on [Phone]. Thank you very much for your time.

With best regards,
[Your name]
[Phone] (phone and WhatsApp)

---

## Email 17 — Dr. Pie Müller

**To:** pie.mueller@swisstph.ch
**Subject:** Phone-based mosquito detection and Aedes surveillance in Switzerland

Dear Dr. Müller,

I'm developing MosquitoNet, a free, open-source web app that detects mosquitoes from their wingbeat tone using an ordinary phone microphone, with all processing on the device and detections reported to a map at a coarse, privacy-blurred location. Your surveillance of invasive Aedes along Swiss traffic axes (Müller et al., PLOS NTD 2020) made me wonder whether many phones could add a cheap, dense layer of reports alongside ovitraps, as Ae. albopictus continues to spread north of the Alps.

The app includes Ae. albopictus and Ae. japonicus with regional settings for Switzerland, but I should be clear about its limits. On real phone recordings it detects only a minority of mosquito flight episodes, it cannot reliably separate species by wingbeat alone, and it has not been tested in Swiss conditions.

I would be very grateful for your critical view on whether reports like these could be useful to surveillance at all, and if so, what quality they would need. I'm also looking for recordings to run preliminary tests; if the Vector Biology Unit could share, or allow me to make, recordings of live Ae. albopictus and Ae. japonicus, that would be extremely helpful, and I would gladly do the recording work myself. If those tests are successful, I would warmly welcome a collaboration, for example in field or semi-field trials.

App: https://uriiger-sketch.github.io/mosquito/
Code: https://github.com/uriiger-sketch/mosquito
Real-data method and results: https://github.com/uriiger-sketch/mosquito/blob/main/docs/REAL_DATA.md

You can reach me by return email, or by phone or WhatsApp on [Phone]. Thank you very much for your time.

With best regards,
[Your name]
[Phone] (phone and WhatsApp)
