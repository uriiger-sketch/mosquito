# Real recordings used to validate and train the detector

None of this audio is committed to the repository. `tools/real/fetch.sh` clones
the exact versions below; every result in `docs/REAL_DATA.md` was produced from them.

| Source | What it contains | Licence | Version |
|---|---|---|---|
| **HumBug project, University of Oxford — [MozzBNN](https://github.com/HumBug-Mosquito/MozzBNN)** (Kiskin et al.; the CDC recordings were made for the Abuzz study, Mukundarajan et al. 2017, *eLife* 6:e27854) | 8 kHz phone recordings: CDC insect-culture *Ae. aegypti* (17.7 min), *Ae. albopictus* (16.6 min), *Cx. quinquefasciatus* (18.6 min), speech segments labelled; Ifakara Health Institute (Tanzania) field recordings (11.5 min); DR Congo field recordings (12.6 min); non-mosquito backgrounds: bus stop (30 min), construction site (30 min), indoor speech (6 min) | MIT (repository) | `8a808ea` |
| **HumBugDB — [Kiskin et al., NeurIPS 2021 Datasets & Benchmarks](https://github.com/HumBug-Mosquito/HumBugDB)** | Metadata of 9,295 recordings (33.5 h; species, sex, device, country). Its audio is on Zenodo (record 4904800), which was not reachable from the build environment. | MIT (code/metadata), CC-BY-4.0 (data) | `5065675` |
| **ESC-50 — [Piczak, ACM MM 2015](https://github.com/karolpiczak/ESC-50)** | 2,000 labelled 5-s real recordings, 50 everyday sound classes (snoring, breathing, fans, rain, insects, speech-like sounds, household appliances …). Used ONLY as a held-out false-alarm test. | CC BY-NC 3.0 (non-commercial; used for evaluation only, never redistributed) | `33c8ce9` |

Split discipline: every MozzBNN recording is split in time — the first half is
the dev split (training and threshold choice), the second half is the test
split (reported only). ESC-50 is never used for any choice.
