# Privacy & data protection — `sec-privacy`

Scope: every byte that leaves the phone, everything the server keeps, and
everything it publishes or shows the admin. Files: the upload/heartbeat code
in `index.html` (`FedClient`, `_locationOffset`, `reportDetection`,
`initServer`), `federated_server.py` (`_public_row`, `_geo_offset`,
`_pseudonym`, hotspots, admin views, logging), `README.md` / `DEPLOY.md`
privacy statements. You think like a data consumer who scrapes `/log` for
months, a curious admin, and a regulator reading the privacy promise.

## The promise (README → Privacy, onboarding text, About)

No audio leaves the device; no account; a detection carries
species/genus, confidence, frequency and a location moved by a fixed secret
per-device offset ≤ ~1.5 km; the server adds a second offset before
publishing and rotates pseudonyms daily; history stays on the device. The
admin view shows only what devices shared.

## Checklist

**Re-identification**
- [ ] Public `/log`: can rows be linked across days despite daily
      pseudonyms (timing, unique species/frequency fingerprints, appVersion,
      snr, posterior vectors, `conf_updated`, sequential ids)? Can averaging
      within one day reveal more than intended?
- [ ] Does any public field leak the stable device hash, eventId or raw
      coordinates? (`_PUBLIC_KEYS` allowlist; `device=` filter on `/log`.)
- [ ] `/hotspots` cell keys: resolution vs. a single household.
- [ ] Does the client offset survive storage loss correctly (no location sent
      if it can't be kept)? Is the jitter adequate vs. the documented model?
- [ ] Timestamps: client `ts` precision (ms) — needed? Rounding helps unlinkability.

**Minimisation & retention**
- [ ] Is every stored field necessary? (`eid`, full posterior, `snr`,
      `appVersion`, `asymptomatic`, `disease` strings.)
- [ ] Retention: the JSONL grows forever. Is there a documented retention
      period or an operator tool to expire old rows? Propose, don't delete data.
- [ ] Logs (`_log`, gunicorn access log, Railway HTTP log): any device id,
      IP + id pairing, coordinates, or admin token? IPs in admin-login logs are
      a deliberate security trade-off — keep but note.

**Admin exposure**
- [ ] Admin views limited to what's documented; no raw device id; coordinates
      are the stored client-offset values only; no bulk export beyond need.
- [ ] Admin session can't outlive its purpose (expiry, logout, reload).

**Transparency**
- [ ] Privacy text in the app matches what the code does (any drift is a
      finding — fix the code or the text).

## Techniques

Simulate a scraper: seed a local server with many reports from a few
synthetic devices (fixed "homes"), pull `/log` and `/hotspots` as the public
would, and try to recover homes or link devices across days. Report the
achievable error in metres.
