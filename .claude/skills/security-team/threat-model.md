# MosquitoNet threat model

Keep this current: any change that adds an endpoint, a stored field, a
third-party origin, a secret or a deploy path updates it.

## System in one paragraph

A PWA (`index.html` + `js/` + `sw.js`, minified by `tools/build-site.sh`) runs
on-device mosquito detection from the microphone and shares confirmed finds
(species, confidence, frequency, an offset location) with a Flask backend
(`federated_server.py`, one gunicorn worker, state in RAM + an append-only
JSONL and a state file in `DATA_DIR` on a Railway volume). The same backend
serves the built app at `https://mosquito.charity`; GitHub Pages serves a copy
at `https://uriiger-sketch.github.io/mosquito/` that calls the API
cross-origin. Every push to `main` runs CI, then `deploy-pages.yml` builds the
site, pushes it with the server code to the `release` branch (Railway deploys
it) and to Pages. An operator can open an admin session with the server's
`ADMIN_PASSWORD`.

## Assets (what an attacker wants)

| Asset | Where | Why it matters |
|---|---|---|
| Admin session / `ADMIN_PASSWORD` | Railway env; token in one tab's memory; hashed tokens in server RAM | Unlocks stored detections (with shared locations), logs, files — deliberately no device list |
| Stored detections | `DATA_DIR/detections.jsonl` (stable device hash, 4-decimal client-offset coordinates, eventIds) | Linkable per device; closest data to a user's home |
| `.server_secret` / `PSEUDONYM_SECRET` | `DATA_DIR`, env | Undoes the public offsets and daily pseudonyms |
| Raw device id | Phone `localStorage`; sent in heartbeat/detection/upload bodies, hashed on arrival | Links a phone's reports across days |
| Client location offset | Phone `localStorage` | With it, a phone's reports point back at its home |
| Service availability | The single process | One process = one outage domain |
| Global federated model | Server RAM/state file | Poisoning degrades every user's detector |
| Public map / hotspots integrity | `/hotspots`, `/log` | Fake outbreaks mislead health decisions |
| Build & release pipeline | GitHub Actions, `release` branch, Railway | Code execution on every user's phone and the server |

## Adversaries

- **Anonymous internet client** — curl, scripts, botnets; chooses every field,
  rotates device ids and IPs (IPv6 /64s), replays, floods.
- **Malicious web page** — runs in a real user's or the admin's browser:
  cross-origin requests, CSRF-style POSTs (text/plain needs no preflight),
  attempts to read admin responses, clickjacking.
- **Curious data consumer** — reads public `/log`, `/log.txt`, `/hotspots`
  over time and tries to re-identify homes or link devices across days.
- **Network attacker** — on hostile Wi-Fi; TLS is the defence; `http://` LAN
  mode exists for development only.
- **Compromised dependency / action / CDN** — npm minifiers, pip packages,
  GitHub Actions referenced by tag, vendored Leaflet, OpenStreetMap tiles.
- **Someone with the phone** — reads `localStorage` (history, device id,
  offset). Out of scope beyond not storing secrets there.
- **Insider with Railway / GitHub access** — trusted; but limit blast radius
  (least-privilege tokens, no secrets in logs).

## Trust boundaries and entry points

| Boundary | Entry points | Existing controls |
|---|---|---|
| Internet → server | `POST /heartbeat`, `/detection`, `/federated/upload`; `GET /log`, `/log.txt`, `/hotspots`, `/health`, `/federated/model`, `/federated/stats`; site files; `OPTIONS` | Per-IP and per-device token buckets, 64 KB body cap, strict validation with bounded strings, control-char stripping, `_q()` log escaping, JSON-only errors, security headers, idempotent eventIds |
| Browser origin → server writes | All POSTs | Origin allowlist enforced server-side (403), reflected CORS only for first-party |
| Admin | `POST /admin/login`, `/admin/logout`; `GET /admin/*` | Off unless `ADMIN_PASSWORD`; constant-time compare; non-scaled login throttles; bearer token (hash stored), expiry, cap; first-party CORS only; SW never caches `/admin/`; `[Admin]` audit lines |
| Server → public | `/log`, `/log.txt`, `/hotspots`, stats | Allowlisted fields, secret per-device offset + rounding, daily keyed pseudonyms, ~5 km hotspot cells |
| Page → browser | DOM rendering of server and stored data | `esc()` on interpolation; CSP with the hashed inline script as a header (backend) AND a `<meta>` (both hosts, `tools/csp-meta.js`; `connect-src` = self + SERVER_URLS); `frame-ancestors 'none'` on the backend |
| Service worker | Fetch interception | API and `/admin/` never intercepted; network-first app code, one cached copy per file; only the scope's HTML becomes the offline shell; bounded tile cache |
| Phone → server (location) | `lat`/`lng` on detections | Snapped to a fixed 0.02° grid, then the phone's secret ≤ 1.5 km offset (`reportDetection`): survives storage wipes |
| Repo → production | `main` push → CI → `release` branch → Railway; Pages | Full CI gate incl. the image test; build job read-only (npm code, no credentials/cache/scripts), deploy job writes but runs no npm code; actions pinned by SHA; build refuses symlinks; server serves only regular files inside SITE_DIR |

## Known accepted risks (re-evaluate, don't rediscover)

- Device ids are client-chosen: per-device limits only catch one misbehaving
  install; per-IP limits are the real abuse control.
- GitHub Pages cannot set response headers: the Pages copy has the `<meta>`
  CSP but no `frame-ancestors` (a meta can't carry it) and no other headers.
- Slow-header clients: gthread reads headers with no timeout, so ~4 stalled
  connections block the one worker; only the platform edge can enforce read
  timeouts (documented in DEPLOY.md). Never "fix" it with more workers.
- `/hotspots` counts cells of the stored (client-offset) point, without the
  server's public offset; the 5 km cells and the client grid bound what that
  reveals. Changing it needs a cell rebuild from the log.
- The private JSONL keeps the stable device hash and client-offset points with
  no retention limit (proposed: `RETENTION_DAYS` rewriting old rows to their
  public projection). Insider/backup exposure only.
- Model updates are noised but give no meaningful differential privacy
  (ε ≈ 400 per step); they carry 16 weights, no audio or location.
- The client grid keeps an install in one cell (≈ 280 m hysteresis), but a
  home within ~280 m of a cell edge still lands in either cell across WIPED
  storage sessions; the server's public offset hides it from public outputs.
- The Pages copy's `script-src 'self'` is the whole `uriiger-sketch.github.io`
  origin: every Pages site of that account. Keep third-party code off them.
- `/federated/upload` is sent as `application/json` (a CORS preflight), unlike
  the other writes; it is not on the no-lost-detections path.
- Releases up to v3.1.11 shipped an unsalted SHA-256 of the old in-page
  Advanced password; it is public forever (history, caches). That password
  must never be an `ADMIN_PASSWORD`.
- The global admin-login bucket can be exhausted by a distributed guesser,
  delaying legitimate admin logins (never granting access).
- Public reads are deliberately unauthenticated and CORS `*`.
