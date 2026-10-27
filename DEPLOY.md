# MosquitoNet — Complete Deployment Guide

## ⚠️ READ FIRST: set DATA_DIR to a persistent volume

`DATA_DIR` defaults to `/tmp`. On Railway, Render, Heroku and Fly, `/tmp` is
**ephemeral** — it is wiped on every restart, redeploy and daily container
recycle. If you go live without setting it you will **permanently lose every
detection, all statistics and the trained federated model**, roughly daily, with
no warning and no way to recover them.

The server logs an error at startup when it detects this, and `GET /health`
reports `"ephemeral": true` plus `ephemeral_reason`. It catches every case seen
in practice: `DATA_DIR` under `/tmp`, a Railway service with **no volume
attached** (production ran like that until October 2026 — the old check called
it persistent), `DATA_DIR` outside the Railway volume, and a container with no
volume mounted at `DATA_DIR`.

**Railway**
1. Service → **Variables** → add `DATA_DIR = /data`
2. Service → **Settings → Volumes** → **Add Volume**, mount path `/data`
3. Redeploy. Confirm `curl https://YOUR-URL/health` shows `"ephemeral": false`
   and `"writable": true`.

**Render**
1. Service → **Disks** → **Add Disk**, mount path `/data` (1 GB is plenty)
2. Service → **Environment** → add `DATA_DIR = /data`
3. Redeploy and check `/health` as above.

**Docker / VPS / Fly** — the `Dockerfile` sets `DATA_DIR=/data` but does **not**
declare `VOLUME /data` (Railway refuses to build an image that does — that broke
every Railway deploy from June to October 2026). So mount a volume yourself:
`docker run -v mosquitonet_data:/data ...` — without one, data lives in the
container's writable layer and is lost when the container is recreated.

The container runs the server as an unprivileged user (uid/gid **10001**). It
starts as root only long enough to `chown` the data directory — platform volumes
are mounted root-owned, and data written by images before v13 is root-owned —
then drops all privileges. With a host bind mount (`-v /srv/mn:/data`) that means
the host directory becomes owned by 10001. If you start the container with
`--user`, prepare the directory yourself: `chown -R 10001:10001 /srv/mn`.

**Verify persistence for real** — the only check that matters:
```bash
curl -s https://YOUR-URL/health | grep -o '"total_detections":[0-9]*'
# restart/redeploy the service, then run it again — the number must NOT reset.
```

Capacity note: the server keeps the most recent `DETECTION_LOG_MAX` (default
20000) detections in RAM for `/log`; the on-disk JSONL keeps the full history and
is append-only. Raise it with the `DETECTION_LOG_MAX` env var if you have memory
to spare, but it is the setting that keeps memory bounded — don't remove it.
The JSONL grows forever (~300–500 bytes per detection): watch `disk_free_mb` in
`/health`. When the disk fills, detections are refused with `503` (clients keep
them queued and retry) and `/health` reports `"status": "degraded"` — nothing is
lost, but nothing new is stored until you free or grow the disk.

**Run exactly one server process per `DATA_DIR`** (`--workers 1`; threads are
fine). All state lives in that process; a second writer silently corrupts it.
The server takes a lock on `DATA_DIR` and logs a loud warning plus
`"single_instance": false` in `/health` if it ever finds another process there.

---

## Server configuration (environment variables)

All optional except `DATA_DIR`. Invalid values are logged and replaced by the
default (a typo never stops the server from booting).

| Variable | Default | What it does |
|---|---|---|
| `DATA_DIR` | `/tmp` (`/data` in Docker) | Where `detections.jsonl` and `mosquitonet_state.json` live. **Must be a persistent volume.** |
| `PORT` | `5001` standalone, `8080` Docker | Listen port (gunicorn binds `0.0.0.0:$PORT`). |
| `ALLOWED_ORIGINS` | — | Comma-separated **extra** origins allowed to POST (e.g. `https://my-fork.netlify.app`). The GitHub Pages app (`https://uriiger-sketch.github.io`), `https://mosquito.charity` and its subdomains, and `localhost`/`127.0.0.1` on any port are always allowed. `ALLOWED_ORIGIN` (older name) is read too. |
| `TRUSTED_PROXY_HOPS` | `1` | Reverse proxies in front of gunicorn whose `X-Forwarded-For` is trusted for per-IP rate limits. `1` for Railway / Render / Fly / nginx / Caddy, `2` for Cloudflare → Railway, **`0` if gunicorn is exposed directly** (otherwise clients can spoof their IP). |
| `RATE_LIMIT_SCALE` | `1` | Multiplies every rate limit below. `0` disables rate limiting. |
| `DETECTION_LOG_MAX` | `20000` | Detections kept in RAM for `/log`. |
| `SEEN_EVENTS_MAX` | `20000` | Size of the idempotency window (eventIds remembered to drop client retries). |
| `DETECTION_CELLS_MAX` | `20000` | Hotspot cells tracked (least recently hit cells are evicted first). |
| `PUBLIC_GEO_OFFSET` | `0.02` | Max secret per-device shift (degrees, ≈ 2.2 km) applied to coordinates in the public `/log` and `/log.txt`. `0` disables it (not recommended). |
| `PUBLIC_COORD_DECIMALS` | `2` | Decimals of lat/lng shown by the public `/log` and `/log.txt` (2 ≈ 1.1 km). The private JSONL keeps 4. |
| `PSEUDONYM_SECRET` | generated once | Key for the public device pseudonyms and offsets. By default a random key is created on first boot in `DATA_DIR/.server_secret` (mode 0600) and reused; set this only to manage the key yourself. Changing it re-randomises every public offset. |
| `FED_CLIP_NORM` | `2.0` | Max L2 distance one federated upload can pull the model per round. |
| `LOG_FORMAT` | `json` on Railway, else `text` | `json`: one JSON object per log line (server **and** gunicorn), which Railway parses into levels and filterable fields. `text`: plain lines. See *Logs*. |
| `ACCESS_LOG` | `errors` | Which requests gunicorn logs: `errors` (4xx/5xx except throttling and 404/405 scanner noise, plus requests slower than 1 s), `all`, or `off`. Railway's HTTP log records every request anyway. |
| `STATS_LOG_INTERVAL` | `900` | Seconds between `[Stats]` summary lines (traffic, rejections, write errors, free disk since the previous line). `0` disables them. |

gunicorn's own settings (one worker, threads, timeouts, logging) live in
`gunicorn.conf.py`, which gunicorn loads from its working directory however it
is started — so no start command needs flags.

### Rate limits

Token buckets (burst, then sustained rate), multiplied by `RATE_LIMIT_SCALE`.
Per-IP limits are generous because whole campuses and mobile carriers share one
address; per-device limits sit well above what the app actually sends.

| Bucket | Burst | Sustained | Applies to |
|---|---|---|---|
| per IP, everything | 600 | 20 / s | all requests except `/health` and `OPTIONS` |
| per IP, writes | 120 | 1 / s | `POST /detection`, `POST /federated/upload` |
| per IP, model uploads | 10 | 1 / min | `POST /federated/upload` (caps how many learning rounds one address can drive) |
| per IP, logs | 30 | 1 / 2 s | `/log`, `/log.txt` |
| per IP, health | 120 | 2 / s | `/health` only (separate, so probes never starve behind busy traffic) |
| per device, detections | 60 | 15 / min | `POST /detection` (burst = draining an offline queue) |
| per device, heartbeats | 60 | 60 / min | `/heartbeat` (one open app sends ~9 / min) |
| per device, uploads | 6 | 2 / min | `POST /federated/upload` |

Throttled requests get `429` + `Retry-After` — except `/detection`, which gets
`503` + `Retry-After`, because deployed apps delete a queued find on any 4xx but
retry 5xx with backoff. A throttled find is delayed, never lost.

### Health check fields

`GET /health` (its own generous rate-limit bucket) returns the public stats plus:

| Field | Meaning |
|---|---|
| `status` | `ok`, or `degraded` when the last write to the detection log failed (disk full / volume gone). HTTP status stays 200. |
| `service` | `MosquitoNet v13` — use it to confirm a deploy went live. |
| `writable`, `ephemeral` | `DATA_DIR` is writable / would be lost on redeploy (must be `true` / `false` in production). |
| `ephemeral_reason` | Why storage is ephemeral (`no Railway volume is attached to this service`, `DATA_DIR is under /tmp`, …), `null` when it is persistent. |
| `single_instance` | `false` = another process holds `DATA_DIR` — fix immediately. |
| `disk_free_mb` | Free space on the `DATA_DIR` volume. |
| `last_state_save` | Time of the last successful state-file save (every ~5 s while there is activity). |
| `client_ip` | The address rate limits use for *your* request. **Check it once after deploying:** it must be your own public IP. A Cloudflare/Railway/proxy address means `TRUSTED_PROXY_HOPS` is wrong and all users share one bucket. |
| `counters` | Since process start: `rejected_detections`, `rejected_uploads`, `rejected_origin`, `rate_limited`, `clipped_updates`, `dropped_fields`, `write_errors`. A climbing `rejected_origin` / `rate_limited` / `clipped_updates` means someone is probing or poisoning. |
| `python`, `log_cap`, `seen_events`, `min_conf`, `data_dir`, `log_file` | Runtime facts (Python is reported as major.minor only). |

### Privacy of the public log

`/log` and `/log.txt` are public. The app jitters each report by up to ~1.4 km,
but independently per report, so averaging one phone's reports used to reveal
its home to ~100 m (rounding doesn't stop that — the jitter is wider than the
grid). The public log therefore shifts every phone's coordinates by its own
**secret, constant** offset of up to `PUBLIC_GEO_OFFSET` (≈ 2.2 km) before
rounding: averaging any number of reports, over any number of days, lands on
the offset point, not the home. Devices appear as a pseudonym that changes every
UTC day. The stable device hash, eventIds and 4-decimal (client-jittered)
coordinates stay in the private JSONL; `/hotspots` uses ~5 km cells.
Keep `DATA_DIR/.server_secret` with the data — losing it re-randomises the
offsets, and averaging across old and new offsets weakens the protection.
Request logs never contain the raw device id: the Docker image's access log
omits query strings and client IPs.

---

## What goes where

```
mosquito/
├── index.html                ← PWA app (GitHub Pages)
├── sw.js                     ← service worker (offline support)
├── manifest.webmanifest      ← PWA manifest
├── icon.svg / *.png          ← app icons
├── federated_server.py       ← Python backend (Railway / Render / Docker)
├── requirements.txt          ← Python deps (fully pinned)
├── requirements-dev.txt      ← test deps (pytest, pytest-cov)
├── pytest.ini                ← pytest config → tests/server/
├── Procfile                  ← start command for buildpack platforms (`gunicorn federated_server:app`)
├── gunicorn.conf.py          ← gunicorn settings + startup guards + log format (read automatically)
├── Dockerfile                ← Docker deployment (VPS / fly.io / GCP / Railway)
├── runtime.txt               ← Python 3.12
├── tests/server/             ← server test suite (pytest)
├── .github/workflows/        ← ci.yml (tests) + deploy-pages.yml (gated deploy)
└── README.md
```

Only the static site is published to GitHub Pages: the deploy workflow copies
the repo into `_site/` and deletes server code, tests, tooling, CI config, docs
and `*.md` (a denylist, so new static folders such as `vendor/` ship
automatically).

`index.html` works with no server (detection is fully on-device).
The server adds: global detection map, federated model, live device count.

---

## Logs

On Railway every line is one JSON object (`LOG_FORMAT=json`): `level` colours
the line and `message` is the text; every other key is a filter in the Log
Explorer. gunicorn's own lines go to stdout as JSON too (on stderr Railway
showed even `Starting gunicorn` as an error).

| What | When | Level | Filter |
|---|---|---|---|
| `[Boot] MosquitoNet v13 pid=… uid=… commit=… storage=persistent …` | each start | info | `@event:boot` |
| `[Storage] EPHEMERAL: <reason>` | start, if data would be lost | **error** | `@event:ephemeral_storage` |
| `ready: listening on 0.0.0.0:8080, workers=1, …` | gunicorn ready | info | `@component:gunicorn` |
| `repaired bind address …` / `workers=N refused …` | start command is wrong | **error** | `@component:gunicorn @level:error` |
| `[Boot] running as root …` | start command bypassed the entrypoint | warn | `@event:running_as_root` |
| `[Det #42] Aedes albopictus conf=0.830 …` | each stored detection | info | `@event:detection` (fields `det_id`, `species`, `conf`, `freq`, `located`) |
| `[Det REJECT 400] …`, `[Fed REJECT 400] …`, `[CORS] rejected …` | invalid or foreign requests | warn | `@level:warn` |
| `[Stats] active=… detections=… (+n) … write_errors=+n disk_free_mb=…` | every 15 min | info; **error** while writes fail | `@event:stats` |
| `[append] FAILED`, `[state] save FAILED` | disk full / volume gone | **error** | `@event:write_failed` |
| `[500] GET /… RuntimeError: …` with a `stack` field | unhandled exception | **error** | `@event:internal_error` |
| `worker … timed out …` with every thread's `stack` | a request hung past 120 s | **error** | `@component:gunicorn` |
| `GET /path 500 12ms` | 4xx/5xx and slow requests (`ACCESS_LOG`) | warn / error | `@component:access` |
| `[Exit] done` with final counters | graceful shutdown | info | `@event:exit` |

Never logged: raw device ids, client IPs, query strings or coordinates.
Client-supplied text appears only inside rejection reasons, escaped and
truncated, so it can't forge log lines. Everything
needed to act on is at `warn` or `error`, so `@level:error` should be empty on
a healthy service.

## The app shows "Offline"

In the app: **About → Advanced → Test server connection** (Advanced asks for
the maintainers' password; the page stores only its SHA-256). It probes
`/health` on every endpoint in `SERVER_URLS` and names the failure:

| Result | Meaning | Fix |
|---|---|---|
| `OK (MosquitoNet v13)` | That endpoint works; the app switches to it on its next heartbeat (≤ 10 s). | If the pill still says Offline, reload the page (an old cached build may be running). |
| `unreachable — DNS, TLS or network failure` | The domain doesn't resolve, has no valid certificate, or nothing is deployed. | Railway → Networking: is the custom domain verified and the certificate issued? Does the DNS record match what Railway shows? Did the last deploy succeed? |
| `HTTP 404 — answered, but not the MosquitoNet server` | Something answers, but not this app: typically Railway's "Application not found" (domain not attached to this service, or a wrong target port) or a parked domain. | Attach the domain to the service (Step 3) and check the port matches the `Listening at:` log line. |
| `HTTP 502` / `503` | The platform reached the service, but it isn't serving: still booting, crashed, or failing its health check. | Railway → Deployments → deploy and runtime logs. |
| `no answer within 8 s` | A firewall or a hung server. | As for 502. |

From a computer, `curl -sS https://mosquito.charity/health` gives the same
information. Railway build failures (`BUILD_IMAGE`, e.g. the Dockerfile
`VOLUME` rejection) leave the **previous** deployment running — or nothing, on a
new service — so always check the latest deploy actually went live.

## Diagnosing "Server Down" (403 / 502 / timeout)

### Check 1: Is gunicorn running?
```bash
ps aux | grep gunicorn
# If nothing → the process crashed. Restart it (see below).
```

### Check 2: Is the port bound?
```bash
ss -tlnp | grep 8080
# Or: netstat -tlnp | grep 8080
```

### Check 3: Does Flask itself respond?
```bash
curl http://localhost:8080/health
# Should return {"status":"ok","service":"MosquitoNet v13",...}
# If this works but the public URL fails → nginx/Caddy proxy is broken
# "status":"degraded" → the disk is full or the volume is gone (see disk_free_mb)
```

### Check 3b: Requests rejected with 403 / 429 / 503?
- `403 {"reason":"origin not allowed"}` — the app is served from an origin the
  server doesn't know. Add it to `ALLOWED_ORIGINS`.
- `429` / `503` with `"rate limited"` — see *Rate limits* above. If *everyone*
  is throttled, the server is probably seeing the proxy's IP for all clients:
  check `TRUSTED_PROXY_HOPS` (e.g. `2` behind Cloudflare → Railway).
- `503 {"reason":"storage unavailable"}` — writes to `DATA_DIR` are failing.

### Check 4: nginx / Caddy proxy
```bash
# For nginx — check the proxy_pass line:
grep -r proxy_pass /etc/nginx/
# Should say: proxy_pass http://127.0.0.1:8080;

# Restart nginx:
sudo systemctl restart nginx

# For Caddy:
sudo systemctl restart caddy
```

### Restart the server (any environment)

**Docker (recommended — isolated, auto-restarts):**
```bash
docker build -t mosquitonet .
docker run -d --name mosquitonet --restart=always \
  -p 8080:8080 -v mosquitonet_data:/data \
  mosquitonet
```

**Without Docker (Railway/Render will do this automatically):**
```bash
pip install -r requirements.txt
PORT=8080 gunicorn federated_server:app     # run from the repo root
```
Everything else comes from `gunicorn.conf.py` (read from the current
directory): one worker — a `--workers N` flag, `WEB_CONCURRENCY` or a `TTIN`
signal is refused with an error in the log — threads, timeouts, logging. Don't
add `--preload`: the background flusher thread must start in the worker, not
the master.

**Railway:** your service → **Deployments** → **⋮ → Redeploy** restarts the
current build. After changing *settings* (start command, healthcheck), use
**Deploy latest commit** or push to `main` instead — see *Railway service
settings*.

**Render:** Go to your Render service → click **Manual Deploy** → **Deploy latest commit**.

---

---

## Stable deploy URL (permanent address, every deploy)

The app's public address is **https://uriiger-sketch.github.io/mosquito/** and it
**never changes** between deployments — a GitHub Pages project URL is fixed by
`<user>.github.io/<repo>/`. It only changes if the repo or account is renamed.

A GitHub Actions workflow (`.github/workflows/deploy-pages.yml`) makes each deploy
deterministic: every push to `main` runs the full test suite and, only if it
passes, republishes the static site to that same URL.

**Every deployment shows its own version.** The workflow stamps `index.html`
(`tools/stamp-version.js`): `APP_VERSION` keeps its hand-set major.minor and
takes the deploy's run number as the last part (`3.1.0` in the repo → `3.1.10`
for deploy run #10), and `APP_BUILD` becomes the commit and date. About shows
`MosquitoNet v3.1.10`; About → Advanced shows `3.1.10 · de25d72 · 2026-10-27`.
Bump only major.minor by hand, for releases worth naming.

**One-time setup to activate it:**
1. Repo **Settings → Pages**
2. **Build and deployment → Source → "GitHub Actions"**
3. Push to `main` (or run the workflow from the Actions tab). Done — future pushes
   auto-deploy to the same address.

Until you switch the Source to "GitHub Actions", the workflow run will fail
harmlessly; any existing branch-based Pages deploy keeps serving the site — but a
branch deploy publishes the whole repo (server code, tests, docs) and skips the
test gate, so do switch.

---

## Continuous integration and the deploy gate

`.github/workflows/ci.yml` runs on every push and pull request:

| Job | What it checks |
|---|---|
| **JS unit tests + detector benchmark** | `npm test`, then `npm run bench:quick` (Node 22) |
| **Server tests (Python 3.12)** | `pytest` over `tests/server/` with coverage (fails under 85 %) |
| **End-to-end (Playwright, Chromium)** | `npm run test:e2e` — the real app in a headless browser: detection with WAV files as the mic, and connectivity against a locally started backend (endpoint failover, heartbeat, upload, diagnostics, outage recovery). Production is never called; no secrets needed |
| **Backend image smoke test** | builds the Dockerfile, starts it on a root-owned volume holding legacy data, checks it runs as uid 10001, accepts a detection, turns Docker-healthy and flushes state on `docker stop` |

`deploy-pages.yml` calls the same workflow and publishes **only if the JS, server
and end-to-end jobs pass** (the image smoke test is skipped there — it doesn't
affect the static site). Waiting for the browser tests matters: the service
worker caches what we ship on users' phones, so a broken release would outlive
its own rollback. In an emergency, run **Actions → Deploy to GitHub Pages → Run
workflow** with **skip_e2e** ticked; unit and server tests still gate it.

Run the same checks locally:
```bash
npm test && npm run bench:quick
pip install -r requirements.txt -r requirements-dev.txt
pytest --cov=federated_server --cov-report=term-missing
```

**Recommended repo settings (manual):**
- **Settings → Branches → Add rule for `main`**: require a pull request and the
  status checks *JS unit tests + detector benchmark*, *Server tests (Python
  3.12)*, *End-to-end (Playwright, Chromium)* and *Backend image smoke test*.
- **Settings → Environments → github-pages**: limit deployments to `main`.
- The backend is **not** deployed by GitHub Actions — Railway/Render deploy it
  straight from the repo on every push. In the service settings, make the
  auto-deploy wait for GitHub CI checks to pass (Railway: **"Wait for CI"**;
  Render: Auto-Deploy **after CI checks pass**) so a red build never reaches the
  server.

**Optional — a custom domain (e.g. app.mosquito.charity):** add a `CNAME` file
containing the domain and point a DNS `CNAME` record at `uriiger-sketch.github.io`.
Do this only once DNS is configured — an unresolvable domain in `CNAME` takes the
site offline.

---

## Part 1: GitHub Pages (the app)

### Push all files to your repo

```bash
# If you haven't cloned it yet:
git clone https://github.com/YOUR-USERNAME/mosquito.git
cd mosquito

# Copy all files in:
cp /path/to/index.html .
cp /path/to/federated_server.py .
cp /path/to/requirements.txt .
cp /path/to/Procfile .
cp /path/to/runtime.txt .
cp /path/to/README.md .

git add .
git commit -m "Add all MosquitoNet files"
git push origin main
```

### Enable GitHub Pages

1. Go to your repo on GitHub
2. Click **Settings** → **Pages** (left sidebar)
3. Under **Build and deployment → Source**, choose **GitHub Actions**
   (not "Deploy from a branch" — see *Stable deploy URL* above)
4. Push to `main`; the *Deploy to GitHub Pages* workflow tests and publishes it

Your app will be live at:
```
https://YOUR-USERNAME.github.io/mosquito/
```
(Takes ~2 minutes to first deploy, then ~30 seconds for updates.)

### Install on iPhone

1. Open the URL above in **Safari** (must be Safari)
2. Tap the **Share button** (□↑)
3. Tap **"Add to Home Screen"**
4. Tap **"Add"**

---

## Part 2: Federated Server (Railway — free tier)

Railway gives you a persistent server with free compute.

### Step 1: Create a Railway account

Go to [railway.app](https://railway.app) → sign up with GitHub (takes 30 seconds).

### Step 2: Deploy from GitHub

1. Click **"New Project"**
2. Click **"Deploy from GitHub repo"**
3. Select your `mosquito` repo
4. Railway builds the root `Dockerfile`.
5. Before (or right after) the first deploy, apply **every** setting in
   *Railway service settings* below — above all the start command, the volume
   and `PORT`. Then **Deploy latest commit**.

Wait ~2 minutes. A good boot logs `ready: listening on 0.0.0.0:8080` and a
`[Boot] … storage=persistent` line — see *Logs*.

### Railway service settings (production, as configured)

| Setting | Value | Why |
|---|---|---|
| **Start command** | `/usr/local/bin/docker-entrypoint gunicorn federated_server:app` (or empty) | Railway runs it in **exec form — no shell** — and it **replaces** the image `ENTRYPOINT`. Never put `$PORT` (or any `$VAR`) in it: it reaches gunicorn literally. Keeping `docker-entrypoint` keeps the drop to uid 10001. |
| Healthcheck path / timeout | `/health` / `300` s | A deploy goes live only once the new process answers. |
| Restart policy | **Always** | A web server should never stay down; `On failure` gives up after N crashes. |
| Draining (teardown) | `40` s | Railway's default is **0 s** — SIGKILL right after SIGTERM — which skips the final fsync/state save (gunicorn's graceful timeout is 30 s). |
| Volume | mount path `/data` | Without it every deploy wipes all data (`/health` → `ephemeral_reason`). |
| Variables | `PORT = 8080`, `DATA_DIR = /data` | `PORT` pins the port the domains target (`8080`); Railway's healthcheck uses the same variable. |
| Replicas | 1 | All state lives in one process; a volume pins the service to one region anyway. |
| Source → **Wait for CI** | on (dashboard toggle) | A red commit never reaches the server. |

Two Railway behaviours behind the October 2026 outage:

- **A start command is exec form.** `gunicorn … --bind 0.0.0.0:$PORT`, which
  Railway had imported from an old `Procfile`, made gunicorn exit with
  `Error: '$PORT' is not a valid port number.` on every boot. The `Procfile` no
  longer has flags, and `gunicorn.conf.py` now repairs such a bind (and refuses
  extra workers) at startup with an `error` log line instead of crashing — but
  fix the setting when you see that line.
- **Redeploy replays the old deployment, start command included.** A changed
  setting only takes effect on a *new* deployment: push to `main`, or
  **Deploy latest commit**.

Configuration as code (`railway.json`) is deprecated by Railway and stops being
read on 2026-12-01, so these settings live on the service; this table is their
record.

### Step 3: Give the service a public address

Railway → your service → **Settings → Networking**:

1. **Generate Domain** — you get e.g. `https://mosquito-production.up.railway.app`.
   If it asks for a port, use the one from the `Listening at:` log line (8080
   unless Railway injected its own `PORT`). A wrong port shows up as a Railway
   error page instead of `/health`.
2. **Custom Domain** → add `mosquito.charity` (the address the app calls by
   default). Railway shows the DNS record to create (a `CNAME`, or `ALIAS`/
   flattened `CNAME` at the apex, plus a `TXT` verification record) — add it at
   your domain registrar/DNS host and wait until Railway shows the domain as
   verified and the certificate as issued. If the DNS is proxied through
   Cloudflare, set `TRUSTED_PROXY_HOPS = 2`.

Check both from any machine: `curl https://mosquito.charity/health` and
`curl https://web-production-de7b5.up.railway.app/health` (the production
service's Railway domain) must return `"service":"MosquitoNet v13"`. The app
lists both in `SERVER_URLS`, so it stays online if the custom domain's DNS or
certificate ever lapses.

### Step 4: Variables (DATA_DIR, and ALLOWED_ORIGINS only if needed)

In Railway → your service → **Variables** tab, set `DATA_DIR = /data` and attach
a volume at `/data` (see *READ FIRST* at the top).

The server only accepts uploads (POST) from known app origins. The GitHub Pages
app, `mosquito.charity` (and subdomains) and `localhost` are built in, so the
official app needs **nothing** here. Only if you serve the app from another
origin (a fork, a Netlify preview, a different custom domain) add it:

```
ALLOWED_ORIGINS = https://YOUR-USERNAME.github.io,https://my-preview.netlify.app
```

Origins are `scheme://host[:port]` — no path (a `/mosquito/` suffix is ignored).
Uploads from any other web page are rejected with `403`. Reads (`GET`) stay open
to everyone.

If you put Cloudflare in front of Railway, also set `TRUSTED_PROXY_HOPS = 2`
(see *Server configuration*).

### Step 5: Point the app at the server (no code change)

The app calls the endpoints in `SERVER_URLS` (top of the `<script>` in
`index.html`; default: `https://mosquito.charity`, then the production
service's Railway domain) **in order**, fails over
to the next one when an endpoint doesn't answer as MosquitoNet, and remembers
the one that worked on each device.

To change the list without editing code, set a repository variable:

1. GitHub repo → **Settings → Secrets and variables → Actions → Variables →
   New repository variable**
2. Name `SERVER_URLS`, value — comma-separated origins (`https://host`, no path):
   ```
   https://mosquito.charity,https://web-production-de7b5.up.railway.app
   ```
3. **Actions → Deploy to GitHub Pages → Run workflow** (or push to `main`).

List the custom domain and the Railway domain **of the same service** (one
data store): the app then stays online while the custom domain's DNS or
certificate is being set up, and moves back to it by itself once it works.
Never list two different backends — they would split the map and the model. The deploy refuses anything that
isn't a plain origin, so the variable can only ever put a URL into the page.

---

## Verify everything works

| Check | How to verify |
|---|---|
| App loads | Open `https://YOUR-USERNAME.github.io/mosquito/` |
| Microphone works | Tap START — should not error |
| Server alive | Open `https://YOUR-SERVER.up.railway.app/health` — `"status":"ok"`, `"service":"MosquitoNet v13"`, `"writable":true`, `"ephemeral":false`, `"single_instance":true` |
| Federation active | Tap Stats tab → Federated Learning → should show "uploaded Xs ago" |

---

## Files NOT pushed to GitHub

These are for local development only — do not push:

```
serve.py       ← local dev server, not needed online
core.js        ← intermediate build artifact
*.pem          ← SSL certificates (never commit these)
```

Add them to `.gitignore`:

```bash
echo "serve.py" >> .gitignore
echo "core.js" >> .gitignore
echo "*.pem" >> .gitignore
git add .gitignore
git commit -m "Add .gitignore"
git push origin main
```

---

## Cost

| Service | Cost |
|---|---|
| GitHub Pages | Free (unlimited for public repos) |
| Railway | Free tier: 5$/month credit, ~500 hours/month |
| Total | **Free** for this scale |

Railway's free tier is enough for hundreds of concurrent devices.
If you outgrow it, Render.com has the same free tier and identical setup.

---

## Render.com (alternative to Railway)

If Railway doesn't work, Render is identical:

1. Go to [render.com](https://render.com) → sign up with GitHub
2. Click **"New"** → **"Web Service"**
3. Connect your `mosquito` repo
4. Runtime: **Python 3** (or **Docker**, which uses the `Dockerfile`)
5. Build command: `pip install -r requirements.txt`
6. Start command — the `Procfile` line: `gunicorn federated_server:app`.
   `gunicorn.conf.py` (read from the repo root) holds it at one worker even if
   Render sets `WEB_CONCURRENCY`.
7. Environment: `DATA_DIR = /data` plus a disk mounted at `/data`
8. Click **"Create Web Service"**

Same `ALLOWED_ORIGINS` / `TRUSTED_PROXY_HOPS` notes as Railway apply.
