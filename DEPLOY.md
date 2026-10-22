# MosquitoNet — Complete Deployment Guide

## ⚠️ READ FIRST: set DATA_DIR to a persistent volume

`DATA_DIR` defaults to `/tmp`. On Railway, Render, Heroku and Fly, `/tmp` is
**ephemeral** — it is wiped on every restart, redeploy and daily container
recycle. If you go live without setting it you will **permanently lose every
detection, all statistics and the trained federated model**, roughly daily, with
no warning and no way to recover them.

The server prints a loud banner at startup when it detects this, and
`GET /health` reports `"ephemeral": true`.

**Railway**
1. Service → **Variables** → add `DATA_DIR = /data`
2. Service → **Settings → Volumes** → **Add Volume**, mount path `/data`
3. Redeploy. Confirm `curl https://YOUR-URL/health` shows `"ephemeral": false`
   and `"writable": true`.

**Render**
1. Service → **Disks** → **Add Disk**, mount path `/data` (1 GB is plenty)
2. Service → **Environment** → add `DATA_DIR = /data`
3. Redeploy and check `/health` as above.

**Docker / VPS / Fly** — already correct: the `Dockerfile` sets `DATA_DIR=/data`
and declares `VOLUME /data`. Just make sure you actually mount it:
`docker run -v mosquitonet_data:/data ...`

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
| `PUBLIC_COORD_DECIMALS` | `2` | Decimals of lat/lng shown by the public `/log` and `/log.txt` (2 ≈ 1.1 km). The private JSONL keeps 4. |
| `PSEUDONYM_SECRET` | random per start | Key for the per-day device pseudonyms in `/log`. Set it only if pseudonyms must stay stable across restarts. |
| `FED_CLIP_NORM` | `2.0` | Max L2 distance one federated upload can pull the model per round. |

### Rate limits

Token buckets (burst, then sustained rate), multiplied by `RATE_LIMIT_SCALE`.
Per-IP limits are generous because whole campuses and mobile carriers share one
address; per-device limits sit well above what the app actually sends.

| Bucket | Burst | Sustained | Applies to |
|---|---|---|---|
| per IP, everything | 600 | 20 / s | all requests except `/health` and `OPTIONS` |
| per IP, writes | 120 | 1 / s | `POST /detection`, `POST /federated/upload` |
| per IP, logs | 30 | 1 / 2 s | `/log`, `/log.txt` |
| per device, detections | 60 | 15 / min | `POST /detection` (burst = draining an offline queue) |
| per device, heartbeats | 60 | 60 / min | `/heartbeat` (one open app sends ~9 / min) |
| per device, uploads | 6 | 2 / min | `POST /federated/upload` |

Throttled requests get `429` + `Retry-After` — except `/detection`, which gets
`503` + `Retry-After`, because deployed apps delete a queued find on any 4xx but
retry 5xx with backoff. A throttled find is delayed, never lost.

### Health check fields

`GET /health` (never rate limited) returns the public stats plus:

| Field | Meaning |
|---|---|
| `status` | `ok`, or `degraded` when the last write to the detection log failed (disk full / volume gone). HTTP status stays 200. |
| `service` | `MosquitoNet v13` — use it to confirm a deploy went live. |
| `writable`, `ephemeral` | `DATA_DIR` is writable / is under `/tmp` (must be `true` / `false` in production). |
| `single_instance` | `false` = another process holds `DATA_DIR` — fix immediately. |
| `disk_free_mb` | Free space on the `DATA_DIR` volume. |
| `last_state_save` | Time of the last successful state-file save (every ~5 s while there is activity). |
| `client_ip` | The address rate limits use for *your* request. **Check it once after deploying:** it must be your own public IP. A Cloudflare/Railway/proxy address means `TRUSTED_PROXY_HOPS` is wrong and all users share one bucket. |
| `counters` | Since process start: `rejected_detections`, `rejected_uploads`, `rejected_origin`, `rate_limited`, `clipped_updates`, `dropped_fields`, `write_errors`. A climbing `rejected_origin` / `rate_limited` / `clipped_updates` means someone is probing or poisoning. |
| `python`, `log_cap`, `seen_events`, `min_conf`, `data_dir`, `log_file` | Runtime facts (Python is reported as major.minor only). |

### Privacy of the public log

`/log` and `/log.txt` are public. They show coordinates rounded to ~1 km and a
device pseudonym that changes every day (UTC), so one phone's reports can't be
linked across days or averaged into its home location. The stable device hash,
eventIds and 4-decimal (client-jittered) coordinates stay in the private JSONL.
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
├── Procfile                  ← Railway/Heroku start command
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
gunicorn federated_server:app \
  --bind 0.0.0.0:${PORT:-8080} \
  --workers 1 --threads 4 --timeout 120 --graceful-timeout 30 --no-control-socket
```
Always `--workers 1` (see above). Don't add `--preload` either: the background
flusher thread must start in the worker, not the master.

**Railway:** Go to your Railway project → your service → **Deployments** → click **Redeploy**.

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
| **End-to-end (Playwright, Chromium)** | `npm run test:e2e` — the real app in a headless browser (production API calls are blocked; no secrets needed) |
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
4. Railway finds the `Dockerfile` at the repo root and builds the image from it
   (a root `Dockerfile` takes precedence over `requirements.txt` + `Procfile`)
5. Click **"Deploy"**

Wait ~2 minutes. Railway will show a deployment log.

### Step 3: Get your server URL

1. In Railway, click your deployment
2. Click **"Settings"** → **"Domains"**
3. Click **"Generate Domain"**
4. You get a URL like: `https://mosquito-production.up.railway.app`

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

### Step 5: Update index.html with the server URL

Open `index.html` and find this line near the top of the `<script>`:

```javascript
const SERVER_URL = '';   // ← PASTE YOUR SERVER URL HERE AFTER DEPLOYING
```

Change it to:

```javascript
const SERVER_URL = 'https://mosquito-production.up.railway.app';
```

Then push:

```bash
git add index.html
git commit -m "Connect app to federated server"
git push origin main
```

GitHub Pages redeploys in ~30 seconds. Done.

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
6. Start command — exactly the `Procfile` line, `--workers 1` included (Render
   may set `WEB_CONCURRENCY`, which would otherwise start several workers):
   `gunicorn federated_server:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 --graceful-timeout 30 --no-control-socket`
7. Environment: `DATA_DIR = /data` plus a disk mounted at `/data`
8. Click **"Create Web Service"**

Same `ALLOWED_ORIGINS` / `TRUSTED_PROXY_HOPS` notes as Railway apply.
