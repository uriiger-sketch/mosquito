# MosquitoNet — Complete Deployment Guide

## What goes where

```
mosquito/
├── index.html                ← PWA app (GitHub Pages / gh-pages branch)
├── federated_server.py       ← Python backend (Railway / Render / Docker)
├── requirements.txt          ← Python deps
├── Procfile                  ← Railway/Heroku start command
├── Dockerfile                ← Docker deployment (VPS / fly.io / GCP)
├── runtime.txt               ← Python 3.12
└── README.md
```

`index.html` is fully self-contained — works with no server.
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
# Should return {"status":"ok","service":"MosquitoNet v10",...}
# If this works but the public URL fails → nginx/Caddy proxy is broken
```

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
  --workers 1 --threads 4 --timeout 120
```

**Railway:** Go to your Railway project → your service → **Deployments** → click **Redeploy**.

**Render:** Go to your Render service → click **Manual Deploy** → **Deploy latest commit**.

---

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
3. Under **Build and deployment → Source**, choose **Deploy from a branch**
4. Branch: **main**, Folder: **/ (root)**
5. Click **Save**

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
4. Railway auto-detects Python from `requirements.txt` and `Procfile`
5. Click **"Deploy"**

Wait ~2 minutes. Railway will show a deployment log.

### Step 3: Get your server URL

1. In Railway, click your deployment
2. Click **"Settings"** → **"Domains"**
3. Click **"Generate Domain"**
4. You get a URL like: `https://mosquito-production.up.railway.app`

### Step 4: Set the ALLOWED_ORIGIN env variable

In Railway → your service → **Variables** tab:

```
ALLOWED_ORIGIN = https://YOUR-USERNAME.github.io
```

This restricts the server to only accept requests from your GitHub Pages app.

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
| Server alive | Open `https://YOUR-SERVER.up.railway.app/health` — should return `{"status":"ok"}` |
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
4. Runtime: **Python 3**
5. Build command: `pip install -r requirements.txt`
6. Start command: `gunicorn federated_server:app --bind 0.0.0.0:$PORT`
7. Click **"Create Web Service"**

Same `ALLOWED_ORIGIN` env variable setup applies.
