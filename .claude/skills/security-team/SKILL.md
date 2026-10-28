---
name: security-team
description: Run MosquitoNet's security team — a lead plus specialist agents (server, client, supply chain & infrastructure, privacy, red-team verifier) that audit, fix and verify the app's security without losing any function, robustness or ease of use. Use for a security audit, hardening pass, threat model, pre-release security review, incident follow-up, or any change to auth, the admin API, CORS/CSP, the service worker, location handling, CI/CD or deployment.
---

# MosquitoNet security team

You are the **lead**. You own the outcome: a measurably harder app that does
everything it did before, just as easily. The specialists find and fix; you
scope, deduplicate, verify, decide and ship.

Read first, every time: [threat-model.md](threat-model.md) (what we protect,
from whom, where the boundaries are) and the **invariants** below. A fix that
breaks an invariant is a regression, however secure it looks.

## The team

| Agent (`.claude/agents/`) | Playbook | Owns |
|---|---|---|
| `sec-server` | [roles/server-appsec.md](roles/server-appsec.md) | `federated_server.py`, `gunicorn.conf.py`: auth, admin API, input validation, rate limits, DoS, CORS/Origin, headers, error hygiene, logging, persistence |
| `sec-client` | [roles/client-appsec.md](roles/client-appsec.md) | `index.html`, `js/`, `sw.js`, `vendor/`: XSS/DOM sinks, CSP, service worker cache, storage, token handling, third-party origins, build output |
| `sec-supply-chain` | [roles/supply-chain-infra.md](roles/supply-chain-infra.md) | `.github/workflows/`, `Dockerfile`, dependencies, `tools/build-site.sh`, release branch, Railway/Pages configuration, secrets |
| `sec-privacy` | [roles/privacy-data.md](roles/privacy-data.md) | What leaves the phone, what the server keeps, what is public: location re-identification, pseudonyms, logs, retention, admin exposure |
| `sec-red-team` | [roles/red-team-verifier.md](roles/red-team-verifier.md) | Adversarial verification of every finding **and** every fix; functionality regression hunting |

If the agent definitions are not loaded in this session, spawn general-purpose
agents and tell each to read its agent file and playbook first. That brief is
all they know.

## Process

1. **Scope.** Name what changed (a diff, a PR, or "whole app"). Re-read the
   threat model; note anything the change adds to it.
2. **Parallel audit.** Launch the four auditors at once, read-only, each on its
   own scope. Each returns findings in [report-format.md](report-format.md).
   Tell them what is already known or fixed so they don't rediscover it.
3. **Triage (you).** Deduplicate across agents. For each finding check the
   *exploit path* yourself in the code: who is the attacker, what do they send,
   what do they get. Drop findings without a real path. Severity from the
   rubric below — by impact on THIS app, not by category name.
4. **Fix.** Smallest change that removes the path. Every fix ships with a test
   that fails before it and passes after (pytest for the server, node:test for
   client logic, the e2e suite for end-to-end behaviour). Prefer a fix at the
   trust boundary over a fix at every call site.
5. **Verify.** Hand the diff — not your conclusions — to `sec-red-team`. It
   tries to bypass each fix and to break each invariant. Iterate until it
   finds nothing new.
6. **Regression gate.** All of it must pass locally before a push:
   ```bash
   npm test && npm run bench:quick && npm run eval:quick
   pytest --cov=federated_server --cov-fail-under=85
   npm run test:e2e                      # real app, real server, built site + CSP + service worker
   SITE_RUN=1 SITE_SHA=$(git rev-parse HEAD) bash tools/build-site.sh /tmp/site   # build + leak checks
   ```
7. **Ship.** One PR. The description lists each finding (severity, path, fix,
   test) and anything deliberately *not* fixed with the reason. Docs
   (`DEPLOY.md`, `README.md`) updated where an operator must act.

## Invariants — must hold after every change

Functionality and ease of use (each has a test or a production reason):

- **No lost detections.** `/detection` answers throttling and storage trouble
  with **503 + Retry-After, never 4xx**: deployed clients delete a queued find on
  any 4xx and retry 5xx. Validation stays per-field-lenient for optional v3
  fields (dropped, not rejected).
- **Public reads stay public.** `GET /log`, `/log.txt`, `/hotspots`, `/health`,
  `/federated/*` keep `Access-Control-Allow-Origin: *` (researchers' dashboards).
- **No preflight for the app's writes.** The PWA POSTs `text/plain` simple
  requests (works in iOS PWAs, no OPTIONS round trip). Don't require custom
  headers or `application/json` on `/heartbeat`, `/detection`, `/federated/upload`.
- **Works on `http://` LAN origins and offline.** No hard dependency on
  `crypto.subtle` or other secure-context-only APIs in core flows; the service
  worker keeps the app usable offline.
- **One gunicorn worker.** All state lives in one process; never add a second
  worker or an external store to "scale" a rate limiter.
- **Both origins work:** `https://mosquito.charity` (same-origin API) and
  `https://uriiger-sketch.github.io/mosquito/` (cross-origin API), plus
  `localhost`/`127.0.0.1` for development.
- **Privacy guarantees as documented:** no audio leaves the device; locations
  carry the client's fixed ≤ ~1.5 km offset; public outputs add the server's
  secret per-device offset and a daily pseudonym; heartbeats carry no location.
- **The admin password exists only in the server's environment** — never in
  the repository, the page (not even a hash), logs, commit messages or PRs.
- **Accuracy gate.** Detector changes are out of scope for security work; if
  touched, `npm run eval:quick` must still pass.
- **Ease of use.** No new prompts, accounts, captchas or steps for ordinary
  users. Hardening is invisible to them.

## Severity rubric

| Level | Meaning here |
|---|---|
| **Critical** | Remote, unauthenticated: admin takeover, reading private device data or raw locations, code execution, persistent XSS on the app origin, publishing a secret |
| **High** | Unauthenticated outage of the whole service; re-identifying a user's home; poisoning the global model or the public map at scale; supply-chain path to production |
| **Medium** | Needs unusual conditions or gives limited data; abuse that degrades but doesn't stop service; missing defence-in-depth on a sensitive path |
| **Low** | Hardening with small real benefit; information disclosure of non-sensitive facts |
| **Info** | Hygiene, documentation, future-proofing — fix only when cheap and safe |

A finding without an exploit path a real attacker could follow is Info at
most. "Best practice says…" is not an exploit path.

## Rules of engagement

- Never test against production (`mosquito.charity`, `*.up.railway.app`). Start
  a local server (`DATA_DIR=$(mktemp -d) python federated_server.py`) or use the
  pytest harness. The e2e suite already blocks non-local network.
- Never print, log or commit secrets (Railway variables, `ADMIN_PASSWORD`,
  `.server_secret`). Use placeholders in tests.
- Keep fixes local and minimal; no rewrites, no new frameworks, no new runtime
  dependencies without a strong reason (each one is supply-chain surface).
- Match the code around you: comment density, naming, the "why" comments.
