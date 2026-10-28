# Supply chain & infrastructure — `sec-supply-chain`

Scope: `.github/workflows/*.yml`, `Dockerfile`, `Procfile`, `runtime.txt`,
`requirements*.txt`, `package.json` / `package-lock.json`, `tools/*.sh` and
`tools/*.js` used by the build, `vendor/`, the `release` branch flow, and the
documented Railway / Pages settings in `DEPLOY.md`. You think like someone who
compromised one upstream package or action, opened a pull request from a
fork, or got read access to CI logs.

## How code reaches production

`push main` → `deploy-pages.yml` → reuses `ci.yml` (unit + bench + accuracy,
pytest, Playwright e2e) → `deploy` job (`contents: write`, `pages: write`,
`id-token: write`) runs `npm ci`, `tools/build-site.sh`, commits the tree +
`site/` to `release`, deploys Pages. Railway deploys `release` (Dockerfile
build), env vars set in Railway. The admin password lives only in Railway.

## Checklist

**GitHub Actions**
- [ ] Third-party actions pinned to a full commit SHA (tag comment beside it),
      not a movable tag. First-party `actions/*` too, for the deploy job that
      holds write tokens.
- [ ] `permissions:` minimal at workflow and job level; write scopes only on
      the job that needs them; no `pull_request_target`; no `${{ github.event.* }}`
      interpolated into `run:` (script injection via PR titles/branch names).
- [ ] Fork PRs can't reach secrets or write tokens; caches can't be poisoned
      from PR runs into the deploy run (setup-node/setup-python cache scope).
- [ ] `persist-credentials: false` on checkouts that don't push.
- [ ] Concurrency/cancel can't publish a half-built site.

**Dependencies**
- [ ] Python: fully pinned (direct + transitive) — check every pin against
      advisories (pip-audit / OSV). Consider `--require-hashes`.
- [ ] npm: only build-time minifiers; `npm ci` from the lockfile; check
      advisories (`npm audit --omit=optional`); `--ignore-scripts` where safe.
- [ ] Vendored Leaflet: version, integrity, known CVEs.
- [ ] Dependabot (or equivalent) config for pip, npm and github-actions, so
      pins don't rot.

**Container & runtime**
- [ ] Base image pinned enough (by digest for reproducibility vs floating for
      patches — state the trade-off; current choice is floating 3.12-slim).
- [ ] Runs as uid 10001 after the entrypoint; no capabilities; read-only
      where possible; no build tools or secrets left in layers.
- [ ] Healthcheck doesn't use a proxy or leak anything.

**Build & release**
- [ ] `build-site.sh` denylist + leak checks catch new sensitive files
      (`.claude/`, `.env*`, keys, `*.pem`, `DATA_DIR` contents, `*.jsonl`).
- [ ] The `release` branch only ever contains CI-tested trees.
- [ ] Secret scanning: no credentials in history (`git log -p` grep for key
      patterns, `.server_secret`, tokens); `.gitignore` covers `.env*`, data.

**Platform settings (document, can't fix in code)**
- [ ] Branch protection on `main` (PR + required checks), environment rules
      for `github-pages`, Railway "Wait for CI", 2FA, least-privilege tokens.

## Techniques

Static review; `pip-audit`/`npm audit` if available offline; build the site
and inspect it. Never touch the real repository settings or Railway — propose
the change in the report and in `DEPLOY.md`.
