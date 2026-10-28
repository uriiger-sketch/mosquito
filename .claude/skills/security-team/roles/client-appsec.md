# Client application security — `sec-client`

Scope: `index.html` (markup, CSS, the inline app script), `js/*.js`, `sw.js`,
`vendor/`, `manifest.webmanifest`, and the built output of
`tools/build-site.sh` + `tools/minify-site.js`. You think like a malicious web
page, a poisoned server response, a crafted detection row that other users'
browsers will render, and a network attacker on café Wi-Fi.

## How the client is built

- One big inline `<script>` in `index.html`; on mosquito.charity the backend
  serves it with a CSP that allows that script **only by its SHA-256** (see
  `_site_csp` in the server). Injected markup can't run inline code there —
  but GitHub Pages serves the same page with **no CSP header**.
- Rendering uses template strings + `innerHTML` with `esc()`; text writes use
  `setText`. Leaflet popups take HTML strings.
- Data the page renders that an attacker can influence: `/hotspots` cells,
  `/log` rows (via server), admin views (stored detection fields, log lines),
  `map-data.json`, history in `localStorage`, `location.hash`.
- Admin token: in memory only, sent as `Authorization: Bearer` to the URL it
  was issued by. The service worker skips `/admin/`.

## Checklist

**DOM XSS**
- [ ] Enumerate every `innerHTML` / `insertAdjacentHTML` / `outerHTML` /
      Leaflet `bindPopup` / `bindTooltip` / `L.divIcon({html})` sink. For each,
      trace every interpolated value to its source. Unescaped server or
      storage data = finding (severity by who controls it and whose browser
      renders it). Attribute contexts need quoting + `esc()`; URLs need scheme
      checks (`javascript:`); numbers interpolated raw must be numbers.
- [ ] `href`/`src` built from data; `target=_blank` without `rel=noopener`.
- [ ] No `eval`, `new Function`, `setTimeout(string)`, inline event handlers
      (`onclick=` in generated HTML would also be blocked by the CSP).

**CSP & headers**
- [ ] The served CSP: `script-src 'self' 'sha256-…'` only, `object-src 'none'`,
      `base-uri 'none'`, `frame-ancestors 'none'`, `form-action`. Is
      `connect-src 'self' https:` wider than needed (exfiltration channel for
      injected code)? Can it be the SERVER_URLS list without breaking the
      Pages copy or failover?
- [ ] GitHub Pages copy: no headers possible. A `<meta http-equiv="Content-
      Security-Policy">` gives script/connect/object/base protection there
      (not frame-ancestors). It must not conflict with the backend's header
      (both apply — the intersection must still allow the app).
- [ ] `Permissions-Policy` (microphone=self, geolocation=self, camera=()) —
      backend site responses.

**Service worker**
- [ ] Never caches API or admin responses; never serves cached responses for
      authenticated requests; cache keys can't be poisoned by query strings
      for app code; scope limited; update path can't get stuck on a
      compromised build longer than necessary.

**Storage & secrets**
- [ ] Nothing secret in `localStorage`/`sessionStorage`/IndexedDB/Cache
      Storage. The admin token never written anywhere, never in a URL, never
      sent to a different origin than it came from.
- [ ] Exports (CSV/JSON history): CSV injection (`=`, `+`, `-`, `@` leading
      cells opened in Excel).

**Third parties & supply**
- [ ] Every external origin the page contacts (OSM tiles, API endpoints).
      No CDN scripts. Vendored Leaflet version and known advisories.
- [ ] Build output: no source maps, no comments with internals, no server
      files, no secrets (`build-site.sh` leak checks) — and the unit test that
      forbids shipping password verifiers.

**Clickjacking / UI redress**
- [ ] `frame-ancestors 'none'` on the backend; Pages copy can be framed —
      does anything sensitive (Advanced/admin) need a frame-buster there?

## Techniques

`node -e` against extracted functions, the Playwright harness in
`tests/e2e/run-app.js` (network isolated), or a local backend serving a built
site (`SITE_DIR=… python federated_server.py`). Prove XSS with a harmless
marker (e.g. set `window.__xss=1`) and a CSP-violation listener.
