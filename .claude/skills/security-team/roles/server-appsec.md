# Server application security — `sec-server`

Scope: `federated_server.py`, `gunicorn.conf.py`, `tests/server/`. You think
like an anonymous client with curl and a botnet, and like a malicious web page
running in the admin's browser.

## How this server is built (know it before judging it)

- One process, one gunicorn worker, 4 threads. All state is module globals
  guarded by `write_lock`; slow I/O under `_io_lock` (lock order: `_io_lock`
  before `write_lock`). Anything an attacker can make slow under `write_lock`
  stalls every request.
- Every per-key structure is keyed by attacker-chosen values (device ids,
  eventIds, cells) and must be bounded (`*_MAX` caps, LRU eviction).
- Bodies are parsed with `_json_body()` regardless of Content-Type (the PWA
  sends `text/plain`). `MAX_CONTENT_LENGTH` is 64 KB.
- Client IP comes from `ProxyFix(x_for=TRUSTED_PROXY_HOPS)`; IPv6 is keyed per /64.
- `/detection` must answer throttling/storage failure with 503, never 4xx.

## Checklist

**Authentication & sessions (admin API)**
- [ ] Every `/admin/*` route is wrapped by `_admin_only` (grep the routes; a new
      route without it is Critical). Disabled mode is a 404 everywhere.
- [ ] Password compare is constant-time on fixed-length digests; no early exit
      on length; no password, digest or token in any log line or response.
- [ ] Token: ≥ 256 bits from `os.urandom`; only a hash stored; expiry enforced
      on every use; logout and eviction really invalidate; cap enforced.
- [ ] Login throttle is not disabled by `RATE_LIMIT_SCALE=0`; per-IP key uses
      the same `_client_ip()` as everything else; check behaviour when
      `TRUSTED_PROXY_HOPS` is wrong (spoofable `X-Forwarded-For` ⇒ unlimited
      per-IP guesses — the global bucket must still hold).
- [ ] CSRF: a foreign page cannot log in, log out or read admin data (Origin
      check on every admin method; no cookies are used — keep it that way).

**Input handling**
- [ ] Every client field type-checked, bounded and canonicalised before it
      reaches RAM, disk, logs or a response. Look for `str(x)` on unbounded
      input, `int()`/`float()` on huge/NaN/Infinity, nested JSON depth,
      unhashable types used as dict keys, booleans passing as ints.
- [ ] Log injection: client text in `_log()` goes through `_q()`; JSON mode
      escapes, text mode must not allow newlines/ANSI/bidi.
- [ ] Query args (`limit`, `from_id`, `device`, `species`, `level`) clamped.

**Availability**
- [ ] Find any request whose cost grows with stored data while holding
      `write_lock` (full scans of `detection_log`, HMACs per row, sorting).
- [ ] Response sizes bounded (`limit` caps); no endpoint returns unbounded lists.
- [ ] Rate-limit maps bounded; an attacker rotating keys can't evict the
      per-IP buckets (separate maps) or grow memory.
- [ ] Slowloris / slow bodies: gunicorn `gthread` with 4 threads and 120 s
      timeout — note what the platform edge absorbs; don't "fix" with workers.
- [ ] Startup: a corrupt/huge `DATA_DIR` file must not crash-loop the boot.

**CORS, headers, errors**
- [ ] Public GETs `*`; writes and all admin traffic reflect only first-party
      origins; `Vary: Origin` set; `null` origin never allowed.
- [ ] API responses: `nosniff`, `DENY`, `no-referrer`, `default-src 'none'`
      CSP, `no-store`, HSTS. Site responses: hashed-script CSP.
- [ ] Errors never include stack traces, paths, exception text with client
      input, or internal config. `/health` discloses only what an operator
      needs (check each field: is it useful to an attacker?).

**Data integrity & abuse**
- [ ] FedAvg poisoning: bounds, clipping, one vote per device per round,
      per-IP upload budget. Can sybils (free device ids, many IPs) still steer
      the model? Is that within the documented accepted risk?
- [ ] Map/hotspot spam: can one IP create hotspots or fake outbreaks at scale?
- [ ] Idempotency: eventId replay, conf-update window abuse.

**Persistence**
- [ ] Files created 0600, directory 0700; no path ever built from request data.
- [ ] Crash/partial-write safety preserved (O_APPEND, rollback, torn tail).

## Techniques

Use the pytest harness (`tests/server/conftest.py`: `harness.start(**env)`,
`post()`, `post_det()`), or a throwaway local server. Write the failing test
first; it becomes the regression test for the fix.
