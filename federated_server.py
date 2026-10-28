#!/usr/bin/env python3
"""
MosquitoNet Federated Server v13
================================
Log format (human-readable at /log.txt):

  #42  |  device: 9f3c1e7a  |  32.08, 34.79  |  2025-01-15 14:32:07 UTC
  Ae. albopictus  |  risk: HIGH  |  668.0 Hz  |  conf: 0.83

v13 — launch hardening (security + reliability audit):
  - Abuse resistance: in-process token buckets per client IP and per device
    (one gunicorn worker owns all state, so no Redis is needed). /detection
    answers 503 + Retry-After when throttled, never 429: deployed clients DELETE
    a queued find on any 4xx but retry 5xx with backoff, so throttling must
    delay a real find, never destroy it.
  - CORS: GETs stay `*` (public, credential-free data). POSTs are reflected only
    for first-party origins (+ ALLOWED_ORIGINS) and REJECTED server-side when a
    browser sends a foreign Origin — CORS alone never stops a text/plain POST
    from being sent, so without this any web page could turn its visitors'
    browsers into a distributed spam / model-poisoning botnet. flask-cors is
    gone (three 2024 advisories in the pinned 5.0.0; the policy is ~40 lines).
  - Every client-supplied field is type-checked, bounded and sanitised before it
    reaches RAM, disk, logs or the public /log (a 250 KB `disease` string was
    stored verbatim — 20k such rows = 5 GB → OOM; an unhashable eventId → 500).
  - Privacy: each report is jittered independently on the client, so averaging
    one phone's reports recovered its home to ~100 m. Public /log and /log.txt
    now shift coordinates by a secret per-device constant offset (≈ ±2 km)
    before rounding, and replace the stable device hash with a per-day keyed
    pseudonym.
  - FedAvg: magnitude bounds, one pending update per device per round, L2
    clipping of each client's delta, step weight capped, the global model
    kept in a bounded box, and a per-IP upload budget — a single sybil can no
    longer steer the global model.
  - Durability: unbuffered O_APPEND writes rolled back on failure (a full disk
    now returns 503 so the client retries, instead of acknowledging a find that
    was never written); torn-tail repair; corrupt state quarantined and rebuilt
    from the append-only log; eventIds persisted in the log so idempotency
    survives a hard kill; single-writer lock on DATA_DIR.
  - v3 client fields (f0, genus, posterior, range, snr, appVersion): optional,
    validated, and dropped individually when invalid (see _parse_detection).

Kept from v12: append-only JSONL with {"op":"conf"} records replayed at load,
bounded deque in RAM, batched fsync/state saves by a background flusher, O(1)
active_now(). MUST run as ONE process (gunicorn --workers 1, threads are fine):
all state lives in this process and a second writer would corrupt DATA_DIR.
"""

import os, re, io, sys, gzip, json, math, time, hmac, errno, atexit, base64, hashlib, shutil
import threading, ipaddress, functools, itertools, traceback
from collections import deque, OrderedDict
from datetime import datetime, timezone
import numpy as np
from flask import Flask, request, jsonify, Response
from werkzeug.exceptions import HTTPException, NotFound
from werkzeug.middleware.proxy_fix import ProxyFix
try:
    import fcntl
except ImportError:          # non-POSIX dev box: the single-writer guard is skipped
    fcntl = None

SERVICE = 'MosquitoNet v13'


# ── Logging ───────────────────────────────────────────────────────────────────
# LOG_FORMAT=json emits one JSON object per line — Railway parses `level` (line
# colour, @level: filter) and `message`, and every other key becomes a
# filterable attribute (@component:Det, @event:detection). LOG_FORMAT=text keeps
# plain lines. Unset = json on Railway, text everywhere else (local, tests).
# gunicorn.conf.py applies the same rule to gunicorn's own log lines.
def _on_railway():
    return bool(os.environ.get('RAILWAY_ENVIRONMENT_ID') or os.environ.get('RAILWAY_PROJECT_ID'))

LOG_FORMAT = os.environ.get('LOG_FORMAT', '').strip().lower()
if LOG_FORMAT not in ('json', 'text'):
    LOG_FORMAT = 'json' if _on_railway() else 'text'

_LOG_LEVELS = ('debug', 'info', 'warn', 'error')
_TAG = re.compile(r'\[([A-Za-z][A-Za-z0-9_-]*)')
# The last ADMIN_LOG_LINES records, for the operator's server view (GET
# /admin/logs). RAM only: the platform keeps the durable copy of stdout.
ADMIN_LOG_LINES = 1000
_recent_logs = deque(maxlen=ADMIN_LOG_LINES)

def _log(msg, level='info', **fields):
    """All server logging goes through here: one record, flushed immediately (a
    piped stdout is block-buffered, so a crash used to swallow the last lines).
    `level` is debug / info / warn / error. Keyword fields become structured
    attributes in JSON mode (text mode prints the message only, so the message
    must stand on its own). Client-supplied values must be interpolated via
    _q() so they can't forge extra log lines; in JSON mode json.dumps escapes
    them as well."""
    if level not in _LOG_LEVELS:
        level = 'info'
    _recent_logs.append({'t': round(time.time(), 3), 'level': level, 'msg': str(msg)[:2000],
                         'event': fields.get('event')})
    if LOG_FORMAT == 'json':
        rec = {'level': level, 'message': msg}
        m = _TAG.match(msg)
        if m:
            rec['component'] = m.group(1)
        for k, v in fields.items():
            if k not in rec and v is not None:
                rec[k] = v
        print(json.dumps(rec, default=str, separators=(',', ':')), flush=True)
    else:
        print(msg, flush=True)

def _stack(e):
    """The traceback frames of exception `e` (code locations only). The
    exception's message — which may echo client input — is deliberately left
    out: callers log it separately through _q()."""
    return ''.join(traceback.format_tb(e.__traceback__)).rstrip() or None

def _q(v, n=64):
    """Log-safe rendering of an untrusted value: truncated, and repr() escapes
    newlines / ANSI escapes / bidi overrides, so a request can't inject lines."""
    return repr(str(v)[:n])

def _env_num(name, default, lo, hi, cast=float):
    """Numeric env setting with bounds. A typo must not crash the boot (that is
    a full outage) — it is logged and the safe default used instead."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        v = cast(raw)
        if lo <= v <= hi:
            return v
    except (TypeError, ValueError):
        pass
    _log(f'[config] ignoring invalid {name}={_q(raw)} (allowed {lo}..{hi}) — using {default}', 'warn')
    return default


app = Flask(__name__)
# Reject oversized bodies before they are parsed (→ 413). A real detection is
# < 2 KB and a model upload < 1 KB; 64 KB is ample headroom while bounding what
# a single request can make json.loads allocate.
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024

# Client IPs for rate limiting. Every documented deployment (Railway, Render,
# Fly, nginx/Caddy) puts exactly ONE reverse proxy in front of gunicorn, which
# appends the real peer address to X-Forwarded-For; ProxyFix trusts only that
# many hops from the right, so a client can't spoof its IP by sending its own
# X-Forwarded-For. Set TRUSTED_PROXY_HOPS=0 if gunicorn is exposed directly
# (else the header is attacker-controlled), 2 behind Cloudflare → Railway, etc.
TRUSTED_PROXY_HOPS = _env_num('TRUSTED_PROXY_HOPS', 1, 0, 5, int)
if TRUSTED_PROXY_HOPS:
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=TRUSTED_PROXY_HOPS)


# ── Persistence ───────────────────────────────────────────────────────────────
DATA_DIR   = os.environ.get('DATA_DIR', '/tmp')
# A freshly-mounted volume may not have the directory yet; create it rather than
# failing every write for the life of the process.
try:
    os.makedirs(DATA_DIR, exist_ok=True)
except Exception as _e:
    _log(f'[DATA_DIR] could not create {DATA_DIR}: {_e} — falling back to /tmp', 'error')
    DATA_DIR = '/tmp'
STATE_FILE = os.path.join(DATA_DIR, 'mosquitonet_state.json')
LOG_FILE   = os.path.join(DATA_DIR, 'detections.jsonl')
LOCK_FILE  = os.path.join(DATA_DIR, '.mosquitonet.lock')

# How many recent detections to keep in RAM. The JSONL on disk keeps the full
# durable history; this only bounds the live window used by /log and /log.txt.
# Without this the process OOMs at ~300k detections AND — because load_all()
# re-read the whole file at boot — could never start again (permanent outage).
DETECTION_LOG_MAX = _env_num('DETECTION_LOG_MAX', 20000, 100, 2_000_000, int)
# Idempotency window (eventId → detection id). Rebuilt from the log at boot, so
# a retry that arrives after a hard kill is still recognised as a duplicate.
SEEN_EVENTS_MAX   = _env_num('SEEN_EVENTS_MAX', 20000, 100, 1_000_000, int)
FLUSH_INTERVAL    = 5.0     # seconds between background fsync + state saves
# Seconds between [Stats] summary lines in the log (0 = off).
STATS_LOG_INTERVAL = _env_num('STATS_LOG_INTERVAL', 900, 0, 86400, int)

# Single lock for in-memory state AND every append to the detection log — no
# races possible. The slow, batched I/O (fsync, state-file rewrite) runs outside
# it under _io_lock. Lock order: _io_lock before write_lock, never the reverse.
write_lock = threading.Lock()
_io_lock   = threading.RLock()

_log_fd       = None    # O_APPEND fd of the JSONL (unbuffered: see _append_entry)
_dirty        = False   # state changed since last successful save
_log_unsynced = False   # bytes written but not yet fsynced
_write_failing = False  # last append failed (disk full / volume gone) → health 'degraded'
_last_state_save = None
_lock_fh      = None    # holds the single-writer flock on DATA_DIR
single_instance = None  # True: we own DATA_DIR; False: another process does (!)
_stop = threading.Event()

# Operational counters — process lifetime only, surfaced by /health so abuse
# and storage trouble are visible without grepping logs.
counters = {k: 0 for k in ('rejected_detections', 'rejected_uploads', 'rejected_origin',
                           'rate_limited', 'clipped_updates', 'dropped_fields',
                           'write_errors', 'admin_logins', 'admin_login_failures')}
_counter_lock = threading.Lock()

def _count(name, n=1):
    with _counter_lock:
        counters[name] = counters.get(name, 0) + n


def _mark_dirty():
    """Flag state for the background flusher instead of paying a full
    serialize+fsync inside the request (that cost ~250 KB and 2 fsyncs per
    detection, capping throughput at ~100 req/s)."""
    global _dirty
    _dirty = True

def _fsync_dir(path):
    """fsync a directory so a just-renamed file survives power loss (the rename
    itself lives in the directory, not the file)."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:    os.fsync(fd)
        finally: os.close(fd)
    except OSError:
        pass     # not supported on every filesystem — best effort

def _open_log():
    """Open the append log unbuffered with O_APPEND, mode 0600 (it holds device
    hashes and coordinates). If the file does not end in a newline, the previous
    process died mid-write: terminate the torn line first — otherwise the next
    record would be glued onto it and BOTH would be unparseable at the next boot."""
    fd = os.open(LOG_FILE, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        size = os.fstat(fd).st_size
        if size:
            with open(LOG_FILE, 'rb') as f:
                f.seek(size - 1)
                if f.read(1) != b'\n':
                    os.write(fd, b'\n')
                    _log('[Log] torn final line (crash mid-write) — terminated so new records stay parseable', 'warn')
    except OSError as e:
        _log(f'[Log] tail check failed: {e}', 'warn')
    return fd

def _append_entry(entry):
    """Append one JSONL line. Called under write_lock. Returns True only if the
    WHOLE line reached the file; on any failure (ENOSPC, EIO, volume detached)
    the partial line is truncated away and False is returned, so the caller can
    answer 503 and the client keeps the find queued. (v12 swallowed the error and
    acknowledged the find — the client then deleted it: silent data loss.)
    Unbuffered os.write: a buffered writer that fails half-way keeps the rest in
    its buffer and would emit it later, corrupting the next line.
    fsync is batched by the background flusher — a hard kill can lose at most
    FLUSH_INTERVAL seconds of OS-buffered writes on power loss."""
    global _log_fd, _log_unsynced, _write_failing
    data = (json.dumps(entry, separators=(',', ':')) + '\n').encode('utf-8')
    try:
        if _log_fd is None:
            _log_fd = _open_log()
        start = os.lseek(_log_fd, 0, os.SEEK_END)
        try:
            view = memoryview(data)
            while view:
                n = os.write(_log_fd, view)
                if n <= 0:
                    raise OSError(errno.EIO, 'short write')
                view = view[n:]
        except OSError:
            try:    os.ftruncate(_log_fd, start)    # drop our partial line
            except OSError: pass
            raise
    except OSError as e:
        _write_failing = True
        _count('write_errors')
        _log(f'[append] FAILED ({errno.errorcode.get(e.errno, e.errno)}): {e}', 'error',
             event='write_failed', errno=errno.errorcode.get(e.errno, e.errno))
        return False
    _log_unsynced = True
    _write_failing = False
    return True

def _state_snapshot():
    """Shallow copy of everything the state file holds. Called under write_lock
    (~1 ms at the caps); serialising and fsyncing then happen OUTSIDE the lock —
    at 20k cells + 20k eventIds that is ~40 ms, plus an fsync that can take far
    longer on a network volume, which v12 made every request wait behind.
    (Cell dicts are only ever mutated by replacing an int value, which is safe
    to serialise concurrently.)"""
    return {
        'stats':           dict(stats),
        'next_id':         next_detection_id[0],
        'detection_cells': dict(detection_cells),
        'hotspot_cells':   dict(hotspot_cells),
        'seen_events':     dict(seen_events),
        'global_W':        global_W.tolist(),
        'global_b':        global_b.tolist(),
    }

def _save_state(payload=None):
    """Atomically replace the state file (tmp + fsync + rename + dir fsync).
    Takes the snapshot itself when not given one, so it must NOT be called with
    write_lock held. Returns True on success — the flusher only clears the dirty
    flag then, so a failed save is retried instead of forgotten."""
    global _last_state_save
    if payload is None:
        with write_lock:
            payload = _state_snapshot()
    try:
        data = json.dumps(payload, separators=(',', ':'))
        tmp = STATE_FILE + '.tmp'
        with _io_lock:                   # one writer of the tmp file at a time
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, STATE_FILE)
            _fsync_dir(DATA_DIR)
        _last_state_save = datetime.now(timezone.utc).isoformat()
        return True
    except Exception as e:
        _count('write_errors')
        _log(f'[state] save FAILED: {e}', 'error', event='write_failed')
        return False

def _flush_once(lock_timeout=-1):
    """One maintenance pass (the flusher's body; also the exit flush):
      • evict devices idle beyond ACTIVE_SEC, keeping active_now() O(1)
      • evict recent_log entries past RECENT_WINDOW (they were retained forever)
      • fsync the append log and save state at most once per FLUSH_INTERVAL
        (previously every detection paid a full state serialize + 2 fsyncs)
    The sweeps and the snapshot run under write_lock; the fsync and the state
    write run after it is released, under _io_lock only. Both registries are
    ordered oldest-first, so sweeping stops at the first live entry instead of
    scanning everything under the lock. Returns False if write_lock could not
    be taken within lock_timeout (exit path: never hang shutdown forever)."""
    global _dirty, _log_unsynced
    with _io_lock:
        if not write_lock.acquire(timeout=lock_timeout):
            return False
        try:
            now = time.time()
            while device_registry:
                h = next(iter(device_registry))
                if now - device_registry[h] <= ACTIVE_SEC:
                    break
                device_registry.pop(h, None)
                session_registry.pop(h, None)
                device_listening.pop(h, None)
            while recent_log:
                k = next(iter(recent_log))
                if now - recent_log[k]['ts'] <= RECENT_WINDOW:
                    break
                recent_log.pop(k, None)
            fd = _log_fd if _log_unsynced else None
            _log_unsynced = False
            snap = _state_snapshot() if _dirty else None
            _dirty = False
        finally:
            write_lock.release()
        if fd is not None:
            try:
                os.fsync(fd)    # fd is only ever closed under _io_lock, which we hold
            except OSError as e:
                _log(f'[sync] {e}', 'error', event='write_failed')
                _log_unsynced = True
        if snap is not None and not _save_state(snap):
            _dirty = True       # retried on the next pass instead of forgotten
        return True


# ── Load-time type checking ───────────────────────────────────────────────────
# Everything read back from disk is re-validated: a hand-edited, truncated or
# legacy file must degrade gracefully, never poison the running process (a
# string in stats['total_detections'] used to make EVERY detection a 500).
def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)

def _finite(v):
    """Strict number → float, or None. Accepts int/float and short numeric
    strings (older clients); rejects bool, NaN and ±Inf (json.loads accepts the
    NaN/Infinity literals, so they DO arrive)."""
    if isinstance(v, bool):
        return None
    try:
        if isinstance(v, (int, float)):
            f = float(v)
        elif isinstance(v, str) and len(v) <= 32:
            f = float(v)
        else:
            return None
    except (ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None

def _bstr(v, n):
    return v[:n] if isinstance(v, str) else None

def _bounded_record(rec):
    """Rebuild a detection record read from the JSONL with known keys only and
    bounded values. Rows written before v13 could carry arbitrarily large
    strings (a few hundred such rows can exhaust RAM at boot) or odd types."""
    out = {'id': rec['id']}
    for k, n in (('ts', 40), ('device', 16), ('species', 40), ('name', 64),
                 ('disease', 128), ('risk', 16), ('conf_updated', 40),
                 ('genus', 24), ('range', 8), ('appVersion', 16), ('eid', 64)):
        if k in rec:
            out[k] = _bstr(rec[k], n)
    for k in ('lat', 'lng', 'freq', 'conf', 'f0', 'snr'):
        if k in rec:
            out[k] = _finite(rec[k])
    if 'asymptomatic' in rec:
        out['asymptomatic'] = rec['asymptomatic'] is True
    post = rec.get('posterior')
    if isinstance(post, list):
        clean = [{'id': p['id'][:40], 'p': _finite(p.get('p'))}
                 for p in post[:POSTERIOR_MAX]
                 if isinstance(p, dict) and isinstance(p.get('id'), str)]
        if clean:
            out['posterior'] = clean
    return out

def _load_state_file():
    """Read and type-check the state file. Returns True if usable state was
    loaded. A file that exists but cannot be parsed is renamed aside (kept for
    forensics) and the caller rebuilds what it can from the append-only log —
    v12 silently started from zero and overwrote the evidence 5 s later."""
    global detection_cells, hotspot_cells, seen_events, global_W, global_b
    try:
        with open(STATE_FILE, encoding='utf-8') as f:
            d = json.load(f)
        if not isinstance(d, dict):
            raise ValueError('state root is not an object')
    except FileNotFoundError:
        _log('[State] no state file — fresh start (cells/counters rebuilt from the log if present)')
        return False
    except Exception as e:
        bad = f'{STATE_FILE}.corrupt-{int(time.time())}'
        try:
            os.replace(STATE_FILE, bad)
        except OSError:
            bad = '(could not move it aside)'
        _log(f'[State] CORRUPT state file ({type(e).__name__}: {_q(e, 120)}) moved to {bad} '
              f'— rebuilding counters and cells from the detection log', 'error')
        return False

    st = d.get('stats') if isinstance(d.get('stats'), dict) else {}
    for k in stats:
        v = st.get(k)
        if k == 'last_aggregate':
            if v is None or isinstance(v, str):
                stats[k] = v if v is None else v[:40]
        elif _is_int(v) and v >= 0:
            stats[k] = v
    if _is_int(d.get('next_id')) and d['next_id'] >= 1:
        next_detection_id[0] = d['next_id']

    def _cells(raw):
        out = {}
        if not isinstance(raw, dict):
            return out
        for k, v in raw.items():
            if (isinstance(k, str) and len(k) <= 80 and isinstance(v, dict)
                    and _is_int(v.get('total')) and _finite(v.get('lat')) is not None
                    and _finite(v.get('lng')) is not None and isinstance(v.get('species'), str)):
                out[k] = {'species': v['species'][:40], 'total': v['total'],
                          'risk': str(v.get('risk', 'UNKNOWN'))[:16],
                          'lat': _finite(v['lat']), 'lng': _finite(v['lng'])}
        return out
    detection_cells = _cells(d.get('detection_cells'))
    hotspot_cells   = _cells(d.get('hotspot_cells'))
    _trim(detection_cells, DETECTION_CELLS_MAX)
    _trim(hotspot_cells, DETECTION_CELLS_MAX)

    se = d.get('seen_events')
    seen_events = {k: v for k, v in se.items() if isinstance(k, str) and len(k) <= 64 and _is_int(v)} \
        if isinstance(se, dict) else {}
    _trim(seen_events, SEEN_EVENTS_MAX)

    # Restore the trained global model — but only if it is well-shaped, finite
    # and in bounds; otherwise keep the seed rather than adopt a corrupt blob.
    if 'global_W' in d and 'global_b' in d:
        if _valid_weights({'W': d['global_W'], 'b': d['global_b']}):
            global_W = np.array(d['global_W'], dtype=float)
            global_b = np.array(d['global_b'], dtype=float)
        else:
            _log('[State] persisted global model invalid — keeping seed', 'warn')
    _log(f'[State] loaded, total={stats["total_detections"]}, seen_events={len(seen_events)}, '
          f'rounds={stats["total_rounds"]}')
    return True

def _trim(d, cap):
    """Evict the oldest entries of an insertion-ordered dict down to `cap`."""
    while len(d) > cap:
        d.pop(next(iter(d)), None)

def _load_log(rebuild_cells):
    """Stream the JSONL. `detection_log` is a bounded deque, so a huge file can
    never blow up RAM at boot — only the tail is kept. {"op":"conf"} lines are
    confidence updates replayed onto the entry they refer to. Records carry
    their eventId ('eid'), so the idempotency map is rebuilt from the durable
    log rather than trusting a state file that may be up to FLUSH_INTERVAL old.
    If the state file was missing/corrupt, hotspot cells are recounted too."""
    conf_updates = {}
    eid_tail = deque(maxlen=SEEN_EVENTS_MAX)
    n_read = n_bad = max_id = 0
    try:
        with open(LOG_FILE, encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    n_bad += 1          # torn line from a crash, or garbage
                    continue
                if not isinstance(rec, dict) or not _is_int(rec.get('id')) or rec['id'] < 1:
                    n_bad += 1
                    continue
                rid, op, eid = rec['id'], rec.get('op'), rec.get('eid')
                if op == 'conf':
                    c = _finite(rec.get('conf'))
                    if c is not None and 0.0 <= c <= 1.0:
                        u = conf_updates.setdefault(rid, {})
                        u['conf'] = c
                        if isinstance(rec.get('at'), str):
                            u['at'] = rec['at'][:40]
                    if isinstance(eid, str) and len(eid) <= 64:
                        eid_tail.append((eid, rid))
                    # conf ops always follow their detection closely; forget the
                    # ones whose record has already scrolled out of the window,
                    # so this map stays bounded on a multi-million-line file.
                    if len(conf_updates) > 2 * DETECTION_LOG_MAX and detection_log:
                        head = detection_log[0]['id']
                        for k in [k for k in conf_updates if k < head]:
                            del conf_updates[k]
                    continue
                if op is not None:
                    n_bad += 1          # unknown record type (from a newer version?)
                    continue
                rec = _bounded_record(rec)
                n_read += 1
                max_id = max(max_id, rid)
                detection_log.append(rec)     # deque(maxlen) evicts the head
                if isinstance(rec.get('eid'), str):
                    eid_tail.append((rec['eid'], rid))
                if rebuild_cells:
                    _count_cell(rec)
    except FileNotFoundError:
        _log(f'[Log] starting fresh at {LOG_FILE}')
        return
    except Exception as e:
        _log(f'[Log load] {e}', 'error', stack=_stack(e))
        return
    for e in detection_log:
        u = conf_updates.get(e['id'])
        if u:
            e['conf'] = u['conf']
            if 'at' in u:
                e['conf_updated'] = u['at']
    for eid, rid in eid_tail:                  # the log is the source of truth:
        seen_events.pop(eid, None)             # newest last, so trimming evicts oldest
        seen_events[eid] = rid
    _trim(seen_events, SEEN_EVENTS_MAX)
    # Self-heal counters from the durable log. State is saved by a background
    # flusher, so a hard kill (SIGKILL/OOM) can leave state.json slightly
    # behind the append-only log. Detection ids are sequential, so the log
    # tells us where to resume — this prevents duplicate detection ids and
    # stops the public "total detections" counter going backwards.
    if max_id:
        if next_detection_id[0] <= max_id:
            _log(f'[State] next_id {next_detection_id[0]} behind log (max {max_id}) — recovering', 'warn')
            next_detection_id[0] = max_id + 1
        if stats['total_detections'] < max_id:
            _log(f'[State] total_detections {stats["total_detections"]} behind log — recovering to {max_id}', 'warn')
            stats['total_detections'] = max_id
    if n_bad:
        _log(f'[Log] skipped {n_bad} unparseable/invalid line(s)', 'warn')
    _log(f'[Log] {n_read} entries on disk; kept most recent {len(detection_log)} '
          f'(cap {DETECTION_LOG_MAX}), {len(conf_updates)} conf-updates replayed, '
          f'{len(eid_tail)} eventIds recovered'
          + (f', {len(detection_cells)} cells rebuilt' if rebuild_cells else ''))

def load_all():
    state_ok = _load_state_file()
    _load_log(rebuild_cells=not state_ok)
    if not state_ok and detection_log:
        _mark_dirty()          # persist the rebuilt state on the first flush

def _acquire_single_writer_lock():
    """Advisory flock on DATA_DIR. Two processes sharing DATA_DIR (gunicorn
    --workers 2, WEB_CONCURRENCY injected by a platform, an overlapping
    deploy) each keep their own in-memory state and overwrite each other's
    state file — silent data loss. We can't safely refuse to boot (that is an
    outage), so we shout in the logs and report it in /health instead."""
    global _lock_fh, single_instance
    if fcntl is None:
        return
    try:
        _lock_fh = open(LOCK_FILE, 'a+')
    except OSError as e:
        # Can't even create the lock file: a permissions/volume problem (which
        # /health's `writable` reports), not a second writer — don't cry wolf.
        _log(f'[lock] cannot open {LOCK_FILE}: {e} — single-writer check skipped', 'warn')
        return
    try:
        fcntl.flock(_lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        single_instance = True
    except OSError as e:
        if e.errno not in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
            _log(f'[lock] flock failed: {e} — single-writer check skipped', 'warn')
            return
        single_instance = False
        _log(f'[lock] WARNING: another process already holds {LOCK_FILE}.\n'
             '          Run exactly ONE worker per DATA_DIR (gunicorn --workers 1);\n'
             '          concurrent writers lose detections and corrupt the state file.',
             'error', event='second_writer')

@atexit.register
def _flush_on_exit():
    global _dirty
    if _stop.is_set():
        return                 # already shut down (tests / double signal)
    _stop.set()
    _log('[Exit] flushing...')
    _dirty = True              # always write a final state file
    # Append-only log: just fsync, never rewrite the whole file. Bounded wait:
    # never hang the shutdown forever behind a stuck request.
    if not _flush_once(lock_timeout=10):
        _log('[Exit] state lock busy for 10 s — fsyncing the log only', 'warn')
        with _io_lock:
            if _log_fd is not None:
                try: os.fsync(_log_fd)
                except OSError: pass
    with _counter_lock:
        ctr = dict(counters)
    _log('[Exit] done', event='exit', total_detections=stats.get('total_detections'),
         uptime_seconds=int(time.time() - start_time), **ctr)

def _graceful_exit(signum, _frame):
    """Plain `python federated_server.py` dies on SIGTERM WITHOUT running atexit,
    which would drop the batched state (counters, seen_events, trained model).
    Flush explicitly, then exit. Under gunicorn the worker raises SystemExit and
    atexit runs, so this is only installed for the standalone path."""
    _log(f'[Signal {signum}] flushing before exit')
    try:
        _flush_on_exit()
    finally:
        os._exit(0)

def _shutdown(flush=True):
    """Stop the flusher and release files/locks. For tests and embedding:
    flush=False emulates a hard kill (no final fsync/state save)."""
    global _log_fd, _lock_fh
    if flush:
        _flush_on_exit()
    _stop.set()
    with _io_lock, write_lock:          # lock order: _io_lock, then write_lock
        if _log_fd is not None:
            try: os.close(_log_fd)
            except OSError: pass
            _log_fd = None
    if _lock_fh is not None:
        try: _lock_fh.close()
        except OSError: pass
        _lock_fh = None


# ── Model ─────────────────────────────────────────────────────────────────────
global_W = np.array([[ 2.8,0.9,1.2,0.8],[-0.6,0.7,1.0,0.7],
                     [-1.1,0.7,1.0,0.6],[ 1.8,0.8,1.3,0.9]], dtype=float)
global_b = np.array([-0.4,-0.3,-0.3,-0.4], dtype=float)

# Poisoning resistance. The model is a 4×4 logistic layer; the seed is within
# ±3 and the client trains with LR 0.01 and clipped gradients, so no honest
# upload comes anywhere near these bounds.
FED_WEIGHT_ABS_MAX = 50.0   # any |w| above this is rejected outright (400)
# Clipping bounds each ROUND, not the sum of rounds: sybils (device ids are free)
# could drive many rounds in one direction until the global model itself passed
# FED_WEIGHT_ABS_MAX — honest clients merging it would then be rejected, and the
# persisted model would fail validation and silently reset at the next boot. So
# the global model is kept inside a box well within the upload bound.
FED_GLOBAL_ABS_MAX = 20.0
FED_CLIP_NORM  = _env_num('FED_CLIP_NORM', 2.0, 0.01, 1000.0)   # L2 cap on (client − global)
FED_MAX_STEPS  = 50         # step weight cap (honest clients send 20; was 500 → 25× leverage)


# ── State ─────────────────────────────────────────────────────────────────────
device_registry   = {}     # ACTIVE devices, oldest-seen first; hash → last_seen
known_devices     = OrderedDict()  # recently seen hashes (ever-count dedupe), LRU-bounded
session_registry  = {}
device_listening  = {}     # active device → its last heartbeat's `listening` flag (RAM only)
pending_updates   = {}     # device hash → latest update this round (one vote per device)
detection_log     = deque(maxlen=DETECTION_LOG_MAX)   # bounded live window
detection_cells   = {}     # LRU by last detection: hot cells stay, cold ones age out
hotspot_cells     = {}
next_detection_id = [1]
recent_log        = {}     # rkey → {det_id, ts, conf, entry}, oldest first (swept by _flusher)
RECENT_WINDOW     = 60     # seconds: conf update allowed (exactly 1 minute)
seen_events       = {}     # clientEventId → det_id (idempotency: retries never double-count)

# Hard caps on every per-key structure. All keys (device ids, coordinates,
# eventIds) are chosen by anonymous clients, so without a cap an attacker who
# rotates them grows memory — and the state file rewritten every 5 s — forever.
ACTIVE_MAX          = 200_000
KNOWN_DEVICES_MAX   = 200_000
RECENT_MAX          = 100_000
DETECTION_CELLS_MAX = _env_num('DETECTION_CELLS_MAX', 20000, 100, 1_000_000, int)

# ── Detection validation (defence in depth; the client also gates at 0.70) ──────
MIN_CONF     = 0.5         # reject anything below this — well under the client's 0.70 gate,
                          # so no real client find is affected, but blocks conf=0 / spoofed junk
FREQ_MIN     = 50.0        # Hz — below any mosquito wingbeat fundamental
FREQ_MAX     = 2000.0      # Hz — above the 3rd harmonic of the highest species
KNOWN_SPECIES = {
    'anopheles', 'anopheles_stephensi', 'aedes_aegypti', 'aedes_albopictus',
    'culex', 'aedes_japonicus', 'aedes_vexans', 'mansonia_uniformis',
    'culex_pipiens', 'culiseta_annulata', 'ochlerotatus_caspius', 'toxorhynchites',
}
KNOWN_RISKS  = {'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NONE', 'UNKNOWN'}
# v3 optional fields
GENERA       = {g.lower(): g for g in ('Anopheles', 'Aedes', 'Culex', 'Mansonia',
                                       'Culiseta', 'Toxorhynchites')}
RANGES       = {'near', 'far'}
POSTERIOR_MAX = 5
POSTERIOR_EPS = 0.01       # Σp may exceed 1 by client-side rounding, not more
SNR_MIN, SNR_MAX = -20.0, 120.0
APP_VERSION_RE = re.compile(r'[0-9A-Za-z.+-]{1,16}')
EVENT_ID_RE    = re.compile(r'[A-Za-z0-9._:-]{1,64}')   # UUID or base36-hex from the client
TS_MAX_FUTURE  = 24 * 3600          # clock skew tolerated on client timestamps
TS_MAX_AGE     = 365 * 24 * 3600    # queued offline finds can be old, not ancient
# Control characters, bidi overrides and zero-width characters: never useful in a
# species/disease label, but they forge lines in /log.txt and spoof display text.
_CTRL_RE = re.compile('[\x00-\x1f\x7f-\x9f​-‏ -‮⁠-⁩﻿]')

stats = {
    'total_detections': 0,
    'total_sessions':   0,
    'total_uploads':    0,
    'total_rounds':     0,
    'last_aggregate':   None,
    'ever_devices':     0,     # monotonic count of distinct devices ever seen
}
ACTIVE_SEC        = 90
HOTSPOT_THRESHOLD = 100
MIN_UPLOADS       = 3
start_time        = time.time()


# ── Public-output privacy ─────────────────────────────────────────────────────
# Clients jitter coordinates by up to ±~1.4 km, but independently per report, so
# the mean of one device's reports converges on its true location — and that
# works even without a device id, by clustering reports around a home. Rounding
# alone doesn't help: the jitter is wider than any useful grid, so it acts as
# dither and the mean of rounded values still converges. Public endpoints
# therefore show each report shifted by a SECRET, PER-DEVICE, CONSTANT offset
# (±PUBLIC_GEO_OFFSET degrees: ≈ ±2.2 km north–south, less east–west away from
# the equator) before rounding: averaging any number of reports, over any number
# of days, converges on true + offset, which an outsider cannot remove. The
# device itself appears as a keyed per-day pseudonym, which stops linking days
# BY ID — but a phone's public point is the same every day, so its days link by
# location. The point is therefore snapped to a PUBLIC_COORD_STEP grid (0.05°,
# the hotspot cell size): many phones share each grid point, and cross-day
# linkage drops from ~80 % to ~20-35 % in a simulated city. The raw jittered
# values and the stable hash stay in the private JSONL for the operator.
PUBLIC_COORD_DECIMALS = _env_num('PUBLIC_COORD_DECIMALS', 2, 0, 4, int)
PUBLIC_GEO_OFFSET     = _env_num('PUBLIC_GEO_OFFSET', 0.02, 0.0, 0.5)
PUBLIC_COORD_STEP     = _env_num('PUBLIC_COORD_STEP', 0.05, 0.0, 1.0)

def _public_coord(v, lo, hi):
    if PUBLIC_COORD_STEP:
        v = round(v / PUBLIC_COORD_STEP) * PUBLIC_COORD_STEP
    return round(min(hi, max(lo, v)), PUBLIC_COORD_DECIMALS)
_DAY_RE = re.compile(r'\d{4}-\d{2}-\d{2}')

def _server_secret():
    """Key for pseudonyms and offsets. It must outlive restarts — a new key
    would give every device a fresh offset, and averaging across keys undoes
    the protection — so it is created once in DATA_DIR (0600) unless
    PSEUDONYM_SECRET provides one."""
    env = os.environ.get('PSEUDONYM_SECRET')
    if env:
        return env.encode()
    path = os.path.join(DATA_DIR, '.server_secret')
    for _ in range(2):
        try:
            with open(path, 'rb') as f:
                key = f.read()
            if len(key) >= 32:
                return key
        except FileNotFoundError:
            pass
        except OSError as e:
            _log(f'[privacy] cannot read {path}: {e}', 'error')
            break
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'wb') as f:
                f.write(os.urandom(32))
                f.flush()
                os.fsync(f.fileno())
        except FileExistsError:
            continue                     # another process won the race: read theirs
        except OSError as e:
            _log(f'[privacy] cannot persist {path}: {e}', 'error')
            break
    _log('[privacy] WARNING: using a temporary key — public offsets/pseudonyms change on restart', 'warn')
    return os.urandom(32)

_PSEUDO_KEY = _server_secret()

@functools.lru_cache(maxsize=65536)
def _pseudonym(device, day):
    return hmac.new(_PSEUDO_KEY, f'{device}|{day}'.encode(), hashlib.sha256).hexdigest()[:16]

@functools.lru_cache(maxsize=65536)
def _geo_offset(device):
    """(Δlat, Δlng) in ±PUBLIC_GEO_OFFSET: uniform, secret, fixed per device."""
    d = hmac.new(_PSEUDO_KEY, f'geo|{device}'.encode(), hashlib.sha256).digest()
    u = [int.from_bytes(d[i:i + 8], 'big') / 2.0 ** 64 for i in (0, 8)]
    return tuple((2.0 * x - 1.0) * PUBLIC_GEO_OFFSET for x in u)

def _public_device(e):
    ts = e.get('ts')
    day = ts[:10] if isinstance(ts, str) and _DAY_RE.match(ts) else 'na'
    return _pseudonym(str(e.get('device', '')), day)

_PUBLIC_KEYS = ('id', 'ts', 'species', 'name', 'disease', 'freq', 'conf', 'risk',
                'asymptomatic', 'conf_updated',
                'f0', 'genus', 'posterior', 'range', 'snr', 'appVersion')

def _public_row(e):
    """What /log may show of a stored entry: an allowlist of keys (private ones
    such as 'eid' never leak), offset + rounded coordinates, pseudonymous device."""
    out = {k: e[k] for k in _PUBLIC_KEYS if k in e}
    out['device'] = _public_device(e)
    lat, lng = _finite(e.get('lat')), _finite(e.get('lng'))
    if lat is None or lng is None:
        out['lat'] = out['lng'] = None
    else:
        dlat, dlng = _geo_offset(str(e.get('device', '')))
        out['lat'] = _public_coord(lat + dlat, -90.0, 90.0)
        out['lng'] = _public_coord(lng + dlng, -180.0, 180.0)
    return out


# ── Helpers ───────────────────────────────────────────────────────────────────
def dh(raw):
    return hashlib.sha256(str(raw)[:256].encode()).hexdigest()[:16]

def active_now():
    """O(1). `device_registry` holds only devices seen within ACTIVE_SEC — the
    background flusher evicts the rest — so its length IS the active count.
    (Previously this scanned a never-evicted dict on every heartbeat, which made
    total work O(devices^2) at 10 heartbeats/minute/device.)"""
    return len(device_registry)

def touch(h):
    # pop + re-insert keeps the dict ordered by last-seen, so the flusher can
    # sweep idle devices from the head in O(evicted) and the cap evicts the
    # longest-idle device.
    device_registry.pop(h, None)
    device_registry[h] = time.time()
    while len(device_registry) > ACTIVE_MAX:
        old = next(iter(device_registry))
        device_registry.pop(old, None)
        session_registry.pop(old, None)
        device_listening.pop(old, None)
    if h in known_devices:
        known_devices.move_to_end(h)
    else:
        # Dedupe window is bounded (it was an unbounded set — a heartbeat flood
        # with random ids grew it forever). A device returning after falling
        # out of the window, or after a restart, is counted again: the public
        # "ever" figure is an upper bound, as it always was across restarts.
        known_devices[h] = None
        if len(known_devices) > KNOWN_DEVICES_MAX:
            known_devices.popitem(last=False)
        stats['ever_devices'] = stats.get('ever_devices', 0) + 1
        _mark_dirty()

def cell_key(lat, lng, sp):
    return f'{round(float(lat)/0.05)*0.05:.3f},{round(float(lng)/0.05)*0.05:.3f},{sp}'

def _count_cell(entry):
    """Count a detection into its 0.05° hotspot cell. Called under write_lock
    (or single-threaded at boot when rebuilding from the log)."""
    lat, lng, sp = entry.get('lat'), entry.get('lng'), entry.get('species')
    if lat is None or lng is None or not isinstance(sp, str):
        return
    try:
        ck = cell_key(lat, lng, sp)
        cell = detection_cells.pop(ck, None)      # re-insert = most recently used
        if cell is None:
            cell = {'species': sp, 'total': 0, 'risk': entry.get('risk') or 'UNKNOWN',
                    'lat': round(float(lat)/0.05)*0.05,
                    'lng': round(float(lng)/0.05)*0.05}
        detection_cells[ck] = cell
        cell['total'] += 1
        if cell['total'] >= HOTSPOT_THRESHOLD:
            hotspot_cells.pop(ck, None)
            hotspot_cells[ck] = dict(cell)
            _trim(hotspot_cells, DETECTION_CELLS_MAX)
        _trim(detection_cells, DETECTION_CELLS_MAX)   # evicts the coldest cell
    except Exception as e:
        _log(f'[Cell] {e}', 'error', stack=_stack(e))

def _remember_event(event_id, det_id):
    """Record an eventId→det_id mapping for idempotency. Called under write_lock.
    Evicts the oldest entries (dicts preserve insertion order) once the cap is hit."""
    seen_events[event_id] = det_id
    _trim(seen_events, SEEN_EVENTS_MAX)

def full_stats():
    return {
        'active_now':       active_now(),
        'unique_devices':   stats.get('ever_devices', 0),
        'ever_connected':   stats.get('ever_devices', 0),
        'total_sessions':   stats['total_sessions'],
        'total_uploads':    stats['total_uploads'],
        'total_rounds':     stats['total_rounds'],
        'total_detections': stats['total_detections'],
        'hotspot_count':    len(hotspot_cells),
        'log_size':         len(detection_log),
        'uptime_seconds':   int(time.time() - start_time),
    }

def _valid_weights(w):
    """True iff w = {'W': 4x4, 'b': len-4}, all finite numbers with |v| ≤
    FED_WEIGHT_ABS_MAX. Rejects the poison payloads that would otherwise
    silently corrupt the global model (1e300 is finite, and survived v12).
    Shape is validated on the RAW lists BEFORE np.array() — otherwise a crafted
    100M-element payload would allocate hundreds of MB and OOM the worker before
    the shape check could reject it."""
    if not isinstance(w, dict):
        return False
    W_raw, b_raw = w.get('W'), w.get('b')
    nrow, ncol = global_W.shape
    if not (isinstance(W_raw, list) and len(W_raw) == nrow):
        return False
    if not (isinstance(b_raw, list) and len(b_raw) == global_b.shape[0]):
        return False
    for row in W_raw:
        if not (isinstance(row, list) and len(row) == ncol):
            return False
    for v in [v for row in W_raw for v in row] + b_raw:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False
        try:
            f = float(v)                     # a 400-digit int overflows here
        except OverflowError:
            return False
        if not (math.isfinite(f) and abs(f) <= FED_WEIGHT_ABS_MAX):
            return False
    return True

def _disk_free_mb():
    try:
        return shutil.disk_usage(DATA_DIR).free // (1024 * 1024)
    except OSError:
        return None

_stats_prev = {}    # counters + total detections as of the previous [Stats] line

def _log_stats():
    """One operational summary line: proves the process is alive in the logs
    and shows the trend (traffic, abuse, storage) since the previous line,
    without logging every heartbeat. Level error while writes are failing."""
    global _stats_prev
    with write_lock:
        snap = dict(full_stats())
        failing = _write_failing
    with _counter_lock:
        now = dict(counters)
    now['detections'] = snap['total_detections']
    d = {k: v - _stats_prev.get(k, v if k == 'detections' else 0) for k, v in now.items()}
    _stats_prev = now
    rejected = d['rejected_detections'] + d['rejected_uploads'] + d['rejected_origin']
    disk = _disk_free_mb()
    _log(f'[Stats] active={snap["active_now"]} detections={snap["total_detections"]} '
         f'(+{d["detections"]}) uploads={snap["total_uploads"]} rounds={snap["total_rounds"]} '
         f'rate_limited=+{d["rate_limited"]} rejected=+{rejected} '
         f'write_errors=+{d["write_errors"]} disk_free_mb={disk}',
         'error' if failing or d['write_errors'] else 'info',
         event='stats', active_now=snap['active_now'], total_detections=snap['total_detections'],
         new_detections=d['detections'], total_uploads=snap['total_uploads'],
         total_rounds=snap['total_rounds'], rate_limited=d['rate_limited'], rejected=rejected,
         write_errors=d['write_errors'], disk_free_mb=disk, uptime_seconds=snap['uptime_seconds'])

def _flusher():
    next_stats = time.monotonic() + STATS_LOG_INTERVAL
    while not _stop.wait(FLUSH_INTERVAL):
        try:
            _flush_once()
            if STATS_LOG_INTERVAL and time.monotonic() >= next_stats:
                next_stats = time.monotonic() + STATS_LOG_INTERVAL
                _log_stats()
        except Exception as e:
            _log(f'[flusher] {e}', 'error', stack=_stack(e))

def fedavg(updates):
    """Robust FedAvg over client MODEL WEIGHTS.
    Each client's delta from the current global model is L2-clipped to
    FED_CLIP_NORM, then the step-weighted mean delta is blended in at 0.7.
    For honest updates (delta under the clip) this is exactly the v12 rule
    0.3·global + 0.7·Σ(steps·W)/Σsteps — but one poisoned upload can now move
    the model by at most 0.7·FED_CLIP_NORM per round, however extreme it is
    (Sun et al. 2019, "Can you really backdoor federated learning?")."""
    global global_W, global_b
    # Keep only structurally valid contributions; an invalid upload must NOT count
    # its steps toward the weighted average (the old code let it bias the result).
    valid = [u for u in updates if _valid_weights(u['weights'])]
    total = sum(u['steps'] for u in valid)
    if not total:
        return
    dW = np.zeros_like(global_W); db = np.zeros_like(global_b)
    for u in valid:
        uW = np.array(u['weights']['W'], dtype=float) - global_W
        ub = np.array(u['weights']['b'], dtype=float) - global_b
        norm = math.sqrt(float(np.sum(uW * uW) + np.sum(ub * ub)))
        if norm > FED_CLIP_NORM:
            uW *= FED_CLIP_NORM / norm
            ub *= FED_CLIP_NORM / norm
            _count('clipped_updates')
        w = u['steps'] / total
        dW += w * uW
        db += w * ub
    # Conservative 0.3 global / 0.7 new keeps a single round from dominating
    # while still converging.
    cand_W = np.clip(global_W + 0.7 * dW, -FED_GLOBAL_ABS_MAX, FED_GLOBAL_ABS_MAX)
    cand_b = np.clip(global_b + 0.7 * db, -FED_GLOBAL_ABS_MAX, FED_GLOBAL_ABS_MAX)
    if not (np.all(np.isfinite(cand_W)) and np.all(np.isfinite(cand_b))):
        _log('[FedAvg] non-finite result — keeping previous global model', 'warn')
        return
    global_W, global_b = cand_W, cand_b
    stats['total_rounds'] += 1
    stats['last_aggregate'] = datetime.now(timezone.utc).isoformat()


# ── Rate limiting ─────────────────────────────────────────────────────────────
class _RateLimiter:
    """Token buckets in an LRU-bounded map (keys are attacker-chosen, so the
    map must not grow without bound; an evicted bucket simply restarts full).
    One process owns all traffic (1 gunicorn worker), so in-process is exact."""

    def __init__(self, max_keys):
        self._lock = threading.Lock()
        self._buckets = OrderedDict()      # key → [tokens, last_refill]
        self._max = max_keys

    def hit(self, key, capacity, rate, now=None):
        """Take one token. Returns 0.0 if allowed, else seconds until one is free."""
        now = time.monotonic() if now is None else now
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = self._buckets[key] = [capacity, now]
                if len(self._buckets) > self._max:
                    self._buckets.popitem(last=False)
            else:
                b[0] = min(capacity, b[0] + (now - b[1]) * rate)
                b[1] = now
                self._buckets.move_to_end(key)
            if b[0] >= 1.0:
                b[0] -= 1.0
                return 0.0
            return (1.0 - b[0]) / rate

# (burst capacity, sustained refill per second), multiplied by RATE_LIMIT_SCALE
# (0 disables limiting). Per-IP buckets are sized for carrier-grade NAT / campus
# Wi-Fi where hundreds of honest phones share one address; per-device buckets
# sit well above the real client's cadence (heartbeat every 10 s + 30 s, stats
# every 15 s, an upload per 20 training steps) but stop a runaway or scripted
# client. Device ids are client-chosen, so the per-IP buckets are the real
# abuse control and the per-device ones catch a single misbehaving install.
RATE_LIMITS = {
    'ip:any':        (600, 20.0),   # every request from one IP
    'ip:write':      (120, 1.0),    # POST /detection + /federated/upload per IP
    'ip:log':        (30, 0.5),     # /log, /log.txt — up to 5000 rows each
    'ip:upload':     (10, 1 / 60),  # model uploads: honest phones send one per 20 training
                                    # steps; this caps how many FedAvg rounds one IP can drive
    'ip:health':     (120, 2.0),    # own bucket: probes must not starve behind other traffic
    'dev:detection': (60, 0.25),    # burst = draining an offline queue; then 15/min
    'dev:heartbeat': (60, 1.0),     # ~9/min per open tab; leaves room for several tabs
    'dev:upload':    (6, 1 / 30),
}
RATE_LIMIT_SCALE = _env_num('RATE_LIMIT_SCALE', 1.0, 0.0, 1000.0)
_ip_limiter  = _RateLimiter(50_000)
_dev_limiter = _RateLimiter(100_000)   # separate map: rotating device ids can't evict IP buckets

def _client_ip():
    ip = request.remote_addr or 'unknown'
    if ':' in ip:
        try:
            addr = ipaddress.ip_address(ip)
            if addr.ipv4_mapped:          # ::ffff:1.2.3.4 (dual-stack socket / proxy)
                return str(addr.ipv4_mapped)   # …would otherwise all collapse into ::/64
            # One IPv6 subscriber typically holds a whole /64; keying on the full
            # address would hand an attacker 2^64 fresh buckets.
            return str(ipaddress.ip_network(f'{ip}/64', strict=False))
        except ValueError:
            pass
    return ip

def _throttled(scope, key):
    """Take a token from bucket (scope, key): 0.0 if allowed, else wait seconds."""
    if RATE_LIMIT_SCALE <= 0:
        return 0.0
    cap, rate = RATE_LIMITS[scope]
    lim = _ip_limiter if scope.startswith('ip:') else _dev_limiter
    wait = lim.hit(f'{scope}|{key}', cap * RATE_LIMIT_SCALE, rate * RATE_LIMIT_SCALE)
    if wait:
        _count('rate_limited')
    return wait

def _rate_limited(wait):
    retry = max(1, math.ceil(wait))
    if request.path == '/detection':
        # 503, not 429: deployed clients drop a queued find on ANY 4xx but retry
        # 5xx with exponential backoff — a throttled find is delayed, never lost.
        resp = jsonify({'received': False, 'reason': 'rate limited', 'retry_after': retry})
        resp.status_code = 503
    else:
        resp = jsonify({'error': 'rate limited', 'retry_after': retry})
        resp.status_code = 429
    resp.headers['Retry-After'] = str(retry)
    return resp


# ── CORS + request guard ──────────────────────────────────────────────────────
# Reads (GET/HEAD) are public, credential-free data: `*` is fine and lets
# researchers' dashboards use /log and /hotspots. Writes (POST) are reflected
# only for first-party origins. ALLOWED_ORIGINS (comma-separated; ALLOWED_ORIGIN
# is accepted too, as older docs named it) EXTENDS the built-in list and never
# replaces it, so a typo in the env var can't lock out the production app.
_ORIGIN_PATTERNS = [re.compile(p) for p in (
    r'https://uriiger-sketch\.github\.io',
    r'https://([a-z0-9-]+\.)*mosquito\.charity',
    r'https?://(localhost|127\.0\.0\.1|\[::1\])(:\d{1,5})?',
)]

def _norm_origin(o):
    """'https://Example.org/path/' → 'https://example.org' (origins carry no path)."""
    o = o.strip().lower()
    m = re.match(r'^([a-z][a-z0-9+.-]*://[^/?#]+)', o)
    return m.group(1) if m else o

_EXTRA_ORIGINS = {
    _norm_origin(o)
    for var in ('ALLOWED_ORIGINS', 'ALLOWED_ORIGIN')
    for o in os.environ.get(var, '').split(',')
    if o.strip() and o.strip() != '*'
}

def _origin_allowed(origin):
    if not origin:
        return False
    o = origin.strip().lower()
    # 'null' (sandboxed iframes, file://) is deliberately NOT allowed: any page
    # can produce it, so it would reopen the drive-by hole.
    return o in _EXTRA_ORIGINS or any(p.fullmatch(o) for p in _ORIGIN_PATTERNS)

def _reject_origin():
    _count('rejected_origin')
    _log(f'[CORS] rejected {request.method} {_q(request.path)} from origin '
          f'{_q(request.headers.get("Origin"))}', 'warn', event='rejected_origin')
    if request.path == '/detection':
        return jsonify({'received': False, 'reason': 'origin not allowed'}), 403
    return jsonify({'error': 'origin not allowed'}), 403

def _is_admin_path():
    return request.path == '/admin' or request.path.startswith('/admin/')

@app.before_request
def _guard():
    """Per-IP rate limits and the Origin check for writes. OPTIONS (preflight,
    no state touched) is exempt; /health has its own generous bucket so
    platform liveness probes never starve behind a busy shared IP."""
    if request.method == 'OPTIONS':
        return None
    ip = _client_ip()
    if request.path == '/health':
        wait = _throttled('ip:health', ip)
        return _rate_limited(wait) if wait else None
    wait = _throttled('ip:any', ip)       # first, so even rejected requests cost a token
    if wait:
        return _rate_limited(wait)
    if request.method == 'POST' or _is_admin_path():
        origin = request.headers.get('Origin')
        # No Origin header = not a browser cross-site request (curl, native
        # app); those can forge anything anyway and are handled by rate limits.
        if origin is not None and not _origin_allowed(origin):
            return _reject_origin()
    if request.method == 'POST' and request.path in ('/detection', '/federated/upload'):
        wait = _throttled('ip:write', ip)
    if not wait and request.method == 'POST' and request.path == '/federated/upload':
        wait = _throttled('ip:upload', ip)
    if not wait and request.path in ('/log', '/log.txt'):
        wait = _throttled('ip:log', ip)
    if wait:
        return _rate_limited(wait)
    return None

@app.after_request
def cors_hdr(r):
    origin = request.headers.get('Origin')
    method = request.method
    if method == 'OPTIONS':
        method = request.headers.get('Access-Control-Request-Method', 'POST').upper()
    # The admin API is never `*`, reads included: only the app's own origins
    # may read an operator's answers.
    if method in ('GET', 'HEAD') and not _is_admin_path():
        r.headers['Access-Control-Allow-Origin'] = '*'
    else:
        r.headers.add('Vary', 'Origin')
        if _origin_allowed(origin):
            r.headers['Access-Control-Allow-Origin'] = origin
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type,Accept,Authorization'
    r.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    if request.method == 'OPTIONS':
        r.headers['Access-Control-Max-Age'] = '600'
    # Security headers: this is a JSON/text API — nothing here is ever meant to
    # be rendered, framed or cached by an intermediary.
    r.headers.setdefault('X-Content-Type-Options', 'nosniff')
    r.headers.setdefault('X-Frame-Options', 'DENY')
    r.headers.setdefault('Referrer-Policy', 'no-referrer')
    r.headers.setdefault('Content-Security-Policy', "default-src 'none'; frame-ancestors 'none'")
    r.headers.setdefault('Cross-Origin-Resource-Policy', 'cross-origin')
    r.headers.setdefault('Strict-Transport-Security', 'max-age=31536000')
    r.headers.setdefault('Cache-Control', 'no-store')
    return r

@app.route('/', defaults={'p':''}, methods=['OPTIONS'])
@app.route('/<path:p>', methods=['OPTIONS'])
def opts(p): return '', 204


# ── Errors: JSON, never a stack trace or an internal path ─────────────────────
@app.errorhandler(HTTPException)
def _http_error(e):
    body = {'error': (e.name or 'error').lower()}
    if request.path == '/detection':
        body = {'received': False, 'reason': body['error']}
    resp = jsonify(body)
    resp.status_code = e.code or 500
    if getattr(e, 'valid_methods', None):          # keep `Allow` on 405
        resp.headers['Allow'] = ', '.join(e.valid_methods)
    return resp

@app.errorhandler(Exception)
def _internal_error(e):
    _log(f'[500] {request.method} {_q(request.path)}: {type(e).__name__}: {_q(e, 200)}', 'error',
         event='internal_error', exc_type=type(e).__name__, stack=_stack(e))
    if request.path == '/detection':
        return jsonify({'received': False, 'reason': 'internal error'}), 500
    return jsonify({'error': 'internal error'}), 500

def _json_body():
    """Parse the body as JSON whatever its Content-Type (the PWA posts
    text/plain to avoid a CORS preflight). Never raises for bad input — a body
    nested 100k deep raises RecursionError, which get_json(silent=True) does not
    catch — but lets 413 (body too large) propagate."""
    try:
        return request.get_json(force=True, silent=True)
    except HTTPException:
        raise
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════

@app.route('/heartbeat', methods=['GET','POST'])
def heartbeat():
    if request.method == 'GET':
        raw_id = request.args.get('deviceId', 'anon')
        sess   = request.args.get('sess', '')
        listening = request.args.get('listening')
    else:
        # A non-dict JSON body (e.g. `[1,2]`) is truthy, so `or {}` would not fire
        # and .get() raised AttributeError → unhandled 500. Guard on the type.
        body   = _json_body()
        body   = body if isinstance(body, dict) else {}
        raw_id = body.get('deviceId', 'anon')
        # v3 clients POST the heartbeat so the raw device id never appears in a
        # URL (URLs end up in platform/CDN access logs next to the client IP).
        # They send the session start time as a number.
        sess   = body.get('sess', '')
        listening = body.get('listening')
        if isinstance(sess, (int, float)) and not isinstance(sess, bool):
            sess = str(int(sess)) if math.isfinite(sess) and sess > 0 else ''
    sess = sess[:32] if isinstance(sess, str) else ''   # stored per device: bound it
    # v3 clients say whether the microphone is on (1) or the app is just open (0).
    listening = (None if listening is None
                 else listening is True or listening == 1 or listening in ('1', 'true'))
    h = dh(raw_id)
    wait = _throttled('dev:heartbeat', h)
    if wait:
        return _rate_limited(wait)
    with write_lock:
        new_s = sess and session_registry.get(h) != sess
        touch(h)
        if listening is not None:
            device_listening[h] = listening
        if new_s:
            session_registry[h] = sess
            stats['total_sessions'] += 1
            _mark_dirty()
        snap = dict(full_stats())   # snapshot under the lock (reading device_registry
                                    # unlocked could raise "dict changed size")
    return jsonify(snap)


def _reject(reason, code=400):
    """Reject a malformed / invalid detection with a logged 4xx (never a 500).
    `reason` may contain client input: it is truncated for the response and
    escaped for the log."""
    _count('rejected_detections')
    reason = str(reason)[:160]
    _log(f'[Det REJECT {code}] {_q(reason, 160)}', 'warn', event='rejected_detection')
    return jsonify({'received': False, 'reason': reason}), code

def _clean_str(v, n):
    """Single-line printable string of at most n chars, or None."""
    if not isinstance(v, str):
        return None
    v = _CTRL_RE.sub(' ', v[:4 * n]).strip()
    return v[:n] or None

def _valid_ts(v, now):
    """A client ISO-8601 timestamp within [now − 1 y, now + 1 d], else None."""
    if not isinstance(v, str) or not (10 <= len(v) <= 40):
        return None
    try:
        dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        t = dt.timestamp()
    except (OverflowError, OSError, ValueError):
        return None
    return v if now - TS_MAX_AGE <= t <= now + TS_MAX_FUTURE else None

def _parse_posterior(v):
    """≤ POSTERIOR_MAX entries of {id: known species, p: 0..1}, distinct ids,
    Σp ≤ 1 + ε. Returns the normalised list or None (whole field dropped: a
    partially-kept distribution would misstate the classifier's belief)."""
    if not isinstance(v, list) or not (1 <= len(v) <= POSTERIOR_MAX):
        return None
    out, seen, total = [], set(), 0.0
    for item in v:
        if not isinstance(item, dict):
            return None
        sid, p = item.get('id'), _finite(item.get('p'))
        if not isinstance(sid, str) or sid not in KNOWN_SPECIES or sid in seen:
            return None
        if p is None or isinstance(item.get('p'), str) or not (0.0 <= p <= 1.0):
            return None
        seen.add(sid)
        total += p
        out.append({'id': sid, 'p': round(p, 4)})
    return out if total <= 1.0 + POSTERIOR_EPS else None

def _parse_detection(d, now):
    """Validate one /detection payload → (fields, None) or (None, reason).

    Core fields (species, confidence, frequency, coordinates, eventId) are
    strict: failing them is a 400, which the client treats as permanent and
    drops — correct, since that payload can never succeed on retry.

    Everything else is best effort. Legacy display fields (speciesName, disease,
    risk, ts) are sanitised/bounded in place. The OPTIONAL v3 fields (f0, genus,
    posterior, range, snr, appVersion) are validated individually and an invalid
    one is DROPPED — named in the response's `dropped` list — while the
    detection is still accepted. Rationale: deployed clients delete a queued
    find permanently on any 4xx, so rejecting the whole upload because, say, a
    newer client sent a posterior entry for a species this server doesn't know
    yet would destroy a real, otherwise valid observation to protect an
    auxiliary field. Dropping keeps the core data and still keeps junk out."""
    dropped = []

    species = d.get('species')
    if not isinstance(species, str) or species not in KNOWN_SPECIES:
        return None, f'unknown species: {str(species)[:40]!r}'

    # ── Numeric parsing — guarded so bad input is a clean 400, not an unhandled 500 ──
    conf, freq = _finite(d.get('confidence')), _finite(d.get('frequency'))
    if conf is None or freq is None:
        return None, 'confidence/frequency not numeric'
    conf, freq = round(conf, 3), round(freq, 1)
    if not (0.0 <= conf <= 1.0):
        return None, f'confidence out of range: {conf}'
    if conf < MIN_CONF:
        return None, f'confidence below floor: {conf} < {MIN_CONF}'
    if not (FREQ_MIN <= freq <= FREQ_MAX):
        return None, f'frequency out of range: {freq}'

    lat_raw, lng_raw = d.get('lat'), d.get('lng')
    lat = None if lat_raw is None else _finite(lat_raw)
    lng = None if lng_raw is None else _finite(lng_raw)
    if (lat_raw is not None and lat is None) or (lng_raw is not None and lng is None):
        return None, 'lat/lng not numeric'
    if lat is not None and not (-90.0 <= lat <= 90.0):
        return None, f'latitude out of range: {lat}'
    if lng is not None and not (-180.0 <= lng <= 180.0):
        return None, f'longitude out of range: {lng}'
    if (lat is None) != (lng is None):
        lat = lng = None            # half a coordinate places nothing on a map
        dropped.append('lat/lng')

    event_id = d.get('eventId')
    if _is_int(event_id):
        event_id = str(event_id)
    if event_id is not None and not (isinstance(event_id, str) and EVENT_ID_RE.fullmatch(event_id)):
        # Was stored verbatim: a list → unhashable → 500; a 60 KB string × the
        # idempotency map = hundreds of MB, rewritten to disk every 5 s.
        return None, 'invalid eventId'

    ts = d.get('ts')
    ts_ok = _valid_ts(ts, now)
    if ts is not None and ts_ok is None:
        dropped.append('ts')        # replaced by server time below
    risk = _clean_str(d.get('risk'), 16)
    risk = risk.upper() if risk and risk.upper() in KNOWN_RISKS else 'UNKNOWN'
    asymp = d.get('asymptomatic', False)

    fields = {
        'event_id':     event_id,
        'species':      species,
        'conf':         conf,
        'freq':         freq,
        'lat':          lat,
        'lng':          lng,
        'ts':           ts_ok or datetime.now(timezone.utc).isoformat(),
        'name':         _clean_str(d.get('speciesName'), 64) or species,
        'disease':      _clean_str(d.get('disease'), 128) or '',
        'risk':         risk,
        'asymptomatic': asymp is True or asymp == 1 or
                        (isinstance(asymp, str) and asymp.strip().lower() in ('true', '1')),
        'extra':        {},
    }

    # ── Optional v3 fields: absent/null = not sent; invalid = dropped ─────────
    for key, check in OPTIONAL_FIELDS:
        if d.get(key) is not None:
            val = check(d[key])
            if val is None:
                dropped.append(key)
            else:
                fields['extra'][key] = val

    fields['dropped'] = dropped
    return fields, None

def _opt_number(lo, hi):
    # New fields are numbers on the wire; unlike the legacy ones, strings are not accepted.
    def check(v):
        f = None if isinstance(v, str) else _finite(v)
        return round(f, 1) if f is not None and lo <= f <= hi else None
    return check

def _opt_choice(canonical, max_len):
    def check(v):
        return canonical.get(v.strip().lower()) if isinstance(v, str) and len(v) <= max_len else None
    return check

OPTIONAL_FIELDS = (                                  # (payload key, validator → value | None)
    ('f0',         _opt_number(FREQ_MIN, FREQ_MAX)),             # measured fundamental, Hz
    ('genus',      _opt_choice(GENERA, 24)),
    ('posterior',  _parse_posterior),
    ('range',      _opt_choice({r: r for r in RANGES}, 8)),
    ('snr',        _opt_number(SNR_MIN, SNR_MAX)),               # dB
    ('appVersion', lambda v: v if isinstance(v, str) and APP_VERSION_RE.fullmatch(v) else None),
)

def _storage_unavailable():
    """The durable write failed (disk full, volume gone). 503 + Retry-After so
    the client keeps the find queued and retries — never a false ack."""
    resp = jsonify({'received': False, 'reason': 'storage unavailable'})
    resp.status_code = 503
    resp.headers['Retry-After'] = '30'
    return resp

@app.route('/detection', methods=['POST'])
def detection():
    d = _json_body()
    if not isinstance(d, dict):
        return _reject('body is not a JSON object')

    now = time.time()
    f, why = _parse_detection(d, now)
    if f is None:
        return _reject(why)
    h = dh(d.get('deviceId', 'anon'))
    wait = _throttled('dev:detection', h)
    if wait:
        return _rate_limited(wait)

    species, conf, event_id = f['species'], f['conf'], f['event_id']
    dropped = f['dropped']
    if dropped:
        _count('dropped_fields', len(dropped))
        _log(f'[Det] dropped invalid optional field(s): {",".join(dropped)}', event='dropped_fields')
    rkey = f'{h}:{species}'

    with write_lock:
        touch(h)

        # ── Idempotency: a retried upload carries the same eventId. Acknowledge
        #    it without creating a duplicate row or bumping the counter. ────────
        if event_id is not None and event_id in seen_events:
            snap = dict(full_stats())
            return jsonify({'received': True, 'duplicate': True,
                            'detection_id': seen_events[event_id], **snap})

        # ── Conf update for recent detection? ─────────────────────────────────
        rec = recent_log.get(rkey)
        if rec and (now - rec['ts']) < RECENT_WINDOW:
            improved = conf > rec['conf']
            if improved or event_id is not None:
                # Record the change as a single appended line (replaced a
                # full-file rewrite that cost ~139ms under the global lock). It
                # also carries the eventId, so this retry stays a duplicate even
                # if the process is killed before the next state save.
                op = {'op': 'conf', 'id': rec['det_id'], 'conf': max(conf, rec['conf'])}
                if improved:
                    op['at'] = datetime.now(timezone.utc).isoformat()
                if event_id is not None:
                    op['eid'] = event_id
                if not _append_entry(op):
                    return _storage_unavailable()
                if improved:
                    # Mutate the entry through the reference we kept when it was
                    # created — O(1), no scan of the log at all. (detection_log
                    # holds the same dict object, so this updates what /log serves.)
                    e = rec.get('entry')
                    if isinstance(e, dict):
                        e['conf'] = conf
                        e['conf_updated'] = op['at']
                    rec['conf'] = conf
                if event_id is not None:
                    _remember_event(event_id, rec['det_id'])
                _mark_dirty()
            snap = dict(full_stats())
            body = {'received': True, 'updated': True, 'detection_id': rec['det_id'], **snap}
            if dropped:
                body['dropped'] = dropped
            return jsonify(body)

        # ── New detection ─────────────────────────────────────────────────────
        det_id = next_detection_id[0]
        entry = {
            'id':           det_id,
            'ts':           f['ts'],
            'device':       h,
            'lat':          round(f['lat'], 4) if f['lat'] is not None else None,
            'lng':          round(f['lng'], 4) if f['lng'] is not None else None,
            'species':      species,
            'name':         f['name'],
            'disease':      f['disease'],
            'freq':         f['freq'],
            'conf':         conf,
            'risk':         f['risk'],
            'asymptomatic': f['asymptomatic'],
            **f['extra'],
        }
        if event_id is not None:
            entry['eid'] = event_id      # private: never served by /log

        # Durable record FIRST, under the same lock. Only once it is on disk do
        # we consume the id and touch in-memory state, so a failed write leaves
        # nothing half-applied. The fsync and the state save are batched by the
        # background flusher (was: a full state serialize + 2 fsyncs on EVERY
        # detection, which capped throughput near 100 req/s).
        if not _append_entry(entry):
            return _storage_unavailable()
        next_detection_id[0] += 1
        stats['total_detections'] += 1
        detection_log.append(entry)
        recent_log.pop(rkey, None)       # re-insert keeps recent_log oldest-first
        recent_log[rkey] = {'det_id': det_id, 'ts': now, 'conf': conf, 'entry': entry}
        _trim(recent_log, RECENT_MAX)
        if event_id is not None:
            _remember_event(event_id, det_id)
        _count_cell(entry)               # hotspot
        _mark_dirty()
        snap = dict(full_stats())   # snapshot inside lock — consistent with what was written

    _log(f'[Det #{det_id}] {entry["name"]} conf={conf:.3f} freq={entry["freq"]:.1f}Hz '
          f'located={entry["lat"] is not None}', event='detection', det_id=det_id,
         species=species, conf=round(conf, 3), freq=entry['freq'],
         located=entry['lat'] is not None)
    body = {'received': True, 'detection_id': det_id, **snap}
    if dropped:
        body['dropped'] = dropped
    return jsonify(body)


def _int_arg(name, default, lo, hi):
    try:
        return max(lo, min(int(request.args.get(name, default)), hi))
    except (TypeError, ValueError):
        return default

def _select_rows(from_id, limit, sp_filter=None, dev_filter=None):
    """Newest-first log entries → (rows, stats snapshot). Unfiltered, walk
    newest→oldest and stop at `limit`, so the lock is held for O(limit) rather
    than O(entire log). A selective filter may have to walk the whole window
    (and HMAC every row for `device`), so then the deque's references are copied
    under the lock — a fast C-level copy — and filtered outside it: one picky
    query must not stall every detection behind the global lock."""
    with write_lock:
        snap = dict(full_stats())
        newest = (e for e in reversed(detection_log) if e['id'] > from_id)
        if not (sp_filter or dev_filter):
            return list(itertools.islice(newest, limit)), snap
        pool = list(detection_log)
    rows = (e for e in reversed(pool)
            if e['id'] > from_id
            and (not sp_filter or e.get('species') == sp_filter)
            and (not dev_filter or _public_device(e) == dev_filter))
    return list(itertools.islice(rows, limit)), snap

@app.route('/log', methods=['GET'])
def log_json():
    # Cap at 5000 (was 50000 → a 13 MB unauthenticated response per request).
    rows, s = _select_rows(_int_arg('from_id', 0, 0, 2**62), _int_arg('limit', 500, 1, 5000),
                           request.args.get('species'),
                           request.args.get('device'))   # the public per-day pseudonym
    # Project outside the lock (pure reads of dicts whose keys we only add).
    return jsonify({'total_ever': s['total_detections'], 'total_log': s['log_size'],
                    'returned': len(rows), 'detections': [_public_row(e) for e in rows]})


@app.route('/log.txt', methods=['GET'])
def log_txt():
    rows, s = _select_rows(_int_arg('from_id', 0, 0, 2**62), _int_arg('limit', 2000, 1, 5000),
                           request.args.get('species'))
    total_ever, total_log = s['total_detections'], s['log_size']

    buf = io.StringIO()
    buf.write('━' * 60 + '\n')
    buf.write(f'  MosquitoNet Detection Log\n')
    buf.write(f'  Total ever: {total_ever}  |  In log: {total_log}\n')
    buf.write(f'  {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} UTC\n')
    buf.write('━' * 60 + '\n\n')

    # Every row is formatted defensively: one malformed/legacy record (a string
    # where a float is expected) must not take the whole endpoint down with a 500.
    def _num(v, fmt):
        try:    return format(float(v), fmt)
        except (TypeError, ValueError): return 'n/a'

    def _txt(v, n):
        # Strip control chars so a stored label can't fake extra log rows.
        return _CTRL_RE.sub(' ', str(v))[:n]

    cfmt = f'.{PUBLIC_COORD_DECIMALS}f'
    for e in rows:
      try:
        p       = _public_row(e)
        det_id  = e.get('id', '?')
        device  = p['device'][:8]
        lat     = _num(p['lat'], cfmt)
        lng     = _num(p['lng'], cfmt)
        try:
            dt = datetime.fromisoformat(str(e.get('ts','')).replace('Z','+00:00'))
            ts = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        except Exception:
            ts = _txt(e.get('ts',''), 19)

        sp_name = _txt(e.get('name') or e.get('species', '?'), 64)
        parts   = sp_name.split()
        if len(parts) >= 2:
            sp_name = parts[0][0] + '. ' + ' '.join(parts[1:])

        risk    = _txt(e.get('risk', '?'), 16)
        freq    = _num(e.get('freq'), '.1f') + ' Hz'
        conf    = _num(e.get('conf'), '.2f')
        upd     = '  *(conf updated)' if e.get('conf_updated') else ''

        buf.write(f'#{det_id}  |  device: {device}  |  {lat}, {lng}  |  {ts}\n')
        buf.write(f'  {sp_name}  |  risk: {risk}  |  {freq}  |  conf: {conf}{upd}\n')
        buf.write('\n')
      except Exception:
        continue   # skip an unrenderable row rather than 500 the endpoint

    return Response(buf.getvalue(), mimetype='text/plain; charset=utf-8')


@app.route('/hotspots', methods=['GET'])
def hotspots():
    # Only references are copied under the lock; the answer (up to
    # DETECTION_CELLS_MAX dicts) is built outside it, so a public read can't
    # hold up detections. Safe: _count_cell only ever bumps a cell's `total`
    # in place — it never adds or removes keys, so copying one can't fail.
    with write_lock:
        hot_src  = list(hotspot_cells.items())
        near_src = [(k, v) for k, v in detection_cells.items()
                    if k not in hotspot_cells and v.get('total', 0) >= 50]
    hot  = [{'key':k,**v} for k,v in hot_src]
    near = [{'key':k,**v,'approaching':True} for k,v in near_src]
    return jsonify({'hotspots':hot,'approaching':near,'threshold':HOTSPOT_THRESHOLD})

def _reject_upload(msg, log_reason):
    _count('rejected_uploads')
    _log(f'[Fed REJECT 400] {log_reason}', 'warn', event='rejected_upload')
    return jsonify({'error': msg}), 400

@app.route('/federated/upload', methods=['POST'])
def upload():
    d = _json_body()
    if not isinstance(d, dict) or not all(k in d for k in ['deviceId','weights','steps']):
        return _reject_upload('missing required fields', 'missing deviceId/weights/steps')
    # Reject malformed / non-finite / wrong-shape / absurd-magnitude model
    # uploads up front so they never reach the aggregator.
    if not _valid_weights(d['weights']):
        return _reject_upload('weights must be 4x4 W and length-4 b, all finite',
                              f'weights not 4x4/len-4 finite with |w|<={FED_WEIGHT_ABS_MAX}')
    steps = _finite(d.get('steps', 1))
    if steps is None:          # was int(): Infinity → OverflowError → 500
        return _reject_upload('steps not an integer', 'steps not numeric')
    steps = int(min(max(steps, 1), FED_MAX_STEPS))
    h = dh(d['deviceId'])
    wait = _throttled('dev:upload', h)
    if wait:
        return _rate_limited(wait)
    with write_lock:
        touch(h); stats['total_uploads'] += 1
        # One vote per device per round: a re-upload replaces the device's
        # pending update, so a single install can't fill a round on its own.
        pending_updates.pop(h, None)
        pending_updates[h] = {'steps': steps, 'weights': d['weights']}
        if len(pending_updates) >= MIN_UPLOADS:
            fedavg(list(pending_updates.values())); pending_updates.clear()
        _mark_dirty()   # global model / counters persisted by the background flusher
        snap = dict(full_stats())
        gW, gB = global_W.tolist(), global_b.tolist()
    return jsonify({'status':'accepted','weights':{'W':gW,'b':gB}, **snap})

@app.route('/federated/model',  methods=['GET'])
def model():
    with write_lock:
        return jsonify({'round':stats['total_rounds'],
                        'weights':{'W':global_W.tolist(),'b':global_b.tolist()}})

@app.route('/federated/stats',  methods=['GET'])
def get_stats():
    with write_lock: return jsonify(full_stats())

@app.route('/health', methods=['GET'])
def health():
    return jsonify(_health_body())

def _health_body():
    with write_lock:
        snap = dict(full_stats())
        n_seen = len(seen_events)
        failing = _write_failing
        saved = _last_state_save
    with _counter_lock:
        ctr = dict(counters)
    disk_free_mb = _disk_free_mb()
    return {
        # 'degraded' = the last durable write failed (disk full / volume gone):
        # detections are being refused with 503 and retried by clients. HTTP
        # stays 200 because restarting the process would not fix the disk.
        'status':       'degraded' if failing else 'ok',
        'service':      SERVICE,
        'data_dir':     DATA_DIR,
        'log_file':     LOG_FILE,
        # major.minor only — the full build string fingerprints the patch level.
        'python':       f'{sys.version_info.major}.{sys.version_info.minor}',
        'writable':     os.access(DATA_DIR, os.W_OK) and not failing,
        # true = the data will NOT survive a redeploy; ephemeral_reason says why.
        'ephemeral':    EPHEMERAL_REASON is not None,
        'ephemeral_reason': EPHEMERAL_REASON,
        'single_instance': single_instance,
        'disk_free_mb': disk_free_mb,
        'last_state_save': saved,
        'min_conf':     MIN_CONF,
        'seen_events':  n_seen,
        'log_cap':      DETECTION_LOG_MAX,
        'rate_limit_scale': RATE_LIMIT_SCALE,
        # The address rate limits key on for THIS request — the caller's own IP,
        # so echoing it is harmless. If it shows a proxy/CDN address instead of
        # your public IP, TRUSTED_PROXY_HOPS is wrong and everyone shares a bucket.
        'client_ip':    _client_ip(),
        'counters':     ctr,
        **snap,
    }

# ── Admin (operator) API ──────────────────────────────────────────────────────
# Off unless ADMIN_PASSWORD is set: every /admin/* path is then a plain 404.
# About → Advanced in the app is checked in the browser against a hash that
# ships in the public page, so it can never guard data; this is the real check.
# The right password returns a random bearer token that the app keeps in memory
# only (that one tab: gone on reload) and that expires after ADMIN_SESSION_SEC.
# An admin sees what this server already stores, nothing more: the active
# devices, with the locations they chose to share with their detections (the
# client's own fixed ≤ ~1.5 km offset, never a raw GPS fix; heartbeats carry no
# location at all), the stored detection records, the recent server log and the
# files in DATA_DIR. Logins, failures and logouts are logged.
ADMIN_PASSWORD     = os.environ.get('ADMIN_PASSWORD', '').strip()
ADMIN_SESSION_SEC  = _env_num('ADMIN_SESSION_SEC', 4 * 3600, 300, 86400, int)
ADMIN_SESSIONS_MAX = 16
ADMIN_LOCATIONS_PER_DEVICE = 10
_ADMIN_PW_DIGEST   = hashlib.sha256(ADMIN_PASSWORD.encode()).digest() if ADMIN_PASSWORD else None
_admin_sessions    = OrderedDict()   # sha256(token) → session, oldest first
_admin_lock        = threading.Lock()
# Password guessing is limited per IP and across all IPs together. Deliberately
# NOT scaled by RATE_LIMIT_SCALE: turning rate limiting off must not open the
# login to brute force.
ADMIN_LOGIN_LIMITS = {'ip': (5, 1 / 60), 'all': (30, 1 / 20)}
_admin_limiter     = _RateLimiter(10_000)
if ADMIN_PASSWORD and len(ADMIN_PASSWORD) < 12:
    _log('[Admin] ADMIN_PASSWORD is shorter than 12 characters — use a long random one', 'warn',
         event='admin_weak_password')

def _admin_enabled():
    if _ADMIN_PW_DIGEST is None:
        raise NotFound()

def _admin_session():
    """The caller's live admin session, or None. Expired sessions are dropped."""
    auth = request.headers.get('Authorization', '')
    if not auth.startswith('Bearer '):
        return None
    key = hashlib.sha256(auth[7:].strip().encode('utf-8', 'replace')).hexdigest()
    now = time.time()
    with _admin_lock:
        s = _admin_sessions.get(key)
        if s is not None and s['exp'] <= now:
            _admin_sessions.pop(key, None)
            s = None
        return s

def _admin_only(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        _admin_enabled()
        s = _admin_session()
        if s is None:
            return jsonify({'error': 'admin session required'}), 401
        return view(s, *args, **kwargs)
    return wrapped

@app.route('/admin/login', methods=['POST'])
def admin_login():
    _admin_enabled()
    ip = _client_ip()
    cap, rate = ADMIN_LOGIN_LIMITS['ip']
    wait = _admin_limiter.hit(f'ip|{ip}', cap, rate)
    if not wait:
        cap, rate = ADMIN_LOGIN_LIMITS['all']
        wait = _admin_limiter.hit('all', cap, rate)
    if wait:
        _log(f'[Admin] login throttled from {ip}', 'warn', event='admin_login_throttled')
        return _rate_limited(wait)
    body = _json_body()
    body = body if isinstance(body, dict) else {}
    pw = body.get('password')
    ok = isinstance(pw, str) and len(pw) <= 256 and hmac.compare_digest(
        hashlib.sha256(pw.encode('utf-8', 'replace')).digest(), _ADMIN_PW_DIGEST)
    if not ok:
        _count('admin_login_failures')
        _log(f'[Admin] login FAILED from {ip}', 'warn', event='admin_login_failed')
        return jsonify({'error': 'wrong password'}), 401
    token = base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip('=')
    now = time.time()
    device = dh(body.get('deviceId', 'anon'))
    with _admin_lock:
        for k in [k for k, v in _admin_sessions.items() if v['exp'] <= now]:
            _admin_sessions.pop(k, None)
        _admin_sessions[hashlib.sha256(token.encode()).hexdigest()] = {
            'created': now, 'exp': now + ADMIN_SESSION_SEC, 'device': device, 'ip': ip}
        while len(_admin_sessions) > ADMIN_SESSIONS_MAX:
            _admin_sessions.popitem(last=False)
    _count('admin_logins')
    # The IP is logged on purpose (who logged in as admin); no device id beside it.
    _log(f'[Admin] login from {ip}', 'warn', event='admin_login')
    return jsonify({'token': token, 'expires_in': ADMIN_SESSION_SEC, 'device': device})

@app.route('/admin/logout', methods=['POST'])
@_admin_only
def admin_logout(s):
    with _admin_lock:
        for k in [k for k, v in _admin_sessions.items() if v is s]:
            _admin_sessions.pop(k, None)
    _log(f'[Admin] logout device {s["device"][:8]}', event='admin_logout', device=s['device'][:8])
    return jsonify({'ok': True})

def _session_start_ms(v):
    """session_registry holds the client's session start (ms since epoch) as a string."""
    try:
        ms = int(v)
    except (TypeError, ValueError):
        return None
    return ms if 1_000_000_000_000 <= ms <= 100_000_000_000_000 else None

def _admin_location(e):
    return {'id': e.get('id'), 'ts': e.get('ts'), 'lat': e.get('lat'), 'lng': e.get('lng'),
            'species': e.get('species'), 'name': e.get('name'), 'conf': e.get('conf')}

@app.route('/admin/devices', methods=['GET'])
@_admin_only
def admin_devices(_s):
    """Devices active within ACTIVE_SEC, most recently seen first, each with the
    detections it shared in the live window and their locations."""
    limit = _int_arg('limit', 500, 1, 5000)
    now = time.time()
    with write_lock:
        active = list(device_registry.items())          # oldest-seen first
        sess = {h: session_registry.get(h) for h, _ in active}
        lis = {h: device_listening.get(h) for h, _ in active}
        pool = list(detection_log)
    shared = {h: [] for h, _ in active}
    for e in reversed(pool):                            # newest first
        rows = shared.get(e.get('device'))
        if rows is not None:
            rows.append(e)
    devices, n_listening, n_located = [], 0, 0
    for h, seen in reversed(active):
        rows = shared[h]
        located = [e for e in rows if _finite(e.get('lat')) is not None and _finite(e.get('lng')) is not None]
        n_listening += lis[h] is True
        n_located += bool(located)
        if len(devices) >= limit:
            continue
        devices.append({
            'device':        h,
            'last_seen_s':   round(max(0.0, now - seen), 1),
            'listening':     lis[h],                    # None: the client did not say
            'session_start': _session_start_ms(sess[h]),
            'detections':    len(rows),
            'last_detection': _admin_location(rows[0]) if rows else None,
            'locations':     [_admin_location(e) for e in located[:ADMIN_LOCATIONS_PER_DEVICE]],
            'app_version':   next((e['appVersion'] for e in rows if e.get('appVersion')), None),
        })
    return jsonify({'active': len(active), 'listening': n_listening, 'with_location': n_located,
                    'active_window_s': ACTIVE_SEC, 'returned': len(devices), 'devices': devices,
                    'location_note': 'Locations are only those shared with detections, as stored: '
                                     "offset on the device by its fixed secret offset (up to ~1.5 km)."})

@app.route('/admin/detections', methods=['GET'])
@_admin_only
def admin_detections(_s):
    """Stored detection records, newest first, every stored field — the private
    hash and the device-offset coordinates, not the public projections."""
    from_id = _int_arg('from_id', 0, 0, 2**62)
    limit = _int_arg('limit', 200, 1, 2000)
    dev = (request.args.get('device') or '').strip().lower()[:16]
    with write_lock:
        snap = dict(full_stats())
        rows = (e for e in reversed(detection_log)
                if e['id'] > from_id and (not dev or str(e.get('device', '')).startswith(dev)))
        rows = [dict(e) for e in itertools.islice(rows, limit)]
    return jsonify({'total_ever': snap['total_detections'], 'total_log': snap['log_size'],
                    'log_cap': DETECTION_LOG_MAX, 'returned': len(rows), 'detections': rows})

@app.route('/admin/logs', methods=['GET'])
@_admin_only
def admin_logs(_s):
    """The most recent server log records (this process only), newest first."""
    limit = _int_arg('limit', 300, 1, ADMIN_LOG_LINES)
    level = request.args.get('level', 'debug')
    floor = _LOG_LEVELS.index(level) if level in _LOG_LEVELS else 0
    recs = [r for r in list(_recent_logs) if _LOG_LEVELS.index(r['level']) >= floor]
    recs.reverse()
    return jsonify({'capacity': ADMIN_LOG_LINES, 'returned': min(limit, len(recs)), 'logs': recs[:limit]})

def _data_dir_files(cap=200):
    """Names, sizes and times of what is in DATA_DIR (never file contents)."""
    out = []
    try:
        with os.scandir(DATA_DIR) as it:
            for ent in itertools.islice(it, cap):
                try:
                    st = ent.stat(follow_symlinks=False)
                except OSError:
                    continue
                out.append({'name': ent.name, 'dir': ent.is_dir(follow_symlinks=False),
                            'size': st.st_size,
                            'mtime': datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()})
    except OSError as e:
        return [{'error': f'{type(e).__name__}: {e}'}]
    return sorted(out, key=lambda f: f['name'])

@app.route('/admin/server', methods=['GET'])
@_admin_only
def admin_server(s):
    now = time.time()
    with _admin_lock:
        n_sessions = sum(1 for v in _admin_sessions.values() if v['exp'] > now)
    return jsonify({
        'health': _health_body(),
        'files':  _data_dir_files(),
        'config': {
            'service': SERVICE, 'commit': (os.environ.get('RAILWAY_GIT_COMMIT_SHA') or '')[:12] or None,
            'data_dir': DATA_DIR, 'log_format': LOG_FORMAT, 'detection_log_max': DETECTION_LOG_MAX,
            'seen_events_max': SEEN_EVENTS_MAX, 'active_sec': ACTIVE_SEC,
            'rate_limit_scale': RATE_LIMIT_SCALE, 'trusted_proxy_hops': TRUSTED_PROXY_HOPS,
            'public_geo_offset': PUBLIC_GEO_OFFSET, 'public_coord_decimals': PUBLIC_COORD_DECIMALS,
            'extra_origins': sorted(_EXTRA_ORIGINS), 'site_files': len(_SITE),
        },
        'admin': {'sessions': n_sessions, 'session_sec': ADMIN_SESSION_SEC,
                  'expires_in': round(s['exp'] - now), 'device': s['device']},
    })


@app.route('/', methods=['GET'])
def index():
    if _SITE:
        return _site_response('index.html')
    return jsonify({'service':SERVICE,'data_dir':DATA_DIR,
                    'endpoints':['GET /heartbeat','POST /detection',
                                 'GET /log','GET /log.txt','GET /hotspots',
                                 'POST /federated/upload','GET /federated/model',
                                 'GET /federated/stats','GET /health']})

# ── The app itself ────────────────────────────────────────────────────────────
# The Pages deploy also builds the minified PWA into ./site on the `release`
# branch, which is what Railway deploys: then this server is a full copy of the
# app at / (https://mosquito.charity), on the same origin as the API. Without a
# built site (main branch, local runs, most tests) nothing below is registered
# and / stays the JSON service description. Files are read once at boot and
# served from memory (about 1 MB), so no filesystem path is ever built from a
# request.
SITE_DIR = os.environ.get('SITE_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'site')
_SITE_TYPES = {
    '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8', '.json': 'application/json',
    '.webmanifest': 'application/manifest+json', '.svg': 'image/svg+xml',
    '.png': 'image/png', '.ico': 'image/x-icon', '.woff2': 'font/woff2', '.woff': 'font/woff',
    '.txt': 'text/plain; charset=utf-8', '.xml': 'application/xml',
}
_SITE_GZIP = ('.html', '.js', '.css', '.json', '.webmanifest', '.svg', '.txt', '.xml')
# Re-validated on every load (an ETag round trip): the page and the service
# worker must pick up a new release at once. Everything else may be cached a day;
# app code is fetched with ?v=<commit>, so a release never pairs with stale code.
_SITE_NO_CACHE = ('index.html', 'sw.js', 'manifest.webmanifest', 'map-data.json')

def _load_site(root):
    if not os.path.isfile(os.path.join(root, 'index.html')):
        return {}
    files = {}
    real_root = os.path.realpath(root)
    for dirpath, dirnames, filenames in os.walk(root):      # never descends symlinked dirs
        dirnames[:] = sorted(d for d in dirnames if not d.startswith('.'))
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if name.startswith('.') or ext not in _SITE_TYPES:
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, '/')
            # Regular files that really live under the site, nothing else: a
            # symlink in a build (a merged PR, a compromised minifier) would
            # publish its target — /proc/self/environ holds ADMIN_PASSWORD.
            if (os.path.islink(full) or not os.path.isfile(full)
                    or os.path.commonpath([real_root, os.path.realpath(full)]) != real_root):
                _log(f'[Site] skipped {_q(rel)}: not a regular file inside SITE_DIR', 'warn',
                     event='site_file_skipped')
                continue
            with open(full, 'rb') as fh:
                body = fh.read()
            gz = gzip.compress(body, 9, mtime=0) if ext in _SITE_GZIP else None
            files[rel] = {
                'body': body, 'gz': gz if gz is not None and len(gz) < len(body) else None,
                'type': _SITE_TYPES[ext], 'etag': hashlib.sha256(body).hexdigest()[:32],
                'cache': 'no-cache' if rel in _SITE_NO_CACHE else 'public, max-age=86400',
            }
    return files if 'index.html' in files else {}

def _site_csp(index_html):
    """The page's Content-Security-Policy. Inline scripts run only if their exact
    text is listed by hash, so markup injected into the page can never run code;
    only the OpenStreetMap tile servers are allowed besides this origin (and
    HTTPS endpoints for the API failover list)."""
    hashes = []
    for attrs, code in re.findall(r'<script\b([^>]*)>(.*?)</script>', index_html, re.S | re.I):
        if re.search(r'\bsrc\s*=', attrs, re.I):
            continue
        t = re.search(r'\btype\s*=\s*["\']?([^"\'\s>]+)', attrs, re.I)
        if t and t.group(1).lower() not in ('text/javascript', 'application/javascript', 'module'):
            continue                                   # data blocks (JSON-LD) never execute
        digest = base64.b64encode(hashlib.sha256(code.encode('utf-8')).digest()).decode()
        hashes.append(f"'sha256-{digest}'")
    return '; '.join([
        "default-src 'self'",
        "script-src 'self' " + ' '.join(hashes),
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: blob: https://*.tile.openstreetmap.org",
        "font-src 'self'",
        "connect-src 'self' https:",
        "manifest-src 'self' blob:",
        "worker-src 'self'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
    ])

def _site_response(rel):
    f = _SITE.get(rel)
    if f is None:
        raise NotFound()
    gz = f['gz'] is not None and 'gzip' in request.headers.get('Accept-Encoding', '')
    r = Response(f['gz'] if gz else f['body'], content_type=f['type'])
    if f['gz'] is not None:
        r.headers['Vary'] = 'Accept-Encoding'
    if gz:
        r.headers['Content-Encoding'] = 'gzip'
    r.set_etag(f['etag'] + ('-gz' if gz else ''))
    r.headers['Cache-Control'] = f['cache']
    r.headers['Content-Security-Policy'] = _SITE_CSP
    # Not the API's no-referrer: the OpenStreetMap tile servers refuse requests
    # that carry no Referer at all. Cross-origin, only the origin is sent.
    r.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return r.make_conditional(request)

_SITE = _load_site(SITE_DIR)
_SITE_CSP = _site_csp(_SITE['index.html']['body'].decode('utf-8')) if _SITE else ''
if _SITE:
    @app.route('/<path:p>', methods=['GET'])
    def site_file(p):
        return _site_response(p)

# ── Storage check ─────────────────────────────────────────────────────────────
# The old check only flagged /tmp, so a Railway service with NO volume attached
# reported "ephemeral": false while every redeploy silently wiped its data.
def _in_container():
    return os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')

def _on_mounted_volume(path):
    """True if `path` lives on a mount other than the root filesystem."""
    p = os.path.realpath(path)
    while True:
        if os.path.ismount(p):
            return p != '/'
        parent = os.path.dirname(p)
        if parent == p:
            return False
        p = parent

def _ephemeral_reason(path):
    """Why data in `path` would NOT survive a restart/redeploy, or None when it
    is (as far as can be told) on persistent storage."""
    path = os.path.abspath(path)
    if path == '/tmp' or path.startswith('/tmp/'):
        return 'DATA_DIR is under /tmp'
    if _on_railway():
        vol = os.environ.get('RAILWAY_VOLUME_MOUNT_PATH', '').strip()
        if not vol:
            return 'no Railway volume is attached to this service'
        vol = os.path.realpath(vol)
        real = os.path.realpath(path)
        if real != vol and not real.startswith(vol.rstrip('/') + '/'):
            return f'DATA_DIR is not on the Railway volume (mounted at {vol})'
        return None
    if _in_container() and not _on_mounted_volume(path):
        return 'DATA_DIR is on the container filesystem (no volume mounted)'
    return None

EPHEMERAL_REASON = _ephemeral_reason(DATA_DIR)


_acquire_single_writer_lock()
load_all()
# Started in the process that serves requests. (Under gunicorn do NOT use
# --preload: the thread would live in the master and die at fork.)
threading.Thread(target=_flusher, daemon=True, name='flusher').start()

def _log_boot():
    """One line with everything needed to tell from the logs alone which build
    is running, as whom, and whether its storage is safe."""
    global _stats_prev
    uid = os.getuid() if hasattr(os, 'getuid') else None
    commit = (os.environ.get('RAILWAY_GIT_COMMIT_SHA') or '')[:12] or None
    with write_lock:
        total, kept = stats['total_detections'], len(detection_log)
    with _counter_lock:
        _stats_prev = {**counters, 'detections': total}
    _log(f'[Boot] {SERVICE} pid={os.getpid()} uid={uid} commit={commit or "-"} '
         f'data_dir={DATA_DIR} storage={"EPHEMERAL" if EPHEMERAL_REASON else "persistent"} '
         f'writable={os.access(DATA_DIR, os.W_OK)} disk_free_mb={_disk_free_mb()} '
         f'detections={total} in_ram={kept} single_instance={single_instance} '
         f'log_format={LOG_FORMAT} site_files={len(_SITE)}',
         event='boot', service=SERVICE, pid=os.getpid(), uid=uid, commit=commit,
         deployment=os.environ.get('RAILWAY_DEPLOYMENT_ID'), data_dir=DATA_DIR,
         ephemeral=EPHEMERAL_REASON is not None, total_detections=total,
         rate_limit_scale=RATE_LIMIT_SCALE, trusted_proxy_hops=TRUSTED_PROXY_HOPS,
         extra_origins=len(_EXTRA_ORIGINS), site_files=len(_SITE))
    if EPHEMERAL_REASON:
        _log(f'[Storage] EPHEMERAL: {EPHEMERAL_REASON} (DATA_DIR={DATA_DIR}).\n'
             '          Every restart/redeploy will PERMANENTLY DELETE all detections,\n'
             '          all statistics and the trained federated model.\n'
             '          Mount a persistent volume at DATA_DIR before going live.',
             'error', event='ephemeral_storage', reason=EPHEMERAL_REASON)
    if uid == 0 and (_in_container() or _on_railway()):
        _log('[Boot] running as root: the platform start command bypassed the image '
             'entrypoint (which drops to uid 10001). Use '
             '"/usr/local/bin/docker-entrypoint gunicorn federated_server:app" '
             'or no custom start command.', 'warn', event='running_as_root')

_log_boot()

if __name__ == '__main__':
    import signal as _signal
    for _s in (_signal.SIGTERM, _signal.SIGINT):
        try: _signal.signal(_s, _graceful_exit)
        except Exception: pass
    port = int(os.environ.get('PORT', 5001))
    _log(f'[Boot] standalone development server on 0.0.0.0:{port} '
         f'(production runs under gunicorn: see gunicorn.conf.py)')
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
