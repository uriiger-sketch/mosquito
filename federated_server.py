#!/usr/bin/env python3
"""
MosquitoNet Federated Server v12
================================
Log format (human-readable at /log.txt):

  #42  |  device: a1b2c3d4  |  32.0821, 34.7913  |  2025-01-15 14:32:07 UTC
  Ae. albopictus  |  RISK: HIGH  |  668.0 Hz  |  conf: 0.83

Fixes vs v11 (production hardening for real traffic):
  - The JSONL is strictly APPEND-ONLY. A confidence update appends a tiny
    {"op":"conf"} record (replayed at load) instead of rewriting the whole
    file, which cost ~139ms under the global lock at 20k rows.
  - detection_log is a bounded deque and load_all() keeps only the tail, so a
    large history can never OOM the process at boot (previously unrecoverable:
    it died during startup before it could serve anything).
  - fsync + state saves are batched by a background flusher every few seconds
    instead of running on every single detection (~250KB + 2 fsyncs per event).
  - active_now() is O(1); idle devices and stale recent_log entries are evicted.
  - Read endpoints bound their work; request bodies are size-capped.
"""

import os, time, hashlib, threading, json, io, atexit
from collections import deque
import numpy as np
from datetime import datetime, timezone
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
# Reject oversized bodies before they are parsed (a huge /federated/upload
# payload could otherwise allocate hundreds of MB and OOM the worker).
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024   # 256 KB is ample for any endpoint
CORS(app, origins='*', supports_credentials=False,
     allow_headers=['Content-Type', 'Accept'],
     methods=['GET', 'POST', 'OPTIONS'])

# ── Persistence ───────────────────────────────────────────────────────────────
DATA_DIR   = os.environ.get('DATA_DIR', '/tmp')
STATE_FILE = os.path.join(DATA_DIR, 'mosquitonet_state.json')
LOG_FILE   = os.path.join(DATA_DIR, 'detections.jsonl')

# How many recent detections to keep in RAM. The JSONL on disk keeps the full
# durable history; this only bounds the live window used by /log and /log.txt.
# Without this the process OOMs at ~300k detections AND — because load_all()
# re-read the whole file at boot — could never start again (permanent outage).
DETECTION_LOG_MAX = int(os.environ.get('DETECTION_LOG_MAX', '20000'))
FLUSH_INTERVAL    = 5.0     # seconds between background fsync + state saves

# Single lock for BOTH in-memory state AND all disk writes — no races possible
write_lock = threading.Lock()

_log_fh     = None      # persistent append handle (avoids open/close per write)
_dirty      = False     # state changed since last save
_log_unsynced = False   # bytes written but not yet fsynced

def _mark_dirty():
    """Flag state for the background flusher instead of paying a full
    serialize+fsync inside the request (that cost ~250 KB and 2 fsyncs per
    detection, capping throughput at ~100 req/s)."""
    global _dirty
    _dirty = True

def _log_handle():
    global _log_fh
    if _log_fh is None:
        _log_fh = open(LOG_FILE, 'a', encoding='utf-8')
    return _log_fh

def _append_entry(entry):
    """Append one JSONL line. Called under write_lock.
    Buffered + flushed, but fsync is batched by the background flusher — so a
    hard kill can lose at most FLUSH_INTERVAL seconds of detections rather than
    costing an fsync on every single request."""
    global _log_unsynced
    try:
        f = _log_handle()
        f.write(json.dumps(entry, separators=(',', ':')) + '\n')
        f.flush()
        _log_unsynced = True
    except Exception as e:
        print(f'[append] {e}')

def _sync_log():
    """fsync the append log. Called by the background flusher / on exit."""
    global _log_unsynced
    if not _log_unsynced:
        return
    try:
        if _log_fh is not None:
            os.fsync(_log_fh.fileno())
        _log_unsynced = False
    except Exception as e:
        print(f'[sync] {e}')

def _save_state():
    """Write state file. Called under write_lock."""
    try:
        payload = {
            'stats':           stats,
            'next_id':         next_detection_id[0],
            'detection_cells': detection_cells,
            'hotspot_cells':   hotspot_cells,
            'seen_events':     seen_events,
            'global_W':        global_W.tolist(),
            'global_b':        global_b.tolist(),
        }
        tmp = STATE_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(payload, f, separators=(',', ':'))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        print(f'[state] {e}')

def load_all():
    global detection_log, detection_cells, hotspot_cells, stats, seen_events, global_W, global_b
    try:
        with open(STATE_FILE) as f:
            d = json.load(f)
        for k in stats:
            if k in d.get('stats', {}):
                stats[k] = d['stats'][k]
        next_detection_id[0] = d.get('next_id', 1)
        detection_cells = d.get('detection_cells', {})
        hotspot_cells   = d.get('hotspot_cells',   {})
        seen_events     = d.get('seen_events',     {})
        # Restore the trained global model — but only if it is well-shaped and
        # finite; otherwise keep the seed rather than adopt a corrupt blob.
        if 'global_W' in d and 'global_b' in d:
            try:
                W = np.array(d['global_W'], dtype=float)
                b = np.array(d['global_b'], dtype=float)
                if W.shape == global_W.shape and b.shape == global_b.shape \
                   and np.all(np.isfinite(W)) and np.all(np.isfinite(b)):
                    global_W, global_b = W, b
                else:
                    print('[State] persisted global model invalid — keeping seed')
            except Exception as e:
                print(f'[State] global model restore failed: {e} — keeping seed')
        print(f'[State] loaded, total={stats["total_detections"]}, seen_events={len(seen_events)}, '
              f'rounds={stats["total_rounds"]}')
    except FileNotFoundError:
        print('[State] fresh start')
    except Exception as e:
        print(f'[State load] {e}')
    # Load the detection log. `detection_log` is a bounded deque, so a huge file
    # can never blow up RAM at boot — we stream the file and keep only the tail.
    # Lines tagged {"op":"conf"} are confidence updates, replayed onto the entry
    # they refer to (this replaces the old full-file rewrite, which cost ~139ms
    # under the global lock at 20k rows).
    try:
        conf_updates = {}
        n_read = 0
        with open(LOG_FILE, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                if rec.get('op') == 'conf':
                    conf_updates[rec.get('id')] = rec.get('conf')
                else:
                    n_read += 1
                    detection_log.append(rec)     # deque(maxlen) evicts the head
        for e in detection_log:
            if e.get('id') in conf_updates:
                c = conf_updates[e['id']]
                if isinstance(c, (int, float)):
                    e['conf'] = c
        # Self-heal counters from the durable log. State is saved by a background
        # flusher, so a hard kill (SIGKILL/OOM) can leave state.json slightly
        # behind the append-only log. Detection ids are sequential, so the log
        # tail tells us where to resume — this prevents duplicate detection ids
        # and stops the public "total detections" counter going backwards.
        max_id = 0
        for e in detection_log:
            try:    max_id = max(max_id, int(e.get('id', 0)))
            except (TypeError, ValueError): pass
        if max_id:
            if next_detection_id[0] <= max_id:
                print(f'[State] next_id {next_detection_id[0]} behind log (max {max_id}) — recovering')
                next_detection_id[0] = max_id + 1
            if stats['total_detections'] < max_id:
                print(f'[State] total_detections {stats["total_detections"]} behind log — recovering to {max_id}')
                stats['total_detections'] = max_id
        print(f'[Log] {n_read} entries on disk; kept most recent {len(detection_log)} '
              f'(cap {DETECTION_LOG_MAX}), {len(conf_updates)} conf-updates replayed')
    except FileNotFoundError:
        print(f'[Log] starting fresh at {LOG_FILE}')
    except Exception as e:
        print(f'[Log load] {e}')

@atexit.register
def _flush_on_exit():
    print('[Exit] flushing...')
    with write_lock:
        _sync_log()      # append-only log: just fsync, never rewrite the whole file
        _save_state()
    print('[Exit] done')

def _graceful_exit(signum, _frame):
    """Plain `python federated_server.py` dies on SIGTERM WITHOUT running atexit,
    which would drop the batched state (counters, seen_events, trained model).
    Flush explicitly, then exit. Under gunicorn the worker raises SystemExit and
    atexit runs, so this is only installed for the standalone path."""
    print(f'[Signal {signum}] flushing before exit')
    try:
        _flush_on_exit()
    finally:
        os._exit(0)

# ── Model ─────────────────────────────────────────────────────────────────────
global_W = np.array([[ 2.8,0.9,1.2,0.8],[-0.6,0.7,1.0,0.7],
                     [-1.1,0.7,1.0,0.6],[ 1.8,0.8,1.3,0.9]], dtype=float)
global_b = np.array([-0.4,-0.3,-0.3,-0.4], dtype=float)

# ── State ─────────────────────────────────────────────────────────────────────
device_registry   = {}     # ACTIVE devices only (swept by _flusher); hash → last_seen
known_devices     = set()  # hashes seen this process lifetime (ever-count dedupe)
session_registry  = {}
pending_updates   = []
detection_log     = deque(maxlen=DETECTION_LOG_MAX)   # bounded live window
detection_cells   = {}
hotspot_cells     = {}
next_detection_id = [1]
recent_log        = {}     # rkey → {det_id, ts, conf}  (swept by _flusher)
RECENT_WINDOW     = 60     # seconds: conf update allowed (exactly 1 minute)
seen_events       = {}     # clientEventId → det_id (idempotency: retries never double-count)
SEEN_EVENTS_MAX   = 5000   # cap the idempotency map; evict oldest beyond this

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

# ── Helpers ───────────────────────────────────────────────────────────────────
def dh(raw):
    return hashlib.sha256(str(raw).encode()).hexdigest()[:16]

def active_now():
    """O(1). `device_registry` holds only devices seen within ACTIVE_SEC — the
    background flusher evicts the rest — so its length IS the active count.
    (Previously this scanned a never-evicted dict on every heartbeat, which made
    total work O(devices^2) at 10 heartbeats/minute/device.)"""
    return len(device_registry)

def touch(h):
    device_registry[h] = time.time()
    if h not in known_devices:
        known_devices.add(h)
        stats['ever_devices'] = stats.get('ever_devices', 0) + 1
        _mark_dirty()

def cell_key(lat, lng, sp):
    return f'{round(float(lat)/0.05)*0.05:.3f},{round(float(lng)/0.05)*0.05:.3f},{sp}'

def _remember_event(event_id, det_id):
    """Record an eventId→det_id mapping for idempotency. Called under write_lock.
    Evicts the oldest entries (dicts preserve insertion order) once the cap is hit."""
    seen_events[event_id] = det_id
    while len(seen_events) > SEEN_EVENTS_MAX:
        seen_events.pop(next(iter(seen_events)), None)   # O(1), no list() copy

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
    """True iff w = {'W': 4x4, 'b': len-4}, all finite. Rejects the poison payloads
    that would otherwise silently corrupt the global model.
    Shape is validated on the RAW lists BEFORE np.array() — otherwise a crafted
    100M-element payload would allocate hundreds of MB and OOM the worker before
    the shape check could reject it."""
    if not isinstance(w, dict):
        return False
    W_raw, b_raw = w.get('W'), w.get('b')
    nrow, ncol = global_W.shape
    if not (isinstance(W_raw, list) and len(W_raw) == nrow):
        return False
    for row in W_raw:
        if not (isinstance(row, list) and len(row) == ncol):
            return False
        for v in row:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return False
    if not (isinstance(b_raw, list) and len(b_raw) == global_b.shape[0]):
        return False
    for v in b_raw:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False
    try:
        W = np.array(W_raw, dtype=float)
        b = np.array(b_raw, dtype=float)
    except Exception:
        return False
    return bool(np.all(np.isfinite(W)) and np.all(np.isfinite(b)))

def _flusher():
    """Background maintenance so the request path stays O(1):
      • fsync the append log and save state at most once per FLUSH_INTERVAL
        (previously every detection paid a full state serialize + 2 fsyncs)
      • evict devices idle beyond ACTIVE_SEC, keeping active_now() O(1)
      • evict recent_log entries past RECENT_WINDOW (they were retained forever)"""
    global _dirty
    while True:
        time.sleep(FLUSH_INTERVAL)
        try:
            now = time.time()
            with write_lock:
                for h in [h for h, t in device_registry.items() if now - t > ACTIVE_SEC]:
                    device_registry.pop(h, None)
                    session_registry.pop(h, None)
                for k in [k for k, v in recent_log.items() if now - v['ts'] > RECENT_WINDOW]:
                    recent_log.pop(k, None)
                _sync_log()
                if _dirty:
                    _save_state()
                    _dirty = False
        except Exception as e:
            print(f'[flusher] {e}')

threading.Thread(target=_flusher, daemon=True, name='flusher').start()

def fedavg(updates):
    global global_W, global_b
    # Keep only structurally valid contributions; an invalid upload must NOT count
    # its steps toward the weighted average (the old code let it bias the result).
    valid = [u for u in updates if _valid_weights(u['weights'])]
    total = sum(u['steps'] for u in valid)
    if not total:
        return
    nW = np.zeros_like(global_W); nb = np.zeros_like(global_b)
    for u in valid:
        w = u['steps'] / total
        nW += w * np.array(u['weights']['W'], dtype=float)
        nb += w * np.array(u['weights']['b'], dtype=float)
    # Federated averaging on client MODEL WEIGHTS: blend the step-weighted client
    # mean into the global model. Conservative 0.3 global / 0.7 new keeps a single
    # device from dominating a round while still converging.
    cand_W = 0.3 * global_W + 0.7 * nW
    cand_b = 0.3 * global_b + 0.7 * nb
    if not (np.all(np.isfinite(cand_W)) and np.all(np.isfinite(cand_b))):
        print('[FedAvg] non-finite result — keeping previous global model')
        return
    global_W, global_b = cand_W, cand_b
    stats['total_rounds'] += 1
    stats['last_aggregate'] = datetime.now(timezone.utc).isoformat()

# ── CORS ──────────────────────────────────────────────────────────────────────
@app.after_request
def cors_hdr(r):
    r.headers['Access-Control-Allow-Origin']  = '*'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type,Accept'
    r.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return r

@app.route('/', defaults={'p':''}, methods=['OPTIONS'])
@app.route('/<path:p>', methods=['OPTIONS'])
def opts(p): return '', 204

# ══════════════════════════════════════════════════════════════════════════════

@app.route('/heartbeat', methods=['GET','POST'])
def heartbeat():
    if request.method == 'GET':
        raw_id = request.args.get('deviceId', 'anon')
        sess   = request.args.get('sess', '')
    else:
        # A non-dict JSON body (e.g. `[1,2]`) is truthy, so `or {}` would not fire
        # and .get() raised AttributeError → unhandled 500. Guard on the type.
        body   = request.get_json(force=True, silent=True)
        body   = body if isinstance(body, dict) else {}
        raw_id = body.get('deviceId', 'anon')
        sess   = ''
    h = dh(raw_id)
    with write_lock:
        new_s = sess and session_registry.get(h) != sess
        touch(h)
        if new_s:
            session_registry[h] = sess
            stats['total_sessions'] += 1
            _mark_dirty()
        snap = dict(full_stats())   # snapshot under the lock (reading device_registry
                                    # unlocked could raise "dict changed size")
    return jsonify(snap)


def _reject(reason, code=400):
    """Reject a malformed / invalid detection with a logged 4xx (never a 500)."""
    print(f'[Det REJECT {code}] {reason}')
    return jsonify({'received': False, 'reason': reason}), code

@app.route('/detection', methods=['POST'])
def detection():
    d       = request.get_json(force=True, silent=True)
    if not isinstance(d, dict):
        return _reject('body is not a JSON object')

    raw_id   = d.get('deviceId', 'anon')
    species  = str(d.get('species', ''))
    event_id = d.get('eventId')

    # ── Numeric parsing — guarded so bad input is a clean 400, not an unhandled 500 ──
    try:
        conf = round(float(d.get('confidence', 0)), 3)
        freq = round(float(d.get('frequency', 0)), 1)
    except (TypeError, ValueError):
        return _reject('confidence/frequency not numeric')

    lat = d.get('lat')
    lng = d.get('lng')
    try:
        lat = float(lat) if lat is not None else None
        lng = float(lng) if lng is not None else None
    except (TypeError, ValueError):
        return _reject('lat/lng not numeric')

    # ── Schema / plausibility validation ──────────────────────────────────────
    if species not in KNOWN_SPECIES:
        return _reject(f'unknown species: {species!r}')
    if not (0.0 <= conf <= 1.0):
        return _reject(f'confidence out of range: {conf}')
    if conf < MIN_CONF:
        return _reject(f'confidence below floor: {conf} < {MIN_CONF}')
    if not (FREQ_MIN <= freq <= FREQ_MAX):
        return _reject(f'frequency out of range: {freq}')
    if lat is not None and not (-90.0 <= lat <= 90.0):
        return _reject(f'latitude out of range: {lat}')
    if lng is not None and not (-180.0 <= lng <= 180.0):
        return _reject(f'longitude out of range: {lng}')

    risk    = d.get('risk', 'UNKNOWN')
    sp_name = d.get('speciesName', species)
    disease = d.get('disease', '')
    asymp   = bool(d.get('asymptomatic', False))
    ts_str  = d.get('ts') or datetime.now(timezone.utc).isoformat()
    h       = dh(raw_id)
    rkey    = f'{h}:{species}'
    now     = time.time()

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
            if conf > rec['conf']:
                # Mutate the entry through the reference we kept when it was
                # created — O(1), no scan of the log at all. (detection_log holds
                # the same dict object, so this updates what /log serves.)
                e = rec.get('entry')
                if isinstance(e, dict):
                    e['conf'] = conf
                    e['conf_updated'] = datetime.now(timezone.utc).isoformat()
                rec['conf'] = conf
                # Record the change as a single appended line; replaced a
                # full-file rewrite that cost ~139ms under the global lock.
                _append_entry({'op': 'conf', 'id': rec['det_id'], 'conf': conf})
                _mark_dirty()
            if event_id is not None:
                _remember_event(event_id, rec['det_id'])
                _mark_dirty()
            snap = dict(full_stats())
            return jsonify({'received': True, 'updated': True,
                            'detection_id': rec['det_id'], **snap})

        # ── New detection ─────────────────────────────────────────────────────
        det_id = next_detection_id[0]
        next_detection_id[0] += 1
        stats['total_detections'] += 1

        entry = {
            'id':           det_id,
            'ts':           ts_str,
            'device':       h,
            'lat':          round(float(lat), 4) if lat is not None else None,
            'lng':          round(float(lng), 4) if lng is not None else None,
            'species':      species,
            'name':         sp_name,
            'disease':      disease,
            'freq':         freq,
            'conf':         conf,
            'risk':         risk,
            'asymptomatic': asymp,
        }
        detection_log.append(entry)
        recent_log[rkey] = {'det_id': det_id, 'ts': now, 'conf': conf, 'entry': entry}
        if event_id is not None:
            _remember_event(event_id, det_id)

        # Hotspot
        if entry['lat'] is not None:
            try:
                ck = cell_key(lat, lng, species)
                if ck not in detection_cells:
                    detection_cells[ck] = {
                        'species': species, 'total': 0, 'risk': risk,
                        'lat': round(float(lat)/0.05)*0.05,
                        'lng': round(float(lng)/0.05)*0.05,
                    }
                detection_cells[ck]['total'] += 1
                if detection_cells[ck]['total'] >= HOTSPOT_THRESHOLD:
                    hotspot_cells[ck] = dict(detection_cells[ck])
            except Exception as e:
                print(f'[Cell] {e}')

        # Append the durable record under the same lock. The fsync and the state
        # save are batched by the background flusher (was: a full state serialize
        # + 2 fsyncs on EVERY detection, which capped throughput near 100 req/s).
        _append_entry(entry)
        _mark_dirty()
        snap = dict(full_stats())   # snapshot inside lock — consistent with what was written

    print(f'[Det #{det_id}] {sp_name} conf={conf:.3f} freq={freq:.1f}Hz '
          f'lat={entry["lat"]} lng={entry["lng"]}')
    return jsonify({'received': True, 'detection_id': det_id, **snap})


@app.route('/log', methods=['GET'])
def log_json():
    sp_filter  = request.args.get('species')
    dev_filter = request.args.get('device')
    # Cap at 5000 (was 50000 → a 13 MB unauthenticated response per request).
    try:   limit   = max(1, min(int(request.args.get('limit', 500)), 5000))
    except: limit  = 500
    try:   from_id = int(request.args.get('from_id', 0))
    except: from_id = 0
    # Walk newest→oldest and stop as soon as we have `limit` rows, so the lock is
    # held for O(limit) rather than O(entire log) on every call.
    rows = []
    with write_lock:
        for e in reversed(detection_log):
            if e.get('id', 0) <= from_id:      continue
            if sp_filter  and e.get('species') != sp_filter:  continue
            if dev_filter and e.get('device')  != dev_filter: continue
            rows.append(e)
            if len(rows) >= limit: break
        s = dict(full_stats())
    return jsonify({'total_ever': s['total_detections'], 'total_log': s['log_size'],
                    'returned': len(rows), 'detections': rows})


@app.route('/log.txt', methods=['GET'])
def log_txt():
    sp_filter = request.args.get('species')
    try:   limit   = max(1, min(int(request.args.get('limit', 2000)), 5000))
    except: limit  = 2000
    try:   from_id = int(request.args.get('from_id', 0))
    except: from_id = 0

    rows = []
    with write_lock:
        for e in reversed(detection_log):
            if e.get('id', 0) <= from_id:  continue
            if sp_filter and e.get('species') != sp_filter: continue
            rows.append(e)
            if len(rows) >= limit: break
        total_ever = stats['total_detections']
        total_log  = len(detection_log)

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

    for e in rows:
      try:
        det_id  = e.get('id', '?')
        device  = str(e.get('device', '?'))[:8]
        lat     = _num(e.get('lat'), '.4f')
        lng     = _num(e.get('lng'), '.4f')
        try:
            dt = datetime.fromisoformat(str(e.get('ts','')).replace('Z','+00:00'))
            ts = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        except Exception:
            ts = str(e.get('ts',''))[:19]

        sp_name = str(e.get('name') or e.get('species', '?'))
        parts   = sp_name.split()
        if len(parts) >= 2:
            sp_name = parts[0][0] + '. ' + ' '.join(parts[1:])

        risk    = str(e.get('risk', '?'))
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
    with write_lock:
        hot  = [{'key':k,**v} for k,v in hotspot_cells.items()]
        near = [{'key':k,**v,'approaching':True} for k,v in detection_cells.items()
                if k not in hotspot_cells and v.get('total', 0) >= 50]
    return jsonify({'hotspots':hot,'approaching':near,'threshold':HOTSPOT_THRESHOLD})

@app.route('/federated/upload', methods=['POST'])
def upload():
    d = request.get_json(force=True, silent=True)
    if not isinstance(d, dict) or not all(k in d for k in ['deviceId','weights','steps']):
        print('[Fed REJECT 400] missing deviceId/weights/steps')
        return jsonify({'error':'missing required fields'}), 400
    # Reject malformed / non-finite / wrong-shape model uploads up front so they
    # never reach the aggregator. Returns 400 rather than silently dropping.
    if not _valid_weights(d['weights']):
        print('[Fed REJECT 400] weights not 4x4/len-4 finite')
        return jsonify({'error':'weights must be 4x4 W and length-4 b, all finite'}), 400
    try:
        steps = min(max(int(d.get('steps', 1)), 1), 500)
    except (TypeError, ValueError):
        return jsonify({'error':'steps not an integer'}), 400
    h = dh(d['deviceId'])
    with write_lock:
        touch(h); stats['total_uploads'] += 1
        pending_updates.append({'steps': steps, 'weights': d['weights']})
        if len(pending_updates) >= MIN_UPLOADS:
            fedavg(pending_updates.copy()); pending_updates.clear()
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
    import sys
    with write_lock:
        snap = dict(full_stats())
        n_seen = len(seen_events)
    return jsonify({
        'status':       'ok',
        'service':      'MosquitoNet v12',
        'data_dir':     DATA_DIR,
        'log_file':     LOG_FILE,
        'python':       sys.version,
        'writable':     os.access(DATA_DIR, os.W_OK),
        'ephemeral':    os.path.abspath(DATA_DIR).startswith('/tmp'),
        'min_conf':     MIN_CONF,
        'seen_events':  n_seen,
        'log_cap':      DETECTION_LOG_MAX,
        **snap,
    })

@app.route('/', methods=['GET'])
def index():
    return jsonify({'service':'MosquitoNet v12','data_dir':DATA_DIR,
                    'endpoints':['GET /heartbeat','POST /detection',
                                 'GET /log','GET /log.txt','GET /hotspots',
                                 'GET /federated/stats','GET /health']})

load_all()
if os.path.abspath(DATA_DIR).startswith('/tmp'):
    print('=' * 72)
    print(f'[WARNING] DATA_DIR={DATA_DIR} is under /tmp — storage is EPHEMERAL.')
    print('          Every restart/redeploy will PERMANENTLY DELETE all detections,')
    print('          all statistics and the trained federated model.')
    print('          Set DATA_DIR to a mounted persistent volume before going live.')
    print('=' * 72)

if __name__ == '__main__':
    import signal as _signal
    for _s in (_signal.SIGTERM, _signal.SIGINT):
        try: _signal.signal(_s, _graceful_exit)
        except Exception: pass
    port = int(os.environ.get('PORT', 5001))
    print(f'\nMosquitoNet v12 — DATA_DIR={DATA_DIR}  port={port}\n')
    print(f'  /health  /detection  /hotspots  /federated/upload  /log.txt\n')
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
