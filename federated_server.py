#!/usr/bin/env python3
"""
MosquitoNet Federated Server v11
================================
Log format (human-readable at /log.txt):

  #42  |  device: a1b2c3d4  |  32.0821, 34.7913  |  2025-01-15 14:32:07 UTC
  Ae. albopictus  |  RISK: HIGH  |  668.0 Hz  |  conf: 0.83

Fixes vs v8:
  - All writes are synchronous under write_lock — no race between
    rewrite and append threads (that caused disappearing entries)
  - No background threads for I/O — gunicorn's threads handle concurrency
  - Single write_lock serialises everything; writes are <2ms each
  - atexit flush is still present for clean shutdown
"""

import os, time, hashlib, threading, json, io, atexit
import numpy as np
from datetime import datetime, timezone
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins='*', supports_credentials=False,
     allow_headers=['Content-Type', 'Accept'],
     methods=['GET', 'POST', 'OPTIONS'])

# ── Persistence ───────────────────────────────────────────────────────────────
DATA_DIR   = os.environ.get('DATA_DIR', '/tmp')
STATE_FILE = os.path.join(DATA_DIR, 'mosquitonet_state.json')
LOG_FILE   = os.path.join(DATA_DIR, 'detections.jsonl')

# Single lock for BOTH in-memory state AND all disk writes — no races possible
write_lock = threading.Lock()

def _append_entry(entry):
    """Append one JSONL line. Called under write_lock."""
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, separators=(',', ':')) + '\n')
            f.flush()
            os.fsync(f.fileno())   # guarantee kernel flushes to disk
    except Exception as e:
        print(f'[append] {e}')

def _rewrite_log():
    """Rewrite full JSONL atomically. Called under write_lock."""
    try:
        tmp = LOG_FILE + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            for e in detection_log:
                f.write(json.dumps(e, separators=(',', ':')) + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, LOG_FILE)
    except Exception as e:
        print(f'[rewrite] {e}')

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
    try:
        with open(LOG_FILE, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        detection_log.append(json.loads(line))
                    except Exception:
                        pass
        print(f'[Log] loaded {len(detection_log)} entries from {LOG_FILE}')
    except FileNotFoundError:
        print(f'[Log] starting fresh at {LOG_FILE}')
    except Exception as e:
        print(f'[Log load] {e}')

@atexit.register
def _flush_on_exit():
    print('[Exit] flushing...')
    with write_lock:
        _rewrite_log()
        _save_state()
    print('[Exit] done')

# ── Model ─────────────────────────────────────────────────────────────────────
global_W = np.array([[ 2.8,0.9,1.2,0.8],[-0.6,0.7,1.0,0.7],
                     [-1.1,0.7,1.0,0.6],[ 1.8,0.8,1.3,0.9]], dtype=float)
global_b = np.array([-0.4,-0.3,-0.3,-0.4], dtype=float)

# ── State ─────────────────────────────────────────────────────────────────────
device_registry   = {}
session_registry  = {}
pending_updates   = []
detection_log     = []
detection_cells   = {}
hotspot_cells     = {}
next_detection_id = [1]
recent_log        = {}     # rkey → {det_id, ts, conf}
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
}
ACTIVE_SEC        = 90
HOTSPOT_THRESHOLD = 100
MIN_UPLOADS       = 3
start_time        = time.time()

# ── Helpers ───────────────────────────────────────────────────────────────────
def dh(raw):
    return hashlib.sha256(str(raw).encode()).hexdigest()[:16]

def active_now():
    c = time.time() - ACTIVE_SEC
    return sum(1 for t in device_registry.values() if t >= c)

def touch(h):
    device_registry[h] = time.time()

def cell_key(lat, lng, sp):
    return f'{round(float(lat)/0.05)*0.05:.3f},{round(float(lng)/0.05)*0.05:.3f},{sp}'

def _remember_event(event_id, det_id):
    """Record an eventId→det_id mapping for idempotency. Called under write_lock.
    Evicts the oldest entries (dicts preserve insertion order) once the cap is hit."""
    seen_events[event_id] = det_id
    if len(seen_events) > SEEN_EVENTS_MAX:
        for k in list(seen_events)[:len(seen_events) - SEEN_EVENTS_MAX]:
            seen_events.pop(k, None)

def full_stats():
    return {
        'active_now':       active_now(),
        'unique_devices':   len(device_registry),
        'ever_connected':   len(device_registry),
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
    that would otherwise silently corrupt the global model."""
    try:
        W = np.array(w['W'], dtype=float)
        b = np.array(w['b'], dtype=float)
    except Exception:
        return False
    if W.shape != global_W.shape or b.shape != global_b.shape:
        return False
    return bool(np.all(np.isfinite(W)) and np.all(np.isfinite(b)))

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
    raw_id = request.args.get('deviceId','anon') if request.method=='GET' \
             else (request.get_json(force=True,silent=True) or {}).get('deviceId','anon')
    sess   = request.args.get('sess','') if request.method=='GET' else ''
    h = dh(raw_id)
    with write_lock:
        new_s = sess and session_registry.get(h) != sess
        touch(h)
        if new_s:
            session_registry[h] = sess
            stats['total_sessions'] += 1
    return jsonify(full_stats())


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
                for e in reversed(detection_log):
                    if e.get('id') == rec['det_id']:
                        old_conf = e['conf']
                        e['conf'] = conf
                        e['conf_updated'] = datetime.now(timezone.utc).isoformat()
                        rec['conf'] = conf
                        print(f'[Det #{rec["det_id"]}] conf {old_conf:.3f}→{conf:.3f}')
                        _rewrite_log()   # rewrite under same lock — no race
                        _save_state()
                        break
            if event_id is not None:
                _remember_event(event_id, rec['det_id'])
                _save_state()
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
        recent_log[rkey] = {'det_id': det_id, 'ts': now, 'conf': conf}
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

        # Write under the same lock — guaranteed no concurrent append/rewrite
        _append_entry(entry)
        _save_state()
        snap = dict(full_stats())   # snapshot inside lock — consistent with what was written

    print(f'[Det #{det_id}] {sp_name} conf={conf:.3f} freq={freq:.1f}Hz '
          f'lat={entry["lat"]} lng={entry["lng"]}')
    return jsonify({'received': True, 'detection_id': det_id, **snap})


@app.route('/log', methods=['GET'])
def log_json():
    sp_filter  = request.args.get('species')
    dev_filter = request.args.get('device')
    try:   limit   = min(int(request.args.get('limit', 500)), 50000)
    except: limit  = 500
    try:   from_id = int(request.args.get('from_id', 0))
    except: from_id = 0
    with write_lock:
        rows = [e for e in detection_log
                if (not sp_filter  or e.get('species')  == sp_filter)
                and (not dev_filter or e.get('device')   == dev_filter)
                and e.get('id', 0) > from_id]
        s = dict(full_stats())
    return jsonify({'total_ever': s['total_detections'], 'total_log': s['log_size'],
                    'returned': len(rows[-limit:]), 'detections': rows[-limit:][::-1]})


@app.route('/log.txt', methods=['GET'])
def log_txt():
    sp_filter = request.args.get('species')
    try:   limit   = min(int(request.args.get('limit', 2000)), 50000)
    except: limit  = 2000
    try:   from_id = int(request.args.get('from_id', 0))
    except: from_id = 0

    with write_lock:
        rows = [e for e in detection_log
                if (not sp_filter or e.get('species') == sp_filter)
                and e.get('id', 0) > from_id]
        total_ever = stats['total_detections']
        total_log  = len(detection_log)

    rows = rows[-limit:][::-1]   # newest first

    buf = io.StringIO()
    buf.write('━' * 60 + '\n')
    buf.write(f'  MosquitoNet Detection Log\n')
    buf.write(f'  Total ever: {total_ever}  |  In log: {total_log}\n')
    buf.write(f'  {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} UTC\n')
    buf.write('━' * 60 + '\n\n')

    for e in rows:
        det_id  = e.get('id', '?')
        device  = str(e.get('device', '?'))[:8]
        lat     = f'{e["lat"]:.4f}' if e.get('lat') is not None else 'n/a'
        lng     = f'{e["lng"]:.4f}' if e.get('lng') is not None else 'n/a'
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
        freq    = f'{float(e.get("freq", 0)):.1f} Hz'
        conf    = f'{float(e.get("conf", 0)):.2f}'
        upd     = '  *(conf updated)' if e.get('conf_updated') else ''

        buf.write(f'#{det_id}  |  device: {device}  |  {lat}, {lng}  |  {ts}\n')
        buf.write(f'  {sp_name}  |  risk: {risk}  |  {freq}  |  conf: {conf}{upd}\n')
        buf.write('\n')

    return Response(buf.getvalue(), mimetype='text/plain; charset=utf-8')


@app.route('/hotspots', methods=['GET'])
def hotspots():
    with write_lock:
        hot  = [{'key':k,**v} for k,v in hotspot_cells.items()]
        near = [{'key':k,**v,'approaching':True} for k,v in detection_cells.items()
                if k not in hotspot_cells and v['total']>=50]
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
            _save_state()   # persist the freshly-aggregated global model
    return jsonify({'status':'accepted','weights':{'W':global_W.tolist(),'b':global_b.tolist()},**full_stats()})

@app.route('/federated/model',  methods=['GET'])
def model(): return jsonify({'round':stats['total_rounds'],'weights':{'W':global_W.tolist(),'b':global_b.tolist()}})

@app.route('/federated/stats',  methods=['GET'])
def get_stats():
    with write_lock: return jsonify(full_stats())

@app.route('/health', methods=['GET'])
def health():
    import sys
    return jsonify({
        'status':       'ok',
        'service':      'MosquitoNet v11',
        'data_dir':     DATA_DIR,
        'log_file':     LOG_FILE,
        'python':       sys.version,
        'writable':     os.access(DATA_DIR, os.W_OK),
        'ephemeral':    os.path.abspath(DATA_DIR).startswith('/tmp'),
        'min_conf':     MIN_CONF,
        'seen_events':  len(seen_events),
        **full_stats(),
    })

@app.route('/', methods=['GET'])
def index():
    return jsonify({'service':'MosquitoNet v11','data_dir':DATA_DIR,
                    'endpoints':['GET /heartbeat','POST /detection',
                                 'GET /log','GET /log.txt','GET /hotspots',
                                 'GET /federated/stats','GET /health']})

load_all()
if os.path.abspath(DATA_DIR).startswith('/tmp'):
    print(f'[WARN] DATA_DIR={DATA_DIR} is under /tmp — data is EPHEMERAL and will be '
          f'lost on restart. Set DATA_DIR to a persistent volume for production.')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f'\nMosquitoNet v11 — DATA_DIR={DATA_DIR}  port={port}\n')
    print(f'  /health  /detection  /hotspots  /federated/upload  /log.txt\n')
    app.run(host='0.0.0.0', port=port, debug=False)
