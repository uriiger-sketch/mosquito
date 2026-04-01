#!/usr/bin/env python3
"""
MosquitoNet Federated Server v6
================================
- Every detection stored with sequential ID
- Log format: id, ts, device, lat, lng, species, freq, conf, asymptomatic
- No server-side cooldown (client handles dedup via 60s UI cooldown)
- Atomic saves on every detection
- Persistent storage across restarts
"""

import os, time, hashlib, threading, json
import numpy as np
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins='*', supports_credentials=False,
     allow_headers=['Content-Type', 'Accept'],
     methods=['GET', 'POST', 'OPTIONS'])

# ── Persistence ───────────────────────────────────────────────────────────────
DATA_DIR  = os.environ.get('DATA_DIR', '/tmp')
DATA_FILE = os.path.join(DATA_DIR, 'mosquitonet_v6.json')

def save_state():
    """Atomic background save — never blocks a request."""
    def _save():
        try:
            payload = {
                'stats':           stats,
                'next_id':         next_detection_id[0],
                'detection_log':   detection_log,   # save ALL
                'detection_cells': detection_cells,
                'hotspot_cells':   hotspot_cells,
            }
            tmp = DATA_FILE + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(payload, f, separators=(',', ':'))
            os.replace(tmp, DATA_FILE)   # atomic rename
        except Exception as e:
            print(f'[Save] {e}')
    threading.Thread(target=_save, daemon=True).start()

def load_state():
    global detection_log, detection_cells, hotspot_cells, stats
    try:
        with open(DATA_FILE) as f:
            d = json.load(f)
        for k in stats:
            if k in d.get('stats', {}):
                stats[k] = d['stats'][k]
        next_detection_id[0] = d.get('next_id', stats['total_detections'] + 1)
        detection_log   = d.get('detection_log', [])
        detection_cells = d.get('detection_cells', {})
        hotspot_cells   = d.get('hotspot_cells', {})
        print(f'[Load] {len(detection_log)} entries in log, '
              f'total_ever={stats["total_detections"]}, next_id={next_detection_id[0]}')
    except FileNotFoundError:
        print('[Load] Fresh start — no saved state')
    except Exception as e:
        print(f'[Load] {e}')

# ── Model ─────────────────────────────────────────────────────────────────────
global_W = np.array([
    [ 2.8, 0.9, 1.2, 0.8],
    [-0.6, 0.7, 1.0, 0.7],
    [-1.1, 0.7, 1.0, 0.6],
    [ 1.8, 0.8, 1.3, 0.9],
], dtype=float)
global_b = np.array([-0.4, -0.3, -0.3, -0.4], dtype=float)

# ── State ─────────────────────────────────────────────────────────────────────
lock              = threading.Lock()
device_registry   = {}
session_registry  = {}
pending_updates   = []
detection_log     = []      # every detection, never cleared, with sequential id
detection_cells   = {}
hotspot_cells     = {}
next_detection_id = [1]     # mutable list so background thread can update

stats = {
    'total_detections': 0,
    'total_sessions':   0,
    'total_uploads':    0,
    'total_rounds':     0,
    'last_aggregate':   None,
}

ACTIVE_SEC        = 90
HOTSPOT_THRESHOLD = 100
MAX_LOG           = 200000  # 200k entries (~100MB max)
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
    clat = round(float(lat) / 0.05) * 0.05
    clng = round(float(lng) / 0.05) * 0.05
    return f'{clat:.3f},{clng:.3f},{sp}'

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
        'pending_updates':  len(pending_updates),
        'last_aggregate':   stats['last_aggregate'],
        'uptime_seconds':   int(time.time() - start_time),
        'log_size':         len(detection_log),
    }

def fedavg(updates):
    global global_W, global_b
    total = sum(u['steps'] for u in updates)
    if not total: return
    nW = np.zeros_like(global_W)
    nb = np.zeros_like(global_b)
    for u in updates:
        w = u['steps'] / total
        try:
            nW += w * np.array(u['weights']['W'])
            nb += w * np.array(u['weights']['b'])
        except Exception: pass
    global_W = 0.3 * global_W + 0.7 * nW
    global_b = 0.3 * global_b + 0.7 * nb
    stats['total_rounds'] += 1
    stats['last_aggregate'] = datetime.now(timezone.utc).isoformat()

# ── CORS ──────────────────────────────────────────────────────────────────────
@app.after_request
def cors(r):
    r.headers['Access-Control-Allow-Origin']  = '*'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type,Accept'
    r.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return r

@app.route('/', defaults={'p': ''}, methods=['OPTIONS'])
@app.route('/<path:p>', methods=['OPTIONS'])
def opts(p): return '', 204

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/heartbeat', methods=['GET', 'POST'])
def heartbeat():
    if request.method == 'GET':
        raw_id   = request.args.get('deviceId', 'anon')
        sess_val = request.args.get('sess', '')
    else:
        d        = request.get_json(force=True, silent=True) or {}
        raw_id   = d.get('deviceId', 'anon')
        sess_val = str(d.get('sessionStart', ''))
    h = dh(raw_id)
    with lock:
        new_s = sess_val and session_registry.get(h) != sess_val
        touch(h)
        if new_s:
            session_registry[h] = sess_val
            stats['total_sessions'] += 1
    return jsonify(full_stats())


@app.route('/detection', methods=['POST'])
def detection():
    """
    Log every single confirmed detection.
    No server-side cooldown — every POST is stored.
    Log entry format (in order): id, ts, device, lat, lng,
                                  species, freq, conf, asymptomatic
    """
    d       = request.get_json(force=True, silent=True) or {}
    raw_id  = d.get('deviceId', 'anon')
    species = str(d.get('species', ''))
    lat     = d.get('lat')
    lng     = d.get('lng')
    conf    = d.get('confidence', 0)
    freq    = d.get('frequency', 0)
    asymp   = bool(d.get('asymptomatic', False))
    risk    = d.get('risk', 'UNKNOWN')
    ts_str  = d.get('ts') or datetime.now(timezone.utc).isoformat()
    h       = dh(raw_id)

    with lock:
        det_id = next_detection_id[0]
        next_detection_id[0] += 1
        touch(h)
        stats['total_detections'] += 1

        # ── Ordered log entry ──────────────────────────────────────────────
        entry = {'id': det_id, 'ts': ts_str, 'device': h}

        if lat is not None and lng is not None:
            try:
                entry['lat'] = round(float(lat), 4)
                entry['lng'] = round(float(lng), 4)
            except Exception:
                entry['lat'] = None
                entry['lng'] = None
        else:
            entry['lat'] = None
            entry['lng'] = None

        entry['species']      = species
        entry['freq']         = round(float(freq), 1)
        entry['conf']         = round(float(conf), 3)
        entry['asymptomatic'] = asymp

        detection_log.append(entry)
        if len(detection_log) > MAX_LOG:
            detection_log.pop(0)

        # ── Hotspot cell accumulation ──────────────────────────────────────
        if entry['lat'] is not None:
            try:
                ck = cell_key(entry['lat'], entry['lng'], species)
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

    print(f'[Det #{det_id}] {species} freq={freq:.1f}Hz conf={conf:.3f} '
          f'lat={entry["lat"]} lng={entry["lng"]} dev={h}')

    # Atomic background save — every single detection persisted
    save_state()

    return jsonify({'received': True, 'detection_id': det_id, **full_stats()})


@app.route('/log', methods=['GET'])
def log():
    """
    Return detection log.
    Field order: id, ts, device, lat, lng, species, freq, conf, asymptomatic
    Query: ?species=X  ?device=Z  ?limit=N  ?offset=N  ?from_id=N
    """
    sp_filter  = request.args.get('species')
    dev_filter = request.args.get('device')
    try:   limit  = min(int(request.args.get('limit', 1000)), 50000)
    except: limit = 1000
    try:   from_id = int(request.args.get('from_id', 0))
    except: from_id = 0

    with lock:
        rows = [e for e in detection_log
                if (not sp_filter  or e.get('species') == sp_filter)
                and (not dev_filter or e.get('device') == dev_filter)
                and e.get('id', 0) > from_id]
        total_ever = stats['total_detections']
        total_log  = len(detection_log)

    # Newest first
    result = rows[-limit:][::-1]

    return jsonify({
        'total_ever':  total_ever,
        'total_log':   total_log,
        'count':       len(result),
        'detections':  result,
    })


@app.route('/hotspots', methods=['GET'])
def hotspots():
    with lock:
        hot  = [{'key': k, **v} for k, v in hotspot_cells.items()]
        near = [{'key': k, **v, 'approaching': True}
                for k, v in detection_cells.items()
                if k not in hotspot_cells and v['total'] >= 50]
    return jsonify({'hotspots': hot, 'approaching': near,
                    'threshold': HOTSPOT_THRESHOLD})


@app.route('/federated/upload', methods=['POST'])
def upload():
    d = request.get_json(force=True, silent=True) or {}
    if not all(k in d for k in ['deviceId', 'weights', 'steps']):
        return jsonify({'error': 'missing'}), 400
    h = dh(d['deviceId'])
    with lock:
        touch(h)
        stats['total_uploads'] += 1
        pending_updates.append({
            'steps':   min(int(d.get('steps', 1)), 500),
            'weights': d['weights'],
        })
        if len(pending_updates) >= MIN_UPLOADS:
            fedavg(pending_updates.copy())
            pending_updates.clear()
    return jsonify({'status': 'accepted',
                    'weights': {'W': global_W.tolist(), 'b': global_b.tolist()},
                    **full_stats()})


@app.route('/federated/model', methods=['GET'])
def model():
    return jsonify({'round': stats['total_rounds'],
                    'weights': {'W': global_W.tolist(), 'b': global_b.tolist()}})


@app.route('/federated/stats', methods=['GET'])
def get_stats():
    with lock: return jsonify(full_stats())


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'MosquitoNet v6', **full_stats()})


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'MosquitoNet v6',
        'log_schema': 'id, ts, device, lat, lng, species, freq, conf, asymptomatic',
        'endpoints': [
            'GET  /heartbeat?deviceId=X',
            'POST /detection  — body: {deviceId, species, confidence, frequency, asymptomatic, ts, lat, lng}',
            'GET  /log?species=X&device=Z&limit=N&from_id=N',
            'GET  /hotspots',
            'POST /federated/upload',
            'GET  /federated/stats',
            'GET  /health',
        ],
    })


# Load persisted state on startup (gunicorn imports this module)
load_state()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f'\nMosquitoNet v6 on :{port}\n')
    app.run(host='0.0.0.0', port=port, debug=False)
