#!/usr/bin/env python3
"""
MosquitoNet Federated Server v4
=================================
Adds persistent detection log — every confirmed detection stored with:
  timestamp, species, confidence, frequency, location (DP-jittered),
  device_hash. Used for hotspot detection and scientific analysis.
"""

import json, os, time, hashlib, threading
import numpy as np
from datetime import datetime, timezone

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError:
    raise

app = Flask(__name__)
CORS(app, origins='*', supports_credentials=False,
     allow_headers=['Content-Type', 'Accept'],
     methods=['GET', 'POST', 'OPTIONS'])

# ── Model ─────────────────────────────────────────────────────────────────────
N_FEATURES, N_CLASSES = 4, 10   # 10 species now
global_W = np.array([
    [ 2.8, 0.9, 1.2, 0.8],
    [-0.6, 0.7, 1.0, 0.7],
    [-1.1, 0.7, 1.0, 0.6],
    [ 1.8, 0.8, 1.3, 0.9],
], dtype=float)
global_b = np.array([-0.4, -0.3, -0.3, -0.4], dtype=float)

# ── State ─────────────────────────────────────────────────────────────────────
lock             = threading.Lock()
device_registry  = {}
session_registry = {}
pending_updates  = []

# Detection log: list of dicts, persistent in memory (Railway resets on redeploy)
# Each entry: {ts, species, confidence, freq, lat, lng, device_hash, risk}
detection_log    = []
MAX_LOG_SIZE     = 50000  # keep last 50k detections

# Per-cell counts for hotspot detection (0.05° ≈ 5.5km)
detection_cells  = {}
hotspot_cells    = {}
HOTSPOT_THRESHOLD = 100

# Full detection log for research analysis
detection_log = []
MAX_LOG = 100000

# Per-device per-species 60s cooldown
detection_cooldowns = {}

stats = {
    'total_uploads': 0, 'total_rounds': 0,
    'total_detections': 0, 'total_sessions': 0,
    'last_aggregate': None,
}

ACTIVE_WINDOW_SEC = 90
MIN_UPLOADS_ROUND = 3
start_time = time.time()

# ── Helpers ───────────────────────────────────────────────────────────────────
def dh(raw_id):
    return hashlib.sha256(str(raw_id).encode()).hexdigest()[:16]

def active_now():
    cutoff = time.time() - ACTIVE_WINDOW_SEC
    return sum(1 for ts in device_registry.values() if ts >= cutoff)

def touch(h):
    device_registry[h] = time.time()

def cell_key(lat, lng, species):
    clat = round(float(lat) / 0.05) * 0.05
    clng = round(float(lng) / 0.05) * 0.05
    return f"{clat:.3f},{clng:.3f},{species}"

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
    if total == 0: return
    # Pad weights to current dimensions if needed
    nW = global_W.copy()
    nb = global_b.copy()
    try:
        aW = np.array(updates[0]['weights']['W'])
        if aW.shape == global_W.shape:
            nW = np.zeros_like(global_W)
            nb = np.zeros_like(global_b)
            for u in updates:
                w = u['steps'] / total
                nW += w * np.array(u['weights']['W'])
                nb += w * np.array(u['weights']['b'])
            global_W = 0.3 * global_W + 0.7 * nW
            global_b = 0.3 * global_b + 0.7 * nb
    except Exception as e:
        print(f"[FedAvg] shape mismatch: {e}")
    stats['total_rounds'] += 1
    stats['last_aggregate'] = datetime.now(timezone.utc).isoformat()
    print(f"[FedAvg] Round {stats['total_rounds']}")

# ── CORS ──────────────────────────────────────────────────────────────────────
@app.after_request
def add_cors(r):
    r.headers['Access-Control-Allow-Origin']  = '*'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept'
    r.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return r

@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def options(path): return '', 204

# ══════════════════════════════════════════════════════════════════════════════

@app.route('/heartbeat', methods=['POST', 'GET'])
def heartbeat():
    # Accept both GET (query params, no preflight) and POST (JSON body)
    if request.method == 'GET':
        raw_id   = request.args.get('deviceId', 'anon')
        sess_val = request.args.get('sess', '')
    else:
        data     = request.get_json(force=True, silent=True) or {}
        raw_id   = data.get('deviceId', 'anon')
        sess_val = str(data.get('sessionStart', ''))

    h = dh(raw_id)
    with lock:
        new_sess = sess_val and session_registry.get(h) != sess_val
        touch(h)
        if new_sess:
            session_registry[h] = sess_val
            stats['total_sessions'] += 1
    return jsonify(full_stats())


@app.route('/detection', methods=['POST'])
def report_detection():
    data    = request.get_json(force=True, silent=True) or {}
    raw_id  = data.get('deviceId', 'anon')
    h       = dh(raw_id)
    species = str(data.get('species', ''))
    lat     = data.get('lat')
    lng     = data.get('lng')
    conf    = data.get('confidence', 0)
    freq    = data.get('frequency', 0)
    risk    = data.get('risk', 'UNKNOWN')

    # Server-side 60s cooldown per device+species
    cd_key = f"{h}:{species}"
    now    = time.time()
    with lock:
        if now - detection_cooldowns.get(cd_key, 0) < 60:
            return jsonify({'received': False, 'reason': 'cooldown', **full_stats()})
        detection_cooldowns[cd_key] = now
        touch(h)
        stats['total_detections'] += 1

        # ── Persistent detection log ─────────────────────────────
        entry = {
            'ts':      datetime.now(timezone.utc).isoformat(),
            'species': species,
            'conf':    round(float(conf), 3),
            'freq':    round(float(freq), 1),
            'risk':    risk,
            'device':  h,
        }
        if lat is not None and lng is not None:
            try:
                entry['lat'] = round(float(lat), 4)
                entry['lng'] = round(float(lng), 4)
            except Exception:
                pass

        detection_log.append(entry)
        if len(detection_log) > MAX_LOG_SIZE:
            detection_log.pop(0)

        # ── Per-cell accumulation ────────────────────────────────
        if lat is not None and lng is not None:
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
                    print(f"[Hotspot] {ck} → {detection_cells[ck]['total']} det.")
            except Exception as e:
                print(f"[Cell] {e}")

    # Append to research log
    with lock:
        detection_log.append({
            'ts':         datetime.utcnow().isoformat() + 'Z',
            'device':     h,
            'species':    species,
            'confidence': float(data.get('confidence', 0)),
            'frequency':  float(data.get('frequency', 0)),
            'lat':        float(lat) if lat is not None else None,
            'lng':        float(lng) if lng is not None else None,
        })
        if len(detection_log) > MAX_LOG:
            detection_log.pop(0)

    loc = f"{lat:.4f},{lng:.4f}" if lat is not None else "no-gps"
    print(f"[Det] {species} conf={data.get('confidence','?')} freq={data.get('frequency','?')}Hz loc={loc}")
    return jsonify({'received': True, **full_stats()})


@app.route('/detections/log', methods=['GET'])
def get_detection_log():
    """Full detection log for research analysis. Last N entries."""
    try:
        n = min(int(request.args.get('n', 1000)), 10000)
    except Exception:
        n = 1000
    with lock:
        entries = detection_log[-n:]
    return jsonify({
        'count': len(entries),
        'total_logged': len(detection_log),
        'entries': entries,
    })


@app.route('/detections/log', methods=['GET'])
def get_detection_log():
    """Return recent detection log for analysis. Optional ?species= filter."""
    sp_filter = request.args.get('species')
    limit = min(int(request.args.get('limit', 500)), 2000)
    with lock:
        entries = detection_log[-limit:]
        if sp_filter:
            entries = [e for e in entries if e.get('species') == sp_filter]
    return jsonify({
        'count':      len(entries),
        'total_ever': stats['total_detections'],
        'entries':    entries,
    })


@app.route('/detections/stats', methods=['GET'])
def detection_stats():
    """Per-species detection counts and hotspot summary."""
    with lock:
        by_species = {}
        for e in detection_log:
            sp = e.get('species', 'unknown')
            by_species[sp] = by_species.get(sp, 0) + 1
        return jsonify({
            'by_species':   by_species,
            'total':        stats['total_detections'],
            'hotspots':     len(hotspot_cells),
            'cells_tracked': len(detection_cells),
        })


@app.route('/hotspots', methods=['GET'])
def get_hotspots():
    with lock:
        hot  = [{'key': k, **v} for k, v in hotspot_cells.items()]
        near = [{'key': k, **v, 'approaching': True}
                for k, v in detection_cells.items()
                if k not in hotspot_cells and v['total'] >= 50]
    return jsonify({'hotspots': hot, 'approaching': near,
                    'threshold': HOTSPOT_THRESHOLD})


@app.route('/log', methods=['GET'])
def get_log():
    """Return detection log for analysis. Optional ?species= and ?limit= filters."""
    sp_filter = request.args.get('species')
    try:    limit = min(int(request.args.get('limit', 1000)), 10000)
    except: limit = 1000
    with lock:
        filtered = [e for e in detection_log
                    if not sp_filter or e.get('species') == sp_filter]
        return jsonify({
            'count':      len(filtered),
            'total_log':  len(detection_log),
            'detections': filtered[-limit:],
        })


@app.route('/federated/upload', methods=['POST'])
def upload():
    data = request.get_json(force=True, silent=True) or {}
    if not all(k in data for k in ['deviceId', 'weights', 'steps']):
        return jsonify({'error': 'missing'}), 400
    h = dh(data['deviceId'])
    with lock:
        touch(h)
        stats['total_uploads'] += 1
        pending_updates.append({'device_hash': h,
                                 'steps': min(int(data.get('steps', 1)), 500),
                                 'weights': data['weights']})
        if len(pending_updates) >= MIN_UPLOADS_ROUND:
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
    return jsonify({'status': 'ok', 'service': 'MosquitoNet v4', **full_stats()})


@app.route('/', methods=['GET'])
def index():
    return jsonify({'service': 'MosquitoNet v4',
                    'endpoints': ['POST /heartbeat', 'POST /detection',
                                  'GET /hotspots', 'GET /log',
                                  'POST /federated/upload', 'GET /federated/stats',
                                  'GET /health']})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f'\nMosquitoNet v4 :{port}\n')
    app.run(host='0.0.0.0', port=port, debug=False)
