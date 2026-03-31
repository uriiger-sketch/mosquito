#!/usr/bin/env python3
"""
MosquitoNet Federated Server v3
================================
Additions over v2:
  • Per-cell detection counts (0.05° ≈ 5.5km cells)
  • Hotspot cells: when a cell reaches 100 detections, it's flagged
  • /hotspots endpoint returns all hotspot cells
  • Per-device per-species 60s rate-limiting enforced server-side too
"""

import json, os, time, hashlib, threading
import numpy as np
from datetime import datetime

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
N_FEATURES, N_CLASSES = 4, 4
global_W = np.array([
    [ 2.8, 0.9, 1.2, 0.8],
    [-0.6, 0.7, 1.0, 0.7],
    [-1.1, 0.7, 1.0, 0.6],
    [ 1.8, 0.8, 1.3, 0.9],
], dtype=float)
global_b = np.array([-0.4, -0.3, -0.3, -0.4], dtype=float)

# ── State ─────────────────────────────────────────────────────────────────────
lock             = threading.Lock()
device_registry  = {}   # hash → last_seen
session_registry = {}   # hash → session_start string
pending_updates  = []

# Per-cell detection counts: cellKey → {species: count, total: count}
# Cell resolution: 0.05° × 0.05° ≈ 5.5km × 5.5km at equator
detection_cells  = {}   # cellKey → {'species': {'anopheles': N, ...}, 'total': N}
hotspot_cells    = {}   # cellKey → cell info (total ≥ HOTSPOT_THRESHOLD)
HOTSPOT_THRESHOLD = 100

# Per-device per-species cooldown (server-enforced: 60s)
detection_cooldowns = {}  # f"{deviceHash}:{species}" → last_report_time

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
    """0.05° ≈ 5.5km cell resolution"""
    clat = round(float(lat) / 0.05) * 0.05
    clng = round(float(lng) / 0.05) * 0.05
    return f"{clat:.3f},{clng:.3f},{species}"

def cell_centroid(key):
    parts = key.split(',')
    return float(parts[0]), float(parts[1]), parts[2]

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
    }

def fedavg(updates):
    global global_W, global_b
    total = sum(u['steps'] for u in updates)
    if total == 0: return
    nW = np.zeros((N_CLASSES, N_FEATURES))
    nb = np.zeros(N_CLASSES)
    for u in updates:
        w = u['steps'] / total
        nW += w * np.array(u['weights']['W'])
        nb += w * np.array(u['weights']['b'])
    global_W = 0.3 * global_W + 0.7 * nW
    global_b = 0.3 * global_b + 0.7 * nb
    stats['total_rounds'] += 1
    stats['last_aggregate'] = datetime.utcnow().isoformat() + 'Z'
    print(f"[FedAvg] Round {stats['total_rounds']}")

# ── CORS on every response ────────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Accept'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    return '', 204

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/heartbeat', methods=['POST', 'GET'])
def heartbeat():
    data   = request.get_json(force=True, silent=True) or {}
    raw_id = data.get('deviceId', request.args.get('deviceId', 'anon'))
    h      = dh(raw_id)
    with lock:
        is_new = (data.get('sessionStart') and
                  session_registry.get(h) != str(data.get('sessionStart')))
        touch(h)
        if is_new:
            session_registry[h] = str(data.get('sessionStart'))
            stats['total_sessions'] += 1
    return jsonify(full_stats())


@app.route('/detection', methods=['POST'])
def report_detection():
    data   = request.get_json(force=True, silent=True) or {}
    raw_id = data.get('deviceId', 'anon')
    h      = dh(raw_id)
    species = data.get('species', '')
    lat     = data.get('lat')
    lng     = data.get('lng')

    # Server-side 60s cooldown per device+species
    cd_key = f"{h}:{species}"
    now    = time.time()
    with lock:
        last = detection_cooldowns.get(cd_key, 0)
        if now - last < 60:
            return jsonify({'received': False, 'reason': 'cooldown', **full_stats()})
        detection_cooldowns[cd_key] = now
        touch(h)
        stats['total_detections'] += 1

        # Store per-cell counts if location provided
        if lat is not None and lng is not None and species:
            try:
                ck = cell_key(lat, lng, species)
                if ck not in detection_cells:
                    detection_cells[ck] = {'species': species, 'total': 0,
                                           'lat': round(float(lat)/0.05)*0.05,
                                           'lng': round(float(lng)/0.05)*0.05}
                detection_cells[ck]['total'] += 1

                # Promote to hotspot when threshold reached
                if detection_cells[ck]['total'] >= HOTSPOT_THRESHOLD:
                    hotspot_cells[ck] = {**detection_cells[ck]}
                    print(f"[Hotspot] {ck} hit {detection_cells[ck]['total']} detections!")
            except Exception as e:
                print(f"[Cell err] {e}")

    conf = data.get('confidence', '?')
    freq = data.get('frequency', '?')
    print(f"[Det] {species} conf={conf} freq={freq}Hz device={h}")
    return jsonify({'received': True, **full_stats()})


@app.route('/hotspots', methods=['GET'])
def get_hotspots():
    """Return all hotspot cells (≥100 detections in a 5km cell)."""
    with lock:
        cells_list = []
        for ck, info in hotspot_cells.items():
            cells_list.append({
                'key':     ck,
                'lat':     info['lat'],
                'lng':     info['lng'],
                'species': info['species'],
                'total':   info['total'],
            })
        # Also include cells approaching hotspot (≥50)
        approaching = []
        for ck, info in detection_cells.items():
            if ck not in hotspot_cells and info['total'] >= 50:
                approaching.append({
                    'key':     ck,
                    'lat':     info['lat'],
                    'lng':     info['lng'],
                    'species': info['species'],
                    'total':   info['total'],
                    'approaching': True,
                })
        return jsonify({
            'hotspots':   cells_list,
            'approaching': approaching,
            'threshold':  HOTSPOT_THRESHOLD,
        })


@app.route('/federated/upload', methods=['POST'])
def upload():
    data = request.get_json(force=True, silent=True) or {}
    if not all(k in data for k in ['deviceId', 'weights', 'steps']):
        return jsonify({'error': 'missing fields'}), 400
    h = dh(data['deviceId'])
    with lock:
        touch(h)
        stats['total_uploads'] += 1
        pending_updates.append({
            'device_hash': h,
            'steps': min(int(data.get('steps', 1)), 500),
            'weights': data['weights'],
        })
        if len(pending_updates) >= MIN_UPLOADS_ROUND:
            fedavg(pending_updates.copy())
            pending_updates.clear()
    return jsonify({'status': 'accepted',
                    'weights': {'W': global_W.tolist(), 'b': global_b.tolist()},
                    **full_stats()})


@app.route('/federated/model', methods=['GET'])
def get_model():
    return jsonify({'round': stats['total_rounds'],
                    'weights': {'W': global_W.tolist(), 'b': global_b.tolist()}})


@app.route('/federated/stats', methods=['GET'])
def get_stats():
    with lock:
        return jsonify(full_stats())


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'MosquitoNet v3', **full_stats()})


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'MosquitoNet v3',
        'endpoints': [
            'POST /heartbeat',
            'POST /detection  — body: {deviceId, species, confidence, frequency, lat, lng}',
            'GET  /hotspots   — returns cells with ≥100 detections',
            'POST /federated/upload',
            'GET  /federated/stats',
            'GET  /health',
        ],
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f'\nMosquitoNet Server v3 on :{port}\n')
    app.run(host='0.0.0.0', port=port, debug=False)
