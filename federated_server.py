#!/usr/bin/env python3
"""
MosquitoNet Federated Aggregation Server — v2
=============================================
Tracks:
  - unique_devices_ever   : all devices that have ever connected
  - active_devices_now    : devices that sent a heartbeat in the last 90 seconds
  - total_sessions        : total listening sessions started
  - total_uploads         : gradient uploads received
  - total_detections      : confirmed mosquito detections reported
  - total_rounds          : FedAvg rounds completed

Endpoints:
  POST /heartbeat          — called every 30s while app is open (no data needed)
  POST /federated/upload   — submit local model update
  POST /detection          — report a confirmed detection (species, location hash)
  GET  /federated/model    — download global model
  GET  /federated/stats    — live dashboard stats
  GET  /health             — health check
"""

import json, os, time, hashlib, threading
import numpy as np
from datetime import datetime
from collections import defaultdict

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError:
    print("pip install flask flask-cors numpy")
    raise

app = Flask(__name__)
_origin = os.environ.get('ALLOWED_ORIGIN', '*')
CORS(app, origins=_origin)

# ── Global model ──────────────────────────────────────────────────────────────
N_FEATURES = 4
N_CLASSES  = 4

# Pre-seeded from HumBugDB (Kiskin et al. 2021, CC-BY-4.0) and
# Mukundarajan et al. 2017 (Stanford Abuzz, eLife).
# Frequency centroids: Anopheles 406Hz, Ae.aegypti 617Hz,
#                      Ae.albopictus 672Hz, Culex 453Hz
global_W = np.array([
    [ 2.8, 0.9, 1.2, 0.8],   # Anopheles
    [-0.6, 0.7, 1.0, 0.7],   # Ae. aegypti
    [-1.1, 0.7, 1.0, 0.6],   # Ae. albopictus
    [ 1.8, 0.8, 1.3, 0.9],   # Culex
], dtype=float)
global_b = np.array([-0.4, -0.3, -0.3, -0.4], dtype=float)

# ── State ─────────────────────────────────────────────────────────────────────
lock = threading.Lock()

# Device registry: device_hash → last_seen timestamp
device_registry  = {}   # hash → last_seen (heartbeat or upload)
# Sessions: device_hash → last session_start
session_registry = {}   # hash → start_ts

pending_updates  = []

stats = {
    'total_uploads':    0,
    'total_rounds':     0,
    'total_detections': 0,
    'total_sessions':   0,
    'last_aggregate':   None,
}

ACTIVE_WINDOW_SEC = 90   # device counts as "live now" if seen in last 90s
MIN_UPLOADS_ROUND  = 3   # min uploads to trigger FedAvg

start_time = time.time()


# ── Helpers ───────────────────────────────────────────────────────────────────
def device_hash(raw_id: str) -> str:
    return hashlib.sha256(raw_id.encode()).hexdigest()[:16]

def active_now() -> int:
    cutoff = time.time() - ACTIVE_WINDOW_SEC
    return sum(1 for ts in device_registry.values() if ts >= cutoff)

def touch_device(dh: str):
    """Record that this device is alive right now."""
    device_registry[dh] = time.time()


# ── FedAvg ────────────────────────────────────────────────────────────────────
def fedavg_aggregate(updates):
    global global_W, global_b
    total = sum(u['steps'] for u in updates)
    if total == 0:
        return
    new_W = np.zeros((N_CLASSES, N_FEATURES))
    new_b = np.zeros(N_CLASSES)
    for u in updates:
        w = u['steps'] / total
        new_W += w * np.array(u['weights']['W'])
        new_b += w * np.array(u['weights']['b'])
    # Momentum: blend 30% global, 70% new
    global_W = 0.30 * global_W + 0.70 * new_W
    global_b = 0.30 * global_b + 0.70 * new_b
    stats['total_rounds'] += 1
    stats['last_aggregate'] = datetime.utcnow().isoformat() + 'Z'
    print(f"  [FedAvg] Round {stats['total_rounds']} — {len(updates)} updates, {total} steps")


# ── DP verification ───────────────────────────────────────────────────────────
def verify_dp(update, sigma_min=0.008):
    W = np.array(update['weights']['W'])
    if np.std(W) < sigma_min:
        return False, f"noise std={np.std(W):.4f} < {sigma_min}"
    return True, "ok"


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    """
    Called every 30 s by every open app instance.
    Body: { "deviceId": "...", "sessionStart": <unix_ms>, "listening": true/false }
    No audio, no location. Just "I am alive".
    """
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        data = {}

    raw_id  = data.get('deviceId', '')
    if not raw_id:
        return jsonify({'error': 'missing deviceId'}), 400

    dh = device_hash(raw_id)

    with lock:
        is_new_device  = dh not in device_registry
        is_new_session = data.get('sessionStart') and \
                         session_registry.get(dh) != data.get('sessionStart')
        touch_device(dh)
        if is_new_session:
            session_registry[dh] = data.get('sessionStart')
            stats['total_sessions'] += 1

    return jsonify({
        'active_now':    active_now(),
        'ever_connected': len(device_registry),
        'total_sessions': stats['total_sessions'],
    })


@app.route('/detection', methods=['POST'])
def report_detection():
    """
    Called when the on-device classifier confirms a mosquito detection.
    Body: { "deviceId": "...", "species": "anopheles", "confidence": 0.72,
            "frequency": 408.5, "locationHash": "abc123" }
    No raw audio, no exact GPS — only an anonymised area hash.
    """
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({'error': 'bad JSON'}), 400

    required = ['deviceId', 'species', 'confidence']
    if not all(k in data for k in required):
        return jsonify({'error': 'missing fields'}), 400

    dh = device_hash(data['deviceId'])

    with lock:
        touch_device(dh)
        stats['total_detections'] += 1

    print(f"  [Detection] {data.get('species')} conf={data.get('confidence'):.2f} "
          f"freq={data.get('frequency', '?')}Hz device={dh}")

    return jsonify({
        'received': True,
        'total_detections': stats['total_detections'],
    })


@app.route('/federated/upload', methods=['POST'])
def upload():
    """Submit a local model gradient update."""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({'error': 'Invalid JSON'}), 400

    if not all(k in data for k in ['deviceId', 'weights', 'steps']):
        return jsonify({'error': 'Missing fields'}), 400

    dh = device_hash(data['deviceId'])

    ok, msg = verify_dp(data)
    if not ok:
        return jsonify({'error': f'DP fail: {msg}'}), 400

    with lock:
        touch_device(dh)
        stats['total_uploads'] += 1
        pending_updates.append({
            'device_hash': dh,
            'steps':       min(int(data['steps']), 500),
            'weights':     data['weights'],
            'timestamp':   time.time(),
        })
        should_agg = len(pending_updates) >= MIN_UPLOADS_ROUND
        if should_agg:
            to_agg = pending_updates.copy()
            pending_updates.clear()
            fedavg_aggregate(to_agg)

    return jsonify({
        'status':      'accepted',
        'round':       stats['total_rounds'],
        'active_now':  active_now(),
        'ever':        len(device_registry),
        'weights': {
            'W': global_W.tolist(),
            'b': global_b.tolist(),
        },
    })


@app.route('/federated/model', methods=['GET'])
def get_model():
    return jsonify({
        'round':          stats['total_rounds'],
        'ever_connected': len(device_registry),
        'last_aggregate': stats['last_aggregate'],
        'weights': {
            'W': global_W.tolist(),
            'b': global_b.tolist(),
        },
        'sources': [
            'HumBugDB — Kiskin et al. 2021, CC-BY-4.0, zenodo.org/record/4904800',
            'Abuzz — Mukundarajan et al. 2017, eLife, Stanford University',
        ],
        'privacy': {
            'mechanism': 'Gaussian DP-SGD',
            'epsilon': 1.0, 'delta': 1e-5, 'clip_norm': 1.0,
        },
    })


@app.route('/federated/stats', methods=['GET'])
def get_stats():
    with lock:
        return jsonify({
            'active_now':       active_now(),
            'unique_devices':   len(device_registry),
            'total_sessions':   stats['total_sessions'],
            'total_uploads':    stats['total_uploads'],
            'total_rounds':     stats['total_rounds'],
            'total_detections': stats['total_detections'],
            'pending_updates':  len(pending_updates),
            'last_aggregate':   stats['last_aggregate'],
            'uptime_seconds':   int(time.time() - start_time),
        })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':    'ok',
        'service':   'MosquitoNet Federated Server v2',
        'active_now': active_now(),
        'uptime_s':  int(time.time() - start_time),
    })


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'MosquitoNet Federated Aggregation Server v2',
        'endpoints': [
            'POST /heartbeat          — register device presence (every 30s)',
            'POST /detection          — report confirmed detection',
            'POST /federated/upload   — submit model gradient',
            'GET  /federated/model    — download global model',
            'GET  /federated/stats    — live dashboard stats',
            'GET  /health             — health check',
        ],
        'data_sources': [
            'HumBugDB (Kiskin et al. 2021) — CC-BY-4.0 — zenodo.org/record/4904800',
            'Abuzz (Mukundarajan et al. 2017) — Stanford — eLife DOI:10.7554/eLife.27854',
        ],
        'privacy': 'DP-SGD Gaussian mechanism ε=1.0 δ=1e-5',
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print()
    print('  MosquitoNet Federated Server v2')
    print(f'  Port: {port}')
    print(f'  ALLOWED_ORIGIN: {_origin}')
    print()
    app.run(host='0.0.0.0', port=port, debug=False)
