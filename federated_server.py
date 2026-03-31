#!/usr/bin/env python3
"""
MosquitoNet Federated Server v2.1
==================================
CORS: always allows all origins — data is public, not sensitive.
Tracks: active_now (90s window), unique_devices, total_sessions,
        total_uploads, total_rounds, total_detections.
"""

import json, os, time, hashlib, threading
import numpy as np
from datetime import datetime
from collections import defaultdict

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError:
    raise

app = Flask(__name__)

# ALWAYS allow all origins — this is public scientific data.
# No credentials, no PII, no auth. CORS should never be the blocker.
CORS(app,
     origins='*',
     supports_credentials=False,
     allow_headers=['Content-Type', 'Accept'],
     methods=['GET', 'POST', 'OPTIONS'])

# ── Model ─────────────────────────────────────────────────────────────────────
N_FEATURES = 4
N_CLASSES  = 4

global_W = np.array([
    [ 2.8, 0.9, 1.2, 0.8],
    [-0.6, 0.7, 1.0, 0.7],
    [-1.1, 0.7, 1.0, 0.6],
    [ 1.8, 0.8, 1.3, 0.9],
], dtype=float)
global_b = np.array([-0.4, -0.3, -0.3, -0.4], dtype=float)

# ── State ─────────────────────────────────────────────────────────────────────
lock             = threading.Lock()
device_registry  = {}   # hash → last_seen timestamp
session_registry = {}   # hash → session_start
pending_updates  = []

stats = {
    'total_uploads':    0,
    'total_rounds':     0,
    'total_detections': 0,
    'total_sessions':   0,
    'last_aggregate':   None,
}

ACTIVE_WINDOW_SEC = 90
MIN_UPLOADS_ROUND = 3
start_time = time.time()


# ── Helpers ───────────────────────────────────────────────────────────────────
def dh(raw_id):
    return hashlib.sha256(raw_id.encode()).hexdigest()[:16]

def active_now():
    cutoff = time.time() - ACTIVE_WINDOW_SEC
    return sum(1 for ts in device_registry.values() if ts >= cutoff)

def touch(h):
    device_registry[h] = time.time()

def full_stats():
    return {
        'active_now':       active_now(),
        'unique_devices':   len(device_registry),
        'ever_connected':   len(device_registry),
        'total_sessions':   stats['total_sessions'],
        'total_uploads':    stats['total_uploads'],
        'total_rounds':     stats['total_rounds'],
        'total_detections': stats['total_detections'],
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
    print(f"[FedAvg] Round {stats['total_rounds']} done")


# ── Handle CORS preflight globally ───────────────────────────────────────────
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
    """Register device as alive. Works as GET too for easy testing."""
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        data = {}

    raw_id = data.get('deviceId', request.args.get('deviceId', 'anonymous'))
    h = dh(raw_id)

    with lock:
        is_new_session = (data.get('sessionStart') and
                          session_registry.get(h) != str(data.get('sessionStart')))
        touch(h)
        if is_new_session:
            session_registry[h] = str(data.get('sessionStart'))
            stats['total_sessions'] += 1

    return jsonify(full_stats())


@app.route('/detection', methods=['POST'])
def report_detection():
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({'error': 'bad json'}), 400

    raw_id = data.get('deviceId', 'anonymous')
    h = dh(raw_id)

    with lock:
        touch(h)
        stats['total_detections'] += 1

    print(f"[Det] {data.get('species')} conf={data.get('confidence','?')} freq={data.get('frequency','?')}Hz")
    return jsonify({'received': True, **full_stats()})


@app.route('/federated/upload', methods=['POST'])
def upload():
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({'error': 'bad json'}), 400

    if not all(k in data for k in ['deviceId', 'weights', 'steps']):
        return jsonify({'error': 'missing fields'}), 400

    h = dh(data['deviceId'])

    with lock:
        touch(h)
        stats['total_uploads'] += 1
        pending_updates.append({
            'device_hash': h,
            'steps':       min(int(data.get('steps', 1)), 500),
            'weights':     data['weights'],
        })
        if len(pending_updates) >= MIN_UPLOADS_ROUND:
            fedavg(pending_updates.copy())
            pending_updates.clear()

    return jsonify({
        'status': 'accepted',
        'weights': {'W': global_W.tolist(), 'b': global_b.tolist()},
        **full_stats(),
    })


@app.route('/federated/model', methods=['GET'])
def get_model():
    return jsonify({
        'round': stats['total_rounds'],
        'weights': {'W': global_W.tolist(), 'b': global_b.tolist()},
    })


@app.route('/federated/stats', methods=['GET'])
def get_stats():
    with lock:
        return jsonify(full_stats())


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'MosquitoNet v2.1', **full_stats()})


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'MosquitoNet Federated Server v2.1',
        'endpoints': [
            'POST /heartbeat  — register presence (fires on app open)',
            'POST /detection  — report confirmed detection',
            'POST /federated/upload',
            'GET  /federated/stats',
            'GET  /health',
        ],
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f'\nMosquitoNet Server v2.1 on port {port}\n')
    app.run(host='0.0.0.0', port=port, debug=False)
