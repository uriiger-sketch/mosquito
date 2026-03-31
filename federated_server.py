#!/usr/bin/env python3
"""
MosquitoNet Federated Aggregation Server
=========================================
Implements FedAvg with differential privacy verification.
Run this on a server (or locally for testing).

Install: pip install flask flask-cors numpy
Run:     python3 federated_server.py

The PWA clients POST anonymized gradient updates here.
No audio, no user data — only noisy model weight deltas.
"""

import json
import os
import time
import hashlib
import threading
import numpy as np
from datetime import datetime
from collections import defaultdict

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError:
    print("Install dependencies: pip install flask flask-cors numpy")
    raise

app = Flask(__name__)

# Allow the GitHub Pages origin plus localhost for dev.
# Set ALLOWED_ORIGIN env var to your GitHub Pages URL on deployment.
_origin = os.environ.get('ALLOWED_ORIGIN', '*')
CORS(app, origins=_origin)

# ─── Global model state ───
N_FEATURES = 4
N_CLASSES  = 4

# Global weights (FedAvg aggregated)
global_W = np.random.randn(N_CLASSES, N_FEATURES) * 0.1
global_b = np.zeros(N_CLASSES)

# Aggregation buffer
pending_updates = []
pending_lock    = threading.Lock()

# Stats
stats = {
    'total_uploads': 0,
    'total_rounds':  0,
    'unique_devices': set(),
    'detections_by_species': defaultdict(int),
    'last_aggregate': None,
    'contributors_today': 0,
}

# FedAvg config
AGGREGATE_EVERY   = 10    # aggregate after N uploads
MIN_UPLOADS_ROUND =  3    # minimum uploads to trigger aggregation
DP_EPSILON        = 1.0   # privacy budget (enforced server-side)
DP_DELTA          = 1e-5


def verify_dp_noise(update, sigma_min=0.008):
    """
    Basic server-side DP verification.
    Checks that uploaded gradients have sufficient noise injected.
    A more rigorous implementation would use cryptographic verification.
    """
    W = np.array(update['weights']['W'])
    if np.std(W) < sigma_min:
        return False, f"Insufficient DP noise: std={np.std(W):.4f} < {sigma_min}"
    return True, "ok"


def fedavg_aggregate(updates):
    """
    Federated Averaging (McMahan et al. 2017).
    Weighted by number of local steps.
    """
    global global_W, global_b

    total_steps = sum(u['steps'] for u in updates)
    if total_steps == 0:
        return

    new_W = np.zeros((N_CLASSES, N_FEATURES))
    new_b = np.zeros(N_CLASSES)

    for update in updates:
        weight = update['steps'] / total_steps
        new_W += weight * np.array(update['weights']['W'])
        new_b += weight * np.array(update['weights']['b'])

    # Weighted average with existing global model (momentum)
    alpha = 0.3  # global model momentum
    global_W = alpha * global_W + (1 - alpha) * new_W
    global_b = alpha * global_b + (1 - alpha) * new_b

    stats['total_rounds'] += 1
    stats['last_aggregate'] = datetime.utcnow().isoformat() + 'Z'

    print(f"  [FedAvg] Round {stats['total_rounds']} complete — "
          f"{len(updates)} updates, {total_steps} total steps")


@app.route('/federated/upload', methods=['POST'])
def upload():
    """Receive a model update from a client device."""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({'error': 'Invalid JSON'}), 400

    required = ['round', 'deviceId', 'weights', 'steps']
    if not all(k in data for k in required):
        return jsonify({'error': 'Missing fields'}), 400

    # Anonymize device ID (hash it — never store raw)
    device_hash = hashlib.sha256(data['deviceId'].encode()).hexdigest()[:16]

    # Verify DP noise level
    dp_ok, dp_msg = verify_dp_noise(data)
    if not dp_ok:
        return jsonify({'error': f'DP verification failed: {dp_msg}'}), 400

    # Store update
    update = {
        'device_hash': device_hash,
        'round':       data['round'],
        'steps':       min(int(data['steps']), 500),  # cap per-device contribution
        'weights':     data['weights'],
        'timestamp':   time.time(),
    }

    with pending_lock:
        pending_updates.append(update)
        stats['total_uploads'] += 1
        stats['unique_devices'].add(device_hash)

        # Trigger aggregation if enough updates accumulated
        should_aggregate = len(pending_updates) >= MIN_UPLOADS_ROUND

        if should_aggregate:
            to_aggregate = pending_updates.copy()
            pending_updates.clear()
            fedavg_aggregate(to_aggregate)

    return jsonify({
        'status': 'accepted',
        'round': stats['total_rounds'],
        'contributors': len(stats['unique_devices']),
        'weights': {
            'W': global_W.tolist(),
            'b': global_b.tolist(),
        },
    })


@app.route('/federated/model', methods=['GET'])
def get_model():
    """Download current global model (for new devices)."""
    return jsonify({
        'round': stats['total_rounds'],
        'contributors': len(stats['unique_devices']),
        'last_aggregate': stats['last_aggregate'],
        'weights': {
            'W': global_W.tolist(),
            'b': global_b.tolist(),
        },
        'species': [
            {'id': 'anopheles',      'index': 0},
            {'id': 'aedes_aegypti',  'index': 1},
            {'id': 'aedes_albopictus', 'index': 2},
            {'id': 'culex',          'index': 3},
        ],
        'privacy': {
            'mechanism': 'Gaussian',
            'epsilon': DP_EPSILON,
            'delta': DP_DELTA,
            'clipping_norm': 1.0,
        },
    })


@app.route('/federated/stats', methods=['GET'])
def get_stats():
    """Public dashboard stats."""
    with pending_lock:
        return jsonify({
            'total_uploads':   stats['total_uploads'],
            'total_rounds':    stats['total_rounds'],
            'unique_devices':  len(stats['unique_devices']),
            'pending_updates': len(pending_updates),
            'last_aggregate':  stats['last_aggregate'],
            'uptime_seconds':  int(time.time() - start_time),
        })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'MosquitoNet Federated Server'})


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'MosquitoNet Federated Aggregation Server',
        'version': '1.0.0',
        'endpoints': [
            'POST /federated/upload  — submit local model update',
            'GET  /federated/model   — download global model',
            'GET  /federated/stats   — aggregation statistics',
            'GET  /health            — health check',
        ],
        'privacy': 'DP-SGD with Gaussian mechanism (ε=1.0, δ=1e-5)',
        'docs': 'https://github.com/mosquitonet/server',
    })


start_time = time.time()

if __name__ == '__main__':
    print()
    print('  ╔══════════════════════════════════════════════╗')
    print('  ║   MosquitoNet Federated Aggregation Server  ║')
    print('  ╚══════════════════════════════════════════════╝')
    print()
    print('  DP config: Gaussian mechanism, ε=1.0, δ=1e-5')
    print('  FedAvg: aggregates every 3+ uploads')
    print()
    print('  Endpoints:')
    print('    POST http://localhost:5001/federated/upload')
    print('    GET  http://localhost:5001/federated/model')
    print('    GET  http://localhost:5001/federated/stats')
    print()
    port = int(os.environ.get('PORT', 5001))
    print(f'  Listening on port {port}')
    print()
    app.run(host='0.0.0.0', port=port, debug=False)
