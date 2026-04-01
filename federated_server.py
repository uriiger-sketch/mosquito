#!/usr/bin/env python3
"""
MosquitoNet Federated Server v5
================================
Every detection saved immediately to disk (background thread).
Detection log includes: ts, species, speciesName, disease, conf, freq,
risk, asymptomatic, lat, lng, device_hash.
/log endpoint returns full log for scientific analysis.
"""

import os, time, hashlib, threading, json
import numpy as np
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins='*', supports_credentials=False,
     allow_headers=['Content-Type','Accept'],
     methods=['GET','POST','OPTIONS'])

# ── Persistence ───────────────────────────────────────────────────────────────
DATA_DIR  = os.environ.get('DATA_DIR', '/tmp')
DATA_FILE = os.path.join(DATA_DIR, 'mosquitonet_v5.json')

def save_state():
    """Save to disk in a background thread — never blocks a request."""
    def _save():
        try:
            payload = {
                'stats':           stats,
                'detection_log':   detection_log[-20000:],
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
        detection_log   = d.get('detection_log', [])
        detection_cells = d.get('detection_cells', {})
        hotspot_cells   = d.get('hotspot_cells', {})
        print(f'[Load] {len(detection_log)} detections, '
              f'{stats["total_detections"]} total, '
              f'{len(hotspot_cells)} hotspots')
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
lock             = threading.Lock()
device_registry  = {}
session_registry = {}
pending_updates  = []
detection_log    = []
detection_cells  = {}
hotspot_cells    = {}
detection_cds    = {}  # f"{hash}:{species}" → last_report_ts

stats = {
    'total_uploads': 0, 'total_rounds': 0,
    'total_detections': 0, 'total_sessions': 0,
    'last_aggregate': None,
}

ACTIVE_SEC        = 90
MIN_UPLOADS       = 3
HOTSPOT_THRESHOLD = 100
MAX_LOG           = 50000
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
    nW = np.zeros_like(global_W); nb = np.zeros_like(global_b)
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
    print(f'[FedAvg] Round {stats["total_rounds"]}')

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
    d        = request.get_json(force=True, silent=True) or {}
    raw_id   = d.get('deviceId', 'anon')
    species  = str(d.get('species', ''))
    lat      = d.get('lat')
    lng      = d.get('lng')
    conf     = d.get('confidence', 0)
    freq     = d.get('frequency', 0)
    risk     = d.get('risk', 'UNKNOWN')
    sp_name  = d.get('speciesName', species)
    disease  = d.get('disease', '')
    asymp    = bool(d.get('asymptomatic', False))
    ts_str   = d.get('ts') or datetime.now(timezone.utc).isoformat()
    h        = dh(raw_id)
    cd_key   = f'{h}:{species}'
    now      = time.time()

    with lock:
        # Server-side 60s cooldown per device+species
        if now - detection_cds.get(cd_key, 0) < 60:
            return jsonify({'received': False, 'reason': 'cooldown', **full_stats()})

        detection_cds[cd_key] = now
        touch(h)
        stats['total_detections'] += 1

        # ── Full detection record ─────────────────────────────────────────
        entry = {
            'ts':           ts_str,
            'species':      species,
            'speciesName':  sp_name,
            'disease':      disease,
            'conf':         round(float(conf), 3),
            'freq':         round(float(freq), 1),
            'risk':         risk,
            'asymptomatic': asymp,
            'device':       h,
        }
        if lat is not None and lng is not None:
            try:
                entry['lat'] = round(float(lat), 4)
                entry['lng'] = round(float(lng), 4)
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
                    print(f'[Hotspot] {ck} → {detection_cells[ck]["total"]}')
            except Exception as e:
                print(f'[Cell] {e}')

        detection_log.append(entry)
        if len(detection_log) > MAX_LOG:
            detection_log.pop(0)

    print(f'[Det] {species} ({sp_name}) conf={conf:.3f} freq={freq:.1f}Hz '
          f'lat={lat} lng={lng} dev={h}')

    # Save after EVERY detection (background thread — does not block response)
    save_state()

    return jsonify({'received': True, **full_stats()})


@app.route('/hotspots', methods=['GET'])
def hotspots():
    with lock:
        hot  = [{'key': k, **v} for k, v in hotspot_cells.items()]
        near = [{'key': k, **v, 'approaching': True}
                for k, v in detection_cells.items()
                if k not in hotspot_cells and v['total'] >= 50]
    return jsonify({'hotspots': hot, 'approaching': near,
                    'threshold': HOTSPOT_THRESHOLD})


@app.route('/log', methods=['GET'])
def log():
    """Full detection log for scientific analysis."""
    sp    = request.args.get('species')
    risk  = request.args.get('risk')
    try:   limit = min(int(request.args.get('limit', 1000)), 10000)
    except: limit = 1000
    with lock:
        rows = [e for e in detection_log
                if (not sp   or e.get('species') == sp)
                and (not risk or e.get('risk') == risk)]
    return jsonify({
        'count':      len(rows),
        'total_log':  len(detection_log),
        'total_ever': stats['total_detections'],
        'detections': rows[-limit:],
    })


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
    return jsonify({
        'status':  'accepted',
        'weights': {'W': global_W.tolist(), 'b': global_b.tolist()},
        **full_stats(),
    })


@app.route('/federated/model', methods=['GET'])
def model():
    return jsonify({'round': stats['total_rounds'],
                    'weights': {'W': global_W.tolist(), 'b': global_b.tolist()}})


@app.route('/federated/stats', methods=['GET'])
def get_stats():
    with lock: return jsonify(full_stats())


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'MosquitoNet v5', **full_stats()})


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'MosquitoNet v5',
        'endpoints': [
            'GET  /heartbeat?deviceId=X&listening=1&sess=Y',
            'POST /detection  — {deviceId,species,speciesName,disease,confidence,frequency,risk,asymptomatic,ts,lat,lng}',
            'GET  /hotspots',
            'GET  /log?species=X&risk=Y&limit=N',
            'POST /federated/upload',
            'GET  /federated/stats',
            'GET  /health',
        ],
    })


# Load persisted state on startup (works for both gunicorn and direct run)
load_state()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f'\nMosquitoNet v5 on :{port}\n')
    app.run(host='0.0.0.0', port=port, debug=False)
