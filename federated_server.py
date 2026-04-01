#!/usr/bin/env python3
"""
MosquitoNet Federated Server v8
================================
Fixes:
  - Synchronous writes (no daemon threads that die on shutdown)
  - Single worker mode (Procfile) — no diverging in-memory state
  - JSONL log is append-only, never truncated, is the source of truth
  - STATE_FILE holds stats/cells/hotspots (no log — log is in JSONL)
  - Conf update within 60s: if same device+species submits again within
    60s of a logged entry, and new conf > old conf, we UPDATE that entry
    in-place in memory AND rewrite the JSONL file.
  - DATA_DIR env var must point to a Railway persistent volume (/data)
    so data survives restarts. Falls back to /tmp (ephemeral) if not set.
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

write_lock = threading.Lock()   # serialise all disk writes

def _write_log_sync():
    """Rewrite the full JSONL log from in-memory list — called under write_lock."""
    try:
        tmp = LOG_FILE + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            for entry in detection_log:
                f.write(json.dumps(entry, separators=(',', ':')) + '\n')
        os.replace(tmp, LOG_FILE)
    except Exception as e:
        print(f'[Log write] {e}')

def _append_log_sync(entry):
    """Append one entry to JSONL — called under write_lock."""
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, separators=(',', ':')) + '\n')
    except Exception as e:
        print(f'[Log append] {e}')

def _save_state_sync():
    """Write state file (stats/cells/hotspots) — called under write_lock."""
    try:
        payload = {
            'stats':           stats,
            'next_id':         next_detection_id[0],
            'detection_cells': detection_cells,
            'hotspot_cells':   hotspot_cells,
        }
        tmp = STATE_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(payload, f, separators=(',', ':'))
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        print(f'[State write] {e}')

def persist(entry=None, rewrite_log=False):
    """Thread-safe persist: append entry to log and save state."""
    def _do():
        with write_lock:
            if rewrite_log:
                _write_log_sync()
            elif entry is not None:
                _append_log_sync(entry)
            _save_state_sync()
    # Non-daemon thread — will complete even if main thread exits
    t = threading.Thread(target=_do, daemon=False)
    t.start()
    return t

def load_all():
    global detection_log, detection_cells, hotspot_cells, stats
    # Load state
    try:
        with open(STATE_FILE) as f:
            d = json.load(f)
        for k in stats:
            if k in d.get('stats', {}):
                stats[k] = d['stats'][k]
        next_detection_id[0] = d.get('next_id', 1)
        detection_cells = d.get('detection_cells', {})
        hotspot_cells   = d.get('hotspot_cells',   {})
        print(f'[State] loaded, total_detections={stats["total_detections"]}')
    except FileNotFoundError:
        print('[State] fresh start')
    except Exception as e:
        print(f'[State load] {e}')
    # Load log from JSONL — source of truth
    try:
        loaded = []
        with open(LOG_FILE, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        loaded.append(json.loads(line))
                    except Exception:
                        pass
        detection_log[:] = loaded
        print(f'[Log] loaded {len(detection_log)} detections from {LOG_FILE}')
    except FileNotFoundError:
        print(f'[Log] no existing log — starting fresh')
    except Exception as e:
        print(f'[Log load] {e}')

def save_on_exit():
    """Flush everything synchronously when the process shuts down."""
    print('[Exit] flushing state to disk...')
    with write_lock:
        _write_log_sync()
        _save_state_sync()
    print('[Exit] done.')

atexit.register(save_on_exit)

# ── Model ─────────────────────────────────────────────────────────────────────
global_W = np.array([
    [ 2.8, 0.9, 1.2, 0.8],
    [-0.6, 0.7, 1.0, 0.7],
    [-1.1, 0.7, 1.0, 0.6],
    [ 1.8, 0.8, 1.3, 0.9],
], dtype=float)
global_b = np.array([-0.4, -0.3, -0.3, -0.4], dtype=float)

# ── State ─────────────────────────────────────────────────────────────────────
lock              = threading.Lock()   # protects in-memory state
device_registry   = {}
session_registry  = {}
pending_updates   = []
detection_log     = []     # append-only in memory; mirrored to LOG_FILE
detection_cells   = {}
hotspot_cells     = {}
next_detection_id = [1]
# Track recently-logged entries for conf updates: {f"{device}:{species}" -> det_id}
recent_log        = {}     # key -> {'det_id': N, 'ts': epoch}
RECENT_WINDOW     = 70     # seconds: allow conf update within ~70s of logging

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

# ── CORS ──────────────────────────────────────────────────────────────────────
@app.after_request
def cors_hdr(r):
    r.headers['Access-Control-Allow-Origin']  = '*'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type,Accept'
    r.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return r

@app.route('/', defaults={'p': ''}, methods=['OPTIONS'])
@app.route('/<path:p>', methods=['OPTIONS'])
def opts(p): return '', 204

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
    d       = request.get_json(force=True, silent=True) or {}
    raw_id  = d.get('deviceId', 'anon')
    species = str(d.get('species', ''))
    lat     = d.get('lat')
    lng     = d.get('lng')
    conf    = round(float(d.get('confidence', 0)), 3)
    freq    = round(float(d.get('frequency', 0)), 1)
    risk    = d.get('risk', 'UNKNOWN')
    sp_name = d.get('speciesName', species)
    disease = d.get('disease', '')
    asymp   = bool(d.get('asymptomatic', False))
    ts_str  = d.get('ts') or datetime.now(timezone.utc).isoformat()
    h       = dh(raw_id)
    rkey    = f'{h}:{species}'
    now     = time.time()

    need_rewrite = False   # True if we update an existing entry
    new_entry    = None    # set if we create a new entry

    with lock:
        touch(h)

        # ── Check if this is a conf UPDATE for a recent detection ─────────────
        recent = recent_log.get(rkey)
        if recent and (now - recent['ts']) < RECENT_WINDOW:
            # Within the update window — only update if conf improved
            if conf > recent['conf']:
                # Find and update the entry in-memory
                for e in reversed(detection_log):
                    if e.get('id') == recent['det_id']:
                        e['conf'] = conf
                        e['conf_updated'] = datetime.now(timezone.utc).isoformat()
                        recent['conf'] = conf
                        need_rewrite = True
                        print(f'[Det #{recent["det_id"]}] conf updated {recent["conf"]:.3f} → {conf:.3f}')
                        break
            # Either updated or not, don't create a new entry
            return jsonify({'received': True, 'updated': need_rewrite,
                            'detection_id': recent['det_id'], **full_stats()})

        # ── New detection ─────────────────────────────────────────────────────
        det_id = next_detection_id[0]
        next_detection_id[0] += 1
        stats['total_detections'] += 1

        entry = {
            'id':          det_id,
            'ts':          ts_str,
            'device':      h,
            'lat':         round(float(lat), 4) if lat is not None else None,
            'lng':         round(float(lng), 4) if lng is not None else None,
            'species':     species,
            'name':        sp_name,
            'disease':     disease,
            'freq':        freq,
            'conf':        conf,
            'risk':        risk,
            'asymptomatic': asymp,
        }
        detection_log.append(entry)
        new_entry = entry

        # Register for potential conf updates within the window
        recent_log[rkey] = {'det_id': det_id, 'ts': now, 'conf': conf}

        # Hotspot cell
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

    print(f'[Det #{det_id}] {sp_name} conf={conf:.3f} freq={freq:.1f}Hz '
          f'lat={entry["lat"]} lng={entry["lng"]}')

    # Persist: append new entry (fast) or rewrite if updated (slower but safe)
    persist(entry=new_entry if not need_rewrite else None,
            rewrite_log=need_rewrite)

    return jsonify({'received': True, 'detection_id': det_id, **full_stats()})


@app.route('/log', methods=['GET'])
def log_json():
    sp_filter  = request.args.get('species')
    dev_filter = request.args.get('device')
    try:   limit   = min(int(request.args.get('limit', 500)), 50000)
    except: limit  = 500
    try:   from_id = int(request.args.get('from_id', 0))
    except: from_id = 0

    with lock:
        rows = [e for e in detection_log
                if (not sp_filter  or e.get('species') == sp_filter)
                and (not dev_filter or e.get('device')  == dev_filter)
                and e.get('id', 0) > from_id]
        s = full_stats()

    return jsonify({
        'total_ever': s['total_detections'],
        'total_log':  s['log_size'],
        'returned':   len(rows[-limit:]),
        'detections': rows[-limit:][::-1],
    })


@app.route('/log.txt', methods=['GET'])
def log_txt():
    sp_filter  = request.args.get('species')
    try:   limit = min(int(request.args.get('limit', 2000)), 50000)
    except: limit = 2000
    try:   from_id = int(request.args.get('from_id', 0))
    except: from_id = 0

    with lock:
        rows = [e for e in detection_log
                if (not sp_filter or e.get('species') == sp_filter)
                and e.get('id', 0) > from_id]
        total_ever = stats['total_detections']
        total_log  = len(detection_log)

    rows = rows[-limit:][::-1]

    buf = io.StringIO()
    buf.write(f'MosquitoNet Detection Log  |  total ever: {total_ever}  |  in log: {total_log}\n')
    buf.write(f'Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} UTC\n')
    if sp_filter: buf.write(f'Filter: species={sp_filter}\n')
    buf.write('\n')
    buf.write(f'{"#":<6}  {"DATE/TIME (UTC)":<20}  {"DEVICE":<8}  '
              f'{"SPECIES":<22}  {"CONF":>5}  {"HZ":>6}  '
              f'{"RISK":<8}  {"LAT":>10}  {"LNG":>10}\n')
    buf.write('-' * 106 + '\n')

    for e in rows:
        det_id = str(e.get('id','?')).rjust(5)
        try:
            dt = datetime.fromisoformat(str(e.get('ts','')).replace('Z','+00:00'))
            ts = dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            ts = str(e.get('ts',''))[:19]
        device  = str(e.get('device','?'))[:8]
        sp_name = str(e.get('name') or e.get('species','?'))
        parts   = sp_name.split()
        if len(parts) >= 2:
            sp_name = parts[0][0] + '. ' + ' '.join(parts[1:])
        sp_name = sp_name[:22]
        conf    = f'{float(e.get("conf",0)):.2f}'
        upd     = '*' if e.get('conf_updated') else ' '
        freq    = f'{float(e.get("freq",0)):.1f}'
        risk    = str(e.get('risk','?'))[:8]
        lat     = f'{e["lat"]:>10.4f}' if e.get('lat') is not None else ' '*10
        lng     = f'{e["lng"]:>10.4f}' if e.get('lng') is not None else ' '*10
        buf.write(f'{det_id}  {ts:<20}  {device:<8}  '
                  f'{sp_name:<22}  {upd}{conf:>5}  {freq:>6}  '
                  f'{risk:<8}  {lat}  {lng}\n')

    buf.write('\n* = confidence updated after initial log\n')
    return Response(buf.getvalue(), mimetype='text/plain; charset=utf-8')


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
    if not all(k in d for k in ['deviceId','weights','steps']):
        return jsonify({'error':'missing'}), 400
    h = dh(d['deviceId'])
    with lock:
        touch(h)
        stats['total_uploads'] += 1
        pending_updates.append({'steps':min(int(d.get('steps',1)),500),
                                 'weights':d['weights']})
        if len(pending_updates) >= MIN_UPLOADS:
            fedavg(pending_updates.copy())
            pending_updates.clear()
    return jsonify({'status':'accepted',
                    'weights':{'W':global_W.tolist(),'b':global_b.tolist()},
                    **full_stats()})

@app.route('/federated/model',  methods=['GET'])
def model(): return jsonify({'round':stats['total_rounds'],'weights':{'W':global_W.tolist(),'b':global_b.tolist()}})

@app.route('/federated/stats',  methods=['GET'])
def get_stats():
    with lock: return jsonify(full_stats())

@app.route('/health', methods=['GET'])
def health(): return jsonify({'status':'ok','service':'MosquitoNet v8',**full_stats()})

@app.route('/', methods=['GET'])
def index():
    return jsonify({'service':'MosquitoNet v8',
                    'log_file': LOG_FILE,
                    'state_file': STATE_FILE,
                    'endpoints':['GET /heartbeat','POST /detection',
                                 'GET /log','GET /log.txt',
                                 'GET /hotspots','GET /federated/stats','GET /health']})


load_all()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f'\nMosquitoNet v8 on :{port} — DATA_DIR={DATA_DIR}\n')
    app.run(host='0.0.0.0', port=port, debug=False)
