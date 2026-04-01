#!/usr/bin/env python3
"""
MosquitoNet Federated Server v9
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
    global detection_log, detection_cells, hotspot_cells, stats
    try:
        with open(STATE_FILE) as f:
            d = json.load(f)
        for k in stats:
            if k in d.get('stats', {}):
                stats[k] = d['stats'][k]
        next_detection_id[0] = d.get('next_id', 1)
        detection_cells = d.get('detection_cells', {})
        hotspot_cells   = d.get('hotspot_cells',   {})
        print(f'[State] loaded, total={stats["total_detections"]}')
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
RECENT_WINDOW     = 70     # seconds: conf update allowed within this window

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
    global_W = 0.3*global_W + 0.7*nW
    global_b = 0.3*global_b + 0.7*nb
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

    with write_lock:
        touch(h)

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
                        _rewrite_log()   # rewrite under same lock — no race possible
                        _save_state()
                        break
            return jsonify({'received': True, 'updated': True,
                            'detection_id': rec['det_id'], **full_stats()})

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

    print(f'[Det #{det_id}] {sp_name} conf={conf:.3f} freq={freq:.1f}Hz '
          f'lat={entry["lat"]} lng={entry["lng"]}')
    return jsonify({'received': True, 'detection_id': det_id, **full_stats()})


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
    d = request.get_json(force=True, silent=True) or {}
    if not all(k in d for k in ['deviceId','weights','steps']):
        return jsonify({'error':'missing'}),400
    h = dh(d['deviceId'])
    with write_lock:
        touch(h); stats['total_uploads']+=1
        pending_updates.append({'steps':min(int(d.get('steps',1)),500),'weights':d['weights']})
        if len(pending_updates)>=MIN_UPLOADS:
            fedavg(pending_updates.copy()); pending_updates.clear()
    return jsonify({'status':'accepted','weights':{'W':global_W.tolist(),'b':global_b.tolist()},**full_stats()})

@app.route('/federated/model',  methods=['GET'])
def model(): return jsonify({'round':stats['total_rounds'],'weights':{'W':global_W.tolist(),'b':global_b.tolist()}})

@app.route('/federated/stats',  methods=['GET'])
def get_stats():
    with write_lock: return jsonify(full_stats())

@app.route('/health', methods=['GET'])
def health(): return jsonify({'status':'ok','service':'MosquitoNet v9',
                               'data_dir':DATA_DIR,'log_file':LOG_FILE,**full_stats()})

@app.route('/', methods=['GET'])
def index():
    return jsonify({'service':'MosquitoNet v9','data_dir':DATA_DIR,
                    'endpoints':['GET /heartbeat','POST /detection',
                                 'GET /log','GET /log.txt','GET /hotspots',
                                 'GET /federated/stats','GET /health']})

load_all()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f'\nMosquitoNet v9 — DATA_DIR={DATA_DIR}  port={port}\n')
    app.run(host='0.0.0.0', port=port, debug=False)
