"""Concurrent writers (gunicorn runs 4 threads) and the hard caps on every
per-key structure an anonymous client can grow."""
import threading
import time

from conftest import detection, post, post_det

THREADS = 8
PER_THREAD = 40


def test_concurrent_posts_lose_and_duplicate_nothing(harness, srv):
    errors, acks = [], []
    barrier = threading.Barrier(THREADS)

    def worker(t):
        c = srv.app.test_client()
        barrier.wait()
        for i in range(PER_THREAD):
            # Each thread is its own device; alternate species so every post is a
            # new detection (same device+species within 60 s would merge).
            p = detection(deviceId=f'dev{t}-{i}', eventId=f't{t}-e{i}')
            for attempt in range(2):              # every event is also retried once
                r = post(c, '/detection', p)
                if r.status_code != 200:
                    errors.append(r.status_code)
                else:
                    acks.append((p['eventId'], r.get_json()['detection_id'],
                                 bool(r.get_json().get('duplicate'))))
            c.get('/heartbeat?deviceId=hb%d' % t)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(THREADS)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    n = THREADS * PER_THREAD
    assert errors == []
    by_event = {}
    for eid, det_id, dup in acks:
        by_event.setdefault(eid, set()).add(det_id)
    assert len(by_event) == n
    assert all(len(ids) == 1 for ids in by_event.values())        # retry → same id
    ids = sorted(next(iter(s)) for s in by_event.values())
    assert ids == list(range(1, n + 1))                             # unique, contiguous
    assert sum(1 for *_, dup in acks if dup) == n                   # every retry deduped
    assert srv.stats['total_detections'] == n
    lines = harness.log_lines()
    assert len(lines) == n and sorted(l['id'] for l in lines) == ids

    srv2 = harness.restart(srv)
    assert srv2.stats['total_detections'] == n and len(srv2.detection_log) == n
    assert srv2.app.test_client().get('/log?limit=5000').get_json()['returned'] == n


def test_concurrent_flushes_and_posts(harness, srv):
    stop = threading.Event()

    def flusher():
        while not stop.is_set():
            srv._flush_once()

    th = threading.Thread(target=flusher)
    th.start()
    try:
        c = srv.app.test_client()
        for i in range(100):
            assert post_det(c, deviceId=f'd{i}').status_code == 200
    finally:
        stop.set()
        th.join()
    srv._flush_once()
    srv2 = harness.restart(srv, graceful=False)
    assert srv2.stats['total_detections'] == 100


def test_idle_devices_and_recent_entries_are_swept(srv, client):
    client.get('/heartbeat?deviceId=old')
    post_det(client, deviceId='old')
    client.get('/heartbeat?deviceId=new')
    with srv.write_lock:
        assert next(iter(srv.device_registry)) == srv.dh('old')     # ordered by last-seen
        srv.device_registry[srv.dh('old')] -= srv.ACTIVE_SEC + 1
        for rec in srv.recent_log.values():
            rec['ts'] -= srv.RECENT_WINDOW + 1
    srv._flush_once()
    assert srv.active_now() == 1 and srv.recent_log == {}


def test_per_key_structures_are_capped(srv, client):
    srv.KNOWN_DEVICES_MAX = 5
    srv.ACTIVE_MAX = 4
    srv.SEEN_EVENTS_MAX = 6
    srv.DETECTION_CELLS_MAX = 3
    for i in range(10):
        assert post_det(client, deviceId=f'd{i}', lat=float(i), lng=float(i)).status_code == 200
    assert len(srv.known_devices) == 5
    assert len(srv.device_registry) == 4 and srv.active_now() == 4
    assert len(srv.seen_events) == 6
    assert len(srv.detection_cells) == 3
    # LRU: the most recently hit cells survive.
    assert all(k.startswith(('7.', '8.', '9.')) for k in srv.detection_cells)
    assert srv.stats['ever_devices'] == 10


def test_hot_cells_survive_eviction(srv, client):
    srv.DETECTION_CELLS_MAX = 2
    post_det(client, deviceId='hot0', lat=1.0, lng=1.0)
    for i in range(5):
        post_det(client, deviceId=f'cold{i}', lat=10.0 + i, lng=10.0)
        post_det(client, deviceId=f'hot{i + 1}', lat=1.0, lng=1.0)     # keeps getting hit
    assert any(k.startswith('1.000,1.000') for k in srv.detection_cells)
    hot = next(v for k, v in srv.detection_cells.items() if k.startswith('1.000,1.000'))
    assert hot['total'] == 6


def test_session_strings_are_bounded(srv, client):
    client.get('/heartbeat?deviceId=d&sess=' + 'x' * 3000)
    assert all(len(s) <= 32 for s in srv.session_registry.values())


def test_flusher_thread_runs(harness, monkeypatch):
    srv = harness.start()
    srv.FLUSH_INTERVAL = 0.05
    # The running thread already waits on the old interval; start a fresh one.
    srv._stop.set()
    time.sleep(0.01)
    srv._stop.clear()
    th = threading.Thread(target=srv._flusher, daemon=True)
    th.start()
    c = srv.app.test_client()
    post_det(c)
    deadline = time.time() + 5
    while srv._last_state_save is None and time.time() < deadline:
        time.sleep(0.02)
    srv._stop.set()
    th.join(timeout=2)
    assert srv._last_state_save is not None
