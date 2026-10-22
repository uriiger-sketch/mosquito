"""Idempotency, the 60 s confidence-update window, durability across graceful
and hard restarts, corrupted-file recovery and storage-failure behaviour."""
import errno
import json
import os

import pytest

from conftest import detection, post, post_det


def test_same_event_id_is_acknowledged_once(client, harness):
    p = detection(eventId='evt-1')
    first = post(client, '/detection', p).get_json()
    again = post(client, '/detection', p).get_json()
    assert first['received'] and not first.get('duplicate')
    assert again == {**again, 'received': True, 'duplicate': True,
                     'detection_id': first['detection_id']}
    assert again['total_detections'] == 1
    assert len(harness.log_lines()) == 1


def test_redetection_within_window_updates_confidence(client, harness, srv):
    a = post_det(client, confidence=0.7, eventId='e1').get_json()
    b = post_det(client, confidence=0.9, eventId='e2').get_json()        # same device+species
    assert b['updated'] is True and b['detection_id'] == a['detection_id']
    assert b['total_detections'] == 1
    row = client.get('/log').get_json()['detections'][0]
    assert row['conf'] == 0.9 and 'conf_updated' in row
    # Lower confidence: still merged, conf unchanged.
    c = post_det(client, confidence=0.6, eventId='e3').get_json()
    assert c['updated'] is True
    assert client.get('/log').get_json()['detections'][0]['conf'] == 0.9
    # Retrying any of the merged events is a duplicate, not a new row.
    assert post_det(client, confidence=0.6, eventId='e3').get_json()['duplicate'] is True
    ops = [l for l in harness.log_lines() if l.get('op') == 'conf']
    assert [o['eid'] for o in ops] == ['e2', 'e3'] and ops[0]['conf'] == 0.9
    # Different species from the same device is a separate detection.
    d = post_det(client, species='culex', eventId='e4').get_json()
    assert d['detection_id'] == 2 and not d.get('updated')


def test_window_expiry_creates_new_row(client, srv):
    post_det(client, eventId='e1')
    for rec in srv.recent_log.values():
        rec['ts'] -= srv.RECENT_WINDOW + 1
    assert post_det(client, eventId='e2').get_json()['detection_id'] == 2


def test_graceful_restart_recovers_everything(harness, srv, client):
    post(client, '/heartbeat?deviceId=x', {})
    client.get('/heartbeat?deviceId=d1&sess=1')
    post_det(client, eventId='e1', deviceId='d1')
    post_det(client, eventId='e2', deviceId='d2', confidence=0.7)
    post_det(client, eventId='e3', deviceId='d2', confidence=0.95)          # conf update of #2
    for i in range(3):
        post(client, '/federated/upload', {'deviceId': f'f{i}', 'steps': 20,
                                           'weights': {'W': [[1.0] * 4] * 4, 'b': [0.0] * 4}})
    W_before = srv.global_W.tolist()

    srv2 = harness.restart(srv, graceful=True)
    c2 = srv2.app.test_client()
    s = c2.get('/federated/stats').get_json()
    assert s['total_detections'] == 2 and s['total_rounds'] == 1 and s['total_uploads'] == 3
    assert s['total_sessions'] == 1
    assert srv2.global_W.tolist() == W_before
    log = c2.get('/log').get_json()
    assert [d['id'] for d in log['detections']] == [2, 1]
    assert log['detections'][0]['conf'] == 0.95 and 'conf_updated' in log['detections'][0]
    # Idempotency survives the restart, and ids continue.
    assert post(c2, '/detection', detection(eventId='e3', deviceId='d2')).get_json()['duplicate']
    assert post_det(c2, eventId='e4', deviceId='d3').get_json()['detection_id'] == 3


def test_hard_kill_self_heals_from_the_log(harness, srv, client):
    """SIGKILL/OOM: no final state save. The append-only log is the truth."""
    post_det(client, eventId='e1', deviceId='d1')
    srv._flush_once()                               # state file now says total=1
    post_det(client, eventId='e2', deviceId='d2')
    post_det(client, eventId='e3', deviceId='d3')   # …but the log has 3
    srv2 = harness.restart(srv, graceful=False)
    c2 = srv2.app.test_client()
    assert c2.get('/federated/stats').get_json()['total_detections'] == 3
    # e3 was never in a saved state file — only in the log — and is still a duplicate.
    assert post_det(c2, eventId='e3', deviceId='d3').get_json()['duplicate'] is True
    assert post_det(c2, eventId='e5', deviceId='d5').get_json()['detection_id'] == 4


def test_corrupt_state_file_is_quarantined_and_rebuilt(harness, srv, client):
    srv.HOTSPOT_THRESHOLD = 2
    for i in range(2):
        post_det(client, eventId=f'e{i}', deviceId=f'd{i}')
    srv._shutdown()
    with open(harness.state_path, 'w') as f:
        f.write('{"stats": {"total_detec')          # truncated mid-write
    srv2 = harness.start()
    corrupt = [n for n in os.listdir(harness.data_dir) if '.corrupt-' in n]
    assert len(corrupt) == 1                        # evidence kept, not overwritten
    c2 = srv2.app.test_client()
    s = c2.get('/federated/stats').get_json()
    assert s['total_detections'] == 2
    cells = srv2.detection_cells
    assert len(cells) == 1 and next(iter(cells.values()))['total'] == 2   # recounted from log
    assert post_det(c2, eventId='e1', deviceId='d1').get_json()['duplicate'] is True
    srv2._flush_once()
    with open(harness.state_path) as f:
        assert json.load(f)['stats']['total_detections'] == 2           # healthy again


@pytest.mark.parametrize('state', [
    [], 'null', {'stats': {'total_detections': 'abc', 'total_rounds': -4, 'last_aggregate': 7},
                 'next_id': 'x', 'detection_cells': [1, 2], 'hotspot_cells': {'k': 'v'},
                 'seen_events': {'a': 'b', 'ok': 3, 'x' * 100: 1},
                 'global_W': [[1e300] * 4] * 4, 'global_b': [0] * 4},
])
def test_mistyped_state_is_ignored_field_by_field(harness, state):
    os.makedirs(harness.data_dir, exist_ok=True)
    with open(harness.state_path, 'w') as f:
        f.write(state if isinstance(state, str) else json.dumps(state))
    srv = harness.start()
    c = srv.app.test_client()
    assert post_det(c).status_code == 200             # used to be a 500 on every detection
    assert c.get('/hotspots').status_code == 200
    s = c.get('/federated/stats').get_json()
    assert s['total_detections'] == 1 and s['total_rounds'] >= 0
    assert abs(srv.global_W).max() < 10                # poisoned persisted model not adopted
    if isinstance(state, dict):
        assert 'ok' in srv.seen_events and 'a' not in srv.seen_events
        assert all(len(k) <= 64 for k in srv.seen_events)


def test_torn_tail_and_garbage_lines_are_survivable(harness, srv, client):
    post_det(client, eventId='e1', deviceId='d1')
    srv._shutdown()
    with open(harness.log_path, 'a') as f:
        f.write('not json\n[1,2]\n{"id": "seven"}\n{"op": "future", "id": 9}\n')
        f.write('{"id": 2, "species": "culex", "conf"')      # torn: no newline
    srv2 = harness.start()
    c2 = srv2.app.test_client()
    assert post_det(c2, eventId='e2', deviceId='d2').get_json()['detection_id'] == 2
    srv3 = harness.restart(srv2)
    log = srv3.app.test_client().get('/log').get_json()
    assert [d['id'] for d in log['detections']] == [2, 1]      # new record not glued to torn line


def test_legacy_rows_are_bounded_at_load(harness):
    os.makedirs(harness.data_dir, exist_ok=True)
    legacy = {'id': 1, 'ts': '2025-01-01T00:00:00Z', 'device': 'abc', 'lat': 'x', 'lng': None,
              'species': 'culex', 'name': 'N' * 100_000, 'disease': {'nested': 1}, 'freq': '450',
              'conf': 0.9, 'risk': 'HIGH', 'asymptomatic': 'yes', 'junk': 'J' * 10_000}
    with open(harness.log_path, 'w') as f:
        f.write(json.dumps(legacy) + '\n')
        f.write(json.dumps({'op': 'conf', 'id': 1, 'conf': 'high'}) + '\n')   # bad conf op ignored
    srv = harness.start()
    rec = srv.detection_log[0]
    assert len(rec['name']) == 64 and rec['disease'] is None and 'junk' not in rec
    assert rec['lat'] is None and rec['freq'] == 450.0 and rec['asymptomatic'] is False
    assert rec['conf'] == 0.9
    c = srv.app.test_client()
    assert c.get('/log').status_code == 200 and c.get('/log.txt').status_code == 200


def test_detection_log_is_capped_and_conf_ops_replayed(harness):
    os.makedirs(harness.data_dir, exist_ok=True)
    with open(harness.log_path, 'w') as f:
        for i in range(1, 501):
            f.write(json.dumps({'id': i, 'species': 'culex', 'conf': 0.6, 'eid': f'e{i}'}) + '\n')
            f.write(json.dumps({'op': 'conf', 'id': i, 'conf': 0.61, 'at': 't'}) + '\n')
    srv = harness.start(DETECTION_LOG_MAX='100', SEEN_EVENTS_MAX='150')
    assert len(srv.detection_log) == 100 and srv.detection_log[0]['id'] == 401
    assert all(e['conf'] == 0.61 and e['conf_updated'] == 't' for e in srv.detection_log)
    assert len(srv.seen_events) == 150 and 'e500' in srv.seen_events and 'e1' not in srv.seen_events
    assert srv.next_detection_id[0] == 501 and srv.stats['total_detections'] == 500


def test_disk_full_returns_503_and_changes_nothing(srv, client):
    if not os.path.exists('/dev/full'):
        pytest.skip('/dev/full not available')
    post_det(client, eventId='e1', deviceId='d1')
    with srv.write_lock:
        good_fd = srv._log_fd
        srv._log_fd = os.open('/dev/full', os.O_WRONLY)     # every write → ENOSPC
    r = post_det(client, eventId='e2', deviceId='d2')
    assert r.status_code == 503 and r.headers['Retry-After'] == '30'
    assert r.get_json() == {'received': False, 'reason': 'storage unavailable'}
    h = client.get('/health').get_json()
    assert h['status'] == 'degraded' and h['writable'] is False and h['counters']['write_errors'] == 1
    assert h['total_detections'] == 1 and 'e2' not in srv.seen_events
    # A conf-update needing a write is refused the same way.
    assert post_det(client, eventId='e3', deviceId='d1', confidence=0.99).status_code == 503
    with srv.write_lock:
        os.close(srv._log_fd)
        srv._log_fd = good_fd
    r = post_det(client, eventId='e2', deviceId='d2')      # the client's retry succeeds
    assert r.status_code == 200 and r.get_json()['detection_id'] == 2
    assert client.get('/health').get_json()['status'] == 'ok'


def test_partial_write_is_truncated(srv, client, harness, monkeypatch):
    post_det(client, eventId='e1', deviceId='d1')
    real_write = os.write
    calls = {'n': 0}

    def flaky(fd, data):
        if fd == srv._log_fd:
            calls['n'] += 1
            if calls['n'] == 1:
                return real_write(fd, bytes(data[:10]))     # short write…
            raise OSError(errno.ENOSPC, 'No space left on device')   # …then disk full
        return real_write(fd, data)

    monkeypatch.setattr(srv.os, 'write', flaky)
    assert post_det(client, eventId='e2', deviceId='d2').status_code == 503
    monkeypatch.setattr(srv.os, 'write', real_write)
    assert post_det(client, eventId='e2', deviceId='d2').status_code == 200
    lines = harness.log_lines()                      # every line parses: no torn fragment
    assert [l['id'] for l in lines] == [1, 2]


def test_failed_state_save_is_retried(srv, client, monkeypatch):
    post_det(client)
    real_replace = os.replace

    def broken_replace(*_a):
        raise OSError(errno.EIO, 'I/O error')

    monkeypatch.setattr(srv.os, 'replace', broken_replace)
    srv._flush_once()
    assert srv._dirty is True                        # not forgotten
    monkeypatch.setattr(srv.os, 'replace', real_replace)
    srv._flush_once()
    assert srv._dirty is False and srv._last_state_save is not None


def test_files_are_private(srv, client, harness):
    post_det(client)
    srv._flush_once()
    for p in (harness.log_path, harness.state_path):
        assert os.stat(p).st_mode & 0o077 == 0, p


def test_second_process_on_same_data_dir_is_flagged(harness, srv):
    from conftest import _import_server
    other = _import_server('federated_server_second')
    try:
        assert srv.single_instance is True
        assert other.single_instance is False
        assert other.app.test_client().get('/health').get_json()['single_instance'] is False
    finally:
        other._shutdown(flush=False)


def test_unopenable_lock_file_is_not_reported_as_a_second_writer(harness):
    os.makedirs(os.path.join(harness.data_dir, '.mosquitonet.lock'))   # open() → EISDIR
    srv = harness.start()
    assert srv.single_instance is None        # unknown — not a phantom second worker
    assert srv.app.test_client().get('/health').get_json()['single_instance'] is None


def test_exit_flush_is_idempotent_and_writes_state(srv, client, harness):
    post_det(client)
    srv._flush_on_exit()
    srv._flush_on_exit()                              # second call: no-op
    with open(harness.state_path) as f:
        assert json.load(f)['stats']['total_detections'] == 1
