"""Every route: happy path + backward-compatible response shapes."""
import json

from conftest import FIRST_PARTY, detection, post, post_det, weights

STATS_KEYS = {'active_now', 'unique_devices', 'ever_connected', 'total_sessions',
              'total_uploads', 'total_rounds', 'total_detections', 'hotspot_count',
              'log_size', 'uptime_seconds'}


def test_index_lists_endpoints(client):
    r = client.get('/')
    assert r.status_code == 200
    body = r.get_json()
    assert body['service'].startswith('MosquitoNet')
    assert 'POST /detection' in body['endpoints'] and 'GET /health' in body['endpoints']


def test_health_fields(client, srv, harness):
    r = client.get('/health')
    assert r.status_code == 200
    h = r.get_json()
    # Fields the docs and the in-app "Test server" button rely on.
    for k in ('status', 'service', 'data_dir', 'log_file', 'python', 'writable', 'ephemeral',
              'min_conf', 'seen_events', 'log_cap', 'active_now', 'total_detections',
              'log_size', 'uptime_seconds'):
        assert k in h, k
    assert STATS_KEYS <= set(h)
    assert h['status'] == 'ok'
    assert h['writable'] is True
    assert h['data_dir'] == harness.data_dir
    # New in v13
    assert h['single_instance'] is True
    assert isinstance(h['disk_free_mb'], int) and h['disk_free_mb'] > 0
    assert set(h['counters']) >= {'rejected_detections', 'rate_limited', 'write_errors'}
    assert h['rate_limit_scale'] == 0
    assert h['client_ip'] == '127.0.0.1'


def test_health_client_ip_follows_trusted_proxy(client):
    h = client.get('/health', headers={'X-Forwarded-For': '203.0.113.9, 198.51.100.7'}).get_json()
    assert h['client_ip'] == '198.51.100.7'          # only the proxy-appended hop is trusted
    # Only major.minor — the full build string is a fingerprint.
    assert h['python'].count('.') == 1


def test_heartbeat_get_counts_device_and_session(client):
    r = client.get('/heartbeat?deviceId=dev1&listening=1&sess=111')
    assert r.status_code == 200
    s = r.get_json()
    assert set(s) == STATS_KEYS
    assert s['active_now'] == 1 and s['unique_devices'] == 1 and s['total_sessions'] == 1
    # Same session again: no new session; a new session id: +1.
    assert client.get('/heartbeat?deviceId=dev1&sess=111').get_json()['total_sessions'] == 1
    assert client.get('/heartbeat?deviceId=dev1&sess=222').get_json()['total_sessions'] == 2
    s = client.get('/heartbeat?deviceId=dev2').get_json()
    assert s['active_now'] == 2 and s['unique_devices'] == 2


def test_heartbeat_post_and_non_object_body(client):
    r = post(client, '/heartbeat', {'deviceId': 'x'})
    assert r.status_code == 200 and r.get_json()['active_now'] == 1
    r = post(client, '/heartbeat', [1, 2])          # used to be an unhandled 500
    assert r.status_code == 200


def test_heartbeat_post_v3_counts_sessions_like_get(client):
    # The v3 client POSTs {deviceId, listening, sess:<number>} so the device id
    # never lands in a URL; sessions must be counted exactly as for GET.
    s = post(client, '/heartbeat', {'deviceId': 'dev1', 'listening': 1, 'sess': 1761160000000}).get_json()
    assert s['active_now'] == 1 and s['total_sessions'] == 1
    s = post(client, '/heartbeat', {'deviceId': 'dev1', 'listening': 1, 'sess': 1761160000000}).get_json()
    assert s['total_sessions'] == 1                  # same session: no double count
    s = post(client, '/heartbeat', {'deviceId': 'dev1', 'listening': 0, 'sess': 1761160099999}).get_json()
    assert s['total_sessions'] == 2
    # Not listening yet (sess 0) and junk values never open a session.
    for junk in (0, -5, True, float('nan'), [1], {'a': 1}):
        body = '{"deviceId": "dev2", "sess": %s}' % ('NaN' if junk != junk else json.dumps(junk))
        r = post(client, '/heartbeat', None, raw=body)
        assert r.status_code == 200
    assert client.get('/heartbeat?deviceId=dev3').get_json()['total_sessions'] == 2


def test_detection_happy_path_and_log(client):
    r = post_det(client)
    assert r.status_code == 200
    body = r.get_json()
    assert body['received'] is True and body['detection_id'] == 1
    assert STATS_KEYS <= set(body)
    assert 'dropped' not in body                     # nothing to drop in a v2 payload

    log = client.get('/log').get_json()
    assert log['total_ever'] == 1 and log['total_log'] == 1 and log['returned'] == 1
    row = log['detections'][0]
    assert set(row) == {'id', 'ts', 'device', 'lat', 'lng', 'species', 'name', 'disease',
                        'freq', 'conf', 'risk', 'asymptomatic'}
    assert row['species'] == 'aedes_albopictus' and row['conf'] == 0.83 and row['freq'] == 668.0
    assert row['ts'] == '2026-10-22T10:00:00.000Z'


def test_log_filters_and_limits(client):
    for i, sp in enumerate(['culex', 'culex', 'anopheles']):
        assert post_det(client, species=sp, deviceId=f'd{i}').status_code == 200
    log = client.get('/log?species=culex').get_json()
    assert log['returned'] == 2 and all(d['species'] == 'culex' for d in log['detections'])
    log = client.get('/log?limit=1').get_json()
    assert log['returned'] == 1 and log['detections'][0]['id'] == 3      # newest first
    log = client.get('/log?from_id=2').get_json()
    assert [d['id'] for d in log['detections']] == [3]
    # Garbage parameters fall back to defaults instead of erroring.
    for q in ('limit=abc', 'limit=-5', 'from_id=x', 'limit=' + '9' * 5000):
        assert client.get('/log?' + q).status_code == 200
    assert client.get('/log?limit=999999').get_json()['returned'] == 3


def test_log_txt_format(client):
    post_det(client)
    r = client.get('/log.txt')
    assert r.status_code == 200
    assert r.mimetype == 'text/plain'
    text = r.get_data(as_text=True)
    assert 'MosquitoNet Detection Log' in text and 'Total ever: 1' in text
    assert '#1  |  device: ' in text
    assert 'A. albopictus  |  risk: HIGH  |  668.0 Hz  |  conf: 0.83' in text
    assert '32.08, 34.79' in text                    # coarsened to 2 decimals
    assert client.get('/log.txt?species=culex&limit=x&from_id=y').status_code == 200


def test_hotspots_shape_and_threshold(client, srv):
    srv.HOTSPOT_THRESHOLD = 3
    for i in range(3):
        assert post_det(client, deviceId=f'dev{i}').status_code == 200
    h = client.get('/hotspots').get_json()
    assert h['threshold'] == 3
    assert len(h['hotspots']) == 1
    spot = h['hotspots'][0]
    assert spot['total'] == 3 and spot['species'] == 'aedes_albopictus'
    assert {'key', 'lat', 'lng', 'risk'} <= set(spot)
    assert h['approaching'] == []


def test_federated_model_stats_and_upload(client, srv):
    m = client.get('/federated/model').get_json()
    assert m['round'] == 0 and len(m['weights']['W']) == 4 and len(m['weights']['b']) == 4
    # The PWA sends this one as application/json (preflighted).
    r = post(client, '/federated/upload', {'deviceId': 'd1', 'weights': weights(0.5), 'steps': 20},
             **{'Content-Type': 'application/json'})
    assert r.status_code == 200
    body = r.get_json()
    assert body['status'] == 'accepted' and STATS_KEYS <= set(body)
    assert body['total_uploads'] == 1 and body['total_rounds'] == 0
    assert client.get('/federated/stats').get_json()['total_uploads'] == 1


def test_options_preflight(client):
    r = client.options('/federated/upload', headers={
        'Origin': FIRST_PARTY, 'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'content-type'})
    assert r.status_code in (200, 204)
    assert r.headers['Access-Control-Allow-Origin'] == FIRST_PARTY
    assert 'POST' in r.headers['Access-Control-Allow-Methods']
    assert 'Content-Type' in r.headers['Access-Control-Allow-Headers']
    assert r.headers['Access-Control-Max-Age'] == '600'
    assert client.options('/anything/at/all').status_code == 204


def test_unknown_route_and_wrong_method_are_json(client):
    # The catch-all OPTIONS route (kept from v12 for preflights) makes unknown
    # GET paths a 405 rather than a 404; either way the body is JSON.
    r = client.get('/nope')
    assert r.status_code == 405 and r.get_json() == {'error': 'method not allowed'}
    r = client.post('/nope', data='{}')
    assert r.status_code == 405
    r = client.get('/detection')
    assert r.status_code == 405
    assert r.get_json() == {'received': False, 'reason': 'method not allowed'}
    assert 'POST' in r.headers['Allow']
