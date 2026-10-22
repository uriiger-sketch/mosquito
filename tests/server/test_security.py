"""CORS / Origin policy, security headers, error hygiene, log injection and the
privacy of the public /log and /log.txt endpoints."""
import pytest

from conftest import FIRST_PARTY, detection, post, post_det

SECURITY_HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'Referrer-Policy': 'no-referrer',
    'Cache-Control': 'no-store',
}


# ── CORS ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize('path', ['/health', '/log', '/hotspots', '/federated/stats',
                                  '/federated/model', '/heartbeat?deviceId=x', '/log.txt', '/'])
def test_reads_are_public(client, path):
    r = client.get(path, headers={'Origin': 'https://researcher.example'})
    assert r.status_code == 200
    assert r.headers['Access-Control-Allow-Origin'] == '*'


@pytest.mark.parametrize('origin', [
    FIRST_PARTY, 'https://mosquito.charity', 'https://app.mosquito.charity',
    'http://localhost:8080', 'http://127.0.0.1:5500', 'http://localhost', 'https://localhost:3000',
    'http://[::1]:8080', 'HTTPS://URIIGER-SKETCH.GITHUB.IO',
])
def test_first_party_writes_allowed_and_reflected(client, origin):
    r = post(client, '/detection', detection(), origin=origin)
    assert r.status_code == 200
    assert r.headers['Access-Control-Allow-Origin'] == origin
    assert 'Origin' in r.headers['Vary']


@pytest.mark.parametrize('origin', [
    'https://evil.example', 'null', 'https://uriiger-sketch.github.io.evil.example',
    'https://evilmosquito.charity', 'http://uriiger-sketch.github.io',            # http, not https
    'https://other-user.github.io', 'http://localhost.evil.example', '',
])
def test_foreign_origin_writes_rejected(client, srv, origin):
    r = post(client, '/detection', detection(), origin=origin)
    assert r.status_code == 403
    assert r.get_json() == {'received': False, 'reason': 'origin not allowed'}
    assert 'Access-Control-Allow-Origin' not in r.headers
    assert srv.stats['total_detections'] == 0
    r = post(client, '/federated/upload', {'deviceId': 'd', 'steps': 1, 'weights': {}}, origin=origin)
    assert r.status_code == 403 and r.get_json() == {'error': 'origin not allowed'}


def test_no_origin_header_is_allowed(client):
    # curl / native clients / same-origin navigations don't send Origin on every request.
    assert post(client, '/detection', detection(), origin=None).status_code == 200


def test_allowed_origins_env_extends_never_replaces(harness):
    srv = harness.start(ALLOWED_ORIGINS='https://Partner.example/some/path/, https://x.netlify.app',
                        ALLOWED_ORIGIN='https://legacy.example')
    c = srv.app.test_client()
    for origin, dev in (('https://partner.example', 'a'), ('https://x.netlify.app', 'b'),
                        ('https://legacy.example', 'c'), (FIRST_PARTY, 'd')):
        assert post(c, '/detection', detection(deviceId=dev), origin=origin).status_code == 200, origin
    assert post(c, '/detection', detection(deviceId='e'), origin='https://evil.example').status_code == 403


def test_wildcard_in_env_is_ignored(harness):
    srv = harness.start(ALLOWED_ORIGINS='*')
    c = srv.app.test_client()
    assert post(c, '/detection', detection(), origin='https://evil.example').status_code == 403


def test_preflight_from_foreign_origin_gets_no_allow_origin(client):
    r = client.options('/federated/upload', headers={
        'Origin': 'https://evil.example', 'Access-Control-Request-Method': 'POST'})
    assert 'Access-Control-Allow-Origin' not in r.headers


def test_preflight_for_get_is_public(client):
    r = client.options('/log', headers={'Origin': 'https://evil.example',
                                        'Access-Control-Request-Method': 'GET'})
    assert r.headers['Access-Control-Allow-Origin'] == '*'


# ── Headers & error hygiene ───────────────────────────────────────────────────
@pytest.mark.parametrize('method,path', [('get', '/health'), ('get', '/log'), ('get', '/nope'),
                                         ('post', '/detection')])
def test_security_headers_everywhere(client, method, path):
    r = getattr(client, method)(path, data='{}', headers={'Origin': FIRST_PARTY})
    for k, v in SECURITY_HEADERS.items():
        assert r.headers.get(k) == v, k
    assert "default-src 'none'" in r.headers['Content-Security-Policy']
    assert r.headers['Strict-Transport-Security'].startswith('max-age=')


def test_internal_errors_are_generic_json(client, srv, monkeypatch):
    def boom():
        raise RuntimeError('secret internal detail /data/path')
    monkeypatch.setattr(srv, 'full_stats', boom)
    r = client.get('/federated/stats')
    assert r.status_code == 500
    assert r.get_json() == {'error': 'internal error'}
    assert b'secret' not in r.data and b'Traceback' not in r.data
    r = post_det(client)
    assert r.status_code == 500 and r.get_json() == {'received': False, 'reason': 'internal error'}


def test_reject_reason_is_bounded_and_log_safe(client, capsys):
    r = post_det(client, species='x\n[Det #1] forged line' + 'y' * 5000)
    assert r.status_code == 400 and len(r.get_json()['reason']) <= 160
    out = capsys.readouterr().out
    assert '\n[Det #1] forged' not in out


def test_log_txt_cannot_be_forged_by_labels(client):
    post_det(client, speciesName='Aedes albopictus\n#999  |  device: deadbeef  |  0.0, 0.0',
             disease='x\r\ny', risk='HIGH\nLOW')
    text = client.get('/log.txt').get_data(as_text=True)
    assert '\n#999' not in text
    assert text.count('\n#') == 1


# ── Privacy of public endpoints ───────────────────────────────────────────────
def test_public_coordinates_are_coarsened(client, srv, harness):
    post_det(client, lat=32.123456, lng=-117.987654)
    row = client.get('/log').get_json()['detections'][0]
    assert row['lat'] == 32.12 and row['lng'] == -117.99
    stored = harness.log_lines()[-1]                  # operator keeps the (jittered) 4-dp value
    assert stored['lat'] == 32.1235 and stored['lng'] == -117.9877


def test_coordinate_precision_is_configurable(harness):
    srv = harness.start(PUBLIC_COORD_DECIMALS='1')
    c = srv.app.test_client()
    post_det(c, lat=32.123456, lng=34.98)
    row = c.get('/log').get_json()['detections'][0]
    assert row['lat'] == 32.1 and row['lng'] == 35.0


def test_device_is_a_per_day_pseudonym(client, srv):
    dev_hash = srv.dh('my-device-id')
    post_det(client, deviceId='my-device-id', ts='2026-10-20T10:00:00Z', species='culex')
    post_det(client, deviceId='my-device-id', ts='2026-10-20T18:00:00Z', species='anopheles')
    post_det(client, deviceId='my-device-id', ts='2026-10-21T10:00:00Z', species='aedes_aegypti')
    rows = client.get('/log').get_json()['detections']
    devices = [r['device'] for r in rows]             # newest first: 21st, 20th, 20th
    assert dev_hash not in devices
    assert devices[1] == devices[2]                   # same day: linkable (researcher dedupe)
    assert devices[0] != devices[1]                   # next day: not linkable
    assert all(len(d) == 16 for d in devices)
    # The /log?device= filter works on the public pseudonym, not the stable hash.
    assert client.get(f'/log?device={devices[1]}').get_json()['returned'] == 2
    assert client.get(f'/log?device={dev_hash}').get_json()['returned'] == 0
    txt = client.get('/log.txt').get_data(as_text=True)
    assert dev_hash[:8] not in txt and devices[0][:8] in txt


def test_pseudonym_key_is_secret(harness, monkeypatch):
    a = harness.start(PSEUDONYM_SECRET=None)
    pa = a._pseudonym('dev', '2026-10-20')
    a._shutdown()
    b = harness.start(PSEUDONYM_SECRET=None)
    assert b._pseudonym('dev', '2026-10-20') != pa     # random per process when unset


def test_private_fields_never_served(client):
    post_det(client, eventId='secret-event-id')
    row = client.get('/log').get_json()['detections'][0]
    assert 'eid' not in row and 'secret-event-id' not in str(row)
    assert 'secret-event-id' not in client.get('/log.txt').get_data(as_text=True)


def test_health_does_not_leak_python_build(client):
    h = client.get('/health').get_json()
    assert 'GCC' not in h['python'] and '(' not in h['python']
