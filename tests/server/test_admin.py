"""The operator admin API (/admin/*): off unless ADMIN_PASSWORD is set, a real
server-side password check with brute-force limits, bearer sessions that
expire, first-party-only CORS, and views that show only what the server
already stores (shared locations as the device sent them, never more)."""
import json

import pytest

from conftest import FIRST_PARTY, detection, post, post_det

PW = 'correct horse battery staple'


@pytest.fixture
def admin_srv(harness):
    return harness.start(ADMIN_PASSWORD=PW)


@pytest.fixture
def ac(admin_srv):
    return admin_srv.app.test_client()


def login(c, password=PW, **headers):
    return post(c, '/admin/login', {'password': password, 'deviceId': 'admin-device'}, **headers)


def bearer(c):
    r = login(c)
    assert r.status_code == 200, r.get_json()
    return {'Authorization': 'Bearer ' + r.get_json()['token']}


def beat(c, device, listening):
    return post(c, '/heartbeat', {'deviceId': device, 'listening': listening, 'sess': 1761650000000})


ADMIN_GETS = ['/admin/devices', '/admin/detections', '/admin/logs', '/admin/server']


# ── Off by default ────────────────────────────────────────────────────────────
@pytest.mark.parametrize('path', ADMIN_GETS)
def test_admin_is_off_without_a_password(client, path):
    assert client.get(path).status_code == 404
    assert client.get(path, headers={'Authorization': 'Bearer x'}).status_code == 404


def test_login_is_off_without_a_password(client):
    assert login(client).status_code == 404


# ── Login ─────────────────────────────────────────────────────────────────────
def test_wrong_password_is_refused_counted_and_logged(ac, admin_srv, capsys):
    for i, bad in enumerate(('nope', '', None, 123, ['x'], 'x' * 300, PW + ' ')):
        r = post(ac, '/admin/login', {'password': bad}, **{'X-Forwarded-For': f'198.51.100.{i}'})
        assert r.status_code == 401 and r.get_json() == {'error': 'wrong password'}
    assert admin_srv.counters['admin_login_failures'] == 7
    assert 'login FAILED' in capsys.readouterr().out
    assert not admin_srv._admin_sessions


def test_right_password_returns_an_expiring_token(ac, admin_srv):
    r = login(ac)
    body = r.get_json()
    assert r.status_code == 200
    assert len(body['token']) >= 40 and body['expires_in'] == admin_srv.ADMIN_SESSION_SEC
    # Only a hash of the token is kept server-side.
    assert body['token'] not in json.dumps(list(admin_srv._admin_sessions))
    assert admin_srv.counters['admin_logins'] == 1


def test_env_password_whitespace_is_ignored(harness):
    srv = harness.start(ADMIN_PASSWORD=f'  {PW}\n')
    assert login(srv.app.test_client()).status_code == 200


def test_short_password_warns_at_boot(harness, capsys):
    harness.start(ADMIN_PASSWORD='short')
    assert 'shorter than 12 characters' in capsys.readouterr().out


def test_login_is_throttled_even_with_rate_limits_off(ac):
    # conftest sets RATE_LIMIT_SCALE=0; the login limiter ignores it on purpose.
    for _ in range(5):
        assert login(ac, 'guess').status_code == 401
    r = login(ac)                                  # even the right password, once throttled
    assert r.status_code == 429 and int(r.headers['Retry-After']) >= 1


def test_login_throttle_is_per_ip(ac):
    for _ in range(5):
        login(ac, 'guess', **{'X-Forwarded-For': '198.51.100.1'})
    assert login(ac, 'guess', **{'X-Forwarded-For': '198.51.100.1'}).status_code == 429
    assert login(ac, **{'X-Forwarded-For': '198.51.100.2'}).status_code == 200


def test_login_from_a_foreign_origin_is_rejected(ac, admin_srv):
    r = login(ac, origin='https://evil.example')
    assert r.status_code == 403
    assert not admin_srv._admin_sessions


# ── Sessions ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize('path', ADMIN_GETS)
def test_views_need_a_live_session(ac, path):
    assert ac.get(path).status_code == 401
    assert ac.get(path, headers={'Authorization': 'Bearer forged'}).status_code == 401
    assert ac.get(path, headers={'Authorization': 'Basic ' + PW}).status_code == 401
    assert ac.get(path, headers=bearer(ac)).status_code == 200


def test_expired_session_is_refused_and_dropped(ac, admin_srv):
    auth = bearer(ac)
    for s in admin_srv._admin_sessions.values():
        s['exp'] = 0
    assert ac.get('/admin/devices', headers=auth).status_code == 401
    assert not admin_srv._admin_sessions


def test_logout_ends_only_that_session(ac):
    a, b = bearer(ac), bearer(ac)
    r = post(ac, '/admin/logout', {}, **a)
    assert r.status_code == 200
    assert ac.get('/admin/devices', headers=a).status_code == 401
    assert ac.get('/admin/devices', headers=b).status_code == 200


def test_session_count_is_bounded(ac, admin_srv):
    first = bearer(ac)
    for i in range(admin_srv.ADMIN_SESSIONS_MAX):
        login(ac, **{'X-Forwarded-For': f'198.51.100.{i // 4 + 10}'})
    assert len(admin_srv._admin_sessions) == admin_srv.ADMIN_SESSIONS_MAX
    assert ac.get('/admin/devices', headers=first).status_code == 401   # oldest evicted


# ── CORS ──────────────────────────────────────────────────────────────────────
def test_admin_reads_are_never_cors_star(ac):
    auth = bearer(ac)
    r = ac.get('/admin/devices', headers={**auth, 'Origin': FIRST_PARTY})
    assert r.status_code == 200
    assert r.headers['Access-Control-Allow-Origin'] == FIRST_PARTY
    assert 'Origin' in r.headers['Vary']
    r = ac.get('/admin/devices', headers={**auth, 'Origin': 'https://evil.example'})
    assert r.status_code == 403 and 'Access-Control-Allow-Origin' not in r.headers


def test_admin_preflight_allows_the_authorization_header(ac):
    r = ac.open('/admin/devices', method='OPTIONS', headers={
        'Origin': FIRST_PARTY, 'Access-Control-Request-Method': 'GET',
        'Access-Control-Request-Headers': 'authorization'})
    assert r.status_code in (200, 204)
    assert r.headers['Access-Control-Allow-Origin'] == FIRST_PARTY
    assert 'Authorization' in r.headers['Access-Control-Allow-Headers']
    r = ac.open('/admin/devices', method='OPTIONS', headers={
        'Origin': 'https://evil.example', 'Access-Control-Request-Method': 'GET'})
    assert 'Access-Control-Allow-Origin' not in r.headers


def test_public_reads_stay_cors_star(ac):
    assert ac.get('/log', headers={'Origin': 'https://researcher.example'}) \
             .headers['Access-Control-Allow-Origin'] == '*'


# ── Devices ───────────────────────────────────────────────────────────────────
def test_devices_show_activity_and_only_shared_locations(ac, admin_srv):
    beat(ac, 'listener', 1)
    beat(ac, 'idle', 0)
    post(ac, '/heartbeat', {'deviceId': 'legacy'})            # old client: no flag
    assert post_det(ac, deviceId='listener', lat=32.08214, lng=34.79133).status_code == 200
    assert post_det(ac, deviceId='idle', lat=None, lng=None).status_code == 200
    d = ac.get('/admin/devices', headers=bearer(ac)).get_json()

    assert d['active'] == 3 and d['listening'] == 1 and d['with_location'] == 1
    by = {x['device']: x for x in d['devices']}
    lis, idle, legacy = by[admin_srv.dh('listener')], by[admin_srv.dh('idle')], by[admin_srv.dh('legacy')]
    assert lis['listening'] is True and idle['listening'] is False and legacy['listening'] is None
    assert lis['session_start'] == 1761650000000
    assert lis['detections'] == 1
    # Exactly what the device sent (it applies its own offset), rounded as stored.
    assert lis['locations'] == [{'id': 1, 'ts': lis['locations'][0]['ts'], 'lat': 32.0821, 'lng': 34.7913,
                                 'species': 'aedes_albopictus', 'name': 'Aedes albopictus', 'conf': 0.83}]
    assert idle['detections'] == 1 and idle['locations'] == []
    assert legacy['detections'] == 0 and legacy['last_detection'] is None
    # The raw device ids never appear, only the server's hashes.
    body = json.dumps(d)
    assert 'listener' not in body and 'legacy' not in body


def test_devices_most_recent_first_and_limited(ac, admin_srv):
    for name in ('a', 'b', 'c'):
        beat(ac, name, 1)
    d = ac.get('/admin/devices?limit=2', headers=bearer(ac)).get_json()
    assert [x['device'] for x in d['devices']] == [admin_srv.dh('c'), admin_srv.dh('b')]
    assert d['active'] == 3 and d['listening'] == 3 and d['returned'] == 2


def test_idle_devices_leave_the_view(ac, admin_srv):
    beat(ac, 'gone', 1)
    for h in admin_srv.device_registry:
        admin_srv.device_registry[h] -= admin_srv.ACTIVE_SEC + 1
    admin_srv._flush_once()
    assert not admin_srv.device_listening
    assert ac.get('/admin/devices', headers=bearer(ac)).get_json()['active'] == 0


def test_get_heartbeat_listening_flag(ac, admin_srv):
    ac.get('/heartbeat?deviceId=g&listening=1')
    assert admin_srv.device_listening[admin_srv.dh('g')] is True


# ── Detections, logs, server ──────────────────────────────────────────────────
def test_detections_are_the_stored_records(ac, admin_srv):
    post_det(ac, deviceId='one', lat=10.12341, lng=20.12341)
    post_det(ac, deviceId='two', species='culex', speciesName='Culex', frequency=480.0)
    auth = bearer(ac)
    d = ac.get('/admin/detections', headers=auth).get_json()
    assert d['total_ever'] == 2 and d['returned'] == 2
    newest = d['detections'][0]
    assert newest['device'] == admin_srv.dh('two') and 'eid' in newest
    assert d['detections'][1]['lat'] == 10.1234              # stored value, no public offset
    one = ac.get('/admin/detections?device=' + admin_srv.dh('one')[:6], headers=auth).get_json()
    assert [x['device'] for x in one['detections']] == [admin_srv.dh('one')]


def test_logs_show_recent_records_newest_first(ac):
    login(ac, 'wrong')
    auth = bearer(ac)
    post_det(ac)
    d = ac.get('/admin/logs', headers=auth).get_json()
    msgs = [r['msg'] for r in d['logs']]
    assert any(m.startswith('[Det #1]') for m in msgs)
    assert msgs.index(next(m for m in msgs if 'login from' in m)) < \
           msgs.index(next(m for m in msgs if 'login FAILED' in m))
    warn = ac.get('/admin/logs?level=warn', headers=auth).get_json()['logs']
    assert warn and all(r['level'] in ('warn', 'error') for r in warn)


def test_server_view_lists_files_never_contents(ac, admin_srv, harness):
    post_det(ac)
    admin_srv._flush_once()
    d = ac.get('/admin/server', headers=bearer(ac)).get_json()
    assert d['health']['service'].startswith('MosquitoNet')
    names = {f['name'] for f in d['files']}
    assert 'detections.jsonl' in names
    assert all(set(f) <= {'name', 'dir', 'size', 'mtime'} for f in d['files'])
    assert d['config']['data_dir'] == harness.data_dir
    assert d['admin']['sessions'] == 1 and 0 < d['admin']['expires_in'] <= admin_srv.ADMIN_SESSION_SEC
    assert 'test-secret' not in json.dumps(d)
