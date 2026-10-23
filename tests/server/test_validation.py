"""/detection input validation: good, bad, boundary, wrong types, oversized —
plus the optional v3 fields, which are dropped (not fatal) when invalid."""
import json

import pytest

from conftest import _DROP, detection, post, post_det


def _ok(r):
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body['received'] is True
    return body


def _bad(r, code=400):
    assert r.status_code == code, r.get_data(as_text=True)
    body = r.get_json()
    assert body['received'] is False and isinstance(body['reason'], str)
    return body


def _last_row(client):
    return client.get('/log?limit=1').get_json()['detections'][0]


# ── Body-level ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize('raw', ['', 'not json', '[1,2,3]', '"str"', '42', 'null',
                                 '{"species": ', '\xff\xfe'])
def test_non_object_bodies_are_400(client, raw):
    _bad(post(client, '/detection', None, raw=raw))


def test_deeply_nested_body_is_400_not_500(client):
    # 20k levels fits under the 64 KB cap; json.loads raises RecursionError,
    # which Flask's get_json(silent=True) does not catch.
    _bad(post(client, '/detection', None, raw='[' * 20000 + ']' * 20000))
    _bad(post(client, '/detection', None, raw='{"a":' * 10000 + '1' + '}' * 10000))


def test_oversized_body_is_413(client, srv):
    big = detection(disease='x' * (srv.app.config['MAX_CONTENT_LENGTH'] + 1))
    body = _bad(post(client, '/detection', big), 413)
    assert body['reason'] == 'request entity too large'


def test_content_type_does_not_matter(client):
    for ct in ('text/plain', 'application/json', 'application/x-www-form-urlencoded'):
        _ok(post(client, '/detection', detection(deviceId=ct), **{'Content-Type': ct}))


# ── Core fields (strict) ──────────────────────────────────────────────────────
@pytest.mark.parametrize('species', ['', 'AEDES_ALBOPICTUS', 'aedes', None, 7, ['culex'],
                                     {'id': 'culex'}, 'x' * 5000])
def test_unknown_or_mistyped_species(client, species):
    body = _bad(post_det(client, species=species))
    assert body['reason'].startswith('unknown species')
    assert len(body['reason']) <= 160            # never echoes a huge input back


@pytest.mark.parametrize('conf', [None, 'abc', True, [], {}, float('nan'), 'NaN', 'Infinity'])
def test_confidence_must_be_numeric(client, conf):
    payload = detection()
    payload['confidence'] = conf
    raw = json.dumps(payload)       # json.dumps emits NaN literally; the server must cope
    _bad(post(client, '/detection', None, raw=raw))


def test_confidence_missing_is_400(client):
    _bad(post_det(client, confidence=_DROP))


# Confidence is rounded to 3 dp before the range check (as in v12), so 0.4996
# counts as 0.5 and 1.0004 as 1.0.
@pytest.mark.parametrize('conf,ok', [(0.5, True), (1.0, True), ('0.9', True), (0.4996, True),
                                      (1.0004, True), (0.4994, False), (0.0, False),
                                      (1.0006, False), (-0.1, False), (5, False)])
def test_confidence_bounds(client, conf, ok):
    r = post_det(client, confidence=conf)
    _ok(r) if ok else _bad(r)


@pytest.mark.parametrize('freq,ok', [(50, True), (2000, True), (49.9, False), (2000.1, False),
                                      ('668', True), (None, False), (float('inf'), False)])
def test_frequency_bounds(client, freq, ok):
    raw = json.dumps(detection(frequency=freq))
    r = post(client, '/detection', None, raw=raw)
    _ok(r) if ok else _bad(r)


@pytest.mark.parametrize('lat,lng,ok', [(90, 180, True), (-90, -180, True), (0, 0, True),
                                         (90.0001, 0, False), (0, -180.0001, False),
                                         ('north', 0, False), (True, 0, False), ([1], 0, False)])
def test_coordinate_bounds(client, lat, lng, ok):
    r = post_det(client, lat=lat, lng=lng)
    _ok(r) if ok else _bad(r)


def test_missing_location_is_fine_and_half_location_is_dropped(client):
    _ok(post_det(client, lat=None, lng=None, deviceId='a'))
    row = _last_row(client)
    assert row['lat'] is None and row['lng'] is None
    body = _ok(post_det(client, lat=10.0, lng=_DROP, deviceId='b'))
    assert body['dropped'] == ['lat/lng']
    row = _last_row(client)
    assert row['lat'] is None and row['lng'] is None


@pytest.mark.parametrize('eid', [['a'], {'a': 1}, 'x' * 65, 'has space', 'bad\nline', '', 1.5, True])
def test_invalid_event_ids_are_rejected(client, eid):
    # Used to be stored verbatim: a list was unhashable (→ 500) and a huge
    # string was kept in the idempotency map and rewritten to disk every 5 s.
    _bad(post_det(client, eventId=eid))


def test_event_id_forms_accepted(client):
    _ok(post_det(client, eventId='0b9f2a4e-8c1d-4e5f-9a7b-3c2d1e0f9a8b', deviceId='1'))
    _ok(post_det(client, eventId='lq3k9x0-1a2b3c4d5e6f7a8b', deviceId='2'))   # fallback form
    _ok(post_det(client, eventId=12345, deviceId='3'))
    _ok(post_det(client, eventId=_DROP, deviceId='4'))                         # pre-eventId clients


# ── Legacy display fields (sanitised, never fatal) ───────────────────────────
def test_display_fields_are_sanitised_and_bounded(client, harness):
    body = _ok(post_det(client, speciesName='Aedes\nalbopictus\r\n#99 | forged ‮',
                        disease='D' * 10_000, risk='high'))
    assert 'dropped' not in body
    row = _last_row(client)
    assert '\n' not in row['name'] and '‮' not in row['name']
    assert row['name'].startswith('Aedes albopictus')
    assert len(row['disease']) == 128
    assert row['risk'] == 'HIGH'
    stored = harness.log_lines()[-1]
    assert len(stored['disease']) == 128 and '\n' not in stored['name']


@pytest.mark.parametrize('risk,expect', [('CRITICAL', 'CRITICAL'), ('none', 'NONE'),
                                          ('<script>', 'UNKNOWN'), (None, 'UNKNOWN'),
                                          (42, 'UNKNOWN'), ('x' * 100, 'UNKNOWN')])
def test_risk_is_normalised(client, risk, expect):
    _ok(post_det(client, risk=risk))
    assert _last_row(client)['risk'] == expect


def test_non_string_display_fields_fall_back(client):
    _ok(post_det(client, speciesName={'a': 1}, disease=['x']))
    row = _last_row(client)
    assert row['name'] == 'aedes_albopictus' and row['disease'] == ''


@pytest.mark.parametrize('val,expect', [(True, True), (False, False), (1, True), (0, False),
                                         ('true', True), ('false', False), (None, False),
                                         ([1], False)])
def test_asymptomatic_coercion(client, val, expect):
    _ok(post_det(client, asymptomatic=val))
    assert _last_row(client)['asymptomatic'] is expect


@pytest.mark.parametrize('ts', ['not a date', '2999-01-01T00:00:00Z', '1999-01-01T00:00:00Z',
                                12345, 'x' * 100])
def test_bad_client_timestamp_replaced_by_server_time(client, ts):
    body = _ok(post_det(client, ts=ts))
    assert body['dropped'] == ['ts']
    row_ts = _last_row(client)['ts']
    assert row_ts != ts and row_ts.startswith('20')


def test_missing_timestamp_uses_server_time_silently(client):
    body = _ok(post_det(client, ts=_DROP))
    assert 'dropped' not in body


def test_naive_timestamp_accepted(client, srv):
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
    body = _ok(post_det(client, ts=ts))
    assert 'dropped' not in body and _last_row(client)['ts'] == ts


# ── Optional v3 fields ────────────────────────────────────────────────────────
V3 = {
    'f0': 612.37,
    'genus': 'Aedes',
    'posterior': [{'id': 'aedes_albopictus', 'p': 0.71}, {'id': 'aedes_aegypti', 'p': 0.2},
                  {'id': 'culex', 'p': 0.05}],
    'range': 'far',
    'snr': 14.26,
    'appVersion': '3.0.0-rc.1+b7',
}


def test_v3_fields_accepted_stored_and_exposed(client, harness):
    body = _ok(post_det(client, **V3))
    assert 'dropped' not in body
    row = _last_row(client)
    assert row['f0'] == 612.4 and row['genus'] == 'Aedes' and row['range'] == 'far'
    assert row['snr'] == 14.3 and row['appVersion'] == '3.0.0-rc.1+b7'
    assert row['posterior'] == [{'id': 'aedes_albopictus', 'p': 0.71},
                                {'id': 'aedes_aegypti', 'p': 0.2}, {'id': 'culex', 'p': 0.05}]
    stored = harness.log_lines()[-1]
    for k in V3:
        assert k in stored


def test_old_payload_has_no_v3_keys(client, harness):
    _ok(post_det(client))
    row = _last_row(client)
    for k in V3:
        assert k not in row and k not in harness.log_lines()[-1]


def test_null_v3_fields_are_treated_as_absent(client):
    body = _ok(post_det(client, **{k: None for k in V3}))
    assert 'dropped' not in body


@pytest.mark.parametrize('field,value', [
    ('f0', 49.9), ('f0', 2000.1), ('f0', 'abc'), ('f0', '600'), ('f0', True), ('f0', [600]),
    ('genus', 'Ochlerotatus'), ('genus', 'Aedes aegypti'), ('genus', 3), ('genus', 'A' * 30),
    ('posterior', []), ('posterior', 'x'), ('posterior', {'id': 'culex', 'p': 1}),
    ('posterior', [{'id': 'culex', 'p': 0.2}] * 2),                                  # duplicate id
    ('posterior', [{'id': s, 'p': 0.1} for s in
                   ('culex', 'anopheles', 'aedes_aegypti', 'aedes_vexans', 'toxorhynchites',
                    'culex_pipiens')]),                                              # 6 > 5
    ('posterior', [{'id': 'not_a_species', 'p': 0.5}]),
    ('posterior', [{'id': 'culex', 'p': 1.2}]),
    ('posterior', [{'id': 'culex', 'p': -0.1}]),
    ('posterior', [{'id': 'culex', 'p': '0.5'}]),
    ('posterior', [{'id': 'culex', 'p': 0.6}, {'id': 'anopheles', 'p': 0.42}]),       # Σ = 1.02
    ('posterior', [{'id': 'culex'}]),
    ('posterior', ['culex']),
    ('range', 'mid'), ('range', 1), ('range', ''),
    ('snr', -20.1), ('snr', 120.1), ('snr', 'loud'), ('snr', False),
    ('appVersion', 'x' * 17), ('appVersion', '3.0 beta'), ('appVersion', '3.0/../'),
    ('appVersion', ''), ('appVersion', 3),
])
def test_invalid_v3_field_is_dropped_but_detection_kept(client, harness, field, value):
    body = _ok(post_det(client, **{**V3, field: value}))
    assert body['dropped'] == [field]
    row = _last_row(client)
    assert field not in row and field not in harness.log_lines()[-1]
    for other in V3:
        if other != field:
            assert other in row             # the valid siblings survive


@pytest.mark.parametrize('field,value,stored', [
    ('f0', 50, 50.0), ('f0', 2000, 2000.0),
    ('snr', -20, -20.0), ('snr', 120, 120.0),
    ('genus', 'anopheles', 'Anopheles'), ('genus', ' TOXORHYNCHITES ', 'Toxorhynchites'),
    ('range', 'NEAR', 'near'),
    ('appVersion', 'a' * 16, 'a' * 16),
    ('posterior', [{'id': 'culex', 'p': 0.6}, {'id': 'anopheles', 'p': 0.405}],     # Σ = 1.005 (rounding)
     [{'id': 'culex', 'p': 0.6}, {'id': 'anopheles', 'p': 0.405}]),
    ('posterior', [{'id': 'culex', 'p': 1, 'extra': 'ignored'}], [{'id': 'culex', 'p': 1.0}]),
])
def test_v3_boundaries_accepted(client, field, value, stored):
    body = _ok(post_det(client, **{field: value}))
    assert 'dropped' not in body
    assert _last_row(client)[field] == stored


def test_multiple_invalid_fields_all_reported(client, srv):
    body = _ok(post_det(client, f0=1, snr=1000, genus='X'))
    assert body['dropped'] == ['f0', 'genus', 'snr']
    assert srv.counters['dropped_fields'] == 3
