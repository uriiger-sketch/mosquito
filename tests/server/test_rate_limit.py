"""Token-bucket rate limiting per IP and per device."""
import pytest

from conftest import detection, post, post_det


@pytest.fixture
def limited(harness):
    """Server with limits on (scale 1) and small, deterministic buckets."""
    srv = harness.start(RATE_LIMIT_SCALE='1')
    srv.RATE_LIMITS.update({
        'ip:any': (1000, 1000.0), 'ip:write': (1000, 1000.0), 'ip:log': (2, 0.001),
        'dev:detection': (3, 0.001), 'dev:heartbeat': (2, 0.001), 'dev:upload': (1, 0.001),
    })
    return srv


def test_detection_throttle_is_503_so_clients_retry(limited):
    c = limited.app.test_client()
    for i in range(3):
        assert post_det(c, deviceId='dev', species=['culex', 'anopheles', 'aedes_aegypti'][i]).status_code == 200
    r = post_det(c, deviceId='dev', species='culex_pipiens')
    # 503 (not 429): the deployed client deletes a find on any 4xx.
    assert r.status_code == 503
    body = r.get_json()
    assert body['received'] is False and body['reason'] == 'rate limited'
    assert int(r.headers['Retry-After']) >= 1 and body['retry_after'] >= 1
    assert post_det(c, deviceId='other').status_code == 200          # per device
    assert c.get('/health').get_json()['counters']['rate_limited'] == 1


def test_heartbeat_and_upload_throttle_is_429(limited):
    c = limited.app.test_client()
    assert c.get('/heartbeat?deviceId=d').status_code == 200
    assert c.get('/heartbeat?deviceId=d').status_code == 200
    r = c.get('/heartbeat?deviceId=d')
    assert r.status_code == 429 and r.get_json()['error'] == 'rate limited' and 'Retry-After' in r.headers
    w = {'W': [[0.0] * 4] * 4, 'b': [0.0] * 4}
    assert post(c, '/federated/upload', {'deviceId': 'u', 'weights': w, 'steps': 1}).status_code == 200
    assert post(c, '/federated/upload', {'deviceId': 'u', 'weights': w, 'steps': 1}).status_code == 429


def test_per_ip_limits_use_the_proxy_appended_address(limited):
    limited.RATE_LIMITS['ip:any'] = (2, 0.001)
    c = limited.app.test_client()

    def get(xff):
        return c.get('/federated/stats', headers={'X-Forwarded-For': xff})

    assert get('1.1.1.1').status_code == 200
    assert get('1.1.1.1').status_code == 200
    assert get('1.1.1.1').status_code == 429
    assert get('2.2.2.2').status_code == 200                     # a different client
    # Spoofing: a client-supplied left-hand XFF entry is ignored; with one
    # trusted hop, only the address the proxy appended (rightmost) counts.
    assert get('9.9.9.9, 1.1.1.1').status_code == 429


def test_ipv6_clients_are_grouped_by_64(limited):
    limited.RATE_LIMITS['ip:any'] = (2, 0.001)
    c = limited.app.test_client()
    h = lambda ip: {'X-Forwarded-For': ip}
    assert c.get('/federated/stats', headers=h('2001:db8:1:2::1')).status_code == 200
    assert c.get('/federated/stats', headers=h('2001:db8:1:2::ffff')).status_code == 200
    assert c.get('/federated/stats', headers=h('2001:db8:1:2:aaaa::9')).status_code == 429
    assert c.get('/federated/stats', headers=h('2001:db8:1:3::1')).status_code == 200


def test_log_endpoints_have_their_own_tighter_bucket(limited):
    c = limited.app.test_client()
    assert c.get('/log').status_code == 200
    assert c.get('/log.txt').status_code == 200
    assert c.get('/log').status_code == 429
    assert c.get('/hotspots').status_code == 200


def test_ipv4_mapped_addresses_are_keyed_as_ipv4(limited):
    limited.RATE_LIMITS['ip:any'] = (1, 0.001)
    c = limited.app.test_client()
    h = lambda ip: {'X-Forwarded-For': ip}
    assert c.get('/federated/stats', headers=h('::ffff:1.2.3.4')).status_code == 200
    # A different IPv4 client must not share the bucket (they all used to map to ::/64)…
    assert c.get('/federated/stats', headers=h('::ffff:5.6.7.8')).status_code == 200
    # …while the same client, mapped or not, does.
    assert c.get('/federated/stats', headers=h('1.2.3.4')).status_code == 429


def test_health_has_its_own_bucket_and_preflight_is_free(limited):
    limited.RATE_LIMITS['ip:any'] = (1, 0.001)
    limited.RATE_LIMITS['ip:health'] = (3, 0.001)
    c = limited.app.test_client()
    c.get('/federated/stats')
    for _ in range(3):                    # other traffic exhausted ip:any; probes still pass
        assert c.get('/health').status_code == 200
        assert c.options('/detection').status_code in (200, 204)
    assert c.get('/health').status_code == 429
    assert c.options('/detection').status_code in (200, 204)
    assert c.get('/federated/stats').status_code == 429


def test_per_ip_upload_budget_limits_rounds(limited):
    limited.RATE_LIMITS['ip:upload'] = (3, 0.001)
    c = limited.app.test_client()
    w = {'W': [[0.0] * 4] * 4, 'b': [0.0] * 4}
    for i in range(3):                    # rotating device ids don't buy more uploads
        assert post(c, '/federated/upload', {'deviceId': f's{i}', 'weights': w, 'steps': 1}).status_code == 200
    assert post(c, '/federated/upload', {'deviceId': 's9', 'weights': w, 'steps': 1}).status_code == 429
    assert post_det(c).status_code == 200            # detections unaffected


def test_ip_write_bucket(limited):
    limited.RATE_LIMITS['ip:write'] = (2, 0.001)
    c = limited.app.test_client()
    assert post_det(c, deviceId='a').status_code == 200
    assert post_det(c, deviceId='b').status_code == 200
    assert post_det(c, deviceId='c').status_code == 503          # rotating device ids don't help
    assert c.get('/heartbeat?deviceId=z').status_code == 200     # reads unaffected


def test_scale_zero_disables(harness):
    srv = harness.start(RATE_LIMIT_SCALE='0')
    srv.RATE_LIMITS['dev:detection'] = (1, 0.001)
    c = srv.app.test_client()
    for sp in ('culex', 'anopheles', 'aedes_aegypti'):
        assert post_det(c, species=sp).status_code == 200


def test_invalid_env_falls_back_to_default(harness):
    srv = harness.start(RATE_LIMIT_SCALE='lots', TRUSTED_PROXY_HOPS='-1', DETECTION_LOG_MAX='abc')
    assert srv.RATE_LIMIT_SCALE == 1.0 and srv.TRUSTED_PROXY_HOPS == 1
    assert srv.DETECTION_LOG_MAX == 20000


def test_no_proxy_mode_ignores_forwarded_for(harness):
    srv = harness.start(RATE_LIMIT_SCALE='1', TRUSTED_PROXY_HOPS='0')
    srv.RATE_LIMITS['ip:any'] = (1, 0.001)
    c = srv.app.test_client()
    assert c.get('/federated/stats', headers={'X-Forwarded-For': '1.1.1.1'}).status_code == 200
    assert c.get('/federated/stats', headers={'X-Forwarded-For': '2.2.2.2'}).status_code == 429


def test_bucket_refill_and_lru_bound(srv):
    lim = srv._RateLimiter(max_keys=3)
    assert lim.hit('k', 2, 1.0, now=0.0) == 0.0
    assert lim.hit('k', 2, 1.0, now=0.0) == 0.0
    wait = lim.hit('k', 2, 1.0, now=0.0)
    assert wait == pytest.approx(1.0)
    assert lim.hit('k', 2, 1.0, now=1.0) == 0.0                  # one token refilled
    assert lim.hit('k', 2, 1.0, now=100.0) == 0.0                # capped at capacity…
    assert lim.hit('k', 2, 1.0, now=100.0) == 0.0
    assert lim.hit('k', 2, 1.0, now=100.0) > 0                   # …not 100 tokens
    for k in ('a', 'b', 'c', 'd'):
        lim.hit(k, 2, 1.0, now=0.0)
    assert len(lim._buckets) == 3 and 'k' not in lim._buckets    # memory bounded
