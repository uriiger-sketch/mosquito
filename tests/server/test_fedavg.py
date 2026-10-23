"""Federated averaging: correctness for honest updates, rejection of malformed
uploads, and bounded influence of poisoned ones."""
import json
import math

import numpy as np
import pytest

from conftest import post, weights

SEED_W = [[2.8, 0.9, 1.2, 0.8], [-0.6, 0.7, 1.0, 0.7], [-1.1, 0.7, 1.0, 0.6], [1.8, 0.8, 1.3, 0.9]]
SEED_B = [-0.4, -0.3, -0.3, -0.4]


def upload(client, device, w, steps=20):
    return post(client, '/federated/upload', {'deviceId': device, 'weights': w, 'steps': steps,
                                              'round': 1, 'version': 2, 'timestamp': 0},
                **{'Content-Type': 'application/json'})


def near(base, delta):
    """Seed weights shifted by `delta` (keeps the update under the clip norm)."""
    return {'W': [[v + delta for v in row] for row in SEED_W], 'b': [v + delta for v in SEED_B]}


def test_fedavg_matches_the_v12_formula_for_honest_updates(client, srv):
    ups = [(near(0, 0.1), 10), (near(0, -0.2), 20), (near(0, 0.3), 30)]
    for i, (w, s) in enumerate(ups):
        r = upload(client, f'dev{i}', w, s)
        assert r.status_code == 200
    body = r.get_json()
    assert body['total_rounds'] == 1
    total = sum(s for _, s in ups)
    mean_W = sum(np.array(w['W']) * s for w, s in ups) / total
    mean_b = sum(np.array(w['b']) * s for w, s in ups) / total
    expect_W = 0.3 * np.array(SEED_W) + 0.7 * mean_W
    expect_b = 0.3 * np.array(SEED_B) + 0.7 * mean_b
    assert np.allclose(body['weights']['W'], expect_W)
    assert np.allclose(body['weights']['b'], expect_b)
    assert srv.counters['clipped_updates'] == 0
    m = client.get('/federated/model').get_json()
    assert m['round'] == 1 and np.allclose(m['weights']['W'], expect_W)
    assert srv.stats['last_aggregate'] is not None


def test_no_round_before_min_uploads(client):
    assert upload(client, 'a', near(0, 0.1)).get_json()['total_rounds'] == 0
    assert upload(client, 'b', near(0, 0.1)).get_json()['total_rounds'] == 0
    assert upload(client, 'c', near(0, 0.1)).get_json()['total_rounds'] == 1


def test_one_device_cannot_fill_a_round(client, srv):
    for _ in range(5):
        assert upload(client, 'same', near(0, 0.5)).status_code == 200
    assert srv.stats['total_rounds'] == 0 and len(srv.pending_updates) == 1
    assert srv.stats['total_uploads'] == 5


def test_poisoned_update_is_clipped(client, srv):
    clip = srv.FED_CLIP_NORM
    upload(client, 'honest1', near(0, 0.0))
    upload(client, 'honest2', near(0, 0.0))
    # Extreme but in-bounds weights with a huge claimed step count.
    body = upload(client, 'evil', weights(-49.0), steps=10_000).get_json()
    assert body['total_rounds'] == 1
    moved = math.sqrt(np.sum((np.array(body['weights']['W']) - SEED_W) ** 2)
                      + np.sum((np.array(body['weights']['b']) - SEED_B) ** 2))
    # steps capped at 50 → weight 50/90; delta clipped to `clip`; blended at 0.7.
    assert moved <= 0.7 * clip * (50 / 90) + 1e-9
    assert srv.counters['clipped_updates'] == 1


@pytest.mark.parametrize('w', [
    None, [], 'weights', {'W': SEED_W}, {'b': SEED_B},
    {'W': SEED_W[:3], 'b': SEED_B},                                   # 3x4
    {'W': [row[:3] for row in SEED_W], 'b': SEED_B},                  # 4x3
    {'W': SEED_W, 'b': SEED_B[:3]},
    {'W': [[0.0] * 4] * 3 + [[0.0] * 4 + [1.0]], 'b': SEED_B},        # ragged
    {'W': [[True] * 4] * 4, 'b': SEED_B},
    {'W': [['1'] * 4] * 4, 'b': SEED_B},
    {'W': [[None] * 4] * 4, 'b': SEED_B},
    {'W': [[0.0] * 4] * 4, 'b': [0, 0, 0, 51]},                       # |b| > 50
    {'W': [[1e300] * 4] * 4, 'b': SEED_B},                            # finite but absurd
    {'W': [[0.0] * 4] * 4, 'b': [0, 0, 0, 10 ** 400]},                # int overflows float
    {'W': [[0.0] * 4] * 100_00, 'b': SEED_B},                         # huge, rejected pre-numpy
])
def test_malformed_weights_rejected(client, srv, w):
    r = upload(client, 'd', w)
    assert r.status_code == 400 or r.status_code == 413
    if r.status_code == 400:
        assert 'error' in r.get_json()
    assert srv.stats['total_uploads'] == 0


@pytest.mark.parametrize('literal', ['NaN', 'Infinity', '-Infinity'])
def test_non_finite_weights_rejected(client, srv, literal):
    raw = ('{"deviceId":"d","steps":20,"weights":{"W":[[%s,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],'
           '"b":[0,0,0,0]}}' % literal)
    r = post(client, '/federated/upload', None, raw=raw)
    assert r.status_code == 400
    assert srv.counters['rejected_uploads'] == 1


@pytest.mark.parametrize('payload', [{}, {'deviceId': 'd', 'weights': weights()},
                                     {'deviceId': 'd', 'steps': 1}, [1]])
def test_missing_fields(client, payload):
    r = post(client, '/federated/upload', payload)
    assert r.status_code == 400 and r.get_json() == {'error': 'missing required fields'}


@pytest.mark.parametrize('steps,ok', [(1, True), ('20', True), (2.7, True), (0, True), (-5, True),
                                      (None, False), ('x', False), (True, False), ([], False)])
def test_steps_parsing(client, steps, ok):
    r = upload(client, 'd', near(0, 0.1), steps=steps)
    assert (r.status_code == 200) is ok


def test_infinite_steps_is_400_not_500(client):
    raw = '{"deviceId":"d","steps":Infinity,"weights":%s}' % json.dumps(weights())
    assert post(client, '/federated/upload', None, raw=raw).status_code == 400


def test_steps_are_capped(srv):
    srv.pending_updates.clear()
    ups = [{'steps': 1, 'weights': near(0, 0.1)}, {'steps': srv.FED_MAX_STEPS, 'weights': near(0, -0.1)}]
    srv.fedavg(ups)
    assert srv.stats['total_rounds'] == 1


def test_sybil_rounds_cannot_push_the_model_out_of_bounds(harness, srv, client):
    """Clipping bounds one round; many sybil rounds must still not drive the
    global model past the upload bound (or honest clients get rejected and the
    persisted model resets at the next boot)."""
    for r in range(60):
        for k in range(3):
            assert upload(client, f'sybil{r}-{k}', weights(49.0), steps=50).status_code == 200
    assert srv.stats['total_rounds'] == 60
    assert np.abs(srv.global_W).max() <= srv.FED_GLOBAL_ABS_MAX
    assert np.abs(srv.global_b).max() <= srv.FED_GLOBAL_ABS_MAX
    # An honest client that merged the global model 50/50 is still accepted…
    merged = {'W': (0.5 * np.array(SEED_W) + 0.5 * srv.global_W).tolist(),
              'b': (0.5 * np.array(SEED_B) + 0.5 * srv.global_b).tolist()}
    assert upload(client, 'honest', merged).status_code == 200
    # …and the trained model survives a restart instead of silently resetting.
    W = srv.global_W.tolist()
    srv2 = harness.restart(srv)
    assert srv2.global_W.tolist() == W


def test_fedavg_ignores_invalid_and_empty(srv):
    before = srv.global_W.copy()
    srv.fedavg([])
    srv.fedavg([{'steps': 5, 'weights': {'W': 'bad', 'b': []}}])
    assert np.array_equal(srv.global_W, before) and srv.stats['total_rounds'] == 0
