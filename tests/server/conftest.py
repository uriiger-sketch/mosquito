"""Shared fixtures for the federated_server test suite.

The server keeps its state in module globals and reads its configuration from
the environment at import time, so every test gets a FRESH module object
imported against its own temporary DATA_DIR. `restart()` simulates a process
restart on the same DATA_DIR: a graceful one (final fsync + state save, as
gunicorn's atexit does) or a hard kill (no final flush, like SIGKILL / OOM).
"""
import importlib.util
import json
import os
import sys
import uuid

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SERVER_PY = os.path.join(ROOT, 'federated_server.py')
FIRST_PARTY = 'https://uriiger-sketch.github.io'


def _import_server(name='federated_server'):
    """Import federated_server.py as a brand-new module object."""
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, SERVER_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Harness:
    def __init__(self, data_dir, monkeypatch):
        self.data_dir = str(data_dir)
        self.mp = monkeypatch
        self.mods = []
        self.env = {
            'DATA_DIR': self.data_dir,
            'RATE_LIMIT_SCALE': '0',        # most tests are not about rate limiting
            'TRUSTED_PROXY_HOPS': '1',
            'PSEUDONYM_SECRET': 'test-secret',
            'PUBLIC_GEO_OFFSET': '0',       # exact coordinates; offset tests turn it on
        }

    def start(self, **env):
        """Import a fresh server. Extra keyword args become env vars (None unsets)."""
        merged = {**self.env, **env}
        for k, v in merged.items():
            if v is None:
                self.mp.delenv(k, raising=False)
            else:
                self.mp.setenv(k, str(v))
        mod = _import_server()
        self.mods.append(mod)
        return mod

    def restart(self, mod, graceful=True, **env):
        mod._shutdown(flush=graceful)
        return self.start(**env)

    def close(self):
        for m in self.mods:
            try:
                m._shutdown(flush=False)
            except Exception:
                pass

    # ── file helpers ─────────────────────────────────────────────────────────
    @property
    def log_path(self):
        return os.path.join(self.data_dir, 'detections.jsonl')

    @property
    def state_path(self):
        return os.path.join(self.data_dir, 'mosquitonet_state.json')

    def log_lines(self):
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, encoding='utf-8') as f:
            return [json.loads(l) for l in f if l.strip()]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    h = Harness(tmp_path / 'data', monkeypatch)
    yield h
    h.close()


@pytest.fixture
def srv(harness):
    return harness.start()


@pytest.fixture
def client(srv):
    return srv.app.test_client()


def detection(**over):
    """A valid v2 (pre-v3) detection payload, as the deployed client sends it."""
    d = {
        'eventId':      str(uuid.uuid4()),
        'deviceId':     'a' * 32,
        'species':      'aedes_albopictus',
        'speciesName':  'Aedes albopictus',
        'disease':      'Dengue / Chikungunya',
        'confidence':   0.83,
        'frequency':    668.0,
        'risk':         'HIGH',
        'asymptomatic': False,
        'ts':           '2026-10-22T10:00:00.000Z',
        'lat':          32.08214,
        'lng':          34.79133,
    }
    for k, v in over.items():
        if v is _DROP:
            d.pop(k, None)
        else:
            d[k] = v
    return d


class _Drop:
    def __repr__(self):
        return 'DROP'


_DROP = _Drop()    # sentinel: detection(lat=DROP) removes the key


def post(client, path, payload, origin=FIRST_PARTY, raw=None, **headers):
    """POST like the PWA does: a text/plain 'simple' request with an Origin."""
    hdrs = {'Content-Type': 'text/plain'}
    if origin is not None:
        hdrs['Origin'] = origin
    hdrs.update(headers)
    body = raw if raw is not None else json.dumps(payload)
    return client.post(path, data=body, headers=hdrs)


def post_det(client, **over):
    return post(client, '/detection', detection(**over))


def weights(scale=0.0, W=None, b=None):
    """A well-formed 4x4 / len-4 weight payload."""
    return {'W': W if W is not None else [[scale] * 4 for _ in range(4)],
            'b': b if b is not None else [scale] * 4}
