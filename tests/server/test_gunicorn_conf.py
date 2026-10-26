"""gunicorn.conf.py: the process model, the logging, and the startup guards.

October 2026: Railway ran the service's start command
`gunicorn federated_server:app --bind 0.0.0.0:$PORT --workers 2 ...` in exec form
(no shell), gunicorn received the literal text '$PORT' and exited, and
production crash-looped for days. The last test replays that exact argv.
"""
import importlib.util
import json
import logging
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

from conftest import ROOT

CONF = os.path.join(ROOT, 'gunicorn.conf.py')


@pytest.fixture
def load_conf(monkeypatch):
    """Execute gunicorn.conf.py the way gunicorn does, under the given env."""
    def load(**env):
        for k, v in env.items():
            if v is None:
                monkeypatch.delenv(k, raising=False)
            else:
                monkeypatch.setenv(k, str(v))
        spec = importlib.util.spec_from_file_location('mn_gunicorn_conf', CONF)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return load


def test_process_model(load_conf):
    conf = load_conf()
    assert conf.workers == 1 and conf.worker_class == 'gthread'
    assert conf.control_socket_disable is True
    assert conf.logger_class is conf.MosquitoLogger
    assert conf.accesslog == '-' and conf.errorlog == '-'


def test_repair_bind(load_conf):
    conf = load_conf()
    env = {'PORT': '8123'}
    ok = ['127.0.0.1:5000', '0.0.0.0:8123', '[::1]', '[::]:80', 'localhost', 'unix:/tmp/s.sock', 'fd://3']
    assert conf.repair_bind(ok, env) == (ok, [])
    fixed, problems = conf.repair_bind(['0.0.0.0:$PORT', '[::]:${PORT}'], env)
    assert fixed == ['0.0.0.0:8123', '[::]:8123'] and len(problems) == 2
    # No usable PORT: the image's port, never a crash.
    assert conf.repair_bind(['0.0.0.0:$PORT'], {})[0] == ['0.0.0.0:8080']
    assert conf.repair_bind(['0.0.0.0:$PORT'], {'PORT': '$PORT'})[0] == ['0.0.0.0:8080']
    assert conf.repair_bind(['0.0.0.0:abc'], env)[0] == ['0.0.0.0:8080']


def test_bind_guard_repairs_values_as_gunicorn_sets_them(load_conf):
    from gunicorn.config import Config
    load_conf(PORT='8124')
    load_conf(PORT='8124')                     # gunicorn re-reads the file on reload
    cfg = Config()
    cfg.set('bind', ['0.0.0.0:$PORT'])         # what `--bind 0.0.0.0:$PORT` does
    assert cfg.bind == ['0.0.0.0:8124']
    assert cfg.address                         # parses: this raised before the fix
    cfg.set('bind', ['127.0.0.1:9000'])
    assert cfg.bind == ['127.0.0.1:9000']


def test_non_numeric_port_variable_gets_the_default_bind(load_conf):
    assert load_conf(PORT='$PORT').bind == ['0.0.0.0:8080']
    assert not hasattr(load_conf(PORT='8080'), 'bind')     # gunicorn's own default stands


def test_log_format_and_access_log_settings(load_conf):
    assert load_conf(LOG_FORMAT=None, RAILWAY_ENVIRONMENT_ID=None,
                     RAILWAY_PROJECT_ID=None).LOG_FORMAT == 'text'
    assert load_conf(LOG_FORMAT=None, RAILWAY_ENVIRONMENT_ID='env').LOG_FORMAT == 'json'
    assert load_conf(LOG_FORMAT='text').LOG_FORMAT == 'text'
    assert load_conf(ACCESS_LOG=None).ACCESS_LOG == 'errors'
    assert load_conf(ACCESS_LOG='bogus').ACCESS_LOG == 'errors'


def test_access_log_filter(load_conf):
    skip = load_conf(ACCESS_LOG=None)._skip_access
    retry = [('Retry-After', '5')]
    assert skip(200, 5, []) and skip(304, 5, [])
    assert skip(404, 5, []) and skip(405, 5, [])            # vulnerability scanners
    assert skip(429, 5, []) and skip(503, 5, retry)         # throttling: counted in /health
    assert not skip(400, 5, []) and not skip(403, 5, []) and not skip(413, 5, [])
    assert not skip(500, 5, []) and not skip(503, 5, [])    # 503 without Retry-After
    assert not skip(200, 1500, [])                          # slow, even if successful
    assert not load_conf(ACCESS_LOG='all')._skip_access(200, 5, [])
    assert load_conf(ACCESS_LOG='off')._skip_access(500, 5000, [])


def test_json_formatter(load_conf):
    fmt = load_conf().JsonFormatter()
    rec = logging.LogRecord('gunicorn.error', logging.WARNING, __file__, 1,
                            'bad %s\n{"forged": 1}', ('thing',), None)
    rec.fields = {'status': 403}
    line = fmt.format(rec)
    assert '\n' not in line
    assert json.loads(line) == {'level': 'warn', 'message': 'bad thing\n{"forged": 1}',
                                'component': 'gunicorn', 'status': 403}


def _free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _request(url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=3) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


@pytest.mark.skipif(sys.platform == 'win32', reason='gunicorn needs a POSIX system')
def test_railway_exec_form_start_command_now_boots(tmp_path):
    port = _free_port()
    env = {**os.environ, 'PORT': str(port), 'DATA_DIR': str(tmp_path / 'data'),
           'LOG_FORMAT': 'json', 'STATS_LOG_INTERVAL': '0', 'RATE_LIMIT_SCALE': '0'}
    # argv exactly as Railway passed the stored start command: no shell, so
    # '$PORT' stays literal, and a second worker that would corrupt DATA_DIR.
    argv = [sys.executable, '-m', 'gunicorn', 'federated_server:app',
            '--bind', '0.0.0.0:$PORT', '--workers', '2', '--timeout', '60']
    proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    base = f'http://127.0.0.1:{port}'
    try:
        deadline = time.monotonic() + 30
        while True:
            assert proc.poll() is None, 'gunicorn exited: ' + proc.stdout.read()
            try:
                status, body = _request(base + '/health')
                if status == 200:
                    break
            except OSError:
                pass
            assert time.monotonic() < deadline, 'server never became healthy'
            time.sleep(0.2)
        assert json.loads(body)['status'] == 'ok'
        assert _request(base + '/wp-json/wp/v2/users')[0] == 405       # not access-logged
        assert _request(base + '/detection', b'{}', {'Origin': 'https://evil.example'})[0] == 403
    finally:
        proc.terminate()
        out = proc.communicate(timeout=30)[0]

    recs = [json.loads(line) for line in out.splitlines() if line.strip()]   # nothing but JSON
    text = '\n'.join(r['message'] for r in recs)
    assert f"repaired bind address: '0.0.0.0:$PORT' -> '0.0.0.0:{port}'" in text
    assert 'workers=2 refused' in text
    assert sum('Booting worker' in r['message'] for r in recs) == 1
    assert {r['level'] for r in recs} <= {'debug', 'info', 'warn', 'error'}
    assert [r['status'] for r in recs if r.get('component') == 'access'] == [403]
    assert any(r.get('event') == 'exit' for r in recs)      # graceful flush on SIGTERM
