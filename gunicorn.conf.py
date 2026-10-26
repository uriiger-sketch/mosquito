"""
Gunicorn settings for the MosquitoNet server.

Gunicorn reads ./gunicorn.conf.py from its working directory by itself (/app in
the Docker image, the repo root for the Procfile), so these apply however the
server is started: the Dockerfile CMD, the Procfile, a platform start command,
or a bare `gunicorn federated_server:app`. No start command needs a flag or a
shell variable: gunicorn binds 0.0.0.0:$PORT on its own when PORT is set.

Why this file exists: in October 2026 production crash-looped on Railway with
"Error: '$PORT' is not a valid port number." — Railway runs a Dockerfile
service's start command in exec form (no shell), so `--bind 0.0.0.0:$PORT`
reached gunicorn as the literal text '$PORT'. on_starting() below repairs such
a bind (and a stale `--workers 2`) with a loud log line instead of crashing.

Logging: LOG_FORMAT=json (the default on Railway) turns gunicorn's own lines
and the access log into one JSON object per line on stdout, like the app's
(see _log in federated_server.py). ACCESS_LOG picks which requests are logged:
  errors (default)  4xx/5xx except throttling (429, and 503 + Retry-After,
                    which /health counts) and 404/405 (the public domain gets a
                    steady stream of WordPress/xmlrpc vulnerability scans), plus
                    any request slower than 1 s
  all               every request
  off               none
Railway's HTTP log already records every request at the edge, so `errors`
keeps the app log readable without losing anything.
"""
import json
import logging
import os
import re
import sys
import traceback

from gunicorn.glogging import Logger

# ── Process model ─────────────────────────────────────────────────────────────
# ONE worker process: all server state lives in it, and a second writer would
# corrupt DATA_DIR. Threads are fine. Command-line flags, GUNICORN_CMD_ARGS and
# WEB_CONCURRENCY all beat this file, so on_starting() and nworkers_changed()
# enforce it again.
workers = 1
worker_class = 'gthread'
threads = 4
timeout = 120
graceful_timeout = 30
# gunicorn >= 25 otherwise opens a runtime admin socket that can add workers.
control_socket_disable = True

DEFAULT_PORT = '8080'      # the port the Docker image and the Railway domains use

# ── Logging ───────────────────────────────────────────────────────────────────
def _on_railway():
    return bool(os.environ.get('RAILWAY_ENVIRONMENT_ID') or os.environ.get('RAILWAY_PROJECT_ID'))

LOG_FORMAT = os.environ.get('LOG_FORMAT', '').strip().lower()
if LOG_FORMAT not in ('json', 'text'):
    LOG_FORMAT = 'json' if _on_railway() else 'text'

ACCESS_LOG = os.environ.get('ACCESS_LOG', '').strip().lower()
if ACCESS_LOG not in ('errors', 'all', 'off'):
    ACCESS_LOG = 'errors'
SLOW_REQUEST_MS = 1000

accesslog = '-'
errorlog = '-'
# Text mode only. No query string (old heartbeat URLs carried the raw device
# id) and no client IP.
access_log_format = '%(t)s "%(m)s %(U)s" %(s)s %(B)sB %(M)sms'

_LEVEL_NAMES = {logging.DEBUG: 'debug', logging.INFO: 'info', logging.WARNING: 'warn',
                logging.ERROR: 'error', logging.CRITICAL: 'error'}


class JsonFormatter(logging.Formatter):
    """One JSON object per record, the shape Railway parses: `level`,
    `message`, and any extra fields passed as extra={'fields': {...}}."""

    def format(self, record):
        rec = {'level': _LEVEL_NAMES.get(record.levelno, 'info'),
               'message': record.getMessage(),
               'component': 'gunicorn'}
        fields = getattr(record, 'fields', None)
        if isinstance(fields, dict):
            rec.update(fields)
        if record.exc_info:
            rec['stack'] = self.formatException(record.exc_info)
        return json.dumps(rec, default=str, separators=(',', ':'))


def _skip_access(status, ms, headers):
    if ACCESS_LOG == 'off':
        return True
    if ACCESS_LOG == 'all' or ms >= SLOW_REQUEST_MS:
        return False
    if status < 400 or status in (404, 405, 429):
        return True
    # /detection answers throttling with 503 + Retry-After (clients retry 5xx
    # but drop a find on 4xx); a full disk is logged by the app itself.
    return status == 503 and any(k.lower() == 'retry-after' for k, _ in headers)


class MosquitoLogger(Logger):
    """gunicorn's logger, plus: JSON lines on stdout in json mode (Railway
    shows anything on stderr as an error) and a filtered access log."""

    def setup(self, cfg):
        super().setup(cfg)
        if LOG_FORMAT != 'json':
            return
        for log in (self.error_log, self.access_log):
            for h in log.handlers:
                if getattr(h, '_gunicorn', False) and isinstance(h, logging.StreamHandler) \
                        and h.stream in (sys.stderr, sys.stdout):
                    h.setStream(sys.stdout)
                    h.setFormatter(JsonFormatter())

    def access(self, resp, req, environ, request_time):
        try:
            status = int(str(resp.status).split(None, 1)[0])
        except (TypeError, ValueError):
            status = 0
        ms = request_time.seconds * 1000 + request_time.microseconds // 1000
        headers = list(getattr(resp, 'headers', None) or [])
        if _skip_access(status, ms, headers):
            return
        if LOG_FORMAT != 'json':
            return super().access(resp, req, environ, request_time)
        method, path = environ.get('REQUEST_METHOD'), environ.get('PATH_INFO')
        level = logging.ERROR if status >= 500 else logging.WARNING if status >= 400 else logging.INFO
        self.access_log.log(level, '%s %s %s %sms', method, path, status, ms,
                            extra={'fields': {'component': 'access', 'method': method,
                                              'path': path, 'status': status,
                                              'duration_ms': ms,
                                              'bytes': getattr(resp, 'sent', None)}})


logger_class = MosquitoLogger


# ── Startup guards ────────────────────────────────────────────────────────────
_VAR = re.compile(r'\$(\{)?PORT(?(1)\})')

def repair_bind(addresses, environ=os.environ):
    """Return (addresses, problems): each `host:$PORT` / `host:${PORT}` with the
    variable substituted, and any port that still isn't a number replaced by
    DEFAULT_PORT. A start command run without a shell passes '$PORT' through
    literally; a PORT *variable* that is not a number (e.g. set to the text
    '$PORT') does the same to gunicorn's default bind."""
    port = (environ.get('PORT') or '').strip()
    port = port if port.isdigit() else DEFAULT_PORT
    fixed, problems = [], []
    for addr in addresses:
        new = _VAR.sub(port, addr)
        # ':' after any IPv6 literal ([::1]:80) separates the port.
        if not addr.startswith(('unix:', 'fd://')) and ':' in new.rsplit(']', 1)[-1]:
            host, _, p = new.rpartition(':')
            if not p.isdigit():
                new = f'{host}:{DEFAULT_PORT}'
        if new != addr:
            problems.append(f'{addr!r} -> {new!r}')
        fixed.append(new)
    return fixed, problems


_BIND_REPAIRS = []      # what the guard below changed, logged by on_starting()

def _install_bind_guard():
    """Run every bind value through repair_bind() as gunicorn sets it.

    The repair can't wait for a server hook: gunicorn parses the bind addresses
    (Arbiter.setup → cfg.address) before on_starting() runs, and a bad one is a
    fatal error. This file is executed after the command line is parsed but
    before its values are applied, and each value — from here,
    GUNICORN_CMD_ARGS or the command line — passes through the setting's
    validator, so wrapping that validator catches `--bind 0.0.0.0:$PORT`.
    Idempotent (gunicorn re-reads this file on reload); if gunicorn's internals
    ever change, the guard is skipped and gunicorn behaves as stock."""
    try:
        from gunicorn.config import Bind
    except ImportError:
        return
    if getattr(Bind.validator, '_mosquito_guard', False):
        return
    stock = Bind.validator

    def validator(val):
        addresses, problems = repair_bind(stock(val))
        _BIND_REPAIRS.extend(problems)
        return addresses

    validator._mosquito_guard = True
    Bind.validator = staticmethod(validator)

_install_bind_guard()

# gunicorn computes its default bind (0.0.0.0:$PORT) when it is imported,
# before this file runs, so a PORT variable that is not a number is replaced here.
if os.environ.get('PORT', '').strip() and not os.environ['PORT'].strip().isdigit():
    bind = [f'0.0.0.0:{DEFAULT_PORT}']
    _BIND_REPAIRS.append(f'PORT={os.environ["PORT"]!r} -> {DEFAULT_PORT}')


def on_starting(server):
    """Say loudly what was repaired: the deploy works, but its configuration
    is wrong and should be fixed at the source."""
    if _BIND_REPAIRS:
        server.log.error('repaired bind address: %s. A start command run without a '
                         'shell (Railway, Dockerfile builds) passes "$PORT" literally, '
                         'and PORT must be a number. Remove --bind from the start '
                         'command: gunicorn binds 0.0.0.0:$PORT by itself',
                         '; '.join(dict.fromkeys(_BIND_REPAIRS)))
    if server.cfg.workers != 1:
        server.cfg.set('workers', 1)     # nworkers_changed() already clamped and logged
        server.num_workers = 1


def nworkers_changed(server, new_value, old_value):
    """Called for the initial worker count (before on_starting) and for every
    change (a TTIN signal): hold it at 1."""
    if new_value != 1:
        server._num_workers = 1
        if server.log:
            server.log.error('workers=%s refused: MosquitoNet keeps all state in ONE '
                             'process (a second worker corrupts DATA_DIR); using 1',
                             new_value)


def when_ready(server):
    server.log.info('ready: listening on %s, workers=1, threads=%s, timeout=%ss, '
                    'log_format=%s, access_log=%s',
                    ','.join(server.cfg.bind), server.cfg.threads, server.cfg.timeout,
                    LOG_FORMAT, ACCESS_LOG)


def worker_abort(worker):
    """A request ran past `timeout` and gunicorn is killing the worker: dump
    every thread's stack so the hang can be diagnosed from the log."""
    stacks = []
    for tid, frame in sys._current_frames().items():
        stacks.append(f'Thread {tid}:\n' + ''.join(traceback.format_stack(frame)))
    dump = '\n'.join(stacks)
    if LOG_FORMAT == 'json':
        worker.log.error('worker %s timed out after %ss; thread stacks attached',
                         worker.pid, worker.cfg.timeout, extra={'fields': {'stack': dump}})
    else:
        worker.log.error('worker %s timed out after %ss; thread stacks:\n%s',
                         worker.pid, worker.cfg.timeout, dump)


def on_exit(server):
    server.log.info('gunicorn master exiting')
