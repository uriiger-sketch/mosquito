# MosquitoNet federated server — production image (Railway builds from this
# file when it is present; also Fly / Render / any VPS).
#
# python:3.12 matches runtime.txt. The tag floats on 3.12.x patch releases, so
# a rebuild picks up CPython security fixes; the Debian release is pinned
# (bookworm) so the OS underneath can't change major version silently.
FROM python:3.12-slim-bookworm

# gunicorn reads GUNICORN_CMD_ARGS itself, so these invariants also apply if a
# platform overrides the start command with a bare `gunicorn federated_server:app`
# (and they beat a platform-injected WEB_CONCURRENCY):
#   --workers 1          ALL state lives in this one process — never raise it (threads are fine)
#   --no-control-socket  gunicorn >= 25 otherwise opens a runtime admin socket that can add workers
#   --access-logformat   no query string (heartbeat URLs carry the raw device id) and no IP
# gunicorn binds 0.0.0.0:$PORT by default, so platform-assigned ports just work.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data \
    PORT=8080 \
    GUNICORN_CMD_ARGS="--workers 1 --threads 4 --timeout 120 --graceful-timeout 30 --no-control-socket --access-logfile - --access-logformat '%(t)s \"%(m)s %(U)s\" %(s)s %(B)sB %(M)sms'"

# Unprivileged runtime user with a fixed UID/GID (10001), so ownership of a
# host bind mount can be prepared with `chown 10001:10001 <dir>`.
#
# The entrypoint starts as root ONLY to hand DATA_DIR to `app` — platform
# volumes (Railway, Render, host bind mounts) are mounted root-owned, and data
# written by images before v13 is root-owned too — then drops privileges for
# good with setpriv: gunicorn runs as `app` with no capabilities and no root
# process remains. If the directory still isn't writable by `app` (e.g. an NFS
# volume that refuses chown) it stays root with a loud warning rather than
# turning every detection into a 503. Started with `--user`, it runs as-is.
# (Single quotes below are deliberate: the script is written out verbatim.)
# hadolint ignore=SC2016
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home --home-dir /nonexistent \
            --shell /usr/sbin/nologin app \
 && install -d -o app -g app -m 0700 /data \
 && printf '%s\n' \
      '#!/bin/sh' \
      'set -u' \
      'if [ "$(id -u)" = 0 ]; then' \
      '  mkdir -p "$DATA_DIR" && chown -R app:app "$DATA_DIR"' \
      '  if setpriv --reuid=app --regid=app --init-groups test -w "$DATA_DIR"; then' \
      '    exec setpriv --reuid=app --regid=app --init-groups --inh-caps=-all "$@"' \
      '  fi' \
      '  echo "[entrypoint] WARNING: $DATA_DIR is not writable by uid 10001 - running as ROOT" >&2' \
      'fi' \
      'exec "$@"' \
      > /usr/local/bin/docker-entrypoint \
 && chmod 0755 /usr/local/bin/docker-entrypoint

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY federated_server.py .

# Persistence: Railway does not support the Dockerfile VOLUME instruction.
# Attach a Railway Volume to the service with mount path /data.
EXPOSE 8080

# Liveness: /health has its own generous rate-limit bucket, so busy traffic
# can't starve the probe. ProxyHandler({}) ignores any proxy env vars so the
# probe always hits the local server.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import os, sys, urllib.request as u; r = u.build_opener(u.ProxyHandler({})).open('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8080'), timeout=4); sys.exit(0 if r.status == 200 else 1)"]

ENTRYPOINT ["/usr/local/bin/docker-entrypoint"]
CMD ["gunicorn", "federated_server:app"]
