"""The backend also serves the built app (./site on the release branch).

Without a built site the server must stay exactly the API it was; with one,
/ is the app, every file is served from memory with the page's CSP, and the API
routes are untouched.
"""
import base64
import gzip
import hashlib

import pytest

from conftest import post

INLINE = "(()=>{'use strict';document.title='MosquitoNet'})();"
PAGE = ('<!doctype html><html><head>'
        '<script type="application/ld+json">{"@type":"WebApplication"}</script>'
        '<script src="js/app.js?v=abc"></script>'
        f'<script>{INLINE}</script>'
        '</head><body>app</body></html>')


@pytest.fixture
def site(tmp_path):
    root = tmp_path / 'site'
    (root / 'js').mkdir(parents=True)
    (root / 'index.html').write_text(PAGE, encoding='utf-8')
    (root / 'js' / 'app.js').write_text('console.log("app");' * 50, encoding='utf-8')
    (root / 'sw.js').write_text('self.addEventListener("fetch",()=>{});', encoding='utf-8')
    (root / 'icon.png').write_bytes(b'\x89PNG\r\n\x1a\n' + bytes(64))
    (root / '.secret').write_text('nope', encoding='utf-8')
    (root / 'README.md').write_text('not published', encoding='utf-8')
    return root


@pytest.fixture
def app_client(harness, site):
    return harness.start(SITE_DIR=str(site)).app.test_client()


def test_without_a_built_site_the_server_is_api_only(harness, tmp_path):
    empty = tmp_path / 'placeholder'
    empty.mkdir()
    (empty / 'README.md').write_text('placeholder', encoding='utf-8')
    c = harness.start(SITE_DIR=str(empty)).app.test_client()
    assert c.get('/').get_json()['service'].startswith('MosquitoNet')
    assert c.get('/index.html').status_code == 405          # unchanged: no GET route


def test_root_is_the_app_with_a_strict_csp(app_client):
    r = app_client.get('/')
    assert r.status_code == 200 and r.content_type == 'text/html; charset=utf-8'
    assert r.get_data(as_text=True) == PAGE
    csp = r.headers['Content-Security-Policy']
    want = base64.b64encode(hashlib.sha256(INLINE.encode()).digest()).decode()
    script_src = next(d for d in csp.split('; ') if d.startswith('script-src'))
    assert script_src == f"script-src 'self' 'sha256-{want}'"     # the inline app script only
    assert "'unsafe-inline'" not in script_src and "'unsafe-eval'" not in csp
    for d in ("default-src 'self'", "object-src 'none'", "base-uri 'none'", "frame-ancestors 'none'",
              "img-src 'self' data: blob: https://*.tile.openstreetmap.org", "manifest-src 'self' blob:"):
        assert d in csp
    assert r.headers['Referrer-Policy'] == 'strict-origin-when-cross-origin'   # OSM tiles need a Referer
    assert r.headers['Cache-Control'] == 'no-cache' and r.headers['ETag']
    assert r.headers['X-Frame-Options'] == 'DENY' and r.headers['X-Content-Type-Options'] == 'nosniff'
    assert app_client.get('/index.html').get_data(as_text=True) == PAGE


def test_files_gzip_and_revalidation(app_client):
    plain = app_client.get('/js/app.js')
    assert plain.status_code == 200 and plain.content_type.startswith('text/javascript')
    assert plain.headers['Cache-Control'] == 'public, max-age=86400'
    assert 'Content-Encoding' not in plain.headers and plain.headers['Vary'] == 'Accept-Encoding'
    gz = app_client.get('/js/app.js', headers={'Accept-Encoding': 'gzip, br'})
    assert gz.headers['Content-Encoding'] == 'gzip'
    assert gzip.decompress(gz.get_data()) == plain.get_data()
    again = app_client.get('/js/app.js', headers={'If-None-Match': plain.headers['ETag']})
    assert again.status_code == 304 and again.get_data() == b''
    assert app_client.get('/sw.js').headers['Cache-Control'] == 'no-cache'
    png = app_client.get('/icon.png', headers={'Accept-Encoding': 'gzip'})
    assert png.content_type == 'image/png' and 'Content-Encoding' not in png.headers


@pytest.mark.parametrize('path', ['/.secret', '/README.md', '/../federated_server.py',
                                  '/js/../../federated_server.py', '/nope', '/js/', '/%2e%2e/etc/passwd'])
def test_only_published_files_are_served(app_client, path):
    r = app_client.get(path)
    assert r.status_code == 404 and r.get_json() == {'error': 'not found'}


def test_the_api_is_unchanged_next_to_the_app(app_client):
    h = app_client.get('/health')
    assert h.status_code == 200 and h.get_json()['status'] == 'ok'
    assert h.headers['Content-Security-Policy'] == "default-src 'none'; frame-ancestors 'none'"
    assert h.headers['Referrer-Policy'] == 'no-referrer'
    hb = post(app_client, '/heartbeat', {'deviceId': 'b' * 32, 'listening': 1, 'sess': 1})
    assert hb.status_code == 200 and 'total_detections' in hb.get_json()
    # The app's own origin may post (it is first-party).
    hb2 = post(app_client, '/heartbeat', {'deviceId': 'c' * 32}, origin='https://mosquito.charity')
    assert hb2.status_code == 200
