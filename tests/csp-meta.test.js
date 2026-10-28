'use strict';
// tools/csp-meta.js gives the built page its CSP as a <meta> (GitHub Pages can't
// send headers), and tools/build-site.sh refuses symlinks, which would publish
// their targets. See the header of tools/csp-meta.js.
const test = require('node:test');
const assert = require('node:assert/strict');
const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const { addCspMeta, cspPolicy } = require('../tools/csp-meta');

const ROOT = path.join(__dirname, '..');
const CODE = "(()=>{document.title='x'})();";
const PAGE = '<!doctype html><html><head><meta charset=utf-8><title>t</title>' +
             '<script type="application/ld+json">{"@type":"WebApplication"}</script>' +
             `<script src="js/a.js?v=1"></script><script>${CODE}</script></head><body></body></html>`;
const URLS = 'https://mosquito.charity,https://web-production-de7b5.up.railway.app';
const directives = (policy) => policy.split('; ');

test('the meta comes right after the charset, with the inline script by hash', () => {
  const { html, policy } = addCspMeta(PAGE, URLS);
  assert.ok(html.startsWith('<!doctype html><html><head><meta charset=utf-8><meta http-equiv="Content-Security-Policy" content="'));
  const hash = crypto.createHash('sha256').update(CODE, 'utf8').digest('base64');
  assert.ok(directives(policy).includes(`script-src 'self' 'sha256-${hash}'`));
  assert.ok(directives(policy).includes(`connect-src 'self' ${URLS.replace(',', ' ')}`));
  for (const bad of ['frame-ancestors', 'report-uri', 'sandbox', "'unsafe-eval'"]) assert.ok(!policy.includes(bad), bad);
  assert.ok(!directives(policy).find(d => d.startsWith('script-src')).includes("'unsafe-inline'"));
  assert.ok(!policy.includes('"'), 'the policy must not break out of the attribute');
});

test('it refuses what would make a wrong policy', () => {
  assert.throws(() => addCspMeta(PAGE.replace(`<script>${CODE}</script>`, ''), URLS), /exactly 1 inline script/);
  assert.throws(() => addCspMeta(PAGE.replace('</head>', '<script>2</script></head>'), URLS), /exactly 1 inline script/);
  assert.throws(() => addCspMeta(PAGE, 'https://ok.example,javascript:alert(1)'), /not an origin/);
  assert.throws(() => addCspMeta(PAGE, 'https://evil.example/path'), /not an origin/);
  assert.throws(() => addCspMeta(PAGE, ''), /no endpoints/);
  assert.throws(() => addCspMeta(PAGE.replace('<meta charset=utf-8>', ''), URLS), /charset/);
  assert.throws(() => addCspMeta(addCspMeta(PAGE, URLS).html, URLS), /already has a CSP/);
});

test('the meta policy matches the header the backend sends (federated_server._site_csp)', () => {
  const src = fs.readFileSync(path.join(ROOT, 'federated_server.py'), 'utf8');
  const body = src.match(/def _site_csp\([\s\S]*?return '; '\.join\(\[([\s\S]*?)\]\)/);
  assert.ok(body, '_site_csp not found');
  const server = [...body[1].matchAll(/^\s*"([^"]+)"/gm)].map(m => m[1]);
  assert.ok(server.length >= 10, 'parsed the server directives');
  const meta = directives(cspPolicy(PAGE, URLS));
  // Deliberate differences: a <meta> can't carry frame-ancestors, connect-src is
  // narrowed to the app's endpoints, and script-src carries the hash.
  for (const d of server) {
    if (/^(frame-ancestors|connect-src|script-src)\b/.test(d)) continue;
    assert.ok(meta.includes(d), `server directive missing from the meta: ${d}`);
  }
  assert.deepEqual(meta.map(d => d.split(' ')[0]).sort(),
                   server.map(d => d.split(' ')[0]).filter(n => n !== 'frame-ancestors').sort());
});

test('the build refuses symlinks (they would publish their targets)', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'mn-build-'));
  const run = (extra) => {
    const repo = path.join(tmp, extra ? 'with-link' : 'clean');
    fs.mkdirSync(path.join(repo, 'tools'), { recursive: true });
    fs.copyFileSync(path.join(ROOT, 'tools', 'build-site.sh'), path.join(repo, 'tools', 'build-site.sh'));
    fs.writeFileSync(path.join(repo, 'index.html'), PAGE);
    if (extra) fs.symlinkSync('/proc/self/environ', path.join(repo, 'licence.txt'));
    spawnSync('git', ['init', '-q'], { cwd: repo });
    return spawnSync('bash', [path.join(repo, 'tools', 'build-site.sh'), path.join(tmp, `out-${extra ? 1 : 0}`)], {
      encoding: 'utf8', env: { ...process.env, SITE_RUN: '1', SITE_SHA: '0123456789abcdef0123456789abcdef01234567' },
    });
  };
  try {
    const bad = run(true);
    assert.notEqual(bad.status, 0);
    assert.match(bad.stdout + bad.stderr, /::error::symlinks or special files in the site:\n\.\/licence\.txt/);
    const clean = run(false);                 // goes past the check (and stops later: no app code here)
    assert.doesNotMatch(clean.stdout + clean.stderr, /symlinks/);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});
