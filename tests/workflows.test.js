'use strict';
// Supply-chain rules for .github/workflows, checked as text (no YAML parser is
// a dependency here). Why each rule exists: see the header of deploy-pages.yml.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');

const DIR = path.join(__dirname, '..', '.github', 'workflows');
const FILES = fs.readdirSync(DIR).filter(f => /\.ya?ml$/.test(f));
const read = (f) => fs.readFileSync(path.join(DIR, f), 'utf8');

// The body of each job under `jobs:` (two-space-indented keys), comments dropped.
function jobs(src) {
  const out = {};
  const lines = src.slice(src.indexOf('\njobs:\n')).split('\n');
  let name = null;
  for (const line of lines) {
    const m = line.match(/^ {2}([A-Za-z0-9_-]+):\s*$/);
    if (m) { name = m[1]; out[name] = ''; continue; }
    if (name && !/^\s*#/.test(line)) out[name] += line + '\n';
  }
  return out;
}

test('every action is pinned to a full commit SHA', () => {
  const loose = [];
  for (const f of FILES) {
    for (const [, ref] of read(f).matchAll(/^\s*-?\s*uses:\s*(\S+)/gm)) {
      if (ref.startsWith('./')) continue;                    // local reusable workflow
      if (!/@[0-9a-f]{40}$/.test(ref)) loose.push(`${f}: ${ref}`);
    }
  }
  assert.deepEqual(loose, [], 'pin these to a commit SHA (keep the tag as a comment)');
});

test('workflows default to no permissions and only the deploy job can write', () => {
  const src = read('deploy-pages.yml');
  assert.match(src, /^permissions: \{\}$/m);
  const j = jobs(src);
  const writers = Object.entries(j).filter(([, body]) => /:\s*write\b/.test(body)).map(([n]) => n);
  assert.deepEqual(writers, ['deploy']);
  assert.match(read('ci.yml'), /^permissions:\n {2}contents: read$/m);
});

test('the job holding write tokens runs no npm code and restores no cache', () => {
  const deploy = jobs(read('deploy-pages.yml')).deploy;
  for (const banned of [/\bnpm\b/, /\bnpx\b/, /build-site/, /\bcache:/, /actions\/cache@/, /setup-node/]) {
    assert.doesNotMatch(deploy, banned, `deploy job: ${banned}`);
  }
});

test('the build job runs the minifiers without credentials, scripts or a cache', () => {
  const build = jobs(read('deploy-pages.yml')).build;
  assert.match(build, /persist-credentials: false/);
  assert.match(build, /npm ci --ignore-scripts/);
  assert.doesNotMatch(build, /\bcache:/);
  assert.doesNotMatch(build, /:\s*write\b/);
});

test('no CI checkout leaves the token in .git/config', () => {
  const src = read('ci.yml');
  const checkouts = src.match(/uses: actions\/checkout@\S+[^\n]*\n(?:\s+with:\n(?:\s{8,}.*\n)*)?/g) || [];
  assert.ok(checkouts.length >= 4);
  for (const c of checkouts) assert.match(c, /persist-credentials: false/, c);
});

test('the deploy gate includes the backend image test (Railway builds the Dockerfile)', () => {
  assert.doesNotMatch(read('deploy-pages.yml'), /skip_docker:\s*true/);
});

test('no workflow interpolates event data into a shell script', () => {
  for (const f of FILES) {
    assert.doesNotMatch(read(f), /\$\{\{\s*github\.event\./, f);
    assert.doesNotMatch(read(f), /pull_request_target/, f);
  }
});
