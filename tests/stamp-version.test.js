'use strict';
// The Pages deploy runs tools/stamp-version.js so every deployment shows a new
// version. It must find the version lines in the real index.html (or the stamp
// silently stops and every deploy shows the same version again), change only
// those lines, and produce a version the backend accepts as appVersion.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const { stampVersion } = require('../tools/stamp-version');

const INDEX = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const SHA = 'de25d7280d6803931d27811c5bbf0cd3e7d876ec';

test('index.html has exactly one APP_VERSION and one APP_BUILD line', () => {
  assert.equal(INDEX.match(/^const APP_VERSION = '\d+\.\d+\.\d+';$/gm).length, 1);
  assert.equal(INDEX.match(/^const APP_BUILD = '[^']*';$/gm).length, 1);
});

test('the run number becomes the last part of the version; only those two lines change', () => {
  const base = INDEX.match(/^const APP_VERSION = '(\d+\.\d+)\.\d+';$/m)[1];
  const { html, version, build } = stampVersion(INDEX, 42, SHA, '2026-10-27');
  assert.equal(version, `${base}.42`);
  assert.equal(build, 'de25d72 · 2026-10-27');
  assert.match(html, new RegExp(`^const APP_VERSION = '${base.replace('.', '\\.')}\\.42';$`, 'm'));
  assert.match(html, /^const APP_BUILD = 'de25d72 · 2026-10-27';$/m);
  const rest = (s) => s.replace(/^const APP_(VERSION|BUILD) = .*$/gm, '');
  assert.equal(rest(html), rest(INDEX));
  assert.notEqual(stampVersion(INDEX, 43, SHA, '2026-10-27').version, version, 'each deploy differs');
});

test('the stamped version is one the server accepts as appVersion', () => {
  const server = fs.readFileSync(path.join(__dirname, '..', 'federated_server.py'), 'utf8');
  const re = new RegExp(`^${server.match(/APP_VERSION_RE = re\.compile\(r'([^']+)'\)/)[1]}$`);
  assert.match(stampVersion(INDEX, 9999999, SHA, '2026-10-27').version, re);
});

test('bad inputs are refused', () => {
  for (const run of [0, -1, '1.5', 'x', '', '12345678']) assert.throws(() => stampVersion(INDEX, run, SHA), /run number/);
  for (const sha of ['', 'xyz1234', "de25d72';alert(1)"]) assert.throws(() => stampVersion(INDEX, 1, sha), /commit SHA/);
  assert.throws(() => stampVersion(INDEX, 1, SHA, '27/10/2026'), /YYYY-MM-DD/);
  assert.throws(() => stampVersion('<html></html>', 1, SHA), /not found/);
});
