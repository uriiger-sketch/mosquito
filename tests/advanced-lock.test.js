'use strict';
// About → Advanced is unlocked by comparing sha256Hex(input) with the hash in
// index.html. The hash is computed by the page's own small SHA-256 (it must work
// where crypto.subtle doesn't), so it is checked here against Node's crypto.
const test = require('node:test');
const assert = require('node:assert/strict');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const INDEX = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const src = INDEX.match(/^function sha256Hex\(str\) \{\n[\s\S]*?\n\}\n/m);
const sha256Hex = src && new Function(`${src[0]}; return sha256Hex;`)();
const nodeSha = (s) => crypto.createHash('sha256').update(s, 'utf8').digest('hex');

test('index.html defines sha256Hex and a SHA-256 for the Advanced password', () => {
  assert.ok(sha256Hex, 'function sha256Hex(str) not found at top level');
  assert.match(INDEX, /^const ADVANCED_PW_SHA256 = '[0-9a-f]{64}';$/m);
});

test('sha256Hex matches Node crypto, across block boundaries and UTF-8', () => {
  const inputs = ['', 'abc', 'password', 'a'.repeat(55), 'a'.repeat(56), 'a'.repeat(63), 'a'.repeat(64),
                  'b'.repeat(119), 'c'.repeat(1000), 'עברית 🦟 ünïcödé'];
  for (const s of inputs) assert.equal(sha256Hex(s), nodeSha(s), JSON.stringify(s.slice(0, 24)));
});
