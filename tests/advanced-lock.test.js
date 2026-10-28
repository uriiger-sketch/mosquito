'use strict';
// About → Advanced opens only when the SERVER accepts the admin password
// (POST /admin/login, checked against its ADMIN_PASSWORD). The page is public,
// so it must never carry the password in any form: a hash shipped in it could
// be guessed offline, and that password also guards the admin views.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const CLIENT = ['index.html', 'sw.js', ...fs.readdirSync(path.join(ROOT, 'js')).map(f => path.join('js', f))];

test('the client ships no password verifier', () => {
  for (const f of CLIENT) {
    const src = fs.readFileSync(path.join(ROOT, f), 'utf8');
    assert.doesNotMatch(src, /[0-9a-f]{40,}/i, `${f}: a long hex constant (a password hash?)`);
    assert.doesNotMatch(src, /PW_SHA|sha256Hex|crypto\.subtle\.digest/, `${f}: client-side password hashing`);
  }
});

test('Advanced unlocks only through the server login', () => {
  const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
  const wire = html.match(/^ {2}wireAdvanced\(\) \{\n[\s\S]*?\n {2}\}\n/m);
  assert.ok(wire, 'wireAdvanced() not found');
  assert.match(wire[0], /await this\.adminLogin\(pw\.value\)/);
  // The only place that unlocks it comes after the server said yes.
  const unlock = wire[0].indexOf('this._advUnlocked = true');
  assert.ok(unlock > wire[0].indexOf('if (!r.ok)'), 'unlocked before the server answered');
});
