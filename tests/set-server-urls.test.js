'use strict';
// tools/set-server-urls.js writes the repository variable SERVER_URLS into the
// page's script at deploy time: it must accept origins only, and must find the
// SERVER_URLS line in the real index.html (or the override silently stops working).
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const { parseUrls, setServerUrls } = require('../tools/set-server-urls');

const INDEX = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');

test('index.html has exactly one SERVER_URLS line, and the override rewrites it', () => {
  assert.equal(INDEX.match(/^const SERVER_URLS = \[.*\];$/gm).length, 1);
  const { html } = setServerUrls(INDEX, 'https://mosquito.charity, https://mosquito-production.up.railway.app/');
  assert.match(html, /^const SERVER_URLS = \['https:\/\/mosquito\.charity', 'https:\/\/mosquito-production\.up\.railway\.app'\];$/m);
  const rest = (s) => s.replace(/^const SERVER_URLS.*$/m, '');
  assert.equal(rest(html), rest(INDEX), 'only that line changed');
});

test('origins are normalised and de-duplicated', () => {
  assert.deepEqual(parseUrls(' HTTPS://Mosquito.Charity/ ,https://mosquito.charity,https://api.example.org:8443'),
                   ['https://mosquito.charity', 'https://api.example.org:8443']);
  assert.deepEqual(parseUrls('http://127.0.0.1:5001,http://localhost'), ['http://127.0.0.1:5001', 'http://localhost']);
});

test('anything but an origin is refused', () => {
  for (const bad of ['', ' , ', 'mosquito.charity', 'http://mosquito.charity', 'https://mosquito.charity/api',
                     "https://x.org'];alert(1);//", 'https://x.org?a=1', 'javascript:alert(1)', 'https://localhost',
                     'https://-x.org', 'https://x', 'ftp://x.org', 'https://x.org:99999999']) {
    assert.throws(() => parseUrls(bad), /not an origin|no endpoints/, JSON.stringify(bad));
  }
});

test('the e2e suites can never reach a backend the app is configured to call', () => {
  const { leavesTestMachine } = require('./e2e/run-app');
  const urls = INDEX.match(/^const SERVER_URLS = \[(.*)\];$/m)[1].match(/'[^']+'/g).map(s => s.slice(1, -1));
  assert.ok(urls.length >= 1);
  for (const u of urls) assert.ok(leavesTestMachine(new URL(u)), `${u} must be blocked in tests`);
  for (const u of ['http://127.0.0.1:5001/health', 'http://localhost:8080/', 'http://[::1]:9/x']) {
    assert.ok(!leavesTestMachine(new URL(u)), `${u} is a local test server`);
  }
});
