'use strict';
// What a phone uploads must never average back to its home — not even when the
// browser's storage is wiped between sessions (private tabs, Safari's 7-day
// limit), which redraws the secret offset. Runs the REAL FederatedLearningClient
// from index.html in a vm with a wipeable localStorage. See _locationOffset.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const src = html.match(/^class FederatedLearningClient \{[\s\S]*?\n\}\n/m);

const HOME = [32.0755, 34.7955];            // ≈ 800 m from the centre of its 0.02° cell
const CELL = [32.07, 34.79];
const M = 111320, COS = Math.cos(HOME[0] * Math.PI / 180);
const metres = (a, b) => Math.hypot((a[0] - b[0]) * M, (a[1] - b[1]) * M * COS);
const SPECIES = { id: 'culex', name: 'Culex', disease: 'West Nile', risk: 'MEDIUM' };

function harness() {
  let store = new Map();
  const ctx = {
    localStorage: { getItem: k => (store.has(k) ? store.get(k) : null), setItem: (k, v) => store.set(k, String(v)),
                    removeItem: k => store.delete(k) },
    crypto: globalThis.crypto, console: { log() {}, warn() {} }, Math, JSON, Date, Number, Array,
    Uint8Array, Uint32Array, parseFloat, setTimeout() {}, clearTimeout() {},
    window: { _app: { deviceLat: HOME[0], deviceLng: HOME[1] }, SERVER_URL: '' },
    APP_VERSION: '3.1.0', fetchWithTimeout() {}, XMLHttpRequest: function () {},
  };
  vm.createContext(ctx);
  vm.runInContext(`${src[0]}\nthis.FederatedLearningClient = FederatedLearningClient;`, ctx);
  return {
    // One listening session on fresh storage: a new id and a new secret offset.
    wipedSession(finds) {
      store = new Map();
      const fc = new ctx.FederatedLearningClient();
      for (let i = 0; i < finds; i++) fc.reportDetection({ species: SPECIES, confidence: 0.9, frequency: 455, f0: 455 });
      return fc._loadQueue().map(p => [p.lat, p.lng]);
    },
  };
}

test('index.html defines FederatedLearningClient at top level', () => assert.ok(src));

test('averaging many storage-wiped sessions ends at the grid cell, not the home', () => {
  const h = harness();
  const pts = [];
  for (let s = 0; s < 300; s++) pts.push(...h.wipedSession(4));
  const mean = [pts.reduce((a, p) => a + p[0], 0) / pts.length, pts.reduce((a, p) => a + p[1], 0) / pts.length];
  assert.ok(metres(mean, CELL) < 200, `mean ${metres(mean, CELL).toFixed(0)} m from the cell centre`);
  assert.ok(metres(mean, HOME) > 500, `mean only ${metres(mean, HOME).toFixed(0)} m from the home`);
});

test('every report stays within the documented distance of the true fix', () => {
  // ≤ half a cell diagonal (~1.6 km here) + the ~1.5 km offset (+ jitter).
  const h = harness();
  for (let s = 0; s < 50; s++) {
    for (const p of h.wipedSession(2)) assert.ok(metres(p, HOME) < 3300, `${metres(p, HOME).toFixed(0)} m`);
  }
});
