'use strict';
// getLocalTaxon: location-aware display names for stand-in catalogue entries.
const test = require('node:test');
const assert = require('node:assert/strict');
const H = require('./helpers');

const eng = H.loadEngine();
const T = eng.getLocalTaxon;

test('getLocalTaxon is exported and returns defaults without a location', () => {
  assert.equal(typeof T, 'function');
  for (const sp of eng.SPECIES_DB) {
    const r = T(sp.id, null, null);
    assert.deepEqual({ ...r }, { name: sp.name, common: sp.common, note: '', standIn: false });  // spread: vm-realm object
  }
  assert.equal(T('no_such_species', 10, 10), null);
});

test('Anopheles outside Africa is never called An. gambiae', () => {
  const places = { bangkok: [13.75, 100.5], telAviv: [32.08, 34.78], mumbai: [19.08, 72.88], riyadh: [24.7, 46.7], rome: [41.9, 12.5] };
  for (const [name, [lat, lng]] of Object.entries(places)) {
    const r = T('anopheles', lat, lng);
    assert.equal(r.name, 'Anopheles sp.', name);
    assert.equal(r.standIn, true, name);
    assert.doesNotMatch(r.common, /African/, name);
    assert.ok(r.note.length > 0, name);
  }
  assert.match(T('anopheles', 13.75, 100.5).note, /dirus/);
  assert.match(T('anopheles', 32.08, 34.78).note, /sacharovi|labranchiae/);
});

test('Anopheles in sub-Saharan Africa is An. gambiae s.l.', () => {
  const r = T('anopheles', 5.6, -0.19);                 // Accra
  assert.equal(r.name, 'Anopheles gambiae s.l.');
  assert.equal(r.standIn, false);
});

test('Toxorhynchites brevipalpis only in Africa; other stand-ins annotated', () => {
  assert.equal(T('toxorhynchites', -1.3, 36.8).name, 'Toxorhynchites brevipalpis');   // Nairobi
  assert.equal(T('toxorhynchites', 13.75, 100.5).name, 'Toxorhynchites sp.');         // Bangkok
  assert.equal(T('culex_pipiens', 1.35, 103.8).name, 'Culex pipiens complex');        // Singapore
  assert.equal(T('culex_pipiens', 48.85, 2.35).name, 'Culex pipiens');                // Paris
  assert.match(T('mansonia_uniformis', 6.5, 3.4).note, /africana/);                    // Lagos
  assert.equal(T('aedes_aegypti', 13.75, 100.5).name, 'Aedes aegypti');               // real species: unchanged
});
