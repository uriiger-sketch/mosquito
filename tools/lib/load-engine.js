'use strict';
/*
 * Load the browser detection engine into Node exactly as the page runs it:
 * js/species.js then js/detector.js evaluated as classic scripts in ONE vm
 * context, so their top-level declarations share a global lexical scope just
 * like <script> tags do. Returns the engine's public names.
 *
 *   const { MosquitoClassifier, SPECIES_DB } = require('./tools/lib/load-engine')();
 *   const eng = require('./tools/lib/load-engine')({ root: '/path/to/checkout' });
 *
 * Names a given checkout does not define (e.g. getLocalTaxon on an older
 * baseline) come back as undefined instead of throwing, so the evaluation
 * harness can drive old and new engines through the same code.
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const DEFAULT_ROOT = path.join(__dirname, '..', '..');
const FILES = ['js/species.js', 'js/detector.js'];
const NAMES = ['SPECIES_DB', 'WINGBEAT_MODEL', 'getNativeSpecies', 'getLocalDisease', 'getLocalTaxon', 'solarTimes',
  'timeOfDayCategory', 'FarFieldTracker', 'MosquitoClassifier', 'SpeciesModel'];

module.exports = function loadEngine(opts = {}) {
  const root = opts.root || DEFAULT_ROOT;
  const quiet = opts.verbose ? console : {
    log: () => {}, info: () => {}, debug: () => {},
    warn: console.warn, error: console.error,
  };
  const ctx = vm.createContext({ console: quiet });
  for (const f of FILES) {
    const file = path.join(root, f);
    vm.runInContext(fs.readFileSync(file, 'utf8'), ctx, { filename: file });
  }
  // Lexical (const/class) globals are not properties of the context object;
  // read them out through a final script evaluated in the same scope.
  const body = NAMES.map(n => `${n}: typeof ${n} === 'undefined' ? undefined : ${n}`).join(',\n');
  return vm.runInContext(`({ ${body} })`, ctx);
};
