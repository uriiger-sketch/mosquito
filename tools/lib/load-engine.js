'use strict';
/*
 * Load the browser detection engine into Node exactly as the page runs it:
 * js/species.js then js/detector.js evaluated as classic scripts in ONE vm
 * context, so their top-level declarations share a global lexical scope just
 * like <script> tags do. Returns the engine's public names.
 *
 *   const { MosquitoClassifier, SPECIES_DB } = require('./tools/lib/load-engine')();
 *   const eng = require('./tools/lib/load-engine')({ root: '/path/to/checkout' });
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const DEFAULT_ROOT = path.join(__dirname, '..', '..');
const FILES = ['js/species.js', 'js/detector.js'];

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
  return vm.runInContext(`({
    SPECIES_DB, getNativeSpecies, getLocalDisease, solarTimes, timeOfDayCategory,
    FarFieldTracker, MosquitoClassifier,
  })`, ctx);
};
