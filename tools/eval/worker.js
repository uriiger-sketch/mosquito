'use strict';
// Worker thread: loads the engine of the checkout under test once, then runs jobs.
const { parentPort, workerData } = require('worker_threads');
const loadEngine = require('../lib/load-engine');
const { execute } = require('./jobs');

const eng = loadEngine({ root: workerData.root });
parentPort.on('message', ({ idx, job }) => {
  try { parentPort.postMessage({ idx, r: execute(job, eng) }); }
  catch (e) { parentPort.postMessage({ idx, err: e && e.stack ? e.stack : String(e) }); }
});
