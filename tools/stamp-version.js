#!/usr/bin/env node
'use strict';
/*
 * Give a built index.html the version of THIS deployment:
 *
 *   node tools/stamp-version.js <index.html> <run-number> <commit-sha> [YYYY-MM-DD]
 *
 * APP_VERSION keeps its hand-set major.minor and takes the deploy's run number
 * as the last part (3.1.0 → 3.1.42), so every deployment shows a new, higher
 * version without anyone remembering to bump it. APP_BUILD becomes
 * "<commit> · <date>" (shown in About → Advanced). Used by the Pages deploy.
 */
const fs = require('fs');

const VERSION_LINE = /^const APP_VERSION = '(\d+)\.(\d+)\.\d+';$/m;
const BUILD_LINE = /^const APP_BUILD = '[^'\n]*';$/m;
const SERVER_APP_VERSION = /^[0-9A-Za-z.+-]{1,16}$/;   // what the backend accepts (federated_server.py)

function stampVersion(html, run, sha, date = new Date().toISOString().slice(0, 10)) {
  const n = String(run);
  if (!/^[1-9]\d{0,6}$/.test(n)) throw new Error(`run number must be a positive integer: ${JSON.stringify(run)}`);
  if (!/^[0-9a-f]{7,40}$/.test(String(sha))) throw new Error(`not a commit SHA: ${JSON.stringify(sha)}`);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(String(date))) throw new Error(`date must be YYYY-MM-DD: ${JSON.stringify(date)}`);
  const m = html.match(VERSION_LINE);
  if (!m) throw new Error("`const APP_VERSION = 'X.Y.Z';` line not found in index.html");
  if (!BUILD_LINE.test(html)) throw new Error("`const APP_BUILD = '…';` line not found in index.html");
  const version = `${m[1]}.${m[2]}.${n}`;
  if (!SERVER_APP_VERSION.test(version)) throw new Error(`version ${version} would be rejected by the server`);
  const build = `${String(sha).slice(0, 7)} · ${date}`;
  return {
    html: html.replace(VERSION_LINE, `const APP_VERSION = '${version}';`)
              .replace(BUILD_LINE, `const APP_BUILD = '${build}';`),
    version, build,
  };
}

module.exports = { stampVersion };

if (require.main === module) {
  const [file, run, sha, date] = process.argv.slice(2);
  try {
    if (!file) throw new Error('usage: stamp-version.js <index.html> <run-number> <commit-sha> [YYYY-MM-DD]');
    const { html, version, build } = stampVersion(fs.readFileSync(file, 'utf8'), run, sha, date || undefined);
    fs.writeFileSync(file, html);
    console.log(`APP_VERSION → ${version} (${build})`);
  } catch (e) {
    console.error(`::error::stamp-version: ${e.message}`);
    process.exit(1);
  }
}
