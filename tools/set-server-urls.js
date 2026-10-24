#!/usr/bin/env node
'use strict';
/*
 * Point a built index.html at a list of backend endpoints, tried in order:
 *
 *   node tools/set-server-urls.js <index.html> "https://mosquito.charity,https://x.up.railway.app"
 *
 * Used by the Pages deploy (repository variable SERVER_URLS) and the e2e tests.
 * Every entry must be an origin — https://host[:port], or http on localhost for
 * tests — so nothing but a URL can ever be written into the page's script.
 */
const fs = require('fs');

const ORIGIN = /^(https:\/\/[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+|http:\/\/(localhost|127\.0\.0\.1))(:\d{1,5})?$/;
const LINE = /^const SERVER_URLS = \[.*\];$/m;

function parseUrls(raw) {
  const urls = String(raw || '').split(',').map(s => s.trim().toLowerCase().replace(/\/+$/, '')).filter(Boolean);
  if (!urls.length) throw new Error('no endpoints given');
  for (const u of urls) {
    if (!ORIGIN.test(u)) throw new Error(`not an origin (https://host[:port], no path): ${JSON.stringify(u)}`);
  }
  return [...new Set(urls)];
}

function setServerUrls(html, raw) {
  const urls = parseUrls(raw);
  if (!LINE.test(html)) throw new Error('`const SERVER_URLS = [...]` line not found in index.html');
  return { html: html.replace(LINE, `const SERVER_URLS = [${urls.map(u => `'${u}'`).join(', ')}];`), urls };
}

module.exports = { parseUrls, setServerUrls };

if (require.main === module) {
  const [file, raw] = process.argv.slice(2);
  try {
    if (!file) throw new Error('usage: set-server-urls.js <index.html> <url[,url…]>');
    const { html, urls } = setServerUrls(fs.readFileSync(file, 'utf8'), raw);
    fs.writeFileSync(file, html);
    console.log(`SERVER_URLS → ${urls.join(', ')}`);
  } catch (e) {
    console.error(`::error::set-server-urls: ${e.message}`);
    process.exit(1);
  }
}
