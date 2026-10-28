#!/usr/bin/env node
'use strict';
/*
 * Give a BUILT, minified index.html its Content-Security-Policy as a <meta>:
 *
 *   node tools/csp-meta.js <index.html> "https://mosquito.charity,https://x.up.railway.app"
 *
 * GitHub Pages cannot send response headers, so without this the Pages copy
 * runs with no CSP at all — one missed esc() there would run script next to an
 * admin session. The policy mirrors the header federated_server.py sends with
 * the page (_site_csp; tests/csp-meta.test.js keeps the two in step), with two
 * differences:
 *   • connect-src is 'self' plus the app's own endpoints (SERVER_URLS), not
 *     every https: host. On the backend both policies apply, so this narrows
 *     the page there too; the service worker runs under the header only (a
 *     <meta> never applies to workers) and keeps reaching the tile servers.
 *   • no frame-ancestors / report-uri / sandbox: a <meta> may not carry them
 *     (Chrome ignores them with a console error).
 * Must run after minification: the inline script is allowed by its exact hash.
 */
const fs = require('fs');
const crypto = require('crypto');
const { parseUrls } = require('./set-server-urls');

// Every executable inline <script>, hashed by the same rule as _site_csp.
function inlineScriptHashes(html) {
  const hashes = [];
  for (const [, attrs, code] of html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)) {
    if (/\bsrc\s*=/i.test(attrs)) continue;
    const t = attrs.match(/\btype\s*=\s*["']?([^"'\s>]+)/i);
    if (t && !['text/javascript', 'application/javascript', 'module'].includes(t[1].toLowerCase())) continue;
    hashes.push(`'sha256-${crypto.createHash('sha256').update(code, 'utf8').digest('base64')}'`);
  }
  return hashes;
}

function cspPolicy(html, rawUrls) {
  const hashes = inlineScriptHashes(html);
  if (hashes.length !== 1) throw new Error(`expected exactly 1 inline script, found ${hashes.length}`);
  const urls = parseUrls(rawUrls);
  return [
    "default-src 'self'",
    `script-src 'self' ${hashes[0]}`,
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob: https://*.tile.openstreetmap.org",
    "font-src 'self'",
    `connect-src 'self' ${urls.join(' ')}`,
    "manifest-src 'self' blob:",
    "worker-src 'self'",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'none'",
  ].join('; ');
}

function addCspMeta(html, rawUrls) {
  if (/http-equiv=["']?content-security-policy/i.test(html)) throw new Error('index.html already has a CSP <meta>');
  const anchor = /<meta charset=["']?utf-8["']?\s*\/?>/i;
  if (!anchor.test(html)) throw new Error('<meta charset="utf-8"> not found');
  const policy = cspPolicy(html, rawUrls);
  // Right after the charset, before anything the policy must govern.
  return { html: html.replace(anchor, (m) => `${m}<meta http-equiv="Content-Security-Policy" content="${policy}">`), policy };
}

module.exports = { inlineScriptHashes, cspPolicy, addCspMeta };

if (require.main === module) {
  const [file, raw] = process.argv.slice(2);
  try {
    if (!file) throw new Error('usage: csp-meta.js <index.html> <url[,url…]>');
    const { html, policy } = addCspMeta(fs.readFileSync(file, 'utf8'), raw);
    fs.writeFileSync(file, html);
    console.log(`CSP meta → ${policy.replace(/'sha256-[^']+'/, "'sha256-…'")}`);
  } catch (e) {
    console.error(`::error::csp-meta: ${e.message}`);
    process.exit(1);
  }
}
