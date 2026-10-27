#!/usr/bin/env node
'use strict';
/*
 * Minify a BUILT site in place (never the repository's own files):
 *
 *   node tools/minify-site.js <site-dir>
 *
 * index.html (with its inline CSS and JS), js/*.js and sw.js lose comments and
 * whitespace and get local names shortened. Top-level names stay as they are:
 * the page's inline script calls the engine scripts through them. Vendor files
 * are already minified. No source maps are written, so the published copy does
 * not point back to the readable source. Used by tools/build-site.sh.
 */
const fs = require('fs');
const path = require('path');
const { minify: minifyJs } = require('terser');
const { minify: minifyHtml } = require('html-minifier-terser');

const TERSER = {
  ecma: 2020,
  compress: { passes: 2 },
  mangle: true,                       // locals only: toplevel stays false
  format: { comments: false },
};

async function minifySite(dir) {
  const report = [];
  const size = (f) => fs.statSync(f).size;
  const js = [path.join(dir, 'sw.js'),
              ...fs.readdirSync(path.join(dir, 'js')).filter(f => f.endsWith('.js')).map(f => path.join(dir, 'js', f))];
  for (const file of js) {
    const before = size(file);
    const out = await minifyJs(fs.readFileSync(file, 'utf8'), TERSER);
    if (!out.code) throw new Error(`terser produced nothing for ${file}`);
    fs.writeFileSync(file, out.code);
    report.push([path.relative(dir, file), before, size(file)]);
  }
  const html = path.join(dir, 'index.html');
  const before = size(html);
  const out = await minifyHtml(fs.readFileSync(html, 'utf8'), {
    collapseWhitespace: true,
    conservativeCollapse: true,       // keep one space where one was: no layout changes
    removeComments: true,
    minifyCSS: true,
    minifyJS: TERSER,                 // the inline app script (JSON-LD is left alone)
    decodeEntities: false,
  });
  fs.writeFileSync(html, out);
  report.push(['index.html', before, size(html)]);
  return report;
}

module.exports = { minifySite };

if (require.main === module) {
  const dir = process.argv[2];
  if (!dir || !fs.existsSync(path.join(dir, 'index.html'))) {
    console.error('::error::usage: minify-site.js <built-site-dir containing index.html>');
    process.exit(1);
  }
  minifySite(dir).then((report) => {
    for (const [f, a, b] of report) console.log(`minified ${f}: ${a} → ${b} bytes`);
  }).catch((e) => {
    console.error(`::error::minify-site: ${e.message}`);
    process.exit(1);
  });
}
