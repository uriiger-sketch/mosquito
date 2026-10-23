/* MosquitoNet service worker.
 *
 * Replaces an inline blob-URL worker that browsers ALWAYS reject (a service
 * worker script must be same-origin HTTP(S); `blob:` is not a valid script URL),
 * so the app previously had no offline support at all and Chrome never fired
 * `beforeinstallprompt` — the "Add to Home Screen" banner was dead.
 *
 * Caching strategy, chosen so the app can never get stuck on a stale build:
 *   • The page itself  → network-first, cache as fallback. Updates always land
 *     when online; the last good copy serves the app when offline.
 *   • App code (JS/CSS/manifest, same origin) → network-first, cache fallback —
 *     the same rule as the page, so a fresh index.html can never run against a
 *     stale cached js/detector.js (version skew between the split files).
 *   • Other static assets (icons, map data) → stale-while-revalidate.
 *   • Map tiles        → cache-first (immutable, and the whole point offline).
 *   • The API + anything non-GET → never touched. Detection uploads must reach
 *     the network untouched (the client deliberately uses XHR for them).
 */

const VERSION      = 'v4';
const SHELL_CACHE  = `mn-shell-${VERSION}`;
const ASSET_CACHE  = `mn-assets-${VERSION}`;
const TILE_CACHE   = 'mn-tiles';          // unversioned: tiles never change
const MAX_TILES    = 400;

const SHELL_URLS = [
  './', './index.html',
  './js/species.js', './js/audio-engine.js', './js/detector.js',
  './vendor/leaflet/leaflet.js', './vendor/leaflet/leaflet.css',
];
const ASSET_URLS = [
  './icon.svg', './favicon-32.png', './favicon-16.png',
  './apple-touch-icon.png', './icon-192.png', './icon-512.png',
  './map-data.json',
  './assets/fonts/instrument-sans-latin-wght.woff2',
  './assets/fonts/instrument-serif-latin-italic.woff2',
];

self.addEventListener('install', (e) => {
  e.waitUntil((async () => {
    // Pre-cache best-effort: one missing asset must not abort the install.
    const shell = await caches.open(SHELL_CACHE);
    // cache: 'reload' — never precache a copy the HTTP cache is still holding.
    await Promise.allSettled(SHELL_URLS.map((u) => shell.add(new Request(u, { cache: 'reload' }))));
    const assets = await caches.open(ASSET_CACHE);
    await Promise.allSettled(ASSET_URLS.map((u) => assets.add(u)));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    const keep = new Set([SHELL_CACHE, ASSET_CACHE, TILE_CACHE]);
    for (const k of await caches.keys()) {
      if (k.startsWith('mn-') && !keep.has(k)) await caches.delete(k);
    }
    await self.clients.claim();
  })());
});

async function trimCache(name, max) {
  try {
    const c = await caches.open(name);
    const keys = await c.keys();
    for (let i = 0; i < keys.length - max; i++) await c.delete(keys[i]);
  } catch (e) { /* cache eviction is best-effort */ }
}

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;                    // uploads pass through

  let url;
  try { url = new URL(req.url); } catch (e) { return; }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return;

  // Never intercept the API — detections, heartbeats and federated traffic must
  // always hit the network directly and must never be served from a cache.
  if (url.hostname === 'mosquito.charity') return;

  // Map tiles: cache-first, and bounded so long sessions can't fill storage.
  if (url.hostname.endsWith('tile.openstreetmap.org')) {
    event.respondWith((async () => {
      const cache = await caches.open(TILE_CACHE);
      const hit = await cache.match(req);
      if (hit) return hit;
      try {
        const res = await fetch(req);
        if (res && res.ok) { cache.put(req, res.clone()); trimCache(TILE_CACHE, MAX_TILES); }
        return res;
      } catch (e) {
        return hit || Response.error();
      }
    })());
    return;
  }

  // The page: network-first so a new build always wins, cache as offline fallback.
  if (req.mode === 'navigate') {
    event.respondWith((async () => {
      try {
        // A navigate-mode Request can't be re-fetched with an init (TypeError):
        // rebuild it from the URL to revalidate past the HTTP cache.
        const res = await fetch(new Request(req.url, { cache: 'no-cache', credentials: 'same-origin' }));
        if (res && res.ok) (await caches.open(SHELL_CACHE)).put('./index.html', res.clone());
        return res;
      } catch (e) {
        const cache = await caches.open(SHELL_CACHE);
        return (await cache.match('./index.html')) || (await cache.match('./')) ||
               new Response('Offline', { status: 503, headers: { 'Content-Type': 'text/plain' } });
      }
    })());
    return;
  }

  // Same-origin app code: network-first (see header — prevents version skew).
  if (url.origin === self.location.origin &&
      (req.destination === 'script' || req.destination === 'style' ||
       req.destination === 'worker' || req.destination === 'audioworklet' ||
       req.destination === 'manifest' || /\.(m?js|css|webmanifest)$/.test(url.pathname))) {
    event.respondWith((async () => {
      const cache = await caches.open(SHELL_CACHE);
      try {
        const res = await fetch(req, { cache: 'no-cache' });   // revalidate: no HTTP-cache skew
        if (res && res.ok) cache.put(req, res.clone());
        return res;
      } catch (e) {
        return (await cache.match(req, { ignoreSearch: true })) || Response.error();
      }
    })());
    return;
  }

  // Other same-origin static assets: stale-while-revalidate.
  if (url.origin === self.location.origin) {
    event.respondWith((async () => {
      const cache = await caches.open(ASSET_CACHE);
      const hit = await cache.match(req);
      const net = fetch(req).then((res) => {
        if (res && res.ok) cache.put(req, res.clone());
        return res;
      }).catch(() => null);
      return hit || (await net) || Response.error();
    })());
  }
  // Cross-origin (fonts, Leaflet CDN): leave to the browser.
});
