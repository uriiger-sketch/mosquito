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
 *   • Static assets    → stale-while-revalidate (instant load, refresh behind).
 *   • Map tiles        → cache-first (immutable, and the whole point offline).
 *   • The API + anything non-GET → never touched. Detection uploads must reach
 *     the network untouched (the client deliberately uses XHR for them).
 */

const VERSION      = 'v1';
const SHELL_CACHE  = `mn-shell-${VERSION}`;
const ASSET_CACHE  = `mn-assets-${VERSION}`;
const TILE_CACHE   = 'mn-tiles';          // unversioned: tiles never change
const MAX_TILES    = 400;

const SHELL_URLS = ['./', './index.html'];
const ASSET_URLS = [
  './icon.svg', './favicon-32.png', './favicon-16.png',
  './apple-touch-icon.png', './icon-192.png', './icon-512.png',
  './manifest.webmanifest',
];

self.addEventListener('install', (e) => {
  e.waitUntil((async () => {
    // Pre-cache best-effort: one missing asset must not abort the install.
    const shell = await caches.open(SHELL_CACHE);
    await Promise.allSettled(SHELL_URLS.map((u) => shell.add(u)));
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
        const res = await fetch(req);
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

  // Same-origin static assets: stale-while-revalidate.
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
