// Service worker for Cloud Cost.
// Caches the app shell so the UI still loads offline (it already has embedded
// fallback pricing data client-side). Both the API and the shell document go
// network-first: if they fail, the app's own tryBackend() fallback (for API
// calls) or the cached shell (for the document) takes over. Network-first for
// the document matters because this is a single-file app under active
// development — a stale cached copy of index.html would silently keep
// serving old UI/JS forever otherwise. Bump CACHE_NAME on any shell change
// that must invalidate old caches immediately.
const CACHE_NAME = "cloud-cost-shell-v2";
const SHELL_URLS = ["/"];

self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(SHELL_URLS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", event => {
  if (event.request.method !== "GET") return;
  const url = new URL(event.request.url);
  if (url.origin !== location.origin) return; // let cross-origin (CDN) requests pass through normally

  if (url.pathname.startsWith("/api/")) {
    event.respondWith(
      fetch(event.request).catch(() => new Response(null, { status: 503, statusText: "Offline" }))
    );
    return;
  }

  // Network-first for the shell itself, so redeploys are picked up on the very
  // next load instead of waiting for a background revalidation to catch up.
  event.respondWith(
    fetch(event.request).then(resp => {
      if (resp && resp.ok) {
        const clone = resp.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
      }
      return resp;
    }).catch(() => caches.match(event.request))
  );
});
