// Service Worker PWA - Apex Quant Markets v1.0
// - Solo intercepta recursos del propio sitio (CDN y fuentes pasan directos, sin guardarse).
// - La clave de caché ignora la query (?t=...): datos.json ocupa UNA sola entrada que se va sobrescribiendo.
// - Tope de entradas por si algún día se añaden más archivos.
const CACHE_NAME = 'apex-mercados-v1.0';
const SHELL = ['./', './index.html', './manifest.json', './icon-192.png', './icon-512.png'];
const MAX_ENTRADAS = 30;

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE_NAME).then((c) => c.addAll(SHELL)).catch(() => {}));
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

async function recortar(cache) {
  const keys = await cache.keys();
  if (keys.length > MAX_ENTRADAS) {
    await Promise.all(keys.slice(0, keys.length - MAX_ENTRADAS).map((k) => cache.delete(k)));
  }
}

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;          // recursos externos: sin tocar

  const clave = new Request(url.origin + url.pathname);     // sin query -> entradas acotadas
  event.respondWith((async () => {
    const cache = await caches.open(CACHE_NAME);
    try {
      const res = await fetch(req);                         // red primero: datos y web siempre frescos
      if (res && res.status === 200 && res.type === 'basic') {
        await cache.put(clave, res.clone());
        recortar(cache);
      }
      return res;
    } catch (err) {
      const guardado = await cache.match(clave);            // sin conexión: última copia buena
      if (guardado) return guardado;
      if (req.mode === 'navigate') {
        const portada = await cache.match('./index.html');
        if (portada) return portada;
      }
      return Response.error();
    }
  })());
});
