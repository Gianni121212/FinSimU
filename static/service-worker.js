const CACHE_NAME = 'finsimu-cache-v2'; // 更新緩存版本號 (如果之前有舊版本)
    const urlsToCache = [
      '/',              // 对应 app.py 中的 finsimu_landing_route
      '/login',         // 对应 app.py 中的 finsimu_login_route
      '/register',      // 对应 app.py 中的 finsimu_register_route
      '/app',           // 对应 app.py 中的 finsimu_app_route
      // 如果你有全局的 CSS 或 JS 文件 (目前看來沒有，都是內聯的)，也需要加入
      // 例如: '/static/css/style.css', '/static/js/main.js'
      // 也可加入 logo 等重要靜態資源
      '/static/icons/icon-192.png',
      '/static/icons/icon-512.png'
      // 注意：manifest.json 通常不需要被 Service Worker 緩存，瀏覽器會自己處理
    ];

    self.addEventListener('install', function(event) {
      console.log('[Service Worker] Install event');
      event.waitUntil(
        caches.open(CACHE_NAME).then(function(cache) {
          console.log('[Service Worker] Opened cache:', CACHE_NAME);
          return cache.addAll(urlsToCache)
            .then(() => console.log('[Service Worker] Files cached successfully'))
            .catch(error => console.error('[Service Worker] Failed to cache files:', error, urlsToCache));
        })
      );
      self.skipWaiting(); // 強制新的 Service Worker 立即激活
    });

    self.addEventListener('activate', event => {
      console.log('[Service Worker] Activate event');
      // 清理舊緩存
      event.waitUntil(
        caches.keys().then(cacheNames => {
          return Promise.all(
            cacheNames.map(cache => {
              if (cache !== CACHE_NAME) {
                console.log('[Service Worker] Clearing old cache:', cache);
                return caches.delete(cache);
              }
            })
          );
        })
      );
      return self.clients.claim(); // 讓新的 Service Worker 立即控制所有打開的客戶端
    });

    self.addEventListener('fetch', function(event) {
      // 我們只對 GET 請求進行緩存處理
      if (event.request.method !== 'GET') {
        event.respondWith(fetch(event.request));
        return;
      }

      // 對於 API 請求，總是從網絡獲取 (不緩存 API 響應)
      if (event.request.url.includes('/api/')) {
        event.respondWith(fetch(event.request));
        return;
      }
      
      // 對於導航請求 (HTML 頁面)，嘗試網絡優先，如果失敗則回退到緩存的 app shell
      // 這有助於用戶總是獲取最新的頁面，但如果離線，則顯示緩存的版本
      if (event.request.mode === 'navigate') {
        event.respondWith(
          fetch(event.request)
            .catch(() => {
              // 如果網絡請求失敗 (例如離線)，嘗試從緩存中獲取 /app (或你的主要入口點)
              // 注意: 這需要 '/app' 真的在 urlsToCache 中並且緩存成功
              return caches.match('/app') || caches.match('/'); // 回退到主應用頁面或根路徑
            })
        );
        return;
      }

      // 對於其他靜態資源 (CSS, JS, 圖片等)，使用 Cache First 策略
      event.respondWith(
        caches.match(event.request).then(function(response) {
          if (response) {
            // console.log('[Service Worker] Serving from cache:', event.request.url);
            return response;
          }
          // console.log('[Service Worker] Fetching from network:', event.request.url);
          return fetch(event.request).then(function(networkResponse) {
            // 如果需要，可以將新的靜態資源動態添加到緩存中
            // 但對於 urlsToCache 中已有的，通常在 install 時已緩存
            // if (networkResponse && networkResponse.status === 200 && !event.request.url.includes('/api/')) {
            //   const responseToCache = networkResponse.clone();
            //   caches.open(CACHE_NAME).then(function(cache) {
            //     cache.put(event.request, responseToCache);
            //   });
            // }
            return networkResponse;
          }).catch(error => {
            console.error('[Service Worker] Fetch failed; returning offline page instead.', error);
            // 可以提供一個通用的離線頁面
            // return caches.match('/offline.html'); // 如果你有 offline.html
          });
        })
      );
    });