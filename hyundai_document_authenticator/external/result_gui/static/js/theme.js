/*
Theme management and local-first asset fallback logic.

- Applies selected theme using data-theme on <html> and <body>.
- Persists theme in localStorage (key: 'ui.theme').
- Honors server-provided default via body[data-theme].
- Ensures Bootstrap CSS/JS present (fallback to CDN already handled in base template for CSS and here for JS if needed).
- In air-gapped scenarios where assets are missing, forces default theme to maintain readability and shows a non-blocking toast.
*/
(function(){
  'use strict';

  var STORAGE_KEY = 'ui.theme';

  function setTheme(theme) {
    try {
      document.documentElement.setAttribute('data-theme', theme);
      document.body.setAttribute('data-theme', theme);
      // Map Bootstrap data-bs-theme. Treat "default" as light so it never falls back to dark.
      if (theme === 'dark') { document.body.setAttribute('data-bs-theme', 'dark'); document.documentElement.setAttribute('data-bs-theme', 'dark'); }
      else if (theme === 'light' || theme === 'default') { document.body.setAttribute('data-bs-theme', 'light'); document.documentElement.setAttribute('data-bs-theme', 'light'); }
      else { document.body.setAttribute('data-bs-theme', ''); document.documentElement.setAttribute('data-bs-theme', ''); }
    } catch (e) {}
  }

  function getSavedTheme() {
    try { return localStorage.getItem(STORAGE_KEY) || null; } catch(e) { return null; }
  }

  function saveTheme(theme) {
    try { localStorage.setItem(STORAGE_KEY, theme); } catch(e) {}
  }

  function detectAssets() {
    // Check a few Bootstrap CSS tokens by computed style.
    var cssOk = (function(){
      try {
        var probe = document.createElement('button');
        probe.className = 'btn btn-primary';
        document.body.appendChild(probe);
        var cs = window.getComputedStyle(probe);
        var pad = parseFloat(cs.paddingLeft)||0; var rad = parseFloat(cs.borderRadius)||0;
        probe.remove();
        return pad >= 6 || rad >= 2;
      } catch(e) { return false; }
    })();

    var jsOk = typeof window.bootstrap !== 'undefined';
    return { cssOk: cssOk, jsOk: jsOk };
  }

  function showToast(message) {
    try {
      var wrap = document.createElement('div');
      wrap.className = 'position-fixed bottom-0 end-0 p-3';
      wrap.style.zIndex = 1080;
      wrap.innerHTML = [
        '<div class="toast align-items-center text-bg-warning border-0" role="alert" aria-live="assertive" aria-atomic="true">',
        '  <div class="d-flex">',
        '    <div class="toast-body">'+ message +'</div>',
        '    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>',
        '  </div>',
        '</div>'
      ].join('');
      document.body.appendChild(wrap);
      var toastEl = wrap.querySelector('.toast');
      if (window.bootstrap && window.bootstrap.Toast) {
        new window.bootstrap.Toast(toastEl, { delay: 5000 }).show();
      } else {
        // Fallback: auto-hide after 5s
        setTimeout(function(){ try { wrap.remove(); } catch(e){} }, 5000);
      }
    } catch(e) {}
  }

  document.addEventListener('DOMContentLoaded', function(){
    var serverTheme = (document.documentElement.getAttribute('data-theme') || '').trim().toLowerCase();
    var saved = getSavedTheme();
    // Prefer server-provided theme (reflects user selection), then saved.
    var theme = serverTheme || saved || 'auto';
    setTheme(theme);
    // Persist the resolved theme so it survives navigation.
    saveTheme(theme);

    // If assets missing and likely offline, fallback to default theme for readability.
    var assets = detectAssets();
    if (!assets.cssOk) {
      // Assume limited visuals; force default theme to ensure neutral palette.
      setTheme('default');
      saveTheme('default');
      showToast('Some UI assets are missing. Using default theme for stability.');
    }
  });
})();
