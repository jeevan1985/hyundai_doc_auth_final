/*
Client-side controls for the Results panel.

This script injects and manages four UI controls that operate on the currently
rendered dataset only (no server round-trip):

1) Key Mode: Adds a "Simplify result" option to prune top_similar_docs per the
   active Score OP/Score filter, mutating only the cell content for the current
   DOM row set. Leaves row count unchanged.

2) Hide columns: Multi-select control populated from the table header. Toggling
   visibility is local-only and persisted in localStorage per user.

3) Rows per page: Number input (5–500, default 20) that re-paginates the
   existing client-side dataset (the rows currently in the table when the page
   loaded). Navigation for client-side pages is provided without reloading.

4) Export buttons: CSV and Excel exports of exactly the current on-screen view
   (filtered rows, hidden columns removed, simplified cells included) without
   making any server calls.

The implementation is defensive and will no-op if expected elements are not
present. It avoids interfering with existing server-side filters/pagination.
*/

(function () {
  'use strict';

  /**
   * Utility: Get current username for namespaced localStorage keys.
   * Falls back to 'anonymous' if not available.
   */
  function getCurrentUsername() {
    const body = document.body;
    const fromDataAttr = body ? body.getAttribute('data-username') : null;
    if (fromDataAttr && fromDataAttr.trim()) return fromDataAttr.trim();
    const navTextEl = document.querySelector('nav .navbar-text');
    if (navTextEl) {
      const txt = navTextEl.textContent || '';
      const m = txt.match(/^(\S+)/);
      if (m && m[1]) return m[1];
    }
    return 'anonymous';
  }

  const USERNAME = getCurrentUsername();
  const LS_PREFIX = 'result_gui';

  function lsKey(key) { return `${LS_PREFIX}.${key}.${USERNAME}`; }

  /**
   * Utility: Find main table and parse headers and current DOM rows into a
   * dataset structure.
   */
  function findResultsTable() {
    // Prefer a table with explicit id if present; fallback to first .table in content.
    const table = document.querySelector('#resultsTable') || document.querySelector('main .table');
    if (!table) return null;

    // Opt-in override: allow controls when data-enable-controls="true"
    const explicitEnable = table.dataset && table.dataset.enableControls === 'true';

    // Opt-out: explicit disable flag; allow admin path when explicitly enabled
    if ((table.dataset && table.dataset.enableControls === 'false')) return null;
    try {
      const path = (window.location && window.location.pathname) ? window.location.pathname : '';
      if (!explicitEnable && path && path.indexOf('/admin') !== -1) return null;
    } catch (_e) { /* no-op */ }

    const thead = table.tHead || table.querySelector('thead');
    const tbody = table.tBodies[0] || table.querySelector('tbody');
    if (!thead || !tbody) return null;

    // Safety guard: if the table body contains interactive form controls, skip
    // attaching client-side re-rendering logic to avoid breaking forms/buttons
    // (e.g., Manage Users page). Pages can opt-in via data-enable-controls="true".
    if (!explicitEnable) {
      const hasInteractive = tbody.querySelector('form, button, input, select, textarea, a[href]');
      if (hasInteractive) return null;
    }

    const headers = Array.from(thead.querySelectorAll('th')).map((th) => (th.textContent || '').trim());
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const data = rows.map((tr) => Array.from(tr.querySelectorAll('td')).map((td) => td.textContent || ''));
    return { table, thead, tbody, headers, data };
  }

  /**
   * Utility: Create an element with attributes and optional innerHTML.
   */
  function el(tag, attrs, html) {
    const e = document.createElement(tag);
    if (attrs) {
      Object.entries(attrs).forEach(([k, v]) => {
        if (v === null || v === undefined) return;
        e.setAttribute(k, String(v));
      });
    }
    if (html !== undefined) e.innerHTML = html;
    return e;
  }

  /**
   * CSV escape.
   */
  function csvEscape(val) {
    const s = String(val ?? '');
    if (s.includes('"') || s.includes(',') || s.includes('\n') || s.includes('\r')) {
      return '"' + s.replaceAll('"', '""') + '"';
    }
    return s;
  }

  /**
   * Parse Score OP and Score value from the existing filter controls.
   */
  function getScoreFilter() {
    const opEl = document.querySelector('select[name="td_score_op"]');
    const valEl = document.querySelector('input[name="td_score_val"]');
    const op = (opEl && opEl.value) ? opEl.value : '>=';
    let val = NaN;
    if (valEl && valEl.value !== '') {
      val = Number(valEl.value);
    }
    return { op, val, hasVal: !Number.isNaN(val) };
  }

  /**
   * Decide if a numeric score passes the filter per the problem's inclusive rule
   * for the '>' example ("> 0.6" keeps >= 0.6) and similarly for other ops.
   */
  function scorePasses(op, threshold, score) {
    if (Number.isNaN(score)) return false;
    if (op === '>=') return score >= threshold;
    if (op === '>') return score >= threshold; // as specified: keep  threshold for >
    if (op === '<=') return score <= threshold;
    if (op === '<') return score < threshold;
    if (op === '=') return score === threshold;
    // default to >= if unknown
    return score >= threshold;
  }

  /**
   * Attempt to parse the top_similar_docs cell into a structure we can filter.
   * Returns an object of form {kind: 'dict'|'list', value: Dict|Array}.
   */
  function parseTopSimilar(cellText) {
    const txt = (cellText || '').trim();
    if (!txt) return { kind: 'empty', value: null };
    try {
      const parsed = JSON.parse(txt);
      if (Array.isArray(parsed)) {
        return { kind: 'list', value: parsed };
      }
      if (parsed && typeof parsed === 'object') {
        return { kind: 'dict', value: parsed };
      }
    } catch (_e) {
      // try to recover from non-strict JSON by replacing single quotes
      try {
        const alt = JSON.parse(txt.replaceAll("'", '"'));
        if (Array.isArray(alt)) return { kind: 'list', value: alt };
        if (alt && typeof alt === 'object') return { kind: 'dict', value: alt };
      } catch (_e2) {
        return { kind: 'raw', value: txt };
      }
    }
    return { kind: 'raw', value: txt };
  }

  /**
   * Filter a parsed top_similar_docs structure per score filter.
   * Preserves the original shape (dict -> dict, list -> list of singleton dicts).
   */
  function filterTopSimilarStructure(parsed, op, threshold) {
    if (!parsed || !parsed.kind) return '';
    if (parsed.kind === 'dict') {
      const obj = parsed.value || {};
      const out = {};
      for (const [k, v] of Object.entries(obj)) {
        const s = Number(v);
        if (!Number.isNaN(s) && scorePasses(op, threshold, s)) {
          out[k] = v;
        }
      }
      return JSON.stringify(out);
    }
    if (parsed.kind === 'list') {
      const arr = Array.isArray(parsed.value) ? parsed.value : [];
      const out = [];
      for (const item of arr) {
        if (item && typeof item === 'object') {
          const entries = Object.entries(item);
          if (entries.length === 1) {
            const [k, v] = entries[0];
            const s = Number(v);
            if (!Number.isNaN(s) && scorePasses(op, threshold, s)) {
              out.push({ [k]: v });
            }
          }
        }
      }
      return JSON.stringify(out);
    }
    // For raw/empty types, return original text unchanged
    return JSON.stringify(parsed.value);
  }

  /**
   * State container for client pagination and original cell values so we can
   * restore when Simplify result is toggled off.
   */
  const State = {
    headers: [],
    data: [],
    originalTopSimilar: new Map(), // key: rowIndex, value: original cell text
    topSimilarColIdx: -1,
    clientPageIndex: 0,
    rowsPerPage: 20,
  };

  /**
   * Initialize state from the DOM table.
   */
  function initState() {
    const tableCtx = findResultsTable();
    if (!tableCtx) return null;
    State.headers = tableCtx.headers.slice();
    State.data = tableCtx.data.slice();
    State.clientPageIndex = 0;
    const idx = State.headers.findIndex((h) => h.toLowerCase() === 'top_similar_docs');
    State.topSimilarColIdx = idx;

    // Capture original top_similar_docs text content by current DOM rows
    const trNodes = Array.from(tableCtx.tbody.querySelectorAll('tr'));
    trNodes.forEach((tr, rIdx) => {
      const td = tr.querySelectorAll('td')[idx];
      const text = td ? td.textContent || '' : '';
      State.originalTopSimilar.set(rIdx, text);
    });
    return tableCtx;
  }

  /**
   * Render the current client page rows into the DOM tbody.
   * Applies hidden columns after render.
   */
  function renderTableBody(tableCtx) {
    const { tbody } = tableCtx;
    tbody.innerHTML = '';
    const start = State.clientPageIndex * State.rowsPerPage;
    const end = Math.min(State.data.length, start + State.rowsPerPage);
    for (let i = start; i < end; i++) {
      const tr = document.createElement('tr');
      const row = State.data[i];
      State.headers.forEach((_, cIdx) => {
        const td = document.createElement('td');
        td.textContent = (row[cIdx] != null) ? String(row[cIdx]) : '';
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    }
    // Apply hidden columns after re-render
    applyHiddenColumns(tableCtx);
    // If Simplify is enabled, apply to the freshly rendered cells
    try {
      applySimplifyToRendered(tableCtx);
    } catch (_e) { /* no-op */ }
  }

  /**
   * Get and validate rowsPerPage from input (5..500), default 20.
   */
  function getRowsPerPageOrDefault() {
    const inp = document.querySelector('[data-testid="rows-per-page-input"]');
    let v = 20;
    if (inp && inp.value) {
      const num = Number(inp.value);
      if (!Number.isNaN(num)) v = num;
    }
    v = Math.max(5, Math.min(500, v));
    return v;
  }

  /**
   * Update client-side page indicator controls.
   */
  function updateClientPager(tableCtx) {
    let pager = document.querySelector('[data-testid="client-pager"]');
    if (!pager) {
      pager = el('div', { 'data-testid': 'client-pager', class: 'd-flex align-items-center gap-2 mt-2' });
      tableCtx.table.parentElement.insertBefore(pager, tableCtx.table.nextSibling);
    }
    const totalPages = Math.max(1, Math.ceil(State.data.length / State.rowsPerPage));
    State.clientPageIndex = Math.max(0, Math.min(State.clientPageIndex, totalPages - 1));
    pager.innerHTML = '';
    const I18N = (window.I18N && window.I18N.dict) || {};
    const prevBtn = el('button', { type: 'button', class: 'btn btn-sm btn-outline-secondary', 'data-testid': 'client-prev' }, I18N['controls.prev'] || 'Prev');
    const nextBtn = el('button', { type: 'button', class: 'btn btn-sm btn-outline-secondary', 'data-testid': 'client-next' }, I18N['controls.next'] || 'Next');
    const label = el('span', { class: 'small text-muted' }, `${I18N['controls.page'] || (I18N['common.page'] || 'Page')} ${State.clientPageIndex + 1} ${I18N['controls.of'] || (I18N['common.of'] || 'of')} ${totalPages}`);
    prevBtn.addEventListener('click', () => {
      if (State.clientPageIndex > 0) {
        State.clientPageIndex -= 1;
        renderTableBody(tableCtx);
        updateClientPager(tableCtx);
      }
    });
    nextBtn.addEventListener('click', () => {
      if (State.clientPageIndex < totalPages - 1) {
        State.clientPageIndex += 1;
        renderTableBody(tableCtx);
        updateClientPager(tableCtx);
      }
    });
    pager.appendChild(prevBtn);
    pager.appendChild(nextBtn);
    pager.appendChild(label);
  }

  /**
   * Inject Hide Columns multi-select and Rows Per Page input if not present.
   */
  function ensureAuxControls(tableCtx) {
    // Create a dedicated controls section directly above the table to avoid polluting navbar or other forms.
    const I18N = (window.I18N && window.I18N.dict) || {};

    // Determine a stable wrapper (prefer parent of .table-responsive if present)
    const tableResponsive = tableCtx.table.closest('.table-responsive');
    const wrapper = tableResponsive ? tableResponsive.parentElement : tableCtx.table.parentElement;

    // Build the "Hide Columns" section with multi-select first, then export buttons
    let section = document.querySelector('[data-testid="hide-columns-section"]');
    if (!section) {
      section = el('div', { class: 'card mb-3', 'data-testid': 'hide-columns-section' });
      const body = el('div', { class: 'card-body' });
      const title = el('h6', { class: 'card-title mb-2' }, I18N['controls.hide_columns'] || 'Hide columns');

      // Multi-select for columns
      const hideSel = el('select', {
        multiple: 'multiple',
        class: 'form-select',
        'data-testid': 'hide-columns-select',
        style: 'min-height: 120px;'
      });
      State.headers.forEach((h, idx) => {
        const opt = el('option', { value: String(idx) }, h);
        hideSel.appendChild(opt);
      });

      // Export buttons below the multi-select (CSV first, then Excel)
      const btnRow = el('div', { class: 'mt-2 d-flex flex-wrap gap-2' });
      const btnCsv = el('button', { type: 'button', class: 'btn btn-sm btn-outline-primary', 'data-testid': 'export-csv-btn', title: I18N['controls.export_csv'] || 'Export CSV' }, `<i class="bi bi-filetype-csv"></i> ${I18N['controls.export_csv'] || 'Export CSV'}`);
      const btnXls = el('button', { type: 'button', class: 'btn btn-sm btn-outline-success', 'data-testid': 'export-xls-btn', title: I18N['controls.export_excel'] || 'Export Excel' }, `<i class="bi bi-file-earmark-excel"></i> ${I18N['controls.export_excel'] || 'Export Excel'}`);
      btnRow.appendChild(btnCsv);
      btnRow.appendChild(btnXls);

      body.appendChild(title);
      body.appendChild(hideSel);
      body.appendChild(btnRow);
      section.appendChild(body);

      // Insert the section above the table-responsive wrapper/table
      const refNode = tableResponsive || tableCtx.table;
      if (wrapper && wrapper.insertBefore && refNode && refNode.parentElement === wrapper) {
        wrapper.insertBefore(section, refNode);
      } else if (tableCtx.table.parentElement && tableCtx.table.parentElement.insertBefore) {
        tableCtx.table.parentElement.insertBefore(section, tableCtx.table);
      } else {
        document.body.insertBefore(section, document.body.firstChild);
      }
    }

    // Rows per page control: place just above the table (below the Hide Columns section)
    let rpp = document.querySelector('[data-testid="rows-per-page-input"]');
    if (!rpp) {
      const rppWrap = el('div', { class: 'd-flex align-items-end gap-2 mb-2' });
      const label2 = el('label', { class: 'form-label mb-0' }, I18N['controls.rows_per_page'] || 'Rows per page');
      rpp = el('input', { type: 'number', min: '5', max: '500', value: '20', class: 'form-control form-control-sm', 'data-testid': 'rows-per-page-input', style: 'width: 120px;' });
      rppWrap.appendChild(label2);
      rppWrap.appendChild(rpp);
      const refNode2 = tableResponsive || tableCtx.table;
      const parent2 = section.parentElement || wrapper || tableCtx.table.parentElement || document.body;
      parent2.insertBefore(rppWrap, (refNode2 && refNode2.parentElement === parent2) ? refNode2 : parent2.firstChild);
    }

    // Sync hidden columns on initial render
    applyHiddenColumns(tableCtx);
  }

  /**
   * Apply hidden columns from localStorage and current multi-select.
   */
  function applyHiddenColumns(tableCtx) {
    const { table, thead, tbody } = tableCtx;
    const ths = Array.from(thead.querySelectorAll('th'));
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const sel = document.querySelector('[data-testid="hide-columns-select"]');

    // Determine hidden set
    const lsKeyHidden = lsKey('hiddenColumns');
    const saved = localStorage.getItem(lsKeyHidden);
    let hiddenIdx = new Set();
    if (saved) {
      try { hiddenIdx = new Set(JSON.parse(saved)); } catch (_e) { hiddenIdx = new Set(); }
    }
    if (sel) {
      // Sync selection UI with saved state (tick options with saved indices)
      Array.from(sel.options).forEach((opt) => {
        const idx = Number(opt.value);
        opt.selected = hiddenIdx.has(idx);
      });
    }

    // Apply hidden styles
    ths.forEach((th, idx) => {
      if (hiddenIdx.has(idx)) th.style.display = 'none'; else th.style.display = '';
    });
    rows.forEach((tr) => {
      const tds = Array.from(tr.querySelectorAll('td'));
      tds.forEach((td, idx) => {
        if (hiddenIdx.has(idx)) td.style.display = 'none'; else td.style.display = '';
      });
    });
  }

  /**
   * Handle changes to Hide Columns selection, persisting to localStorage and
   * updating the table view.
   */
  function bindHideColumns(tableCtx) {
    const sel = document.querySelector('[data-testid="hide-columns-select"]');
    if (!sel) return;
    const key = lsKey('hiddenColumns');

    function saveAndApply() {
      const selected = Array.from(sel.options).filter((opt) => opt.selected).map((opt) => Number(opt.value));
      localStorage.setItem(key, JSON.stringify(selected));
      applyHiddenColumns(tableCtx);
    }

    // Default change handler (covers keyboard selection and programmatic changes)
    sel.addEventListener('change', saveAndApply);

    // Enhance UX: toggle select options on simple click (no Ctrl/Cmd required),
    // allowing deselecting the last selected option to show all columns.
    sel.addEventListener('mousedown', function (e) {
      const t = e.target;
      if (t && t.tagName && t.tagName.toUpperCase() === 'OPTION') {
        e.preventDefault(); // prevent native selection behavior
        // Toggle the clicked option
        t.selected = !t.selected;
        // Persist and apply immediately
        saveAndApply();
      }
    });
  }

  /**
   * Inject the Simplify option into Key Mode dropdown and bind change handlers
   * for Simplify and Score filters.
   */
  // Apply simplify to the currently rendered table only; does not re-render.
  function applySimplifyToRendered(tableCtx) {
    const box = document.querySelector('#simplifyToggle');
    if (!box || !box.checked) return false;
    if (State.topSimilarColIdx < 0) {
      // No target column; uncheck to reflect non-applicability
      box.checked = false; localStorage.setItem(lsKey('simplify_enabled'), 'false');
      return false;
    }
    const score = getScoreFilter();
    if (!score.hasVal) {
      // Without numeric threshold, skip and uncheck for clarity
      box.checked = false; localStorage.setItem(lsKey('simplify_enabled'), 'false');
      return false;
    }
    const trs = Array.from(tableCtx.tbody.querySelectorAll('tr'));
    let applied = false;
    trs.forEach((tr) => {
      const td = tr.querySelectorAll('td')[State.topSimilarColIdx];
      if (!td) return;
      const original = td.textContent || '';
      const parsed = parseTopSimilar(original);
      const filteredStr = filterTopSimilarStructure(parsed, score.op, score.val);
      if (filteredStr !== original) { applied = true; }
      td.textContent = filteredStr;
    });
    return applied;
  }

  function bindSimplify(tableCtx) {
    const simplifyBox = document.querySelector('#simplifyToggle');
    const opEl = document.querySelector('select[name="td_score_op"]');
    const valEl = document.querySelector('input[name="td_score_val"]');
    if (!simplifyBox) return;

    const LSKEY = lsKey('simplify_enabled');

    // Load saved state and reflect in UI (application happens after first render)
    const saved = localStorage.getItem(LSKEY);
    simplifyBox.checked = (saved === 'true');

    // Persist on toggle and re-render to ensure a clean base, then apply
    simplifyBox.addEventListener('change', () => {
      localStorage.setItem(LSKEY, simplifyBox.checked ? 'true' : 'false');
      renderTableBody(tableCtx); // re-render from State.data snapshot
    });

    // When score filters change, re-render then re-apply simplify if enabled
    function onScoreChange() { renderTableBody(tableCtx); }
    if (opEl) opEl.addEventListener('change', onScoreChange);
    if (valEl) valEl.addEventListener('input', onScoreChange);
  }

  /**
   * Bind Rows per Page control and client pagination buttons.
   */
  function bindRowsPerPage(tableCtx) {
    const inp = document.querySelector('[data-testid="rows-per-page-input"]');
    if (!inp) return;

    // Detect Admin page explicit enablement to sync with server pagination instead of client-only
    let isAdminPath = false;
    try {
      const path = (window.location && window.location.pathname) ? window.location.pathname : '';
      isAdminPath = (path && path.indexOf('/admin') !== -1);
    } catch(_e) { isAdminPath = false; }

    if (isAdminPath) {
      // Reflect current server-side per_page into the control
      const form = document.querySelector('.card-header form');
      const perPageHidden = document.querySelector('input[name="per_page"]');
      const pageHidden = document.querySelector('input[name="page"]');
      if (perPageHidden && perPageHidden.value) {
        const num = Number(perPageHidden.value);
        if (!Number.isNaN(num)) {
          // Clamp to server constraints (1..100 for admin)
          const clamped = Math.max(1, Math.min(100, num));
          inp.value = String(clamped);
        }
      }
      inp.addEventListener('change', function(){
        let v = Number(inp.value);
        if (Number.isNaN(v)) v = 10;
        // Server constraints: 1..100 for admin
        v = Math.max(1, Math.min(100, v));
        if (perPageHidden) perPageHidden.value = String(v);
        if (pageHidden) pageHidden.value = '1'; // reset to first page for consistency
        if (form && form.submit) form.submit();
      });
      return; // skip client-side re-pagination on admin
    }

    // Default client-side behavior for non-admin pages
    const key = lsKey('rowsPerPage');

    // Load saved
    const saved = localStorage.getItem(key);
    if (saved) {
      const num = Number(saved);
      if (!Number.isNaN(num)) inp.value = String(Math.max(5, Math.min(500, num)));
    }

    function rePaginate() {
      State.rowsPerPage = getRowsPerPageOrDefault();
      localStorage.setItem(key, String(State.rowsPerPage));
      State.clientPageIndex = 0;
      renderTableBody(tableCtx);
      updateClientPager(tableCtx);
    }
    inp.addEventListener('change', rePaginate);

    // Initialize pager once
    State.rowsPerPage = getRowsPerPageOrDefault();
    renderTableBody(tableCtx);
    updateClientPager(tableCtx);
  }

  /**
   * Compute visible column indices based on current hidden selection.
   */
  function getVisibleColumnIndices(tableCtx) {
    const ths = Array.from(tableCtx.thead.querySelectorAll('th'));
    const visible = [];
    ths.forEach((th, idx) => { if (th.style.display !== 'none') visible.push(idx); });
    return visible;
  }

  /**
   * Collect current on-screen rows for export (exact view), respecting client
   * pagination, hidden columns, and simplified cells.
   */
  function collectVisibleData(tableCtx) {
    const headers = tableCtx.headers;
    const visibleCols = getVisibleColumnIndices(tableCtx);
    const start = State.clientPageIndex * State.rowsPerPage;
    const end = Math.min(State.data.length, start + State.rowsPerPage);

    // Build rows from DOM to capture any simplified content exactly as shown
    const domRows = Array.from(tableCtx.tbody.querySelectorAll('tr'));
    const rows = domRows.map((tr) => Array.from(tr.querySelectorAll('td')).map((td) => td.textContent || ''));

    const outHeaders = visibleCols.map((i) => headers[i]);
    const outRows = rows.map((r) => visibleCols.map((i) => r[i]));
    return { outHeaders, outRows };
  }

  /**
   * Trigger download of a Blob as a named file.
   */
  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 0);
  }

  /**
   * Bind export buttons.
   */
  function bindExport(tableCtx) {
    const btnCsv = document.querySelector('[data-testid="export-csv-btn"]');
    const btnXls = document.querySelector('[data-testid="export-xls-btn"]');
    if (!btnCsv || !btnXls) return;

    // Determine context-specific base filename from the table attribute, fallback to 'results_view'.
    const baseName = (tableCtx.table && tableCtx.table.dataset && tableCtx.table.dataset.exportFilename)
      ? String(tableCtx.table.dataset.exportFilename)
      : 'results_view';

    btnCsv.addEventListener('click', () => {
      const { outHeaders, outRows } = collectVisibleData(tableCtx);
      const lines = [];
      lines.push(outHeaders.map(csvEscape).join(','));
      outRows.forEach((r) => { lines.push(r.map(csvEscape).join(',')); });
      const csv = lines.join('\r\n');
      // Prepend UTF-8 BOM so Excel correctly detects encoding for non-ASCII headers
      const blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8;' });
      downloadBlob(blob, `${baseName}.csv`);
    });

    // Helper: ensure XLSX library is available (dynamically loads from CDN if needed)
    function ensureXLSXLib() {
      return new Promise((resolve, reject) => {
        if (window.XLSX) { resolve(window.XLSX); return; }
        // Try local vendor first, then CDN, otherwise let caller fall back to XML
        const localScript = document.createElement('script');
        localScript.src = '/hyundai_document_authenticator/external/result_gui/static/vendor/xlsx/xlsx.full.min.js';
        localScript.onload = () => resolve(window.XLSX);
        localScript.onerror = () => {
          const cdnScript = document.createElement('script');
          cdnScript.src = 'https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js';
          cdnScript.onload = () => resolve(window.XLSX);
          cdnScript.onerror = () => reject(new Error('Failed to load XLSX library'));
          document.head.appendChild(cdnScript);
        };
        document.head.appendChild(localScript);
      });
    }

    function exportAsXLSX(headers, rows) {
      const wb = window.XLSX.utils.book_new();
      const aoa = [headers, ...rows];
      const ws = window.XLSX.utils.aoa_to_sheet(aoa);
      window.XLSX.utils.book_append_sheet(wb, ws, 'Results');
      const wbout = window.XLSX.write(wb, { bookType: 'xlsx', type: 'array' });
      const blob = new Blob([wbout], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
      downloadBlob(blob, `${baseName}.xlsx`);
    }

    function exportAsXML(headers, rows) {
      function xmlEscape(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
      let xml = '';
      xml += '<?xml version="1.0"?>\n';
      xml += '<?mso-application progid="Excel.Sheet"?>\n';
      xml += '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:x="urn:schemas-microsoft-com:office:excel" xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet" xmlns:html="http://www.w3.org/TR/REC-html40">';
      xml += '<Worksheet ss:Name="Results"><Table>';
      xml += '<Row>' + headers.map((h) => `<Cell><Data ss:Type="String">${xmlEscape(h)}</Data></Cell>`).join('') + '</Row>';
      rows.forEach((r) => {
        xml += '<Row>' + r.map((c) => `<Cell><Data ss:Type="String">${xmlEscape(c)}</Data></Cell>`).join('') + '</Row>';
      });
      xml += '</Table></Worksheet></Workbook>';
      const blob = new Blob([xml], { type: 'application/vnd.ms-excel' });
      downloadBlob(blob, `${baseName}.xml`);
    }

    btnXls.addEventListener('click', () => {
      const { outHeaders, outRows } = collectVisibleData(tableCtx);
      ensureXLSXLib()
        .then(() => {
          try { exportAsXLSX(outHeaders, outRows); }
          catch (_e) { exportAsXML(outHeaders, outRows); }
        })
        .catch(() => {
          exportAsXML(outHeaders, outRows);
        });
    });
  }

  /**
   * Wire up everything on DOMContentLoaded.
   */
  function main() {
    const tableCtx = initState();
    if (!tableCtx) return; // no table on page

    // Inject controls (hide columns, rows per page, export)
    ensureAuxControls(tableCtx);

    // Apply saved hidden columns and bind change listener
    applyHiddenColumns(tableCtx);
    bindHideColumns(tableCtx);

    // Simplify mode logic
    bindSimplify(tableCtx);

    // Rows per page and client pager
    bindRowsPerPage(tableCtx);

    // Export buttons
    bindExport(tableCtx);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', main);
  } else {
    main();
  }
})();
