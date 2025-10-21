# Result GUI Admin & i18n Guide

This guide documents admin endpoints, the System Health schema, i18n usage, and access control for the self-contained Result GUI module.

## Endpoints

- GET /admin
  - Admin dashboard (role: admin only). Shows metrics, recent activity, and System Health panel.
- GET /admin/system_health.json
  - JSON payload of System Health. Role: admin only.
- GET /admin/system_health.csv
  - CSV export (attachment) of the same metrics. Filename: system_health.csv. Role: admin only.
- GET /users
  - Manage Users page (list, search). Role: admin only.
- POST /users/create
  - Create user with username, password, role (viewer|admin). Role: admin only.
- POST /users/<int:user_id>/role
  - Update role for a user (viewer|admin). Role: admin only.
- POST /users/<int:user_id>/password
  - Set/reset password for a user. Role: admin only.
- POST /users/<int:user_id>/delete
  - Delete a user. Role: admin only.
- GET|POST /profile
  - Profile page for the current user to change their password. Requires verifying current password and confirming new password.
- POST /set_lang
  - Persist language selection in the session. Body: lang=en|ko. Redirects to referer or index.

## System Health Schema

JSON shape returned by /admin/system_health.json:

{
  "server": {
    "status": "ok" | "degraded" | "down" (string),
    "hostname": string,
    "platform": string,
    "uptime_seconds": int,
    "load_1m": float|null,
    "load_5m": float|null,
    "load_15m": float|null
  },
  "database": {
    "connected": bool,
    "error": string|null
  },
  "queue": {
    "status": string,     // e.g., "ok", "backlog", "unknown"
    "backlog": int
  },
  "storage": {
    "disk_total_bytes": int,
    "disk_used_bytes": int,
    "disk_free_bytes": int
  },
  "recent_events": [ {"level": string, "message": string, "ts": int } ],
  "generated_at": int     // epoch seconds when generated on server
}

- Update cadence: the Admin page auto-refreshes every 15 seconds using fetch().
- Error behavior: when fetch fails or the server returns error, a warning alert is shown with a translated message. CSV endpoint is always downloadable and reflects current values when requested.

CSV shape at /admin/system_health.csv:
- Header: key,value
- Rows:
  - server.status, server.hostname, server.platform, server.uptime_seconds,
  - server.load_1m, server.load_5m, server.load_15m,
  - database.connected, database.error,
  - queue.status, queue.backlog,
  - storage.disk_total_bytes, storage.disk_used_bytes, storage.disk_free_bytes,
  - generated_at

## Internationalization (i18n)

- Language codes: "en" and "ko" are supported by default.
- Language selection persists in the Flask session via POST /set_lang.
- Default language is selected from the Accept-Language header on first request; fallback to English.
- All UI strings in templates and client-side JS reference keys in i18n.TRANSLATIONS.
- In templates, translations are injected as TRANSLATIONS and selected language is LANG.
- For client-side JS, window.I18N = { lang, dict } is injected by base.html. Use window.I18N.dict["key"] with English fallback.
- Adding a new language:
  1) Extend SUPPORTED_LANGS in i18n.py.
  2) For every key in TRANSLATIONS, add the new language entry.
  3) Templates and JS will automatically use the provided translation with English fallback.

### Using translations in templates
- Example: {{ TRANSLATIONS['nav.admin'] }}
- For dynamic interpolations where needed, format on the server before rendering or use .format in Python code (see flash messages).

### Using translations in JavaScript
- Example:
  const I18N = (window.I18N && window.I18N.dict) || {};
  const label = I18N['controls.prev'] || 'Prev';

## Result Controls and Persistence

The Results page exposes client-side controls that operate without server round-trips. These controls only affect the current on-screen dataset (the rows currently rendered):

- Hide Columns
  - A multi-select placed in a dedicated "Hide Columns" section above the table.
  - Persists per user via localStorage key: result_gui.hiddenColumns.<username>
  - Hides selected columns using CSS display: none (headers and cells).
- Rows per page
  - Numeric input (5–500). Re-paginates the current on-screen dataset client-side.
  - Persists per user via localStorage key: result_gui.rowsPerPage.<username>
- Key Mode (Simplify result)
  - Augments the Key Mode dropdown with a "Simplify result" option.
  - When active and a numeric Score filter is set, prunes top_similar_docs values to those passing the (inclusive) threshold logic.
  - Persists per user via localStorage key: result_gui.td_key_mode.<username>. It is restored on load and only changes when the user updates it manually; it does not reset on searches.
- Export CSV / Excel
  - Export exactly what is visible on screen: honors hidden columns, client pagination, and simplified cell content.
  - The export base filename is taken from the table attribute data-export-filename when present; otherwise defaults to results_view.
  - CSV is UTF-8 with BOM; Excel export uses XLSX when the library loads (falls back to Excel XML otherwise).

Accessibility & Non-interference
- Controls are not injected into pages where table rows contain interactive form elements (e.g., Manage Users) unless explicitly enabled.
- All user-facing strings used by these controls are translatable via i18n TRANSLATIONS or window.I18N.

## Manage Users & Profile: Access and Behavior

- Manage Users is accessible only to users with role == 'admin'. Non-admins receive 403.
- Create, update role, set password, and delete actions require admin role.
- Profile page is available to authenticated users (admin and viewer). Users can change their own password by providing the current password and confirming the new password. Passwords are hashed using bcrypt.

## Clear Recent Activity (Admin Only)

- Location: Admin > Recent Activity (toolbar, right side).
- UI: Inline form with a range dropdown and a Clear button.
  - Ranges:
    - All
    - Today
    - Past N days (1..31)
  - Confirmation: A confirmation prompt appears before deletion.
- Endpoint: POST /admin/activity/clear
  - Body: range_choice = one of { all | today | days_N }
  - RBAC: Requires admin role (403 for non-admins).
  - Behavior: Deletes rows from auth schema activity_log per selection.
  - Safety: Parameterized SQL, server-side validation. Invalid input flashes a localized error and deletes nothing.
  - Result: Localized flash indicates how many records were deleted.

## Recent Activity Exports

- In Admin > Recent Activity table, export file base name is recent_activity for client-side CSV/XLSX exports from the table toolbar (no server round-trip). The actual attachment filename will be recent_activity.csv or recent_activity.xlsx depending on the export action.

## Manual Test Checklist

Language/i18n
- Switch language from navbar selector; verify selection persists across page navigations and browser refresh.
- Verify translations appear correctly on:
  - Index (filters, pagination, table empty state)
  - Admin (cards, table headers, System Health panel, alerts, buttons)
  - Manage Users (headings, search, form labels, buttons, delete confirm)
  - Login (labels, submit text)
  - Profile (labels, submit text)

Admin System Health
- Navigate to /admin as admin, ensure auto-refresh updates values every ~15 seconds.
- Verify CSV download link saves as system_health.csv.
- Temporarily stop DB or break credentials; JSON fetch shows translated error alert; CSV still downloads with database.connected=false and error message populated.

Manage Users
- As admin, create a user, update the role, set a password, delete a user. Flash messages are localized.
- As viewer, attempt to access /admin and /users; receive 403.

Profile
- Change password with correct current password; success flash is localized.
- Attempt with wrong current password; localized error.
- Attempt with mismatched confirmation; localized error.

Recent Activity Exports
- Admin > Recent Activity table export buttons produce recent_activity.csv/.xlsx with correct on-screen view.

Result Controls
- On Results page, verify Hide columns, Rows per page, Simplify result (in Key Mode), Key Mode selection persists across reloads/searches, and CSV/XLSX export still work.
- On Manage Users page, verify controls are disabled (data-enable-controls=false) and do not interfere with forms.

## Lightweight Unit Tests

Create tests in a suitable tests/ directory (example names shown):
- test_i18n.py
  - test_normalize_lang_variants(): 'en', 'EN', 'ko-KR', None -> expected normalized values.
  - test_translations_for_fallback(): missing ko key falls back to English; missing key returns the key.
- test_routes.py
  - test_system_health_shape(client): use Flask test client, monkeypatch Database._get_conn to simulate DB up/down; assert JSON has expected keys and types, and CSV contains required rows.

Note: Unit tests should import from result_gui.i18n and result_gui.routes via the module path appropriate for your test runner context.
