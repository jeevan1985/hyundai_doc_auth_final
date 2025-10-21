# Result GUI (Self-Contained Flask Module)

This directory contains an optional, self-contained Flask web GUI for securely viewing result data. It includes user authentication (PostgreSQL + Flask-Login + bcrypt) and supports two data sources: a debug CSV file or a PostgreSQL table used by the main system.

Removing this `result_gui/` folder has zero impact on the rest of the project.

## Features
- Authentication with users stored in PostgreSQL (`users` table)
- Flask-Login session management
- Bcrypt password hashing
- View result data from a CSV or a PostgreSQL table (configurable via YAML)
- Internationalization (i18n) with English/Korean and client-side injection
- Search and pagination
- Minimal responsive UI (Bootstrap)
- Optional role-based access control example (admin route)
- Local-first static assets with automatic CDN fallback
- Theming system with 5 themes (Default, Light, Dark, Retro Futurism, Cyberpunk, Glassmorphism)

## Structure
- `app.py` – Flask entry point
- `assets_manager.py` – Local-first asset bootstrapper; auto-downloads missing vendor files when online
- `config_loader.py` – Typed YAML config loader with validation
- `db.py` – PostgreSQL access layer and helpers
- `auth.py` – Authentication (Flask-Login, bcrypt)
- `routes.py` – Routes for results and an example admin page
- `templates/`, `static/` – UI assets
- `static/css/themes.css` – Theme variables and component styling
- `static/js/theme.js` – Client theme management and offline fallback
- `requirements.txt` – Module-specific dependencies only
- `config.yaml` – Configuration (edit as needed)
- `debug_results.csv` – Sample CSV for CSV mode
- `manage_users.py` – CLI helper to create/update users

## Setup

1. Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows
pip install -r requirements.txt
```

2. Ensure PostgreSQL is running and reachable per your `config.yaml` values. The module will create the `users` table automatically if missing (in `auth_db_schema`, default `public`).

3. Configure `config.yaml`:
- `use_csv`: true to read from CSV, false to read from DB table
- `csv_path`: path to `debug_results.csv` (if using CSV)
- `db_*`: connection settings for PostgreSQL
- `results_table`: schema-qualified table name (for DB results mode)
- `auth_db_schema`: schema for the `users` table
- `web_port`, `debug_mode`: server settings
- `secret_key`: optional fixed secret; otherwise a secure random key is generated on each run

## Local-First Assets & Fallback

- Vendor assets are expected under `static/vendor/`:
  - Bootstrap CSS: `static/vendor/bootstrap/css/bootstrap.min.css`
  - Bootstrap JS: `static/vendor/bootstrap/js/bootstrap.bundle.min.js`
  - Bootstrap Icons CSS: `static/vendor/bootstrap-icons/font/bootstrap-icons.min.css`
  - Bootstrap Icons fonts: `static/vendor/bootstrap-icons/font/fonts/*`
- On app startup, `ensure_assets()` checks for these files and downloads any missing ones from a CDN when internet is available. When offline, the app continues to run and the UI gracefully falls back.
- Templates load local assets first. If a local asset is missing, a small inline script injects CDN links. If also offline, the UI forces a safe default theme and shows a non-blocking toast about limited visuals.

## Theming

- Theme switcher is exposed in the navbar. The selected theme is saved in session and also persisted in the browser via `localStorage` by `static/js/theme.js` (key: `ui.theme`).
- Themes provided: Default, Light, Dark, Retro Futurism (`retro`), Cyberpunk (`cyberpunk`), and Glassmorphism (`glass`).
- `static/css/themes.css` defines CSS variables per theme and applies them to key components: navbar, forms, tables, cards, pagination, modals, alerts, buttons, inputs, dropdowns, and tooltips.

## Testing Scenarios

- Theme switching: Verify that selecting each theme updates the entire UI consistently and persists across page loads.
- Dropdown alignment: Theme and language dropdowns are constrained in width and include spacing so they do not collide with the username.
- Air-gapped mode: Disconnect internet and remove a vendor file temporarily; the app should still render using the default theme without console errors.
- Connected mode: With vendor files removed, refresh; CDN fallback should load assets automatically.

## Run

```bash
python app.py
```

Navigate to `http://localhost:<web_port>` (default 8080).

For production, run via a WSGI server (e.g., gunicorn/uwsgi) behind a reverse proxy and set a fixed `secret_key`.
