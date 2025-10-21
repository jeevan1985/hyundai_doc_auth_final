"""Flask entry point for the self-contained Result GUI module.

This module boots a minimal, production-ready Flask application that is fully
self-contained within the result_gui directory. It wires configuration loading,
PostgreSQL initialization for authentication, blueprint registration, and
application startup.

The GUI supports an optional CSV-based results viewer or a PostgreSQL-backed
results viewer, selectable via the config.yaml file located in the same
directory.

This module is intentionally isolated and should have zero impact on the main
project if removed.

Usage:
    $ python app.py

The application will read configuration from config.yaml located alongside this
file. Ensure dependencies in requirements.txt are installed in your
environment.

Note:
- Other local modules such as config_loader.py, db.py, auth.py, and routes.py
  must exist in this directory. They are created as part of the Result GUI
  module and are intentionally not imported from outside the module.

"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
from datetime import timedelta
import os

from flask import Flask, g, request, session, render_template, jsonify, Response
from flask_login import LoginManager
from assets_manager import ensure_assets

# Local imports (modules reside within this same directory)
try:
    from config_loader import AppConfig, load_config
    from db import Database
    from auth import init_auth
    from routes import init_routes
except Exception as exc:  # pragma: no cover - helpful startup message during scaffolding
    # Provide a meaningful error if supporting files are not yet created.
    raise RuntimeError(
        "Required local modules are missing. Ensure config_loader.py, db.py, "
        "auth.py, and routes.py exist in the same directory as app.py."
    ) from exc


def create_app(config_path: Optional[Path] = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        config_path: Optional explicit path to a YAML configuration file. If not
            provided, defaults to a file named "config.yaml" in the same
            directory as this module.

    Returns:
        A configured Flask application instance.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        ValueError: If the configuration is invalid.
        RuntimeError: If database initialization or blueprint registration
            fails.
    """
    base_dir = Path(__file__).resolve().parent
    cfg_path = config_path or (base_dir / "config.yaml")

    # Load application configuration from YAML
    config: AppConfig = load_config(cfg_path)

    # Initialize Flask app using explicit template and static folders
    app = Flask(
        __name__,
        template_folder=str(base_dir / "templates"),
        static_folder=str(base_dir / "static"),
    )

    # Configure application logging early to capture initialization issues.
    try:
        import logging
        from logging.handlers import RotatingFileHandler
        # Honor centralized logging root via APP_LOG_DIR; fallback to container-friendly default
        env_log_dir = os.getenv("APP_LOG_DIR", "/home/appuser/app/logs")
        log_dir = Path(env_log_dir) / "result_gui"
        log_dir.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        # File handler with rotation
        fh = RotatingFileHandler(str(log_dir / "app.log"), maxBytes=1_048_576, backupCount=5, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        # Stream handler for stderr
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        # Attach handlers only once
        if not app.logger.handlers:
            app.logger.addHandler(fh)
            app.logger.addHandler(sh)
        app.logger.setLevel(logging.INFO)
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
    except Exception:
        # Never break app startup due to logging config issues.
        pass

    # Make the typed config available to blueprints/extensions via app.config.
    app.config["APP_CONFIG"] = config

    # Ensure local vendor assets exist; attempt CDN download if missing.
    try:
        ensure_assets(base_dir / "static")
    except Exception:
        # Do not crash the app if asset checks fail. UI will apply graceful fallbacks.
        pass

    # Apply security-relevant settings
    # Use configured secret key; if missing, load_config should provide a
    # cryptographically secure default for the process lifetime.
    app.config["SECRET_KEY"] = config.secret_key
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    # Force exceptions to be handled by our error handlers even if debug is enabled.
    app.config["PROPAGATE_EXCEPTIONS"] = False
    app.config["TRAP_HTTP_EXCEPTIONS"] = False

    # Configure Flask-Login remember-me duration from YAML (in days)
    try:
        remember_days = int(getattr(config, 'remember_days', 14) or 14)
    except Exception:
        remember_days = 14
    app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=max(1, min(365, remember_days)))

    # Initialize database layer dedicated to this module (auth and results)
    db = Database(config=config)
    # Ensure authentication schema and users table exist (idempotent)
    db.ensure_auth_schema_and_table()

    # Initialize auth (Flask-Login) and register login routes/blueprint
    init_auth(app=app, db=db)

    # i18n helpers: store chosen language in session and provide g.lang
    from i18n import normalize_lang, translations_for  # local import by design

    @app.before_request
    def _set_lang() -> None:
      """Set the request language using session value or Accept-Language header."""
      lang = session.get('lang')
      if not lang:
          # Basic Accept-Language parse (take first token)
          hdr = request.headers.get('Accept-Language', '')
          token = (hdr.split(',')[0].split('-')[0].strip() if hdr else None)
          lang = normalize_lang(token)
          session['lang'] = lang
      g.lang = normalize_lang(lang)

      # Theme preference
      theme = (session.get('theme') or 'auto').strip().lower()
      # Support extended themes; map unknowns to 'auto'.
      allowed_themes = {'light', 'dark', 'auto', 'retro', 'cyberpunk', 'glass', 'default'}
      if theme not in allowed_themes:
          theme = 'auto'
          session['theme'] = theme
      g.theme = theme

    @app.context_processor
    def inject_translations() -> dict[str, object]:
        """Inject i18n utilities into Jinja templates.

        Returns:
            dict[str, object]: Context values including LANG, THEME, TRANSLATIONS, APP_CONFIG.
        """
        # Provide t() in JS through a flattened dict if needed
        lang = getattr(g, 'lang', 'en')
        theme = getattr(g, 'theme', 'auto')
        return {
            'LANG': lang,
            'THEME': theme,
            'TRANSLATIONS': translations_for(lang),
            'APP_CONFIG': config,
        }

    # Register application routes/blueprint for results viewing
    init_routes(app=app, db=db, config=config)

    # ----------------------------
    # Global error handlers
    # ----------------------------
    import psycopg2  # local import to avoid hard dependency during tooling/lint
    from werkzeug.exceptions import NotFound  # noqa: F401  (imported for type context / future use)

    @app.errorhandler(psycopg2.OperationalError)
    def _handle_db_operational_error(exc: Exception) -> tuple[str, int]:
        """Handle database connectivity issues without leaking details.

        Args:
            exc: The original exception instance.

        Returns:
            A Flask response rendering a generic 500 error page.
        """
        app.logger.exception("Database OperationalError encountered during request.")
        # Do not expose driver/DSN details to client.
        return render_template("500.html", title="Database Error", message="Database connection failed. Please try again later."), 500

    @app.errorhandler(404)
    def _handle_404(_exc: Exception) -> tuple[str, int]:
        """Render a friendly 404 page and log the event."""
        try:
            path = request.path
        except Exception:
            path = "<unknown>"
        app.logger.warning("404 Not Found: %s", path)
        return render_template("404.html"), 404

    @app.errorhandler(Exception)
    def _handle_unexpected_error(exc: Exception) -> tuple[str, int]:
        """Catch-all handler for unexpected exceptions.

        Logs the full stack trace server-side and returns a generic error page
        without exposing internals to the user.
        """
        app.logger.exception("Unhandled exception occurred.")
        return render_template("500.html", title="Server Error", message="An unexpected error occurred. Please try again later."), 500

    # ----------------------------
    # Health check endpoint
    # ----------------------------
    @app.get("/health")
    def _health() -> tuple[Response, int]:
        """Basic health probe for monitoring systems.

        Returns:
            JSON with overall status and database connectivity. Uses HTTP 200
            when healthy and 503 when unhealthy.
        """
        ok = db.ping()
        status = "ok" if ok else "unhealthy"
        code = 200 if ok else 503
        return jsonify({"status": status, "database": {"connected": ok}}), code

    return app


if __name__ == "__main__":
    application = create_app()
    # Run with host 0.0.0.0 to allow external access when deployed behind a
    # reverse proxy; port and debug are configured via YAML.
    # For production, prefer running via a WSGI server (e.g., gunicorn/uwsgi)
    # and a fronting proxy (e.g., nginx). This direct run is suitable for local
    # development and testing.
    cfg = load_config(Path(__file__).resolve().parent / "config.yaml")
    application.run(host="0.0.0.0", port=cfg.web_port, debug=cfg.debug_mode)
