"""Authentication utilities using Flask-Login and bcrypt.

This module defines the Flask-Login integration, the user model wrapper, and
blueprint routes for login/logout. Password hashing is handled via bcrypt.
User data is persisted in PostgreSQL through the local ``Database`` layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from flask import Blueprint, Flask, flash, redirect, render_template, request, url_for, current_app, session
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
import bcrypt
import psycopg2
import pyotp
import urllib.parse

from db import Database, UserRecord


@dataclass
class User(UserMixin):
    """Flask-Login user wrapper around a persistent user record.

    Attributes:
        id: Unique user ID as string (Flask-Login requires str ID).
        username: Username string.
        role: Role string (e.g., "admin", "viewer").
    """

    id: str
    username: str
    role: str


login_manager = LoginManager()
login_manager.login_view = "auth.login"


def init_auth(app: Flask, db: Database) -> None:
    """Initialize Flask-Login and register authentication routes.

    Args:
        app: The Flask application instance.
        db: Database layer for fetching users.
    """
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id: str) -> Optional[User]:
        """Load a user by ID for Flask-Login session rehydration.

        Args:
            user_id: ID string stored in the session.

        Returns:
            A ``User`` instance or ``None`` if not found.
        """
        try:
            rec = db.get_user_by_id(int(user_id))
        except psycopg2.OperationalError:
            # Avoid leaking DB details during session rehydration. Log and treat as no user.
            current_app.logger.exception("Database OperationalError in user_loader")
            return None
        except Exception:
            current_app.logger.exception("Unexpected error in user_loader")
            return None
        if not rec:
            return None
        return User(id=str(rec.id), username=rec.username, role=rec.role)

    # Blueprint for auth routes
    bp = Blueprint("auth", __name__, template_folder="templates")

    @bp.route("/login", methods=["GET", "POST"])
    def login():  # type: ignore[override]
        """Render login form and perform authentication on POST.

        Behavior:
            - When 2FA is disabled: verify password and complete login.
            - When 2FA is enabled: after password verification, redirect to
              the OTP entry/setup flow (/login/2fa). The user is not logged in
              until OTP is validated.
        """
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "").encode("utf-8")
            remember = (request.form.get("remember") == "on")
            try:
                rec: Optional[UserRecord] = db.get_user_by_username(username)
                if rec and bcrypt.checkpw(password, rec.password_hash.encode("utf-8")):
                    # Check if 2FA is enabled via config.
                    cfg = current_app.config.get("APP_CONFIG")
                    enable_2fa: bool = bool(getattr(cfg, "enable_2f_authentication", False))
                    if enable_2fa:
                        # Stash pending login context and send to OTP flow.
                        session["2fa_pending_user_id"] = int(rec.id)
                        session["2fa_pending_username"] = rec.username
                        session["2fa_pending_role"] = rec.role
                        session["2fa_remember"] = bool(remember)
                        return redirect(url_for("auth.login_2fa"))
                    # 2FA disabled: complete login immediately.
                    user = User(id=str(rec.id), username=rec.username, role=rec.role)
                    login_user(user, remember=remember)
                    # Log activity (best-effort)
                    try:
                        db.log_activity(
                            username=rec.username,
                            action="login",
                            ip=request.remote_addr,
                            user_agent=request.headers.get("User-Agent"),
                        )
                    except Exception:
                        pass
                    return redirect(url_for("main.index"))
                from i18n import t
                from flask import g
                flash(t("alerts.invalid_credentials", getattr(g, 'lang', 'en')), "danger")
            except psycopg2.OperationalError:
                # Friendly message without leaking DB details.
                current_app.logger.exception("Database OperationalError during login")
                return render_template("500.html", title="Database Error", message="Database connection failed. Please try again later."), 500
            except Exception:
                current_app.logger.exception("Unexpected error during login")
                return render_template("500.html", title="Server Error", message="An unexpected error occurred. Please try again later."), 500
        return render_template("login.html")

    @bp.route("/logout")
    @login_required
    def logout():  # type: ignore[override]
        """Log out the current user and redirect to login page."""
        try:
            if current_user and current_user.is_authenticated:  # type: ignore[attr-defined]
                db.log_activity(
                    username=current_user.username,  # type: ignore[attr-defined]
                    action="logout",
                    ip=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
        except Exception:
            pass
        logout_user()
        return redirect(url_for("auth.login"))

    @bp.route("/login/2fa", methods=["GET", "POST"])
    def login_2fa():  # type: ignore[override]
        """Two-Factor Authentication (TOTP) verification and setup flow.

        GET:
            - If the user has no TOTP secret, present a setup screen with a
              newly generated secret and otpauth URI for enrollment.
            - If the user already has a TOTP secret, present the OTP entry form.

        POST:
            - Validate the submitted TOTP code against the user's secret.
            - For first-time setup, persist the generated secret only after a
            correct OTP is provided (prevents accidental lockouts).

        Returns:
            A rendered template for OTP entry/setup on failure, or redirects to
            the main index upon successful verification and login.
        """
        pending_id_raw = session.get("2fa_pending_user_id")
        if not pending_id_raw:
            return redirect(url_for("auth.login"))
        try:
            pending_user_id = int(pending_id_raw)
        except Exception:
            # Defensive cleanup
            session.pop("2fa_pending_user_id", None)
            session.pop("2fa_pending_username", None)
            session.pop("2fa_pending_role", None)
            session.pop("2fa_remember", None)
            session.pop("2fa_setup_secret", None)
            return redirect(url_for("auth.login"))

        # Determine if this is a provisioning step (no stored secret).
        try:
            existing_secret = db.get_user_otp_secret(pending_user_id)
        except psycopg2.OperationalError:
            current_app.logger.exception("Database OperationalError fetching OTP secret")
            return render_template("500.html", title="Database Error", message="Database connection failed. Please try again later."), 500
        except Exception:
            current_app.logger.exception("Unexpected error fetching OTP secret")
            return render_template("500.html", title="Server Error", message="An unexpected error occurred. Please try again later."), 500

        provisioning: bool = existing_secret is None
        # Generate a setup secret when needed but do not persist yet.
        if provisioning and not session.get("2fa_setup_secret"):
            session["2fa_setup_secret"] = pyotp.random_base32()

        # Compute presentation fields
        username = str(session.get("2fa_pending_username") or "user")
        issuer = "Result GUI"
        secret_for_display = existing_secret or str(session.get("2fa_setup_secret") or "")
        otpauth_uri: str = ""
        if secret_for_display:
            try:
                otpauth_uri = pyotp.TOTP(secret_for_display).provisioning_uri(name=username, issuer_name=issuer)
            except Exception:
                otpauth_uri = ""
        qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=200x200&data={urllib.parse.quote_plus(otpauth_uri)}" if otpauth_uri else ""

        if request.method == "POST":
            code = (request.form.get("otp_code") or "").strip()
            # Choose the secret to validate against.
            chosen_secret = existing_secret or str(session.get("2fa_setup_secret") or "")
            if not chosen_secret:
                flash("Two-factor setup is not available at this time.", "danger")
                return render_template("login_2fa.html", provisioning=True, secret=secret_for_display, otpauth_uri=otpauth_uri, qr_url=qr_url)
            try:
                totp = pyotp.TOTP(chosen_secret)
                ok = bool(totp.verify(code, valid_window=1))
            except Exception:
                ok = False
            if not ok:
                flash("Invalid authentication code. Please try again.", "danger")
                return render_template("login_2fa.html", provisioning=provisioning, secret=secret_for_display, otpauth_uri=otpauth_uri, qr_url=qr_url)

            # Success: persist setup secret if provisioning
            if provisioning:
                try:
                    db.set_user_otp_secret(user_id=pending_user_id, secret=chosen_secret)
                except psycopg2.OperationalError:
                    current_app.logger.exception("Database OperationalError saving OTP secret")
                    return render_template("500.html", title="Database Error", message="Database connection failed. Please try again later."), 500
                except Exception:
                    current_app.logger.exception("Unexpected error saving OTP secret")
                    return render_template("500.html", title="Server Error", message="An unexpected error occurred. Please try again later."), 500
                finally:
                    session.pop("2fa_setup_secret", None)

            # Finalize login
            remember = bool(session.get("2fa_remember", False))
            pending_username = str(session.get("2fa_pending_username") or username)
            pending_role = str(session.get("2fa_pending_role") or "viewer")
            user = User(id=str(pending_user_id), username=pending_username, role=pending_role)
            login_user(user, remember=remember)
            # Activity log is best-effort.
            try:
                db.log_activity(
                    username=pending_username,
                    action="login",
                    ip=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception:
                pass
            # Cleanup pending session state
            session.pop("2fa_pending_user_id", None)
            session.pop("2fa_pending_username", None)
            session.pop("2fa_pending_role", None)
            session.pop("2fa_remember", None)
            return redirect(url_for("main.index"))

        # GET: render setup or verification page
        return render_template("login_2fa.html", provisioning=provisioning, secret=secret_for_display, otpauth_uri=otpauth_uri, qr_url=qr_url)

    app.register_blueprint(bp)


def hash_password(plaintext: str) -> str:
    """Hash a plaintext password using bcrypt.

    Args:
        plaintext: The plaintext password to hash.

    Returns:
        The hashed password as a UTF-8 string.
    """
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(plaintext.encode("utf-8"), salt).decode("utf-8")
