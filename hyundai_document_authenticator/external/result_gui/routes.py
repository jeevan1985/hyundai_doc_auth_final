"""Application routes for the Result GUI module.

This blueprint exposes the main index route for viewing results from either a
CSV file or a PostgreSQL table. It includes pagination, global search,
column-specific filtering, numeric threshold filtering, sorting, and a basic
role-restricted admin page.
"""
from __future__ import annotations

import csv
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os
import shutil
import time
import platform

from flask import Blueprint, Flask, abort, render_template, request, redirect, url_for, flash, jsonify, Response, current_app
from flask_login import login_required, current_user
import bcrypt
import psycopg2

from config_loader import AppConfig
from db import Database
from auth import hash_password
from i18n import normalize_lang, t


def init_routes(app: Flask, db: Database, config: AppConfig) -> None:
    """Register the main blueprint routes on the Flask application.

    Args:
        app: Flask application.
        db: Database access layer.
        config: Application configuration.
    """
    bp = Blueprint("main", __name__, template_folder="templates")

    # Process start time for uptime calculation
    _process_start_time: float = time.time()

    def _system_health() -> Dict[str, Any]:
        """Collect lightweight system health metrics for the Admin dashboard.

        Returns:
            A dictionary containing server and subsystem metrics suitable for JSON/CSV export.
        """
        # Server uptime and (best-effort) load
        now_ts: float = time.time()
        uptime_seconds: int = int(now_ts - _process_start_time)
        try:
            if hasattr(os, 'getloadavg'):
                load1, load5, load15 = os.getloadavg()  # type: ignore[attr-defined]
            else:
                load1 = load5 = load15 = None
        except Exception:
            load1 = load5 = load15 = None

        # Database connectivity check
        db_ok: bool = False
        db_err: Optional[str] = None
        try:
            # Lightweight ping using a trivial SELECT 1
            with db._get_conn() as conn:  # type: ignore[attr-defined]
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    _ = cur.fetchone()
                    db_ok = True
        except Exception as exc:  # noqa: BLE001
            db_ok = False
            db_err = str(exc)

        # Disk usage for current drive
        try:
            usage = shutil.disk_usage(os.getcwd())
            disk_total = int(usage.total)
            disk_used = int(usage.used)
            disk_free = int(usage.free)
        except Exception:
            disk_total = disk_used = disk_free = 0

        # Processing queue (placeholder hooks to be extended later)
        queue_status: str = "unknown"
        queue_backlog: int = 0

        # Recent errors/warnings (placeholder: empty list)
        recent_events: List[Dict[str, str]] = []

        return {
            "server": {
                "status": "ok",
                "hostname": platform.node(),
                "platform": platform.platform(),
                "uptime_seconds": uptime_seconds,
                "load_1m": load1,
                "load_5m": load5,
                "load_15m": load15,
            },
            "database": {
                "connected": db_ok,
                "error": db_err,
            },
            "queue": {
                "status": queue_status,
                "backlog": queue_backlog,
            },
            "storage": {
                "disk_total_bytes": disk_total,
                "disk_used_bytes": disk_used,
                "disk_free_bytes": disk_free,
            },
            "recent_events": recent_events,
            "generated_at": int(now_ts),
        }

    def _paginate(total: int, page: int, per_page: int) -> Dict[str, Any]:
        """Compute pagination metadata for templates.

        Args:
            total: Total number of items across all pages.
            page: Current 1-based page index.
            per_page: Items displayed per page.

        Returns:
            A dictionary with keys used by the template to render pagination
            controls and summary information.
        """
        total_pages = max(1, ceil(total / per_page))
        return {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
            "has_prev": page > 1,
            "has_next": page < total_pages,
            "prev_page": page - 1,
            "next_page": page + 1,
        }

    def _parse_pagination() -> Tuple[int, int]:
        """Parse pagination inputs from the query string with sane defaults.

        Returns:
            A tuple of (page, per_page) as validated integers.
        """
        try:
            page = int(request.args.get("page", "1"))
            per_page = int(request.args.get("per_page", "25"))
        except ValueError:
            page, per_page = 1, 25
        page = max(1, page)
        per_page = max(1, min(per_page, 200))
        return page, per_page

    def _safe_float(value: str) -> Optional[float]:
        """Safely convert a string to float.

        Args:
            value: String value to convert.

        Returns:
            A float if conversion succeeds; otherwise ``None``.
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _csv_apply_global_search(rows: List[List[str]], q: Optional[str]) -> List[List[str]]:
        """Filter CSV rows by a global case-insensitive search.

        Args:
            rows: CSV body rows (no header) as a list of lists of strings.
            q: Optional search query.

        Returns:
            Filtered rows that contain the query in any cell.
        """
        if not q:
            return rows
        needle = q.lower()
        return [r for r in rows if any(needle in str(val).lower() for val in r)]

    def _csv_apply_column_filter(
        columns: List[str],
        rows: List[List[str]],
        filter_col: Optional[str],
        filter_mode: str,
        filter_val: Optional[str],
    ) -> List[List[str]]:
        """Apply a column-specific string filter to CSV rows.

        Args:
            columns: Header column names.
            rows: CSV body rows.
            filter_col: Column name to filter on (case-insensitive match).
            filter_mode: Either "equals" or "contains".
            filter_val: String to compare against cell values.

        Returns:
            Filtered rows.
        """
        if not filter_col or filter_val is None:
            return rows
        idx_map = {c.lower(): i for i, c in enumerate(columns)}
        key = filter_col.lower()
        if key not in idx_map:
            return rows
        idx = idx_map[key]
        fval = filter_val.lower()
        if filter_mode not in {"equals", "contains"}:
            filter_mode = "equals"
        if filter_mode == "equals":
            return [r for r in rows if (r[idx] or "").lower() == fval]
        return [r for r in rows if fval in (r[idx] or "").lower()]

    def _csv_apply_threshold_filter(
        columns: List[str],
        rows: List[List[str]],
        threshold_col: Optional[str],
        threshold_op: str,
        threshold_val: Optional[float],
    ) -> List[List[str]]:
        """Apply a numeric threshold filter to CSV rows.

        Args:
            columns: Header column names.
            rows: CSV body rows.
            threshold_col: Column name for numeric comparison.
            threshold_op: One of ">=", ">", "<=", "<", "=".
            threshold_val: Numeric threshold.

        Returns:
            Filtered rows.
        """
        if not threshold_col or threshold_val is None:
            return rows
        idx_map = {c.lower(): i for i, c in enumerate(columns)}
        key = threshold_col.lower()
        if key not in idx_map:
            return rows
        idx = idx_map[key]
        allowed = {">=", ">", "<=", "<", "="}
        if threshold_op not in allowed:
            threshold_op = ">="

        def match(cell: str) -> bool:
            v = _safe_float(cell)
            if v is None:
                return False
            if threshold_op == ">=":
                return v >= threshold_val  # type: ignore[operator]
            if threshold_op == ">":
                return v > threshold_val  # type: ignore[operator]
            if threshold_op == "<=":
                return v <= threshold_val  # type: ignore[operator]
            if threshold_op == "<":
                return v < threshold_val  # type: ignore[operator]
            return v == threshold_val  # type: ignore[operator]

        return [r for r in rows if match(r[idx])]

    def _csv_apply_sort(
        columns: List[str], rows: List[List[str]], sort_by: Optional[str], sort_dir: str
    ) -> List[List[str]]:
        """Sort CSV rows by a column, attempting numeric-aware ordering.

        Args:
            columns: Header column names.
            rows: CSV body rows.
            sort_by: Column to sort by.
            sort_dir: Sort direction, either "asc" or "desc".

        Returns:
            Sorted rows (new list).
        """
        if not sort_by:
            return rows
        idx_map = {c.lower(): i for i, c in enumerate(columns)}
        key = sort_by.lower()
        if key not in idx_map:
            return rows
        idx = idx_map[key]
        reverse = (sort_dir or "asc").lower() == "desc"

        def k(row: List[str]) -> Tuple[int, Any]:
            v = row[idx]
            f = _safe_float(v)
            if f is None:
                return (1, (v or "").lower())
            return (0, f)

        return sorted(rows, key=k, reverse=reverse)

    def _csv_apply_advanced_filter(
        columns: List[str],
        rows: List[List[str]],
        filter_fields: Sequence[str],
        td_key: Optional[str],
        td_key_mode: str,
        td_score_op: str,
        td_score_val: Optional[float],
        ia_mode: str,
        ia_val: Optional[str],
    ) -> List[List[str]]:
        """Apply advanced JSON-aware filters with AND semantics for CSV.

        Behavior:
        - If both 'top_similar_docs' and 'image_authenticity' are selected, a row must satisfy both conditions.
        - With one selection, only that condition applies. With none, rows are returned unchanged.
        """
        import json  # local to ensure no top-level overhead
        if not filter_fields:
            return rows
        fields = [(f or "").strip().lower() for f in filter_fields]
        idx_map = {c.lower(): i for i, c in enumerate(columns)}

        def _match_top_sim(doc_map: Dict[str, Any]) -> bool:
            def _key_ok(k: str) -> bool:
                if not td_key:
                    return True
                tgt = td_key or ""
                if td_key_mode == "contains":
                    return tgt.lower() in k.lower()
                return k.lower() == tgt.lower()

            def _score_ok(v: Any) -> bool:
                if td_score_val is None:
                    return True
                try:
                    s = float(v)
                except Exception:
                    return False
                if td_score_op == ">=":
                    return s >= td_score_val
                if td_score_op == ">":
                    return s > td_score_val
                if td_score_op == "<=":
                    return s <= td_score_val
                if td_score_op == "<":
                    return s < td_score_val
                return s == td_score_val

            for k, v in (doc_map or {}).items():
                if _key_ok(str(k)) and _score_ok(v):
                    return True
            return False

        def _match_auth(auth_map: Dict[str, Any]) -> bool:
            if ia_val is None or ia_val == "":
                return True
            tgt = ia_val or ""
            for _k, cls in (auth_map or {}).items():
                sval = str(cls or "")
                if ia_mode == "contains":
                    if tgt.lower() in sval.lower():
                        return True
                else:
                    if sval.lower() == tgt.lower():
                        return True
            return False

        out: List[List[str]] = []
        for r in rows:
            try:
                ok = True
                if 'top_similar_docs' in fields:
                    i = idx_map.get('top_similar_docs')
                    if i is None:
                        ok = False
                    else:
                        try:
                            m = json.loads(r[i] or "{}")
                            if isinstance(m, list):
                                merged: Dict[str, Any] = {}
                                for d in m:
                                    if isinstance(d, dict):
                                        merged.update(d)
                                m = merged
                        except Exception:
                            m = {}
                        ok = ok and _match_top_sim(m)
                if ok and 'image_authenticity' in fields:
                    i = idx_map.get('image_authenticity')
                    if i is None:
                        ok = False
                    else:
                        try:
                            m = json.loads(r[i] or "{}")
                        except Exception:
                            m = {}
                        ok = ok and _match_auth(m)
                if ok:
                    out.append(r)
            except Exception:
                continue
        return out

    @bp.route("/")
    @login_required
    def index():  # type: ignore[override]
        """Render the main results table (CSV or DB based on configuration).

        Query parameters supported:
            page: 1-based page number (int)
            per_page: rows per page (int, max 200)
            q: global search across all columns (string)
            filter_col: column name for string filter (case-insensitive)
            filter_mode: equals|contains (default equals)
            filter_val: value for string filter (string)
            threshold_col: column name for numeric threshold
            threshold_op: one of >=, >, <=, <, = (default >=)
            threshold_val: numeric value for threshold
            sort_by: column name to sort
            sort_dir: asc|desc (default asc)
        """
        page, per_page = _parse_pagination()

        # Common filters from query params
        search = request.args.get("q")
        filter_col = request.args.get("filter_col") or None
        filter_mode = (request.args.get("filter_mode") or "equals").lower()
        filter_val = request.args.get("filter_val")
        # Advanced JSON-aware filters (support multi-select fields)
        raw_fields = request.args.getlist("filter_field")
        filter_fields: List[str] = []
        for f in raw_fields:
            val = (f or "").strip().lower()
            if val in {"top_similar_docs", "image_authenticity"} and val not in filter_fields:
                filter_fields.append(val)
        td_key = request.args.get("td_key") or None
        td_key_mode = (request.args.get("td_key_mode") or "equals").lower()
        td_score_op = (request.args.get("td_score_op") or ">=")
        td_score_val = _safe_float(request.args.get("td_score_val", ""))
        ia_mode = (request.args.get("ia_mode") or "equals").lower()
        ia_val = request.args.get("ia_val") or None
        sort_by = request.args.get("sort_by") or None
        sort_dir = (request.args.get("sort_dir") or "asc").lower()

        if config.use_csv:
            # Load from CSV
            assert config.csv_path is not None
            csv_path = Path(config.csv_path).expanduser().resolve()
            if not csv_path.exists():
                abort(404, description=f"CSV not found: {csv_path}")
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
            if not rows:
                columns: List[str] = []
                body: List[List[str]] = []
            else:
                columns = [c.strip() for c in rows[0]]
                body = [r for r in rows[1:]]

            # Apply filters in order: global search, column filter, advanced JSON filters, sort
            body = _csv_apply_global_search(body, search)
            body = _csv_apply_column_filter(columns, body, filter_col, filter_mode, filter_val)
            body = _csv_apply_advanced_filter(
                columns,
                body,
                filter_fields,
                td_key,
                td_key_mode,
                td_score_op,
                td_score_val,
                ia_mode,
                ia_val,
            )
            body = _csv_apply_sort(columns, body, sort_by, sort_dir)

            total = len(body)
            start = (page - 1) * per_page
            end = start + per_page
            page_rows = body[start:end]
            pagination = _paginate(total=total, page=page, per_page=per_page)
            # Build helper lists for UI (distinct classes from current CSV view if needed)
            ia_classes: List[str] = []
            try:
                import json as _json
                ia_idx = {c.lower(): i for i, c in enumerate(columns)}.get("image_authenticity")
                if ia_idx is not None:
                    seen: set[str] = set()
                    for r in body:
                        try:
                            m = _json.loads(r[ia_idx] or "{}")
                            for _k, v in (m or {}).items():
                                sval = str(v or "")
                                if sval:
                                    seen.add(sval)
                        except Exception:
                            continue
                    ia_classes = sorted(seen)
            except Exception:
                ia_classes = []

            return render_template(
                "index.html",
                columns=columns,
                rows=page_rows,
                pagination=pagination,
                search=search,
                use_csv=True,
                # Filter/sort context
                available_columns=columns,
                filter_col=filter_col,
                filter_mode=filter_mode,
                filter_val=filter_val,
                # Advanced filters context
                filter_fields=filter_fields,
                td_key=td_key,
                td_key_mode=td_key_mode,
                td_score_op=td_score_op,
                td_score_val=(request.args.get("td_score_val", "")),
                ia_mode=ia_mode,
                ia_val=ia_val or "",
                ia_classes=ia_classes,
                sort_by=sort_by,
                sort_dir=sort_dir,
            )
        else:
            # Load from DB results table
            assert config.results_table is not None
            try:
                columns, rows, total = db.fetch_results_page(
                    table_name=config.results_table,
                    page=page,
                    per_page=per_page,
                    search=search,
                    filter_col=filter_col,
                    filter_mode=filter_mode,
                    filter_val=filter_val,
                    # Advanced filters
                    filter_fields=filter_fields,
                    td_key=td_key,
                    td_key_mode=td_key_mode,
                    td_score_op=td_score_op,
                    td_score_val=td_score_val,
                    ia_mode=ia_mode,
                    ia_val=ia_val,
                    sort_by=sort_by,
                    sort_dir=sort_dir,
                )
                pagination = _paginate(total=total, page=page, per_page=per_page)
                # For DB mode, fetch distinct image_authenticity classes for dropdown suggestions
                try:
                    ia_classes = db.list_distinct_image_authenticity_classes(config.results_table)
                except Exception:
                    ia_classes = []
                return render_template(
                    "index.html",
                    columns=columns,
                    rows=rows,
                    pagination=pagination,
                    search=search,
                    use_csv=False,
                    # Filter/sort context
                    available_columns=columns,
                    filter_col=filter_col,
                    filter_mode=filter_mode,
                    filter_val=filter_val,
                    # Advanced filters context
                    filter_fields=filter_fields,
                    td_key=td_key,
                    td_key_mode=td_key_mode,
                    td_score_op=td_score_op,
                    td_score_val=(request.args.get("td_score_val", "")),
                    ia_mode=ia_mode,
                    ia_val=ia_val or "",
                    ia_classes=ia_classes,
                    sort_by=sort_by,
                    sort_dir=sort_dir,
                )
            except psycopg2.OperationalError:
                current_app.logger.exception("Database OperationalError on results page")
                return render_template("500.html", title="Database Error", message="Database connection failed. Please try again later."), 500
            except Exception:
                current_app.logger.exception("Unexpected error on results page")
                return render_template("500.html", title="Server Error", message="An unexpected error occurred. Please try again later."), 500

    @bp.route("/admin")
    @login_required
    def admin():  # type: ignore[override]
        """Render a real admin dashboard with metrics and recent activity."""
        # Basic RBAC: only allow role 'admin'
        role = getattr(current_user, "role", "viewer")
        if role != "admin":
            abort(403)
        # Metrics and recent activity
        try:
            metrics = db.get_admin_metrics()
            online_users = set(db.list_current_online_users())
        except psycopg2.OperationalError:
            current_app.logger.exception("Database OperationalError in admin dashboard")
            return render_template("500.html", title="Database Error", message="Database connection failed. Please try again later."), 500
        except Exception:
            current_app.logger.exception("Unexpected error in admin dashboard")
            return render_template("500.html", title="Server Error", message="An unexpected error occurred. Please try again later."), 500
        try:
            page = int(request.args.get("page", "1"))
            per_page = int(request.args.get("per_page", "10"))
        except ValueError:
            page, per_page = 1, 10
        page = max(1, page)
        per_page = max(1, min(per_page, 100))

        # Sorting parameters from query string
        sort_by = request.args.get("sort_by") or None
        sort_dir = (request.args.get("sort_dir") or "desc").lower()
        if sort_dir not in {"asc", "desc"}:
            sort_dir = "desc"

        # Filters
        f_username = (request.args.get("f_username") or "").strip() or None
        f_action = (request.args.get("f_action") or "").strip() or None
        f_ip = (request.args.get("f_ip") or "").strip() or None
        f_ts = (request.args.get("f_ts") or "").strip() or None
        f_user_agent = (request.args.get("f_user_agent") or "").strip() or None
        f_user_id_raw = (request.args.get("f_user_id") or "").strip()
        try:
            f_user_id = int(f_user_id_raw) if f_user_id_raw else None
        except ValueError:
            f_user_id = None
        f_entry_raw = (request.args.get("f_entry") or "").strip()
        try:
            f_entry = int(f_entry_raw) if f_entry_raw else None
        except ValueError:
            f_entry = None
        # Optional status filter from unified control support (online/offline), kept hidden for now
        f_status = (request.args.get("f_status") or "").strip().lower() or None

        try:
            activity, total = db.list_recent_activity(
                page=page,
                per_page=per_page,
                sort_by=sort_by,
                sort_dir=sort_dir,
                username=f_username,
                action=f_action,
                ip=f_ip,
                ts_query=f_ts,
                user_agent=f_user_agent,
                user_id=f_user_id,
                entry=f_entry,
                status=f_status,
            )
        except psycopg2.OperationalError:
            current_app.logger.exception("Database OperationalError while listing recent activity")
            return render_template("500.html", title="Database Error", message="Database connection failed. Please try again later."), 500
        except Exception:
            current_app.logger.exception("Unexpected error while listing recent activity")
            return render_template("500.html", title="Server Error", message="An unexpected error occurred. Please try again later."), 500
        pagination = {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": max(1, (total + per_page - 1) // per_page),
            "has_prev": page > 1,
            "has_next": page * per_page < total,
            "prev_page": page - 1,
            "next_page": page + 1,
        }
        # Provide initial system health snapshot for template if needed
        sys_health = _system_health()
        return render_template(
            "admin.html",
            metrics=metrics,
            activity=activity,
            pagination=pagination,
            system_health=sys_health,
            sort_by=sort_by or "timestamp",
            sort_dir=sort_dir,
            f_username=f_username or "",
            f_action=f_action or "",
            f_ip=f_ip or "",
            f_ts=f_ts or "",
            f_user_agent=f_user_agent or "",
            f_user_id=str(f_user_id or ""),
            f_entry=str(f_entry or ""),
            online_users=online_users,
            f_status=f_status or "",
        )

    @bp.route("/admin/activity/clear", methods=["POST"])
    @login_required
    def clear_activity() -> Any:  # type: ignore[override]
        """Clear recent activity logs for admin-selected time range.

        This endpoint enforces admin RBAC. It accepts a form field
        'range_choice' with one of: 'all', 'today', or 'days_N' where N is 1..31.
        It maps the selection to the database clearing function and flashes a
        localized message indicating the number of records deleted. Invalid
        input results in no deletion and a localized error flash.

        Returns:
            A redirect back to the Admin dashboard.
        """
        _require_admin()
        from flask import g
        lang = getattr(g, 'lang', 'en')

        choice = (request.form.get("range_choice") or "").strip().lower()
        try:
            if choice == "all":
                deleted = db.clear_activity_logs(days=None, today=False)
            elif choice == "today":
                deleted = db.clear_activity_logs(today=True)
            elif choice.startswith("days_"):
                try:
                    n = int(choice.split("_", 1)[1])
                except Exception as exc:  # noqa: BLE001
                    raise ValueError("Invalid days value") from exc
                deleted = db.clear_activity_logs(days=n, today=False)
            else:
                raise ValueError("Invalid selection")
            flash(t("alerts.activity_cleared", lang).format(count=deleted), "success")
        except ValueError:
            # Defensive: invalid input or out-of-range N
            flash(t("alerts.activity_clear_invalid", lang), "danger")
        except Exception:
            # Any unexpected error: do not leak details to user
            flash(t("alerts.activity_clear_invalid", lang), "danger")
        return redirect(url_for("main.admin"))

    # ----------------------------
    # Admin: Manage Users
    # ----------------------------

    def _require_admin() -> None:
        """Abort with 403 if current user is not an admin."""
        role = getattr(current_user, "role", "viewer")
        if role != "admin":
            abort(403)

    @bp.route("/users", methods=["GET"])
    @login_required
    def users_index():  # type: ignore[override]
        """List users with pagination and optional search (admin-only)."""
        _require_admin()
        try:
            page = int(request.args.get("page", "1"))
            per_page = int(request.args.get("per_page", "25"))
        except ValueError:
            page, per_page = 1, 25
        page = max(1, page)
        per_page = max(1, min(per_page, 200))
        user_q = request.args.get("user_q")

        users, total = db.list_users(page=page, per_page=per_page, search=user_q)
        pagination = {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": max(1, (total + per_page - 1) // per_page),
            "has_prev": page > 1,
            "has_next": page * per_page < total,
            "prev_page": page - 1,
            "next_page": page + 1,
        }
        return render_template(
            "manage_users.html",
            users=users,
            pagination=pagination,
            user_q=user_q,
        )

    @bp.route("/users/create", methods=["POST"])
    @login_required
    def users_create():  # type: ignore[override]
        """Create a new user (admin-only)."""
        _require_admin()
        username = (request.form.get("username") or "").strip()
        password = (request.form.get("password") or "").strip()
        role = (request.form.get("role") or "viewer").strip()
        if not username or not password:
            from flask import g
            flash(t("alerts.username_password_required", getattr(g, 'lang', 'en')), "danger")
            return redirect(url_for("main.users_index"))
        if role not in {"admin", "viewer"}:
            role = "viewer"
        try:
            pwd_hash = hash_password(password)
            db.create_user(username=username, password_hash=pwd_hash, role=role)
            from flask import g
            flash(t("alerts.user_created", getattr(g, 'lang', 'en')).format(username=username), "success")
        except Exception as exc:  # noqa: BLE001
            flash(str(exc), "danger")
        return redirect(url_for("main.users_index"))

    @bp.route("/users/<int:user_id>/role", methods=["POST"])
    @login_required
    def users_update_role(user_id: int):  # type: ignore[override]
        """Update a user's role (admin-only)."""
        _require_admin()
        role = (request.form.get("role") or "viewer").strip()
        if role not in {"admin", "viewer"}:
            from flask import g
            flash(t("alerts.invalid_role", getattr(g, 'lang', 'en')), "danger")
            return redirect(url_for("main.users_index"))
        try:
            db.update_user_role(user_id=user_id, role=role)
            from flask import g
            flash(t("alerts.role_updated", getattr(g, 'lang', 'en')), "success")
        except Exception as exc:  # noqa: BLE001
            flash(str(exc), "danger")
        return redirect(url_for("main.users_index"))

    @bp.route("/users/<int:user_id>/password", methods=["POST"])
    @login_required
    def users_update_password(user_id: int):  # type: ignore[override]
        """Reset a user's password (admin-only)."""
        _require_admin()
        new_password = (request.form.get("password") or "").strip()
        if not new_password:
            from flask import g
            flash(t("alerts.password_required", getattr(g, 'lang', 'en')), "danger")
            return redirect(url_for("main.users_index"))
        try:
            pwd_hash = hash_password(new_password)
            db.update_user_password_by_id(user_id=user_id, new_password_hash=pwd_hash)
            from flask import g
            flash(t("alerts.password_updated", getattr(g, 'lang', 'en')), "success")
        except Exception as exc:  # noqa: BLE001
            flash(str(exc), "danger")
        return redirect(url_for("main.users_index"))

    @bp.route("/users/<int:user_id>/delete", methods=["POST"])
    @login_required
    def users_delete(user_id: int):  # type: ignore[override]
        """Delete a user (admin-only)."""
        _require_admin()
        try:
            db.delete_user(user_id)
            from flask import g
            flash(t("alerts.user_deleted", getattr(g, 'lang', 'en')), "success")
        except Exception as exc:  # noqa: BLE001
            flash(str(exc), "danger")
        return redirect(url_for("main.users_index"))

    @bp.route("/admin/system_health.json")
    @login_required
    def system_health_json():  # type: ignore[override]
        """Return system health metrics as JSON (admin-only)."""
        role = getattr(current_user, "role", "viewer")
        if role != "admin":
            abort(403)
        return jsonify(_system_health())

    @bp.route("/admin/system_health.csv")
    @login_required
    def system_health_csv():  # type: ignore[override]
        """Download system health metrics as CSV (admin-only)."""
        role = getattr(current_user, "role", "viewer")
        if role != "admin":
            abort(403)
        m = _system_health()
        # Flatten metrics into key,value rows for ease of consumption
        rows: List[Tuple[str, str]] = []
        rows.append(("server.status", str(m.get("server", {}).get("status"))))
        rows.append(("server.hostname", str(m.get("server", {}).get("hostname"))))
        rows.append(("server.platform", str(m.get("server", {}).get("platform"))))
        rows.append(("server.uptime_seconds", str(m.get("server", {}).get("uptime_seconds"))))
        rows.append(("server.load_1m", str(m.get("server", {}).get("load_1m"))))
        rows.append(("server.load_5m", str(m.get("server", {}).get("load_5m"))))
        rows.append(("server.load_15m", str(m.get("server", {}).get("load_15m"))))
        rows.append(("database.connected", str(m.get("database", {}).get("connected"))))
        rows.append(("database.error", str(m.get("database", {}).get("error"))))
        rows.append(("queue.status", str(m.get("queue", {}).get("status"))))
        rows.append(("queue.backlog", str(m.get("queue", {}).get("backlog"))))
        rows.append(("storage.disk_total_bytes", str(m.get("storage", {}).get("disk_total_bytes"))))
        rows.append(("storage.disk_used_bytes", str(m.get("storage", {}).get("disk_used_bytes"))))
        rows.append(("storage.disk_free_bytes", str(m.get("storage", {}).get("disk_free_bytes"))))
        rows.append(("generated_at", str(m.get("generated_at"))))

        out_lines = ["key,value"] + [f"{k},{v}" for (k, v) in rows]
        csv_text = "\r\n".join(out_lines)
        resp = Response(csv_text, mimetype="text/csv; charset=utf-8")
        resp.headers["Content-Disposition"] = "attachment; filename=system_health.csv"
        return resp

    @bp.route("/profile", methods=["GET", "POST"])
    @login_required
    def profile():  # type: ignore[override]
        """Allow the current user to view profile and change password.

        POST expects fields: current_password, new_password, confirm_password.
        """
        if request.method == "POST":
            current_password = (request.form.get("current_password") or "").encode("utf-8")
            new_password = (request.form.get("new_password") or "").strip()
            confirm_password = (request.form.get("confirm_password") or "").strip()
            from flask import g
            lang = getattr(g, 'lang', 'en')
            if not new_password:
                flash(t("alerts.password_required", lang), "danger")
                return redirect(url_for("main.profile"))
            if new_password != confirm_password:
                flash(t("alerts.password_mismatch", lang), "danger")
                return redirect(url_for("main.profile"))
            # Verify current password
            try:
                rec = db.get_user_by_id(int(current_user.id))  # type: ignore[arg-type]
            except Exception:
                rec = None
            if not rec or not bcrypt.checkpw(current_password, rec.password_hash.encode("utf-8")):
                flash(t("alerts.current_password_incorrect", lang), "danger")
                return redirect(url_for("main.profile"))
            try:
                pwd_hash = hash_password(new_password)
                db.update_user_password_by_id(user_id=int(current_user.id), new_password_hash=pwd_hash)  # type: ignore[arg-type]
                flash(t("alerts.password_updated", lang), "success")
            except Exception as exc:  # noqa: BLE001
                flash(str(exc), "danger")
            return redirect(url_for("main.profile"))
        return render_template("profile.html")

    @bp.route("/set_lang", methods=["POST"])
    def set_lang():  # type: ignore[override]
        """Set the user's preferred language in session (available pre-login).

        Accepts a form value 'lang' (e.g., 'en' or 'ko') and persists it in the
        session so future requests render with the selected language.
        """
        from flask import session  # local import to avoid circulars
        lang = normalize_lang(request.form.get("lang"))
        session["lang"] = lang
        # Redirect back to referrer or index
        ref = request.headers.get("Referer")
        return redirect(ref or url_for("main.index"))

    @bp.route("/set_theme", methods=["POST"])
    def set_theme():  # type: ignore[override]
        """Set the user's preferred theme in session (available pre-login).

        Supported values are 'default', 'light', 'dark', 'retro', 'cyberpunk',
        'glass', and 'auto'. Defaults to 'default' when an unsupported value is
        provided. The selection is stored in the session and applied via data
        attributes and CSS variables.
        """
        from flask import session  # local import to avoid circulars
        choice = (request.form.get("theme") or "default").strip().lower()
        allowed = {"default", "light", "dark", "retro", "cyberpunk", "glass", "auto"}
        theme = choice if choice in allowed else "default"
        session["theme"] = theme
        ref = request.headers.get("Referer")
        return redirect(ref or url_for("main.index"))

    app.register_blueprint(bp)
