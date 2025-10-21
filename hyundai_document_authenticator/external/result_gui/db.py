"""Database utilities for the Result GUI module.

This module encapsulates all PostgreSQL interactions required by the GUI. It
manages a connection pool, ensures the authentication schema and users table
exist, provides user management helpers, and offers paginated retrieval of
result rows from a configured results table.

The implementation uses psycopg2 with a simple connection pool and SQL
composition for safe, injection-resistant queries.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

import psycopg2
from psycopg2 import sql
from psycopg2.pool import SimpleConnectionPool

from config_loader import AppConfig


@dataclass(frozen=True)
class UserRecord:
    """Typed view of a user record from the database.

    Attributes:
        id: Unique user identifier.
        username: Username for login.
        password_hash: Bcrypt-hashed password string.
        role: Role of the user (e.g., "admin", "viewer").
        created_at: Timestamp string (ISO-ish) of creation.
    """

    id: int
    username: str
    password_hash: str
    role: str
    created_at: str


@dataclass(frozen=True)
class ActivityRecord:
    """Typed view of an activity log entry.

    Attributes:
        id: Unique activity identifier.
        user_id: Optional foreign key back to users.id when available.
        username: Username associated with the event.
        action: One of "login" or "logout".
        ip: Remote IP address string, if available.
        user_agent: User agent string, if available.
        created_at: Timestamp string of when the event occurred.
    """

    id: int
    username: str
    action: str
    ip: Optional[str]
    user_agent: Optional[str]
    created_at: str
    user_id: Optional[int]


class Database:
    """PostgreSQL database access layer with connection pooling.

    This class is dedicated to the Result GUI module and must not be imported by
    or rely on the main project.
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialize the database connection pool.

        Args:
            config: The loaded application configuration, providing database
                connection settings and schema names.
        """
        self._config = config
        dsn = (
            f"host={config.db_host} port={config.db_port} dbname={config.db_name} "
            f"user={config.db_user} password={config.db_password}"
        )
        # Lazily create the connection pool on first use to avoid startup failures
        # when the database is temporarily unavailable.
        self._dsn = dsn
        self._pool: Optional[SimpleConnectionPool] = None

    def _ensure_pool(self) -> None:
        """Ensure the PostgreSQL connection pool is initialized.

        This defers actual database connectivity until first use, allowing the
        application to start even if the database is temporarily unavailable.
        """
        if self._pool is None:
            self._pool = SimpleConnectionPool(minconn=1, maxconn=10, dsn=self._dsn)

    @contextmanager
    def _get_conn(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """Context manager yielding a pooled connection.

        Yields:
            A psycopg2 pooled connection.
        """
        self._ensure_pool()
        assert self._pool is not None
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    def ping(self) -> bool:
        """Check database connectivity using a lightweight query.

        Returns:
            True if the database is reachable and responds to a trivial query;
            otherwise False.

        Notes:
            This method is designed to be side-effect free and fast. It avoids
            leaking driver or DSN details by swallowing all exceptions and
            returning a boolean result.
        """
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    _ = cur.fetchone()
            return True
        except Exception:
            return False

    def ensure_auth_schema_and_table(self) -> None:
        """Create authentication schema and users table if they do not exist.

        This method is idempotent and safe to call multiple times.
        """
        schema_ident = sql.Identifier(self._config.auth_db_schema)
        table_ident = sql.Identifier(self._config.auth_db_schema, self._config.app_user_table)
        activity_ident = sql.Identifier(self._config.auth_db_schema, "activity_log")

        create_schema = sql.SQL("CREATE SCHEMA IF NOT EXISTS {schema}").format(
            schema=schema_ident
        )
        create_users_table = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table} (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'viewer',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        ).format(table=table_ident)
        create_activity_table = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table} (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NULL,
                username TEXT NOT NULL,
                action TEXT NOT NULL CHECK (action IN ('login','logout')),
                ip TEXT NULL,
                user_agent TEXT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        ).format(table=activity_ident)
        # Optional column for Two-Factor Authentication (TOTP) secret per user.
        alter_users_add_otp = sql.SQL(
            "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS otp_secret TEXT NULL"
        ).format(table=table_ident)

        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_schema)
                    cur.execute(create_users_table)
                    cur.execute(create_activity_table)
                    cur.execute(alter_users_add_otp)
                    conn.commit()
        except psycopg2.OperationalError:
            # Start gracefully when DB is unavailable; the schema/table will be
            # created once the database becomes reachable.
            return

    def get_user_by_username(self, username: str) -> Optional[UserRecord]:
        """Fetch a user by username.

        Args:
            username: Username to search for.

        Returns:
            A ``UserRecord`` if found; otherwise ``None``.
        """
        query = sql.SQL(
            "SELECT id, username, password_hash, role, created_at "
            "FROM {table} WHERE username = %s"
        ).format(table=sql.Identifier(self._config.auth_db_schema, self._config.app_user_table))

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (username,))
                row = cur.fetchone()
                if not row:
                    return None
                return UserRecord(
                    id=row[0], username=row[1], password_hash=row[2], role=row[3], created_at=str(row[4])
                )

    def get_user_by_id(self, user_id: int) -> Optional[UserRecord]:
        """Fetch a user by ID.

        Args:
            user_id: User identifier.

        Returns:
            A ``UserRecord`` if found; otherwise ``None``.
        """
        query = sql.SQL(
            "SELECT id, username, password_hash, role, created_at "
            "FROM {table} WHERE id = %s"
        ).format(table=sql.Identifier(self._config.auth_db_schema, self._config.app_user_table))

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (user_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return UserRecord(
                    id=row[0], username=row[1], password_hash=row[2], role=row[3], created_at=str(row[4])
                )

    def create_user(self, username: str, password_hash: str, role: str = "viewer") -> int:
        """Create a new user account.

        Args:
            username: Desired username (must be unique).
            password_hash: Bcrypt-hashed password string.
            role: Role for the user (default "viewer").

        Returns:
            The ``id`` of the newly created user.

        Raises:
            ValueError: If the username already exists.
        """
        insert = sql.SQL(
            "INSERT INTO {table} (username, password_hash, role) VALUES (%s, %s, %s) RETURNING id"
        ).format(table=sql.Identifier(self._config.auth_db_schema, self._config.app_user_table))

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(insert, (username, password_hash, role))
                    new_id = cur.fetchone()[0]
                    conn.commit()
                    return int(new_id)
                except psycopg2.errors.UniqueViolation as exc:  # pragma: no cover - depends on DB constraint timing
                    conn.rollback()
                    raise ValueError(f"Username already exists: {username}") from exc

    def update_user_password_by_id(self, user_id: int, new_password_hash: str) -> None:
        """Update the password hash for the given user ID.

        Args:
            user_id: Identifier of the user to update.
            new_password_hash: New bcrypt password hash to store.

        Raises:
            ValueError: If the user does not exist.
        """
        update = sql.SQL(
            "UPDATE {table} SET password_hash=%s WHERE id=%s RETURNING id"
        ).format(table=sql.Identifier(self._config.auth_db_schema, self._config.app_user_table))
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(update, (new_password_hash, user_id))
                row = cur.fetchone()
                if not row:
                    conn.rollback()
                    raise ValueError(f"User not found: id={user_id}")
                conn.commit()

    def update_user_role(self, user_id: int, role: str) -> None:
        """Update the role of a user.

        Args:
            user_id: Identifier of the user to update.
            role: New role string (e.g., "admin" or "viewer").

        Raises:
            ValueError: If the user does not exist.
        """
        update = sql.SQL(
            "UPDATE {table} SET role=%s WHERE id=%s RETURNING id"
        ).format(table=sql.Identifier(self._config.auth_db_schema, self._config.app_user_table))
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(update, (role, user_id))
                row = cur.fetchone()
                if not row:
                    conn.rollback()
                    raise ValueError(f"User not found: id={user_id}")
                conn.commit()

    def delete_user(self, user_id: int) -> None:
        """Delete a user by ID.

        Args:
            user_id: Identifier of the user to delete.

        Raises:
            ValueError: If the user does not exist.
        """
        delete = sql.SQL(
            "DELETE FROM {table} WHERE id=%s RETURNING id"
        ).format(table=sql.Identifier(self._config.auth_db_schema, self._config.app_user_table))
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(delete, (user_id,))
                row = cur.fetchone()
                if not row:
                    conn.rollback()
                    raise ValueError(f"User not found: id={user_id}")
                conn.commit()

    def list_users(self, page: int, per_page: int, search: Optional[str] = None) -> Tuple[List[UserRecord], int]:
        """List users with pagination and optional username search.

        Args:
            page: 1-based page index.
            per_page: Number of users per page (capped at 200).
            search: Optional case-insensitive substring to match in usernames.

        Returns:
            A tuple (users, total) where users is a list of UserRecord for the
            requested page and total is the total number of matching users.
        """
        page = max(1, int(page))
        per_page = max(1, min(int(per_page), 200))
        offset = (page - 1) * per_page

        where = sql.SQL("")
        params: List[Any] = []
        if search:
            where = sql.SQL(" WHERE username ILIKE %s ")
            params.append(f"%{search}%")

        count_q = (
            sql.SQL("SELECT COUNT(*) FROM {table}").format(
                table=sql.Identifier(self._config.auth_db_schema, self._config.app_user_table)
            )
            + where
        )
        data_q = (
            sql.SQL(
                "SELECT id, username, password_hash, role, created_at FROM {table}"
            ).format(table=sql.Identifier(self._config.auth_db_schema, self._config.app_user_table))
            + where
            + sql.SQL(" ORDER BY id ASC LIMIT %s OFFSET %s")
        )

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(count_q, params or None)
                total = int(cur.fetchone()[0])
                cur.execute(data_q, (params + [per_page, offset]) if params else [per_page, offset])
                rows = cur.fetchall()

        users = [
            UserRecord(id=r[0], username=r[1], password_hash=r[2], role=r[3], created_at=str(r[4])) for r in rows
        ]
        return users, total

    def get_user_otp_secret(self, user_id: int) -> Optional[str]:
        """Retrieve the TOTP secret for a user if configured.

        Args:
            user_id: Identifier of the user.

        Returns:
            The base32 TOTP secret string if present; otherwise None.
        """
        q = sql.SQL("SELECT otp_secret FROM {t} WHERE id=%s").format(
            t=sql.Identifier(self._config.auth_db_schema, self._config.app_user_table)
        )
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(q, (user_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return str(row[0]) if row[0] is not None else None

    def set_user_otp_secret(self, user_id: int, secret: str) -> None:
        """Set or update the TOTP secret for a user.

        Args:
            user_id: Identifier of the user.
            secret: Base32 TOTP secret string.
        """
        q = sql.SQL("UPDATE {t} SET otp_secret=%s WHERE id=%s").format(
            t=sql.Identifier(self._config.auth_db_schema, self._config.app_user_table)
        )
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(q, (secret, user_id))
                conn.commit()

    def list_table_columns(self, table_name: str) -> List[str]:
        """List column names for a given table in ordinal order.

        Args:
            table_name: Target table name, optionally schema-qualified
                (e.g., "public.my_table").

        Returns:
            A list of column names in order.
        """
        return self._get_table_columns(table_name)

    def _get_table_columns(self, table_name: str) -> List[str]:
        """Retrieve ordered column names for the given table.

        Args:
            table_name: Target table name. May include schema (e.g., "public.my_table").

        Returns:
            A list of column names in ordinal position order.
        """
        # Support optional schema-qualified table names
        if "." in table_name:
            schema, table = table_name.split(".", 1)
            ident_table = sql.Identifier(schema, table)
            schema_name = schema
            table_only = table
        else:
            ident_table = sql.Identifier(table_name)
            schema_name = "public"
            table_only = table_name

        query = sql.SQL(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
            """
        )
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (schema_name, table_only))
                rows = cur.fetchall()
                if rows:
                    return [r[0] for r in rows]

        # Fallback: SELECT * LIMIT 0 to inspect cursor description (less reliable if no privileges)
        query2 = sql.SQL("SELECT * FROM {table} LIMIT 0").format(table=ident_table)
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query2)
                return [desc.name for desc in cur.description]

    def fetch_results_page(
        self,
        table_name: str,
        page: int,
        per_page: int,
        search: Optional[str] = None,
        *,
        filter_col: Optional[str] = None,
        filter_mode: str = "equals",
        filter_val: Optional[str] = None,
        filter_field: Optional[str] = None,
        filter_fields: Optional[Sequence[str]] = None,
        td_key: Optional[str] = None,
        td_key_mode: str = "equals",
        td_score_op: str = ">=",
        td_score_val: Optional[float] = None,
        ia_mode: str = "equals",
        ia_val: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_dir: str = "asc",
    ) -> Tuple[List[str], List[Sequence[Any]], int]:
        """Fetch a page of results with optional filters and sorting.

        Args:
            table_name: Name of the results table (optionally schema-qualified).
            page: 1-based page number to retrieve.
            per_page: Number of rows per page (capped at 200 for safety).
            search: Optional case-insensitive search string to match across all
                columns (ILIKE against text casts), combined with other filters
                using AND semantics.
            filter_col: Optional column name for a column-specific string filter.
            filter_mode: One of {"equals", "contains"}. Defaults to "equals".
                - equals: case-insensitive equality comparison (ILIKE without %).
                - contains: case-insensitive substring match (ILIKE with %value%).
            filter_val: Filter value for filter_col.
            filter_field: Optional legacy single advanced filter field name ('top_similar_docs' or 'image_authenticity').
            filter_fields: Optional sequence of advanced filter fields. When multiple are provided,
                their conditions are combined with AND semantics.
            td_key: Optional document key to match inside top_similar_docs mapping.
            td_key_mode: equals|contains for td_key.
            td_score_op: One of ">=", ">", "<=", "<", "=" for top_similar_docs score comparison.
            td_score_val: Numeric score value to compare against in top_similar_docs.
            ia_mode: equals|contains for image_authenticity class values.
            ia_val: Optional class value to match inside image_authenticity mapping.
            sort_by: Optional column name to sort by. If not provided, defaults
                to ordering by the first column.
            sort_dir: Sort direction: "asc" or "desc" (default "asc").

        Returns:
            A tuple of (columns, rows, total_count), where:
                - columns: List of column names.
                - rows: List of row sequences for the requested page.
                - total_count: Total number of rows matching the filter.

        Raises:
            ValueError: If invalid operators or column names are provided.
        """
        page = max(1, int(page))
        per_page = max(1, min(int(per_page), 200))  # cap per_page to 200 for safety
        offset = (page - 1) * per_page

        # Handle schema-qualified names properly
        if "." in table_name:
            schema, table = table_name.split(".", 1)
            ident_table = sql.Identifier(schema, table)
        else:
            ident_table = sql.Identifier(table_name)

        columns = self._get_table_columns(table_name)
        col_map = {c.lower(): c for c in columns}

        # Validate sort parameters
        sort_dir_norm = (sort_dir or "asc").lower()
        if sort_dir_norm not in {"asc", "desc"}:
            raise ValueError("sort_dir must be 'asc' or 'desc'")

        order_by_sql = sql.SQL(" ORDER BY 1 ")
        if sort_by:
            key = sort_by.lower()
            if key not in col_map:
                raise ValueError(f"Unknown sort_by column: {sort_by}")
            order_by_sql = sql.SQL(" ORDER BY {col} {dir} ").format(
                col=sql.Identifier(col_map[key]),
                dir=sql.SQL(sort_dir_norm.upper()),
            )

        # Build WHERE components
        where_parts: List[sql.SQL] = []
        where_params: List[Any] = []

        # Global search across columns
        if search:
            like = f"%{search}%"
            ors = [sql.SQL("{col}::text ILIKE %s").format(col=sql.Identifier(col)) for col in columns]
            where_parts.append(sql.SQL("(") + sql.SQL(" OR ").join(ors) + sql.SQL(")"))
            where_params.extend([like] * len(columns))

        # Column-specific string filter
        if filter_col and filter_val is not None:
            fkey = filter_col.lower()
            if fkey not in col_map:
                raise ValueError(f"Unknown filter_col: {filter_col}")
            if filter_mode not in {"equals", "contains"}:
                raise ValueError("filter_mode must be 'equals' or 'contains'")
            pattern = filter_val if filter_mode == "equals" else f"%{filter_val}%"
            where_parts.append(
                sql.SQL("{col}::text ILIKE %s").format(col=sql.Identifier(col_map[fkey]))
            )
            where_params.append(pattern)

        # Advanced JSON-aware filters (support multi-select via AND semantics)
        selected_fields: List[str] = []
        if filter_field:
            f = (filter_field or "").strip().lower()
            if f in {"top_similar_docs", "image_authenticity"} and f not in selected_fields:
                selected_fields.append(f)
        if filter_fields:
            for f in filter_fields:
                ff = (f or "").strip().lower()
                if ff in {"top_similar_docs", "image_authenticity"} and ff not in selected_fields:
                    selected_fields.append(ff)

        # top_similar_docs condition
        if "top_similar_docs" in selected_fields:
            conds_ts: List[sql.SQL] = []
            params_ts: List[Any] = []
            key_expr = sql.SQL("(kv.key)")
            val_expr = sql.SQL("(kv.value)::numeric")
            join_ts = sql.SQL(
                " EXISTS (SELECT 1 FROM jsonb_each_text({col}) AS kv WHERE TRUE"
            ).format(col=sql.Identifier(col_map.get("top_similar_docs", "top_similar_docs")))
            if td_key:
                if td_key_mode == "contains":
                    conds_ts.append(sql.SQL(" LOWER({k}) LIKE %s ").format(k=key_expr))
                    params_ts.append(f"%{td_key.lower()}%")
                else:
                    conds_ts.append(sql.SQL(" LOWER({k}) = %s ").format(k=key_expr))
                    params_ts.append(td_key.lower())
            if (td_score_val is not None) and (td_score_op in {">=", ">", "<=", "<", "="}):
                conds_ts.append(sql.SQL(" {v} {op} %s ").format(v=val_expr, op=sql.SQL(td_score_op)))
                params_ts.append(td_score_val)
            # Only add EXISTS when there is at least one condition for the field
            if conds_ts:
                where_parts.append(join_ts + sql.SQL(" AND ") + sql.SQL(" AND ").join(conds_ts) + sql.SQL(")"))
                where_params.extend(params_ts)

        # image_authenticity condition
        if "image_authenticity" in selected_fields:
            conds_ia: List[sql.SQL] = []
            params_ia: List[Any] = []
            val_expr = sql.SQL("LOWER(kv.value)")
            join_ia = sql.SQL(
                " EXISTS (SELECT 1 FROM jsonb_each_text({col}) AS kv WHERE TRUE"
            ).format(col=sql.Identifier(col_map.get("image_authenticity", "image_authenticity")))
            if ia_val:
                if ia_mode == "contains":
                    conds_ia.append(sql.SQL(" {v} LIKE %s ").format(v=val_expr))
                    params_ia.append(f"%{(ia_val or '').lower()}%")
                else:
                    conds_ia.append(sql.SQL(" {v} = %s ").format(v=val_expr))
                    params_ia.append((ia_val or '').lower())
            if conds_ia:
                where_parts.append(join_ia + sql.SQL(" AND ") + sql.SQL(" AND ").join(conds_ia) + sql.SQL(")"))
                where_params.extend(params_ia)

        where_clause = sql.SQL("")
        if where_parts:
            where_clause = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_parts)

        count_query = sql.SQL("SELECT COUNT(*) FROM {table}").format(table=ident_table) + where_clause
        data_query = (
            sql.SQL("SELECT * FROM {table}").format(table=ident_table)
            + where_clause
            + order_by_sql
            + sql.SQL(" LIMIT %s OFFSET %s")
        )

        rows: List[Sequence[Any]] = []
        total: int = 0
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                # total count
                cur.execute(count_query, where_params or None)
                total = int(cur.fetchone()[0])

                # page data
                cur.execute(data_query, (where_params + [per_page, offset]) if where_params else [per_page, offset])
                rows = cur.fetchall()

        return columns, rows, total

    def list_distinct_image_authenticity_classes(self, table_name: str) -> List[str]:
        """List distinct class values present inside image_authenticity JSONB values.

        Args:
            table_name: Name of the results table (optionally schema-qualified).

        Returns:
            A sorted list of distinct class strings (lowercased) found in image_authenticity.
        """
        # Resolve identifier
        if "." in table_name:
            schema, table = table_name.split(".", 1)
            ident_table = sql.Identifier(schema, table)
        else:
            ident_table = sql.Identifier(table_name)
        query = sql.SQL(
            """
            SELECT DISTINCT LOWER(value) AS cls
            FROM (
                SELECT jsonb_each_text(image_authenticity)::text AS kv
                FROM {t}
                WHERE image_authenticity IS NOT NULL
            ) AS t
            CROSS JOIN LATERAL jsonb_each_text((kv)::jsonb) AS e(key, value)
            WHERE value IS NOT NULL AND value <> ''
            ORDER BY cls ASC
            """
        ).format(t=ident_table)
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(query)
                    rows = cur.fetchall()
                    return [str(r[0]) for r in rows if r and r[0]]
                except Exception:
                    return []

    # ----------------------------
    # Admin metrics and activity
    # ----------------------------

    def log_activity(self, username: str, action: str, ip: Optional[str], user_agent: Optional[str]) -> None:
        """Log a user activity event (login/logout).

        Args:
            username: Username associated with the event.
            action: Either "login" or "logout".
            ip: Remote IP address if available.
            user_agent: User agent string if available.
        """
        if action not in {"login", "logout"}:
            return
        # Lookup user_id if present
        subq = sql.SQL("SELECT id FROM {ut} WHERE username=%s").format(
            ut=sql.Identifier(self._config.auth_db_schema, self._config.app_user_table)
        )
        ins = sql.SQL(
            "INSERT INTO {at} (user_id, username, action, ip, user_agent) VALUES (%s,%s,%s,%s,%s)"
        ).format(at=sql.Identifier(self._config.auth_db_schema, "activity_log"))
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(subq, (username,))
                r = cur.fetchone()
                uid = int(r[0]) if r else None
                cur.execute(ins, (uid, username, action, ip, user_agent))
                conn.commit()

    def list_recent_activity(
        self,
        page: int,
        per_page: int,
        sort_by: Optional[str] = None,
        sort_dir: str = "desc",
        *,
        username: Optional[str] = None,
        action: Optional[str] = None,
        ip: Optional[str] = None,
        ts_query: Optional[str] = None,
        user_agent: Optional[str] = None,
        user_id: Optional[int] = None,
        entry: Optional[int] = None,
        status: Optional[str] = None,
    ) -> Tuple[List[ActivityRecord], int]:
        """List recent activity log entries with pagination, sorting, and filters.

        Args:
            page: 1-based page index.
            per_page: Number of records per page (capped at 200).
            sort_by: Optional sort key. Supported values: "entry" (or "id"),
                "username", "timestamp" (or "created_at"), "ip". Defaults to
                "timestamp" when not provided or invalid.
            sort_dir: Sort direction, either "asc" or "desc". Defaults to
                "desc".
            username: Optional case-insensitive substring to match in username.
            action: Optional action filter; only "login" or "logout" are accepted.
            ip: Optional case-insensitive substring to match in IP address.
            ts_query: Optional timestamp search. If value matches "YYYY-MM-DD",
                records from that day are returned. If value matches
                "YYYY-MM-DD HH:MM:SS", records whose timestamp truncated to the
                second equals the provided value are returned. Invalid formats
                are ignored and do not filter results.
            user_agent: Optional case-insensitive substring to match in user agent.
            user_id: Optional exact numeric user ID match.
            entry: Optional exact numeric entry id (same as id) match.

        Returns:
            A tuple of (records, total_count) for the requested page.
        """
        page = max(1, int(page))
        per_page = max(1, min(int(per_page), 200))
        offset = (page - 1) * per_page

        table_ident = sql.Identifier(self._config.auth_db_schema, "activity_log")
        base_count = sql.SQL("SELECT COUNT(*) FROM {t}").format(t=table_ident)
        base_data = sql.SQL(
            """
            SELECT id, username, action, ip, user_agent, created_at, user_id
            FROM {t}
            """
        ).format(t=table_ident)

        # Build WHERE filters
        where_parts: List[sql.SQL] = []
        params: List[Any] = []

        if username:
            where_parts.append(sql.SQL(" username ILIKE %s "))
            params.append(f"%{username}%")

        if action:
            act = (action or "").strip().lower()
            if act in {"login", "logout"}:
                where_parts.append(sql.SQL(" action = %s "))
                params.append(act)

        if ip:
            where_parts.append(sql.SQL(" ip ILIKE %s "))
            params.append(f"%{ip}%")

        if user_agent:
            where_parts.append(sql.SQL(" user_agent ILIKE %s "))
            params.append(f"%{user_agent}%")

        if user_id is not None:
            where_parts.append(sql.SQL(" user_id = %s "))
            params.append(int(user_id))

        if entry is not None:
            where_parts.append(sql.SQL(" id = %s "))
            params.append(int(entry))

        if ts_query:
            ts = (ts_query or "").strip()
            # Date-only: YYYY-MM-DD
            if len(ts) == 10 and ts[4] == "-" and ts[7] == "-":
                where_parts.append(sql.SQL(" created_at >= %s::date AND created_at < (%s::date + INTERVAL '1 day') "))
                params.extend([ts, ts])
            # Full timestamp with seconds: YYYY-MM-DD HH:MM:SS
            elif len(ts) >= 19 and ts[4] == "-" and ts[7] == "-" and ts[10] == " " and ts[13] == ":" and ts[16] == ":":
                where_parts.append(sql.SQL(" date_trunc('second', created_at) = %s::timestamp "))
                params.append(ts[:19])
            # Otherwise: ignore invalid pattern

        # Status filter ('online' or 'offline') based on last action per username
        if status:
            st = (status or "").strip().lower()
            if st in {"online", "offline"}:
                if st == "online":
                    where_parts.append(sql.SQL(
                        " username IN (SELECT username FROM (SELECT username, action, ROW_NUMBER() OVER (PARTITION BY username ORDER BY created_at DESC) rn FROM {at}) AS last WHERE rn=1 AND action='login') "
                    ).format(at=sql.Identifier(self._config.auth_db_schema, "activity_log")))
                else:
                    where_parts.append(sql.SQL(
                        " (username IS NULL OR username NOT IN (SELECT username FROM (SELECT username, action, ROW_NUMBER() OVER (PARTITION BY username ORDER BY created_at DESC) rn FROM {at}) AS last WHERE rn=1 AND action='login')) "
                    ).format(at=sql.Identifier(self._config.auth_db_schema, "activity_log")))

        where_sql = sql.SQL("")
        if where_parts:
            where_sql = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_parts)

        # Map external sort keys to database columns
        col_map = {
            "entry": "id",
            "id": "id",
            "username": "username",
            "timestamp": "created_at",
            "created_at": "created_at",
            "ip": "ip",
        }
        key = (sort_by or "timestamp").strip().lower() if sort_by else "timestamp"
        order_col = col_map.get(key, "created_at")
        dir_norm = (sort_dir or "desc").strip().lower()
        dir_norm = "desc" if dir_norm not in {"asc", "desc"} else dir_norm

        # Primary ORDER BY plus deterministic tiebreaker
        order_sql = sql.SQL(" ORDER BY {col} {dir}, id {dir2} ").format(
            col=sql.Identifier(order_col),
            dir=sql.SQL(dir_norm.upper()),
            dir2=sql.SQL(dir_norm.upper()),
        )

        q_count = base_count + where_sql
        q_data = base_data + where_sql + order_sql + sql.SQL(" LIMIT %s OFFSET %s")

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(q_count, params or None)
                total = int(cur.fetchone()[0])
                cur.execute(q_data, (params + [per_page, offset]) if params else [per_page, offset])
                rows = cur.fetchall()

        records: List[ActivityRecord] = [
            ActivityRecord(
                id=int(r[0]),
                username=str(r[1]),
                action=str(r[2]),
                ip=(str(r[3]) if r[3] is not None else None),
                user_agent=(str(r[4]) if r[4] is not None else None),
                created_at=str(r[5]),
                user_id=(int(r[6]) if r[6] is not None else None),
            )
            for r in (rows or [])
        ]
        return records, total

    def list_current_online_users(self) -> List[str]:
        """List usernames that are currently online.

        A user is considered online if their most recent activity action is 'login'.

        Returns:
            A list of usernames currently online.
        """
        q = sql.SQL(
            """
            WITH last AS (
                SELECT username, action,
                       ROW_NUMBER() OVER (PARTITION BY username ORDER BY created_at DESC) AS rn
                FROM {at}
            )
            SELECT username FROM last WHERE rn=1 AND action='login' AND username IS NOT NULL
            """
        ).format(at=sql.Identifier(self._config.auth_db_schema, "activity_log"))
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(q)
                rows = cur.fetchall()
                return [str(r[0]) for r in rows if r and r[0]]

    def clear_activity_logs(self, days: Optional[int] = None, today: bool = False) -> int:
        """Clear activity logs by time range.

        Behavior:
        - When days is None and today is False: delete all rows.
        - When today is True: delete rows with created_at >= date_trunc('day', NOW()).
        - When days is provided (1..31): delete rows with created_at >= NOW() - make_interval(days => days).

        Args:
            days: Optional number of past days to retain (1..31). When provided,
                rows within this interval are deleted.
            today: If True, delete only today's rows (from start of current day).

        Returns:
            The number of rows deleted.

        Raises:
            ValueError: If parameter combination is invalid, or days is out of range (1..31).
        """
        # Validate mutually exclusive parameters
        if days is not None and today:
            raise ValueError("Specify either 'days' or 'today', not both")

        table_ident = sql.Identifier(self._config.auth_db_schema, "activity_log")

        where_sql: Optional[sql.SQL] = None
        params: List[Any] = []

        if days is None and not today:
            # Delete all rows
            where_sql = None
        elif today:
            # Delete from start of current day
            where_sql = sql.SQL(" WHERE created_at >= date_trunc('day', NOW()) ")
        else:
            # days provided
            try:
                d = int(days) if days is not None else None
            except Exception as exc:  # noqa: BLE001
                raise ValueError("Invalid days value") from exc
            if d is None or d < 1 or d > 31:
                raise ValueError("days must be between 1 and 31")
            where_sql = sql.SQL(" WHERE created_at >= NOW() - make_interval(days => %s) ")
            params.append(d)

        delete_sql = sql.SQL("DELETE FROM {t}").format(t=table_ident)
        if where_sql:
            delete_sql = delete_sql + where_sql

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(delete_sql, params or None)
                deleted = int(cur.rowcount)
                conn.commit()
        return deleted

    def get_admin_metrics(self) -> Dict[str, int]:
        """Compute key admin metrics: total users, sign-ups this week, and online users.

        Returns:
            A dictionary with keys: total_users, signups_week, total_active_sessions.
            The key active_sessions_12h is preserved for backward compatibility and
            equals total_active_sessions.
        """
        q_total = sql.SQL(
            "SELECT COUNT(*) FROM {ut}"
        ).format(ut=sql.Identifier(self._config.auth_db_schema, self._config.app_user_table))
        q_week = sql.SQL(
            "SELECT COUNT(*) FROM {ut} WHERE created_at >= NOW() - INTERVAL '7 days'"
        ).format(ut=sql.Identifier(self._config.auth_db_schema, self._config.app_user_table))
        # Online users: last action per username (no time window) is 'login'
        q_online = sql.SQL(
            """
            WITH last AS (
                SELECT username, action,
                       ROW_NUMBER() OVER (PARTITION BY username ORDER BY created_at DESC) AS rn
                FROM {at}
            )
            SELECT COUNT(*) FROM last WHERE rn=1 AND action='login'
            """
        ).format(at=sql.Identifier(self._config.auth_db_schema, "activity_log"))

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(q_total)
                total_users = int(cur.fetchone()[0])
                cur.execute(q_week)
                signups_week = int(cur.fetchone()[0])
                cur.execute(q_online)
                total_active = int(cur.fetchone()[0])
        return {
            "total_users": total_users,
            "signups_week": signups_week,
            "total_active_sessions": total_active,
            "active_sessions_12h": total_active,
        }

    