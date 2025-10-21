#!/usr/bin/env python3
"""Database connectivity and schema tester. 🛡️

This tool validates PostgreSQL connectivity and common operational checks using
connection parameters supplied via environment variables (typically from a .env
file), with optional CLI overrides. It is designed to be safe in production
(enforced read-only by default) and to provide explicit subcommands for common
validation tasks.

Subcommands
- ping: Verify that a connection to the configured database can be established.
- ensure-database: Ensure the configured database exists; create it if missing (requires createdb privileges).
- drop-database: Drop a database by name (requires --confirm-side-effects). Terminates active sessions first.
- table-exists: Check if a specific table exists (optionally under a given schema).
- describe: List columns for a specific table with types and nullability.
- write-test: Create a short-lived test table, run a CRUD cycle, then drop it
  (skipped unless explicitly requested to avoid side effects).
- drop-table: Drop a specified table (requires --confirm-side-effects) for safe, explicit schema resets.
- ensure-doc-sim-table: Ensure the document similarity results table exists with canonical schema; optionally add missing columns.

Environment variables (fallbacks for CLI options)
- POSTGRES_DB
- POSTGRES_HOST
- POSTGRES_PORT
- POSTGRES_USER
- POSTGRES_PASSWORD

Usage
  python -m hyundai_document_authenticator.tool_database_tester ping
  python -m hyundai_document_authenticator.tool_database_tester ensure-database --db mydb
  python -m hyundai_document_authenticator.tool_database_tester drop-database --dbname mydb --confirm-side-effects
  python hyundai_document_authenticator/tool_database_tester.py table-exists --table doc_similarity_results
  python -m hyundai_document_authenticator.tool_database_tester describe --table doc_similarity_results
  python hyundai_document_authenticator/tool_database_tester.py write-test --confirm-side-effects
  python -m hyundai_document_authenticator.tool_database_tester drop-table --table doc_similarity_results --schema public --confirm-side-effects
  python -m hyundai_document_authenticator.tool_database_tester ensure-doc-sim-table --table doc_similarity_results --schema public
  python -m hyundai_document_authenticator.tool_database_tester ensure-doc-sim-table --table doc_similarity_results --schema public --add-missing-columns
  
  python hyundai_document_authenticator/tool_database_tester.py \      
      ping --host 127.0.0.1 --port 5432 --db mydb --user myuser --password secret

Exit codes
  0 on success, non-zero on failures.

Note
  - Default operations are read-only and safe for production validation.
  - Subcommands that perform DDL require explicit flags:
    * write-test: creates and drops a temporary test table only when --confirm-side-effects is provided.
    * drop-table: drops the specified table only when --confirm-side-effects is provided.
    * drop-database: drops a database only when --confirm-side-effects is provided; will terminate active sessions.
    * ensure-doc-sim-table: creates the canonical document-sim results table when missing; with --add-missing-columns it will ALTER TABLE to add missing non-PK columns.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent


def setup_logging(level: str = "INFO") -> None:
    """Configure logging.

    Args:
        level (str): Logging level name (e.g., "DEBUG", "INFO").
    """
    # Centralize logs under APP_LOG_DIR/tools when available; fall back to console-only otherwise.
    env_log_dir = os.getenv("APP_LOG_DIR")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if env_log_dir:
        try:
            tools_dir = Path(env_log_dir) / "tools"
            tools_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(tools_dir / "tool_database_tester.log"), encoding="utf-8")
            handlers.append(fh)
        except Exception:
            # Proceed with console-only logging on failure
            pass

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
        handlers=handlers,
    )


@dataclass(frozen=True)
class PgConfig:
    """Immutable PostgreSQL connection parameters.

    Attributes:
        db (str): Database name.
        host (str): Hostname or IP of the server.
        port (int): TCP port.
        user (str): Database user.
        password (str): Database password.
    """

    db: str
    host: str
    port: int
    user: str
    password: str


def load_env() -> None:
    """Load environment variables from a local .env if available. 🧯

    This function is defensive: it logs when .env is absent and continues.
    """
    try:
        from dotenv import load_dotenv

        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            LOGGER.info("Loaded .env from %s", env_path)
        else:
            LOGGER.debug("No .env found at %s; relying on process env.", env_path)
    except Exception as e:  # keep tool robust even if dotenv is missing
        LOGGER.debug("dotenv not used: %s", e)


def env_or_default(key: str, default: Optional[str] = None) -> Optional[str]:
    """Fetch an environment variable with a default fallback.

    Args:
        key (str): Environment variable name.
        default (Optional[str]): Optional default value.

    Returns:
        Optional[str]: The environment value or the default.
    """
    import os

    return os.getenv(key, default)


def resolve_pg_config(args: argparse.Namespace) -> PgConfig:
    """Resolve PostgreSQL connection parameters from CLI and environment.

    Priority (highest first): CLI option -> environment -> default.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        PgConfig: Sanitized, complete connection configuration.
    """
    db = args.db or env_or_default("POSTGRES_DB") or "postgres"
    host = args.host or env_or_default("POSTGRES_HOST") or "localhost"
    port_str = args.port or env_or_default("POSTGRES_PORT") or "5432"
    user = args.user or env_or_default("POSTGRES_USER") or "postgres"
    password = args.password or env_or_default("POSTGRES_PASSWORD") or ""

    try:
        port = int(port_str)
    except Exception:
        raise ValueError(f"Invalid port value: {port_str}")

    return PgConfig(db=db, host=host, port=port, user=user, password=password)


def connect(cfg: PgConfig):
    """Establish and return a psycopg2 connection using the given config.

    Args:
        cfg (PgConfig): Connection parameters.

    Returns:
        Any: psycopg2 connection object.

    Raises:
        ImportError: If psycopg2 is not installed.
        psycopg2.Error: On connection failure.
    """
    try:
        import psycopg2

        return psycopg2.connect(
            dbname=cfg.db,
            user=cfg.user,
            password=cfg.password,
            host=cfg.host,
            port=cfg.port,
        )
    except ImportError as e:
        LOGGER.error("psycopg2 is required for database operations: %s", e)
        raise


def cmd_ping(cfg: PgConfig) -> int:
    """Ping the database and report connectivity and basic info. 🩺

    Args:
        cfg (PgConfig): Connection parameters.

    Returns:
        int: 0 on success; 1 on failure.
    """
    try:
        with connect(cfg) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version: str = cur.fetchone()[0]
                LOGGER.info("Connected to '%s' at %s:%d as %s", cfg.db, cfg.host, cfg.port, cfg.user)
                LOGGER.info("Server version: %s", version)
        return 0
    except Exception as e:
        LOGGER.error("Ping failed: %s", e, exc_info=True)
        return 1


def table_exists(conn: Any, table: str, schema: str = "public") -> bool:
    """Check if a table exists in the given schema.

    Args:
        conn (Any): psycopg2 connection.
        table (str): Table name.
        schema (str): Schema name (default: public).

    Returns:
        bool: True if table exists; False otherwise.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
            LIMIT 1
            """,
            (schema, table),
        )
        return cur.fetchone() is not None


def describe_table(conn: Any, table: str, schema: str = "public") -> List[Tuple[str, str, str, Optional[str]]]:
    """Return column metadata for a table.

    Args:
        conn (Any): psycopg2 connection.
        table (str): Table name.
        schema (str): Schema name (default: public).

    Returns:
        List[Tuple[str, str, str, Optional[str]]]: Columns as (name, data_type, is_nullable, default).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
            """,
            (schema, table),
        )
        return [(r[0], r[1], r[2], r[3]) for r in cur.fetchall()]


def cmd_table_exists(cfg: PgConfig, table: str, schema: str = "public") -> int:
    """CLI wrapper to check table existence.

    Args:
        cfg (PgConfig): Connection parameters.
        table (str): Table name.
        schema (str): Schema name.

    Returns:
        int: 0 if exists; 1 otherwise.
    """
    try:
        with connect(cfg) as conn:
            exists = table_exists(conn, table=table, schema=schema)
            if exists:
                LOGGER.info("Table %s.%s exists.", schema, table)
                return 0
            LOGGER.warning("Table %s.%s does not exist.", schema, table)
            return 1
    except Exception as e:
        LOGGER.error("table-exists failed: %s", e, exc_info=True)
        return 2


def cmd_describe(cfg: PgConfig, table: str, schema: str = "public") -> int:
    """CLI wrapper to describe a table.

    Args:
        cfg (PgConfig): Connection parameters.
        table (str): Table name.
        schema (str): Schema name.

    Returns:
        int: 0 on success; non-zero on failure.
    """
    try:
        with connect(cfg) as conn:
            if not table_exists(conn, table=table, schema=schema):
                LOGGER.warning("Table %s.%s does not exist.", schema, table)
                return 1
            cols = describe_table(conn, table=table, schema=schema)
            if not cols:
                LOGGER.warning("No columns found for %s.%s.", schema, table)
                return 1
            LOGGER.info("Columns for %s.%s:", schema, table)
            for name, dtype, nullable, default in cols:
                LOGGER.info("  - %-24s %-16s null=%-3s default=%s", name, dtype, nullable, default)
            return 0
    except Exception as e:
        LOGGER.error("describe failed: %s", e, exc_info=True)
        return 2


def cmd_write_test(cfg: PgConfig, confirm_side_effects: bool, schema: str = "public") -> int:
    """Create, write, read, and drop a temporary test table to validate DDL/DML privileges. ✍️

    Args:
        cfg (PgConfig): Connection parameters.
        confirm_side_effects (bool): Must be True to proceed; prevents accidental writes in production.
        schema (str): Schema to create the table in (default: public).

    Returns:
        int: 0 on success; non-zero on failures.
    """
    if not confirm_side_effects:
        LOGGER.error("Refusing to perform write-test without --confirm-side-effects.")
        return 2

    test_table = "_db_test_smoke"
    try:
        from psycopg2 import sql  # type: ignore

        with connect(cfg) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                ident = sql.Identifier(schema, test_table)
                # Drop if exists
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(ident))
                # Create
                cur.execute(
                    sql.SQL(
                        "CREATE TABLE {} (id SERIAL PRIMARY KEY, info TEXT NOT NULL, created_at TIMESTAMPTZ DEFAULT NOW());"
                    ).format(ident)
                )
                # Insert
                cur.execute(sql.SQL("INSERT INTO {} (info) VALUES (%s) RETURNING id;").format(ident), ("hello",))
                new_id = cur.fetchone()[0]
                # Select
                cur.execute(sql.SQL("SELECT id, info FROM {} WHERE id = %s;").format(ident), (new_id,))
                row = cur.fetchone()
                assert row[1] == "hello"
                LOGGER.info("Write-test succeeded. Inserted row id=%s", new_id)
                # Cleanup
                cur.execute(sql.SQL("DROP TABLE {};").format(ident))
                LOGGER.info("Cleanup complete: dropped %s.%s", schema, test_table)
        return 0
    except Exception as e:
        LOGGER.error("write-test failed: %s", e, exc_info=True)
        return 1


def cmd_drop_table(cfg: PgConfig, table: str, schema: str = "public", confirm_side_effects: bool = False) -> int:
    """Drop a specified table in the given schema (guarded by a confirmation flag).

    This subcommand is intentionally explicit to avoid accidental destructive actions
    in production. It performs a schema-qualified drop and returns an appropriate
    exit code without raising.

    Args:
        cfg (PgConfig): Connection parameters.
        table (str): Table name to drop.
        schema (str): Schema name (default: public).
        confirm_side_effects (bool): Must be True to proceed; protects production.

    Returns:
        int: 0 on success; non-zero on failures or refusal.

    Raises:
        None
    """
    if not confirm_side_effects:
        LOGGER.error("Refusing to drop table without --confirm-side-effects.")
        return 2

    try:
        from psycopg2 import sql  # type: ignore
        with connect(cfg) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                ident = sql.Identifier(schema, table)
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {} RESTRICT;").format(ident))
                LOGGER.info("Dropped table if existed: %s.%s", schema, table)
        return 0
    except Exception as e:
        LOGGER.error("drop-table failed: %s", e, exc_info=True)
        return 1


def cmd_ensure_doc_sim_table(
    cfg: PgConfig,
    table: str = "doc_similarity_results",
    schema: str = "public",
    add_missing_columns: bool = False,
) -> int:
    """Ensure the document similarity results table exists with the expected schema.

    Behavior:
    - If the table does not exist, it is created with the canonical schema used by
      the TIF document similarity save path (dynamic JSONB columns included by default).
    - If the table exists, the current columns are introspected. When
      --add-missing-columns is provided, any missing non-PK columns are added. The
      primary key column (id SERIAL PRIMARY KEY) is not added to existing tables; if
      it is absent, a drop/recreate is recommended instead of in-place migration.

    Expected schema (canonical):
      id SERIAL PRIMARY KEY
      run_identifier TEXT
      requesting_username TEXT
      search_timestamp TIMESTAMPTZ DEFAULT NOW()
      parent_document_name TEXT
      highest_similarity_score FLOAT
      sim_img_check JSONB
      image_authenticity JSONB
      fraud_doc_probability JSONB
      global_top_docs JSONB

    Args:
        cfg (PgConfig): Connection parameters.
        table (str): Target table name (default: doc_similarity_results).
        schema (str): Schema name (default: public).
        add_missing_columns (bool): When True, attempt to ALTER TABLE ADD COLUMN
            for any missing non-PK columns.

    Returns:
        int: 0 on success; non-zero when a mismatch is detected and not corrected,
        or on errors.

    Raises:
        None
    """
    try:
        from psycopg2 import sql  # type: ignore
        with connect(cfg) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                # Helper: list existing columns
                cur.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (schema, table),
                )
                existing_cols = [r[0] for r in cur.fetchall()]
                exists = len(existing_cols) > 0

                # Canonical expected columns and types (excluding SERIAL PK for add-missing path)
                expected_types = {
                    "id": "SERIAL PRIMARY KEY",
                    "run_identifier": "TEXT",
                    "requesting_username": "TEXT",
                    "search_timestamp": "TIMESTAMPTZ",
                    "parent_document_name": "TEXT",
                    "highest_similarity_score": "FLOAT",
                    "sim_img_check": "JSONB",
                    "image_authenticity": "JSONB",
                    "fraud_doc_probability": "JSONB",
                    "global_top_docs": "JSONB",
                }

                if not exists:
                    ident = sql.Identifier(schema, table)
                    cols_sql = sql.SQL(
                        ",\n    ".join(
                            [
                                "id SERIAL PRIMARY KEY",
                                "run_identifier TEXT",
                                "requesting_username TEXT",
                                "search_timestamp TIMESTAMPTZ DEFAULT NOW()",
                                "parent_document_name TEXT",
                                "highest_similarity_score FLOAT",
                                "sim_img_check JSONB",
                                "image_authenticity JSONB",
                                "fraud_doc_probability JSONB",
                                "global_top_docs JSONB",
                            ]
                        )
                    )
                    ddl = sql.SQL("CREATE TABLE {} (\n    {}\n);").format(ident, cols_sql)
                    cur.execute(ddl)
                    LOGGER.info("Created table %s.%s with the canonical document-sim schema.", schema, table)
                    return 0

                # Table exists: compute missing columns
                existing_set = set(existing_cols)
                missing = [col for col in expected_types.keys() if col not in existing_set]
                if not missing:
                    LOGGER.info("Table %s.%s already has all expected columns.", schema, table)
                    return 0

                # If id is missing, refuse to add and recommend drop/recreate
                if "id" in missing:
                    LOGGER.warning(
                        "Existing table %s.%s is missing primary key column 'id'. "
                        "In-place addition is not supported by this tool. Consider dropping the table "
                        "and recreating it with the ensure command.",
                        schema,
                        table,
                    )
                    missing = [c for c in missing if c != "id"]

                if not missing:
                    return 1

                if not add_missing_columns:
                    LOGGER.warning(
                        "Table %s.%s is missing columns: %s. Re-run with --add-missing-columns to apply ALTERs.",
                        schema,
                        table,
                        ", ".join(missing),
                    )
                    return 1

                ident_tbl = sql.Identifier(schema, table)
                for col in missing:
                    col_ident = sql.Identifier(col)
                    # For search_timestamp, set default NOW() when adding
                    if col == "search_timestamp":
                        alter = sql.SQL("ALTER TABLE {} ADD COLUMN {} TIMESTAMPTZ DEFAULT NOW();").format(ident_tbl, col_ident)
                    else:
                        col_type_sql = sql.SQL(expected_types[col])
                        alter = sql.SQL("ALTER TABLE {} ADD COLUMN {} {};").format(ident_tbl, col_ident, col_type_sql)
                    cur.execute(alter)
                    LOGGER.info("Added missing column %s %s to %s.%s", col, expected_types[col], schema, table)

                LOGGER.info("Schema alignment complete for %s.%s.", schema, table)
                return 0
    except Exception as e:
        LOGGER.error("ensure-doc-sim-table failed: %s", e, exc_info=True)
        return 1


def cmd_ensure_database(cfg: PgConfig, target_db: Optional[str] = None) -> int:
    """Ensure a PostgreSQL database exists; create it if missing.

    Connects to the control DB (postgres, falling back to template1) and creates
    the target database when it does not exist. Requires CREATEDB privileges or
    appropriate role permissions.

    Returns: 0 on success; non-zero on failure.
    """
    try:
        import psycopg2
        from psycopg2 import sql
        control_db_candidates = ["postgres", "template1"]
        dbname = target_db or cfg.db
        for ctrl in control_db_candidates:
            try:
                with psycopg2.connect(dbname=ctrl, user=cfg.user, password=cfg.password, host=cfg.host, port=cfg.port) as conn:
                    conn.autocommit = True
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
                        if cur.fetchone():
                            LOGGER.info("Database '%s' already exists.", dbname)
                            return 0
                        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
                        LOGGER.info("Created database '%s'.", dbname)
                        return 0
            except psycopg2.OperationalError:
                continue
        LOGGER.error("Could not connect to a control database (postgres/template1) to create '%s'.", dbname)
        return 1
    except Exception as e:
        LOGGER.error("ensure-database failed: %s", e, exc_info=True)
        return 1


def cmd_drop_database(cfg: PgConfig, target_db: str, confirm_side_effects: bool = False) -> int:
    """Drop a PostgreSQL database by name.

    To avoid DROP DATABASE failures due to active connections, this function first
    terminates other sessions connected to the target database. Requires sufficient
    privileges and --confirm-side-effects.

    Returns: 0 on success; non-zero on failure/refusal.
    """
    if not confirm_side_effects:
        LOGGER.error("Refusing to drop database without --confirm-side-effects.")
        return 2
    try:
        import psycopg2
        from psycopg2 import sql
        control_db_candidates = ["postgres", "template1"]
        last_error: Optional[Exception] = None
        for ctrl in control_db_candidates:
            try:
                with psycopg2.connect(dbname=ctrl, user=cfg.user, password=cfg.password, host=cfg.host, port=cfg.port) as conn:
                    conn.autocommit = True
                    with conn.cursor() as cur:
                        # Terminate other sessions on target_db (ignore current backend)
                        cur.execute(
                            """
                            SELECT pg_terminate_backend(pid)
                            FROM pg_stat_activity
                            WHERE datname = %s AND pid <> pg_backend_pid();
                            """,
                            (target_db,),
                        )
                        cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(target_db)))
                        LOGGER.info("Dropped database if existed: %s", target_db)
                        return 0
            except Exception as e:
                last_error = e
                continue
        if last_error:
            LOGGER.error("drop-database failed: %s", last_error, exc_info=True)
        else:
            LOGGER.error("drop-database failed: could not reach a control database to execute DROP.")
        return 1
    except Exception as e:
        LOGGER.error("drop-database failed: %s", e, exc_info=True)
        return 1



def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser with subcommands.
    """
    p = argparse.ArgumentParser(
        prog="tool_database_tester",
        description="PostgreSQL connectivity and schema test utility.",
    )
    p.add_argument("--log", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")

    # Global connection overrides
    p.add_argument("--db", help="Database name (fallback: POSTGRES_DB)")
    p.add_argument("--host", help="PostgreSQL host (fallback: POSTGRES_HOST)")
    p.add_argument("--port", help="PostgreSQL port (fallback: POSTGRES_PORT)")
    p.add_argument("--user", help="PostgreSQL user (fallback: POSTGRES_USER)")
    p.add_argument("--password", help="PostgreSQL password (fallback: POSTGRES_PASSWORD)")

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ping", help="Verify connectivity and print server version.")

    sp_exists = sub.add_parser("table-exists", help="Check if a table exists.")
    sp_exists.add_argument("--table", required=True, help="Table name to check.")
    sp_exists.add_argument("--schema", default="public", help="Schema name (default: public).")

    sp_desc = sub.add_parser("describe", help="Describe a table's columns.")
    sp_desc.add_argument("--table", required=True, help="Table name to describe.")
    sp_desc.add_argument("--schema", default="public", help="Schema name (default: public).")

    sp_write = sub.add_parser(
        "write-test",
        help="Perform a short-lived create/insert/select/drop cycle in a temporary test table.",
    )
    sp_write.add_argument(
        "--confirm-side-effects",
        action="store_true",
        help="Required to proceed; prevents accidental writes in production.",
    )
    sp_write.add_argument("--schema", default="public", help="Schema name (default: public).")

    sp_drop = sub.add_parser(
        "drop-table",
        help="Drop a specified table in a schema (requires --confirm-side-effects).",
    )
    sp_drop.add_argument("--table", required=True, help="Table name to drop.")
    sp_drop.add_argument("--schema", default="public", help="Schema name (default: public).")
    sp_drop.add_argument(
        "--confirm-side-effects",
        action="store_true",
        help="Required to proceed; prevents accidental destructive operations.",
    )

    sp_db_ensure = sub.add_parser(
        "ensure-database",
        help="Ensure the configured database exists; create it if missing.",
    )
    sp_db_ensure.add_argument(
        "--dbname",
        help="Target database name (default: --db or POSTGRES_DB).",
    )

    sp_db_drop = sub.add_parser(
        "drop-database",
        help="Drop a database (requires --confirm-side-effects). Terminates active sessions first.",
    )
    sp_db_drop.add_argument("--dbname", required=True, help="Database name to drop.")
    sp_db_drop.add_argument(
        "--confirm-side-effects",
        action="store_true",
        help="Required to proceed; prevents accidental destructive operations.",
    )

    sp_ensure = sub.add_parser(
        "ensure-doc-sim-table",
        help="Ensure the document similarity results table exists and optionally add missing columns.",
    )
    sp_ensure.add_argument(
        "--table",
        default="doc_similarity_results",
        help="Target table name (default: doc_similarity_results).",
    )
    sp_ensure.add_argument("--schema", default="public", help="Schema name (default: public).")
    sp_ensure.add_argument(
        "--add-missing-columns",
        action="store_true",
        help="When provided, ALTER TABLE to add any missing non-PK columns.",
    )

    return p


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv (Optional[List[str]]): Optional argument vector for testing.

    Returns:
        int: Process exit code (0 on success; non-zero on failure).
    """
    load_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log)

    try:
        cfg = resolve_pg_config(args)
    except Exception as e:
        LOGGER.error("Invalid connection options: %s", e)
        return 2

    if args.cmd == "ping":
        return cmd_ping(cfg)
    if args.cmd == "table-exists":
        return cmd_table_exists(cfg, table=args.table, schema=args.schema)
    if args.cmd == "describe":
        return cmd_describe(cfg, table=args.table, schema=args.schema)
    if args.cmd == "write-test":
        return cmd_write_test(cfg, confirm_side_effects=bool(args.confirm_side_effects), schema=args.schema)
    if args.cmd == "drop-table":
        return cmd_drop_table(cfg, table=args.table, schema=args.schema, confirm_side_effects=bool(args.confirm_side_effects))
    if args.cmd == "ensure-database":
        return cmd_ensure_database(cfg, target_db=(args.dbname or cfg.db))
    if args.cmd == "drop-database":
        return cmd_drop_database(cfg, target_db=args.dbname, confirm_side_effects=bool(args.confirm_side_effects))
    if args.cmd == "ensure-doc-sim-table":
        return cmd_ensure_doc_sim_table(cfg, table=args.table, schema=args.schema, add_missing_columns=bool(args.add_missing_columns))

    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
