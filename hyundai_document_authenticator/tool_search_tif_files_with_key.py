#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Key-driven TIF search and validation tool.

A production-oriented command-line utility that validates and leverages the
key-driven TIF helper (external/key_input) to:
- Stream filenames from a key table (CSV/Excel/JSON).
- Resolve filenames to actual TIF files via local folders or PostgreSQL.
- Apply mapping rules (e.g., insert token before tail_len with glob suffix).
- Report per-batch results and an overall summary.
- Optionally write a JSON manifest.
- Optionally trigger the existing pipeline via run_key_input_pipeline.

This tool does not modify pipeline configuration; it validates/leverages the
helper module to find and batch TIFs before running the standard pipeline.

Examples
- Dry-run (local mode), first 2 batches, showing 10 samples per batch:
  python search_tif_files_with_key.py \
      --key-config external/key_input/key_input_config.yaml \
      --limit-batches 2 --show-samples 10 --log-level INFO

- Validate against database (PostgreSQL):
  python search_tif_files_with_key.py \
      --key-config external/key_input/key_input_config.yaml \
      --mode database --db-host 127.0.0.1 --db-name postgres --db-user postgres \
      --limit-batches 1

- Write a manifest and trigger the pipeline:
  python search_tif_files_with_key.py \
      --key-config external/key_input/key_input_config.yaml \
      --manifest-out pipeline_inputs.json \
      --execute-now --main-config configs/image_similarity_config.yaml

Exit codes:
- 0: success
- 1: invalid arguments or missing files
- 2: runtime error during validation/pipeline
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import yaml

# ──────────────────────────────────────────────────────────────────────────────
# 📁 Imports and project path setup
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
EXTERNAL_DIR = PROJECT_ROOT / "external"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if EXTERNAL_DIR.is_dir() and str(EXTERNAL_DIR) not in sys.path:
    sys.path.append(str(EXTERNAL_DIR))

# External helper API (validated by Key_Driven_TIF_Validation.ipynb)
from external.key_input.key_input_orchestrator import (  # type: ignore
    KeyInputConfig,
    KeyInputLoader,
    LocalFetchConfig,
    LocalFolderFetcher,
    DatabaseConfig,
    DatabaseFetcher,
    NameMappingConfig,
    run_key_input_pipeline,
    _ensure_iterable_chunks,  # robust batching
)

# ─���────────────────────────────────────────────────────────────────────────────
# 🔧 Logging
# ──────────────────────────────────────────────────────────────────────────────
LOGGER = logging.getLogger("key_input_validator")


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure logging with centralized APP_LOG_DIR/tools routing.

    When ``log_file`` is not provided and the environment variable
    ``APP_LOG_DIR`` is set, a file handler is created at
    ``$APP_LOG_DIR/tools/tool_search_tif_files_with_key.log`` in addition
    to console logging. If ``APP_LOG_DIR`` is not set or file handler setup
    fails, logging falls back to console-only.

    Args:
        level: Logging level name (e.g., "DEBUG", "INFO").
        log_file: Optional explicit log file path.
    """
    LOGGER.handlers.clear()
    LOGGER.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Always attach console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOGGER.addHandler(sh)

    # Centralize into APP_LOG_DIR/tools if not explicitly provided
    try:
        target_path: Optional[Path] = None
        if log_file is not None:
            target_path = log_file
        else:
            env_log_dir = os.getenv("APP_LOG_DIR")
            if env_log_dir:
                tools_dir = Path(env_log_dir) / "tools"
                tools_dir.mkdir(parents=True, exist_ok=True)
                target_path = tools_dir / "tool_search_tif_files_with_key.log"
        if target_path is not None:
            fh = logging.FileHandler(str(target_path), encoding="utf-8")
            fh.setFormatter(fmt)
            LOGGER.addHandler(fh)
    except Exception as e:
        # File logging is optional; warn and continue.
        LOGGER.warning("File logger setup failed; continuing with console only: %s", e)


# ──────────────────────────────────────────────────────────────────────────────
# 🧰 Utilities
# ──────────────────────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file into a dictionary.

    Args:
        path (Path): Path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed YAML content or an empty dict on missing/empty file.

    Raises:
        OSError: If the file cannot be read due to I/O errors.
        yaml.YAMLError: If the YAML content is malformed.
    """
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _default_file_name_column(value: Optional[str]) -> str:
    """Return the default file name column if the provided value is empty.

    Args:
        value (Optional[str]): Candidate column name.

    Returns:
        str: A non-empty column name. Defaults to the commonly used Korean column name.
    """
    # This default helps align with notebooks already in use. It's safe to override via CLI.
    if value and isinstance(value, str) and value.strip():
        return value
    return "파일명"  # fallback alternative: "file_name"


def _redact_tail(value: str, keep: int = 3) -> str:
    """Redact all but the last N characters of a string.

    Args:
        value (str): Sensitive string (e.g., password) to redact.
        keep (int): Number of trailing characters to keep.

    Returns:
        str: Redacted string, or empty string if input is None.
    """
    if value is None:
        return ""
    s = str(value)
    if len(s) <= keep:
        return "*" * len(s)
    return "*" * (len(s) - keep) + s[-keep:]


# ──────────────────────────────────────────────────────────────────────────────
# 🧩 Config builders (map YAML + CLI overrides → dataclasses)
# ──────────────────────────────────────────��───────────────────────────────────

def build_key_input_config(cfg_dict: Dict[str, Any], args: argparse.Namespace) -> KeyInputConfig:
    """Construct a KeyInputConfig from YAML and CLI overrides.

    Args:
        cfg_dict (Dict[str, Any]): Parsed YAML dictionary.
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        KeyInputConfig: Configuration for key input behavior.
    """
    section = dict(cfg_dict.get("key_input", {}) or {})
    # CLI overrides
    if args.input_table:
        section["input_table_path"] = str(args.input_table)
    if args.file_name_column is not None:
        section["file_name_column"] = args.file_name_column
    if args.format:
        section["format"] = args.format
    if args.batch_size is not None:
        section["batch_size"] = args.batch_size
    if args.deduplicate is not None:
        section["deduplicate"] = bool(args.deduplicate)
    if args.strip_whitespace is not None:
        section["strip_whitespace"] = bool(args.strip_whitespace)
    if args.case_insensitive_match is not None:
        section["case_insensitive_match"] = bool(args.case_insensitive_match)

    return KeyInputConfig(
        input_table_path=Path(section.get("input_table_path")) if section.get("input_table_path") else Path(""),
        file_name_column=_default_file_name_column(section.get("file_name_column")),
        format=str(section.get("format", "auto")),
        json_array_field=section.get("json_array_field"),
        json_records_is_lines=bool(section.get("json_records_is_lines", False)),
        batch_size=int(section.get("batch_size", 200)),
        deduplicate=bool(section.get("deduplicate", True)),
        strip_whitespace=bool(section.get("strip_whitespace", True)),
    )


def build_name_mapping_config(cfg_dict: Dict[str, Any], args: argparse.Namespace) -> NameMappingConfig:
    """Construct a NameMappingConfig from YAML and CLI overrides.

    Args:
        cfg_dict (Dict[str, Any]): Parsed YAML dictionary.
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        NameMappingConfig: Configuration for filename transformation rules.
    """
    section = dict(cfg_dict.get("name_mapping", {}) or {})
    # CLI overrides
    if args.name_mapping_enabled is not None:
        section["enabled"] = bool(args.name_mapping_enabled)
    if args.tail_len is not None:
        section["tail_len"] = int(args.tail_len)
    if args.insert_token is not None:
        section["insert_token"] = args.insert_token
    if args.glob_suffix is not None:
        section["glob_suffix"] = args.glob_suffix
    if args.use_rglob_any_depth is not None:
        section["use_rglob_any_depth"] = bool(args.use_rglob_any_depth)
    if args.db_like_template is not None:
        section["db_like_template"] = args.db_like_template

    return NameMappingConfig(
        enabled=bool(section.get("enabled", True)),
        tail_len=int(section.get("tail_len", 5)),
        insert_token=str(section.get("insert_token", "001")),
        glob_suffix=str(section.get("glob_suffix", "_*.tif")),
        use_rglob_any_depth=bool(section.get("use_rglob_any_depth", True)),
        db_like_template=str(section.get("db_like_template", "{prefix}{insert}{suffix}_%.tif")),
    )


def build_local_fetch_config(cfg_dict: Dict[str, Any], args: argparse.Namespace) -> LocalFetchConfig:
    """Construct a LocalFetchConfig from YAML and CLI overrides.

    Args:
        cfg_dict (Dict[str, Any]): Parsed YAML dictionary.
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        LocalFetchConfig: Configuration for local filesystem fetching.
    """
    local = dict(cfg_dict.get("data_source", {}).get("local", {}) or {})
    # CLI overrides
    if args.root:
        local["search_roots"] = [str(p) for p in args.root]
    if args.recursive is not None:
        local["recursive"] = bool(args.recursive)
    if args.allowed_extensions:
        local["allowed_extensions"] = args.allowed_extensions
    if args.resolve_without_extension is not None:
        local["resolve_without_extension"] = bool(args.resolve_without_extension)
    if args.stop_on_first_match is not None:
        local["stop_on_first_match"] = bool(args.stop_on_first_match)

    return LocalFetchConfig(
        search_roots=[Path(p) for p in local.get("search_roots", [])],
        recursive=bool(local.get("recursive", True)),
        allowed_extensions=tuple(local.get("allowed_extensions", [".tif", ".tiff"])),
        resolve_without_extension=bool(local.get("resolve_without_extension", True)),
        case_insensitive_match=bool(cfg_dict.get("key_input", {}).get("case_insensitive_match", True)),
        stop_on_first_match=bool(local.get("stop_on_first_match", True)),
    )


def build_database_config(cfg_dict: Dict[str, Any], args: argparse.Namespace) -> DatabaseConfig:
    """Construct a DatabaseConfig from YAML and CLI overrides.

    Args:
        cfg_dict (Dict[str, Any]): Parsed YAML dictionary.
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        DatabaseConfig: Configuration for database lookups of TIF locations.
    """
    db = dict(cfg_dict.get("data_source", {}).get("database", {}) or {})
    # CLI overrides
    if args.db_host:
        db["host"] = args.db_host
    if args.db_port:
        db["port"] = args.db_port
    if args.db_name:
        db["database"] = args.db_name
    if args.db_user:
        db["user"] = args.db_user
    if args.db_password_env_var:
        db["password_env_var"] = args.db_password_env_var
    if args.sslmode:
        db["sslmode"] = args.sslmode
    if args.fetch_mode:
        db["fetch_mode"] = args.fetch_mode
    if args.query_template:
        db["query_template"] = args.query_template
    if args.path_column:
        db["path_column"] = args.path_column
    if args.blob_column:
        db["blob_column"] = args.blob_column
    if args.blob_temp_dir:
        db["blob_temp_dir"] = str(args.blob_temp_dir)

    return DatabaseConfig(
        driver=str(db.get("driver", "postgresql")),
        host=str(db.get("host", "127.0.0.1")),
        port=int(db.get("port", 5432)),
        database=str(db.get("database", "postgres")),
        user=str(db.get("user", "postgres")),
        password_env_var=str(db.get("password_env_var", "POSTGRES_PASSWORD")),
        sslmode=str(db.get("sslmode", "prefer")),
        fetch_mode=str(db.get("fetch_mode", "path")).lower(),
        query_template=str(db.get("query_template", "SELECT file_path FROM tif_documents WHERE file_name = %(file_name)s")),
        path_column=str(db.get("path_column", "file_path")),
        blob_column=str(db.get("blob_column", "file_blob")),
        blob_temp_dir=Path(db.get("blob_temp_dir")) if db.get("blob_temp_dir") else None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 🧭 CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI options and arguments.
    """
    p = argparse.ArgumentParser(
        description="Validate key-driven TIF filename mapping and batch resolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General
    gen = p.add_argument_group("General")
    gen.add_argument("--key-config", type=Path, default=Path("external/key_input/key_input_config.yaml"),
                    help="Path to key_input_config.yaml")
    gen.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    help="Logging verbosity level")
    gen.add_argument("--log-file", type=Path, help="Optional path to write a log file")
    gen.add_argument("--limit-batches", type=int, help="Limit number of batches for validation")
    gen.add_argument("--show-samples", type=int, default=5, help="Number of resolved samples to show per batch")
    gen.add_argument("--manifest-out", type=Path, help="Write a JSON manifest summarizing resolved files per batch")

    # Key input overrides
    key = p.add_argument_group("Key input overrides")
    key.add_argument("--input-table", type=Path, help="Override: path to CSV/Excel/JSON key file")
    key.add_argument("--file-name-column", type=str, help="Override: filename column name")
    key.add_argument("--format", type=str, choices=["auto", "csv", "excel", "json"], help="Override: key file format")
    key.add_argument("--batch-size", type=int, help="Override: batch size")
    key.add_argument("--deduplicate", type=int, choices=[0, 1], help="Override: deduplicate (1/0)")
    key.add_argument("--strip-whitespace", type=int, choices=[0, 1], help="Override: strip whitespace (1/0)")
    key.add_argument("--case-insensitive-match", type=int, choices=[0, 1], help="Override: case-insensitive matching (1/0)")

    # Mode
    mode = p.add_argument_group("Resolution mode")
    mode.add_argument("--mode", type=str, choices=["local", "database"], help="Override: data source mode for resolution")

    # Local mode overrides
    loc = p.add_argument_group("Local mode overrides")
    loc.add_argument("--root", type=Path, action="append", help="Local search root (repeatable)")
    loc.add_argument("--recursive", type=int, choices=[0, 1], help="Search recursively (1/0)")
    loc.add_argument("--allowed-extensions", nargs="+", help="Allowed extensions list, e.g. .tif .tiff")
    loc.add_argument("--resolve-without-extension", type=int, choices=[0, 1], help="Try .tif/.tiff if missing (1/0)")
    loc.add_argument("--stop-on-first-match", type=int, choices=[0, 1], help="Stop after first match (1/0)")

    # Database mode overrides
    db = p.add_argument_group("Database mode overrides")
    db.add_argument("--db-host", type=str, help="DB host")
    db.add_argument("--db-port", type=int, help="DB port")
    db.add_argument("--db-name", type=str, help="DB name")
    db.add_argument("--db-user", type=str, help="DB user")
    db.add_argument("--db-password-env-var", type=str, help="Env var for DB password")
    db.add_argument("--sslmode", type=str, help="DB sslmode, e.g. prefer, require")
    db.add_argument("--fetch-mode", type=str, choices=["path", "blob"], help="Fetch mode for DB results")
    db.add_argument("--query-template", type=str, help="SQL query template with %(file_name)s placeholder")
    db.add_argument("--path-column", type=str, help="Result column name for path mode")
    db.add_argument("--blob-column", type=str, help="Result column name for blob mode")
    db.add_argument("--blob-temp-dir", type=Path, help="Directory to write blobs (blob mode)")

    # Optional pipeline run
    run = p.add_argument_group("Pipeline run")
    run.add_argument("--execute-now", action="store_true", help="After validation, run run_key_input_pipeline")
    run.add_argument("--main-config", type=Path, default=Path("configs/image_similarity_config.yaml"),
                    help="Path to image_similarity_config.yaml for pipeline execution")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 🚀 Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    """Entry point for the key-driven TIF search and validation tool.

    Returns:
        int: Exit code. 0 on success, 1 for invalid args/missing files, 2 for runtime failures.

    Notes:
        The function logs a structured per-batch and overall summary and optionally
        writes a manifest JSON and triggers the pipeline. It avoids raising exceptions
        on non-critical issues to remain CLI-friendly.
    """
    args = parse_args()
    setup_logging(args.log_level, args.log_file)

    if not args.key_config or not args.key_config.exists():
        LOGGER.error("❌ key-config not found: %s", args.key_config)
        return 1

    cfg_dict = load_yaml(args.key_config)

    # Optional override of mode
    mode_cfg = cfg_dict.setdefault("data_source", {})
    if args.mode:
        mode_cfg["mode"] = args.mode
    mode = str(mode_cfg.get("mode", "local")).lower()

    # Build configs
    ki_cfg = build_key_input_config(cfg_dict, args)
    nm_cfg = build_name_mapping_config(cfg_dict, args)
    local_cfg = build_local_fetch_config(cfg_dict, args)
    db_cfg = build_database_config(cfg_dict, args)

    # Validate key file path early
    if not ki_cfg.input_table_path or not ki_cfg.input_table_path.exists():
        LOGGER.error("❌ input table not found: %s", ki_cfg.input_table_path)
        return 1

    # Build loader and fetcher
    loader = KeyInputLoader(ki_cfg, LOGGER)
    if mode == "local":
        LOGGER.info("🧰 Mode: local | roots=%s", [str(p) for p in local_cfg.search_roots])
        fetcher = LocalFolderFetcher(local_cfg, LOGGER, name_map=nm_cfg)
    elif mode == "database":
        pw_env = db_cfg.password_env_var or ""
        pw = os.environ.get(pw_env, "")
        # Redacting sensitive information in logs avoids accidental leakage.
        LOGGER.info(
            "🧰 Mode: database | host=%s db=%s user=%s fetch_mode=%s pw_env=%s(%s)",
            db_cfg.host, db_cfg.database, db_cfg.user, db_cfg.fetch_mode, pw_env, _redact_tail(pw)
        )
        fetcher = DatabaseFetcher(db_cfg, LOGGER, name_map=nm_cfg)
    else:
        LOGGER.error("❌ Unsupported mode: %s", mode)
        return 1

    # Iterate and validate
    total_requested = 0
    total_resolved = 0
    batches_done = 0
    manifest: List[Dict[str, Any]] = []

    start_ts = time.time()

    try:
        for batch in _ensure_iterable_chunks(loader.iter_filenames(), max(1, ki_cfg.batch_size)):
            batches_done += 1
            t0 = time.time()

            total_requested += len(batch)
            try:
                resolved_paths = fetcher.fetch_batch(batch)
            except Exception as e:
                # Robust logging here ensures operational clarity; we continue to keep other batches running.
                LOGGER.exception("⚠️ fetch_batch failed for batch %d: %s", batches_done, e)
                resolved_paths = []

            total_resolved += len(resolved_paths)
            dt = time.time() - t0

            LOGGER.info(
                "📦 Batch %d: requested=%d | resolved=%d | missing=%d | %.3fs",
                batches_done, len(batch), len(resolved_paths), len(batch) - len(resolved_paths), dt
            )

            # Show up to N samples
            for pth in resolved_paths[: max(0, int(args.show_samples or 0))]:
                LOGGER.info("  ✅ RESOLVED: %s", pth)

            batch_entry = {
                "batch_index": batches_done,
                "requested_count": len(batch),
                "resolved_count": len(resolved_paths),
                "resolved_paths": [str(p) for p in resolved_paths],
                "missing_count": len(batch) - len(resolved_paths),
                "duration_sec": round(dt, 3),
            }
            manifest.append(batch_entry)

            if args.limit_batches and batches_done >= args.limit_batches:
                LOGGER.info("⏹️ Batch limit reached (%d). Stopping validation loop.", args.limit_batches)
                break
    except KeyboardInterrupt:
        LOGGER.warning("⏹️ Interrupted by user. Summarizing partial results…")
    except Exception as e:
        LOGGER.exception("❌ Unexpected error during validation: %s", e)
        return 2

    total_dt = time.time() - start_ts
    LOGGER.info(
        "📊 Summary: batches=%d | requested=%d | resolved=%d | missing=%d | total %.3fs",
        batches_done, total_requested, total_resolved, total_requested - total_resolved, total_dt
    )

    if args.manifest_out:
        try:
            outp = args.manifest_out
            outp.parent.mkdir(parents=True, exist_ok=True)
            payload: Dict[str, Any] = {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "mode": mode,
                "config": {
                    "key_config_path": str(args.key_config),
                    "input_table_path": str(ki_cfg.input_table_path),
                    "file_name_column": ki_cfg.file_name_column,
                    "batch_size": ki_cfg.batch_size,
                    "name_mapping": {
                        "enabled": nm_cfg.enabled,
                        "tail_len": nm_cfg.tail_len,
                        "insert_token": nm_cfg.insert_token,
                        "glob_suffix": nm_cfg.glob_suffix,
                        "use_rglob_any_depth": nm_cfg.use_rglob_any_depth,
                    },
                    "local": {
                        "roots": [str(p) for p in local_cfg.search_roots],
                        "recursive": local_cfg.recursive,
                        "allowed_extensions": list(local_cfg.allowed_extensions),
                        "resolve_without_extension": local_cfg.resolve_without_extension,
                        "case_insensitive_match": getattr(local_cfg, "case_insensitive_match", True),
                        "stop_on_first_match": local_cfg.stop_on_first_match,
                    },
                    "database": {
                        "driver": db_cfg.driver,
                        "host": db_cfg.host,
                        "port": db_cfg.port,
                        "database": db_cfg.database,
                        "user": db_cfg.user,
                        "sslmode": db_cfg.sslmode,
                        "fetch_mode": db_cfg.fetch_mode,
                    },
                },
                "batches": manifest,
                "summary": {
                    "batches": batches_done,
                    "requested": total_requested,
                    "resolved": total_resolved,
                    "missing": total_requested - total_resolved,
                    "duration_sec": round(total_dt, 3),
                },
            }
            with outp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            LOGGER.info("📝 Wrote manifest to %s", outp)
        except Exception as e:
            LOGGER.exception("⚠️ Failed to write manifest: %s", e)
            return 2

    if args.execute_now:
        try:
            LOGGER.info("▶️ Executing pipeline via run_key_input_pipeline using %s", args.main_config)
            result = run_key_input_pipeline(args.main_config)
            LOGGER.info("🏁 Pipeline execution result:\n%s", json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as e:
            LOGGER.exception("❌ Pipeline execution failed: %s", e)
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
