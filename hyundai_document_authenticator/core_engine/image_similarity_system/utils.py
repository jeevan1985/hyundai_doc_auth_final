# core_engine/image_similarity_system/utils.py
"""Utility helpers for the image similarity system. ⚙️

This module centralizes general-purpose functionality used across the image
similarity system, including image metadata extraction, filesystem helpers,
PostgreSQL setup/checks, and persistence utilities. The implementation is kept
intentionally lightweight and dependency-free beyond the project's existing
requirements. No functional logic has been altered. ✨

The functions are documented with comprehensive Google-style docstrings to be
Sphinx-ready and fully type-annotated for better IDE support and static
analysis. Only non-obvious sections contain inline comments that capture the
intent (the "why").
"""

from __future__ import annotations

# =============================================================================
# 1. Standard Library Imports
# =============================================================================
import csv
import json
import logging
import os
import getpass
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, Callable

# =============================================================================
# 2. Third-Party Library Imports
# =============================================================================
import cv2
import numpy as np

# =============================================================================
# 3. Application-Specific Imports
# =============================================================================
from .constants import (
    ALLOWED_IMAGE_EXTENSIONS,
    JSON_SUMMARY_DEFAULT_FILENAME,
    QUERY_SAVE_SUBFOLDER_NAME,
    DEFAULT_REMOVE_COLUMNS_FROM_RESULTS,
)
from .removal_filter import effective_removal_set

# =============================================================================
# 4. Module-level Logger Setup
# =============================================================================
logger = logging.getLogger(__name__)


# =============================================================================
# 5. Utility Functions
# =============================================================================

def detect_requesting_username() -> str:
    """Resolve the current OS username in a cross-platform, defensive manner.

    The function attempts multiple strategies to obtain the username and then
    sanitizes the value for safe usage in file and directory names.

    Resolution order:
      1) getpass.getuser()
      2) Environment variables: LOGNAME, USER, LNAME, USERNAME

    The returned value is sanitized by replacing reserved path characters with
    underscores. When no username can be resolved or the sanitized result is
    empty, the function returns the fallback value "cli_user".

    Returns:
        str: A sanitized username suitable for audit columns and filesystem paths.
    """
    username: Optional[str]
    try:
        username = getpass.getuser()
    except Exception:
        username = None

    if not username:
        for env_var in ("LOGNAME", "USER", "LNAME", "USERNAME"):
            candidate = os.environ.get(env_var)
            if candidate:
                username = candidate
                break

    # Sanitize to ensure safe path segment usage across OSes
    if username:
        # Remove path separators and reserved characters
        sanitized = re.sub(r'[\\/:\"*?<>|]+', "_", username.strip())
        # Collapse repeated underscores and trim
        sanitized = re.sub(r"_+", "_", sanitized).strip("._ ")
    else:
        sanitized = ""

    return sanitized or "cli_user"


def get_image_metadata(image_path_str: str) -> Dict[str, Any]:
    """Collect image metadata such as dimensions and file size. 📐

    The function attempts to read the image via OpenCV to obtain dimensions.
    If the file is not readable or does not exist, safe defaults are returned.

    Args:
        image_path_str (str): Absolute or relative path to the image file.

    Returns:
        Dict[str, Any]: A mapping with the following keys:
            - dimensions_str (str): Width x Height, or "N/A" on failure.
            - size_bytes (int): File size in bytes, or -1 on failure.

    Raises:
        None: This function is intentionally defensive and will not raise on
            IO/decoding errors; instead, it logs and returns defaults. ⚠️
    """
    dimensions_str = "N/A"
    size_bytes = -1
    image_path = Path(image_path_str)

    if not image_path.is_file():
        logger.warning("get_image_metadata: File not found at %s", image_path_str)
        return {"dimensions_str": dimensions_str, "size_bytes": size_bytes}

    resolved_image_path = image_path.resolve()
    logger.debug(
        "get_image_metadata: Attempting to read image at resolved path: %s",
        resolved_image_path,
    )

    try:
        # Get image dimensions using OpenCV
        img = cv2.imread(str(resolved_image_path))
        if img is not None:
            height, width = img.shape[:2]
            dimensions_str = f"{width} x {height}"
            logger.debug(
                "get_image_metadata: Successfully read dimensions for %s: %s",
                resolved_image_path,
                dimensions_str,
            )
        else:
            logger.warning(
                "cv2.imread returned None for image: %s (Resolved path: %s)",
                image_path_str,
                resolved_image_path,
            )
    except Exception as e_dim:  # noqa: BLE001 - broad by design; logs and continues
        logger.warning(
            "Could not get dimensions for image %s (Resolved path: %s): %s",
            image_path_str,
            resolved_image_path,
            e_dim,
        )

    try:
        # Get file size in bytes
        size_bytes = image_path.stat().st_size
    except Exception as e_size:  # noqa: BLE001 - broad by design; logs and continues
        logger.warning("Could not get file size for image %s: %s", image_path_str, e_size)

    return {"dimensions_str": dimensions_str, "size_bytes": size_bytes}


def image_path_generator(folder: Path, scan_subfolders: bool = False) -> Generator[Path, None, None]:
    """Yield valid image file paths from a directory. 🖼️

    Iterates over the provided directory (optionally recursively) and yields
    paths whose extensions are in ``ALLOWED_IMAGE_EXTENSIONS``.

    Args:
        folder (Path): Directory to scan.
        scan_subfolders (bool): If True, scan recursively; otherwise only the
            top-level directory is scanned.

    Yields:
        Path: Paths of files that appear to be valid images for this system.

    Raises:
        None: Generator is defensive and will silently skip inaccessible files.
    """
    glob_pattern = "**/*" if scan_subfolders else "*"
    for item_path in folder.glob(glob_pattern):
        if item_path.is_file() and item_path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
            yield item_path


def ensure_postgresql_database_exists(
    database_name: str,
    host: str,
    port: Union[str, int],
    user: str,
    password: str,
    purpose: str = "general",
) -> bool:
    """Ensure that a PostgreSQL database exists, creating it if necessary. 🗄️

    This function is designed to be generic and can be used for any internal
    database (e.g., search results, user authentication).

    Args:
        database_name (str): Name of the database.
        host (str): Database host.
        port (Union[str, int]): Database port.
        user (str): Database user name.
        password (str): User password.
        purpose (str): Human-readable purpose for logging context.

    Returns:
        bool: True if the database exists or is created successfully; False
        otherwise (e.g., package missing or insufficient privileges).

    Raises:
        None: Errors are logged and converted to a boolean outcome. 🧯
    """
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    except ImportError:
        logger.warning(
            f"psycopg2 not available. Cannot check/create PostgreSQL database for {purpose}."
        )
        return False

    try:
        # First, try to connect to the target database to see if it exists
        test_conn = psycopg2.connect(
            dbname=database_name, user=user, password=password, host=host, port=port
        )
        test_conn.close()
        logger.info(
            "PostgreSQL database '%s' already exists for %s.", database_name, purpose
        )
        return True

    except psycopg2.OperationalError as e:
        if "does not exist" in str(e).lower():
            logger.info(
                "PostgreSQL database '%s' does not exist. Attempting to create it for %s...",
                database_name,
                purpose,
            )

            try:
                # Connect to a system database to create the target database
                try:
                    admin_conn = psycopg2.connect(
                        dbname="postgres",
                        user=user,
                        password=password,
                        host=host,
                        port=port,
                    )
                except psycopg2.OperationalError:
                    logger.warning(
                        "Could not connect to 'postgres' database to create new database. "
                        "Trying 'template1' as a fallback."
                    )
                    admin_conn = psycopg2.connect(
                        dbname="template1",
                        user=user,
                        password=password,
                        host=host,
                        port=port,
                    )
                admin_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

                with admin_conn.cursor() as cursor:
                    # Check if database already exists (race condition protection)
                    cursor.execute(
                        "SELECT 1 FROM pg_database WHERE datname = %s",
                        (database_name,),
                    )
                    if cursor.fetchone():
                        logger.info(
                            "Database '%s' was created by another process.", database_name
                        )
                        admin_conn.close()
                        return True

                    # Create the database
                    cursor.execute(f'CREATE DATABASE "{database_name}"')
                    logger.info(
                        "Successfully created PostgreSQL database '%s' for %s.",
                        database_name,
                        purpose,
                    )

                admin_conn.close()
                return True

            except psycopg2.Error as create_error:
                error_msg = str(create_error).lower()
                if "permission denied" in error_msg or "must be owner" in error_msg:
                    # 🔐 Privileges problem; provide actionable guidance.
                    logger.warning(
                        "Insufficient privileges to create database '%s' for %s.",
                        database_name,
                        purpose,
                    )
                    logger.warning(
                        "Please ask your database administrator to create the database, or"
                    )
                    logger.warning(
                        "grant CREATEDB privilege to user '%s'.", user
                    )
                    logger.warning(
                        "Manual command: CREATE DATABASE \"%s\";", database_name
                    )
                elif "already exists" in error_msg:
                    logger.info(
                        "Database '%s' already exists (created concurrently).",
                        database_name,
                    )
                    return True
                else:
                    logger.error(
                        "Failed to create database '%s' for %s: %s",
                        database_name,
                        purpose,
                        create_error,
                    )
                return False
        else:
            # Some other connection error (wrong credentials, server down, etc.)
            logger.error(
                "Cannot connect to PostgreSQL server for %s: %s", purpose, e
            )
            return False

    except Exception as e:  # noqa: BLE001 - defensive catch with logging
        logger.error(
            "Unexpected error while checking PostgreSQL database for %s: %s",
            purpose,
            e,
        )
        return False


def ensure_postgresql_database_exists_from_config(
    pg_config: Dict[str, Any],
    purpose: str = "general",
) -> bool:
    """Ensure a PostgreSQL database exists using a configuration mapping. 🧩

    Args:
        pg_config (Dict[str, Any]): Mapping that contains connection details.
            Expected keys: "database_name", "host", "port", "user", "password".
        purpose (str): Human-readable purpose for logging context.

    Returns:
        bool: True if the database exists or is created successfully; False
        otherwise.

    Raises:
        None
    """
    database_name = pg_config.get("database_name")
    host = pg_config.get("host")
    port = pg_config.get("port", 5432)
    user = pg_config.get("user")
    password = pg_config.get("password")

    if not all([database_name, host, user, password]):
        logger.warning(
            f"Incomplete PostgreSQL configuration for {purpose}. Cannot check/create database."
        )
        return False

    # Expand environment variables for robustness in containerized/runtime envs
    database_name = os.path.expandvars(str(database_name))
    host = os.path.expandvars(str(host))
    user = os.path.expandvars(str(user))
    password = os.path.expandvars(str(password))
    port = os.path.expandvars(str(port))

    return ensure_postgresql_database_exists(database_name, host, port, user, password, purpose)


def create_database_if_not_exists(
    db_name: str,
    db_user: str,
    db_password: str,
    db_host: str,
    db_port: Union[str, int],
) -> None:
    """Create a PostgreSQL database if it does not already exist. 🏗️

    Args:
        db_name (str): Target database name.
        db_user (str): Database user.
        db_password (str): Database password.
        db_host (str): Database host.
        db_port (Union[str, int]): Database port.

    Returns:
        None

    Raises:
        ImportError: If psycopg2 is not installed.
        psycopg2.Error: If an error occurs interacting with PostgreSQL.
    """
    try:
        import psycopg2
    except ImportError:
        logger.error(
            "psycopg2 is not installed. Please install it to use the PostgreSQL feature."
        )
        raise

    try:
        # Connect to the default 'postgres' database to check for the target database
        conn = psycopg2.connect(
            dbname="postgres",
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Check if the database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = cursor.fetchone()

        if not exists:
            # Create the database if it does not exist
            cursor.execute(f'CREATE DATABASE "{db_name}"')
            logger.info("Database '%s' created successfully.", db_name)
        else:
            logger.info("Database '%s' already exists.", db_name)

        cursor.close()
        conn.close()

    except psycopg2.Error as e:
        logger.error("Error interacting with PostgreSQL: %s", e)
        # Re-raise: upstream callers decide if the application can proceed.
        raise


def convert_numpy_to_native_python(data: Any) -> Any:
    """Recursively convert NumPy scalars/arrays to JSON-serializable Python types. 🔁

    This is useful before serialization (e.g., to JSON) to avoid "not serializable"
    errors when data structures embed NumPy-specific types.

    Args:
        data (Any): Arbitrary nested structure (lists, dicts, scalars).

    Returns:
        Any: Structure with NumPy types replaced by their Python equivalents.

    Raises:
        None
    """
    if isinstance(data, list):
        return [convert_numpy_to_native_python(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_numpy_to_native_python(value) for key, value in data.items()}
    elif isinstance(data, (np.float16, np.float32, np.float64)):
        return float(data)
    elif isinstance(
        data,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    return data


def save_similar_images_to_folder(
    search_results: List[Tuple[str, float]],
    output_folder: Union[str, Path],
    query_image_path: Union[str, Path],
    config_for_summary: Dict[str, Any],
    search_method_actually_used: str,
    model_name_used: str,
    total_search_time_seconds: float,
    json_filename: str = JSON_SUMMARY_DEFAULT_FILENAME,
    copier: Optional[Callable[[Union[str, Path], Union[str, Path]], Any]] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, str], Optional[Path], Optional[Dict[str, Any]], int]:
    """Persist search artifacts (optional images and a JSON summary) to disk. 📦

    The function copies the query image and the similar result images based on
    configuration flags, compiles a summary of the run, and optionally writes a
    JSON file to the output directory. It also attempts to save results to
    PostgreSQL if enabled in configuration. No behavior is changed. ✅

    Args:
        search_results (List[Tuple[str, float]]): List of pairs (path, score)
            representing similarity results.
        output_folder (Union[str, Path]): Directory where outputs will be saved.
        query_image_path (Union[str, Path]): Path to the original query image.
        config_for_summary (Dict[str, Any]): Full configuration for logging
            parameters and flags.
        search_method_actually_used (str): Search method used.
        model_name_used (str): Name of model used for embedding/comparison.
        total_search_time_seconds (float): Total time taken by the search.
        json_filename (str): Output JSON filename for the summary.
        copier (Optional[Callable[[Union[str, Path], Union[str, Path]], Any]]): Optional
            file-copy callable used to copy files. If None, a default copier is used
            that preserves file metadata to maintain backward-compatible behavior.
            Useful for injecting fakes in tests. ⚙️
        **kwargs (Any): Additional, optional parameters (e.g., feature_dimension_used).

    Returns:
        Tuple[Dict[str, str], Optional[Path], Optional[Dict[str, Any]], int]:
            - Mapping from original absolute image paths to copied file paths.
            - Path to the JSON summary file if written.
            - The in-memory JSON-serializable summary payload.
            - Count of result files that could not be copied (missing/failed).

    Raises:
        None: All errors are handled via logging, preserving original behavior.

        """
    output_folder_path = Path(output_folder)
    query_image_path_obj = Path(query_image_path)
    copied_image_paths_map: Dict[str, str] = {}
    missing_files_count = 0  # Tracks files we expected to copy but could not.

    search_task_conf = config_for_summary.get("search_task", {})

    # Evaluate save/copy decisions once to avoid repeated dictionary lookups.
    should_copy_query = search_task_conf.get("copy_query_image_to_output", True)
    should_copy_similar = search_task_conf.get("copy_similar_images_to_output", True)
    should_save_json = search_task_conf.get("save_search_summary_json", True)
    # Select copier implementation. Defaults to shutil.copy2 for metadata preservation and
    # backward-compatible behavior. Tests can inject a fake to avoid real IO. ✨
    copy_fn: Callable[[Union[str, Path], Union[str, Path]], Any] = copier or shutil.copy2
    
    # Only create output folder if we're actually saving something to it.
    if should_copy_query or should_copy_similar or should_save_json:
        output_folder_path.mkdir(parents=True, exist_ok=True)

    # --- Copy Query Image (conditional) ---
    if should_copy_query:
        query_dest_folder = output_folder_path
        if search_task_conf.get("save_query_in_separate_subfolder_if_copied", True):
            query_dest_folder = output_folder_path / QUERY_SAVE_SUBFOLDER_NAME

        query_dest_folder.mkdir(exist_ok=True)
        copied_query_path = query_dest_folder / query_image_path_obj.name
        copy_fn(query_image_path_obj, copied_query_path)
        copied_image_paths_map[str(query_image_path_obj.resolve())] = str(
            copied_query_path.resolve()
        )

    # --- Copy Similar Images (conditional) ---
    if should_copy_similar:
        for rank, (original_path_str, _) in enumerate(search_results, 1):
            try:
                original_path = Path(original_path_str)
                if not original_path.is_file():
                    logger.warning(
                        "Could not copy result file for rank %d: Source file not found at '%s'. Skipping this file.",
                        rank,
                        original_path,
                    )
                    missing_files_count += 1
                    continue
                ranked_filename = f"rank_{rank:03d}_{original_path.name}"
                dest_path = output_folder_path / ranked_filename
                copy_fn(original_path, dest_path)
                copied_image_paths_map[str(original_path.resolve())] = str(
                    dest_path.resolve()
                )
            except FileNotFoundError:
                logger.warning(
                    "Could not copy result file for rank %d: File not found at '%s'. It may have been moved or deleted. Skipping.",
                    rank,
                    original_path_str,
                )
                missing_files_count += 1
                continue
            except Exception as e:  # noqa: BLE001 - log and continue per original behavior
                logger.error(
                    "An unexpected error occurred while copying result file '%s': %s",
                    original_path_str,
                    e,
                    exc_info=False,
                )
                missing_files_count += 1
                continue
    else:
        logger.info(
            "Skipping copy of similar images to output folder as per configuration."
        )

    # --- Prepare Summary Data (always needed for PostgreSQL) ---
    summary_data: Dict[str, Any] = {
        "search_run_timestamp": datetime.now().isoformat(),
        "query_image_details": {
            "original_path": str(query_image_path_obj.resolve()),
            "copied_to": copied_image_paths_map.get(str(query_image_path_obj.resolve())),
            "metadata": get_image_metadata(str(query_image_path_obj)),
        },
        "search_parameters_overview": {
            "top_k_requested": search_task_conf.get("top_k"),
            "search_method_used": search_method_actually_used,
            "model_name_used": model_name_used,
            "feature_dimension_used": kwargs.get("feature_dimension_used"),
        },
        "performance_metrics": {
            "total_search_operation_time_seconds": total_search_time_seconds
        },
        "results_summary": {
            "number_of_similar_images_found": len(search_results),
            "search_statistics": {"missing_result_files": missing_files_count},
        },
        "results": [],
    }
    for rank, (original_path, score) in enumerate(search_results, 1):
        summary_data["results"].append(
            {
                "rank": rank,
                "original_path": str(Path(original_path).resolve()),
                "copied_to": copied_image_paths_map.get(str(Path(original_path).resolve())),
                "similarity_score": score,
                "metadata": get_image_metadata(original_path),
            }
        )

    serializable_summary = convert_numpy_to_native_python(summary_data)
    json_summary_file_path: Optional[Path] = None

    # --- Save JSON Summary (conditional) ---
    if should_save_json:
        json_summary_file_path = output_folder_path / json_filename
        with open(json_summary_file_path, "w", encoding="utf-8") as f:
            json.dump(serializable_summary, f, indent=4)

    # --- Save Results to PostgreSQL (conditional, image-only flow) ---
    if search_task_conf.get("save_results_to_postgresql", False):
        pg_config = config_for_summary.get("results_postgresql") or {}
        try:
            save_image_search_results_to_postgresql(serializable_summary, pg_config)
        except Exception as e:
            logger.error(
                "Failed to save image-only search results to PostgreSQL: %s",
                e,
                exc_info=True,
            )

    return copied_image_paths_map, json_summary_file_path, serializable_summary, missing_files_count


def save_tif_per_query_results_to_postgresql(
    per_query_results: List[Dict[str, Any]],
    pg_config: Dict[str, Any],
    run_identifier: str,
    requesting_username: str,
    *,
    per_query_sim_checks: Optional[Dict[str, Any]] = None,
    per_query_timestamps: Optional[Dict[str, str]] = None,
    global_top_docs: Optional[List[str]] = None,
    extra_columns: Optional[List[str]] = None,
    per_query_extra_values: Optional[Dict[str, Dict[str, Any]]] = None,
    remove_columns_from_results: Optional[List[str]] = None,
) -> None:
    """Persist per-query TIF results to PostgreSQL (row-per-query), mirroring CSV.

    This inserts one row per query_document with the exact shape used by
    write_tif_per_query_results_csv, including dynamic column suppression via
    remove_columns_from_results and optional enrichment columns.

    Columns (before removal filtering):
      - run_identifier (TEXT)
      - requesting_username (TEXT)
      - search_time_stamp (TIMESTAMPTZ)
      - query_document (TEXT)
      - matched_query_document (TEXT)
      - top_similar_docs (JSONB)               # {"<doc>": score, ...}
      - threshold_match_count (INTEGER)
      - sim_img_check (JSONB)
      - image_authenticity (JSONB)
      - fraud_doc_probability (TEXT)
      - global_top_docs (JSONB)
      - [extra_columns...] (TEXT)

    Table resolution:
      - Uses results_postgresql.table_name_tif_per_query when provided; else
        results_postgresql.table_name; else defaults to 'doc_similarity_per_query'.
      - Environment variable POSTGRES_TABLE_NAME overrides the table name if set.

    Raises:
      ImportError: If psycopg2 is not installed.
      ValueError: If the configuration is incomplete.
      psycopg2.Error: For database-level issues.
    """
    try:
        import psycopg2  # type: ignore
        from psycopg2 import sql  # type: ignore
        from psycopg2.extras import Json, execute_batch  # type: ignore
    except ImportError:
        logger.error(
            "psycopg2 is not installed. Please install it to save per-query TIF results to PostgreSQL."
        )
        raise

    if not isinstance(pg_config, dict):
        raise ValueError("Invalid 'pg_config' provided for PostgreSQL save operation.")

    # Resolve connection credentials (env overrides YAML values)
    database_name = os.getenv("POSTGRES_DB", pg_config.get("database_name"))
    host = os.getenv("POSTGRES_HOST", pg_config.get("host"))
    port = os.getenv("POSTGRES_PORT", pg_config.get("port"))
    user = os.getenv("POSTGRES_USER", pg_config.get("user"))
    password = os.getenv("POSTGRES_PASSWORD", pg_config.get("password"))

    # Resolve per-query table name with environment override
    env_table = os.getenv("POSTGRES_TABLE_NAME")
    preferred_table = (
        pg_config.get("table_name_tif_per_query")
        or pg_config.get("table_name")
        or "doc_similarity_per_query"
    )
    table_name = env_table or preferred_table

    if not all([database_name, host, port, user, password, table_name]):
        raise ValueError(
            "Incomplete PostgreSQL configuration for saving per-query TIF results."
        )

    # Normalize removal list into a set consistent with CSV logic
    removal_set = effective_removal_set(remove_columns_from_results or [], DEFAULT_REMOVE_COLUMNS_FROM_RESULTS)

    # Filter extra columns by removal list
    extra_columns = extra_columns or []
    filtered_extra_columns = [c for c in extra_columns if c not in removal_set]

    # Ensure DB exists or create it when permitted
    try:
        create_database_if_not_exists(database_name, user, password, host, port)  # type: ignore[arg-type]
    except Exception as e_db:
        logger.warning("Could not create or verify database '%s': %s", database_name, e_db)

    # Table DDL creation (only when missing)
    def _create_table_if_needed(cur: Any, tbl: str) -> None:
        """Create per-query results table with dynamic columns if it does not exist.

        Args:
            cur: psycopg2 cursor.
            tbl: Target table name.
        """
        base_columns: List[sql.SQL] = [
            sql.SQL("id SERIAL PRIMARY KEY"),
            sql.SQL("run_identifier TEXT"),
            sql.SQL("requesting_username TEXT"),
            sql.SQL("search_time_stamp TIMESTAMPTZ"),
            sql.SQL("query_document TEXT"),
            sql.SQL("matched_query_document TEXT"),
        ]
        # Optional columns suppressed by removal_set
        optional_specs: List[tuple[str, str]] = [
            ("top_similar_docs", "JSONB"),
            ("threshold_match_count", "INTEGER"),
            ("sim_img_check", "JSONB"),
            ("image_authenticity", "JSONB"),
            ("fraud_doc_probability", "TEXT"),
            ("global_top_docs", "JSONB"),
        ]
        for col_name, col_type in optional_specs:
            if col_name not in removal_set:
                base_columns.append(sql.SQL(f"{col_name} {col_type}"))
        # Add enrichment columns as TEXT (safe default)
        for col in filtered_extra_columns:
            base_columns.append(sql.SQL(f"{col} TEXT"))
        ddl = sql.SQL("CREATE TABLE {tbl} (\n    {cols}\n)").format(
            tbl=sql.Identifier(tbl),
            cols=sql.SQL(",\n    ").join(base_columns),
        )
        cur.execute(ddl)
        logger.info("Created PostgreSQL table '%s' for per-query TIF results.", tbl)

    # Prepare records from per_query_results mirroring CSV writer logic
    per_query_extra_values = per_query_extra_values or {}
    per_query_sim_checks = per_query_sim_checks or {}
    per_query_timestamps = per_query_timestamps or {}

    records: List[Dict[str, Any]] = []
    for entry in per_query_results or []:
        qname = str(entry.get("query_document") or "").strip()
        matched = entry.get("matched_query_document") or qname
        top_docs = entry.get("top_docs") or []
        # Build compact mapping dict for CSV/DB consistency: {"<doc>": score, ...}
        mapped: Dict[str, Any] = {}
        for td in top_docs:
            try:
                d = td.get("document"); s = td.get("score")
                if d is not None:
                    mapped[str(d)] = s
            except Exception:
                continue
        row_map: Dict[str, Any] = {
            "run_identifier": run_identifier,
            "requesting_username": requesting_username,
            "search_time_stamp": per_query_timestamps.get(qname, datetime.now().isoformat()),
            "query_document": qname,
            "matched_query_document": matched,
        }
        if "top_similar_docs" not in removal_set:
            row_map["top_similar_docs"] = mapped
        if "threshold_match_count" not in removal_set:
            row_map["threshold_match_count"] = int(entry.get("threshold_match_count") or 0)
        if "sim_img_check" not in removal_set:
            row_map["sim_img_check"] = per_query_sim_checks.get(qname, {})
        if "image_authenticity" not in removal_set:
            row_map["image_authenticity"] = entry.get("image_authenticity") or {}
        if "fraud_doc_probability" not in removal_set:
            row_map["fraud_doc_probability"] = entry.get("fraud_doc_probability") or "No"
        if "global_top_docs" not in removal_set:
            row_map["global_top_docs"] = global_top_docs or []
        # Add enrichment values
        extras = per_query_extra_values.get(qname, {})
        for col in filtered_extra_columns:
            row_map[col] = extras.get(col)
        records.append(row_map)

    try:
        import psycopg2  # type: ignore  # re-import for type checkers
        with psycopg2.connect(
            dbname=database_name, user=user, password=password, host=host, port=port
        ) as conn:
            with conn.cursor() as cursor:
                # Introspect existing table columns
                cursor.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (table_name,),
                )
                existing_cols = [r[0] for r in cursor.fetchall()]
                if not existing_cols:
                    _create_table_if_needed(cursor, table_name)  # type: ignore[arg-type]
                    cursor.execute(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = %s
                        ORDER BY ordinal_position
                        """,
                        (table_name,),
                    )
                    existing_cols = [r[0] for r in cursor.fetchall()]

                existing_cols_set = set(existing_cols)

                # Compose insert columns: base + optional + enrichment
                base_headers = [
                    "run_identifier",
                    "requesting_username",
                    "search_time_stamp",
                    "query_document",
                    "matched_query_document",
                ]
                optional_headers = [
                    "top_similar_docs",
                    "threshold_match_count",
                    "sim_img_check",
                    "image_authenticity",
                    "fraud_doc_probability",
                    "global_top_docs",
                ]
                headers = [h for h in base_headers if h in existing_cols_set]
                headers += [h for h in optional_headers if h in existing_cols_set]
                headers += [c for c in filtered_extra_columns if c in existing_cols_set]

                # Warn when some requested optional columns are missing from existing table
                missing_requested = [h for h in optional_headers if h not in existing_cols_set and h not in removal_set]
                if missing_requested:
                    logger.warning(
                        "Per-query table is missing some requested columns (%s). Existing columns will be populated only.",
                        ", ".join(missing_requested),
                    )

                # Build parameterized INSERT and execute batch
                col_identifiers = [sql.Identifier(h) for h in headers]
                placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in headers)
                insert_query = sql.SQL(
                    "INSERT INTO {table} ( {cols} ) VALUES ( {vals} )"
                ).format(table=sql.Identifier(table_name), cols=sql.SQL(", ").join(col_identifiers), vals=placeholders)

                def _serialize_row(row: Dict[str, Any]) -> Tuple[Any, ...]:
                    """Serialize a row dict into a tuple aligned with headers for INSERT.

                    Args:
                        row: Mapping from column name to value.

                    Returns:
                        Tuple[Any, ...]: Values in the same order as headers.
                    """
                    values: List[Any] = []
                    for h in headers:
                        v = row.get(h)
                        if h in ("top_similar_docs", "sim_img_check", "image_authenticity", "global_top_docs"):
                            values.append(Json(v) if v is not None else None)
                        else:
                            values.append(v)
                    return tuple(values)

                batch_values = [_serialize_row(r) for r in records]
                if batch_values:
                    execute_batch(cursor, insert_query, batch_values)
                    logger.info(
                        "Inserted %d per-query TIF result row(s) into PostgreSQL table '%s'",
                        len(batch_values),
                        table_name,
                    )
                conn.commit()
    except psycopg2.Error as e:  # type: ignore[assignment]
        logger.error(
            "A PostgreSQL error occurred during per-query TIF save operation: %s",
            e,
            exc_info=True,
        )
        raise


def save_image_search_results_to_postgresql(summary_data: Dict[str, Any], pg_config: Dict[str, Any]) -> None:
    """Persist image-only search results into a dedicated PostgreSQL table.

    This function is intentionally isolated from TIF document similarity save
    logic. It writes to a separate table designed for image-only queries and
    does not interfere with any TIF-related tables or workflows.

    Table management:
      - Creates the table if it does not exist.
      - Uses the table name from ``results_postgresql.table_name_img_sim``;
        defaults to "image_similarity_results" when not provided.

    The expected schema is:
      id SERIAL PRIMARY KEY,
      search_run_timestamp TIMESTAMPTZ,
      query_image_path TEXT,
      top_k_requested INT,
      search_method_used TEXT,
      model_name_used TEXT,
      feature_dimension_used INT,
      total_search_time_seconds FLOAT,
      results JSONB

    Args:
        summary_data (Dict[str, Any]): JSON-serializable summary payload built
            by ``save_similar_images_to_folder`` containing keys such as
            ``search_run_timestamp``, ``query_image_details``,
            ``search_parameters_overview``, ``performance_metrics``, and
            ``results``.
        pg_config (Dict[str, Any]): PostgreSQL configuration mapping under the
            ``results_postgresql`` section. Expected keys for connection:
            ``database_name``, ``host``, ``port``, ``user``, ``password``.
            Optional key ``table_name_img_sim`` for the target table name.

    Returns:
        None

    Raises:
        None: Any underlying database errors are logged; the caller is expected
        to handle operational decisions. Import errors for psycopg2 are handled
        gracefully with logging and a no-op return.
    """
    # Import lazily to keep the module importable in environments without psycopg2
    try:
        import psycopg2  # type: ignore
        from psycopg2 import sql  # type: ignore
        from psycopg2.extras import Json  # type: ignore
    except ImportError:
        logger.warning(
            "psycopg2 is not installed. Skipping PostgreSQL save for image-only results."
        )
        return

    # Resolve connection parameters. Allow env overrides for credentials but not for table name
    # to maintain strict separation from document similarity workflows.
    database_name = os.getenv("POSTGRES_DB", pg_config.get("database_name"))
    host = os.getenv("POSTGRES_HOST", pg_config.get("host"))
    port = os.getenv("POSTGRES_PORT", pg_config.get("port"))
    user = os.getenv("POSTGRES_USER", pg_config.get("user"))
    password = os.getenv("POSTGRES_PASSWORD", pg_config.get("password"))

    table_name: str = (
        (pg_config.get("table_name_img_sim") or "image_similarity_results")
        if isinstance(pg_config, dict)
        else "image_similarity_results"
    )

    # Validate configuration; fail gracefully if incomplete
    if not all([database_name, host, port, user, password, table_name]):
        logger.warning(
            "Incomplete PostgreSQL configuration for image-only results; skipping save."
        )
        return

    # Ensure DB exists before connecting; do not abort if creation fails
    try:
        create_database_if_not_exists(database_name, user, password, host, port)  # type: ignore[arg-type]
    except Exception as e_db_create:  # noqa: BLE001
        logger.warning(
            "Database existence check/creation failed for '%s': %s. Proceeding to connect...",
            database_name,
            e_db_create,
        )

    # Extract and normalize fields from summary_data
    try:
        search_ts = summary_data.get("search_run_timestamp")
        q_details = summary_data.get("query_image_details", {}) or {}
        q_path = q_details.get("original_path")
        params = summary_data.get("search_parameters_overview", {}) or {}
        perf = summary_data.get("performance_metrics", {}) or {}
        results_payload = summary_data.get("results", [])

        top_k = params.get("top_k_requested")
        method_used = params.get("search_method_used")
        model_used = params.get("model_name_used")
        feat_dim = params.get("feature_dimension_used")
        elapsed = perf.get("total_search_operation_time_seconds")

        # Connect and ensure the target table exists
        with psycopg2.connect(
            dbname=database_name, user=user, password=password, host=host, port=port
        ) as conn:
            with conn.cursor() as cur:
                # Create table if it does not exist
                ddl = sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {tbl} (
                        id SERIAL PRIMARY KEY,
                        search_run_timestamp TIMESTAMPTZ,
                        query_image_path TEXT,
                        top_k_requested INT,
                        search_method_used TEXT,
                        model_name_used TEXT,
                        feature_dimension_used INT,
                        total_search_time_seconds FLOAT,
                        results JSONB
                    )
                    """
                ).format(tbl=sql.Identifier(table_name))
                cur.execute(ddl)

                # Prepare INSERT statement
                insert_query = sql.SQL(
                    "INSERT INTO {tbl} (search_run_timestamp, query_image_path, top_k_requested, "
                    "search_method_used, model_name_used, feature_dimension_used, total_search_time_seconds, results) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                ).format(tbl=sql.Identifier(table_name))

                cur.execute(
                    insert_query,
                    (
                        search_ts,
                        q_path,
                        int(top_k) if top_k is not None else None,
                        method_used,
                        model_used,
                        int(feat_dim) if feat_dim is not None else None,
                        float(elapsed) if elapsed is not None else None,
                        Json(results_payload),
                    ),
                )
                conn.commit()
                logger.info(
                    "Saved image-only search results into PostgreSQL table '%s'",
                    table_name,
                )
    except Exception as e:  # noqa: BLE001
        logger.error("Failed during image-only PostgreSQL save operation: %s", e, exc_info=True)
        # Do not re-raise to preserve non-breaking behavior
        return


def maintain_log_files(
    log_dir: Union[str, Path],
    *,
    stem: Optional[str] = None,
    remove_logs_days: int = 7,
    backup_logs: bool = False,
) -> int:
    """Clean up log files in a directory with optional backup.

    This utility removes log files older than a specified number of days. When
    ``backup_logs`` is True, each file selected for removal is first copied to
    a sibling path with the ``.bk`` suffix appended to the original filename
    (e.g., ``simil.log`` -> ``simil.log.bk``) before the original file is
    deleted. The function is defensive and will not raise on I/O errors; it
    logs issues and continues.

    Args:
        log_dir (Union[str, Path]): Directory where log files reside.
        stem (Optional[str]): Optional filename stem to filter by. When
            provided, only files whose names start with this stem are
            considered (pattern ``f"{stem}*"``). When None, the default
            pattern ``"*.log"`` is used.
        remove_logs_days (int): Files strictly older than this number of days
            are removed. Defaults to 7.
        backup_logs (bool): When True, copy each selected file to a ``.bk``
            path before deletion. Defaults to False.

    Returns:
        int: Number of files removed.

    Raises:
        None
    """
    removed_count: int = 0
    try:
        base_dir = Path(log_dir).resolve()
        if not base_dir.exists() or not base_dir.is_dir():
            return 0

        # Determine matching pattern
        pattern = f"{stem}*" if stem else "*.log"
        cutoff_ts = time.time() - max(0, int(remove_logs_days)) * 24 * 60 * 60

        for p in base_dir.glob(pattern):
            try:
                if not p.is_file():
                    continue
                # Skip already-backed-up files to avoid cascading backups
                if p.name.endswith(".bk"):
                    continue
                # Skip if not older than the cutoff
                if p.stat().st_mtime >= cutoff_ts:
                    continue

                # Backup (copy) before removal if requested
                if backup_logs:
                    backup_path = Path(str(p) + ".bk")
                    try:
                        # Overwrite any existing backup to avoid accumulation
                        if backup_path.exists():
                            try:
                                backup_path.unlink()
                            except Exception:
                                pass
                        shutil.copy2(p, backup_path)
                    except Exception as e_bk:  # noqa: BLE001
                        logger.warning("Log backup failed for %s: %s", p, e_bk)

                # Remove original file
                try:
                    p.unlink()
                    removed_count += 1
                    logger.info("Removed old log file: %s", p)
                except Exception as e_rm:  # noqa: BLE001
                    logger.error("Failed to remove old log %s: %s", p, e_rm)
            except Exception as e:  # noqa: BLE001
                logger.error("Error while processing log candidate %s: %s", p, e)
        return removed_count
    except Exception as e:  # noqa: BLE001
        logger.error("Unexpected error during log maintenance in '%s': %s", log_dir, e)
        return removed_count


def export_table_rows_to_csv(
    conn: Any,
    table_name: str,
    output_csv_path: Union[str, Path],
    run_identifier: Optional[str] = None,
) -> None:
    """Export rows from a PostgreSQL table into a CSV file. 📤

    Args:
        conn (Any): Open psycopg2 connection.
        table_name (str): Source table.
        output_csv_path (Union[str, Path]): Destination CSV path.
        run_identifier (Optional[str]): If provided, filters rows by run_identifier;
            otherwise, exports the most recent 100 rows.

    Returns:
        None

    Raises:
        ImportError: If psycopg2 is not installed.
    """
    try:
        import psycopg2  # noqa: F401
    except ImportError:
        logger.error("psycopg2 is not installed. Cannot export CSV.")
        raise

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with conn.cursor() as cursor:
        if run_identifier:
            cursor.execute(
                f"SELECT * FROM {table_name} WHERE run_identifier = %s ORDER BY id ASC",
                (run_identifier,),
            )
        else:
            logger.warning(
                "No run_identifier provided for CSV export. Exporting last 100 rows as fallback."
            )
            cursor.execute(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 100")
        rows = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(colnames)
        for row in rows:
            serialized: List[Any] = []
            for val in row:
                if isinstance(val, (dict, list)):
                    serialized.append(json.dumps(val, ensure_ascii=False))
                else:
                    serialized.append(val)
            writer.writerow(serialized)
