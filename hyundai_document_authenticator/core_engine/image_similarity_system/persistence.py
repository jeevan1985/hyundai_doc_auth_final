"""
Persistence utilities for image similarity results.

Responsibilities:
- Resolve table names with environment overrides
- Write CSV debug exports for both image similarity and TIF document aggregation
- Thin wrappers around database save operations
"""
from __future__ import annotations

import csv
import json
from .serialization import json_dumps
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .removal_filter import filter_csv_headers, effective_removal_set
from .serialization import csv_safe
from .constants import DEFAULT_REMOVE_COLUMNS_FROM_RESULTS
from .utils import save_tif_per_query_results_to_postgresql

logger = logging.getLogger(__name__)

# ----------------------------
# Table-name resolution helpers
# ----------------------------

def resolve_table_name(pg_config: Dict[str, Any], mode: str) -> str:
    """Resolve the target table name for a given results mode.

    mode:
      - 'img_sim'  -> results_postgresql.table_name_img_sim | table_name | 'image_similarity_search_results'  [legacy]
      - 'tif_doc'  -> results_postgresql.table_name_doc_sim | table_name | 'doc_similarity_results'
      Both are overridden by POSTGRES_TABLE_NAME if set.
    """
    env_table = os.getenv("POSTGRES_TABLE_NAME")
    if env_table:
        return env_table

    if mode == "tif_doc":
        return (
            pg_config.get("table_name_doc_sim")
            or pg_config.get("table_name")
            or "doc_similarity_results"
        )
    else:
        raise ValueError(f"Unsupported mode for table resolution: {mode}")


# ----------------------------
# CSV debug export writers
# ----------------------------



def write_tif_doc_results_csv(
    final_top_documents: List[Dict[str, Any]],
    sim_img_checks: Dict[str, Any],
    global_payload: Dict[str, Any],
    export_dir: Union[str, Path],
    run_id: str,
    user: str,
    table_name: str,
    extra_columns: Optional[List[str]] = None,
    per_doc_extra_values: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Path:
    """Write TIF doc aggregation results to CSV compatible with DB schema, with optional extra columns."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / f"postgres_export_{table_name}_{run_id}.csv"

    headers = [
        "run_identifier",
        "requesting_username",
        "search_timestamp",
        "parent_document_name",
        "highest_similarity_score",
        "sim_img_check",
    ]

    extra_columns = extra_columns or []
    per_doc_extra_values = per_doc_extra_values or {}
    headers.extend(extra_columns)

    from datetime import datetime

    with open(export_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(headers)
        for item in final_top_documents:
            base_row = [
                run_id,
                user,
                datetime.now().isoformat(),
                item["document"],
                item["score"],
                json_dumps(sim_img_checks.get(item["document"], {"note": "not available"})),
            ]
            # Append extra column values in the configured order
            extra_vals_for_doc = per_doc_extra_values.get(item["document"], {})
            for col in extra_columns:
                base_row.append(extra_vals_for_doc.get(col))
            writer.writerow(base_row)

    logger.info("CSV export (tif doc) written: %s", export_path)
    return export_path


def write_tif_per_query_results_csv(
    per_query_results: List[Dict[str, Any]],
    export_dir: Union[str, Path],
    run_id: str,
    user: str,
    extra_columns: Optional[List[str]] = None,
    per_query_extra_values: Optional[Dict[str, Dict[str, Any]]] = None,
    per_query_sim_checks: Optional[Dict[str, Any]] = None,
    per_query_timestamps: Optional[Dict[str, str]] = None,
    filename_stem: str = "postgres_export_tif_per_query",
    global_top_docs: Optional[List[str]] = None,
    remove_columns: Optional[List[str]] = None,
) -> Path:
    """Write per-query TIF results to CSV with a top_similar_docs column (JSON), plus optional extra columns.

    Columns (before filtering):
    - run_identifier
    - requesting_username
    - search_time_stamp
    - query_document
    - matched_query_document
    - top_similar_docs (JSON object mapping "<doc>": score)
    - threshold_match_count (integer; count of docs in top_similar_docs with score >= SIMILAR_DOC_FLAG_THRESHOLD)
    - sim_img_check (JSON)
    - image_authenticity (JSON)
    - fraud_doc_probability
    - global_top_docs (JSON array)
    - [extra_columns...]

    The 'remove_columns' list can hide any of these by header name. Extra columns
    (from key enrichment) are also filtered by matching names.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / f"{filename_stem}_{run_id}.csv"

    extra_columns = extra_columns or []
    per_query_extra_values = per_query_extra_values or {}
    removal_set = effective_removal_set(remove_columns or [], DEFAULT_REMOVE_COLUMNS_FROM_RESULTS)

    base_headers = [
        "run_identifier",
        "requesting_username",
        "search_time_stamp",
        "query_document",
        "matched_query_document",
        "top_similar_docs",
        "threshold_match_count",
        "sim_img_check",
        "image_authenticity",
        "fraud_doc_probability",
        "global_top_docs",
    ]

    # Filter extra columns by removal list and compute final headers
    filtered_extra_columns = [c for c in extra_columns if c not in removal_set]
    headers = filter_csv_headers(base_headers, filtered_extra_columns, removal_set)

    with open(export_path, "w", newline="", encoding="utf-8-sig") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(headers)
        for entry in per_query_results:
            qname = entry.get("query_document")
            top_docs = entry.get("top_docs") or []
            # Transform to {"<doc>": score, ...} for a concise map-style object
            mapped: Dict[str, Any] = {}
            for td in top_docs:
                try:
                    d = td.get("document")
                    s = td.get("score")
                    if d is not None:
                        mapped[str(d)] = s
                except Exception:
                    continue
            row_map: Dict[str, Any] = {
                "run_identifier": run_id,
                "requesting_username": user,
                "search_time_stamp": (per_query_timestamps or {}).get(qname, "") if per_query_timestamps else "",
                "query_document": qname,
                "matched_query_document": entry.get("matched_query_document") or entry.get("matched__query_document") or qname,
                "top_similar_docs": mapped,
                "threshold_match_count": int(entry.get("threshold_match_count") or 0),
                "sim_img_check": (per_query_sim_checks or {}).get(qname) if per_query_sim_checks else {},
                "image_authenticity": entry.get("image_authenticity") or {},
                "fraud_doc_probability": entry.get("fraud_doc_probability") or "No",
                "global_top_docs": (global_top_docs or []),
            }
            # Prepare extra column values
            extras = per_query_extra_values.get(qname, {})
            for col in filtered_extra_columns:
                row_map[col] = extras.get(col)

            # Serialize cells consistently and write in header order
            serialized_row = []
            for h in headers:
                serialized_row.append(csv_safe(row_map.get(h)))
            writer.writerow(serialized_row)

    logger.info("CSV export (tif per-query) written: %s", export_path)
    return export_path

# ----------------------------
# Database write wrappers
# ----------------------------
def try_save_tif_per_query_to_db(
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
    """Save per-query TIF results to PostgreSQL (row-per-query), mirroring CSV.

    Args:
        per_query_results (List[Dict[str, Any]]): Per-query entries from the workflow.
        pg_config (Dict[str, Any]): PostgreSQL configuration mapping.
        run_identifier (str): Unique run ID.
        requesting_username (str): The initiating user ID/name.
        per_query_sim_checks (Optional[Dict[str, Any]]): Optional explainability map per query.
        per_query_timestamps (Optional[Dict[str, str]]): Map of query_document -> ISO timestamp.
        global_top_docs (Optional[List[str]]): Optional list of global top docs to attach per row.
        extra_columns (Optional[List[str]]): Enrichment column names to add as extra fields.
        per_query_extra_values (Optional[Dict[str, Dict[str, Any]]]): Map of query_document -> enrichment values.
        remove_columns_from_results (Optional[List[str]]): Columns to suppress. Mirrors CSV removal logic.

    Returns:
        None
    """
    save_tif_per_query_results_to_postgresql(
        per_query_results=per_query_results,
        pg_config=pg_config,
        run_identifier=run_identifier,
        requesting_username=requesting_username,
        per_query_sim_checks=per_query_sim_checks,
        per_query_timestamps=per_query_timestamps,
        global_top_docs=global_top_docs,
        extra_columns=extra_columns,
        per_query_extra_values=per_query_extra_values,
        remove_columns_from_results=remove_columns_from_results,
    )



