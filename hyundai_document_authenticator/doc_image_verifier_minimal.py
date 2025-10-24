#!/usr/bin/env python
"""Minimal TIF Document Similarity CLI (client-friendly, output-identical).

This CLI keeps the surface area small while preserving the exact console output
format of the full CLI for the supported flows. It defers almost all behavior
to the YAML configuration and provides a few optional overrides.

Supported commands
- build-image-index
- search-doc

Overrides
- --config-path: Path to YAML config (defaults to configs/image_similarity_config.yaml)
- --folder:      Optional override for the relevant input TIF folder
- --doc-input-start: Override input_mode.doc_input_start: tif|key

Examples (Windows CMD/PowerShell)
- python hyundai_document_authenticator\doc_image_verifier_minimal.py build-image-index
- python hyundai_document_authenticator\doc_image_verifier_minimal.py search-doc --doc-input-start key

Examples (Linux/macOS Bash)
- python3 hyundai_document_authenticator/doc_image_verifier_minimal.py build-image-index
- python3 hyundai_document_authenticator/doc_image_verifier_minimal.py search-doc --doc-input-start key
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import sys
import tempfile

import typer
import yaml
from dotenv import load_dotenv, find_dotenv

# Robust package import for direct execution
sys.path.append(str(Path(__file__).resolve().parent))

# Optional key-driven pipeline orchestrator (keeps parity with full CLI)
try:
    from external.key_input.key_input_orchestrator import (
        run_key_input_pipeline,
        run_key_input_pipeline_for_index,
    )
except Exception:
    run_key_input_pipeline = None  # type: ignore
    run_key_input_pipeline_for_index = None  # type: ignore

try:
    from core_engine.image_similarity_system.config_loader import load_and_merge_configs  # type: ignore
    from core_engine.image_similarity_system.workflow import (  # type: ignore
        build_index_from_tif_folder_workflow,
        execute_tif_batch_search_workflow,
    )
    from core_engine.image_similarity_system.log_utils import setup_logging  # type: ignore
except Exception as e:
    print(
        "Error: Failed to import required modules. Ensure you are running from the project root and dependencies are installed.",
        file=sys.stderr,
    )
    print(f"Details: {e}", file=sys.stderr)
    sys.exit(1)


def _load_project_dotenv(start: Optional[Path] = None, filename: str = ".env") -> Optional[Path]:
    """Locate and load a project-level .env file.

    This mirrors the full CLI behavior and ensures environment variables are
    available (e.g., for DB connections) even when running directly.

    Args:
        start: Optional starting directory for find_dotenv (uses CWD when None).
        filename: Name of the environment file to search for.

    Returns:
        The resolved path if a .env was found and loaded; otherwise None.
    """
    try:
        path_str: str = find_dotenv(filename=filename, usecwd=True)
        if path_str:
            load_dotenv(dotenv_path=path_str, override=False)
            return Path(path_str).resolve()
        # Fallback to repo root relative to this file
        repo_root = Path(__file__).resolve().parents[1]
        candidate = repo_root / filename
        if candidate.exists():
            load_dotenv(dotenv_path=str(candidate), override=False)
            return candidate.resolve()
    except Exception:
        return None
    return None


# Load .env early (do not break CLI if missing)
try:
    _ = _load_project_dotenv()
except Exception:
    pass

# Initialize logging (matching full CLI intent; never fail on logging issues)
try:
    setup_logging()
except Exception:
    pass

app: typer.Typer = typer.Typer(add_completion=False, help="Minimal TIF Document Similarity CLI")


def _override_path(cfg: Dict[str, Any], dotted_key: str, value: Optional[Path]) -> None:
    """Override a string path in a nested mapping if provided.

    Args:
        cfg: Loaded configuration mapping.
        dotted_key: Dot-delimited key path (e.g., "search_task.input_tif_folder_for_search").
        value: Optional path to apply.
    """
    if value is None:
        return
    node = cfg
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = str(value)


@app.command("search-doc")
def search_doc(
    config_path: Path = typer.Option(Path("configs/image_similarity_config.yaml"), help="Path to YAML config."),
    doc_input_start: Optional[str] = typer.Option(None, "--doc-input-start", help="Override input_mode.doc_input_start: tif|key"),
    folder: Optional[Path] = typer.Option(None, "--folder", help="Optional override for search_task.input_tif_folder_for_search."),
) -> None:
    """Run TIF batch search and print results identically to the full CLI.

    Behavior:
    - If effective input mode is 'key', delegates to the key-input pipeline and
      prints batch-formatted results exactly like the full CLI.
    - Otherwise, runs the TIF folder workflow and prints the standard per-query
      and summary lines.
    """
    try:
        cfg = load_and_merge_configs(config_path)
        _override_path(cfg, "search_task.input_tif_folder_for_search", folder)

        # Determine effective input mode (mirror full CLI semantics)
        on_disk_start = str((cfg.get("input_mode", {}) or {}).get("doc_input_start", "key")).lower()
        effective_start = str(doc_input_start or on_disk_start).lower()
        if effective_start not in {"tif", "key"}:
            print(
                f"Warning: input_mode.doc_input_start='{effective_start}' is not supported; falling back to default 'key' input mode.",
                file=sys.stderr,
            )
            effective_start = "key"

        # Key-driven pipeline
        if effective_start == "key":
            if run_key_input_pipeline is None:
                print(
                    "Error: key-driven input requested but external.filename_mapper module is unavailable.",
                    file=sys.stderr,
                )
                raise typer.Exit(code=2)
            # If override requested and on-disk value is not 'key', write a temp config with key mode
            cfg_path_for_orch = config_path
            tmp_cfg_file: Optional[Path] = None
            if (doc_input_start is not None) and (on_disk_start != "key"):
                try:
                    cfg_copy = dict(cfg)
                    im = cfg_copy.setdefault("input_mode", {})
                    im["doc_input_start"] = "key"
                    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml", encoding="utf-8") as tf:
                        yaml.safe_dump(cfg_copy, tf, allow_unicode=True)
                        tmp_cfg_file = Path(tf.name)
                    cfg_path_for_orch = tmp_cfg_file
                except Exception as _e:
                    print(
                        f"Warning: failed to write temp config for key-mode override: {_e}. Falling back to config file.",
                        file=sys.stderr,
                    )
            result = run_key_input_pipeline(cfg_path_for_orch)  # type: ignore[arg-type]
            if tmp_cfg_file:
                try:
                    tmp_cfg_file.unlink()
                except Exception:
                    pass
            # Pretty-print results like the full CLI
            if isinstance(result, dict):
                batch_results = result.get("batch_results") or []
                for bi, br in enumerate(batch_results, 1):
                    res = br.get("result") or {}
                    if not isinstance(res, dict):
                        continue
                    print(
                        "\n=== Per-Query Results (batch %d/%d, mode=%s, requested=%s, resolved=%s) ==="
                        % (bi, len(batch_results), br.get("mode"), br.get("requested"), br.get("resolved"))
                    )
                    for entry in res.get("per_query", []) or []:
                        try:
                            print(
                                f"Query: {entry['query_document']}  photos={entry['num_query_photos']}  "
                                f"agg={entry['aggregation_strategy_used']}  t={entry['elapsed_seconds']:.3f}s"
                            )
                            for j, d in enumerate(entry.get("top_docs", []) or [], 1):
                                print(f"   {j:02d}. {d['document']}  score={float(d['score']):.4f}")
                        except Exception:
                            print(entry)
                    if res.get("fallback_to_bruteforce_message"):
                        print(f"\n[Fallback] {res['fallback_to_bruteforce_message']}")
                    if res.get("tif_run_output_path"):
                        print(f"\nRun output folder: {res['tif_run_output_path']}")
                    if res.get("tif_run_summary_json_path"):
                        print(f"Summary JSON: {res['tif_run_summary_json_path']}")
                    if res.get("tif_run_db_export_csv_path"):
                        print(f"PostgreSQL CSV export: {res['tif_run_db_export_csv_path']}")
            else:
                print(result)
            return

        # TIF folder workflow path (standard prints)
        project_root = Path(__file__).resolve().parent
        result: Dict[str, Any] = execute_tif_batch_search_workflow(cfg, project_root)
        print("\n=== Per-Query Results ===")
        for entry in result.get("per_query", []) or []:
            try:
                print(
                    f"Query: {entry['query_document']}  photos={entry['num_query_photos']}  "
                    f"agg={entry['aggregation_strategy_used']}  t={entry['elapsed_seconds']:.3f}s"
                )
                for j, d in enumerate(entry.get("top_docs", []) or [], 1):
                    print(f"   {j:02d}. {d['document']}  score={float(d['score']):.4f}")
            except Exception:
                print(entry)
        if result.get("fallback_to_bruteforce_message"):
            print(f"\n[Fallback] {result['fallback_to_bruteforce_message']}")
        if result.get("tif_run_output_path"):
            print(f"\nRun output folder: {result['tif_run_output_path']}")
        if result.get("tif_run_summary_json_path"):
            print(f"Summary JSON: {result['tif_run_summary_json_path']}")
        if result.get("tif_run_db_export_csv_path"):
            print(f"PostgreSQL CSV export: {result['tif_run_db_export_csv_path']}")

    except typer.Exit:
        raise
    except Exception as e:
        print(f"Error: search failed: {e}", file=sys.stderr)
        raise typer.Exit(code=2)


@app.command("build-image-index")
def build_image_index(
    config_path: Path = typer.Option(Path("configs/image_similarity_config.yaml"), help="Path to YAML config."),
    doc_input_start: Optional[str] = typer.Option(None, "--doc-input-start", help="Override input_mode.doc_input_start: tif|key"),
    folder: Optional[Path] = typer.Option(None, "--folder", help="Optional override for indexing_task.input_tif_folder_for_indexing."),
) -> None:
    """Build an image index; print raw result mapping exactly like the full CLI.

    Honors key-driven mode like the full CLI. When key mode is selected, delegates
    to the key-input indexing pipeline and prints its raw result.
    """
    try:
        cfg = load_and_merge_configs(config_path)
        _override_path(cfg, "indexing_task.input_tif_folder_for_indexing", folder)

        # Determine effective input mode
        on_disk_start = str((cfg.get("input_mode", {}) or {}).get("doc_input_start", "key")).lower()
        effective_start = str(doc_input_start or on_disk_start).lower()
        if effective_start not in {"tif", "key"}:
            effective_start = "key"
        if effective_start == "key":
            if run_key_input_pipeline_for_index is None:
                print(
                    "Error: key-driven input requested for build, but key input orchestrator is unavailable.",
                    file=sys.stderr,
                )
                raise typer.Exit(code=2)
            cfg_path_for_orch = config_path
            tmp_cfg_file: Optional[Path] = None
            if (doc_input_start is not None) and (on_disk_start != "key"):
                try:
                    cfg_copy = dict(cfg)
                    im = cfg_copy.setdefault("input_mode", {})
                    im["doc_input_start"] = "key"
                    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml", encoding="utf-8") as tf:
                        yaml.safe_dump(cfg_copy, tf, allow_unicode=True)
                        tmp_cfg_file = Path(tf.name)
                    cfg_path_for_orch = tmp_cfg_file
                except Exception as _e:
                    print(
                        f"Warning: failed to write temp config for key-mode build override: {_e}. Falling back to config file.",
                        file=sys.stderr,
                    )
            result = run_key_input_pipeline_for_index(cfg_path_for_orch)  # type: ignore[arg-type]
            if tmp_cfg_file:
                try:
                    tmp_cfg_file.unlink()
                except Exception:
                    pass
            print(result)
            return

        # TIF folder indexing path (print raw mapping)
        project_root = Path(__file__).resolve().parent
        result: Dict[str, Any] = build_index_from_tif_folder_workflow(cfg, project_root)
        print(result)

    except typer.Exit:
        raise
    except Exception as e:
        print(f"Error: index build failed: {e}", file=sys.stderr)
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
