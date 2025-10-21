#!/usr/bin/env python3
"""Smoke test and demo runner for the `key_input` pipeline.

This script validates that the key-driven pipeline can run end-to-end with the
current environment and configuration. It is designed to be:
- Deterministic and safe to run repeatedly
- Minimal yet informative (clear summary, non-zero exit on failure)
- Aligned with production logging and config conventions

Run examples:
  python test_key_input.py
  python test_key_input.py --config configs/image_similarity_config.yaml

Behavior notes:
- Loads .env from the repository root (if present).
- Resolves relative config paths against the package root
  (hyundai_document_authenticator/).
- Emits a concise per-query summary similar to the main CLI for consistency.
"""
from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv, find_dotenv

# Ensure package imports work when executed as a standalone script.
# We add the package root (the directory that contains `external/`) to sys.path.
THIS_FILE: Path = Path(__file__).resolve()
# Package root is the directory that contains the 'external/' package
# For: hyundai_document_authenticator/external/key_input/test_key_input.py
# parents[0]=key_input, [1]=external, [2]=hyundai_document_authenticator
PKG_ROOT: Path = THIS_FILE.parents[2]
REPO_ROOT: Path = PKG_ROOT.parent      # outer repository directory
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

# Load .env early so downstream components see environment variables.
def _load_project_dotenv() -> Optional[Path]:
    """Load a .env file from the project tree if available.

    Returns:
        Optional[Path]: Resolved path to the .env file if found and loaded, else None.
    """
    try:
        path_str = find_dotenv(filename=".env", usecwd=True)
        if path_str:
            load_dotenv(dotenv_path=path_str, override=False)
            return Path(path_str).resolve()
        candidate = REPO_ROOT / ".env"
        if candidate.exists():
            load_dotenv(dotenv_path=str(candidate), override=False)
            return candidate.resolve()
    except Exception:
        # Env loading should never break a smoke test.
        return None
    return None

_ = _load_project_dotenv()

# Ensure relative paths in configs resolve against the package root, so any
# paths like 'mock_api_TEST/filtered_rows.xlsx' are correct regardless of where
# this script is launched from.
try:
    os.chdir(str(PKG_ROOT))
except Exception:
    pass

# Configure logging similar to production setup
try:
    from core_engine.image_similarity_system.log_utils import setup_logging  # type: ignore
    setup_logging()  # default folder/name/level; integrates transformers/httpx
except Exception:
    pass


def _print_summary(result: Dict[str, Any]) -> None:
    """Print a concise per-query summary to stdout.

    Args:
        result: Result dictionary returned by the pipeline.
    """
    try:
        batch_results = result.get("batch_results") or []
        if not batch_results:
            print("No batch results returned from key_input pipeline.")
            return
        for bi, br in enumerate(batch_results, 1):
            res = br.get("result") or {}
            if not isinstance(res, dict):
                print(f"Batch {bi}: unexpected result shape -> {type(res)}")
                continue
            print(
                "\n=== Per-Query Results (batch %d/%d, mode=%s, requested=%s, resolved=%s) ==="
                % (bi, len(batch_results), br.get("mode"), br.get("requested"), br.get("resolved"))
            )
            for entry in res.get("per_query", []) or []:
                try:
                    print(
                        f"Query: {entry['query_document']}  photos={entry.get('num_query_photos','?')}  "
                        f"agg={entry.get('aggregation_strategy_used','-')}  t={entry.get('elapsed_seconds','?')}s"
                    )
                    for j, d in enumerate(entry.get("top_docs", []) or [], 1):
                        doc = d.get("document", "?")
                        score = d.get("score", 0.0)
                        print(f"   {j:02d}. {doc}  score={score:.4f}")
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
    except Exception as _e:
        # Avoid failing on printing summary; this is diagnostic only.
        print(f"Summary print warning: {_e}")


def _resolve_config_path(config_path: Path) -> Path:
    """Resolve a possibly relative config path against the package root.

    Args:
        config_path: Path provided via CLI.

    Returns:
        Path: Absolute path pointing to the configuration file.
    """
    if config_path.is_absolute():
        return config_path
    return (PKG_ROOT / config_path).resolve()


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for running the key_input smoke test.

    Args:
        argv: Optional list of CLI arguments. Defaults to sys.argv if None.

    Returns:
        int: Process exit code (0 on success, non-zero on failure).

    Raises:
        None: All exceptions are caught and converted to non-zero exit codes.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test for external.key_input pipeline. Loads config, runs the pipeline, "
            "and prints a concise summary."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/image_similarity_config.yaml",
        help="Path to YAML config file (relative to package root or absolute)",
    )

    args = parser.parse_args(argv)
    cfg_path = _resolve_config_path(Path(args.config))

    try:
        # Import late to ensure sys.path and env are in place
        from external.key_input.key_input_orchestrator import run_key_input_pipeline  # type: ignore

        if not cfg_path.exists():
            print(f"Config file not found at: {cfg_path}")
            return 2

        result: Any = run_key_input_pipeline(cfg_path)
        if isinstance(result, dict):
            _print_summary(result)
        else:
            # Fallback: print raw result in JSON if possible
            try:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            except Exception:
                print(result)
        return 0
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130
    except Exception as exc:
        print(f"Error running key_input pipeline: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
