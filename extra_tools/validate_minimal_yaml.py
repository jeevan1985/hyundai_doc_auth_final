"""Validate minimal YAML configurations for external modules.

This script loads each newly created minimal YAML file to ensure syntactic
validity. It prints a concise report and exits with a non-zero status on error.

Usage:
    python extra_tools/validate_minimal_yaml.py

Notes:
    - Paths are defined relative to the repository root (this script's location
      is d:/frm_git/hyundai_document_authenticator/extra_tools/).
    - Requires PyYAML to be installed in the active Python environment.
"""
from __future__ import annotations

from pathlib import Path
from typing import List
import sys

try:
    import yaml
except Exception as exc:  # pragma: no cover
    print(f"ERROR: PyYAML not available: {exc}")
    sys.exit(2)


def main() -> int:
    """Entry point for validation of minimal YAML files.

    Returns:
        int: Process exit code (0 on success, non-zero on failure).
    """
    repo_root = Path(__file__).resolve().parents[1]

    files: List[Path] = [
        repo_root / "hyundai_document_authenticator" / "external" / "image_authenticity_classifier" / "classifier_config_minimal.yaml",
        repo_root / "hyundai_document_authenticator" / "external" / "key_input" / "key_input_config_minimal.yaml",
        repo_root / "hyundai_document_authenticator" / "external" / "result_gui" / "config_minimal.yaml",
    ]

    all_ok = True
    for p in files:
        try:
            if not p.exists():
                print(f"MISSING: {p}")
                all_ok = False
                continue
            with p.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            # Only basic structural check: top-level must be a mapping or list
            if not isinstance(data, (dict, list, type(None))):
                print(f"INVALID STRUCTURE (not mapping/list): {p}")
                all_ok = False
            else:
                print(f"OK: {p}")
        except Exception as exc:  # pragma: no cover
            print(f"ERROR parsing {p}: {exc}")
            all_ok = False
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
