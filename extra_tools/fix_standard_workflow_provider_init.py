"""Fix the legacy provider-init block in execute_tif_batch_search_workflow.

This script performs an anchored replacement in
hyundai_document_authenticator/core_engine/image_similarity_system/workflow.py
for the standard (non-augmented) workflow. It replaces the duplicated and
currently malformed provider-initialization block with a call to the centralized
helper `_initialize_provider_manager` using legacy message style, while
preserving variable names, logs, and downstream behavior.

It also compiles the file afterwards to verify syntax correctness.

Why: The legacy block is syntactically corrupted (missing commas/parentheses),
which breaks imports. Anchored replacement ensures precise, behavior-identical
refactor without touching unrelated logic.

Usage:
    python extra_tools/fix_standard_workflow_provider_init.py

Raises:
    RuntimeError: When anchors cannot be found or when the compile check fails.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Tuple
import py_compile

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "hyundai_document_authenticator" / "core_engine" / "image_similarity_system" / "workflow.py"

START_ANCHOR_SUBSTR = "Vector DB Search: provider="
END_ANCHOR_PREFIX = "privacy_mode_enabled = bool("

REPLACEMENT_BLOCK = (
    "logger.info(\"Vector DB Search: provider=%s, allow_fallback=%s, fallback_choice=%s\", provider, allow_fallback, fallback_choice)\n\n"
    "base_manager, indexed_item_count, final_index_file_path_str, fallback_to_bruteforce_message, selected_fallback_mode, _fatal_error = _initialize_provider_manager(\n"
    "    provider=provider,\n"
    "    db_conf=db_conf,\n"
    "    feature_extractor=feature_extractor,\n"
    "    project_root=project_root,\n"
    "    search_task_conf=search_conf,\n"
    "    full_config=cloned_config,\n"
    "    message_style='legacy',\n"
    ")\n"
    "if _fatal_error:\n"
    "    return {\"status\": \"error\", \"exit_code\": 1, \"message\": _fatal_error}\n\n"
    "vector_db_manager = base_manager\n\n"
    "# Privacy guardrail: disallow bruteforce when privacy_mode=true (no persistence)\n"
    "privacy_mode_enabled = bool(search_conf.get('privacy_mode', True))\n"
)


def _detect_indentation(line: str) -> str:
    """Return the leading whitespace of a line."""
    return line[: len(line) - len(line.lstrip(" \t"))]


def _indent_block(block: str, indent: str) -> str:
    """Indent every non-empty line of block by indent."""
    out_lines = []
    for ln in block.splitlines():
        if ln:
            out_lines.append(indent + ln)
        else:
            out_lines.append(ln)
    return "\n".join(out_lines)


def replace_block(file_path: Path) -> Tuple[int, int]:
    """Replace the provider-init block in standard workflow with helper call.

    Args:
        file_path: Path to workflow.py

    Returns:
        A tuple (start_line_idx, end_line_idx) for the replaced block (0-based).

    Raises:
        RuntimeError: When anchors cannot be located or replacement fails.
    """
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Find the start anchor
    start_idx = None
    for i, ln in enumerate(lines):
        if START_ANCHOR_SUBSTR in ln:
            start_idx = i
            break
    if start_idx is None:
        raise RuntimeError("Start anchor not found: 'Vector DB Search: provider='")

    # Find the end anchor (line starting with privacy_mode_enabled = bool()
    end_idx = None
    for j in range(start_idx, min(len(lines), start_idx + 3000)):
        if lines[j].lstrip().startswith(END_ANCHOR_PREFIX):
            end_idx = j
            break
    if end_idx is None:
        raise RuntimeError("End anchor not found after start anchor: 'privacy_mode_enabled = bool('")

    # Determine indentation from the start line
    indent = _detect_indentation(lines[start_idx])
    replacement = _indent_block(REPLACEMENT_BLOCK, indent)

    # Build new content
    new_lines = lines[:start_idx] + replacement.splitlines() + lines[end_idx + 1 :]
    file_path.write_text("\n".join(new_lines) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")
    return start_idx, end_idx


def main() -> None:
    """Run anchored replacement and compile verification."""
    if not TARGET.exists():
        raise RuntimeError(f"Target file not found: {TARGET}")

    backup = TARGET.with_suffix(TARGET.suffix + ".bak")
    shutil.copy2(TARGET, backup)

    try:
        s, e = replace_block(TARGET)
        # Compile check
        py_compile.compile(str(TARGET), doraise=True)
        print(f"APPLIED: replaced lines {s+1}..{e+1} and compiled successfully.")
    except Exception as exc:
        # Restore backup on failure
        shutil.copy2(backup, TARGET)
        print(f"ERROR: {exc}")
        raise
    finally:
        # Keep backup for inspection
        pass


if __name__ == "__main__":
    main()
