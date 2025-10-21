"""
Serialization helpers for consistent JSON formatting across CSV/JSON fields.

This module centralizes JSON serialization behavior to avoid divergence between
call sites. It uses the standard library only and mirrors existing behavior.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def json_dumps(data: Any) -> str:
    """Serialize a Python value to a JSON string using ensure_ascii=False.

    Behavior:
    - None -> "{}" when used for dict-like fields, or "[]" for list-like fields. This function
      does not infer intent automatically; callers should pass {} or [] accordingly. When None
      is passed here, it will serialize to "null" to avoid silent type changes.

    Args:
        data: A JSON-serializable value.

    Returns:
        A JSON string.
    """
    return json.dumps(data, ensure_ascii=False)


def csv_safe(value: Any) -> Any:
    """Return a CSV-safe representation for a cell value.

    Rules:
    - Pass through primitives (str, int, float, bool, None)
    - For dict or list, return json_dumps(value)
    - For other objects, return json_dumps(value) as a best-effort fallback

    Args:
        value: Any value destined for a CSV cell.

    Returns:
        A primitive or JSON string.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (dict, list)):
        return json_dumps(value)
    return json_dumps(value)
