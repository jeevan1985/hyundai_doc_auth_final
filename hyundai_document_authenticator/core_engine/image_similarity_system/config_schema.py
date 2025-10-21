"""
Lightweight configuration schema helpers (stdlib only).

These helpers coerce raw configuration values into expected Python types, using
provided defaults on any error. They are intentionally defensive and minimal to
avoid any behavior changes or new dependencies.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def coerce_bool(value: Any, default: bool) -> bool:
    """Coerce a value to bool with a default fallback.

    Accepted truthy values (case-insensitive): 'true', '1', 'yes', 'y', 1
    Accepted falsy values: 'false', '0', 'no', 'n', 0

    Args:
        value: Raw input value.
        default: Default to return on errors or None.

    Returns:
        A boolean value.
    """
    if isinstance(value, bool):
        return value
    try:
        if value is None:
            return bool(default)
        if isinstance(value, (int, float)):
            return bool(int(value) != 0)
        s = str(value).strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
    except Exception as e:
        logger.debug("coerce_bool: error '%s' on value=%r -> default=%r", e, value, default)
    return bool(default)


def coerce_int(value: Any, default: int) -> int:
    """Coerce to int with default fallback.

    Args:
        value: Raw value.
        default: Default to use on failure.

    Returns:
        int
    """
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception as e:
        logger.debug("coerce_int: error '%s' on value=%r -> default=%r", e, value, default)
        return int(default)


def coerce_float(value: Any, default: float) -> float:
    """Coerce to float with default fallback.

    Args:
        value: Raw value.
        default: Default to use on failure.

    Returns:
        float
    """
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception as e:
        logger.debug("coerce_float: error '%s' on value=%r -> default=%r", e, value, default)
        return float(default)


def coerce_list_str(value: Any, default: List[str]) -> List[str]:
    """Coerce to a list[str] with default fallback.

    Accepts comma-separated strings or generic iterables of strings.

    Args:
        value: Raw value.
        default: Default list to use on failure.

    Returns:
        List[str]
    """
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        try:
            return [str(x) for x in value]
        except Exception as e:
            logger.debug("coerce_list_str: error '%s' on list value=%r -> default=%r", e, value, default)
            return list(default or [])
    try:
        s = str(value)
        if not s.strip():
            return list(default or [])
        return [part.strip() for part in s.split(',')]
    except Exception as e:
        logger.debug("coerce_list_str: error '%s' on value=%r -> default=%r", e, value, default)
        return list(default or [])


def resolve_path_maybe_relative(path_str: Optional[str], project_root: Optional[Path]) -> Optional[Path]:
    """Resolve a path string to an absolute path, supporting project-root-relative values.

    Args:
        path_str: Input string path, possibly relative.
        project_root: Project root directory for resolution when path is not absolute.

    Returns:
        Absolute Path or None when input is falsy.
    """
    if not path_str:
        return None
    try:
        p = Path(path_str)
        if p.is_absolute():
            return p.resolve()
        base = project_root if project_root else Path.cwd()
        return (base / p).resolve()
    except Exception as e:
        logger.debug("resolve_path_maybe_relative: error '%s' on value=%r", e, path_str)
        return None
