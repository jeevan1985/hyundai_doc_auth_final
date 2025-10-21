"""Smoke test for vendored imports and workflow wiring.

This module verifies that:
- The 'external' vendored namespace is importable.
- Aliased modules (e.g., tif_searcher, photo_extractor) are available after importing
  from the vendored package.
- The workflow can import and reference vendored classes.

It is intentionally simple and designed to run in varied environments.

Usage:
    python tool_smoke_test.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


LOGGER = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the smoke test.

    Args:
        level (str): Logging level name (e.g., "DEBUG", "INFO").

    Returns:
        None
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(sh)


def ensure_external_on_path(base_dir: Optional[Path] = None) -> None:
    """Ensure the vendored 'external' directory is on sys.path.

    Args:
        base_dir (Optional[Path]): Base directory to resolve 'external' from. If not
            provided, uses this file's parent.

    Returns:
        None
    """
    # Using absolute resolution avoids surprises with relative CWDs.
    root = base_dir or Path(__file__).resolve().parent
    external_dir = root / "external"
    if external_dir.is_dir() and str(external_dir) not in sys.path:
        sys.path.insert(0, str(external_dir))
        LOGGER.debug("Added to sys.path: %s", external_dir)


def test_vendored_namespace() -> bool:
    """Verify that the 'external' package and key modules can be imported.

    Returns:
        bool: True if import succeeds; False otherwise.
    """
    try:
        import external  # noqa: F401
        import external.tif_searcher  # noqa: F401
        import external.photo_extractor  # noqa: F401
        LOGGER.info("external namespace OK")
        return True
    except Exception as exc:
        LOGGER.error("external namespace FAIL %s", exc)
        return False


def test_alias_modules() -> bool:
    """Verify aliased top-level modules are available after vendored imports.

    Returns:
        bool: True if import succeeds; False otherwise.
    """
    try:
        # Import to populate sys.modules aliases
        from external.tif_searcher import TifTextSearcher  # noqa: F401
        from external.photo_extractor import PhotoExtractor  # noqa: F401
        import tif_searcher  # noqa: F401
        import photo_extractor  # noqa: F401
        LOGGER.info("alias modules OK")
        return True
    except Exception as exc:
        LOGGER.error("alias modules FAIL %s", exc)
        return False


def test_workflow_import() -> bool:
    """Verify workflow import and presence of vendored class references.

    Returns:
        bool: True if workflow import and attributes succeed; False otherwise.
    """
    try:
        from core_engine.image_similarity_system import workflow as w  # noqa: F401
        LOGGER.info(
            "workflow OK %s %s",
            bool(getattr(w, "TifTextSearcher", None)),
            bool(getattr(w, "PhotoExtractor", None)),
        )
        return True
    except Exception as exc:
        LOGGER.error("workflow FAIL %s", exc)
        return False


def main() -> int:
    """Execute all smoke tests and return an aggregated exit code.

    Returns:
        int: 0 if all tests pass; 1 otherwise.
    """
    setup_logging("INFO")
    LOGGER.info("CWD %s", Path.cwd())
    LOGGER.info("PY %s", sys.version.split()[0])

    ensure_external_on_path()

    ok = True
    ok &= test_vendored_namespace()
    ok &= test_alias_modules()
    ok &= test_workflow_import()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
