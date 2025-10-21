"""
Vendored TIF text search library export.

Public API:
    - TifTextSearcher: High-level class to find 1-based page indices in a multi-page
      TIF document that contain configured target text. It encapsulates OCR engine
      selection and normalization logic. See `tif_searcher_core.searcher.TifTextSearcher`.

Usage:
    from external.tif_searcher import TifTextSearcher
    searcher = TifTextSearcher()
    matched_pages = searcher.find_text_pages("/path/to/document.tif")

Notes:
    - This vendored package mirrors the import path of an installable package.
      The core engine attempts to import from `external.*` first, then falls back
      to installed package names if available in the environment.
"""
from __future__ import annotations

from .tif_searcher_core import TifTextSearcher

__all__ = ["TifTextSearcher"]