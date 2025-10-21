"""Core API exports for the vendored TIF searcher engine.

Exposes TifTextSearcher, the primary interface for searching multi-page TIFs
for pages containing a configured text using configurable OCR backends.
"""
from __future__ import annotations

from .searcher import TifTextSearcher

__all__ = ["TifTextSearcher"]
