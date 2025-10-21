"""
QueryMerger: Merge base and transient search results, de-duplicate, and filter self-matches.
"""
from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Any, Optional, Set
from pathlib import Path

from .tif_utils import parent_doc_from_db_stem

logger = logging.getLogger(__name__)


class QueryMerger:
    """Merge and filter search results across base and transient indexes.

    - De-duplicate by item unique name/path
    - Apply self-match filtering per configuration
    - Return ranked merged results by score (descending)
    """

    @staticmethod
    def merge_results(
        base_results: List[Tuple[str, float]],
        transient_results: List[Tuple[str, float]],
        top_k: int,
        include_query_image_to_result: bool,
        query_parent_document_name: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Merge and filter two result lists.

        Args:
            base_results: list of (path_or_name, score)
            transient_results: list of (path_or_name, score)
            top_k: cap after merge and filtering
            include_query_image_to_result: when False, filter same-parent and exact self vectors
            query_parent_document_name: parent's name of the current query document (e.g., SomeDoc.tif)

        Returns:
            Merged, de-duplicated, filtered and ranked results.
        """
        combined = list(base_results or []) + list(transient_results or [])
        if not combined:
            return []

        # De-duplicate by key (normalized path or unique name)
        best_by_key: Dict[str, float] = {}
        for name_or_path, score in combined:
            key = QueryMerger._normalize_key(name_or_path)
            prev = best_by_key.get(key)
            if prev is None or score > prev:
                best_by_key[key] = float(score)

        # Filter self matches if required
        items: List[Tuple[str, float]] = []

        # Normalize the query's base core (stem) once for consistent comparison.
        # We purposefully DO NOT include any page/photo suffix so that all variants
        # like "<stem>_pageY[_photoZ]", "<stem> - Copy_pageY", "<stem>_3_pageY" are excluded.
        normalized_query_core: Optional[str] = None
        mapped_query_core: Optional[str] = None
        if not include_query_image_to_result and query_parent_document_name:
            try:
                from pathlib import Path as _P
                normalized_query_core = _P(query_parent_document_name).stem
            except Exception:
                normalized_query_core = str(query_parent_document_name)

            # Also derive mapped core (e.g., insert_token before tail_len) to catch storage-mapped variants
            # Example mapping: prefix=file_name[:-5], suffix=file_name[-5:], mapped_core=f"{prefix}001{suffix}"
            try:
                from external.key_input.key_input_orchestrator import NameMappingConfig as _NMC
                _cfg = _NMC()  # use default mapping parameters unless overridden elsewhere
                if normalized_query_core:
                    n = len(normalized_query_core)
                    t = max(0, int(getattr(_cfg, 'tail_len', 5)))
                    insert_token = str(getattr(_cfg, 'insert_token', '001'))
                    if n > t:
                        prefix = normalized_query_core[: n - t]
                        suffix = normalized_query_core[n - t :]
                        mapped_query_core = f"{prefix}{insert_token}{suffix}"
            except Exception:
                mapped_query_core = None

        for key, score in best_by_key.items():
            if include_query_image_to_result:
                items.append((key, score))
                continue
            # When filtering, skip any result whose stem starts with the query's base core
            # OR the mapped core. This excludes exact self, page/crop variants
            # (generate_virtual_crop_name), local copy/version patterns, and mapped-core variants.
            if normalized_query_core or mapped_query_core:
                try:
                    hit_stem = Path(key).stem
                except Exception:
                    hit_stem = Path(str(key)).stem
                if (
                    (normalized_query_core and hit_stem.startswith(normalized_query_core))
                    or (mapped_query_core and hit_stem.startswith(mapped_query_core))
                ):
                    continue
            items.append((key, score))

        # Rank by score desc and cap to top_k
        items.sort(key=lambda kv: kv[1], reverse=True)
        return items[:top_k]

    @staticmethod
    def _normalize_key(name_or_path: str) -> str:
        """Normalize a result identifier to a stable de-duplication key.

        Prefer an absolute resolved path when the input points to a real file; otherwise
        return the original string (e.g., a virtual name).
        """
        try:
            p = Path(name_or_path)
            if p.exists():
                return str(p.resolve())
        except Exception:
            pass
        return str(name_or_path)

    @staticmethod
    def _parent_from_key(key: str) -> Optional[str]:
        """Derive parent TIF document name from a search hit key.

        Accepts either a filesystem path or a virtual item name and returns the
        normalized parent TIF filename used for self-match filtering.
        """
        try:
            stem = Path(key).stem
        except Exception:
            # If key is not a path, fall back to treating it as an item name
            stem = Path(str(key)).stem
        return parent_doc_from_db_stem(stem)
