"""
TransientIndexManager: FAISS-based temporary index for batch embeddings.

Used when augmentation_mode=transient_query_index (do not modify the main corpus).
Supports in-memory indexing for small batches and an on-disk temporary index for large runs.
Maintains an ID→name map for resolving search results without touching the base mapping.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover (environment without faiss)
    faiss = None  # type: ignore

logger = logging.getLogger(__name__)


class TransientIndexManager:
    """A lightweight FAISS index scoped to a single run/batch.

    This index is discarded after the run ends. It carries its own ID→name mapping
    to resolve item names in results and does not alter the persistent corpus mapping.
    """

    def __init__(
        self,
        feature_dim: int,
        index_type: str = "flat",
        hnsw_m: int = 32,
        hnsw_efconstruction: int = 40,
        use_on_disk: bool = False,
        temp_dir: Optional[Path] = None,
    ) -> None:
        """Create a transient FAISS index for a given feature dimension.

        Args:
            feature_dim (int): Dimensionality of the embedding vectors.
            index_type (str): 'flat' (L2) or 'hnsw' (HNSW-Flat) index variant.
            hnsw_m (int): HNSW graph M parameter (neighbors per node) when index_type='hnsw'.
            hnsw_efconstruction (int): HNSW efConstruction parameter for build time quality.
            use_on_disk (bool): If True and temp_dir provided, prepare a temp directory (future extension).
            temp_dir (Optional[pathlib.Path]): Optional temp directory for on-disk artifacts.
        """
        if faiss is None:
            raise ImportError("faiss is not available for TransientIndexManager")
        self.feature_dim = int(feature_dim)
        self.index_type = index_type.lower()
        self.hnsw_m = int(hnsw_m)
        self.hnsw_efconstruction = int(hnsw_efconstruction)
        self.use_on_disk = bool(use_on_disk)
        self.temp_dir = temp_dir.resolve() if temp_dir else None
        if self.use_on_disk and self.temp_dir:
            self.temp_dir.mkdir(parents=True, exist_ok=True)

        self._index: Optional[Any] = None
        self._id_to_name: Dict[int, str] = {}

        self._init_index()
        logger.info("TransientIndexManager initialized: type=%s, on_disk=%s, temp_dir=%s", self.index_type, self.use_on_disk, str(self.temp_dir) if self.temp_dir else None)

    def _init_index(self) -> None:
        """Initialize the underlying FAISS index based on the configured type."""
        if self.index_type == "flat":
            self._index = faiss.IndexFlatL2(self.feature_dim)
        elif self.index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(self.feature_dim, self.hnsw_m)
            self._index.hnsw.efConstruction = self.hnsw_efconstruction
        else:
            # Keeping IVF out to avoid training for transient indexes
            raise ValueError("Unsupported transient index_type. Use 'flat' or 'hnsw'.")

    def is_ready(self) -> bool:
        """Return True when the index exists and contains at least one item."""
        return self._index is not None and self._index.ntotal > 0

    def add(self, vectors: np.ndarray, names: List[str]) -> None:
        """Add a batch of vectors with aligned names to the transient index.

        Expects vectors.shape == (N, D) and len(names) == N. Stores an internal
        ID→name map to resolve search hits later. Vectors must match feature_dim.
        """
        if self._index is None:
            raise RuntimeError("Transient index not initialized")
        if vectors.ndim != 2 or vectors.shape[1] != self.feature_dim:
            raise ValueError(f"vectors must have shape (N, {self.feature_dim})")
        if vectors.shape[0] != len(names):
            raise ValueError("vectors count and names length mismatch")
        start_id = self._index.ntotal
        self._index.add(vectors)
        for i, n in enumerate(names):
            self._id_to_name[start_id + i] = str(n)

    def search(self, query_vector: np.ndarray, top_k: int, hnsw_efsearch: Optional[int] = None) -> List[Tuple[str, float]]:
        """Search top_k nearest items and return (name, similarity) pairs.

        For HNSW, a temporary efSearch override may be applied for this call only.
        Similarity is derived from L2 distance assuming normalized vectors.
        """
        if self._index is None or self._index.ntotal == 0:
            return []
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        # set HNSW efSearch temporarily
        original_ef = None
        if hnsw_efsearch is not None and hasattr(self._index, 'hnsw'):
            original_ef = self._index.hnsw.efSearch
            self._index.hnsw.efSearch = hnsw_efsearch
        try:
            distances, indices = self._index.search(query_vector, top_k)
        finally:
            if original_ef is not None:
                self._index.hnsw.efSearch = original_ef
        results: List[Tuple[str, float]] = []
        for i in range(indices.shape[1]):
            idx = int(indices[0, i])
            if idx == -1:
                continue
            name = self._id_to_name.get(idx)
            if not name:
                continue
            # Convert L2 distance to similarity for normalized vectors
            dist_sq = float(distances[0, i])
            sim = max(0.0, 1.0 - (dist_sq / 2.0))
            results.append((name, sim))
        return results

    def total_items(self) -> int:
        """Return total items currently stored in the transient index."""
        return int(self._index.ntotal) if self._index is not None else 0

    def get_name_map(self) -> Dict[int, str]:
        """Return a copy of the internal ID→name mapping for debugging/inspection."""
        return dict(self._id_to_name)

    def clear(self) -> None:
        """Reset the index and mapping to an empty state (kept initialized)."""
        self._index = None
        self._id_to_name.clear()
        self._init_index()
