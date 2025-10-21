"""
EmbeddingStore: Vectors-only persistence for scalable, privacy-first batch processing.

This module provides a lightweight storage layer to support the two-phase pipeline:
- Phase A: extract/compute embeddings once (streaming), optionally flushing to disk
- Augmentation: add vectors to persistent or transient indices
- Phase B: query by reusing Phase A vectors without re-extracting or re-embedding

Design goals:
- Privacy-first: persist only vectors and minimal metadata (names); never persist pixels
- Scalable: handle millions of items via chunked (sharded) NPZ files
- Simple dependencies: rely on NumPy only for shard storage to avoid heavy IO stacks
- Strong typing and unit-testable functions

Data model per vector:
- vector: np.ndarray of shape (D,), dtype float32/float16
- parent_document_name: str (e.g., "SomeDoc.tif")
- item_name: str (e.g., virtual crop name "SomeDoc_page3_photo0.jpg" or input file name)

Shard format (.npz):
- vectors: float32/float16 array of shape (N, D)
- parent_names: numpy object array of shape (N,)
- item_names: numpy object array of shape (N,)

Typical usage:
    store = EmbeddingStore(base_dir=Path("instance/transient_emb_store/run_20250101_120000"),
                           vector_dim=1280, dtype=np.float16, shard_size=20000, persist_to_disk=True)
    store.add_batch(parent_names=[...], item_names=[...], vectors=vectors_batch)
    ... (repeat for all chunks) ...
    store.finalize()  # flush remaining

    # Phase B (per query)
    vecs, item_names = store.load_vectors_for_parent("SomeDoc.tif")

Note: When persist_to_disk=False, the store remains in-memory (suitable for small batches).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Iterable, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShardInfo:
    """Metadata for a persisted shard file."""
    path: Path
    count: int


class EmbeddingStore:
    """Vectors-only, chunked embedding store.

    Supports in-memory accumulation with optional on-disk sharding to cap RAM usage.
    Provides per-parent retrieval without re-extraction or re-embedding.
    """

    def __init__(
        self,
        base_dir: Optional[Path],
        vector_dim: int,
        dtype: np.dtype = np.float32,
        shard_size: int = 20000,
        persist_to_disk: bool = False,
        run_id: Optional[str] = None,
    ) -> None:
        """Initialize the embedding store.

        Args:
            base_dir: Directory under which shards will be stored (used when persist_to_disk=True).
            vector_dim: Dimension of each feature vector.
            dtype: NumPy dtype for vectors (np.float32 or np.float16 recommended).
            shard_size: Number of rows per shard file when flushing to disk.
            persist_to_disk: When True, store shards as .npz files in base_dir, else keep in memory.
            run_id: Optional identifier to include in shard filenames for traceability.
        """
        if dtype not in (np.float32, np.float16):
            raise ValueError("dtype must be float32 or float16 for compatibility/performance")
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        if persist_to_disk and not base_dir:
            raise ValueError("base_dir must be provided when persist_to_disk=True")

        self.vector_dim: int = int(vector_dim)
        self.dtype: np.dtype = dtype
        self.shard_size: int = int(shard_size)
        self.persist_to_disk: bool = bool(persist_to_disk)
        self.run_id: str = run_id or "run"

        self.base_dir: Optional[Path] = base_dir.resolve() if base_dir else None
        if self.persist_to_disk and self.base_dir:
            self.base_dir.mkdir(parents=True, exist_ok=True)

        # Buffers for current shard accumulation
        self._vec_buf: Optional[np.ndarray] = None  # shape (n, D)
        self._parent_buf: List[str] = []
        self._item_buf: List[str] = []

        # Persisted shards metadata
        self._shards: List[ShardInfo] = []

        # In-memory fallback (when not persisting)
        self._mem_vecs: Optional[np.ndarray] = None
        self._mem_parents: List[str] = []
        self._mem_items: List[str] = []

        self._total_count: int = 0
        self._flush_count: int = 0

        logger.info(
            "EmbeddingStore initialized: dim=%d, dtype=%s, shard_size=%d, persist_to_disk=%s, base_dir=%s",
            self.vector_dim, str(self.dtype), self.shard_size, self.persist_to_disk, str(self.base_dir) if self.base_dir else None
        )

    # -----------------------------
    # Public API
    # -----------------------------
    def add_batch(self, parent_names: List[str], item_names: List[str], vectors: np.ndarray) -> None:
        """Append a batch of vectors and aligned names.

        Args:
            parent_names: List of parent document names (len=N).
            item_names: List of item (crop/image) names (len=N).
            vectors: Array of shape (N, D), will be cast to configured dtype.
        """
        n = len(parent_names)
        if n != len(item_names):
            raise ValueError("parent_names and item_names length mismatch")
        if vectors.ndim != 2 or vectors.shape[1] != self.vector_dim:
            raise ValueError(f"vectors must have shape (N, {self.vector_dim})")
        if vectors.shape[0] != n:
            raise ValueError("vectors row count must match names length")

        # Normalize dtype and ensure contiguous array
        vecs = np.asarray(vectors, dtype=self.dtype)
        if not vecs.flags.c_contiguous:
            vecs = np.ascontiguousarray(vecs)

        if self.persist_to_disk:
            # Initialize buffer if empty
            if self._vec_buf is None or self._vec_buf.shape[0] == 0:
                self._vec_buf = np.empty((0, self.vector_dim), dtype=self.dtype)

            # Concatenate into current shard buffer
            self._vec_buf = np.vstack([self._vec_buf, vecs])  # type: ignore[assignment]
            self._parent_buf.extend(parent_names)
            self._item_buf.extend(item_names)

            # Flush if threshold reached
            while self._vec_buf.shape[0] >= self.shard_size:
                self._flush_shard(self.shard_size)
        else:
            # In-memory mode
            if self._mem_vecs is None or self._mem_vecs.shape[0] == 0:
                self._mem_vecs = vecs.copy()
            else:
                self._mem_vecs = np.vstack([self._mem_vecs, vecs])
            self._mem_parents.extend(parent_names)
            self._mem_items.extend(item_names)

        self._total_count += n

    def finalize(self) -> None:
        """Flush any remaining buffered rows to disk (when persist_to_disk=True)."""
        if self.persist_to_disk:
            if self._vec_buf is not None and self._vec_buf.shape[0] > 0:
                self._flush_shard(self._vec_buf.shape[0])
            logger.info("EmbeddingStore finalize: total=%d, shards=%d", self._total_count, len(self._shards))
        else:
            logger.info("EmbeddingStore finalize (in-memory): total=%d", self._total_count)

    def total_count(self) -> int:
        """Return total number of vectors added (across shards + memory)."""
        return int(self._total_count)

    def list_shards(self) -> List[ShardInfo]:
        """List persisted shard files (empty in in-memory mode)."""
        return list(self._shards)

    def load_vectors_for_parent(self, parent_document_name: str) -> Tuple[np.ndarray, List[str]]:
        """Load all vectors and item names for a given parent document.

        Returns:
            (vectors, item_names): vectors shape (M, D) and parallel list of length M.
        """
        parent = str(parent_document_name)
        if self.persist_to_disk:
            vecs_accum: List[np.ndarray] = []
            items_accum: List[str] = []
            for shard in self._shards:
                try:
                    with np.load(shard.path, allow_pickle=True) as data:
                        parents = data["parent_names"]  # object array
                        mask = (parents == parent)
                        count = int(np.count_nonzero(mask))
                        if count == 0:
                            continue
                        vecs = data["vectors"][mask]
                        names = data["item_names"][mask].tolist()
                        if vecs.size > 0:
                            vecs_accum.append(vecs)
                            items_accum.extend(names)
                except Exception as e:
                    logger.warning("Failed to read shard %s: %s", shard.path, e)
            if not vecs_accum:
                return np.empty((0, self.vector_dim), dtype=self.dtype), []
            return np.vstack(vecs_accum), items_accum
        else:
            if self._mem_vecs is None or not self._mem_parents:
                return np.empty((0, self.vector_dim), dtype=self.dtype), []
            parents_arr = np.array(self._mem_parents, dtype=object)
            mask = (parents_arr == parent)
            if not np.any(mask):
                return np.empty((0, self.vector_dim), dtype=self.dtype), []
            vecs = self._mem_vecs[mask]
            names = np.array(self._mem_items, dtype=object)[mask].tolist()
            return vecs, names

    def iter_parents(self) -> Iterable[str]:
        """Yield unique parent_document_name entries present in the store."""
        if self.persist_to_disk:
            # For large scale, iterate shards and yield unique names lazily
            seen: set[str] = set()
            for shard in self._shards:
                try:
                    with np.load(shard.path, allow_pickle=True) as data:
                        parents = data["parent_names"].tolist()
                        for p in parents:
                            if p not in seen:
                                seen.add(p)
                                yield p
                except Exception as e:
                    logger.warning("Failed to read shard %s: %s", shard.path, e)
        else:
            for p in sorted(set(self._mem_parents)):
                yield p

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _flush_shard(self, rows_to_flush: int) -> None:
        """Flush the first rows_to_flush rows from buffers to a new shard file."""
        assert self.persist_to_disk and self.base_dir is not None
        assert self._vec_buf is not None
        n = int(rows_to_flush)
        if n <= 0:
            return

        # Slice current buffers
        vecs = self._vec_buf[:n]
        parents = np.array(self._parent_buf[:n], dtype=object)
        items = np.array(self._item_buf[:n], dtype=object)

        # Persist npz shard
        self._flush_count += 1
        shard_path = self.base_dir / f"emb_shard_{self.run_id}_{self._flush_count:06d}.npz"
        try:
            np.savez_compressed(shard_path, vectors=vecs, parent_names=parents, item_names=items)
            self._shards.append(ShardInfo(path=shard_path, count=n))
            logger.debug("Flushed shard: %s (%d rows)", shard_path.name, n)
        except Exception as e:
            logger.error("Failed to write shard %s: %s", shard_path, e)
            raise

        # Remove flushed rows from buffers
        remain = self._vec_buf.shape[0] - n
        if remain > 0:
            self._vec_buf = self._vec_buf[n:]
            self._parent_buf = self._parent_buf[n:]
            self._item_buf = self._item_buf[n:]
        else:
            # Reset empty buffers
            self._vec_buf = np.empty((0, self.vector_dim), dtype=self.dtype)
            self._parent_buf = []
            self._item_buf = []


# Convenience factory

def create_in_memory_store(vector_dim: int, dtype: np.dtype = np.float32) -> EmbeddingStore:
    """Create an in-memory EmbeddingStore (no disk IO)."""
    return EmbeddingStore(base_dir=None, vector_dim=vector_dim, dtype=dtype, shard_size=1_000_000, persist_to_disk=False)


def create_disk_store(base_dir: Path, vector_dim: int, shard_size: int = 20000, dtype: np.dtype = np.float16, run_id: Optional[str] = None) -> EmbeddingStore:
    """Create a disk-backed EmbeddingStore with recommended defaults for large runs."""
    return EmbeddingStore(base_dir=base_dir, vector_dim=vector_dim, dtype=dtype, shard_size=shard_size, persist_to_disk=True, run_id=run_id)
