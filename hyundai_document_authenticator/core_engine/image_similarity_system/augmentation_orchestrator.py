"""
AugmentationOrchestrator: orchestrates augmentation of the searchable corpus with batch embeddings.

Responsibilities:
- Enforce privacy constraints (no bruteforce when persist_query_crops=false)
- Persistent augmentation: add vectors to FAISS/Qdrant and extend mappings/payloads
- Transient augmentation: build and query a side FAISS index and keep an ID→name map
- Provide search helpers to merge base and transient results
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from .faiss_manager import FaissIndexManager
try:
    from .qdrant_manager import QdrantManager  # type: ignore
except Exception:
    QdrantManager = None  # type: ignore
from .transient_index_manager import TransientIndexManager
from .query_merger import QueryMerger

logger = logging.getLogger(__name__)


class AugmentationOrchestrator:
    """Decide and execute augmentation mode and provide merged search utilities."""

    def __init__(
        self,
        provider: str,
        base_manager: Optional[Any],
        faiss_search_config: Dict[str, Any],
        persist_query_crops: bool,
        augmentation_mode: str,
        transient_batch_index: bool,
        feature_dim: int,
        tmp_dir: Optional[Path] = None,
    ) -> None:
        self.provider = provider.lower()
        self.base_manager = base_manager
        self.faiss_search_config = faiss_search_config or {}
        # Persist flag is configured upstream from privacy_mode; avoid redundant bool() casts for clarity
        self.persist_query_crops = persist_query_crops
        self.augmentation_mode = augmentation_mode
        self.transient_batch_index = transient_batch_index
        self.feature_dim = int(feature_dim)
        self.tmp_dir = tmp_dir

        self.transient_index: Optional[TransientIndexManager] = None

    # ---------------- Privacy Guardrail ----------------
    def validate_privacy_vs_provider(self) -> None:
        """Enforce privacy guardrail against incompatible providers.

        When persist_query_crops is false, prohibit bruteforce because it requires
        persisted crops on disk. Raise with the exact policy message.
        """
        if not self.persist_query_crops and self.provider == 'bruteforce':
            raise RuntimeError(
                "Security mode active (persist_query_crops=false): bruteforce requires persisted crops. Cannot proceed. "
                "Configure provider=faiss or qdrant, or enable persist_query_crops."
            )

    # ---------------- Augmentation Phase ----------------
    def augment_with_batch(self, batch_vectors: np.ndarray, item_names: List[str]) -> None:
        """Augment according to mode.

        - persistent_query_index: add to base provider
        - transient_query_index: build a side FAISS index
        """
        mode = (self.augmentation_mode or 'persistent_query_index').lower()
        if mode not in {"persistent_query_index", "transient_query_index"}:
            logger.warning("Unknown augmentation_mode '%s'. Falling back to transient_query_index.", mode)
            mode = "transient_query_index"

        if mode == "persistent_query_index":
            self._augment_persistent(batch_vectors, item_names)
        else:
            self._augment_transient(batch_vectors, item_names)

    def _augment_persistent(self, batch_vectors: np.ndarray, item_names: List[str]) -> None:
        """Persistent augmentation with deduplication and IVF fallback."""
        if self.provider == 'faiss':
            if not isinstance(self.base_manager, FaissIndexManager):
                raise RuntimeError("FAISS persistent augmentation requires a loaded FaissIndexManager.")
            if self.base_manager.faiss_index is None:
                logger.debug(
                    "FAISS base index is None; initializing (type=%s, sharded=%s).",
                    getattr(self.base_manager, "index_type", "unknown"),
                    getattr(self.base_manager, "_sharded", False),
                )
                self.base_manager._ensure_active_index_initialized()
                active_desc = None
                if getattr(self.base_manager, "_sharded", False):
                    sds = list(getattr(self.base_manager, "shard_descriptors", []) or [])
                    active_idx = int(getattr(self.base_manager, "active_shard_idx", 0))
                    if sds and 0 <= active_idx < len(sds):
                        active_desc = sds[active_idx]
                active_index_str = (
                    str(active_desc.index_path)
                    if (active_desc is not None and hasattr(active_desc, "index_path"))
                    else str(getattr(self.base_manager, "index_path", "n/a"))
                )
                logger.info(
                    "FAISS initialized: type=%s, sharded=%s, active_index=%s, shard_capacity=%s",
                    getattr(self.base_manager, "index_type", "unknown"),
                    getattr(self.base_manager, "_sharded", False),
                    active_index_str,
                    str(getattr(self.base_manager, "total_indexes_per_file", None)),
                )
            # IVF training policy: if untrained, switch to transient when allowed; otherwise abort
            if self.base_manager.index_type == 'ivf' and not self.base_manager.faiss_index.is_trained:
                msg = "Base index is IVF and untrained; training during search run is disabled."
                if self.transient_batch_index:
                    logger.info("%s Switching this run to transient augmentation.", msg)
                    self._augment_transient(batch_vectors, item_names)
                    return
                raise RuntimeError("FAISS IVF index is untrained; training during search run is disabled. Use transient augmentation or pre-train.")
            # Deduplicate by name/ID against existing mapping
            keep_vecs: List[np.ndarray] = []
            keep_names: List[str] = []
            for v, n in zip(batch_vectors, item_names):
                try:
                    if not self.base_manager.is_path_indexed(Path(n)):
                        keep_vecs.append(np.asarray(v))
                        keep_names.append(n)
                except (KeyError, OSError, RuntimeError) as e:
                    # Be conservative on known/expected errors (e.g., mapping not ready, IO issues)
                    logger.warning("is_path_indexed check failed for '%s'; keeping for augmentation. Error: %s", n, e)
                    keep_vecs.append(np.asarray(v))
                    keep_names.append(n)
                except Exception as e:
                    # Unknown errors should be visible to aid debugging
                    logger.warning("Unexpected error during is_path_indexed for '%s': %s", n, e)
                    keep_vecs.append(np.asarray(v))
                    keep_names.append(n)
            logger.debug(
                "Persistent FAISS augmentation: input=%d, new=%d, duplicates=%d",
                int(batch_vectors.shape[0]),
                int(len(keep_names)),
                int(batch_vectors.shape[0] - len(keep_names)),
            )
            if not keep_names:
                logger.info("Persistent FAISS augmentation: all %d items were duplicates; nothing to add.", batch_vectors.shape[0])
                return
            to_add = np.vstack(keep_vecs).astype(batch_vectors.dtype, copy=False)
            try:
                if getattr(self.base_manager, "_sharded", False):
                    logger.debug(
                        "FAISS add: adding %d vector(s) to active shard (capacity=%s)",
                        int(to_add.shape[0]),
                        str(getattr(self.base_manager, "total_indexes_per_file", None)),
                    )
                else:
                    logger.debug(
                        "FAISS add: adding %d vector(s) to legacy single index.",
                        int(to_add.shape[0]),
                    )
            except Exception as e:
                # Avoid masking unexpected attribute/state errors during logging
                logger.debug("Could not log FAISS add details: %s", e)
            self.base_manager.add_vectors_to_index(to_add, [Path(n) for n in keep_names])
            self.base_manager.save_index()
            logger.info("Added %d vectors to main corpus (provider=faiss); mapping updated.", to_add.shape[0])
            total_items = int(self.base_manager.get_total_indexed_items())
            if getattr(self.base_manager, "_sharded", False):
                sds = list(getattr(self.base_manager, "shard_descriptors", []) or [])
                num_shards = int(len(sds))
                active_idx = int(getattr(self.base_manager, "active_shard_idx", 0))
                active = sds[active_idx] if (sds and 0 <= active_idx < len(sds)) else None
                active_size = int(active.index.ntotal) if (active and getattr(active, "index", None)) else None
                active_shard_label = (f"s{active.shard_id:04d}" if active else "n/a")
                logger.info(
                    "FAISS post-add: total_items=%d, shards=%d, active_shard=%s, active_shard_size=%s",
                    total_items,
                    num_shards,
                    active_shard_label,
                    (str(active_size) if active_size is not None else "n/a"),
                )
            else:
                logger.info("FAISS post-add: total_items=%d (legacy single index)", total_items)
        elif self.provider == 'qdrant':
            if QdrantManager is None:
                raise RuntimeError("QdrantManager unavailable for persistent augmentation.")
            # If base manager is unavailable (e.g., embedded storage locked or init failure),
            # degrade persistent augmentation to transient when allowed. This mirrors IVF governance
            # for FAISS and prevents hard failure during runs where provider fallback occurred.
            if self.base_manager is None:
                if self.transient_batch_index:
                    logger.info(
                        "Persistent augmentation requested but Qdrant base manager is unavailable; "
                        "switching this run to transient augmentation."
                    )
                    self._augment_transient(batch_vectors, item_names)
                    return
                raise RuntimeError(
                    "Qdrant base manager not provided for persistent augmentation and transient fallback disabled."
                )
            # Deduplicate using QdrantManager.is_path_indexed
            filtered: List[Tuple[np.ndarray, str]] = []
            for v, n in zip(batch_vectors, item_names):
                try:
                    if not self.base_manager.is_path_indexed(Path(n)):
                        filtered.append((np.asarray(v), n))
                except Exception:
                    filtered.append((np.asarray(v), n))
            if not filtered:
                logger.info(
                    "Persistent Qdrant augmentation: all %d items were duplicates; nothing to upsert.",
                    batch_vectors.shape[0]
                )
                return
            try:
                from qdrant_client.http.models import PointStruct
                import uuid
                vecs = np.vstack([fv for fv, _ in filtered])
                names = [fn for _, fn in filtered]
                points = [
                    PointStruct(id=str(uuid.uuid4()), vector=v.tolist(), payload={"image_path": str(Path(n).resolve())})
                    for v, n in zip(vecs, names)
                ]
                op = self.base_manager.client.upsert(
                    collection_name=self.base_manager.collection_name, points=points, wait=True
                )
                logger.info(
                    "Added %d vectors to main corpus (provider=qdrant); payloads updated.", len(points)
                )
            except Exception as e:
                raise RuntimeError(f"Qdrant persistent augmentation failed: {e}")
        else:
            raise RuntimeError("Persistent augmentation is unsupported for provider='bruteforce'.")

    def _augment_transient(self, batch_vectors: np.ndarray, item_names: List[str]) -> None:
        """Build a per-run FAISS side index and insert the batch vectors.

        Independent of the base provider; uses FLAT or HNSW based on config. Can
        create an on-disk temporary index when tmp_dir is provided for large batches.
        """
        if self.provider not in {"faiss", "qdrant", "bruteforce"}:  # transient index independent of base
            raise RuntimeError(f"Unsupported provider '{self.provider}' for transient mode")
        index_type = 'hnsw' if (self.faiss_search_config.get('index_type') == 'hnsw') else 'flat'
        hnsw_m = self.faiss_search_config.get('hnsw_m', 32)
        hnsw_efc = self.faiss_search_config.get('hnsw_ef_construction', 40)
        use_on_disk = self.transient_batch_index and self.tmp_dir
        self.transient_index = TransientIndexManager(
            feature_dim=self.feature_dim,
            index_type=index_type,
            hnsw_m=hnsw_m,
            hnsw_efconstruction=hnsw_efc,
            use_on_disk=use_on_disk,
            temp_dir=self.tmp_dir,
        )
        self.transient_index.add(batch_vectors, item_names)
        logger.info(
            "Created transient batch index with %d vectors (FAISS %s%s); will be discarded after run.",
            batch_vectors.shape[0], index_type, " on-disk" if use_on_disk else " in-memory"
        )

    # ---------------- Search Merge Helpers ----------------
    def merged_search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        include_query_image_to_result: bool,
        query_parent_document_name: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Run search against base provider and transient index (if any) and merge results."""
        base_res: List[Tuple[str, float]] = []
        if self.base_manager is not None and self.base_manager.is_index_loaded_and_ready():
            # Dispatch to FAISS or Qdrant
            try:
                base_res = self.base_manager.search_similar_images(
                    query_vector=query_vector,
                    top_k=top_k,
                    ivf_nprobe_search=self.faiss_search_config.get('ivf_nprobe_search'),
                    hnsw_efsearch_search=self.faiss_search_config.get('hnsw_efsearch_search'),
                )
            except Exception as e:
                logger.warning("Base provider search failed: %s", e)
                base_res = []

        transient_res: List[Tuple[str, float]] = []
        if self.transient_index is not None and self.transient_index.is_ready():
            try:
                transient_res = self.transient_index.search(
                    query_vector=query_vector,
                    top_k=top_k,
                    hnsw_efsearch=self.faiss_search_config.get('hnsw_efsearch_search'),
                )
            except Exception as e:
                logger.warning("Transient index search failed: %s", e)
                transient_res = []

        return QueryMerger.merge_results(
            base_results=base_res,
            transient_results=transient_res,
            top_k=top_k,
            include_query_image_to_result=include_query_image_to_result,
            query_parent_document_name=query_parent_document_name,
        )
