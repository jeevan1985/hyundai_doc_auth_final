# core_engine/image_similarity_system/qdrant_manager.py
"""
Qdrant manager with optional sharded collections.

This module manages creation, discovery, upsert, search, and maintenance of
Qdrant vector collections for image similarity. It preserves the legacy public
API while adding support for sharding: when a configured capacity is reached,
new points are inserted into a new collection suffixed by an ordinal shard
number. Search queries span all discovered collections and return a merged
global top-k result.

Behavior overview
- Legacy mode (default): when ``max_points_per_collection`` is not configured or
  is None, exactly one collection is used with the legacy base name
  ``{collection_name_stem}_{model}``.
- Sharded mode: when ``max_points_per_collection`` is configured (integer N),
  the manager creates or discovers collections named
  ``{collection_name_stem}_{model}_sNNNN`` and rolls over to the next shard when
  the active one reaches/exceeds N points. The unsharded base collection, if it
  exists, is included as well for backward compatibility and treated as a valid
  shard for search and total counts.

Collections are auto-discovered at startup by listing available collections and
filtering on the configured naming base and shard pattern. This enables portable
index import when collections already exist in the target Qdrant instance.

Docstrings follow Google Style and are PEP 257 compliant. All functions include
precise type hints. Comments explain non-obvious design decisions (the "why").
"""

from __future__ import annotations

# =============================================================================
# 1. Standard Library Imports
# =============================================================================
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import re

# =============================================================================
# 2. Third-Party Library Imports
# =============================================================================
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    UpdateStatus,
    VectorParams,
)

# =============================================================================
# 3. Application-Specific Imports
# =============================================================================
from .feature_extractor import FeatureExtractor
from .utils import image_path_generator
from .vector_db_base import VectorDBManager, parse_partition_capacity

# =============================================================================
# 4. Module-level Logger Setup
# =============================================================================
logger = logging.getLogger(__name__)


# =============================================================================
# 5. Data Structures
# =============================================================================
@dataclass
class _CollectionDesc:
    """Descriptor for a Qdrant collection used as a shard."""

    name: str
    shard_id: Optional[int]  # None for legacy base collection without suffix


# =============================================================================
# 6. QdrantManager Class Definition
# =============================================================================
class QdrantManager(VectorDBManager):
    """Manage Qdrant collections for vector similarity with optional sharding.

    Two operational modes are supported:
    - Legacy (single-collection) mode when ``max_points_per_collection`` is None.
    - Sharded mode when ``max_points_per_collection`` is an integer N.

    Discovery policy
    - The manager enumerates available collections and filters by the naming base
      ``{collection_name_stem}_{model}`` (base) and ``..._sNNNN`` (shards). When
      sharding is enabled, the base collection, if present, is also included to
      maintain backward compatibility with previously created indexes.

    Public compatibility
    - The public API is preserved. The attribute ``collection_name`` points to
      the active collection (highest ordinal shard when sharded; base otherwise).
      The property ``collection_names`` lists all discovered collections used
      by this manager for logging and multi-collection search.

    Args:
        feature_dim: Vector dimensionality.
        collection_name_stem: Stem for base collection naming.
        model_name: Feature extraction model name, lower-cased for naming.
        qdrant_config: Provider configuration mapping (embedded/server modes).
        project_root_path: Project root for resolving on-disk paths.
    """

    def __init__(
        self,
        feature_dim: int,
        collection_name_stem: str,
        model_name: str,
        qdrant_config: Dict[str, Any],
        project_root_path: Optional[Path] = None,
    ) -> None:
        self.feature_dim: int = int(feature_dim)
        self.model_name: str = str(model_name).lower()
        self.collection_name_base: str = f"{collection_name_stem}_{self.model_name}"
        self.qdrant_config: Dict[str, Any] = qdrant_config
        self.project_root_path: Path = project_root_path or Path.cwd()

        # Standardized partition capacity parsing with backward compatibility.
        # partition_capacity takes precedence; legacy key (max_points_per_collection)
        # remains supported for existing configurations.
        self.partition_capacity: Optional[int] = parse_partition_capacity(qdrant_config, provider_default=500000)
        # Backward-compat attribute retained for existing call sites and logs
        self.max_points_per_collection = self.partition_capacity

        self.client: QdrantClient = self._initialize_client()

        # Opt-in lazy connection to avoid network calls on initialization.
        # When enabled, discovery and ensure/create are deferred until first use
        # (e.g., load_index or build_index_from_folder). Default remains eager
        # to preserve legacy behavior.
        self.lazy_connect: bool = bool(self.qdrant_config.get("lazy_connect", False))

        # Discovery of collections (eager by default)
        self._collections: List[_CollectionDesc] = []
        if not self.lazy_connect:
            self._discover_collections()
            if not self._collections:
                # Create base or shard 1 according to mode
                if self.partition_capacity is None:
                    self._ensure_collection_exists(self.collection_name_base)
                    self._collections = [_CollectionDesc(name=self.collection_name_base, shard_id=None)]
                else:
                    first = f"{self.collection_name_base}_s0001"
                    self._ensure_collection_exists(first)
                    self._collections = [_CollectionDesc(name=first, shard_id=1)]
        # Active collection name (fallback to base when lazy and undiscovered)
        self.active_collection_name: str = (
            self._collections[-1].name if self._collections else self.collection_name_base
        )
        self.collection_name: str = self.active_collection_name  # legacy attribute

        logger.debug(
            "QdrantManager initialized (sharded=%s). Base=%s, active=%s, total=%d",
            self.partition_capacity is not None,
            self.collection_name_base,
            self.active_collection_name,
            len(self._collections),
        )

    # ------------------------------------------------------------------
    # Client initialization
    # ------------------------------------------------------------------
    def _initialize_client(self) -> QdrantClient:
        """Initialize Qdrant client for embedded or server mode.

        Returns:
            QdrantClient: Ready-to-use client instance.
        """
        location = self.qdrant_config.get("location")
        if location:
            if location == ":memory:":
                return QdrantClient(location=":memory:")
            db_path = Path(str(location))
            if not db_path.is_absolute():
                db_path = (self.project_root_path / db_path).resolve()
            db_path.mkdir(parents=True, exist_ok=True)
            return QdrantClient(path=str(db_path))

        # Server mode (prefer env variables)
        host = os.getenv("QDRANT_HOST", self.qdrant_config.get("host", "localhost"))
        port = int(os.getenv("QDRANT_PORT", self.qdrant_config.get("port", 6333)))
        grpc_port = int(os.getenv("QDRANT_GRPC_PORT", self.qdrant_config.get("grpc_port", 6334)))
        prefer_grpc = bool(self.qdrant_config.get("prefer_grpc", False))
        https_enable = bool(self.qdrant_config.get("https_enable", False))
        api_key = os.getenv("QDRANT_API_KEY", self.qdrant_config.get("api_key"))
        return QdrantClient(
            host=host,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            api_key=api_key,
            https=https_enable,
        )

    # ------------------------------------------------------------------
    # Discovery and creation
    # ------------------------------------------------------------------
    def _discover_collections(self) -> None:
        """Discover relevant collections according to naming base and sharding.

        The method populates ``self._collections`` with zero or more descriptors.
        """
        try:
            resp = self.client.get_collections()
            names: List[str] = [c.name for c in getattr(resp, "collections", [])]
        except Exception as e:
            logger.warning("Failed to list collections: %s", e)
            names = []

        base = self.collection_name_base
        pattern = re.compile(rf"^{re.escape(base)}_s(\d{{4}})$")
        found: List[_CollectionDesc] = []
        base_exists = False
        for n in names:
            if n == base:
                base_exists = True
                found.append(_CollectionDesc(name=n, shard_id=None))
            else:
                m = pattern.match(n)
                if m:
                    sid = int(m.group(1))
                    found.append(_CollectionDesc(name=n, shard_id=sid))
        # Sort: base first (shard_id None), then shards ascending
        found.sort(key=lambda d: (-1 if d.shard_id is None else d.shard_id))
        self._collections = found

        if found:
            logger.info(
                "Qdrant discovery: %d collection(s) for base '%s'. First=%s",
                len(found),
                base,
                found[0].name,
            )
        else:
            logger.info("Qdrant discovery: no collections found for base '%s'", base)

    def _ensure_collection_exists(self, collection_name: str) -> None:
        """Ensure a collection with correct vector params and payload index exists.

        Args:
            collection_name: Name of the collection to validate or create.
        """
        try:
            self.client.get_collection(collection_name=collection_name)
            return
        except Exception:
            pass

        # Create collection with vector params
        distance_metric_str = str(self.qdrant_config.get("distance_metric", "Cosine")).upper()
        try:
            distance_metric = getattr(Distance, distance_metric_str)
        except AttributeError:
            distance_metric = Distance.COSINE
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self.feature_dim, distance=distance_metric),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, always_ram=True
                )
            )
            if self.qdrant_config.get("enable_quantization")
            else None,
            hnsw_config=models.HnswConfigDiff(on_disk=self.qdrant_config.get("on_disk_hnsw_indexing"))
            if self.qdrant_config.get("on_disk_hnsw_indexing") is not None
            else None,
        )
        # Ensure payload index on image_path for fast deduplication
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="image_path",
                field_schema=PayloadSchemaType.KEYWORD,
                wait=True,
            )
        except Exception:
            # Index may already exist; ignore errors
            pass
        logger.info("Created Qdrant collection '%s'.", collection_name)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def collection_names(self) -> List[str]:
        """List names of all discovered collections used by this manager."""
        return [d.name for d in self._collections]

    # ------------------------------------------------------------------
    # Duplicate check
    # ------------------------------------------------------------------
    def is_path_indexed(self, image_path: Path | str) -> bool:
        """Check if a given path exists in any of the collections.

        Args:
            image_path: Path to check for existence.

        Returns:
            bool: True if found, otherwise False.
        """
        resolved = str(Path(image_path).resolve())
        for desc in self._collections:
            try:
                points, _ = self.client.scroll(
                    collection_name=desc.name,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="image_path", match=MatchValue(value=resolved))]
                    ),
                    limit=1,
                    with_payload=False,
                    with_vectors=False,
                )
                if points:
                    return True
            except Exception:
                continue
        return False

    # ------------------------------------------------------------------
    # Load/readiness
    # ------------------------------------------------------------------
    def load_index(self) -> bool:
        """Validate connectivity and presence of at least one collection.

        Returns:
            bool: True if collections are accessible (even when empty), else False.
        """
        try:
            # Refresh discovery to include any externally created shards
            self._discover_collections()
            if not self._collections:
                # In lazy_connect mode, do not auto-create collections here; caller may auto-build.
                return False
            # Ready if any collection exists; data readiness checked separately
            return True
        except Exception as e:
            logger.error("Failed to load/discover Qdrant collections: %s", e, exc_info=False)
            return False

    def is_index_loaded_and_ready(self) -> bool:
        """Return True if at least one collection contains points."""
        return self.get_total_indexed_items() > 0

    def get_total_indexed_items(self) -> int:
        """Return total points across all collections."""
        total = 0
        for desc in self._collections:
            try:
                count = self.client.count(collection_name=desc.name, exact=False)
                total += int(getattr(count, "count", 0))
            except Exception:
                continue
        return total

    def save_index(self) -> None:
        """No-op for Qdrant (persistence managed by engine)."""
        logger.debug("QdrantManager.save_index() called; no action required.")

    # ------------------------------------------------------------------
    # Index building/upsert
    # ------------------------------------------------------------------
    def build_index_from_folder(
        self,
        feature_extractor: FeatureExtractor,
        image_folder: str,
        batch_size: int = 32,
        force_rebuild: bool = False,
        scan_subfolders: bool = False,
        **kwargs: Any,
    ) -> int:
        """Build or update Qdrant collections from a folder of images.

        Args:
            feature_extractor: Initialized feature extractor.
            image_folder: Database image folder.
            batch_size: Extraction batch size.
            force_rebuild: When True, delete and recreate collections before indexing.
            scan_subfolders: Recurse into subdirectories for discovery.
            **kwargs: Unused, kept for compatibility with FAISS call sites.

        Returns:
            int: 0 on success, 1 on failure.
        """
        if force_rebuild:
            logger.info("force_rebuild=True -> clearing index (all collections) before build.")
            self.clear_index()
        # Ensure at least one collection exists before upsert when running in lazy mode
        if not self._collections:
            try:
                if self.partition_capacity is None:
                    self._ensure_collection_exists(self.collection_name_base)
                    self._collections = [_CollectionDesc(name=self.collection_name_base, shard_id=None)]
                else:
                    first = f"{self.collection_name_base}_s0001"
                    self._ensure_collection_exists(first)
                    self._collections = [_CollectionDesc(name=first, shard_id=1)]
                self.active_collection_name = self._collections[-1].name
                self.collection_name = self.active_collection_name
            except Exception as e:
                logger.error("Failed to ensure initial collection before build: %s", e)
                return 1

        new_image_paths = self._get_new_image_paths_to_index(Path(image_folder), scan_subfolders)
        if not new_image_paths:
            logger.info("No new images found to add to Qdrant.")
            return 0

        logger.info("Starting indexing of %d images into Qdrant (may span shards).", len(new_image_paths))
        try:
            for i in tqdm(range(0, len(new_image_paths), batch_size), desc="Indexing to Qdrant", unit="batch"):
                batch_paths = new_image_paths[i : i + batch_size]
                batch_features = feature_extractor.extract_features(batch_paths)
                if batch_features is None or batch_features.shape[0] == 0:
                    continue
                self._upsert_with_shard_rollover(batch_features, batch_paths)
            return 0
        except Exception as e:
            logger.critical("Critical error during Qdrant indexing: %s", e, exc_info=True)
            return 1

    def _upsert_with_shard_rollover(self, vectors: np.ndarray, paths: Sequence[Path]) -> None:
        """Upsert a batch of vectors with shard rollover according to capacity.

        Args:
            vectors: 2D array of vectors.
            paths: Corresponding image paths.
        """
        remaining_vecs = vectors
        remaining_paths = list(paths)
        while remaining_vecs.shape[0] > 0:
            active = self._collections[-1]
            # Ensure active collection exists
            self._ensure_collection_exists(active.name)
            # Determine room in active collection when sharding is enabled
            room = None
            if self.partition_capacity is not None:
                try:
                    c = self.client.count(collection_name=active.name, exact=False)
                    current = int(getattr(c, "count", 0))
                except Exception:
                    current = 0
                room = max(0, int(self.partition_capacity) - current)
                if room <= 0:
                    # Roll to a new shard
                    next_id = (active.shard_id + 1) if active.shard_id is not None else 1
                    name = f"{self.collection_name_base}_s{next_id:04d}"
                    self._ensure_collection_exists(name)
                    self._collections.append(_CollectionDesc(name=name, shard_id=next_id))
                    self.active_collection_name = name
                    continue
            # Take portion that fits (or all if room is None)
            take = remaining_vecs.shape[0] if room is None else min(room, remaining_vecs.shape[0])
            if take == 0:
                # Capacity zero -> force rollover
                next_id = (active.shard_id + 1) if active.shard_id is not None else 1
                name = f"{self.collection_name_base}_s{next_id:04d}"
                self._ensure_collection_exists(name)
                self._collections.append(_CollectionDesc(name=name, shard_id=next_id))
                self.active_collection_name = name
                continue
            vecs = remaining_vecs[:take]
            pths = remaining_paths[:take]
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=v.tolist(),
                    payload={"image_path": str(Path(p).resolve())},
                )
                for v, p in zip(vecs, pths)
            ]
            op = self.client.upsert(collection_name=active.name, wait=True, points=points)
            if getattr(op, "status", UpdateStatus.COMPLETED) != UpdateStatus.COMPLETED:
                logger.error("Qdrant upsert failed for collection '%s' (status=%s)", active.name, getattr(op, "status", None))
            # Consume processed
            remaining_vecs = remaining_vecs[take:]
            remaining_paths = remaining_paths[take:]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search_similar_images(self, query_vector: np.ndarray, top_k: int, **kwargs: Any) -> List[Tuple[str, float]]:
        """Search across all collections and merge a global top-k.

        Args:
            query_vector: 1D or single-row 2D query vector (normalized).
            top_k: Number of neighbors to retrieve.
            **kwargs: Unused, kept for compatibility with FAISS call sites.

        Returns:
            List[Tuple[str, float]]: (image_path, score) sorted by score descending.
        """
        if query_vector.ndim == 2:
            q = query_vector[0].tolist()
        else:
            q = query_vector.tolist()

        if not self.is_index_loaded_and_ready():
            logger.error("Qdrant index is not ready or empty.")
            return []

        merged: List[Tuple[str, float]] = []
        for desc in self._collections:
            try:
                hits = self.client.search(
                    collection_name=desc.name, query_vector=q, limit=top_k, with_payload=True
                )
            except Exception:
                continue
            for h in hits:
                try:
                    p = h.payload.get("image_path") if h.payload else None
                    if p is None:
                        continue
                    merged.append((str(p), float(h.score)))
                except Exception:
                    continue
        # Global top-k with deterministic tie-breaker (score desc, path asc)
        merged.sort(key=lambda t: (-t[1], t[0]))
        return merged[:top_k]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def clear_index(self, save_empty: bool = True) -> None:  # NOSONAR save_empty for API parity
        """Delete all discovered collections and recreate an empty base/shard.

        Args:
            save_empty: Ignored; kept for FAISS parity.
        """
        logger.info("Clearing Qdrant collections for base '%s'", self.collection_name_base)
        for desc in self._collections:
            try:
                self.client.delete_collection(collection_name=desc.name)
            except Exception:
                pass
        # Recreate according to mode
        self._collections = []
        if self.partition_capacity is None:
            self._ensure_collection_exists(self.collection_name_base)
            self._collections = [_CollectionDesc(name=self.collection_name_base, shard_id=None)]
        else:
            first = f"{self.collection_name_base}_s0001"
            self._ensure_collection_exists(first)
            self._collections = [_CollectionDesc(name=first, shard_id=1)]
        self.active_collection_name = self._collections[-1].name
        self.collection_name = self.active_collection_name

    # ------------------------------------------------------------------
    # Discovery of new images versus current corpus
    # ------------------------------------------------------------------
    def _get_new_image_paths_to_index(self, image_folder_path: Path, scan_subfolders: bool = False) -> List[Path]:
        """Scan for image files and return those not present in any collection.

        Args:
            image_folder_path: Root for discovery.
            scan_subfolders: Whether to recurse into subdirectories.

        Returns:
            List[Path]: Absolute image paths that are not yet indexed.
        """
        new_paths: List[Path] = []
        for image_file_path in image_path_generator(image_folder_path, scan_subfolders=scan_subfolders):
            try:
                if not self.is_path_indexed(image_file_path):
                    new_paths.append(Path(image_file_path).resolve())
            except Exception:
                new_paths.append(Path(image_file_path).resolve())
        logger.info(
            "Scan complete. %d new images found under '%s' to add to Qdrant.",
            len(new_paths),
            image_folder_path,
        )
        return new_paths

