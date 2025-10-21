"""Unified vector database manager interface and capacity parsing utilities.

This module defines a provider-agnostic interface for vector database managers
and a robust helper for parsing the standardized partition capacity setting.

Goals
- Provide a single, consistent interface that FAISS and Qdrant managers
  implement, simplifying orchestration and reducing maintenance overhead.
- Standardize configuration using the provider-agnostic key "partition_capacity"
  while preserving backward compatibility with legacy keys used in existing
  deployments (e.g., "total_indexes_per_file" for FAISS and
  "max_points_per_collection" for Qdrant).

Design notes
- The ``parse_partition_capacity`` helper prioritizes the new, unified key and
  gracefully falls back to legacy keys when necessary. It also supports a set of
  disable keywords ("none", "legacy", "single", etc.) to explicitly opt out of
  sharding.
- Defaults are provider-specific and injected by the caller to avoid surprising
  behavioral changes: FAISS continues to default to 250_000 items per shard and
  Qdrant continues to default to 500_000 points per collection unless an
  explicit configuration overrides this via "partition_capacity".

All public functions and classes include precise type hints and follow Google
Style docstrings. Only non-obvious decisions are commented with the "why".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import logging


logger = logging.getLogger(__name__)


class VectorDBManager(ABC):
    """Abstract base class for vector database managers.

    Concrete implementations must provide compatible behavior for core index
    lifecycle operations including loading, saving, building, searching, and
    maintenance. This unifies higher-level orchestration regardless of the
    underlying provider (FAISS, Qdrant, etc.).

    Implementations should expose a provider-specific notion of partitioning
    controlled via the provider-agnostic configuration key ``partition_capacity``.

    Methods are intentionally provider-agnostic to avoid leaking storage details
    into orchestration layers.
    """

    # -------------------------
    # Discovery / lifecycle
    # -------------------------
    @abstractmethod
    def load_index(self) -> bool:
        """Load or validate the index/storage for search operations.

        Returns:
            bool: True if the index or collections are accessible (even if
            empty), otherwise False.
        """

    @abstractmethod
    def save_index(self) -> None:
        """Persist in-memory index updates if applicable.

        Providers with external persistence (e.g., Qdrant server) may treat this
        as a no-op.
        """

    @abstractmethod
    def is_index_loaded_and_ready(self) -> bool:
        """Return whether the index is available and contains data to search."""

    @abstractmethod
    def get_total_indexed_items(self) -> int:
        """Return the total number of vectors/points across all partitions."""

    # -------------------------
    # Build / update
    # -------------------------
    @abstractmethod
    def build_index_from_folder(
        self,
        feature_extractor: Any,
        image_folder: str,
        batch_size: int = 32,
        force_rebuild: bool = False,
        scan_subfolders: bool = False,
        **kwargs: Any,
    ) -> int:
        """Build or update the index from a folder of images.

        Args:
            feature_extractor: Provider-agnostic feature extractor.
            image_folder (str): Root folder containing images.
            batch_size (int): Batch size for extraction or upsert.
            force_rebuild (bool): When True, clear existing data before build.
            scan_subfolders (bool): Whether to recurse into subdirectories.
            **kwargs: Provider-specific options retained for compatibility.

        Returns:
            int: 0 on success, non-zero on failure.
        """

    # -------------------------
    # Query
    # -------------------------
    @abstractmethod
    def search_similar_images(
        self,
        query_vector: np.ndarray,
        top_k: int,
        **kwargs: Any,
    ) -> List[Tuple[str, float]]:
        """Search for nearest neighbors and return (path, score) pairs.

        Args:
            query_vector (np.ndarray): Normalized query vector.
            top_k (int): Number of nearest neighbors to return.
            **kwargs: Provider-specific search parameters retained for
                compatibility.

        Returns:
            List[Tuple[str, float]]: Results sorted by descending similarity.
        """

    # -------------------------
    # Maintenance
    # -------------------------
    @abstractmethod
    def clear_index(self, save_empty: bool = True) -> None:
        """Clear or reset the index/collections to an empty state.

        Args:
            save_empty (bool): Provider-specific hint to persist an empty state
                for file-based stores. Ignored by providers with external
                persistence models.
        """

    @abstractmethod
    def is_path_indexed(self, image_path: Path | str) -> bool:
        """Return True if the given path is already indexed."""


def parse_partition_capacity(config: Dict[str, Any], provider_default: int) -> Optional[int]:
    """Parse a provider-agnostic partition capacity from configuration.

    Precedence (highest to lowest):
      1. ``partition_capacity`` (unified key)
      2. Provider legacy keys: ``total_indexes_per_file`` (FAISS),
         ``max_points_per_collection`` (Qdrant)
      3. Provider-specific default (supplied by caller)

    The following values disable partitioning and force legacy single-partition
    behavior: None, "none", "null", "legacy", "single", or an empty string.

    Args:
        config (Dict[str, Any]): Configuration mapping for the provider.
        provider_default (int): Default capacity to use when no explicit setting
            is provided. Kept provider-specific to avoid unexpected changes in
            behavior for existing deployments.

    Returns:
        Optional[int]: Parsed positive integer capacity, or None when
        partitioning is disabled.

    Raises:
        ValueError: If a provided capacity value is invalid (e.g., negative)
            after normalization.
    """
    disable_keywords = {"", "none", "null", "legacy", "single"}

    def _normalize(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, str):
            v = value.strip().lower()
            if v in disable_keywords:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid partition capacity value: {value!r}")
        if isinstance(value, (int, float)):
            return int(value)
        raise ValueError(f"Unrecognized partition capacity type: {type(value).__name__}")

    # 1) Unified key takes precedence
    if "partition_capacity" in config:
        cap = _normalize(config.get("partition_capacity"))
        if cap is None:
            return None
        if cap <= 0:
            raise ValueError("partition_capacity must be a positive integer or a disabling keyword")
        return cap

    # 2) Legacy keys for backward compatibility (emit deprecation warning)
    for legacy_key in ("total_indexes_per_file", "max_points_per_collection"):
        if legacy_key in config:
            cap = _normalize(config.get(legacy_key))
            if cap is None:
                return None
            if cap <= 0:
                raise ValueError(f"{legacy_key} must be a positive integer or a disabling keyword")
            logger.warning(
                "Deprecated config key '%s' detected; prefer 'partition_capacity'. Applying backward-compatible mapping.",
                legacy_key,
            )
            return cap

    # 3) Default when no key is provided
    return provider_default
