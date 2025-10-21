# core_engine/image_similarity_system/faiss_manager.py
"""
FAISS index manager with optional sharded indexing.

This module manages creation, loading, saving, and searching of FAISS indexes
for image similarity. It supports both legacy single-file mode and sharded
mode, where vectors are split across multiple index artifacts according to a
configurable capacity. Shards are auto-discovered from disk, enabling portable
index import (copy in shard files and they will be used automatically).

Behavior overview
- Legacy mode (default): when ``total_indexes_per_file`` is not configured or
  is None, a single index file and a single ID→path mapping file are used.
- Sharded mode: when ``total_indexes_per_file`` is an integer N, the manager
  rolls to a new shard once the active shard reaches N items. On startup, shard
  files are auto-discovered using the configured naming pattern and loaded.

Search executes across all loaded shards, merging the per-shard top-k results
into a global top-k according to score, with deterministic tie-breaking.

Design notes
- The manager preserves the public API from prior versions, including:
  load_index, save_index, add_vectors_to_index, get_total_indexed_items,
  search_similar_images, clear_index. It also continues to expose ``index_path``
  (active shard path in sharded mode) and introduces ``index_identifiers``
  (all shard index paths) for logging purposes.
- For IVF and HNSW, search-time parameters (nprobe, efSearch) are applied per
  shard during search and restored to their original values after each shard
  query to avoid side effects.
- Indexes are eagerly loaded in sharded mode during ``load_index`` for
  simplicity and deterministic validation of compatibility (dimension checks).
  This is documented in the class docstring.

Docstrings follow Google Style and are PEP 257 compliant. All functions include
precise type hints. Comments explain non-obvious design decisions (the "why").
"""

from __future__ import annotations

# =============================================================================
# 1. Standard Library Imports
# =============================================================================
import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# =============================================================================
# 2. Third-Party Library Imports
# =============================================================================
import faiss  # type: ignore
import numpy as np
from tqdm import tqdm

# =============================================================================
# 3. Application-Specific Imports
# =============================================================================
from .feature_extractor import FeatureExtractor
from .gpu_utils import clear_gpu_memory
from .constants import (
    DEFAULT_HNSW_EF_CONSTRUCTION,
    DEFAULT_HNSW_M,
    DEFAULT_IVF_NLIST,
)
from .utils import image_path_generator
from .vector_db_base import VectorDBManager, parse_partition_capacity

# =============================================================================
# 4. Module-level Logger Setup
# =============================================================================
logger = logging.getLogger(__name__)

# Module-level log de-duplication guards to prevent repeated info logs across
# multiple FaissIndexManager instances that reference the same index base.
# Key format: f"{output_directory}|{descriptive_core}"
_LOG_SEEN_EMPTY_INIT: Set[str] = set()
_LOG_SEEN_DISCOVERY_SUMMARY: Set[str] = set()
_LOG_SEEN_LOADED_SUMMARY: Set[str] = set()
# Prevent repeated adoption logs when legacy single index is adopted as s0001 multiple times
_LOG_SEEN_ADOPT_LEGACY: Set[str] = set()


# =============================================================================
# 5. Data Structures
# =============================================================================
@dataclass
class _ShardDescriptor:
    """Descriptor for a single FAISS shard and its mapping file.

    Attributes:
        shard_id: Ordinal shard number (1-based).
        index_path: Path to the FAISS index file for this shard.
        mapping_path: Path to the pickle mapping file for this shard.
        index: Loaded FAISS index instance for this shard, or None until loaded/initialized.
        id_to_path_map: Local ID→absolute-path mapping for this shard (IDs are 0..ntotal-1).
        modified: Whether this shard has been modified in this process and needs saving.
    """

    shard_id: int
    index_path: Path
    mapping_path: Path
    index: Optional[faiss.Index] = None
    id_to_path_map: Dict[int, str] = field(default_factory=dict)
    modified: bool = False


# =============================================================================
# 6. FaissIndexManager Class Definition
# =============================================================================
class FaissIndexManager(VectorDBManager):
    """Manage FAISS index lifecycle with optional sharded indexing.

    The manager supports two modes:
    - Legacy (single-file) mode when ``total_indexes_per_file`` is None.
    - Sharded mode when ``total_indexes_per_file`` is an integer N.

    Sharded auto-discovery
    - On initialization and load, the manager searches the output directory for
      files matching the naming pattern and validates shard compatibility
      (vector dimension, presence of mapping file). Valid shards are tracked in
      ``shard_descriptors``. Invalid pairs are ignored with a warning.

    Index loading policy (sharded mode)
    - Indexes are eagerly loaded during ``load_index``. This simplifies result
      merging and validates compatibility upfront. The overhead is amortized by
      fast in-process reuse and avoids repeated open/close costs across queries.

    Public compatibility
    - The public API is preserved. The attribute ``index_path`` points to the
      active shard's index path in sharded mode (highest ordinal). The property
      ``index_identifiers`` provides all shard index paths when sharded, or a
      single path list in legacy mode.

    Args:
        feature_dim: Dimensionality of the feature vectors.
        output_directory: Directory where index and mapping artifacts are stored.
        filename_stem: Base filename stem.
        index_type: Index type to use ("flat" | "ivf" | "hnsw").
        model_name: Model name used in filename composition.
        faiss_config: FAISS-specific configuration mapping.
        project_root_path: Project root for resolving relative output_directory.
    """

    def __init__(
        self,
        feature_dim: int,
        output_directory: str,
        filename_stem: str,
        index_type: str,
        model_name: str,
        faiss_config: Dict[str, Any],
        project_root_path: Optional[Path] = None,
    ) -> None:
        self.feature_dim: int = int(feature_dim)

        self.output_directory: Path = Path(output_directory)
        if not self.output_directory.is_absolute():
            base_path = project_root_path if project_root_path else Path.cwd()
            self.output_directory = (base_path / self.output_directory).resolve()
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.filename_stem: str = str(filename_stem)
        self.index_type: str = str(index_type).lower()
        self.model_name: str = str(model_name).lower()

        # Parameters from config
        self.ivf_nlist: int = int(faiss_config.get("ivf_nlist", DEFAULT_IVF_NLIST))
        self.hnsw_m: int = int(faiss_config.get("hnsw_m", DEFAULT_HNSW_M))
        self.hnsw_ef_construction: int = int(
            faiss_config.get("hnsw_ef_construction", DEFAULT_HNSW_EF_CONSTRUCTION)
        )
        # Internal: log guards to avoid duplicate info logs across repeated discovery/load calls
        self._logged_empty_shard_init: bool = False
        self._logged_discovery_summary: bool = False
        self._logged_loaded_summary: bool = False
        # Standardized partition capacity parsing with backward compatibility.
        # partition_capacity takes precedence; legacy keys (total_indexes_per_file)
        # remain supported for existing configurations.
        self.partition_capacity: Optional[int] = parse_partition_capacity(faiss_config, provider_default=250000)
        # Backward-compat attribute retained for existing call sites and logs
        self.total_indexes_per_file = self.partition_capacity

        # Derived naming core
        self._descriptive_core: str = f"{self.filename_stem}_{self.model_name}_{self.index_type}"

        # Legacy attributes maintained for compatibility
        self.index_path: Path = self.output_directory / f"{self._descriptive_core}.index"
        self.mapping_path: Path = self.output_directory / f"{self._descriptive_core}_mapping.pkl"

        # Active index handle (legacy or active shard in sharded mode)
        self.faiss_index: Optional[faiss.Index] = None
        # Legacy single-file mapping (in sharded mode, this is kept in the active shard descriptor only)
        self.id_to_path_map: Dict[int, str] = {}

        # Shard state
        self._sharded: bool = self.partition_capacity is not None
        self.shard_descriptors: List[_ShardDescriptor] = []
        self.active_shard_idx: int = 0  # index into shard_descriptors

        # Set of known absolute paths across all shards for deduplication
        self._known_paths_set: Set[str] = set()

        logger.debug(
            "FaissIndexManager initialized (index_type=%s, model=%s, sharded=%s).",
            self.index_type,
            self.model_name,
            self._sharded,
        )
        if self.index_type == "ivf":
            logger.debug("  IVF nlist: %d", self.ivf_nlist)
        elif self.index_type == "hnsw":
            logger.debug(
                "  HNSW M: %d, efConstruction: %d", self.hnsw_m, self.hnsw_ef_construction
            )

        # Prepare shard descriptors according to mode
        if self._sharded:
            self._discover_or_initialize_shards()
        else:
            # Legacy single-file mode: paths stay as set above
            pass

    # ---------------------------------------------------------------------
    # Shard helpers
    # ---------------------------------------------------------------------
    def _build_shard_paths(self, shard_id: int) -> Tuple[Path, Path]:
        """Return index/mapping paths for the given shard ordinal.

        Args:
            shard_id: 1-based shard ordinal.

        Returns:
            Tuple[Path, Path]: Index file path and mapping file path.
        """
        suffix = f"_s{shard_id:04d}"
        idx = self.output_directory / f"{self._descriptive_core}{suffix}.index"
        mp = self.output_directory / f"{self._descriptive_core}{suffix}_mapping.pkl"
        return idx, mp

    def _discover_or_initialize_shards(self) -> None:
        """Discover shards on disk by naming pattern or initialize an empty shard.

        The function searches for index files matching
        ``{stem}_{model}_{index_type}_s*.index`` and requires a corresponding
        ``..._mapping.pkl`` file to consider the shard valid. Incompatible
        shards (dimension mismatch) or incomplete pairs are skipped with a
        warning, to guard against partially written artifacts.
        """
        pattern = re.compile(rf"{re.escape(self._descriptive_core)}_s(\d{{4}})\.index$")
        candidates: List[Tuple[int, Path]] = []
        for f in self.output_directory.glob(f"{self._descriptive_core}_s*.index"):
            m = pattern.match(f.name)
            if not m:
                continue
            shard_num = int(m.group(1))
            candidates.append((shard_num, f))
        candidates.sort(key=lambda x: x[0])

        valid_count = 0
        for shard_num, idx_path in candidates:
            map_path = self.output_directory / f"{self._descriptive_core}_s{shard_num:04d}_mapping.pkl"
            if not map_path.exists():
                logger.warning(
                    "Ignoring shard s%04d: mapping file missing (%s)", shard_num, map_path
                )
                continue
            try:
                # Eagerly load the index to validate dimension
                idx = faiss.read_index(str(idx_path))
                if idx.d != self.feature_dim:
                    logger.warning(
                        "Ignoring shard s%04d: dimension mismatch (index=%d vs expected=%d)",
                        shard_num,
                        idx.d,
                        self.feature_dim,
                    )
                    continue
                with open(map_path, "rb") as f:
                    id_map: Dict[int, str] = pickle.load(f)
                if idx.ntotal != len(id_map):
                    logger.warning(
                        "Shard s%04d size mismatch: index.ntotal=%d, mapping=%d (continuing)",
                        shard_num,
                        idx.ntotal,
                        len(id_map),
                    )
                self.shard_descriptors.append(
                    _ShardDescriptor(
                        shard_id=shard_num,
                        index_path=idx_path,
                        mapping_path=map_path,
                        index=idx,
                        id_to_path_map=id_map,
                        modified=False,
                    )
                )
                valid_count += 1
            except Exception as e:
                logger.warning(
                    "Ignoring shard s%04d due to load error: %s", shard_num, e
                )
                continue

        if valid_count == 0:
            # No shard files discovered. Attempt to adopt legacy single-file artifacts
            # as the first shard to preserve backward compatibility and avoid data loss.
            try:
                legacy_idx_path = self.output_directory / f"{self._descriptive_core}.index"
                legacy_map_path = self.output_directory / f"{self._descriptive_core}_mapping.pkl"
                if legacy_idx_path.exists() and legacy_map_path.exists():
                    idx = faiss.read_index(str(legacy_idx_path))
                    if idx.d != self.feature_dim:
                        logger.warning(
                            "Cannot adopt legacy index: dimension mismatch (index=%d vs expected=%d). Initializing empty shard.",
                            idx.d,
                            self.feature_dim,
                        )
                    else:
                        with open(legacy_map_path, "rb") as f:
                            id_map: Dict[int, str] = pickle.load(f)
                        if idx.ntotal != len(id_map):
                            logger.warning(
                                "Adopted legacy artifacts have size mismatch: index.ntotal=%d, mapping=%d (continuing)",
                                idx.ntotal,
                                len(id_map),
                            )
                        adopted = _ShardDescriptor(
                            shard_id=1,
                            index_path=legacy_idx_path,
                            mapping_path=legacy_map_path,
                            index=idx,
                            id_to_path_map=id_map,
                            modified=False,
                        )
                        self.shard_descriptors = [adopted]
                        self.active_shard_idx = 0
                        # Update legacy-compat handles
                        self.index_path = adopted.index_path
                        self.mapping_path = adopted.mapping_path
                        self.faiss_index = adopted.index
                        self.id_to_path_map = adopted.id_to_path_map
                        self._rebuild_known_paths_set()
                        _key = f"{str(self.output_directory)}|{self._descriptive_core}"
                        if _key not in _LOG_SEEN_ADOPT_LEGACY:
                            logger.info(
                                "FAISS sharded mode: adopted legacy single index as shard s%04d (items=%d).",
                                adopted.shard_id,
                                int(adopted.index.ntotal) if adopted.index else 0,
                            )
                            _LOG_SEEN_ADOPT_LEGACY.add(_key)
                        return
            except Exception as e:
                logger.warning("Failed to adopt legacy single index: %s. Initializing empty shard.", e)

            # Initialize a new empty shard 1 in-memory; files will be written on save
            idx_path, map_path = self._build_shard_paths(1)
            desc = _ShardDescriptor(
                shard_id=1,
                index_path=idx_path,
                mapping_path=map_path,
                index=None,
                id_to_path_map={},
                modified=False,
            )
            self.shard_descriptors = [desc]
            self.active_shard_idx = 0
            # Set legacy-compat paths and index handle to the active shard
            self.index_path = desc.index_path
            self.mapping_path = desc.mapping_path
            self.faiss_index = None
            self.id_to_path_map = desc.id_to_path_map
            _key = f"{str(self.output_directory)}|{self._descriptive_core}"
            if _key not in _LOG_SEEN_EMPTY_INIT:
                logger.info(
                    "FAISS sharded mode: no existing shards found. Initialized empty shard s%04d.",
                    1,
                )
                _LOG_SEEN_EMPTY_INIT.add(_key)
        else:
            # Track active shard as the highest ordinal discovered
            self.shard_descriptors.sort(key=lambda d: d.shard_id)
            self.active_shard_idx = len(self.shard_descriptors) - 1
            # Update legacy-compat attributes to point to active shard
            active = self.shard_descriptors[self.active_shard_idx]
            self.index_path = active.index_path
            self.mapping_path = active.mapping_path
            self.faiss_index = active.index
            self.id_to_path_map = active.id_to_path_map
            # Build union of known paths for fast deduplication
            self._rebuild_known_paths_set()
            _key = f"{str(self.output_directory)}|{self._descriptive_core}"
            if _key not in _LOG_SEEN_DISCOVERY_SUMMARY:
                logger.info(
                    "FAISS sharded mode: discovered %d shard(s). Active shard: s%04d (%s)",
                    len(self.shard_descriptors),
                    active.shard_id,
                    str(active.index_path),
                )
                _LOG_SEEN_DISCOVERY_SUMMARY.add(_key)

    def _rebuild_known_paths_set(self) -> None:
        """Rebuild a global set of known absolute paths across all shards.

        This supports O(1) deduplication checks across the entire corpus.
        """
        known: Set[str] = set()
        if self._sharded:
            for d in self.shard_descriptors:
                for p in d.id_to_path_map.values():
                    try:
                        known.add(str(Path(p).resolve()))
                    except Exception:
                        known.add(str(p))
        else:
            for p in self.id_to_path_map.values():
                try:
                    known.add(str(Path(p).resolve()))
                except Exception:
                    known.add(str(p))
        self._known_paths_set = known
        logger.debug("Rebuilt known paths set with %d entries.", len(self._known_paths_set))

    def _ensure_active_index_initialized(self) -> None:
        """Ensure the active FAISS index object exists and is initialized.

        This method respects the configured index type and parameters.
        """
        if self._sharded:
            active = self.shard_descriptors[self.active_shard_idx]
            if active.index is None:
                active.index = self._create_empty_index()
                self.faiss_index = active.index
                logger.debug("Initialized FAISS index for active shard s%04d.", active.shard_id)
        else:
            if self.faiss_index is None:
                self.faiss_index = self._create_empty_index()
                logger.debug("Initialized single-file FAISS index.")

    def _create_empty_index(self) -> faiss.Index:
        """Create a new empty FAISS index according to ``index_type``.

        Returns:
            faiss.Index: New empty index instance (trained=false for IVF).

        Raises:
            ValueError: When ``index_type`` is unsupported.
        """
        if self.index_type == "flat":
            idx = faiss.IndexFlatL2(self.feature_dim)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.feature_dim)
            idx = faiss.IndexIVFFlat(quantizer, self.feature_dim, self.ivf_nlist, faiss.METRIC_L2)
        elif self.index_type == "hnsw":
            idx = faiss.IndexHNSWFlat(self.feature_dim, self.hnsw_m, faiss.METRIC_L2)
            # Construction parameter controls graph quality during insertion
            idx.hnsw.efConstruction = self.hnsw_ef_construction  # type: ignore[attr-defined]
        else:
            raise ValueError(f"Unsupported FAISS index type: '{self.index_type}'")
        return idx

    # ---------------------------------------------------------------------
    # Public helpers and properties
    # ---------------------------------------------------------------------
    @property
    def index_identifiers(self) -> List[str]:
        """Return a list of index identifiers (paths) for logging.

        - Sharded mode: paths to all shard index files.
        - Legacy mode: list with a single path to the index file.

        Returns:
            List[str]: One or more absolute paths to index files.
        """
        if self._sharded:
            return [str(d.index_path.resolve()) for d in self.shard_descriptors]
        return [str(self.index_path.resolve())]

    def is_path_indexed(self, image_path: Path | str) -> bool:
        """Check if a path is already present in any shard mapping.

        Args:
            image_path: Image path to check.

        Returns:
            bool: True when the absolute path is present in the index mappings.
        """
        if not self._known_paths_set:
            self._rebuild_known_paths_set()
        resolved = str(Path(image_path).resolve())
        return resolved in self._known_paths_set

    def is_index_loaded_and_ready(self) -> bool:
        """Check if any vectors are available for search.

        Returns:
            bool: True if at least one vector is indexed and available.
        """
        total = self.get_total_indexed_items()
        return total > 0

    # ---------------------------------------------------------------------
    # Loading and saving
    # ---------------------------------------------------------------------
    def load_index(self) -> bool:
        """Load index and mapping artifacts from disk.

        In legacy mode, loads a single index/mapping pair. In sharded mode,
        discovers shards and eagerly loads all valid shards.

        Returns:
            bool: True when loading succeeded (even if empty), False otherwise.
        """
        if not self._sharded:
            if not self.index_path.is_file() or not self.mapping_path.is_file():
                logger.info(
                    "No existing FAISS index or mapping file found for this configuration (legacy mode)."
                )
                self.faiss_index = None
                self.id_to_path_map = {}
                self._known_paths_set = set()
                return False
            try:
                self.faiss_index = faiss.read_index(str(self.index_path))
                if self.faiss_index.d != self.feature_dim:
                    logger.error(
                        "Dimension mismatch: index.d=%d != expected=%d.",
                        self.faiss_index.d,
                        self.feature_dim,
                    )
                    self.faiss_index = None
                    return False
                with open(self.mapping_path, "rb") as f:
                    self.id_to_path_map = pickle.load(f)
                if self.faiss_index.ntotal != len(self.id_to_path_map):
                    logger.warning(
                        "Data inconsistency: index.ntotal=%d, mapping=%d (continuing)",
                        self.faiss_index.ntotal,
                        len(self.id_to_path_map),
                    )
                self._rebuild_known_paths_set()
                logger.info(
                    "Loaded FAISS index (legacy) with %d vectors. Dim=%d, Trained=%s",
                    self.faiss_index.ntotal,
                    self.faiss_index.d,
                    self.faiss_index.is_trained,
                )
                return True
            except Exception as e:
                logger.error("Failed to load FAISS index: %s", e, exc_info=True)
                self.faiss_index = None
                self.id_to_path_map = {}
                self._known_paths_set = set()
                return False

        # Sharded mode
        try:
            # Discovery was done in __init__, but shards may have been added/copied after
            self.shard_descriptors.clear()
            self._discover_or_initialize_shards()

            # If discovered shards are all empty, attempt to adopt legacy single-file artifacts
            # to avoid transient fallback and preserve existing corpora.
            try:
                total_after_discovery = 0
                for d in self.shard_descriptors:
                    if d.index is None:
                        if d.index_path.exists():
                            d.index = faiss.read_index(str(d.index_path))
                        else:
                            continue
                    total_after_discovery += int(d.index.ntotal)
                if total_after_discovery == 0:
                    legacy_idx_path = self.output_directory / f"{self._descriptive_core}.index"
                    legacy_map_path = self.output_directory / f"{self._descriptive_core}_mapping.pkl"
                    if legacy_idx_path.exists() and legacy_map_path.exists():
                        try:
                            idx = faiss.read_index(str(legacy_idx_path))
                            if idx.d == self.feature_dim and int(idx.ntotal) > 0:
                                with open(legacy_map_path, "rb") as f:
                                    id_map: Dict[int, str] = pickle.load(f)
                                adopted = _ShardDescriptor(
                                    shard_id=1,
                                    index_path=legacy_idx_path,
                                    mapping_path=legacy_map_path,
                                    index=idx,
                                    id_to_path_map=id_map,
                                    modified=False,
                                )
                                self.shard_descriptors = [adopted]
                                self.active_shard_idx = 0
                                _key = f"{str(self.output_directory)}|{self._descriptive_core}"
                                if _key not in _LOG_SEEN_ADOPT_LEGACY:
                                    logger.info(
                                        "FAISS sharded load: discovered shards empty; adopted legacy single index as shard s%04d (items=%d).",
                                        adopted.shard_id,
                                        int(idx.ntotal),
                                    )
                                    _LOG_SEEN_ADOPT_LEGACY.add(_key)
                        except Exception as e_ad:
                            logger.warning("Failed to adopt legacy single index during load: %s", e_ad)
            except Exception:
                pass

            active = self.shard_descriptors[self.active_shard_idx]
            self.faiss_index = active.index
            self.id_to_path_map = active.id_to_path_map
            self._rebuild_known_paths_set()
            total = self.get_total_indexed_items()
            _key = f"{str(self.output_directory)}|{self._descriptive_core}"
            if _key not in _LOG_SEEN_LOADED_SUMMARY:
                logger.info(
                    "Loaded FAISS sharded index: %d shard(s), total items=%d. First=%s",
                    len(self.shard_descriptors),
                    total,
                    str(self.shard_descriptors[0].index_path) if self.shard_descriptors else "(none)",
                )
                _LOG_SEEN_LOADED_SUMMARY.add(_key)
            return True
        except Exception as e:
            logger.error("Failed to load FAISS sharded index: %s", e, exc_info=True)
            return False

    def save_index(self) -> None:
        """Persist modified index shards and their mappings to disk.

        - Legacy: writes the single index/mapping pair when in-memory index exists.
        - Sharded: writes only shards flagged as modified.
        """
        if not self._sharded:
            if self.faiss_index is None:
                logger.warning("Attempted to save legacy index, but it is not initialized.")
                return
            try:
                self.output_directory.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self.faiss_index, str(self.index_path))
                with open(self.mapping_path, "wb") as f:
                    pickle.dump(self.id_to_path_map, f)
                logger.info("FAISS index and mapping saved (legacy mode).")
            except Exception as e:
                logger.error("Failed to save FAISS index/mapping: %s", e, exc_info=True)
            return

        # Sharded mode: save only modified shards
        for d in self.shard_descriptors:
            if not d.modified:
                continue
            if d.index is None:
                # No index data to write; write empty index
                d.index = self._create_empty_index()
            try:
                faiss.write_index(d.index, str(d.index_path))
                with open(d.mapping_path, "wb") as f:
                    pickle.dump(d.id_to_path_map, f)
                d.modified = False
                logger.info("Saved shard s%04d (%s).", d.shard_id, d.index_path.name)
            except Exception as e:
                logger.error("Failed to save shard s%04d: %s", d.shard_id, e, exc_info=True)

    # ---------------------------------------------------------------------
    # Training helpers
    # ---------------------------------------------------------------------
    def train_ivf_index(self, training_vectors: np.ndarray) -> None:
        """Train an IVF-type index with provided vectors.

        Args:
            training_vectors: 2D array of training vectors.
        """
        if self.index_type != "ivf":
            return
        if self.faiss_index is None:
            self._ensure_active_index_initialized()
        if self.faiss_index is None:
            raise RuntimeError("IVF index is not initialized.")
        if self.faiss_index.is_trained:
            logger.info("IVF index is already trained. Skipping training.")
            return
        if training_vectors.shape[0] < self.ivf_nlist:
            raise ValueError(
                f"Not enough training vectors for IVF: got {training_vectors.shape[0]}, need >= {self.ivf_nlist}."
            )
        logger.debug("Training IVF index with %d vectors...", training_vectors.shape[0])
        self.faiss_index.train(training_vectors)
        logger.info("IVF training complete.")
        clear_gpu_memory()

    # ---------------------------------------------------------------------
    # Index building and updates
    # ---------------------------------------------------------------------
    def add_vectors_to_index(self, vectors: np.ndarray, image_paths: Sequence[Path]) -> None:
        """Add vectors and paths to the index, honoring shard capacity when enabled.

        The mapping is stored per shard with local IDs 0..ntotal-1.

        Args:
            vectors: 2D array of feature vectors to add.
            image_paths: Sequence of image paths corresponding to the vectors.

        Raises:
            RuntimeError: When the index is not initialized.
            ValueError: On vector/path count mismatch or IVF training insufficiency for a new shard.
        """
        if vectors.shape[0] != len(image_paths):
            raise ValueError("Mismatch between number of vectors and image paths.")
        if vectors.ndim != 2 or vectors.shape[1] != self.feature_dim:
            raise ValueError("Vectors must be 2D with correct feature dimension.")

        if not self._sharded:
            if self.faiss_index is None:
                raise RuntimeError("Cannot add vectors: FAISS index is not initialized.")
            start_id = self.faiss_index.ntotal
            self.faiss_index.add(vectors)
            for i, p in enumerate(image_paths):
                self.id_to_path_map[start_id + i] = str(Path(p).resolve())
            self._rebuild_known_paths_set()
            return

        # Sharded mode
        remaining_vectors = vectors
        remaining_paths = list(image_paths)
        while remaining_vectors.shape[0] > 0:
            active = self.shard_descriptors[self.active_shard_idx]
            if active.index is None:
                active.index = self._create_empty_index()
                # Expose via legacy handles for training helpers
                self.faiss_index = active.index
                self.id_to_path_map = active.id_to_path_map

            capacity = int(self.partition_capacity or 0)
            current = int(active.index.ntotal)
            room = max(0, capacity - current)
            if room <= 0:
                # Roll to a new shard
                new_shard_id = (self.shard_descriptors[-1].shard_id + 1) if self.shard_descriptors else 1
                idx_path, map_path = self._build_shard_paths(new_shard_id)
                new_desc = _ShardDescriptor(
                    shard_id=new_shard_id,
                    index_path=idx_path,
                    mapping_path=map_path,
                    index=self._create_empty_index(),
                    id_to_path_map={},
                    modified=False,
                )
                self.shard_descriptors.append(new_desc)
                self.active_shard_idx = len(self.shard_descriptors) - 1
                active = new_desc
                # Update legacy-compat handles
                self.index_path = active.index_path
                self.mapping_path = active.mapping_path
                self.faiss_index = active.index
                self.id_to_path_map = active.id_to_path_map
                room = int(self.partition_capacity or 0)
                logger.info("Rolled over to new FAISS shard s%04d.", active.shard_id)

            take = min(room, remaining_vectors.shape[0])
            batch_vecs = remaining_vectors[:take]
            batch_paths = remaining_paths[:take]

            # IVF shard may require training before adding
            if self.index_type == "ivf" and active.index is not None and (not active.index.is_trained):
                if batch_vecs.shape[0] < self.ivf_nlist:
                    raise ValueError(
                        f"Cannot train new IVF shard s{active.shard_id:04d}: insufficient vectors to train "
                        f"(have {batch_vecs.shape[0]}, need >= {self.ivf_nlist}). Consider increasing batch size "
                        f"or partition_capacity, or pre-train shard with dedicated samples."
                    )
                logger.info(
                    "Training IVF for shard s%04d with %d vectors before adding...",
                    active.shard_id,
                    batch_vecs.shape[0],
                )
                active.index.train(batch_vecs)

            start_id = int(active.index.ntotal) if active.index is not None else 0
            active.index.add(batch_vecs)  # type: ignore[union-attr]
            for i, p in enumerate(batch_paths):
                active.id_to_path_map[start_id + i] = str(Path(p).resolve())
            active.modified = True

            # Update global known-paths set incrementally
            for p in batch_paths:
                try:
                    self._known_paths_set.add(str(Path(p).resolve()))
                except Exception:
                    self._known_paths_set.add(str(p))

            # Consume from remaining
            remaining_vectors = remaining_vectors[take:]
            remaining_paths = remaining_paths[take:]

    def get_total_indexed_items(self) -> int:
        """Return total number of indexed items across all shards.

        Returns:
            int: Total count of indexed vectors.
        """
        if not self._sharded:
            return int(self.faiss_index.ntotal) if self.faiss_index is not None else 0
        total = 0
        for d in self.shard_descriptors:
            if d.index is not None:
                total += int(d.index.ntotal)
            else:
                # Fallback to mapping length if index is not loaded yet
                try:
                    if d.mapping_path.exists():
                        with open(d.mapping_path, "rb") as f:
                            id_map = pickle.load(f)
                            total += int(len(id_map))
                except Exception:
                    continue
        return total

    # ---------------------------------------------------------------------
    # End-to-end building from a folder (unchanged interface)
    # ---------------------------------------------------------------------
    def _get_new_image_paths_to_index(self, image_folder_path: Path, scan_subfolders: bool = False) -> List[Path]:
        """Scan the folder and list images not yet indexed in any shard.

        Args:
            image_folder_path: Root folder to scan.
            scan_subfolders: Whether to recurse into subdirectories.

        Returns:
            List[Path]: Absolute paths for images not present in index mappings.
        """
        logger.debug(
            "Scanning '%s' for new images to index (recursive=%s).",
            image_folder_path,
            scan_subfolders,
        )
        new_paths: List[Path] = []
        if not self._known_paths_set:
            self._rebuild_known_paths_set()
        for image_path in image_path_generator(image_folder_path, scan_subfolders):
            if not self.is_path_indexed(image_path):
                new_paths.append(Path(image_path).resolve())
        logger.info(
            "Found %d new images in '%s' that are not in the current index.",
            len(new_paths),
            image_folder_path,
        )
        return new_paths

    def _extract_features_for_paths(
        self,
        image_paths: List[Path],
        feature_extractor: FeatureExtractor,
        batch_size: int,
        desc: str = "Extracting Features",
    ) -> Optional[np.ndarray]:
        """Extract features in batches for provided image paths.

        Args:
            image_paths: Images to embed.
            feature_extractor: Initialized extractor.
            batch_size: Batch size for embedding.
            desc: Progress description.

        Returns:
            Optional[np.ndarray]: 2D array of features or None when no features.
        """
        if not image_paths:
            return None
        all_features: List[np.ndarray] = []
        progress_bar = tqdm(range(0, len(image_paths), batch_size), desc=desc, unit="batch")
        for i in progress_bar:
            batch_paths = image_paths[i : i + batch_size]
            batch_features = feature_extractor.extract_features(batch_paths)
            if batch_features is not None and batch_features.shape[0] > 0:
                all_features.append(batch_features)
        if not all_features:
            return None
        return np.concatenate(all_features, axis=0)

    def build_index_from_folder(
        self,
        feature_extractor: FeatureExtractor,
        image_folder: str,
        batch_size: int = 32,
        force_rebuild: bool = False,
        scan_subfolders: bool = False,
        ivf_train_samples_ratio: float = 0.1,
        ivf_train_samples_max: int = 50000,
    ) -> int:
        """Build or update the FAISS index from a folder of images.

        This method mirrors legacy behavior while delegating to sharded add/save
        when sharding is enabled.

        Args:
            feature_extractor: Feature extractor instance.
            image_folder: Path to database images.
            batch_size: Embedding batch size.
            force_rebuild: When True, clear existing index files first.
            scan_subfolders: Recurse into subdirectories for image discovery.
            ivf_train_samples_ratio: Fraction of new images to use for IVF training.
            ivf_train_samples_max: Absolute cap on samples used for IVF training.

        Returns:
            int: 0 on success, 1 on failure.
        """
        if force_rebuild:
            logger.info("force_rebuild=True. Clearing existing FAISS index data.")
            self.clear_index(save_empty=False)

        if not self.faiss_index:
            self.load_index()
        if not self.faiss_index:
            self._ensure_active_index_initialized()
            if not self.faiss_index:
                logger.error("Failed to initialize a FAISS index.")
                return 1

        new_image_paths = self._get_new_image_paths_to_index(Path(image_folder), scan_subfolders)
        if not new_image_paths:
            logger.info("No new images found to add to the FAISS index.")
            # Save possibly empty structures (e.g., after clear) to ensure files exist
            self.save_index()
            return 0

        # IVF training for the current (active) index only. Each IVF shard is trained independently.
        if self.index_type == "ivf" and not self.faiss_index.is_trained:
            num_train_samples = min(ivf_train_samples_max, int(len(new_image_paths) * ivf_train_samples_ratio))
            num_train_samples = max(self.ivf_nlist, num_train_samples)
            num_train_samples = min(num_train_samples, len(new_image_paths))
            if num_train_samples < self.ivf_nlist:
                logger.error(
                    "Cannot train IVF index: not enough new images (%d) to gather %d required training samples.",
                    len(new_image_paths),
                    self.ivf_nlist,
                )
                return 1
            training_paths = new_image_paths[:num_train_samples]
            logger.info("Extracting %d vectors for IVF training...", len(training_paths))
            training_vectors = self._extract_features_for_paths(
                training_paths, feature_extractor, batch_size, "IVF Training"
            )
            if training_vectors is None or training_vectors.shape[0] < self.ivf_nlist:
                logger.error("Failed to extract sufficient training vectors for IVF index.")
                return 1
            self.train_ivf_index(training_vectors)

        # Extract and add all new images
        logger.info("Extracting features for %d new images...", len(new_image_paths))
        all_new_features = self._extract_features_for_paths(
            new_image_paths, feature_extractor, batch_size, "Main Indexing"
        )
        if all_new_features is None:
            logger.error("Feature extraction produced no vectors.")
            return 1
        try:
            self.add_vectors_to_index(all_new_features, new_image_paths)
        except Exception as e:
            logger.error("Failed to add vectors to index: %s", e, exc_info=True)
            return 1
        self.save_index()
        logger.info("Index build complete. Total items: %d", self.get_total_indexed_items())
        return 0

    # ---------------------------------------------------------------------
    # Search
    # ---------------------------------------------------------------------
    def search_similar_images(
        self,
        query_vector: np.ndarray,
        top_k: int,
        ivf_nprobe_search: Optional[int] = None,
        hnsw_efsearch_search: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Search for the most similar images across all shards.

        Performs a per-shard top-k search and merges all shard candidates into a
        single, globally sorted result set. This guarantees that the returned
        results are the true global top-k, not just the last-processed shard.

        Args:
            query_vector (np.ndarray): Query feature vector. Accepts 1D or 2D (1 x D).
            top_k (int): Number of nearest neighbors to return.
            ivf_nprobe_search (Optional[int]): Per-search IVF ``nprobe`` override.
            hnsw_efsearch_search (Optional[int]): Per-search HNSW ``efSearch`` override.

        Returns:
            List[Tuple[str, float]]: Sorted list of (image_path, similarity) pairs
            in descending order of similarity. Similarity is derived from squared
            L2 distance under the assumption of L2-normalized vectors.

        Raises:
            ValueError: If ``query_vector`` dimension does not match ``feature_dim``.
        """
        if top_k <= 0:
            return []

        # Ensure correct shape and dtype for FAISS
        if query_vector.ndim == 1:
            query = query_vector.reshape(1, -1)
        else:
            query = query_vector
        if query.shape[1] != self.feature_dim:
            raise ValueError("Query vector dimension mismatch.")
        if query.dtype != np.float32:
            query = query.astype(np.float32, copy=False)
        query = np.ascontiguousarray(query)

        if not self.is_index_loaded_and_ready():
            logger.error("Cannot search: FAISS index is not loaded or empty.")
            return []

        def _search_one(idx: faiss.Index, dmap: Dict[int, str], shard_k: int) -> List[Tuple[str, float]]:
            """Search a single index with temporary parameter overrides.

            Args:
                idx: FAISS index to query.
                dmap: Local ID->path mapping for the shard.
                shard_k: Per-shard candidate count to request.

            Returns:
                List of (path, similarity) pairs for valid hits in this shard.
            """
            orig_nprobe: Optional[int] = None
            orig_efsearch: Optional[int] = None
            try:
                if self.index_type == "ivf" and ivf_nprobe_search is not None:
                    orig_nprobe = getattr(idx, "nprobe", None)
                    idx.nprobe = int(ivf_nprobe_search)  # type: ignore[attr-defined]
                if self.index_type == "hnsw" and hnsw_efsearch_search is not None and hasattr(idx, "hnsw"):
                    orig_efsearch = idx.hnsw.efSearch  # type: ignore[attr-defined]
                    idx.hnsw.efSearch = int(hnsw_efsearch_search)  # type: ignore[attr-defined]
                distances, indices = idx.search(query, shard_k)
            finally:
                # Always restore search-time params to avoid cross-search side effects
                try:
                    if orig_nprobe is not None:
                        idx.nprobe = orig_nprobe  # type: ignore[attr-defined]
                    if orig_efsearch is not None and hasattr(idx, "hnsw"):
                        idx.hnsw.efSearch = orig_efsearch  # type: ignore[attr-defined]
                except Exception:
                    pass  # Defensive: do not propagate restoration issues

            merged: List[Tuple[str, float]] = []
            if indices.size == 0:
                return merged
            # indices/distances have shape (1, shard_k) for a single query
            for i in range(indices.shape[1]):
                faiss_id = int(indices[0, i])
                if faiss_id < 0:
                    continue  # no hit
                path = dmap.get(faiss_id)
                if not path:
                    continue  # ignore unmapped IDs defensively
                dist_sq = float(distances[0, i])
                sim = max(0.0, 1.0 - (dist_sq / 2.0))  # for L2-normalized vectors
                merged.append((path, sim))
            return merged

        # Collect candidates from all shards (or single legacy index)
        candidates: List[Tuple[str, float, int]] = []  # (path, score, shard_id)
        if not self._sharded:
            if self.faiss_index is None:
                return []
            per = _search_one(self.faiss_index, self.id_to_path_map, min(top_k, int(self.faiss_index.ntotal)))
            candidates.extend([(p, s, 0) for (p, s) in per])
        else:
            for d in self.shard_descriptors:
                # Lazily load shard index if needed
                if d.index is None:
                    try:
                        if d.index_path.exists():
                            d.index = faiss.read_index(str(d.index_path))
                        else:
                            continue
                    except Exception:
                        # Skip shards that fail to load
                        continue
                # Determine per-shard k (cannot exceed shard size)
                shard_size = int(d.index.ntotal) if d.index is not None else 0
                if shard_size <= 0:
                    continue
                per = _search_one(d.index, d.id_to_path_map, min(top_k, shard_size))
                # Merge shard candidates into the global pool
                if per:
                    candidates.extend([(p, s, d.shard_id) for (p, s) in per])

        if not candidates:
            return []

        # Global top-k with deterministic tie-breaking: score desc, path asc, shard_id asc
        candidates.sort(key=lambda t: (-t[1], t[0], t[2]))
        final = [(p, s) for (p, s, _) in candidates[:top_k]]
        return final

    # ---------------------------------------------------------------------
    # Maintenance
    # ---------------------------------------------------------------------
    def clear_index(self, save_empty: bool = True) -> None:
        """Clear index artifacts and optionally save an empty structure.

        - Legacy: delete in-memory structures and reinitialize an empty index.
        - Sharded: reset to a single empty shard s0001 and remove others.

        Args:
            save_empty: When True, persist the cleared state to disk.
        """
        logger.info("Clearing FAISS index artifacts (sharded=%s).", self._sharded)

        if not self._sharded:
            self.faiss_index = self._create_empty_index()
            self.id_to_path_map = {}
            self._known_paths_set = set()
            if save_empty:
                self.save_index()
            return

        # Sharded mode: remove all shard files
        for d in self.shard_descriptors:
            try:
                if d.index_path.exists():
                    d.index_path.unlink(missing_ok=True)  # type: ignore[call-arg]
                if d.mapping_path.exists():
                    d.mapping_path.unlink(missing_ok=True)  # type: ignore[call-arg]
            except Exception:
                # Best-effort; leave partials in place if deletion fails
                pass
        # Reset to a single empty shard
        idx_path, map_path = self._build_shard_paths(1)
        self.shard_descriptors = [
            _ShardDescriptor(
                shard_id=1,
                index_path=idx_path,
                mapping_path=map_path,
                index=self._create_empty_index(),
                id_to_path_map={},
                modified=True,
            )
        ]
        self.active_shard_idx = 0
        active = self.shard_descriptors[0]
        self.index_path = active.index_path
        self.mapping_path = active.mapping_path
        self.faiss_index = active.index
        self.id_to_path_map = active.id_to_path_map
        self._known_paths_set = set()
        if save_empty:
            self.save_index()

