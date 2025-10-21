"""FAISS shard and single-file migration utility.

This script provides two migration paths for FAISS indexes:

1) shards-to-single (default):
   Consolidate multiple shard files into a single legacy index (.index + _mapping.pkl).

2) single-to-shards:
   Split a legacy single index (.index + _mapping.pkl) into multiple shard files
   based on a configured shard size (capacity), producing sNNNN artifacts.

Both paths avoid re-extracting image features from source files by reconstructing
vectors directly from existing FAISS indexes.

Usage examples
- Shards -> Single index:
    python tools/faiss_shard_migrator.py \
        --mode shards-to-single \
        --output-directory instance/faiss_indices \
        --filename-stem faiss_collection \
        --model-name efficientnet \
        --index-type flat \
        --archive-shards --force-overwrite

- Single -> Shards (shard size 300000):
    python tools/faiss_shard_migrator.py \
        --mode single-to-shards \
        --output-directory instance/faiss_indices \
        --filename-stem faiss_collection \
        --model-name efficientnet \
        --index-type ivf \
        --shard-size 300000 \
        --ivf-nlist 100 --archive-single

Notes and constraints
- IVF indexes require training. This utility trains IVF target indexes using
  vectors from the source (sampled subset for shards-to-single; shard segment
  itself for single-to-shards). For single-to-shards, ensure shard_size >= ivf_nlist
  to allow training for each shard; otherwise, the operation will fail with a
  clear error.
- Index types supported here match the project: IndexFlatL2 (flat), IndexIVFFlat
  (ivf with Flat quantizer), and IndexHNSWFlat (hnsw). All support vector
  reconstruction via reconstruct(i) in FAISS Python API.
- For large corpora, reconstruction is the dominant cost. This implementation
  reconstructs in manageable batches to limit memory usage.

Author: Hyundai Document Authenticator
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import pickle
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import faiss  # type: ignore
import numpy as np


logger = logging.getLogger("faiss_shard_migrator")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ------------------------------
# Common helpers
# ------------------------------

def _descriptive_core(filename_stem: str, model_name: str, index_type: str) -> str:
    """Build the filename core shared by artifacts.

    Args:
        filename_stem: Base filename stem.
        model_name: Model name (lowercase recommended).
        index_type: Index type (lowercase recommended).

    Returns:
        str: Core name "{stem}_{model}_{index_type}".
    """
    return f"{filename_stem}_{model_name}_{index_type}"


def _create_empty_index(index_type: str, dim: int, ivf_nlist: int, hnsw_m: int, hnsw_efc: int) -> faiss.Index:
    """Create a new empty FAISS index based on type and parameters.

    Args:
        index_type: "flat" | "ivf" | "hnsw".
        dim: Feature dimension.
        ivf_nlist: IVF cluster count (IVF only).
        hnsw_m: HNSW edges per node (HNSW only).
        hnsw_efc: HNSW efConstruction (HNSW only).

    Returns:
        faiss.Index: New index instance.

    Raises:
        ValueError: For unsupported index types.
    """
    index_type = index_type.lower()
    if index_type == "flat":
        return faiss.IndexFlatL2(dim)
    if index_type == "ivf":
        q = faiss.IndexFlatL2(dim)
        return faiss.IndexIVFFlat(q, dim, ivf_nlist, faiss.METRIC_L2)
    if index_type == "hnsw":
        idx = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_L2)
        idx.hnsw.efConstruction = int(hnsw_efc)  # type: ignore[attr-defined]
        return idx
    raise ValueError(f"Unsupported index_type: {index_type}")


def _reconstruct_range(idx: faiss.Index, start: int, end: int) -> np.ndarray:
    """Reconstruct vectors in [start, end) from index.

    Args:
        idx: FAISS index instance.
        start: Inclusive start id.
        end: Exclusive end id.

    Returns:
        np.ndarray: (end-start, dim) float32 matrix.
    """
    n = max(0, end - start)
    d = int(idx.d)
    out = np.empty((n, d), dtype=np.float32)
    for i in range(n):
        out[i] = idx.reconstruct(start + i)  # type: ignore[attr-defined]
    return out


def _reconstruct_all_vectors(idx: faiss.Index, batch_size: int = 16384) -> np.ndarray:
    """Reconstruct all vectors from a FAISS index in batches.

    Args:
        idx: Loaded FAISS index.
        batch_size: Batch size for reconstruction.

    Returns:
        np.ndarray: 2D array of reconstructed vectors (ntotal, dim).
    """
    total = int(idx.ntotal)
    d = int(idx.d)
    out = np.empty((total, d), dtype=np.float32)
    start = 0
    while start < total:
        end = min(start + batch_size, total)
        for i in range(start, end):
            out[i] = idx.reconstruct(i)  # type: ignore[attr-defined]
        start = end
    return out


def _load_mapping(map_path: Path) -> Dict[int, str]:
    """Load a mapping pickle (id->path).

    Args:
        map_path: Path to mapping pickle.

    Returns:
        Dict[int, str]: Mapping of local ids to image paths.
    """
    with open(map_path, "rb") as f:
        return pickle.load(f)


# ------------------------------
# Shards -> Single
# ------------------------------

def _find_shards(output_dir: Path, core: str) -> List[Tuple[int, Path, Path]]:
    """Discover shard files and return ordered descriptors.

    Args:
        output_dir: Directory containing FAISS artifacts.
        core: Descriptive core name for artifacts.

    Returns:
        List[Tuple[int, Path, Path]]: Sorted (shard_id, index_path, mapping_path).
    """
    shards: List[Tuple[int, Path, Path]] = []
    for idx_path in output_dir.glob(f"{core}_s*.index"):
        name = idx_path.name
        try:
            shard_str = name.split("_s")[-1].split(".")[0]
            shard_id = int(shard_str)
        except Exception:
            continue
        map_path = output_dir / f"{core}_s{shard_id:04d}_mapping.pkl"
        if not map_path.exists():
            logger.warning("Skipping shard %s: mapping file missing: %s", idx_path.name, map_path)
            continue
        shards.append((shard_id, idx_path, map_path))
    shards.sort(key=lambda t: t[0])
    return shards


def migrate_shards_to_single(
    output_directory: Path,
    filename_stem: str,
    model_name: str,
    index_type: str,
    ivf_nlist: int = 100,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 40,
    ivf_train_samples: int = 50000,
    archive_shards: bool = False,
    force_overwrite: bool = False,
) -> Tuple[Path, Path]:
    """Migrate shard-suffixed FAISS indexes into a single legacy file.

    The function discovers shards, validates compatibility, reconstructs vectors
    shard-by-shard, and builds a consolidated target index plus mapping.

    Args:
        output_directory: Directory containing FAISS artifacts.
        filename_stem: Base filename stem.
        model_name: Model name (lowercase recommended for consistency).
        index_type: "flat" | "ivf" | "hnsw".
        ivf_nlist: IVF nlist to use when building the consolidated IVF index.
        hnsw_m: HNSW parameter m for consolidated HNSW index.
        hnsw_ef_construction: HNSW efConstruction for consolidated index.
        ivf_train_samples: Maximum number of samples to train IVF centroids.
        archive_shards: When True, moves shard files into an archive subfolder
            after successful migration.
        force_overwrite: When True, overwrites target files if they already exist.

    Returns:
        Tuple[Path, Path]: Paths to the consolidated index and mapping files.

    Raises:
        FileNotFoundError: When no valid shards are found.
        ValueError: On dimension mismatch across shards or unsupported index types.
    """
    out_dir = output_directory.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    core = _descriptive_core(filename_stem, model_name.lower(), index_type.lower())
    shards = _find_shards(out_dir, core)
    if not shards:
        raise FileNotFoundError(f"No shard files found matching '{core}_sNNNN.index' under {out_dir}")

    # Validate shards and determine dimension
    dim: Optional[int] = None
    shard_metas: List[Tuple[int, faiss.Index, Dict[int, str]]] = []
    for sid, idx_path, map_path in shards:
        idx = faiss.read_index(str(idx_path))
        if dim is None:
            dim = int(idx.d)
        elif int(idx.d) != dim:
            raise ValueError(
                f"Shard s{sid:04d} dimension {idx.d} != expected {dim}. Aborting."
            )
        idmap = _load_mapping(map_path)
        shard_metas.append((sid, idx, idmap))

    assert dim is not None

    # Prepare target artifacts
    target_index_path = out_dir / f"{core}.index"
    target_map_path = out_dir / f"{core}_mapping.pkl"
    if target_index_path.exists() or target_map_path.exists():
        if not force_overwrite:
            raise FileExistsError(
                f"Target files exist: {target_index_path} / {target_map_path}. Use --force-overwrite to replace."
            )
        target_index_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        target_map_path.unlink(missing_ok=True)  # type: ignore[arg-type]

    # Create target index
    tgt = _create_empty_index(index_type, dim, ivf_nlist, hnsw_m, hnsw_ef_construction)

    # IVF training: collect sample vectors up to ivf_train_samples
    if index_type.lower() == "ivf":
        samples: List[np.ndarray] = []
        remaining = int(ivf_train_samples)
        for sid, idx, _ in shard_metas:
            nt = int(idx.ntotal)
            if nt == 0 or remaining <= 0:
                continue
            take = min(remaining, nt)
            part = _reconstruct_all_vectors(idx, batch_size=min(take, 16384))[:take]
            samples.append(part)
            remaining -= take
            if remaining <= 0:
                break
        if samples:
            train_mat = np.vstack(samples)
            logger.info("Training IVF index with %d vectors (nlist=%d).", train_mat.shape[0], ivf_nlist)
            tgt.train(train_mat)
        else:
            raise ValueError("Cannot train IVF index: no sample vectors available from shards.")

    # Build consolidated mapping and add vectors shard-by-shard
    global_map: Dict[int, str] = {}
    global_cursor = 0
    for sid, idx, idmap in shard_metas:
        nt = int(idx.ntotal)
        if nt == 0:
            continue
        logger.info("Adding %d vectors from shard s%04d...", nt, sid)
        vecs = _reconstruct_all_vectors(idx)
        tgt.add(vecs)
        for local_id in range(nt):
            path = idmap.get(local_id)
            if path is not None:
                global_map[global_cursor + local_id] = path
        global_cursor += nt

    # Persist target artifacts
    faiss.write_index(tgt, str(target_index_path))
    with open(target_map_path, "wb") as f:
        pickle.dump(global_map, f)
    logger.info("Consolidated FAISS index saved: %s (total=%d)", target_index_path.name, global_cursor)

    # Optionally archive shards
    if archive_shards:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        arch_dir = out_dir / f"_archived_shards_{ts}"
        arch_dir.mkdir(parents=True, exist_ok=True)
        for sid, idx_path, map_path in shards:
            for p in (idx_path, map_path):
                dest = arch_dir / p.name
                shutil.move(str(p), str(dest))
        logger.info("Archived shard artifacts to %s", arch_dir)

    return target_index_path, target_map_path


# ------------------------------
# Single -> Shards
# ------------------------------

def migrate_single_to_shards(
    output_directory: Path,
    filename_stem: str,
    model_name: str,
    index_type: str,
    shard_size: int,
    ivf_nlist: int = 100,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 40,
    archive_single: bool = False,
    force_overwrite: bool = False,
) -> List[Tuple[Path, Path]]:
    """Split a legacy single index into shard-suffixed indexes.

    This function reads the non-suffixed index and mapping files, reconstructs
    vectors, and writes shard files in segments of shard_size.

    Args:
        output_directory: Directory containing FAISS artifacts.
        filename_stem: Base filename stem.
        model_name: Model name (lowercase recommended for consistency).
        index_type: "flat" | "ivf" | "hnsw".
        shard_size: Maximum vectors per shard (must be >= ivf_nlist for IVF).
        ivf_nlist: IVF nlist for per-shard indexes.
        hnsw_m: HNSW m for per-shard indexes.
        hnsw_ef_construction: HNSW efConstruction for per-shard indexes.
        archive_single: When True, moves the original single-file artifacts into
            an archive subfolder after successful migration.
        force_overwrite: When True, overwrites existing shard files if they exist.

    Returns:
        List[Tuple[Path, Path]]: List of created (index_path, mapping_path) pairs ordered by shard.

    Raises:
        FileNotFoundError: When the single-file artifacts are missing.
        ValueError: If IVF shard_size is smaller than ivf_nlist.
    """
    out_dir = output_directory.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    core = _descriptive_core(filename_stem, model_name.lower(), index_type.lower())
    single_index_path = out_dir / f"{core}.index"
    single_map_path = out_dir / f"{core}_mapping.pkl"
    if not single_index_path.exists() or not single_map_path.exists():
        raise FileNotFoundError(
            f"Legacy single artifacts not found: {single_index_path} / {single_map_path}"
        )

    if index_type.lower() == "ivf" and shard_size < int(ivf_nlist):
        raise ValueError(
            f"IVF shard_size ({shard_size}) must be >= ivf_nlist ({ivf_nlist}) to allow per-shard training."
        )

    idx = faiss.read_index(str(single_index_path))
    dim = int(idx.d)
    idmap = _load_mapping(single_map_path)
    ntotal = int(idx.ntotal)
    if ntotal == 0:
        logger.info("Single index is empty; nothing to shard.")
        return []

    # Compute next shard ordinal after any existing shards
    existing = _find_shards(out_dir, core)
    start_ordinal = existing[-1][0] + 1 if existing else 1

    created: List[Tuple[Path, Path]] = []
    cursor = 0
    shard_id = start_ordinal
    while cursor < ntotal:
        end = min(cursor + shard_size, ntotal)
        seg_len = end - cursor

        # Reconstruct this segment
        logger.info("Building shard s%04d with %d vectors...", shard_id, seg_len)
        seg_vecs = _reconstruct_range(idx, cursor, end)

        # Create and optionally train per-shard index
        shard_index = _create_empty_index(index_type, dim, ivf_nlist, hnsw_m, hnsw_ef_construction)
        if index_type.lower() == "ivf":
            # Train using the segment itself
            if seg_len < int(ivf_nlist):
                raise ValueError(
                    f"Cannot train IVF for shard s{shard_id:04d}: segment size {seg_len} < ivf_nlist {ivf_nlist}. "
                    f"Increase shard_size or reduce ivf_nlist."
                )
            shard_index.train(seg_vecs)

        shard_index.add(seg_vecs)

        # Prepare file paths
        idx_path = out_dir / f"{core}_s{shard_id:04d}.index"
        map_path = out_dir / f"{core}_s{shard_id:04d}_mapping.pkl"
        if (idx_path.exists() or map_path.exists()) and not force_overwrite:
            raise FileExistsError(f"Shard artifacts exist: {idx_path} / {map_path}")

        # Write shard files
        faiss.write_index(shard_index, str(idx_path))
        # Build per-shard mapping (local ids 0..seg_len-1)
        local_map: Dict[int, str] = {}
        for i_local in range(seg_len):
            path = idmap.get(cursor + i_local)
            if path is not None:
                local_map[i_local] = path
        with open(map_path, "wb") as f:
            pickle.dump(local_map, f)

        created.append((idx_path, map_path))
        shard_id += 1
        cursor = end

    if archive_single:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        arch_dir = out_dir / f"_archived_single_{ts}"
        arch_dir.mkdir(parents=True, exist_ok=True)
        for p in (single_index_path, single_map_path):
            dest = arch_dir / p.name
            shutil.move(str(p), str(dest))
        logger.info("Archived legacy single artifacts to %s", arch_dir)

    logger.info("Created %d shard(s) from legacy single index.", len(created))
    return created


# ------------------------------
# CLI entrypoint
# ------------------------------

def main() -> None:
    """CLI entrypoint for FAISS shard/single-file migration utility.

    Supports two modes: 'shards-to-single' and 'single-to-shards'.
    """
    ap = argparse.ArgumentParser(description="FAISS shard/single-file migration utility")
    ap.add_argument("--mode", choices=["shards-to-single", "single-to-shards"], default="shards-to-single", help="Migration direction")
    ap.add_argument("--output-directory", required=True, help="Directory containing FAISS artifacts")
    ap.add_argument("--filename-stem", required=True, help="FAISS filename stem, e.g., faiss_collection")
    ap.add_argument("--model-name", required=True, help="Model name used in artifact names, e.g., efficientnet")
    ap.add_argument("--index-type", required=True, choices=["flat", "ivf", "hnsw"], help="Index type")
    # Common params
    ap.add_argument("--ivf-nlist", type=int, default=100, help="IVF nlist for consolidated or per-shard index (IVF only)")
    ap.add_argument("--hnsw-m", type=int, default=32, help="HNSW m for index (HNSW only)")
    ap.add_argument("--hnsw-ef-construction", type=int, default=40, help="HNSW efConstruction (HNSW only)")
    ap.add_argument("--force-overwrite", action="store_true", help="Overwrite target artifacts if they exist")

    # shards->single specifics
    ap.add_argument("--ivf-train-samples", type=int, default=50000, help="Max vectors to train IVF centroids (shards-to-single)")
    ap.add_argument("--archive-shards", action="store_true", help="Archive shard files after shards-to-single migration")

    # single->shards specifics
    ap.add_argument("--shard-size", type=int, help="Vectors per shard (required for single-to-shards mode)")
    ap.add_argument("--archive-single", action="store_true", help="Archive legacy single files after single-to-shards migration")

    args = ap.parse_args()

    out_dir = Path(args.output_directory)

    if args.mode == "shards-to-single":
        idx_path, map_path = migrate_shards_to_single(
            output_directory=out_dir,
            filename_stem=args.filename_stem,
            model_name=args.model_name,
            index_type=args.index_type,
            ivf_nlist=args.ivf_nlist,
            hnsw_m=args.hnsw_m,
            hnsw_ef_construction=args.hnsw_ef_construction,
            ivf_train_samples=args.ivf_train_samples,
            archive_shards=bool(args.archive_shards),
            force_overwrite=bool(args.force_overwrite),
        )
        print(f"Consolidation complete. Index: {idx_path} | Mapping: {map_path}")
    else:
        if args.shard_size is None or int(args.shard_size) <= 0:
            raise SystemExit("--shard-size must be provided and > 0 for single-to-shards mode")
        created = migrate_single_to_shards(
            output_directory=out_dir,
            filename_stem=args.filename_stem,
            model_name=args.model_name,
            index_type=args.index_type,
            shard_size=int(args.shard_size),
            ivf_nlist=args.ivf_nlist,
            hnsw_m=args.hnsw_m,
            hnsw_ef_construction=args.hnsw_ef_construction,
            archive_single=bool(args.archive_single),
            force_overwrite=bool(args.force_overwrite),
        )
        if not created:
            print("No shards created (source was empty).")
        else:
            print("Created shards:")
            for ip, mp in created:
                print(f"  {ip.name} | {mp.name}")


if __name__ == "__main__":
    main()
