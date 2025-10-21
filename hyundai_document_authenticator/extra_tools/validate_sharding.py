from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Ensure package import path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Generate test images if missing
try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # type: ignore

def ensure_test_images(base: Path, count: int = 12) -> List[Path]:
    base.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for i in range(count):
        p = base / f"test_img_{i:02d}.jpg"
        if not p.exists() and Image is not None:
            img = Image.new('RGB', (64, 64), ((i*20) % 255, (i*40) % 255, (i*60) % 255))
            img.save(str(p), format='JPEG', quality=85)
        paths.append(p)
    return paths

# ---------------- FAISS Validation ----------------

def validate_faiss_sharding() -> None:
    print("[FAISS] Validation started")
    from hyundai_document_authenticator.core_engine.image_similarity_system.faiss_manager import (
        FaissIndexManager,
    )
    # Test parameters
    feature_dim = 16
    shard_cap = 5
    img_dir = ROOT / "instance" / "database_images_test"
    out_dir = ROOT / "instance" / "faiss_indices" / "validate_faiss"
    # Ensure clean output folder
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = ensure_test_images(img_dir, 12)
    # Create manager in sharded mode
    mgr = FaissIndexManager(
        feature_dim=feature_dim,
        output_directory=str(out_dir),
        filename_stem="val",
        index_type="flat",
        model_name="unit",
        faiss_config={"total_indexes_per_file": shard_cap},
        project_root_path=ROOT,
    )
    # Start from a clean state
    mgr.clear_index(save_empty=True)
    # Add 12 vectors -> expect 3 shards (5,5,2)
    rng = np.random.default_rng(42)
    vecs = rng.normal(size=(12, feature_dim)).astype(np.float32)
    # L2 normalization for distance->similarity conversion correctness
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms
    mgr.add_vectors_to_index(vecs, paths)
    mgr.save_index()

    # Validate
    total = mgr.get_total_indexed_items()
    idents = getattr(mgr, "index_identifiers", [str(mgr.index_path)])
    print(f"[FAISS] Total items: {total}")
    print(f"[FAISS] Identifiers: {idents}")
    print(f"[FAISS] Shard count: {len(idents)}")

    # Cross-shard search
    q = vecs[0]
    res: List[Tuple[str, float]] = mgr.search_similar_images(q, top_k=5)
    print(f"[FAISS] Search top-5 results: {[(Path(p).name, round(s,4)) for p,s in res]}")

# ---------------- Qdrant Validation ----------------

def validate_qdrant_sharding() -> None:
    print("[QDRANT] Validation started")
    try:
        from hyundai_document_authenticator.core_engine.image_similarity_system.qdrant_manager import (
            QdrantManager,
        )
    except Exception as e_imp:
        print(f"[QDRANT] Skipped: import failed: {e_imp}")
        return

    feature_dim = 16
    shard_cap = 5
    img_dir = ROOT / "instance" / "database_images_test"
    paths = ensure_test_images(img_dir, 12)

    qconf = {"location": str(ROOT / "instance" / "qdrant_db"), "max_points_per_collection": shard_cap}
    mgr = QdrantManager(
        feature_dim=feature_dim,
        collection_name_stem="val_coll",
        model_name="unit",
        qdrant_config=qconf,
        project_root_path=ROOT,
    )
    # Clean base collections for this test
    mgr.clear_index()

    # Upsert 12 vectors -> expect 3 collections
    rng = np.random.default_rng(123)
    vecs = rng.normal(size=(12, feature_dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms
    try:
        # use internal helper for rollover to validate sharding behavior without feature extractor
        mgr._upsert_with_shard_rollover(vecs, paths)  # type: ignore[attr-defined]
    except Exception as e_up:
        print(f"[QDRANT] Upsert failed: {e_up}")
        return

    total = mgr.get_total_indexed_items()
    names = getattr(mgr, "collection_names", [mgr.collection_name])
    print(f"[QDRANT] Total items: {total}")
    print(f"[QDRANT] Collections: {names}")
    print(f"[QDRANT] Collection count: {len(names)}")

    # Search across collections
    try:
        res = mgr.search_similar_images(vecs[0], top_k=5)
        print(f"[QDRANT] Search top-5 results: {[(Path(p).name, round(s,4)) for p,s in res]}")
    except Exception as e_s:
        print(f"[QDRANT] Search failed: {e_s}")

if __name__ == "__main__":
    try:
        validate_faiss_sharding()
    except Exception as e:
        print(f"[FAISS] Validation error: {e}")
    try:
        validate_qdrant_sharding()
    except Exception as e:
        print(f"[QDRANT] Validation error: {e}")
