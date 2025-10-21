"""
TIF-specific utilities to support batch TIF workflows.
- Parent document derivation from DB image stems
- sim_img_check mapping construction
- TIF preview generation helpers
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


def parent_doc_from_db_stem(stem: str) -> str:
    """Return stem up to and including '_pageY' and optional '_photoZ', no extension.

    Examples:
    - '..._12_page1_photo0_extra' -> '..._12_page1_photo0'
    - '..._12_page1' -> '..._12_page1'
    - '..._12' -> '..._12' (no page/photo found)
    """
    m = re.search(r"_page\d+(?:_photo\d+)?", stem)
    if m:
        return stem[: m.end()]
    return stem


def generate_virtual_crop_name(tif_stem: str, page_num: int, idx: int) -> str:
    """Generate a stable, virtual filename for a TIF crop without persisting a file.

    Parameters
    ----------
    tif_stem : str
        The stem (name without extension) of the source TIF document.
    page_num : int
        1-based page number from which the crop was created.
    idx : int
        Zero-based index of the crop on that page.

    Returns
    -------
    str
        A deterministic virtual filename in the form '<stem>_page<page>_photo<idx>.jpg'.
    """
    return f"{tif_stem}_page{int(page_num)}_photo{int(idx)}.jpg"


def build_sim_img_checks_map(
    final_top_documents: List[Dict[str, Any]],
    per_doc_neighbors: Dict[str, List[Dict[str, Any]]],
    top_k: int,
    max_k_cap: Optional[int] = None,
) -> Dict[str, Any]:
    """Construct sim_img_check payload for each parent doc in final_top_documents.

    Picks up to effective_k images per doc (unique by path with max score kept).
    """
    effective_k = min(top_k, int(max_k_cap)) if max_k_cap is not None else top_k
    sim_checks_map: Dict[str, Any] = {}
    for item in final_top_documents:
        doc = item["document"]
        neigh_list = per_doc_neighbors.get(doc)
        if not neigh_list:
            sim_checks_map[doc] = {"note": "not available"}
            continue
        agg: Dict[str, float] = {}
        for n in neigh_list:
            pth = n["path"]
            sc = float(n["score"])
            if pth in agg:
                if sc > agg[pth]:
                    agg[pth] = sc
            else:
                agg[pth] = sc
        top_items = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:effective_k]
        # Use a mapping for fast key-based access in JSONB and simpler querying
        sim_checks_map[doc] = {
            "parent_document_name": doc,
            "explanatory_db_images": {p: s for p, s in top_items},
        }
    return sim_checks_map


def generate_tif_preview(src_tif: Path, dest_jpg: Path, size=(900, 900)) -> None:
    """Generate a JPG preview for the first page of a TIF document."""
    try:
        with Image.open(src_tif) as _img:
            _img.seek(0)
            page = _img.convert("RGB")
            page.thumbnail(size)
            dest_jpg.parent.mkdir(parents=True, exist_ok=True)
            page.save(dest_jpg, "JPEG", quality=85)
    except Exception as e_prev:
        logger.warning("Failed to create preview for '%s': %s", src_tif, e_prev)
