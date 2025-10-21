#!/usr/bin/env python3
"""Smoke test and demo runner for the `photo_extractor` module.

This script validates that photo extraction works over a small TIF sample using
either:
- BBox mode (default): crops photos using bounding boxes
- YOLO mode: detects photos with a YOLO model and crops them

Run examples:
  # BBox mode (auto-detect first TIF under data_real, page 1)
  python test_photo_extractor.py

  # BBox mode with explicit TIF and page
  python test_photo_extractor.py --mode bbox --tif-path data_real/N2024030602104THA00100001_12.tif --page 1

  # YOLO mode (requires model path)
  python test_photo_extractor.py --mode yolo --tif-path data_real --page 1 \
    --yolo-model trained_model/yolo_photo_extractor/weights/best.pt

Behavior notes:
- Loads .env from the repository root (if present).
- Resolves relative paths against the package root (hyundai_document_authenticator/).
- Emits a concise summary and exits non-zero on failure.
"""
from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from dotenv import load_dotenv, find_dotenv

# Establish import context: add package root (the directory containing `external/`).
THIS_FILE: Path = Path(__file__).resolve()
MODULE_DIR: Path = THIS_FILE.parent
# For: hyundai_document_authenticator/external/photo_extractor/test_photo_extractor.py
# parents[0]=photo_extractor, [1]=external, [2]=hyundai_document_authenticator
PKG_ROOT: Path = THIS_FILE.parents[2]
REPO_ROOT: Path = PKG_ROOT.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

# Load .env early

def _load_project_dotenv() -> Optional[Path]:
    """Locate and load a .env file from CWD upward or repository root.

    Returns:
        Optional[Path]: Resolved path to the loaded .env file if found; otherwise None.
    """
    try:
        path_str = find_dotenv(filename=".env", usecwd=True)
        if path_str:
            load_dotenv(dotenv_path=path_str, override=False)
            return Path(path_str).resolve()
        candidate = REPO_ROOT / ".env"
        if candidate.exists():
            load_dotenv(dotenv_path=str(candidate), override=False)
            return candidate.resolve()
    except Exception:
        return None
    return None

_ = _load_project_dotenv()

# Ensure relative paths in configs resolve against the package root
try:
    os.chdir(str(PKG_ROOT))
except Exception:
    pass

# Configure logging similar to production setup
try:
    from core_engine.image_similarity_system.log_utils import setup_logging  # type: ignore
    setup_logging()
except Exception:
    pass

# Import extractor after sys.path/env are set
from external.photo_extractor import PhotoExtractor  # type: ignore
try:
    from external.photo_extractor.photo_extractor.config import DEFAULT_CONFIG as PE_DEFAULT_CONFIG  # type: ignore
except Exception:
    PE_DEFAULT_CONFIG = {}  # type: ignore


def _resolve_path(p: Path) -> Path:
    """Resolve a possibly relative path against the package root."""
    return p if p.is_absolute() else (PKG_ROOT / p).resolve()


def _find_first_tif(under: Path) -> Optional[Path]:
    """Find the first .tif/.tiff under a directory or return the file if a TIF path.

    Args:
        under: Path to a directory or a TIF/TIFF file.

    Returns:
        Optional[Path]: Resolved path to a TIF file or None if not found.
    """
    p = _resolve_path(under)
    if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}:
        return p
    if p.is_dir():
        for ext in ("*.tif", "*.tiff"):
            matches = sorted(p.rglob(ext))
            if matches:
                return matches[0].resolve()
    return None


def _central_bbox_for(image: Image.Image, frac_w: float = 0.4, frac_h: float = 0.4) -> List[int]:
    """Compute a central xyxy bbox covering a fraction of the image.

    Args:
        image: PIL image (RGB).
        frac_w: Fraction of width to cover.
        frac_h: Fraction of height to cover.

    Returns:
        List[int]: [x1, y1, x2, y2] integer bbox.
    """
    W, H = image.size
    bw, bh = int(W * frac_w), int(H * frac_h)
    x1 = (W - bw) // 2
    y1 = (H - bh) // 2
    x2 = x1 + bw
    y2 = y1 + bh
    return [max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)]


def _bbox_params_from_config_or_fallback(image: Image.Image, bbox_list_json: Optional[str]) -> Tuple[List[List[int]], str, bool]:
    """Resolve bbox params from CLI/DEFAULT_CONFIG with a robust fallback.

    If no bbox list is available, a central bbox is generated to enable a
    minimal smoke test run.

    Args:
        image: PIL image for computing a fallback bbox.
        bbox_list_json: Optional JSON string for bbox list.

    Returns:
        Tuple[List[List[int]], str, bool]: (bbox_list, bbox_format, normalized)
    """
    if bbox_list_json:
        try:
            bboxes = json.loads(bbox_list_json)
            return bboxes, "xyxy", False
        except Exception:
            pass

    try:
        bcfg = PE_DEFAULT_CONFIG.get("bbox_extraction", {})  # type: ignore[attr-defined]
        bboxes = bcfg.get("bbox_list") or []
        bformat = (bcfg.get("bbox_format") or "xyxy").lower()
        bnorm = bool(bcfg.get("normalized", False))
        if bboxes:
            return bboxes, bformat, bnorm
    except Exception:
        pass

    # Fallback: create a central bbox to ensure the smoke test extracts something
    return [_central_bbox_for(image)], "xyxy", False


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for running the photo_extractor smoke test.

    Args:
        argv: Optional CLI arguments list.

    Returns:
        int: Exit code (0 success, non-zero failure).
    """
    parser = argparse.ArgumentParser(description="Smoke test for external.photo_extractor module.")
    parser.add_argument("--mode", choices=["bbox", "yolo"], default="bbox", help="Extraction mode")
    parser.add_argument("--tif-path", type=str, default="data_real", help="Path to a TIF file or a directory")
    parser.add_argument("--page", type=int, default=1, help="1-based page index to process")
    parser.add_argument("--yolo-model", type=str, default="", help="Path to YOLO model (.pt) for yolo mode")
    parser.add_argument("--bbox-list", type=str, default="", help="JSON list of bbox coordinates for bbox mode")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MODULE_DIR / "test_outputs" / "photo_extractor"),
        help="Output directory for extracted crops",
    )

    args = parser.parse_args(argv)
    mode = args.mode.lower()

    # Resolve inputs
    tif_candidate = _find_first_tif(Path(args.tif_path))
    if tif_candidate is None:
        print(f"No TIF found under: {args.tif_path}")
        return 2

    output_dir = _resolve_path(Path(args.output_dir)) / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        extractor = PhotoExtractor()

        if mode == "yolo":
            if not args.yolo_model:
                print("YOLO mode requires --yolo-model path to a .pt file.")
                return 2
            config_override = {"yolo_object_detection": {"model_path": str(_resolve_path(Path(args.yolo_model)))}}
            extractor = PhotoExtractor(config_override=config_override)
            photos = extractor.extract_photos(tif_candidate, int(args.page))
        else:
            with Image.open(tif_candidate) as tif_img:
                total_pages = getattr(tif_img, "n_frames", 1)
                if not (1 <= args.page <= total_pages):
                    print(f"Page {args.page} out of bounds. File has {total_pages} pages.")
                    return 2
                tif_img.seek(int(args.page) - 1)
                page_image = tif_img.convert("RGB")
                bboxes, bbox_format, normalized = _bbox_params_from_config_or_fallback(page_image, args.bbox_list or None)
                photos = extractor.extract_photos_from_bboxes(page_image, bboxes, bbox_format=bbox_format, normalized=normalized)

        saved = 0
        for i, im in enumerate(photos):
            save_path = output_dir / f"{tif_candidate.stem}_page{int(args.page)}_photo{i}.jpg"
            try:
                im.save(save_path, "JPEG")
                saved += 1
            except Exception:
                pass

        if saved == 0:
            print("No crops were saved. Smoke test did not find any photos to extract.")
            return 1

        print(f"Saved {saved} crops to: {output_dir}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130
    except Exception as exc:
        print(f"Error running photo_extractor smoke test: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
