#!/usr/bin/env python3
"""Smoke test and demo runner for the `tif_searcher` package.

Overview:
    This module validates OCR-driven text search using the vendored
    `tif_searcher_core` APIs. It supports two execution paths:

    1) TifTextSearcher (default; --engine searcher)
       - Object-oriented workflow that honors zonal OCR via the
         `search_location` config (top/bottom/center fractions of the page).
       - Preferred for production: composable, testable, and flexible
         (e.g., per-zone debug logging, early exit upon match, batch helpers).

    2) Legacy workflow (--engine legacy)
       - Thin CLI-style runner that OCRs the entire page (ignores search_location)
         and prints a human-friendly console report.
       - Useful for quick demos, simple pipelines, and compatibility with
         existing automation expecting this console output format.

Usage examples:
    # Default (TifTextSearcher) scanning full image
    python test_tif_searcher.py --input data_real

    # TifTextSearcher with zonal OCR: search only the top 30% of each page
    python test_tif_searcher.py --input data_real --search-top 0.3

    # TifTextSearcher with combined zones and custom search text
    python test_tif_searcher.py --input data_real --text "가맹점 실사 사진" \
        --search-top 0.05 --search-center 0.2

    # TifTextSearcher with JSON override for search_location
    python test_tif_searcher.py --search-location-json "{\"top\": 0.1, \"center\": 0.2}"

    # Legacy full-page workflow (ignores search_location)
    python test_tif_searcher.py --engine legacy --input data_real

    # Legacy with custom text (still scans full page)
    python test_tif_searcher.py --engine legacy --text "가맹점 실사 사진"

Engine selection rationale:
    - Use TifTextSearcher (default) when you need performance/precision via zonal
      OCR, reusable objects, or programmatic access to per-page results.
    - Use the legacy workflow for minimal, quick demonstrations or to retain
      backward compatibility with scripts parsing its console output.

Args (CLI):
    --input (str): Folder or file to search under. Defaults to "data_real".
    --text (str): Override the search text phrase.
    --search-top (float): Fraction [0,1] of image height from top to search.
    --search-bottom (float): Fraction [0,1] from bottom to search.
    --search-center (float): Fraction [0,1] of central band height to search.
    --search-full-image (flag): Clear search_location and scan the full page.
    --search-location-json (str): JSON dict for search_location, e.g., "{\"top\":0.1}".
    --engine (str): Execution mode. "searcher" (default) uses TifTextSearcher
        with zonal OCR; "legacy" runs the full-page CLI workflow (ignores
        search_location).

Returns:
    None. Prints a concise summary and exits with a process status code
    (0 on success, non-zero on error).

Raises:
    None. All errors are reported to the console and converted to non-zero exit codes.

Behavior notes:
    - Loads .env from the repository root (if present).
    - Resolves relative paths against the package root (hyundai_document_authenticator/).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv, find_dotenv

# Establish import context: add package root (the directory containing `external/`).
THIS_FILE: Path = Path(__file__).resolve()
# For: hyundai_document_authenticator/external/tif_searcher/test_tif_searcher.py
# parents[0]=tif_searcher, [1]=external, [2]=hyundai_document_authenticator
PKG_ROOT: Path = THIS_FILE.parents[2]
REPO_ROOT: Path = PKG_ROOT.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

# Load .env early

def _load_project_dotenv() -> Optional[Path]:
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

# Ensure relative paths resolve against the package root to avoid path issues
# when invoking this script from different working directories.
try:
    os.chdir(str(PKG_ROOT))
except Exception:
    pass

# Provide a runtime alias package 'tif_searcher' that points to the vendored
# 'tif_searcher_core' directory, so imports like 'tif_searcher.config' work.
try:
    import types as _types
    _core_dir = (THIS_FILE.parent / "tif_searcher_core").resolve()
    if "tif_searcher" not in sys.modules:
        _pkg = _types.ModuleType("tif_searcher")
        _pkg.__path__ = [str(_core_dir)]  # make it a namespace/package
        sys.modules["tif_searcher"] = _pkg
except Exception:
    pass

# Configure logging similar to production setup
try:
    from core_engine.image_similarity_system.log_utils import setup_logging  # type: ignore
    setup_logging()
except Exception:
    pass

# Import tif_searcher API
from external.tif_searcher.tif_searcher_core.config import config as default_config  # type: ignore
from external.tif_searcher.tif_searcher_core.searcher import TifTextSearcher  # type: ignore
from external.tif_searcher.tif_searcher_core.utils import get_image_paths  # type: ignore
# Legacy, full-page CLI-style workflow (does not honor search_location)
from external.tif_searcher.tif_searcher_core.ocr_engines import get_ocr_engine  # type: ignore
from external.tif_searcher.tif_searcher_core.main import run_text_search_workflow  # type: ignore


def _resolve_path(p: Path) -> Path:
    return p if p.is_absolute() else (PKG_ROOT / p).resolve()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test for external.tif_searcher package.")
    parser.add_argument("--input", type=str, default="data_real", help="Folder or file to search under")
    parser.add_argument("--text", type=str, default="", help="Override search text")
    # Search-area overrides
    parser.add_argument("--search-top", type=float, default=None, help="Fraction (0-1) of image height at the top to search")
    parser.add_argument("--search-bottom", type=float, default=None, help="Fraction (0-1) of image height at the bottom to search")
    parser.add_argument("--search-center", type=float, default=None, help="Fraction (0-1) of central height to search")
    parser.add_argument("--search-full-image", action="store_true", help="Search the full image; override search_location to {}")
    parser.add_argument("--search-location-json", type=str, default="", help='JSON dict for search_location, e.g., "{\"top\": 0.1}"')
    # Engine selector: 'searcher' uses TifTextSearcher with zonal OCR;
    # 'legacy' runs the original CLI-style workflow that OCRs full pages.
    parser.add_argument(
        "--engine",
        choices=["searcher", "legacy"],
        default="searcher",
        help=(
            "Implementation to run: 'searcher' (TifTextSearcher with zonal OCR, recommended) "
            "or 'legacy' (run_text_search_workflow full-page, simpler: fast demos/compat)."
        ),
    )
    args = parser.parse_args(argv)

    # Prepare config
    cfg = dict(default_config)
    input_path = _resolve_path(Path(args.input))
    cfg["input_path"] = str(input_path)
    cfg.setdefault("output_folder", str((PKG_ROOT / "search_results").resolve()))
    cfg.setdefault("create_csv_report", False)

    if args.text:
        cfg["search_text"] = args.text

    # search_location override logic
    loc_override = None
    if args.search_full_image:
        loc_override = {}
    elif args.search_location_json:
        try:
            data = json.loads(args.search_location_json)
            if isinstance(data, dict):
                loc_override = {k: float(v) for k, v in data.items() if k in {"top", "bottom", "center"}}
        except Exception:
            loc_override = None
    else:
        loc = {}
        if args.search_top is not None:
            loc["top"] = float(args.search_top)
        if args.search_bottom is not None:
            loc["bottom"] = float(args.search_bottom)
        if args.search_center is not None:
            loc["center"] = float(args.search_center)
        if loc:
            loc_override = loc
    if loc_override is not None:
        cfg["search_location"] = loc_override

    # SECTION: Engine dispatch
    #   - 'searcher' (default): Uses TifTextSearcher (object-oriented API) which supports
    #     zonal OCR via cfg["search_location"]. This is recommended for production
    #     because it is composable, testable, and more flexible (e.g., per-zone logging,
    #     early exit on match, batch extraction helpers).
    #   - 'legacy': Runs the original CLI-style workflow run_text_search_workflow,
    #     which always OCRs whole pages and prints a human-friendly report.
    #     It is useful for quick demonstrations, smoke tests, or to maintain
    #     compatibility with older automation where the simple stdout format matters.
    if args.engine == "legacy":
        # NOTE: The legacy workflow ignores search_location. If the user provided
        # zonal overrides, call them out explicitly so expectations are clear.
        if cfg.get("search_location"):
            print("[Info] Ignoring search_location overrides in legacy mode; full-page OCR will be used.")
            try:
                del cfg["search_location"]
            except Exception:
                pass
        try:
            ocr_engine = get_ocr_engine(cfg)
        except Exception as exc:
            print(f"Failed to initialize OCR engine: {exc}")
            return 1
        try:
            # SECTION: Legacy full-page workflow example
            # Why/when to use:
            # - Minimal, single-call entrypoint for demos and simple pipelines.
            # - Stable, human-readable console output without object orchestration.
            # Limitations vs TifTextSearcher:
            # - No zonal OCR (cannot restrict OCR to top/bottom/center slices).
            # - Less composable (no direct per-page result aggregation APIs).
            run_text_search_workflow(ocr_engine, cfg)
            return 0
        except KeyboardInterrupt:
            print("Interrupted by user.")
            return 130
        except Exception as exc:
            print(f"Error during legacy workflow: {exc}")
            return 1

    # Default path: TifTextSearcher honoring zonal OCR
    try:
        searcher = TifTextSearcher(
            search_text=cfg.get("search_text"),
            language=cfg.get("language"),
            ocr_backend=cfg.get("ocr_engine"),
            search_mode=cfg.get("search_mode"),
            allow_normalization=cfg.get("allow_recognition_normalization"),
            remove_spaces_in_normalization=cfg.get("remove_spaces_in_normalization"),
            use_angle_cls=cfg.get("use_angle_cls"),
            use_gpu_for_paddle=cfg.get("use_gpu_for_paddle") or cfg.get("use_gpu"),
            paddle_batch_size=cfg.get("paddle_batch_size") or cfg.get("rec_batch_num"),
            use_offline_models=cfg.get("use_offline_models"),
            download_path_det_model=cfg.get("download_path_det_model"),
            download_path_rec_model=cfg.get("download_path_rec_model"),
            download_path_cls_model=cfg.get("download_path_cls_model"),
            offline_paddle_det_model_dir=cfg.get("offline_paddle_det_model_dir") or cfg.get("offline_det_model_dir"),
            offline_paddle_rec_model_dir=cfg.get("offline_paddle_rec_model_dir") or cfg.get("offline_rec_model_dir"),
            offline_paddle_cls_model_dir=cfg.get("offline_paddle_cls_model_dir") or cfg.get("offline_cls_model_dir"),
            offline_easyocr_model_dir=cfg.get("offline_model_dir"),
            search_location=cfg.get("search_location", {}),
            recognized_text_debug=cfg.get("recognized_text_debug"),
        )
    except Exception as exc:
        print(f"Failed to initialize TifTextSearcher: {exc}")
        return 1

    try:
        image_paths = get_image_paths(input_path, cfg.get("supported_formats", [".tif", ".tiff"]))
        if not image_paths:
            print("No images found to process.")
            return 0
        total_found = 0
        for i, img_path in enumerate(image_paths, 1):
            # Process only TIF/TIFF with page semantics for this test
            lower = str(img_path).lower()
            if not (lower.endswith(".tif") or lower.endswith(".tiff")):
                continue
            print(f"\nProcessing TIF {i}/{len(image_paths)}: {img_path}")
            try:
                pages = searcher.find_text_pages(img_path)
                if pages:
                    total_found += 1
                    pages_str = ", ".join(map(str, pages))
                    print(f"  ✔️ Found text on page(s): {pages_str}")
                else:
                    print("  ➖ No matching pages found.")
            except Exception as exc2:
                print(f"  ❌ ERROR processing file: {exc2}")
        print("\n=== Summary ===")
        print(f"TIFs scanned: {len([p for p in image_paths if str(p).lower().endswith(('.tif', '.tiff'))])}")
        print(f"TIFs with matches: {total_found}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted by user.")
        return 130
    except Exception as exc:
        print(f"Error during tif_searcher scan: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
