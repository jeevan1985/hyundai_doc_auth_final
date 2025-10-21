#!/usr/bin/env python
"""
Standalone Photo Extraction Utility (Refactored)

This script provides a command-line interface to the refactored PhotoExtractor
library, supporting both YOLO-based and BBox-based extraction from TIF pages.

Usage (YOLO Mode):
    python photo_extractor.py yolo --tif-path path/to/doc.tif --page-numbers 3 7 \
        --model-path models/photo_extractor/best.pt

Usage (BBox Mode):
    python photo_extractor.py bbox --tif-path path/to/doc.tif --page-numbers 3 7 \
        --bbox-list '[[50, 60, 300, 260]]'
"""
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import typer
from PIL import Image, ImageSequence

# --- Environment Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from external.photo_extractor import PhotoExtractor
from external.photo_extractor.photo_extractor.config import DEFAULT_CONFIG as PE_DEFAULT_CONFIG

# --- App Initialization ---
app = typer.Typer(help="Photo extraction utility using the PhotoExtractor library.")


def configure_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s - %(message)s")

@app.command(name="yolo", help="Extract photos using a YOLO model.")
def run_yolo_extraction(
    tif_path: Path = typer.Option(..., exists=True, help="Path to a TIF file or a folder of TIF files."),
    page_numbers: Optional[List[int]] = typer.Option(None, "--page-numbers", help="1-based page indices. Use --all-pages to process all pages."),
    all_pages: bool = typer.Option(False, "--all-pages", help="Process all pages in each TIF. If False, --page-numbers is required."),
    model_path: str = typer.Option(..., help="Path to the YOLOv8 model (.pt)."),
    output_dir: Path = typer.Option(Path("./extracted_crops/yolo"), help="Output directory."),
    verbose: bool = typer.Option(False, "-v", help="Enable debug logging.")
):
    configure_logging(verbose)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Running in YOLO mode. Output directory: {output_dir}")

    # Build list of TIFs to process
    tif_files: List[Path] = []
    if tif_path.is_dir():
        tif_files = sorted([p for p in tif_path.glob("*.tif")] + [p for p in tif_path.glob("*.tiff")])
    else:
        tif_files = [tif_path]

    failed_log_path = output_dir / "failed_pages.txt"
    failed_entries: List[str] = []

    try:
        config = {"yolo_object_detection": {"model_path": model_path}}
        extractor = PhotoExtractor(config_override=config)

        for tif_file in tif_files:
            try:
                with Image.open(tif_file) as tif_img:
                    total_pages = getattr(tif_img, 'n_frames', 1)

                    if not page_numbers and not all_pages:
                        logging.error("You must provide --page-numbers or specify --all-pages.")
                        raise typer.Exit(code=2)

                    pages_to_process = (list(range(1, total_pages + 1)) if all_pages else list(page_numbers or []))

                    for page_num in pages_to_process:
                        if not (1 <= page_num <= total_pages):
                            msg = f"{tif_file.name}: Page {page_num} is out of bounds (total {total_pages}). Skipping."
                            logging.warning(msg)
                            failed_entries.append(msg)
                            continue

                        # Use the high-level extractor which internally opens and reads the page
                        photos = extractor.extract_photos(tif_file, page_num)
                        for i, photo in enumerate(photos):
                            save_path = output_dir / f"{tif_file.stem}_page{page_num}_photo{i}.jpg"
                            photo.save(save_path, "JPEG")
                        logging.info(f"{tif_file.name}: Saved {len(photos)} YOLO crops from page {page_num}.")
            except Exception as e:
                logging.error(f"Failed processing TIF: {tif_file} -> {e}", exc_info=True)
                failed_entries.append(f"{tif_file.name}: {e}")
    except Exception as e:
        logging.error(f"YOLO extraction failed: {e}", exc_info=True)
        raise typer.Exit(code=1)
    finally:
        if failed_entries:
            try:
                with open(failed_log_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(failed_entries))
                logging.info(f"Failed page indexes were logged to: {failed_log_path}")
            except Exception:
                pass

@app.command(name="bbox", help="Extract photos using predefined bounding boxes.")
def run_bbox_extraction(
    tif_path: Path = typer.Option(..., exists=True, help="Path to a TIF file or a folder of TIF files."),
    page_numbers: Optional[List[int]] = typer.Option(None, "--page-numbers", help="1-based page indices. Use --all-pages to process all pages."),
    all_pages: bool = typer.Option(False, "--all-pages", help="Process all pages in each TIF. If False, --page-numbers is required."),
    bbox_list_json: Optional[str] = typer.Option(None, "--bbox-list", help='JSON string of bounding boxes, e.g., "[[50,60,300,260]]". If omitted, values are read from config.'),
    bbox_format: Optional[str] = typer.Option(None, help="BBox format: xyxy, xywh, or cxcywh. If omitted, read from config."),
    normalized: Optional[bool] = typer.Option(None, help="Set if coordinates are normalized [0,1]. If omitted, read from config."),
    output_dir: Path = typer.Option(Path("./extracted_crops/bbox"), help="Output directory."),
    verbose: bool = typer.Option(False, "-v", help="Enable debug logging.")
):
    """BBox-based extraction with config fallback and directory support.

    - If --bbox-list/--bbox-format/--normalized are not provided, values are
      sourced from external.photo_extractor.photo_extractor.config.DEFAULT_CONFIG.
    - If --tif-path points to a directory, all .tif/.tiff files are processed.
    - If a page index is invalid, it is skipped with a warning.
    """
    configure_logging(verbose)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Running in BBox mode. Output directory: {output_dir}")

    # Resolve bbox parameters from CLI or config
    try:
        if bbox_list_json:
            bboxes = json.loads(bbox_list_json)
        else:
            bboxes = PE_DEFAULT_CONFIG.get("bbox_extraction", {}).get("bbox_list", [])
        b_format = (bbox_format or PE_DEFAULT_CONFIG.get("bbox_extraction", {}).get("bbox_format", "xyxy")).lower()
        b_norm = normalized if normalized is not None else bool(PE_DEFAULT_CONFIG.get("bbox_extraction", {}).get("normalized", False))
    except Exception as e:
        logging.error(f"Failed to resolve BBox parameters: {e}")
        raise typer.Exit(code=1)

    if not bboxes:
        logging.error("No bounding boxes provided via --bbox-list and none found in config. Aborting.")
        raise typer.Exit(code=1)

    # Build list of TIFs to process
    tif_files: List[Path] = []
    if tif_path.is_dir():
        tif_files = sorted([p for p in tif_path.glob("*.tif")] + [p for p in tif_path.glob("*.tiff")])
    else:
        tif_files = [tif_path]

    extractor = PhotoExtractor()  # No YOLO config needed for bbox mode
    failed_log_path = output_dir / "failed_pages.txt"
    failed_entries: List[str] = []

    try:
        for tif_file in tif_files:
            with Image.open(tif_file) as tif_img:
                total_pages = getattr(tif_img, 'n_frames', 1)
                if not page_numbers and not all_pages:
                    logging.error("You must provide --page-numbers or specify --all-pages.")
                    raise typer.Exit(code=2)
                pages_to_process = (list(range(1, total_pages + 1)) if all_pages else list(page_numbers or []))

                for page_num in pages_to_process:
                    if not (1 <= page_num <= total_pages):
                        msg = f"{tif_file.name}: Page {page_num} is out of bounds (total {total_pages}). Skipping."
                        logging.warning(msg)
                        failed_entries.append(msg)
                        continue

                    tif_img.seek(page_num - 1)
                    page_image = tif_img.convert("RGB")

                    photos = extractor.extract_photos_from_bboxes(
                        page_image, bboxes, bbox_format=b_format, normalized=b_norm
                    )
                    for i, photo in enumerate(photos):
                        save_path = output_dir / f"{tif_file.stem}_page{page_num}_photo{i}.jpg"
                        photo.save(save_path, "JPEG")
                    logging.info(f"{tif_file.name}: Saved {len(photos)} BBox crops from page {page_num}.")
    except json.JSONDecodeError:
        logging.error("Invalid JSON for --bbox-list. Please provide a valid list of lists.")
        raise typer.Exit(code=1)
    except Exception as e:
        logging.error(f"BBox extraction failed: {e}", exc_info=True)
        raise typer.Exit(code=1)
    finally:
        if failed_entries:
            try:
                with open(failed_log_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(failed_entries))
                logging.info(f"Failed page indexes were logged to: {failed_log_path}")
            except Exception:
                pass

if __name__ == "__main__":
    app()
