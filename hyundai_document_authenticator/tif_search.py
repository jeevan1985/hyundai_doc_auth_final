#!/usr/bin/env python
"""
TIF Document Page Search CLI

A production-grade, single-file CLI built on Typer for searching multi-page TIF
(or other supported image formats) for pages containing specific text. It uses
an underlying OCR-based searcher and provides extensive runtime customization
via CLI flags. Defaults are sourced from the vendored configuration at:
    external.tif_searcher.tif_searcher_core.config

Any CLI option provided here will override the corresponding default for this
invocation only.

Key capabilities:
- Single file mode: Process a single TIF
- Directory mode: Recursively scan a directory for supported image formats
- OCR engine selection: PaddleOCR, EasyOCR, or Tesseract
- Zonal OCR: Restrict OCR to top/bottom/center regions for performance
- CSV reporting: Optional detailed run report
- Saving matched pages as images

Examples
- Single file
    python tif_search.py --tif-path path/to/doc.tif --search-text "My Title"
- Directory (batch)
    python tif_search.py --tif-path path/to/dir --search-text "My Title"
- Let the CLI use configured input_path from config when --tif-path is omitted
    python tif_search.py --search-text "Some Text"

Notes
- Help output is rendered with Rich (when available) and includes grouped help
  panels for better discoverability.
- Use --help to view all available options.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import typer

# === Environment Setup ========================================================
# Ensure the local project root is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the search implementation and defaults
from external.tif_searcher import TifTextSearcher
from external.tif_searcher.tif_searcher_core.config import config as default_config

# === App Initialization =======================================================
APP_NAME = "TIF Search"
APP_VERSION = "1.0.0"

app = typer.Typer(
    help=(
        "[bold cyan]TIF page search utility[/bold cyan] using the "
        "[bold]TifTextSearcher[/bold] library.\n\n"
        "- Supports both single-file and recursive directory processing\n"
        "- Configurable OCR engines (PaddleOCR, EasyOCR, Tesseract)\n"
        "- Zonal OCR to speed up matching\n"
        "- Rich, colored output and grouped help panels"
    ),
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode="rich",
)


# === Utilities ================================================================

def configure_logging(verbose: bool) -> None:
    """Configure root logging for the CLI.

    Args:
        verbose: If True, configure logging at DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s - %(message)s")


def _normalize_exts(exts: Sequence[str] | None) -> List[str]:
    """Normalize file extensions: ensure lowercase and start with a dot.

    Args:
        exts: A sequence of extensions (e.g., [".tif", "jpg"]) or None.

    Returns:
        A list of normalized extensions like [".tif", ".jpg"].
    """
    if not exts:
        return []
    norm: List[str] = []
    for e in exts:
        if not e:
            continue
        e = e.strip().lower()
        if not e:
            continue
        if not e.startswith('.'):
            e = f'.{e}'
        norm.append(e)
    return sorted(set(norm))


def _iter_image_files_in_dir(directory: Path, allowed_exts: Sequence[str]) -> List[Path]:
    """Recursively iterate image files with allowed extensions under directory.

    Args:
        directory: Root directory to recursively scan.
        allowed_exts: Allowed file extensions (lowercase, with leading dots).

    Returns:
        Sorted list of file Paths.
    """
    exts = set(_normalize_exts(allowed_exts))
    return sorted([p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def _save_pages(
    tif_file: Path,
    page_numbers: List[int],
    output_dir: Path,
    img_format: str = "png",
) -> List[Path]:
    """Save specific pages from a multi-page TIF to an output directory.

    Pages are saved using a standardized naming scheme: "<stem>_page{N}.<ext>".

    Args:
        tif_file: The source multi-page TIF file.
        page_numbers: 1-based page numbers to save.
        output_dir: Directory where images will be written. Created if missing.
        img_format: Output format (e.g., "png", "jpg", "jpeg").

    Returns:
        A list of Paths to the saved images.
    """
    from PIL import Image

    saved: List[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = img_format.strip().lower()
    with Image.open(tif_file) as img:
        for page_num in page_numbers:
            try:
                img.seek(page_num - 1)
            except Exception:
                continue
            base_name = tif_file.stem
            out_name = f"{base_name}_page{page_num}.{fmt}"
            save_path = output_dir / out_name
            # Always convert to RGB for consistent downstream behavior
            img.convert("RGB").save(save_path, format=fmt.upper())
            saved.append(save_path)
    return saved


# === Command =================================================================
@app.command()
def main(
    # --- Input & Execution ----------------------------------------------------
    tif_path: Optional[Path] = typer.Option(
        None,
        "--tif-path",
        "-t",
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help=(
            "Path to a TIF file or a directory to scan recursively. "
            "If omitted, defaults to [bold]config.input_path[/bold] (from vendored config)."
        ),
        rich_help_panel="Input & Execution",
    ),
    supported_formats: Optional[List[str]] = typer.Option(
        None,
        "--supported-format",
        "-S",
        help=(
            "Override the set of file extensions to scan in directory mode. "
            "Provide multiple times, e.g.: -S .tif -S .tiff -S .jpg. "
            "Defaults to [bold]config.supported_formats[/bold]."
        ),
        rich_help_panel="Input & Execution",
    ),

    # --- Output ---------------------------------------------------------------
    output_folder: Optional[Path] = typer.Option(
        None,
        "--output-folder",
        "-o",
        help=(
            "Directory to save matched pages. Defaults to "
            "[bold]config.output_folder[/bold]."
        ),
        rich_help_panel="Output",
    ),
    img_format: str = typer.Option(
        "png",
        "--img-format",
        "-f",
        case_sensitive=False,
        help=(
            "Image format for saved pages. Common values: png, jpg, jpeg. "
            "Saved files use the naming '<stem>_pageN.<ext>'."
        ),
        rich_help_panel="Output",
        show_default=True,
    ),
    create_csv_report: Optional[bool] = typer.Option(
        None,
        "--create-csv-report/--no-create-csv-report",
        help=(
            "Generate a CSV summary of processing results. Defaults to "
            "[bold]config.create_csv_report[/bold]."
        ),
        rich_help_panel="Output",
    ),
    csv_output_path: Optional[Path] = typer.Option(
        None,
        "--csv-output-path",
        help=(
            "Path (file) where the CSV report is written. Defaults to "
            "[bold]config.csv_output_path[/bold]."
        ),
        rich_help_panel="Output",
    ),

    # --- Search & Normalization ----------------------------------------------
    search_text: Optional[str] = typer.Option(
        None,
        "--search-text",
        "-q",
        help=(
            "Text to search for. When omitted, uses [bold]config.search_text[/bold]."
        ),
        rich_help_panel="Search & Normalization",
    ),
    search_mode: Optional[str] = typer.Option(
        None,
        "--search-mode",
        help=(
            "Search mode: [bold]exact_phrase[/bold] (substring match) or "
            "[bold]all_words[/bold] (set inclusion of words). Defaults to "
            "[bold]config.search_mode[/bold]."
        ),
        rich_help_panel="Search & Normalization",
        metavar="[exact_phrase|all_words]",
    ),
    allow_normalization: Optional[bool] = typer.Option(
        None,
        "--allow-normalization/--no-allow-normalization",
        help=(
            "If enabled, normalizes OCR text before matching. Defaults to "
            "[bold]config.allow_recognition_normalization[/bold]."
        ),
        rich_help_panel="Search & Normalization",
    ),
    remove_spaces_in_normalization: Optional[bool] = typer.Option(
        None,
        "--remove-spaces/--keep-spaces",
        help=(
            "In exact_phrase mode, remove spaces from both query and OCR text "
            "during normalization for robust matching. Defaults to "
            "[bold]config.remove_spaces_in_normalization[/bold]."
        ),
        rich_help_panel="Search & Normalization",
    ),
    search_location: Optional[str] = typer.Option(
        None,
        "--search-location",
        help=(
            "Zonal OCR configuration as JSON, e.g. '{\"top\":0.1,\"bottom\":0.1}'. "
            "Use '{}' for full-page OCR. Defaults to [bold]config.search_location[/bold]."
        ),
        rich_help_panel="Search & Normalization",
    ),
    recognized_text_debug: Optional[bool] = typer.Option(
        None,
        "--recognized-text-debug/--no-recognized-text-debug",
        help=(
            "Log OCR raw/normalized snippets per page/zone for debugging. "
            "Defaults to [bold]config.recognized_text_debug[/bold]."
        ),
        rich_help_panel="Search & Normalization",
    ),

    # --- OCR Engine Selection & Options --------------------------------------
    ocr_engine: Optional[str] = typer.Option(
        None,
        "--ocr-engine",
        help=(
            "OCR backend to use. Options: [bold]paddleocr[/bold], [bold]easyocr[/bold], "
            "[bold]tesseract[/bold]. Defaults to [bold]config.ocr_engine[/bold]."
        ),
        rich_help_panel="OCR Engine",
        metavar="[paddleocr|easyocr|tesseract]",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help=(
            "Primary language code for OCR (e.g., 'ko', 'en'). Defaults to "
            "[bold]config.language[/bold]."
        ),
        rich_help_panel="OCR Engine",
    ),
    use_angle_cls: Optional[bool] = typer.Option(
        None,
        "--use-angle-cls/--no-use-angle-cls",
        help=(
            "(PaddleOCR) Enable text angle classification. Defaults to "
            "[bold]config.use_angle_cls[/bold]."
        ),
        rich_help_panel="OCR Engine",
    ),
    use_gpu_for_paddle: Optional[bool] = typer.Option(
        None,
        "--use-gpu/--no-use-gpu",
        help=(
            "(PaddleOCR) Use GPU acceleration if available. Defaults to "
            "[bold]config.use_gpu_for_paddle[/bold] or [bold]config.use_gpu[/bold]."
        ),
        rich_help_panel="OCR Engine",
    ),
    paddle_batch_size: Optional[int] = typer.Option(
        None,
        "--paddle-batch-size",
        help=(
            "(PaddleOCR) Recognition batch size. Defaults to "
            "[bold]config.paddle_batch_size[/bold] or [bold]config.rec_batch_num[/bold]."
        ),
        rich_help_panel="OCR Engine",
    ),
    use_offline_models: Optional[bool] = typer.Option(
        None,
        "--use-offline-models/--online-models",
        help=(
            "Force using only local OCR models (no downloads). Defaults to "
            "[bold]config.use_offline_models[/bold]."
        ),
        rich_help_panel="OCR Engine",
    ),
    tesseract_cmd_path: Optional[Path] = typer.Option(
        None,
        "--tesseract-cmd-path",
        help=(
            "(Tesseract) Path to tesseract executable. Defaults to "
            "[bold]config.tesseract_cmd_path[/bold]."
        ),
        rich_help_panel="OCR Engine",
    ),

    # --- Model Path Overrides -------------------------------------------------
    download_path_det_model: Optional[str] = typer.Option(
        None,
        "--download-path-det-model",
        help="Custom URL for PaddleOCR Detection model .tar.",
        rich_help_panel="Model Paths",
    ),
    download_path_rec_model: Optional[str] = typer.Option(
        None,
        "--download-path-rec-model",
        help="Custom URL for PaddleOCR Recognition model .tar.",
        rich_help_panel="Model Paths",
    ),
    download_path_cls_model: Optional[str] = typer.Option(
        None,
        "--download-path-cls-model",
        help="Custom URL for PaddleOCR Classification model .tar.",
        rich_help_panel="Model Paths",
    ),
    offline_paddle_det_model_dir: Optional[Path] = typer.Option(
        None,
        "--offline-paddle-det-model-dir",
        help="Offline dir containing PaddleOCR Detection model (inference.pdmodel).",
        rich_help_panel="Model Paths",
    ),
    offline_paddle_rec_model_dir: Optional[Path] = typer.Option(
        None,
        "--offline-paddle-rec-model-dir",
        help="Offline dir containing PaddleOCR Recognition model (inference.pdmodel).",
        rich_help_panel="Model Paths",
    ),
    offline_paddle_cls_model_dir: Optional[Path] = typer.Option(
        None,
        "--offline-paddle-cls-model-dir",
        help="Offline dir containing PaddleOCR Classification model (inference.pdmodel).",
        rich_help_panel="Model Paths",
    ),
    offline_easyocr_model_dir: Optional[Path] = typer.Option(
        None,
        "--offline-easyocr-model-dir",
        help="Offline dir for EasyOCR models.",
        rich_help_panel="Model Paths",
    ),

    # --- Diagnostics ----------------------------------------------------------
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging for troubleshooting.",
        rich_help_panel="Diagnostics",
    ),
) -> None:
    """Searches a file or directory of images for pages that contain the target text.

    Behavior
    - If --tif-path is a directory, all supported files are processed recursively.
    - If --tif-path is a file, only that file is processed.
    - If --tif-path is omitted, the value from config.input_path is used.

    Output
    - Matched pages are exported as images to --output-folder (or config default).
    - Optional CSV report can be generated for directory scans or single file runs.
    """
    # --- Configure logging early --------------------------------------------
    configure_logging(verbose)

    # --- Resolve effective input path ---------------------------------------
    eff_input: Optional[Path]
    if tif_path is not None:
        eff_input = tif_path
    else:
        cfg_input = default_config.get("input_path")
        eff_input = Path(cfg_input) if cfg_input else None

    if not eff_input:
        typer.secho("FATAL: No input path provided and config.input_path is not set.", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=2)

    eff_input = eff_input.resolve()
    if not eff_input.exists():
        typer.secho(f"FATAL: Input path does not exist: {eff_input}", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=2)

    # --- Resolve output folder ----------------------------------------------
    default_out = default_config.get("output_folder", "search_results")
    eff_output_folder = output_folder if output_folder else Path(default_out)
    if not eff_output_folder.is_absolute():
        eff_output_folder = (Path.cwd() / eff_output_folder).resolve()

    # --- Resolve supported extensions ----------------------------------------
    cfg_exts = default_config.get("supported_formats", ['.tif', '.tiff'])
    eff_exts = _normalize_exts(supported_formats if supported_formats else cfg_exts)

    # --- Parse search_location JSON if provided ------------------------------
    cli_search_location: Optional[Dict[str, float]] = None
    if search_location is not None:
        try:
            parsed = json.loads(search_location)
            if isinstance(parsed, dict):
                cli_search_location = parsed
            else:
                typer.secho(
                    "--search-location should be a JSON object like '{\"top\":0.1}'. Ignoring value.",
                    fg=typer.colors.YELLOW,
                )
        except Exception as _e:
            typer.secho(f"Could not parse --search-location as JSON: {_e}. Ignoring.", fg=typer.colors.YELLOW)
            cli_search_location = None

    try:
        # --- Initialize searcher (CLI values override config defaults) -------
        searcher = TifTextSearcher(
            search_text=search_text,
            language=language,
            ocr_backend=ocr_engine,
            search_mode=search_mode,
            allow_normalization=allow_normalization,
            remove_spaces_in_normalization=remove_spaces_in_normalization,
            use_angle_cls=use_angle_cls,
            use_gpu_for_paddle=use_gpu_for_paddle,
            paddle_batch_size=paddle_batch_size,
            use_offline_models=use_offline_models,
            recognized_text_debug=recognized_text_debug,
            download_path_det_model=download_path_det_model,
            download_path_rec_model=download_path_rec_model,
            download_path_cls_model=download_path_cls_model,
            offline_paddle_det_model_dir=str(offline_paddle_det_model_dir) if offline_paddle_det_model_dir else None,
            offline_paddle_rec_model_dir=str(offline_paddle_rec_model_dir) if offline_paddle_rec_model_dir else None,
            offline_paddle_cls_model_dir=str(offline_paddle_cls_model_dir) if offline_paddle_cls_model_dir else None,
            offline_easyocr_model_dir=str(offline_easyocr_model_dir) if offline_easyocr_model_dir else None,
            search_location=cli_search_location,
            tesseract_cmd_path=str(tesseract_cmd_path) if tesseract_cmd_path else None,
        )

        # === Directory Mode ===================================================
        if eff_input.is_dir():
            eff_search_text = search_text or default_config.get("search_text", "")
            typer.secho(
                f"Scanning directory: {eff_input} for text: '{eff_search_text}'",
                fg=typer.colors.CYAN,
            )

            tif_files = _iter_image_files_in_dir(eff_input, eff_exts)
            if not tif_files:
                typer.secho("Result: Directory contains no matching files.", fg=typer.colors.YELLOW)
                raise typer.Exit(code=0)

            # Prepare CSV if requested
            want_csv = create_csv_report if create_csv_report is not None else default_config.get("create_csv_report", False)
            csv_path = (
                Path(csv_output_path)
                if csv_output_path
                else Path(default_config.get("csv_output_path", "search_results/search_report.csv"))
            )
            csv_fp = None
            writer: Optional[csv.DictWriter] = None
            if want_csv:
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                csv_fp = open(csv_path, "w", newline="", encoding="utf-8-sig")
                writer = csv.DictWriter(
                    csv_fp,
                    fieldnames=[
                        "filename",
                        "matched_pages",
                        "saved_count",
                        "error",
                        "recognized_text_debug_enabled",
                        "searched_text",
                        "ocr_time_ms_list",
                        "raw_text_snippets",
                        "normalized_text_snippets",
                    ],
                )
                writer.writeheader()
                typer.secho(f"CSV report will be written to: {csv_path.resolve()}", fg=typer.colors.CYAN)

            total = len(tif_files)
            any_match = False
            total_saved = 0

            with typer.progressbar(tif_files, label="Processing files", length=total) as files:
                for f in files:
                    try:
                        matched_pages = searcher.find_text_pages(f)
                        # Debug aggregation for CSV if requested
                        dbg_enabled = bool(
                            recognized_text_debug
                            if recognized_text_debug is not None
                            else default_config.get("recognized_text_debug", False)
                        )
                        raw_text_snips: List[str] = []
                        norm_text_snips: List[str] = []
                        ocr_times_ms: List[float] = []
                        if dbg_enabled:
                            for page_dbg in getattr(searcher, "_last_debug_info", []) or []:
                                for z in page_dbg.get("zones", []):
                                    raw_text_snips.append(z.get("raw_text", ""))
                                    norm_text_snips.append(z.get("normalized_text", ""))
                                    ocr_times_ms.append(z.get("ocr_time_ms", 0.0))

                        if matched_pages:
                            any_match = True
                            saved_paths = _save_pages(f, matched_pages, eff_output_folder, img_format=img_format)
                            total_saved += len(saved_paths)
                            typer.secho(
                                f"{f.name}: Found on pages {matched_pages}. Saved: {len(saved_paths)} page(s).",
                                fg=typer.colors.GREEN,
                            )
                            if writer:
                                row = {
                                    "filename": f.name,
                                    "matched_pages": matched_pages,
                                    "saved_count": len(saved_paths),
                                    "error": "",
                                    "recognized_text_debug_enabled": dbg_enabled,
                                    "searched_text": eff_search_text,
                                    "ocr_time_ms_list": ocr_times_ms,
                                    "raw_text_snippets": raw_text_snips,
                                    "normalized_text_snippets": norm_text_snips,
                                }
                                writer.writerow(row)
                        else:
                            typer.secho(f"{f.name}: No matching pages found.", fg=typer.colors.YELLOW)
                            if writer:
                                row = {
                                    "filename": f.name,
                                    "matched_pages": [],
                                    "saved_count": 0,
                                    "error": "",
                                    "recognized_text_debug_enabled": dbg_enabled,
                                    "searched_text": eff_search_text,
                                    "ocr_time_ms_list": ocr_times_ms,
                                    "raw_text_snippets": raw_text_snips,
                                    "normalized_text_snippets": norm_text_snips,
                                }
                                writer.writerow(row)
                    except Exception as e:
                        logging.warning("Skipping %s due to error: %s", f.name, e)
                        if writer:
                            row = {
                                "filename": f.name,
                                "matched_pages": [],
                                "saved_count": 0,
                                "error": str(e),
                                "recognized_text_debug_enabled": bool(
                                    recognized_text_debug
                                    if recognized_text_debug is not None
                                    else default_config.get("recognized_text_debug", False)
                                ),
                                "searched_text": eff_search_text,
                                "ocr_time_ms_list": [],
                                "raw_text_snippets": [],
                                "normalized_text_snippets": [],
                            }
                            writer.writerow(row)
                        continue

            if csv_fp:
                csv_fp.close()
                logging.info("CSV report generated at: %s", csv_path.resolve())

            if not any_match:
                typer.secho("Result: No matching pages found in any files.", fg=typer.colors.YELLOW)
            else:
                typer.secho(
                    f"Done. Total pages saved: {total_saved}. Output: {eff_output_folder}",
                    fg=typer.colors.GREEN,
                    bold=True,
                )
            return

        # === Single File Mode =================================================
        eff_search_text = search_text or default_config.get("search_text", "")
        typer.secho(
            f"Scanning {eff_input} for text: '{eff_search_text}'",
            fg=typer.colors.CYAN,
        )
        matched_pages = searcher.find_text_pages(eff_input)
        if matched_pages:
            saved_paths = _save_pages(eff_input, matched_pages, eff_output_folder, img_format=img_format)
            typer.secho(
                f"Success: Found text on pages: {matched_pages}. Saved: {len(saved_paths)} page(s) to: {eff_output_folder}",
                fg=typer.colors.GREEN,
                bold=True,
            )
            # Optional single-file CSV report
            want_csv = create_csv_report if create_csv_report is not None else default_config.get("create_csv_report", False)
            if want_csv:
                csv_path = (
                    Path(csv_output_path)
                    if csv_output_path
                    else Path(default_config.get("csv_output_path", "search_results/search_report.csv"))
                )
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                with open(csv_path, "w", newline="", encoding="utf-8-sig") as fp:
                    writer = csv.DictWriter(
                        fp,
                        fieldnames=[
                            "filename",
                            "matched_pages",
                            "saved_count",
                            "error",
                            "recognized_text_debug_enabled",
                            "searched_text",
                            "ocr_time_ms_list",
                            "raw_text_snippets",
                            "normalized_text_snippets",
                        ],
                    )
                    writer.writeheader()
                    # Debug aggregation if available
                    dbg_enabled = bool(
                        recognized_text_debug
                        if recognized_text_debug is not None
                        else default_config.get("recognized_text_debug", False)
                    )
                    raw_text_snips: List[str] = []
                    norm_text_snips: List[str] = []
                    ocr_times_ms: List[float] = []
                    if dbg_enabled:
                        for page_dbg in getattr(searcher, "_last_debug_info", []) or []:
                            for z in page_dbg.get("zones", []):
                                raw_text_snips.append(z.get("raw_text", ""))
                                norm_text_snips.append(z.get("normalized_text", ""))
                                ocr_times_ms.append(z.get("ocr_time_ms", 0.0))
                    writer.writerow(
                        {
                            "filename": eff_input.name,
                            "matched_pages": matched_pages,
                            "saved_count": len(saved_paths),
                            "error": "",
                            "recognized_text_debug_enabled": dbg_enabled,
                            "searched_text": eff_search_text,
                            "ocr_time_ms_list": ocr_times_ms,
                            "raw_text_snippets": raw_text_snips,
                            "normalized_text_snippets": norm_text_snips,
                        }
                    )
                typer.secho(f"CSV report written to: {csv_path.resolve()}", fg=typer.colors.CYAN)
        else:
            typer.secho("Result: No matching pages found.", fg=typer.colors.YELLOW)

    except Exception as e:
        logging.error("TIF search failed: %s", e, exc_info=True)
        typer.secho(f"TIF search failed: {e}", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
