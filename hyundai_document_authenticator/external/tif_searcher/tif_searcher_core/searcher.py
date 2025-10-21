"""📝
Production-ready wrapper that provides TifTextSearcher for identifying
pages in multi-page TIF documents that contain a configurable search title.

This package is designed to be reusable across projects. It defaults to
PaddleOCR for robust Korean OCR, with an automatic fallback to Tesseract
if PaddleOCR is unavailable. If a project-level config exists at
`searcher_core.config.config`, its values are used unless explicitly overridden.

Public API:
    - class TifTextSearcher
        - find_text_pages(tif_path: Union[str, Path]) -> List[int]

Usage:
    from tif_searcher import TifTextSearcher
    searcher = TifTextSearcher()  # Will read defaults from config
    pages = searcher.find_text_pages('path/to/document.tif')

Notes:
    - Page indices returned are 1-based (first page is 1) for usability.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Union, Tuple, Dict
import logging

from PIL import Image, ImageSequence
import numpy as np

logger = logging.getLogger(__name__)

# Try to import project-specific config (prefer vendored, then installed)
try:
    from .config import config as _core_cfg  # vendored config
except Exception:
    try:
        from tif_searcher.config import config as _core_cfg  # installed package fallback
    except Exception:
        _core_cfg = {}

# FIXME: Unused helper; consider removal or integration with utils.normalize_text for consistency.
def _normalize_text(s: str) -> str:
    """Normalize text for robust substring matching."""
    return "".join(s.split())

from .ocr_engines import get_ocr_engine
from .utils import normalize_text

class TifTextSearcher:
    """Detect pages in multi-page TIFs containing specified text.

    This class provides flexible OCR-based search with configurable engines,
    normalization, and optional zonal OCR to constrain recognition to portions
    of each page for improved performance and precision.

    Args:
        search_text (Optional[str]): The target phrase to search for. If None,
            value is taken from config.
        language (Optional[str]): OCR language code (e.g., 'ko', 'en'). If None,
            value is taken from config.
        ocr_backend (Optional[str]): Backend identifier: 'paddleocr', 'easyocr', or 'tesseract'.
        custom_engine (Optional[Any]): Custom OCR engine with an extract_text(np.ndarray) -> str API.
        search_mode (Optional[str]): 'exact_phrase' or 'all_words'. Defaults from config.
        allow_normalization (Optional[bool]): When True, normalizes query and OCR text before matching.
        remove_spaces_in_normalization (Optional[bool]): When True with exact_phrase,
            matches "HelloWorld" to "Hello World" by stripping spaces.
        use_angle_cls (Optional[bool]): PaddleOCR angle classification toggle.
        use_gpu_for_paddle (Optional[bool]): PaddleOCR GPU enablement.
        paddle_batch_size (Optional[int]): PaddleOCR recognition batch size.
        use_offline_models (Optional[bool]): Whether to force offline model usage.
        download_path_det_model (Optional[str]): Path for downloading detection model.
        download_path_rec_model (Optional[str]): Path for downloading recognition model.
        download_path_cls_model (Optional[str]): Path for downloading classification model.
        offline_paddle_det_model_dir (Optional[str]): Local dir for PaddleOCR det model.
        offline_paddle_rec_model_dir (Optional[str]): Local dir for PaddleOCR rec model.
        offline_paddle_cls_model_dir (Optional[str]): Local dir for PaddleOCR cls model.
        offline_easyocr_model_dir (Optional[str]): Local dir for EasyOCR models.
        search_location (Optional[dict]): Optional vertical zone percentages for OCR
            with keys in {"top", "bottom", "center"} and float values in [0, 1].
        recognized_text_debug (Optional[bool]): Log raw/normalized OCR snippets per page/zone.
        tesseract_cmd_path (Optional[str]): Optional explicit path to Tesseract executable.

    Attributes:
        _engine: OCR engine instance providing extract_text(np.ndarray) -> str.
        _search_text: Query string used for matching.
        _search_mode: Matching mode: 'exact_phrase' or 'all_words'.
        _allow_normalization: Toggle for normalization prior to matching.
        _remove_spaces: Controls whitespace removal under exact match mode.
        _search_location: Vertical OCR zones configuration.
        _recognized_text_debug: Whether to log OCR text snippets for debugging.
    """

    def __init__(
        self,
        search_text: Optional[str] = None,
        language: Optional[str] = None,
        ocr_backend: Optional[str] = None,
        custom_engine: Optional[Any] = None,
        # Search & Normalization
        search_mode: Optional[str] = None,
        allow_normalization: Optional[bool] = None,
        remove_spaces_in_normalization: Optional[bool] = None,
        # PaddleOCR Performance & Accuracy
        use_angle_cls: Optional[bool] = None,
        use_gpu_for_paddle: Optional[bool] = None,
        paddle_batch_size: Optional[int] = None,
        # Model Paths
        use_offline_models: Optional[bool] = None,
        download_path_det_model: Optional[str] = None,
        download_path_rec_model: Optional[str] = None,
        download_path_cls_model: Optional[str] = None,
        offline_paddle_det_model_dir: Optional[str] = None,
        offline_paddle_rec_model_dir: Optional[str] = None,
        offline_paddle_cls_model_dir: Optional[str] = None,
        offline_easyocr_model_dir: Optional[str] = None,
        search_location: Optional[dict] = None,
        recognized_text_debug: Optional[bool] = None,
        tesseract_cmd_path: Optional[str] = None,
    ) -> None:
        cfg = _core_cfg

        # --- Set Search and Normalization Parameters ---
        self._search_text = search_text if search_text is not None else cfg.get("search_text", "")
        self._search_mode = search_mode if search_mode is not None else cfg.get("search_mode", "exact_phrase")
        self._allow_normalization = allow_normalization if allow_normalization is not None else cfg.get("allow_recognition_normalization", True)
        self._remove_spaces = remove_spaces_in_normalization if remove_spaces_in_normalization is not None else cfg.get("remove_spaces_in_normalization", True)
        # --- Zonal OCR configuration ---
        # Read the vertical search zones specification from config unless explicitly overridden.
        # If absent or empty, full-page OCR is performed.
        self._search_location = search_location if search_location is not None else (cfg.get("search_location", {}) or {})
        # Debug flag: print/log raw OCR text per page before matching when enabled
        self._recognized_text_debug = recognized_text_debug if recognized_text_debug is not None else cfg.get("recognized_text_debug", False)

        # --- Initialize OCR Engine ---
        if custom_engine:
            if not hasattr(custom_engine, "extract_text"):
                raise TypeError("custom_engine must provide an extract_text method.")
            self._engine = custom_engine
            logger.info("🔎 TifTextSearcher using custom OCR engine.")
        else:
            factory_cfg = {
                "ocr_engine": ocr_backend or cfg.get("ocr_engine", "paddleocr"),
                "language": language or cfg.get("language", "ko"),
                "language_code_map": cfg.get("language_code_map", {}),
                "use_angle_cls": use_angle_cls if use_angle_cls is not None else cfg.get("use_angle_cls"),
                "use_gpu_for_paddle": use_gpu_for_paddle if use_gpu_for_paddle is not None else (cfg.get("use_gpu_for_paddle") if "use_gpu_for_paddle" in cfg else cfg.get("use_gpu")),
                "paddle_batch_size": paddle_batch_size if paddle_batch_size is not None else (cfg.get("paddle_batch_size") if "paddle_batch_size" in cfg else cfg.get("rec_batch_num")),
                "use_offline_models": use_offline_models if use_offline_models is not None else cfg.get("use_offline_models"),
                "download_path_det_model": download_path_det_model or cfg.get("download_path_det_model"),
                "download_path_rec_model": download_path_rec_model or cfg.get("download_path_rec_model"),
                "download_path_cls_model": download_path_cls_model or cfg.get("download_path_cls_model"),
                "offline_paddle_det_model_dir": offline_paddle_det_model_dir or cfg.get("offline_paddle_det_model_dir") or cfg.get("offline_det_model_dir"),
                "offline_paddle_rec_model_dir": offline_paddle_rec_model_dir or cfg.get("offline_paddle_rec_model_dir") or cfg.get("offline_rec_model_dir"),
                "offline_paddle_cls_model_dir": offline_paddle_cls_model_dir or cfg.get("offline_paddle_cls_model_dir") or cfg.get("offline_cls_model_dir"),
                "offline_easyocr_model_dir": offline_easyocr_model_dir or cfg.get("offline_easyocr_model_dir") or cfg.get("offline_model_dir"),
                "tesseract_cmd_path": tesseract_cmd_path or cfg.get("tesseract_cmd_path"),
            }
            self._engine = get_ocr_engine(factory_cfg)
            if not self._engine:
                raise RuntimeError("Failed to initialize OCR engine.")

    def find_text_pages(self, tif_path: Union[str, Path]) -> List[int]:
        """Find 1-based page numbers that contain the configured target text.

        Applies optional zonal OCR if search_location is set, performing OCR on
        requested vertical slices and combining results. Matching honors the
        configured search mode and normalization settings.

        Args:
            tif_path (Union[str, Path]): Path to a multi-page TIF document.

        Returns:
            List[int]: 1-based page numbers where the query is considered present.

        Raises:
            FileNotFoundError: If tif_path does not exist or is not a file.
            RuntimeError: If the OCR engine is not initialized (should not occur after __init__).

        Note:
            Per-page OCR exceptions are caught and logged; processing continues with
            subsequent pages for robustness. See TODO in exception handler for narrowing
            exception types.
        """
        tif_file = Path(tif_path)
        if not tif_file.is_file():
            raise FileNotFoundError(f"TIF not found: {tif_file}")

        matched_pages: List[int] = []
        # Reset debug info for this run
        self._last_debug_info: List[dict] = []

        def _clamp_pct(val: float) -> float:
            """Clamp a percentage value to [0.0, 1.0]."""
            try:
                v = float(val)
            except Exception:
                return 0.0
            if v < 0.0:
                logger.warning("search_location value %.4f < 0.0. Clamping to 0.0.", v)
                v = 0.0
            if v > 1.0:
                logger.warning("search_location value %.4f > 1.0. Clamping to 1.0.", v)
                v = 1.0
            return v

        def _build_zone_boxes(h: int, w: int) -> List[Tuple[int, int, int, int, str]]:
            """Return a list of crop boxes with their kind.

            Args:
                h (int): Image height.
                w (int): Image width.

            Returns:
                List[Tuple[int, int, int, int, str]]: Crop boxes (left, upper, right, lower, kind).
            """
            loc = self._search_location if isinstance(self._search_location, dict) else {}
            if not loc:
                return []  # empty => full image
            boxes: List[Tuple[int, int, int, int, str]] = []
            for key, val in loc.items():
                if key not in ("top", "bottom", "center"):
                    logger.debug("Ignoring unknown search_location key: %s", key)
                    continue
                p = _clamp_pct(val)
                if p <= 0.0:
                    continue
                zone_h = max(1, int(round(h * p)))
                if key == "top":
                    boxes.append((0, 0, w, min(h, zone_h), "top"))
                elif key == "bottom":
                    top = max(0, h - zone_h)
                    boxes.append((0, top, w, h, "bottom"))
                elif key == "center":
                    start = max(0, int(round((h - zone_h) / 2)))
                    end = min(h, start + zone_h)
                    boxes.append((0, start, w, end, "center"))
            # Remove duplicates and invalid boxes
            unique_boxes: List[Tuple[int, int, int, int, str]] = []
            for b in boxes:
                l, u, r, d, k = b
                if r - l <= 0 or d - u <= 0:
                    continue
                if b not in unique_boxes:
                    unique_boxes.append(b)
            return unique_boxes

        with Image.open(tif_file) as img:
            for idx, page in enumerate(ImageSequence.Iterator(img), start=1):
                try:
                    page_rgb = page.convert("RGB")
                    h, w = page_rgb.height, page_rgb.width
                    zone_boxes = _build_zone_boxes(h, w)

                    if not zone_boxes:  # Empty dict => full page
                        import time
                        np_image = np.array(page_rgb)
                        t0 = time.time()
                        text = self._engine.extract_text(np_image)
                        dt_ms = (time.time() - t0) * 1000.0
                        # Compute normalized haystack the same way matching does
                        remove_spaces = self._remove_spaces and self._search_mode == 'exact_phrase'
                        norm_text = normalize_text(text, remove_spaces=remove_spaces) if self._allow_normalization and text else (text or "")
                        if self._recognized_text_debug and text is not None:
                            raw_snip = text if len(text) <= 600 else (text[:600] + "...")
                            norm_snip = norm_text if len(norm_text) <= 600 else (norm_text[:600] + "...")
                            logger.info("[OCR DEBUG] %s page %d (%.1f ms) RAW: %s", tif_file.name, idx, dt_ms, raw_snip)
                            logger.info("[OCR DEBUG] %s page %d (%.1f ms) NORM: %s", tif_file.name, idx, dt_ms, norm_snip)
                        # Record debug info
                        self._last_debug_info.append({
                            "page": idx,
                            "zones": [{
                                "zone_index": 1,
                                "zone_type": "full",
                                "ocr_time_ms": dt_ms,
                                "raw_text": text or "",
                                "normalized_text": norm_text,
                            }]
                        })
                        if text and self._is_match(text):
                            matched_pages.append(idx)
                        continue

                    # Zonal OCR: process each zone independently and combine results
                    combined_texts: List[str] = []
                    page_debug = {"page": idx, "zones": []}
                    for b_idx, (l, u, r, d, k) in enumerate(zone_boxes, start=1):
                        import time
                        zone_img = page_rgb.crop((l, u, r, d))
                        np_zone = np.array(zone_img)
                        t0 = time.time()
                        text = self._engine.extract_text(np_zone)
                        dt_ms = (time.time() - t0) * 1000.0
                        if text:
                            remove_spaces = self._remove_spaces and self._search_mode == 'exact_phrase'
                            norm_text = normalize_text(text, remove_spaces=remove_spaces) if self._allow_normalization else text
                            if self._recognized_text_debug:
                                raw_snip = text if len(text) <= 600 else (text[:600] + "...")
                                norm_snip = norm_text if len(norm_text) <= 600 else (norm_text[:600] + "...")
                                logger.info("[OCR DEBUG] %s page %d zone %d (%s, %.1f ms) RAW: %s", tif_file.name, idx, b_idx, k, dt_ms, raw_snip)
                                logger.info("[OCR DEBUG] %s page %d zone %d (%s, %.1f ms) NORM: %s", tif_file.name, idx, b_idx, k, dt_ms, norm_snip)
                            combined_texts.append(text)
                            page_debug["zones"].append({
                                "zone_index": b_idx,
                                "zone_type": k,
                                "ocr_time_ms": dt_ms,
                                "raw_text": text,
                                "normalized_text": norm_text,
                            })
                            # Early match: if any zone matches, accept the page
                            if self._is_match(text):
                                matched_pages.append(idx)
                                self._last_debug_info.append(page_debug)
                                break
                    else:
                        # If loop didn't break, optionally check combined text
                        if combined_texts:
                            if self._is_match("\n".join(combined_texts)):
                                matched_pages.append(idx)
                        # Save page debug even if no early break
                        self._last_debug_info.append(page_debug)
                except Exception as e:
                    # TODO: Narrow exception types; engine/PIL may raise OSError, ValueError, etc.
                    logger.warning("OCR failed on %s page %d: %s", tif_file.name, idx, e)
                    continue
        return matched_pages

    def _is_match(self, ocr_text: str) -> bool:
        """Internal logic to determine if the OCR text matches the search query."""
        # --- Prepare Query Text ---
        remove_spaces = self._remove_spaces and self._search_mode == 'exact_phrase'

        if self._allow_normalization:
            query_text = normalize_text(self._search_text, remove_spaces=remove_spaces)
        else:
            query_text = self._search_text

        # --- Prepare OCR Haystack ---
        if self._allow_normalization:
            haystack = normalize_text(ocr_text, remove_spaces=remove_spaces)
        else:
            haystack = ocr_text

        # --- Perform Match ---
        if self._search_mode == 'exact_phrase':
            return query_text in haystack
        elif self._search_mode == 'all_words':
            query_words = set(query_text.split())
            haystack_words = set(haystack.split())
            return query_words.issubset(haystack_words)

        return False

    def find_and_extract_images_from_batch(
        self,
        tif_files: Sequence[Union[str, Path]],
        output_dir: Union[str, Path],
        img_format: str = "png",
    ) -> List[Path]:
        """Process a batch of TIF files, extract matched pages, and save as images.

        Args:
            tif_files (Sequence[Union[str, Path]]): Paths to TIF files to process.
            output_dir (Union[str, Path]): Directory to save the extracted page images.
            img_format (str): Output image format (e.g., 'png', 'jpg').

        Returns:
            List[Path]: Paths to saved images.

        Raises:
            OSError: If a file cannot be opened or saved by Pillow.

        Note:
            Non-existent TIFs are skipped with a warning; processing continues.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        extracted_image_paths: List[Path] = []

        for tif_path in tif_files:
            tif_file = Path(tif_path)
            if not tif_file.is_file():
                logger.warning("Skipping non-existent file: %s", tif_file)
                continue

            page_numbers = self.find_text_pages(tif_file)
            if not page_numbers:
                logger.info("No matching pages found in %s", tif_file.name)
                continue

            with Image.open(tif_file) as img:
                for page_num in page_numbers:
                    img.seek(page_num - 1)

                    base_name = tif_file.stem
                    # Standardized naming to match pipeline: '<stem>_page{N}.<ext>'
                    output_filename = f"{base_name}_page{page_num}.{img_format}"
                    save_path = output_path / output_filename

                    img.save(save_path, format=img_format.upper())
                    extracted_image_paths.append(save_path)
                    logger.info("Saved page %d from %s to %s", page_num, tif_file.name, save_path)

        return extracted_image_paths


__all__ = ["TifTextSearcher"]
