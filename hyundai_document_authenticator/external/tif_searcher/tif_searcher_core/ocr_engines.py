"""OCR engine abstractions and implementations for TIF text search workflow.

This module defines a standard OCR engine interface and concrete implementations
based on PaddleOCR, EasyOCR, and Tesseract. It provides a factory to construct
an engine instance from configuration, supporting both offline and online model
workflows and optional angle classification.

Notes:
- Avoid configuring logging handlers in this module; applications/CLIs should
  configure handlers. This module uses a module-level logger only.
- Some implementations may download models when not configured for offline use.
  This behavior is preserved for backward compatibility.
"""

from __future__ import annotations

import logging
import os
import tarfile
from os import PathLike
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import requests
from tqdm import tqdm
from PIL import Image
import numpy as np

# TODO: Consider lazy-importing paddleocr to reduce import-time overhead.
import paddleocr

logger = logging.getLogger(__name__)


# ==============================================================================
# === 1. ABSTRACT BASE CLASS ===================================================
# ==============================================================================

class OCREngine(ABC):
    """Abstract base class defining a standard OCR engine interface.

    Subclasses must implement initialization and text extraction from images.

    Args:
        config (Dict[str, Any]): Configuration dictionary for the OCR engine.

    Raises:
        SystemExit: If engine initialization fails and no engine instance is available.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.engine: Any = self._initialize_engine()
        if not self.engine:
            # Preserve existing behavior to stop workflow if engine is not usable.
            raise SystemExit("Halting workflow due to OCR engine initialization failure.")

    @abstractmethod
    def _initialize_engine(self) -> Any:
        """Engine-specific initialization and model loading.

        Returns:
            Any: The underlying engine object if successful; otherwise a falsy value.
        """
        raise NotImplementedError

    @abstractmethod
    def extract_text(self, image_data: Union[np.ndarray, Image.Image, str, PathLike]) -> str:
        """Perform OCR on an image and return the extracted text.

        Args:
            image_data (Union[numpy.ndarray, PIL.Image.Image, str, PathLike]): Input image.
                Strings and PathLike are treated as file paths.

        Returns:
            str: Concatenated text extracted by the OCR engine. Empty string on no text.
        """
        raise NotImplementedError


# ==============================================================================
# === 2. PADDLEOCR IMPLEMENTATION =============================================
# ==============================================================================

class PaddleOCREngine(OCREngine):
    """PaddleOCR-based implementation with robust model resolution and logging.

    Supports offline model directories, custom download URLs, and default model
    usage (auto-download) when not in offline mode.
    """

    def _download_and_extract_model(self, url: str, destination_folder: Union[str, Path], model_name: str) -> bool:
        """Download and extract a model archive with progress.

        Args:
            url (str): Direct URL to a .tar model archive.
            destination_folder (Union[str, Path]): Directory to store the extracted model.
            model_name (str): Name for logging (e.g., "Detection").

        Returns:
            bool: True on success; False on failure or when url is empty.
        """
        if not url:
            return True  # Skipping optional model is not a failure.

        dest = Path(destination_folder)
        dest.mkdir(parents=True, exist_ok=True)
        tar_filename = dest / os.path.basename(url)

        logger.info("Downloading %s model from: %s", model_name, url)
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                with open(tar_filename, "wb") as f, tqdm(total=total_size, unit="iB", unit_scale=True, desc=model_name) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

            logger.info("Extracting %s model...", model_name)
            with tarfile.open(tar_filename) as tf:
                tf.extractall(path=str(dest))

            try:
                os.remove(tar_filename)
            except OSError:
                # Non-fatal cleanup failure; continue.
                pass

            logger.info("%s model is ready.", model_name)
            return True
        except Exception as e:
            logger.error("Failed to download or extract %s model. Reason: %s", model_name, e)
            return False

    def _resolve_model_path(self, base_path: Union[str, Path], model_name: str) -> Optional[str]:
        """Resolve directory containing the model's *.pdmodel file.

        Checks in order:
        - Direct match when *.pdmodel resides in base_path
        - Recursive search under base_path
        - Single-subdirectory pattern after extraction

        Args:
            base_path (Union[str, Path]): Base directory to search.
            model_name (str): Model label for logs.

        Returns:
            Optional[str]: Resolved directory path as a string, or None if not found.
        """
        if not base_path:
            return None
        base_path_str = str(base_path)
        if not os.path.isdir(base_path_str):
            return None

        # Check 1: *.pdmodel present in base path
        try:
            if any(fname.endswith(".pdmodel") for fname in os.listdir(base_path_str)):
                logger.info("Resolved %s path: %s", model_name, base_path_str)
                return base_path_str
        except Exception:
            pass

        # Check 2: Recursive search
        try:
            for root, _dirs, files in os.walk(base_path_str):
                if any(fn.endswith(".pdmodel") for fn in files):
                    logger.info("Resolved %s path: %s", model_name, root)
                    return root
        except Exception:
            pass

        # Check 3: Single-subdirectory case
        try:
            subdirs = [d for d in os.listdir(base_path_str) if os.path.isdir(os.path.join(base_path_str, d))]
            if len(subdirs) == 1:
                potential_path = os.path.join(base_path_str, subdirs[0])
                if any(fname.endswith(".pdmodel") for fname in os.listdir(potential_path)):
                    logger.info("Resolved %s path: %s", model_name, potential_path)
                    return potential_path
        except Exception:
            pass

        return None

    def _introspect_cached_dir(self, engine_obj: Any, kind: str) -> Optional[str]:
        """Best-effort introspection of default PaddleOCR cached model directory.

        Args:
            engine_obj (Any): PaddleOCR instance.
            kind (str): One of {'det', 'rec', 'cls'}.

        Returns:
            Optional[str]: Directory path if found.
        """
        try:
            if kind == "det":
                candidates = [
                    getattr(engine_obj, "det_model_dir", None),
                    getattr(getattr(engine_obj, "text_detector", None), "det_model_dir", None),
                    getattr(getattr(engine_obj, "text_detector", None), "args", None)
                    and getattr(engine_obj.text_detector.args, "det_model_dir", None),
                ]
            elif kind == "rec":
                candidates = [
                    getattr(engine_obj, "rec_model_dir", None),
                    getattr(getattr(engine_obj, "text_recognizer", None), "rec_model_dir", None),
                    getattr(getattr(engine_obj, "text_recognizer", None), "args", None)
                    and getattr(engine_obj.text_recognizer.args, "rec_model_dir", None),
                ]
            elif kind == "cls":
                candidates = [
                    getattr(engine_obj, "cls_model_dir", None),
                    getattr(getattr(engine_obj, "text_classifier", None), "cls_model_dir", None),
                    getattr(getattr(engine_obj, "text_classifier", None), "args", None)
                    and getattr(engine_obj.text_classifier.args, "cls_model_dir", None),
                ]
            else:
                candidates = []

            for c in candidates:
                if isinstance(c, str) and os.path.isdir(c):
                    return os.path.abspath(c)
        except Exception:
            pass

        # Fallback: search default PaddleOCR cache
        try:
            home = os.path.expanduser("~")
            base = os.path.join(home, ".paddleocr", "whl")
            sub = {"det": "det", "rec": "rec", "cls": "cls"}.get(kind)
            if sub:
                root = os.path.join(base, sub)
                if os.path.isdir(root):
                    for r, _d, files in os.walk(root):
                        if any(fn.endswith(".pdmodel") for fn in files):
                            return os.path.abspath(r)
        except Exception:
            pass
        return None

    def _initialize_engine(self) -> Any:
        """Initialize PaddleOCR engine with flexible model resolution.

        Honors offline model directories; otherwise supports custom model downloads
        before falling back to PaddleOCR defaults (cached/auto-download).

        Returns:
            Any: PaddleOCR instance on success, falsy value on failure.
        """
        logger.info("OCR ENGINE INITIALIZATION (PaddleOCR)")

        # --- Stage 1: Configuration Gathering ---
        use_offline = self.config.get("use_offline_models", False)
        use_angle_cls = self.config.get("use_angle_cls", False)
        lang_code = self.config.get("language", "en")

        # Use the configurable language map to translate the code if needed.
        lang_map = self.config.get("language_code_map", {})
        if lang_code in lang_map:
            mapped_code = lang_map[lang_code]
            logger.info("Mapped language code '%s' to '%s' via config.", lang_code, mapped_code)
            lang_code = mapped_code
        else:
            # Fallback mapping for common short codes when config is missing
            fallback_map = {"ko": "korean", "jp": "japan"}
            if lang_code in fallback_map:
                mapped_code = fallback_map[lang_code]
                logger.info("Mapped language code '%s' to '%s' (fallback).", lang_code, mapped_code)
                lang_code = mapped_code

        # Model-specific configurations
        model_configs: Dict[str, Dict[str, Any]] = {
            "det": {
                "name": "Detection",
                "base_path": self.config.get("offline_paddle_det_model_dir"),
                "url": self.config.get("download_path_det_model"),
                "final_path": None,
                "required": True,
            },
            "rec": {
                "name": "Recognition",
                "base_path": self.config.get("offline_paddle_rec_model_dir"),
                "url": self.config.get("download_path_rec_model"),
                "final_path": None,
                "required": True,
            },
            "cls": {
                "name": "Classification",
                "base_path": self.config.get("offline_paddle_cls_model_dir"),
                "url": self.config.get("download_path_cls_model"),
                "final_path": None,
                "required": use_angle_cls,
            },
        }

        # --- Stage 2: Model Path Resolution and Downloading ---
        logger.info("Checking for PaddleOCR models...")
        fallback_usage: Dict[str, bool] = {"det": False, "rec": False, "cls": False}
        # Persist for later (post-first-inference introspection)
        self._fallback_usage = fallback_usage
        self._rec_cls_path_logged = False
        for key, mc in model_configs.items():
            if not mc["required"]:
                continue

            # Priority 1: Check if the model exists in the offline path
            mc["final_path"] = self._resolve_model_path(mc["base_path"], mc["name"])  # type: ignore[arg-type]

            # Priority 2: If not found and not in offline mode, try downloading from custom URL
            if not mc["final_path"] and not use_offline and mc["url"]:
                logger.info("Model '%s' not found locally. Attempting download from custom URL.", mc["name"]) 
                if self._download_and_extract_model(mc["url"], mc["base_path"] or ".", mc["name"]):
                    mc["final_path"] = self._resolve_model_path(mc["base_path"], mc["name"])  # type: ignore[arg-type]

            # If still unresolved and not offline, allow PaddleOCR defaults
            if not mc["final_path"] and not use_offline:
                logger.info("Model '%s' not resolved. Will use PaddleOCR defaults (auto-download/cached).", mc["name"]) 
                fallback_usage[key] = True

        # --- Stage 3: Assemble Initialization Parameters ---
        final_det_path = model_configs["det"]["final_path"]
        final_rec_path = model_configs["rec"]["final_path"]
        final_cls_path = model_configs["cls"]["final_path"]

        # Critical check for required models if in offline mode
        if use_offline and (not final_det_path or not final_rec_path):
            logger.error("In offline mode, but Detection or Recognition models not found at specified paths.")
            return None

        # Determine if angle classification can be truly enabled
        angle_cls_enabled = use_angle_cls and (bool(final_cls_path) or not use_offline)
        use_gpu = self.config.get("use_gpu_for_paddle", False)

        logger.info("Assembling PaddleOCR params: lang='%s', GPU=%s, angle_cls=%s", lang_code, use_gpu, angle_cls_enabled)
        paddle_params: Dict[str, Any] = {
            "use_angle_cls": angle_cls_enabled,
            "lang": lang_code,
            "det_model_dir": final_det_path,
            "rec_model_dir": final_rec_path,
            "cls_model_dir": final_cls_path,
            "rec_batch_num": self.config.get("paddle_batch_size", 6),
            "show_log": self.config.get("recognized_text_debug", False),
        }

        logger.info("Initializing PaddleOCR engine with parameters (non-None):")
        final_params = {k: v for k, v in paddle_params.items() if v is not None}
        for key, value in final_params.items():
            logger.info("  %s: %s", key, value)

        try:
            engine_instance = paddleocr.PaddleOCR(**final_params, use_gpu=use_gpu)
            # After initialization, if some models used defaults, try to log cached directories
            for _k, _label in [("det", "Detection"), ("rec", "Recognition"), ("cls", "Classification")]:
                if fallback_usage.get(_k):
                    cached_dir = self._introspect_cached_dir(engine_instance, _k)
                    if cached_dir:
                        logger.info("PaddleOCR resolved %s model directory to: %s", _label, cached_dir)
                    else:
                        logger.info("PaddleOCR is using default %s model (cached location not introspectable).", _label)
            return engine_instance
        except Exception as e:
            logger.error("Fatal error during PaddleOCR initialization: %s", e)
            if "character dictionary file" in str(e):
                logger.info("Hint: Ensure language code '%s' matches the recognition model.", lang_code)
            if "CUDA" in str(e) or "cuDNN" in str(e):
                logger.info("Hint: Check GPU availability and CUDA/cuDNN versions.")
            return None

    def extract_text(self, image_input: Union[np.ndarray, Image.Image, str, PathLike]) -> str:
        """Perform OCR by laundering the image to a clean PNG before recognition.

        Laundering avoids deep library errors common with complex TIFFs.

        Args:
            image_input (Union[numpy.ndarray, PIL.Image.Image, str, PathLike]): Input image
                or path to an image file. Arrays and PIL images are converted to RGB PNGs.

        Returns:
            str: Concatenated text extracted from the image. Empty string if nothing detected.

        Raises:
            TypeError: If image_input is not an accepted type.
        """
        temp_dir = Path("temp_images_for_ocr")
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_image_path: Optional[Path] = None
        try:
            pil_img: Optional[Image.Image] = None
            base_name = "input"

            if isinstance(image_input, np.ndarray):
                pil_img = Image.fromarray(image_input)
                base_name = "ndarray_input"
            elif isinstance(image_input, Image.Image):
                pil_img = image_input
                base_name = "pil_input"
            elif isinstance(image_input, (str, os.PathLike)):
                with Image.open(image_input) as img:
                    pil_img = img.convert("RGB")
                base_name = os.path.basename(str(image_input))
            else:
                raise TypeError(f"Unsupported image_input type: {type(image_input)}")

            temp_image_path = temp_dir / f"{base_name}.png"
            assert pil_img is not None
            pil_img.convert("RGB").save(temp_image_path, "PNG")

            use_cls = getattr(self.engine, "use_angle_cls", False)
            result = self.engine.ocr(str(temp_image_path), cls=use_cls)

            # Post-inference introspection for rec/cls model paths if fallback was used
            try:
                if getattr(self, "_fallback_usage", None) and not getattr(self, "_rec_cls_path_logged", False):
                    for _k, _label in [("rec", "Recognition"), ("cls", "Classification")]:
                        if self._fallback_usage.get(_k):
                            cached_dir = self._introspect_cached_dir(self.engine, _k)
                            if cached_dir:
                                logger.info("PaddleOCR resolved %s model directory to: %s", _label, cached_dir)
                    self._rec_cls_path_logged = True
            except Exception:
                pass

            if not result or not result[0]:
                return ""

            text_lines = result[0]
            extracted_texts: List[str] = []
            if text_lines:
                for line in text_lines:
                    try:
                        text = line[1][0]
                        if isinstance(text, str):
                            extracted_texts.append(text)
                    except (IndexError, TypeError):
                        continue
            # Save last recognized lines for downstream debug/CSV consumers
            try:
                self.last_text_lines = extracted_texts[:]
            except Exception:
                self.last_text_lines = None
            return " ".join(extracted_texts)

        finally:
            try:
                if temp_image_path and temp_image_path.exists():
                    temp_image_path.unlink()
            except OSError:
                # Non-fatal cleanup error
                pass


# ==============================================================================
# === 3. OTHER ENGINE IMPLEMENTATIONS =========================================
# ==============================================================================

class EasyOCREngine(OCREngine):
    """EasyOCR-based implementation.

    Note:
        This implementation sets gpu=False by default. Offline model usage can be
        enabled via config if supported by easyocr.Reader.
    """

    def _initialize_engine(self) -> Any:
        import easyocr
        lang_value = self.config.get("language")
        lang_list = [lang_value] if isinstance(lang_value, str) else lang_value
        logger.info("OCR ENGINE INITIALIZATION (EasyOCR)")
        if self.config.get("use_offline_models", False):
            return easyocr.Reader(
                lang_list,
                model_storage_directory=self.config.get("offline_easyocr_model_dir"),
                download_enabled=False,
                gpu=False,
            )
        else:
            return easyocr.Reader(lang_list, gpu=False)

    def extract_text(self, image_data: Union[np.ndarray, Image.Image, str, PathLike]) -> str:
        result = self.engine.readtext(image_data, detail=0)
        try:
            self.last_text_lines = list(map(str, result))
        except Exception:
            self.last_text_lines = None
        return " ".join(result)


class TesseractOCREngine(OCREngine):
    """Tesseract-based implementation."""

    def _initialize_engine(self) -> Any:
        import pytesseract
        logger.info("OCR ENGINE INITIALIZATION (Tesseract)")
        try:
            pytesseract.pytesseract.tesseract_cmd = self.config.get("tesseract_cmd_path")
            # Verify path by checking the version.
            pytesseract.get_tesseract_version()
            logger.info("Tesseract executable found at: %s", self.config.get("tesseract_cmd_path"))
            return pytesseract
        except Exception as e:
            logger.error(
                "Tesseract not found or configured correctly. Path: '%s'. Reason: %s",
                self.config.get("tesseract_cmd_path"),
                e,
            )
            return None

    def extract_text(self, image_data: Union[np.ndarray, Image.Image, str, PathLike]) -> str:
        from PIL import Image as PILImage
        if isinstance(image_data, np.ndarray):
            pil_image = PILImage.fromarray(image_data)
        elif isinstance(image_data, PILImage.Image):
            pil_image = image_data
        elif isinstance(image_data, (str, os.PathLike)):
            with PILImage.open(image_data) as img:
                pil_image = img.convert("RGB")
        else:
            raise TypeError(f"Unsupported image_input type: {type(image_data)}")
        lang_code = "kor" if self.config.get("language") == "ko" else "eng"
        text = self.engine.image_to_string(pil_image, lang=lang_code)
        try:
            lines = [ln for ln in text.splitlines() if ln.strip()]
            self.last_text_lines = lines
        except Exception:
            self.last_text_lines = None
        return text


# ==============================================================================
# === 4. ENGINE FACTORY FUNCTION ===============================================
# ==============================================================================


def get_ocr_engine(config: Dict[str, Any]) -> Optional[OCREngine]:
    """Create and return the appropriate OCR engine instance.

    Args:
        config (Dict[str, Any]): Application configuration including 'ocr_engine'.

    Raises:
        ValueError: If an unknown engine name is specified in the config.

    Returns:
        Optional[OCREngine]: Initialized OCR engine instance, or None on failure.

    TODO:
        Validate presence of 'ocr_engine' key explicitly and raise a clearer
        error rather than relying on KeyError.
    """
    engine_map = {
        "paddleocr": PaddleOCREngine,
        "easyocr": EasyOCREngine,
        "tesseract": TesseractOCREngine,
    }
    engine_class = engine_map.get(config["ocr_engine"].lower())

    if engine_class:
        return engine_class(config)

    raise ValueError(f"FATAL: Unknown OCR engine specified: '{config['ocr_engine']}'.")
