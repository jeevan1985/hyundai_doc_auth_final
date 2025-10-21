"""Configuration control panel for the TIF text searcher core package.

This module defines a structured configuration for OCR-driven text search across
multi-page TIF documents. The final `config` dictionary is produced by layering
engine-specific settings over the common defaults.

Conventions:
- Do not perform expensive work at import time.
- Module acts purely as a source of configuration data.
- Logger is provided for consistency; handlers should be configured by the caller/CLI.

Usage:
    from tif_searcher.tif_searcher_core.config import config
    engine_name = config.get('ocr_engine')

Notes:
- All existing keys and semantics are preserved to maintain backward compatibility.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ==============================================================================
# === CONFIGURATION CONTROL PANEL (v2.1, Self-Documenting) ===================
# ==============================================================================
#
# This file is organized into sections for maximum clarity. The final 'config'
# dictionary used by the application is built by merging the 'common' settings
# with the settings from the currently selected OCR engine.
#
# ==============================================================================

_structured_config: Dict[str, Any] = {
    # ==========================================================================
    # === ⚙️ COMMON CONFIGURATION (Applies to all engines) ======================
    # ==========================================================================
    "common": {
        # --- OCR Engine Selection ---
        # Select the primary OCR engine to use for text recognition.
        # The section below with the matching name will be loaded.
        # Available Options: 'paddleocr', 'easyocr', 'tesseract'
        "ocr_engine": "paddleocr",

        # --- Language & Search Parameters ---
        # The primary language of the text in your documents.
        # Examples: "ko" (Korean), "en" (English), "jp" (Japanese)
        "language": "ko",

        # The exact text phrase the system will search for within the images.
        "search_text": "가맹점 실사 사진",

        # The logic used to find a match.
        # Available Options:
        #   - 'exact_phrase': The entire `search_text` must appear consecutively.
        #   - 'all_words': All individual words in `search_text` must be on the
        #                  page, but can be in any order or position.
        "search_mode": "exact_phrase",

        # --- Search Area Control ---
        # Defines specific vertical zones within the image to perform OCR, saving resources.
        # To search the entire image, set this to an empty dictionary: {}.
        # Values are percentages (0.0 to 1.0) of the image height.
        #   - 'top': The top X% of the image.
        #   - 'bottom': The bottom X% of the image.
        #   - 'center': A central slice of X% of the image's height.
        "search_location": {"top": 0.05},

        # --- File Paths & Reporting ---
        # The top-level folder containing your images. The tool will search recursively.
        "input_path": r"../doc_image_synthetic_datagen/final_synthetic_dataset",

        # The directory where found images or pages will be copied.
        "output_folder": "search_results",

        # If True, a detailed CSV report of all processed files will be generated.
        "create_csv_report": True,
        "csv_output_path": "search_results/search_report.csv",

        # --- Normalization & Debugging ---
        # If True, cleans text (removes punctuation, standardizes case) before matching.
        # Recommended to keep True for robust searching.
        "allow_recognition_normalization": True,

        # If True (and in 'exact_phrase' mode), all spaces are removed from both the
        # search query and OCR text. Helps match "Hello World" to "HelloWorld".
        "remove_spaces_in_normalization": True,

        # If True, prints all raw text extracted from each image to the console.
        # Extremely useful for debugging why a search term might not be found.
        "recognized_text_debug": True,

        # --- Offline Mode ---
        # If True, forces the system to ONLY use local model paths and prevents
        # any attempt to download models from the internet. Ideal for air-gapped systems.
        "use_offline_models": False,

        # --- Advanced Settings ---
        # A list of file extensions to be considered for processing.
        "supported_formats": ['.tif', '.tiff', '.jpg', '.jpeg', '.png'],

        # Provides a way to map user-friendly language codes (like 'ko') to the
        # specific internal names an OCR engine expects (e.g., 'korean').
        "language_code_map": {
            "ko": "korean",
            "jp": "japan"
        },

        # --- Logging policy (future readiness; currently console-only) ---
        "logging": {
            # Default (hardcoded): false
            "backup_logs": False,
            # Default (hardcoded): 7
            "remove_logs_days": 7,
        },
    },

    # ==========================================================================
    # === 🧠 PADDLEOCR ENGINE CONFIGURATION =====================================
    # ==========================================================================
    "paddleocr": {
        # --- Performance & Accuracy ---
        # If True, will use a compatible NVIDIA GPU. Requires CUDA/cuDNN to be installed.
        "use_gpu": True, #False,

        # If True, uses the text angle classifier to correct rotated text before recognition.
        # Improves accuracy but adds a small performance overhead.
        "use_angle_cls": True,

        # The number of images to process in a single batch. Higher numbers can improve
        # throughput on powerful GPUs.
        "rec_batch_num": 6,

        # The language for the text angle classification model. Typically 'ch' for Chinese/English.
        "classification_language": "ch",

        # --- Model Download URLs (Used if 'use_offline_models' is False) ---
        # Leave a path as "" (empty string) to let PaddleOCR download the default model automatically.
        # Provide a specific URL to download a custom or different version of a model.
        "download_path_det_model": "",
        "download_path_rec_model": "",
        "download_path_cls_model": "",

        # --- Offline Model Directories (Used if 'use_offline_models' is True) ---
        # These must point to the directories containing the 'inference.pdmodel' files.
        # "offline_det_model_dir": r"C:\\Users\\jeeb\\.paddlex\\official_models\\PP-OCRv3_det",
        # "offline_rec_model_dir": r"C:\\Users\\jeeb\\.paddlex\\official_models\\korean_PP-OCRv3_rec",
        # "offline_cls_model_dir": r"C:\\Users\\jeeb\\.paddlex\\official_models\\ch_ppocr_mobile_v2.0_cls_infer"
        "offline_det_model_dir": r"trained_model/tif_text_searcher/.paddleocr/whl/det/ml/Multilingual_PP-OCRv3_det_infer",
        "offline_rec_model_dir": r"trained_model/tif_text_searcher/.paddleocr/whl/rec/korean/korean_PP-OCRv4_rec_infer/",
        "offline_cls_model_dir": r"trained_model/tif_text_searcher/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer"
    },

    # ==========================================================================
    # === ⬜ TESSERACT ENGINE CONFIGURATION =====================================
    # ==========================================================================
    "tesseract": {
        # Tesseract is always used offline. This path MUST point to the tesseract.exe file.
        "tesseract_cmd_path": 'tesseract'  # On Windows, this might be: r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        },

    # ==========================================================================
    # === 🇪 EASYOCR ENGINE CONFIGURATION =======================================
    # ==========================================================================
    "easyocr": {
        # This path is ONLY used if 'use_offline_models' is True.
        # It should point to the parent folder where EasyOCR models are stored
        # (the folder you might copy from a machine with internet access).
        "offline_model_dir": r"C:\\ocr_models\\.EasyOCR"
    }
}

# ==============================================================================
# === CONFIGURATION MERGING LOGIC (Do not edit below this line) ================
# ==============================================================================
# This logic creates the final 'config' dictionary that the application uses.
# It takes the 'common' settings and merges the settings for the selected
# OCR engine on top of them, ensuring the correct parameters are active.

config: Dict[str, Any] = _structured_config['common'].copy()
_engine_name: Optional[str] = config.get('ocr_engine')

if _engine_name and _engine_name in _structured_config:
    # Merge the engine-specific settings into the main config
    config.update(_structured_config[_engine_name])

# TODO: Consider validating _engine_name against known engines at startup
# and providing a clearer error if the engine is not recognized.
