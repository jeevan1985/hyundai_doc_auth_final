# docprosight/config.py
"""
Configuration management for the DocProSight pipeline.

This module provides a default configuration dictionary and functions to load
and merge user-provided YAML configuration files.

Do not perform any heavy work at import time; this module only defines data and
helpers for configuration handling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import yaml

logger = logging.getLogger(__name__)

# --- Default Configuration ---
# This dictionary contains all configurable parameters with sensible defaults.
# Users can override these settings with a custom `config.yaml` file.
DEFAULT_CONFIG: Dict[str, Any] = {
    "project_meta": {
        # Logging level for console and file; choices: "DEBUG", "INFO", "WARNING", "ERROR"
        "log_level": "INFO",
    },
    "document_input": {
        # DPI for PDF rasterization. Has no effect for TIFF/JPEG/PNG inputs.
        "dpi": 300,
        # Poppler bin directory (Windows) if PDF support is needed; leave None on Linux/macOS or if not using PDFs.
        "poppler_path": None,
    },
    # Primary extraction mode selection: 'yolo' (detector) or 'bbox' (static boxes)
    # This can be overridden by higher-level configs.
    "photo_extraction_mode": "bbox",
    # Module-level debug switch; higher-level configs can override this.
    "photo_extractor_debug": False,
    # Logging policy for this module (housekeeping of run logs)
    "logging": {
        # Default (hardcoded): false
        "backup_logs": False,
        # Default (hardcoded): 7
        "remove_logs_days": 7,
    },
    "yolo_object_detection": {
        # Absolute/relative path to YOLOv8 .pt model. Required for detection-based extraction.
        "model_path": "trained_model/yolo_photo_extractor/weights/best.pt",
        "inference": {
            # Confidence threshold [0.0, 1.0]
            "confidence_threshold": 0.30,
            # IoU threshold for NMS [0.0, 1.0]
            "iou_threshold": 0.45,
            # Names must match your model's class names; only these will be cropped.
            "target_object_names": ["photo", "image", "figure"],
            # YOLO inference image size. Typical values: 416, 512, 640.
            "imgsz": 640,
            # Hardware selection for inference: 'true' (GPU only), 'false' (CPU only), 'auto' (GPU with CPU fallback)
            "gpu_inference": "auto",
        },
    },
    "bbox_extraction": {
        # Optional static bounding boxes mode. Used when YOLO is not desired.
        # bbox_list: list of boxes; each box as [x1, y1, x2, y2] for xyxy, etc.
        "bbox_list": [[171,236,1480,1100],[171,1168,1480,2032]], #[],  # e.g., [[50,60,300,260], [400,80,700,280]]
        # bbox_format choices: "xyxy" (x1,y1,x2,y2), "xywh" (x,y,width,height), "cxcywh" (center x,y,width,height)
        "bbox_format": "xyxy",
        # If True, bbox coordinates are normalized in [0,1] relative to image size
        "normalized": False
    },
    "results": {
        # Save cropped images to disk
        "save_cropped_images": True,
        # Destination base folder for all outputs (crops, visualizations, summary).
        # If not provided, CLI --output or './photo_extractor_results' will be used.
        "output_folder_path": None,
        # File name for JSON summary written at the end of a run
        "summary_filename": "docprosight_summary.json",
        # If True, also produce annotated page visualizations under 'visualizations/<doc_stem>/'
        "visualize": False,
    },
}

def deep_merge(source: Dict[str, Any], destination: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges two dictionaries. The `source` dictionary's values
    overwrite the `destination` dictionary's values.

    Args:
        source (Dict): The dictionary with values to merge.
        destination (Dict): The dictionary to be merged into.

    Returns:
        Dict: The merged dictionary.
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # Get node or create one
            node = destination.setdefault(key, {})
            deep_merge(value, node)
        else:
            destination[key] = value
    return destination

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file and deep merges it with the default config.

    This ensures that all necessary configuration keys are present, and allows
    the user to only specify the settings they wish to change.

    Args:
        config_path (str): The path to the user's YAML configuration file.

    Returns:
        Dict[str, Any]: The final, merged configuration dictionary.
    """
    try:
        config_path_obj = Path(config_path)
        with config_path_obj.open('r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f) or {}

        # Start with a copy of the default config and merge user's config on top
        final_config = DEFAULT_CONFIG.copy()
        final_config = deep_merge(user_config, final_config)
        return final_config

    except FileNotFoundError:
        raise ValueError(f"Configuration file not found at: {config_path}")
    except Exception as e:
        raise ValueError(f"Error parsing configuration file {config_path}: {e}")