"""Utility helpers for image discovery, normalization, text processing, and CSV export.

This module provides functions to:
- Recursively discover image files of supported formats from an input path.
- Standardize images into RGB NumPy arrays that are C-contiguous for OCR backends.
- Normalize free-form text for matching/search.
- Persist results to CSV with deterministic headers.

Usage examples are intended for internal library consumption; CLI code should handle
user interaction and configure logging handlers if desired.
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence, List, Union

import numpy as np
from PIL import Image

# Module-level logger (handlers configured by application/CLI layers)
logger = logging.getLogger(__name__)


def get_image_paths(input_path: Union[str, Path], supported_formats: Sequence[str]) -> List[str]:
    """Discover valid image files from the input path.

    If the path is a file, it is validated against the supported extensions.
    If the path is a directory, it is scanned recursively for matching files.

    Args:
        input_path (Union[str, Path]): File or directory to scan.
        supported_formats (Sequence[str]): Lowercase file extensions including the leading dot
            (e.g., [".png", ".jpg", ".tif"]). Matching is case-insensitive.

    Returns:
        List[str]: Absolute or relative file paths discovered, in the traversal order.

    Raises:
        None: Function logs and returns an empty list when the path does not exist.
    """
    p = Path(input_path)
    image_paths: List[str] = []

    if not p.exists():
        logger.error("Input path does not exist: %s", p)
        return image_paths

    # Normalize supported formats to lowercase with leading dot for robust checks.
    normalized_exts = tuple(ext.lower() for ext in supported_formats)

    if p.is_file():
        if p.suffix.lower() in normalized_exts or any(
            str(p).lower().endswith(ext) for ext in normalized_exts
        ):
            image_paths.append(str(p))
    elif p.is_dir():
        logger.info("Recursively scanning directory: %s", p)
        for dirpath, _dirnames, filenames in os.walk(p):
            for filename in filenames:
                if filename.lower().endswith(normalized_exts):
                    full_path = Path(dirpath) / filename
                    image_paths.append(str(full_path))

    logger.info("Found %d supported images to process.", len(image_paths))
    return image_paths


def standardize_image_for_ocr(image_path: Union[str, Path]) -> np.ndarray:
    """Open an image and return a C-contiguous RGB NumPy array.

    Certain formats (e.g., multi-page or compressed TIFF) can produce arrays with
    non-standard memory layouts. This function converts the image to 3-channel RGB
    and enforces a C-contiguous NumPy array to avoid downstream C/C++ library issues
    (e.g., in OpenCV-backed OCR engines).

    Args:
        image_path (Union[str, Path]): Path to an image file supported by Pillow.

    Returns:
        numpy.ndarray: C-contiguous RGB array with shape (H, W, 3) and dtype inferred
            by Pillow -> NumPy conversion.

    Raises:
        FileNotFoundError: If the image_path does not exist.
        OSError: If the image cannot be opened/decoded by Pillow.
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image path not found: {p}")

    # Context manager guarantees file handle is closed deterministically.
    with Image.open(p) as img:
        rgb_img = img.convert("RGB")
        numpy_array = np.array(rgb_img)
        # Ensure array's memory layout is C-contiguous for downstream C/C++ libraries.
        return np.ascontiguousarray(numpy_array)


def normalize_text(text: str, remove_spaces: bool = False) -> str:
    """Normalize a string by removing punctuation and condensing whitespace.

    Args:
        text (str): The input string to be normalized.
        remove_spaces (bool): When True, removes all spaces after normalization.

    Returns:
        str: The cleaned and normalized string.

    Raises:
        None

    TODO: Consider using Unicode categories (e.g., regex with \p{P}) for broader
        punctuation handling in multilingual contexts.
    """
    if not isinstance(text, str):
        return ""

    punctuation_to_remove = "'\"`.,-~*!@#$%^&()_+=[]{}|\\:;<>?"
    for punc in punctuation_to_remove:
        # Replace with a space to avoid accidental token concatenation
        text = text.replace(punc, " ")

    normalized_text = " ".join(text.split())

    if remove_spaces:
        return "".join(normalized_text.split())

    return normalized_text


def write_results_to_csv(
    output_path: Union[str, Path],
    results_data: Sequence[Mapping[str, Any]],
    header: Sequence[str],
) -> None:
    """Write collected results to a CSV file, ensuring the directory exists.

    Args:
        output_path (Union[str, Path]): Destination CSV file path.
        results_data (Sequence[Mapping[str, Any]]): Iterable of row dictionaries to write.
        header (Sequence[str]): Column names to use as CSV headers; must cover keys used
            in results_data. Extra keys in rows are ignored by DictWriter.

    Returns:
        None

    Raises:
        IOError: If the file cannot be written due to permissions or path issues.
    """
    p = Path(output_path)
    logger.info("Generating CSV report at: %s", p)

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(header))
            writer.writeheader()
            # DictWriter will ignore extra keys by default; this preserves current behavior.
            writer.writerows(results_data)
        logger.info("Successfully generated CSV report: %s", p)
    except IOError as e:
        # Log and re-raise to allow callers to decide how to handle failures.
        logger.error("Could not write CSV file at %s. Reason: %s", p, e)
        raise
