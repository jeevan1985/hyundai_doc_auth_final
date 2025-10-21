# docprosight/utils.py
"""
Core utility functions for image, file, and text handling in the DocProSight pipeline.
This module provides standalone helper functions that are used across different parts
of the application, such as image format conversion, document rasterization,
text normalization, and visualization.
"""
from __future__ import annotations

import cv2
import logging
import re
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, List, Optional, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Image Conversion and Drawing Utilities ---

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Converts a PIL.Image object to an OpenCV (NumPy BGR) image.

    Args:
        pil_image (Image.Image): The input image in PIL format.

    Returns:
        np.ndarray: The converted image in OpenCV's BGR format.
    """
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

def draw_bounding_boxes(
    image: Image.Image,
    detections: Sequence[Sequence[float]],
    class_names_map: Mapping[int, str],
    font_size: int = 15
) -> Image.Image:
    """
    Draws bounding boxes and labels on a copy of a PIL image for visualization.

    Args:
        image (Image.Image): The original image to draw on.
        detections (list): A list of detection results from YOLO (e.g., preds[0].boxes.data).
        class_names_map (dict): A dictionary mapping class index to class name (e.g., model.names).
        font_size (int): The font size for the labels.

    Returns:
        Image.Image: A new image with bounding boxes and labels drawn on it.
    """
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    try:
        # Attempt to use a common, more readable font.
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # Fallback to a basic default font if Arial is not found.
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2, conf, cls_idx_float = det
        cls_idx = int(cls_idx_float)
        class_name = class_names_map.get(cls_idx, f"class_{cls_idx}")
        label = f"{class_name} {conf:.2f}"

        # Draw the bounding box rectangle
        draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline="red", width=2)
        
        # Create a solid background for the text label for better visibility
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = int(x1)
        text_y = int(y1) - text_height - 5  # Position text above the box

        # Adjust position if the label would go off the top of the image
        if text_y < 0:
            text_y = int(y1) + 5

        draw.rectangle([text_x, text_y, text_x + text_width + 4, text_y + text_height + 4], fill="red")
        
        # Draw the text itself
        draw.text((text_x + 2, text_y + 2), label, fill="white", font=font)

    return img_with_boxes

# --- Document Processing Utilities ---

def get_doc_page_images(
    doc_path: Union[str, Path], dpi: int, poppler_path: Optional[str] = None
) -> List[Image.Image]:
    """
    Converts a document (PDF, TIFF, or standard image) into a list of PIL Images.

    Args:
        doc_path (Path): Path to the document file.
        dpi (int): Dots per inch for rasterizing PDFs.
        poppler_path (Optional[str]): Path to the Poppler binary directory (for Windows).

    Returns:
        List[Image.Image]: A list of page images as PIL.Image objects.

    Raises:
        FileNotFoundError: If the document path does not exist.
        ValueError: If pdf2image is required but not installed, or if the file type is unsupported.
    """
    p = Path(doc_path)
    if not p.is_file():
        raise FileNotFoundError(f"Document not found at: {p}")

    images_pil: List[Image.Image] = []
    suffix = p.suffix.lower()
    logger.info("Converting document '%s' (type: %s) to images.", p.name, suffix)

    try:
        if suffix == ".pdf":
            if not PDF2IMAGE_AVAILABLE:
                raise ValueError(
                    "PDF processing requires 'pdf2image' and Poppler. "
                    "Please install them to handle PDF files."
                )
            images_pil = convert_from_path(
                p, dpi=dpi, poppler_path=poppler_path, fmt="jpeg", thread_count=4
            )
        elif suffix in [".tif", ".tiff"]:
            with Image.open(p) as img:
                for i in range(img.n_frames):
                    img.seek(i)
                    images_pil.append(img.convert("RGB").copy())
        elif suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
            with Image.open(p) as img:
                images_pil.append(img.convert("RGB").copy())
        else:
            raise ValueError(f"Unsupported document type: '{suffix}' for file '{doc_path.name}'")
        
        logger.info("Successfully converted '%s' into %d page image(s).", p.name, len(images_pil))
        return images_pil
    except Exception as e:
        logger.error("An error occurred while converting document '%s': %s", p.name, e, exc_info=True)
        raise  # Re-raise the exception to be handled by the main pipeline

# --- Text Processing Utilities ---

def normalize_ocr_text(text: str, normalization_cfg: Mapping[str, Any]) -> str:
    """
    Normalizes OCR'd text based on specified configuration rules.
    This typically involves converting to lowercase, removing punctuation, and standardizing whitespace.

    Args:
        text (str): The raw text extracted from OCR.
        normalization_cfg (dict): Configuration dictionary for normalization steps.

    Returns:
        str: The normalized text.
    """
    if normalization_cfg.get("to_lowercase", True):
        text = text.lower()
    if punc_regex := normalization_cfg.get("remove_punctuation_regex"):
        text = re.sub(punc_regex, "", text)
    if ws_regex := normalization_cfg.get("remove_whitespace_regex"):
        text = re.sub(ws_regex, " ", text).strip()
    return text

# --- Bounding box-based extraction utility ---

def extract_with_bbox(
    image: Image.Image,
    bbox_list: Sequence[Sequence[float]],
    bbox_format: str = "xyxy",
    normalized: bool = False,
    clip: bool = True,
) -> List[Image.Image]:
    """
    Extract cropped images from a PIL image based on provided bounding boxes.

    Supports common formats:
      - "xyxy": [x1, y1, x2, y2]
      - "xywh": [x, y, w, h] (top-left based)
      - "cxcywh": [cx, cy, w, h] (center-based)

    Args:
        image: PIL.Image to crop from.
        bbox_list: List of bounding boxes.
        bbox_format: One of {"xyxy", "xywh", "cxcywh"}.
        normalized: If True, coordinates are in [0,1] relative to image width/height.
        clip: If True, clip the box to image bounds.

    Returns:
        List[Image.Image]: Cropped images in the order of the input boxes.
    """
    W, H = image.size
    fmt = bbox_format.lower()
    crops: List[Image.Image] = []

    def to_xyxy(box: List[float]) -> Optional[List[int]]:
        """Convert a bbox from the configured format to integer xyxy coordinates.

        Args:
            box (List[float]): A single bounding box in the format specified by `bbox_format`.

        Returns:
            Optional[List[int]]: [x1, y1, x2, y2] as integers, or None if invalid.
        """
        try:
            if fmt == "xyxy":
                x1, y1, x2, y2 = box
            elif fmt == "xywh":
                x, y, w, h = box
                x1, y1, x2, y2 = x, y, x + w, y + h
            elif fmt == "cxcywh":
                cx, cy, w, h = box
                x1, y1 = cx - w / 2.0, cy - h / 2.0
                x2, y2 = cx + w / 2.0, cy + h / 2.0
            else:
                logger.warning("Unsupported bbox_format '%s'", bbox_format)
                return None

            if normalized:
                x1, x2 = x1 * W, x2 * W
                y1, y2 = y1 * H, y2 * H

            if clip:
                x1 = max(0.0, min(float(x1), float(W - 1)))
                y1 = max(0.0, min(float(y1), float(H - 1)))
                x2 = max(0.0, min(float(x2), float(W)))
                y2 = max(0.0, min(float(y2), float(H)))

            ix1, iy1, ix2, iy2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            if ix2 <= ix1 or iy2 <= iy1:
                return None
            return [ix1, iy1, ix2, iy2]
        except Exception as e:
            logger.warning("Invalid bbox %s: %s", box, e)
            return None

    for b in bbox_list:
        xyxy = to_xyxy(b)
        if not xyxy:
            continue
        x1, y1, x2, y2 = xyxy
        try:
            crops.append(image.crop((x1, y1, x2, y2)))
        except Exception as e:
            logger.warning("Failed to crop box %s: %s", xyxy, e)
            continue

    return crops