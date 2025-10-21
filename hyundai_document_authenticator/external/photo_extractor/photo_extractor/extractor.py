"""
Provides a simplified, high-level interface for extracting photos from documents.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union, Tuple

from PIL import Image, ImageSequence

from .config import DEFAULT_CONFIG, deep_merge
from .detection import YOLODetector

logger = logging.getLogger(__name__)


class PhotoExtractor:
    """High-level facade for extracting photos from document pages.

    Supports:
        1) YOLO-based detection via extract_photos (automatic discovery).
        2) Bounding-box-based cropping via extract_photos_from_bboxes (predefined ROIs).

    Args:
        config_override (Optional[dict]): Partial overrides applied over DEFAULT_CONFIG.
            For YOLO-based extraction, must provide 'yolo_object_detection.model_path'.

    Attributes:
        mode (str): Extraction mode: 'yolo' or 'bbox'.
        detector (Optional[YOLODetector]): YOLO detector when mode='yolo' and model is configured.
    """

    def __init__(self, config_override: Optional[dict] = None):
        """
        Initializes the PhotoExtractor.

        Notes:
            - If you plan to use YOLO-based extraction via `extract_photos`, you must
              provide `yolo_object_detection.model_path` in the configuration.
            - For bounding-box-only usage via `extract_photos_from_bboxes`, the YOLO
              detector is not required and may be omitted.
        """
        # Start with the default config and merge any overrides (user overrides win)
        cfg = DEFAULT_CONFIG.copy()
        if config_override:
            cfg = deep_merge(config_override, cfg)

        # Resolve mode with safe default
        self.mode = str(cfg.get("photo_extraction_mode", "bbox")).lower()

        # Initialize YOLO only for explicit yolo mode and when a model path is provided
        self.detector: Optional[YOLODetector] = None
        if self.mode == "yolo":
            yolo_config = cfg.get("yolo_object_detection", {}) or {}
            model_path = yolo_config.get("model_path")
            if model_path:
                self.detector = YOLODetector(
                    model_path=model_path,
                    inference_cfg=yolo_config.get("inference", {}) or {},
                )

        logger.info("PhotoExtractor init: mode=%s, yolo_initialized=%s", self.mode, str(self.detector is not None).lower())

    def extract_photos(
        self, tif_path: Union[str, Path], page_number: int
    ) -> List[Image.Image]:
        """
        Extracts all detected photos from a single page of a TIF document using YOLO.

        Args:
            tif_path (Union[str, Path]): The full path to the TIF document.
            page_number (int): The 1-based page number from which to extract photos.

        Returns:
            List[Image.Image]: A list of PIL Images, each being a cropped photo.
                               Returns an empty list if no photos are detected.

        Raises:
            FileNotFoundError: If the specified TIF file does not exist.
            ValueError: If the page_number is out of bounds.
            RuntimeError: If YOLO detector is not initialized (no model provided).
        """
        # Enforce mode gating
        if getattr(self, "mode", "bbox") != "yolo":
            raise RuntimeError(
                f"extract_photos is only available in 'yolo' mode. Current mode is '{getattr(self, 'mode', 'bbox')}'. "
                "Use 'extract_photos_from_bboxes' or set 'photo_extraction_mode: yolo'."
            )
        if self.detector is None:
            raise RuntimeError(
                "YOLO requested but model not provided/initialized. Provide 'yolo_object_detection.model_path' "
                "in the config. No bbox path will be used in 'yolo' mode."
            )

        tif_file = Path(tif_path)
        if not tif_file.is_file():
            raise FileNotFoundError(f"TIF file not found at: {tif_file}")

        try:
            with Image.open(tif_file) as img:
                if not (1 <= page_number <= img.n_frames):
                    raise ValueError(
                        f"Invalid page number: {page_number}. "
                        f"Document has {img.n_frames} pages."
                    )

                # Seek to the desired page (0-indexed)
                img.seek(page_number - 1)
                # Ensure image is in a standard format for detection
                page_image = img.convert("RGB")

        except Exception as e:
            raise IOError(f"Failed to read page {page_number} from {tif_file}: {e}") from e

        # The detector expects a list of images
        detection_results, _ = self.detector.detect_and_crop([page_image])

        # Return only the cropped PIL images
        return [res.cropped_image for res in detection_results]

    @staticmethod
    def _to_xyxy_boxes(
        bboxes: List[Tuple[float, float, float, float]],
        img_w: int,
        img_h: int,
        bbox_format: str = "xyxy",
        normalized: bool = False,
    ) -> List[Tuple[int, int, int, int]]:
        """Convert boxes to absolute pixel XYXY format and clip to image bounds.

        Args:
            bboxes (List[Tuple[float, float, float, float]]): Input boxes; interpretation depends on bbox_format.
            img_w (int): Image width in pixels.
            img_h (int): Image height in pixels.
            bbox_format (str): One of {'xyxy', 'xywh', 'cxcywh'}.
            normalized (bool): If True, input coords are in [0,1] relative to image size.

        Returns:
            List[Tuple[int, int, int, int]]: Clipped (x1, y1, x2, y2) integer boxes with positive area.
        """
        xyxy_boxes: List[Tuple[int, int, int, int]] = []
        for box in bboxes:
            x1, y1, x2, y2 = 0, 0, 0, 0
            try:
                if bbox_format == "xyxy":
                    x1, y1, x2, y2 = box
                elif bbox_format == "xywh":
                    x, y, w, h = box
                    x1, y1, x2, y2 = x, y, x + w, y + h
                elif bbox_format == "cxcywh":
                    cx, cy, w, h = box
                    x1, y1 = cx - w / 2.0, cy - h / 2.0
                    x2, y2 = cx + w / 2.0, cy + h / 2.0
                else:
                    # Unknown format, skip
                    continue

                if normalized:
                    x1 *= img_w; x2 *= img_w
                    y1 *= img_h; y2 *= img_h

                xi1 = max(0, int(round(x1)))
                yi1 = max(0, int(round(y1)))
                xi2 = min(img_w, int(round(x2)))
                yi2 = min(img_h, int(round(y2)))

                if xi1 < xi2 and yi1 < yi2:
                    xyxy_boxes.append((xi1, yi1, xi2, yi2))
            except Exception:
                # Skip malformed boxes
                continue
        return xyxy_boxes

    def extract_photos_from_bboxes(
        self,
        image: Image.Image,
        bboxes: List[Tuple[float, float, float, float]],
        bbox_format: str = "xyxy",
        normalized: bool = False,
    ) -> List[Image.Image]:
        """Crop photos from an image using pre-defined bounding boxes.

        Args:
            image (PIL.Image.Image): Single page image to crop from.
            bboxes (List[Tuple[float, float, float, float]]): Bounding boxes; interpretation depends on bbox_format.
            bbox_format (str): One of: 'xyxy' -> (x1,y1,x2,y2); 'xywh' -> (x,y,width,height);
                'cxcywh' -> (cx,cy,width,height).
            normalized (bool): If True, input coords are in [0,1] relative to image size.

        Returns:
            List[Image.Image]: Cropped images. Empty list if no valid boxes.
        """
        page = image.convert("RGB")
        w, h = page.width, page.height
        boxes_xyxy = self._to_xyxy_boxes(bboxes, w, h, bbox_format=bbox_format, normalized=normalized)
        crops: List[Image.Image] = []
        for (x1, y1, x2, y2) in boxes_xyxy:
            crops.append(page.crop((x1, y1, x2, y2)))
        return crops


__all__ = ["PhotoExtractor"]
