# docprosight/detection.py
"""
Object detection module using YOLOv8. This module defines the `YOLODetector`
class, responsible for loading a trained model and performing inference on
single or batch images to detect and crop target objects.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union, Sequence, Optional, Any

from PIL import Image

from .utils import draw_bounding_boxes

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Structured information describing a single detected object.

    Attributes:
        cropped_image (PIL.Image.Image): Cropped image of the detected object.
        class_name (str): Detected class name.
        confidence (float): Confidence score in [0, 1].
        box_xyxy (Tuple[int, int, int, int]): Bounding box (x1, y1, x2, y2) in pixels.
        source_page_index (int): 0-based index of the page/image in the processed batch.
        object_index_on_page (int): 0-based index of the object on the page.
    """
    cropped_image: Image.Image
    class_name: str
    confidence: float
    box_xyxy: Tuple[int, int, int, int]
    source_page_index: int
    object_index_on_page: int

class YOLODetector:
    """YOLOv8 wrapper for batch detection, cropping, and visualization.

    Args:
        model_path (str): Path to the trained YOLOv8 .pt model file.
        inference_cfg (Dict): Inference configuration including thresholds, img size,
            target_object_names, and gpu_inference mode ('true'|'false'|'auto').

    Raises:
        ImportError: If ultralytics is not installed.
        FileNotFoundError: If the model file path is invalid or missing.
    """
    def __init__(self, model_path: str, inference_cfg: Dict) -> None:
        """Initialize the YOLODetector with a model and inference configuration.

        See class docstring for details. Raises if prerequisites are not met.
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("The 'ultralytics' package is required for detection. Please install it.")
        
        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"YOLO model not found at: {model_file}")

        self.model = YOLO(model_file)
        self.config = inference_cfg
        self.target_indices = self._get_target_indices(self.config.get('target_object_names', []))
        logger.info(f"YOLO Detector initialized with model: {model_path}")
        logger.info(f"Targeting object classes: {self.config.get('target_object_names', [])}")

    def _get_target_indices(self, target_names: List[str]) -> set:
        """Get integer class indices for the configured target class names.

        Args:
            target_names (List[str]): Target class names expected in the model's class list.

        Returns:
            set: Set of integer indices corresponding to target_names that exist in model.names.

        Note:
            If any names are not found, a warning is logged and they are ignored.
        """
        model_names = self.model.names
        name_to_idx = {name: idx for idx, name in model_names.items()}
        indices = {name_to_idx[name] for name in target_names if name in name_to_idx}
        if len(indices) != len(target_names):
            logger.warning("Some target class names in config were not found in the YOLO model's class list.")
        return indices

    def detect_and_crop(
        self,
        images: Union[Image.Image, List[Image.Image]],
        visualize: bool = False,
        start_page_idx: int = 0
    ) -> Tuple[List[DetectionResult], List[Image.Image]]:
        """Run detection on images, crop targets, and optionally visualize results.

        Args:
            images (Union[Image.Image, List[Image.Image]]): A single PIL Image or a list of them.
            visualize (bool): If True, returns annotated versions of the input images.
            start_page_idx (int): Starting page index for labeling, useful when processing in chunks.

        Returns:
            Tuple[List[DetectionResult], List[Image.Image]]: Detection results and optional visualizations.
        """
        images_to_process = [images] if isinstance(images, Image.Image) else images
        
        all_detection_results: List[DetectionResult] = []
        visualized_images: List[Image.Image] = []

        if not images_to_process:
            return all_detection_results, visualized_images

        logger.info("Running YOLO detection on a batch of %d page(s)...", len(images_to_process))
        
        # Determine device behavior from config
        gpu_mode = str(self.config.get('gpu_inference', 'auto')).lower()
        if gpu_mode not in {'true', 'false', 'auto'}:
            logger.warning(f"Unknown gpu_inference value '{gpu_mode}', defaulting to 'auto'.")
            gpu_mode = 'auto'

        def _set_device(device_opt: Union[int, str]) -> None:
            """Move the YOLO model to the requested device.

            Args:
                device_opt (Union[int, str]): Device index or name (e.g., 0, 'cpu', 'cuda:0').
            """
            try:
                self.model.to(device_opt)
            except Exception:
                # Fallback to explicit string devices if numeric index unsupported
                dev = 'cuda:0' if device_opt == 0 else 'cpu'
                self.model.to(dev)

        def _predict(device_opt: Union[int, str]) -> Any:
            """Run YOLO prediction on the prepared list of images for a device.

            Args:
                device_opt (Union[int, str]): Device selection (index or name) used by YOLO.

            Returns:
                Any: The predictions object produced by ultralytics.YOLO.predict.
            """
            _set_device(device_opt)
            return self.model.predict(
                source=images_to_process,
                conf=self.config.get('confidence_threshold', 0.25),
                iou=self.config.get('iou_threshold', 0.45),
                imgsz=self.config.get('imgsz', 640),
                device=device_opt,
                verbose=False
            )

        try:
            if gpu_mode == 'false':
                predictions = _predict('cpu')
            elif gpu_mode == 'true':
                predictions = _predict(0)
            else:  # auto
                try:
                    predictions = _predict(0)
                except Exception as e:
                    msg = str(e)
                    if 'torchvision::nms' in msg and 'CUDA' in msg:
                        logger.warning("CUDA NMS not available; falling back to CPU as per 'auto' mode.")
                        predictions = _predict('cpu')
                    else:
                        # For any other exception in auto mode, re-raise
                        raise
        except NotImplementedError as e:
            if gpu_mode == 'true' and 'torchvision::nms' in str(e) and 'CUDA' in str(e):
                # Explicit GPU-only mode should not fallback silently; surface the error
                logger.error("GPU-only mode requested but CUDA NMS is unavailable.")
            raise
        
        for i, preds in enumerate(predictions):
            page_idx = start_page_idx + i
            page_pil = images_to_process[i]
            boxes_data = preds.boxes.data.cpu().numpy()

            if visualize:
                annotated_image = draw_bounding_boxes(page_pil, boxes_data, self.model.names)
                visualized_images.append(annotated_image)

            for obj_idx, det in enumerate(boxes_data):
                x1, y1, x2, y2, conf, cls_idx_float = det
                cls_idx = int(cls_idx_float)
                if cls_idx not in self.target_indices:
                    continue

                class_name = self.model.names.get(cls_idx, f"class_{cls_idx}")
                crop_box = (
                    max(0, int(x1)), max(0, int(y1)),
                    min(page_pil.width, int(x2)), min(page_pil.height, int(y2))
                )
                if crop_box[0] >= crop_box[2] or crop_box[1] >= crop_box[3]:
                    continue  # Skip zero-area boxes
                
                cropped_pil_img = page_pil.crop(crop_box)
                all_detection_results.append(
                    DetectionResult(
                        cropped_image=cropped_pil_img,
                        class_name=class_name,
                        confidence=float(conf),
                        box_xyxy=crop_box,
                        source_page_index=page_idx,
                        object_index_on_page=obj_idx
                    )
                )

        return all_detection_results, visualized_images