# docprosight/pipeline.py
"""
The main pipeline orchestrator for the DocProSight application. This module
defines the `DocumentPipeline` class, which brings all the components (detection,
OCR, similarity) together to process documents from start to finish.
"""
from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from tqdm import tqdm

from . import utils
from .detection import YOLODetector

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """Orchestrate the end-to-end document processing workflow.

    This pipeline discovers documents, converts them to images, runs object
    detection, optionally saves visualizations and crops, and produces a
    JSON summary.

    Attributes:
        config (Dict[str, Any]): Loaded configuration dictionary.
        detector (YOLODetector): Object detector for crop extraction and visualization.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the pipeline with a given configuration and sets up the
        necessary model engines.

        Args:
            config (Dict[str, Any]): The full configuration dictionary.
        """
        self.config = config
        self.detector = YOLODetector(
            model_path=config["yolo_object_detection"]["model_path"],
            inference_cfg=config["yolo_object_detection"]["inference"],
        )

    def run(self, input_path: Union[str, Path], output_dir: Union[str, Path]) -> None:
        """Execute the pipeline: discover, process, and summarize documents.

        Finds documents under the provided path, processes each through detection
        and optional visualization/cropping, and writes a JSON summary.

        Args:
            input_path (Union[str, Path]): Path to a document or a directory of documents.
            output_dir (Union[str, Path]): Directory where results (crops, visualizations, summary) are saved.

        Returns:
            None

        Raises:
            FileNotFoundError: If the input path does not exist.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        docs_to_process = self._get_docs_to_process(input_path)
        if not docs_to_process:
            logger.warning("No valid documents found to process at the specified input path.")
            return

        all_results: List[Dict[str, Any]] = []
        for doc in tqdm(docs_to_process, desc="Processing Documents"):
            try:
                result = self._process_single_document(doc, output_path)
                all_results.append(result)
            except Exception as e:
                logger.error("Failed to process document '%s': %s", doc, e, exc_info=True)
                all_results.append({
                    "document_path": str(Path(doc).resolve()),
                    "status": "Failed",
                    "error_message": str(e),
                    "extracted_images_details": []
                })

        summary_path = output_path / self.config["results"]["summary_filename"]
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        logger.info("======> Analysis complete. Full summary saved to: %s", summary_path)

    def _get_docs_to_process(self, input_path: Union[str, Path]) -> List[Path]:
        """Identify document files to be processed.

        Args:
            input_path (Union[str, Path]): Path to a single document or a directory.

        Returns:
            List[Path]: Valid document paths filtered by supported suffixes, sorted by name.

        Raises:
            FileNotFoundError: If the input path does not exist.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        docs: List[Path] = []
        supported_suffixes = {".pdf", ".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}
        if input_path.is_dir():
            docs.extend([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in supported_suffixes])
        elif input_path.is_file() and input_path.suffix.lower() in supported_suffixes:
            docs.append(input_path)
        # Non-invasive improvement: deterministic processing order
        docs.sort(key=lambda p: p.name.lower())
        return docs

    def _process_single_document(self, doc_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Execute the analysis pipeline on a single input document.

        Args:
            doc_path (Path): Path to the input document.
            output_dir (Path): Directory where outputs (visualizations, crops) are written.

        Returns:
            Dict[str, Any]: Summary including status, error text if any, and extraction details.

        Raises:
            Exception: Exceptions are caught and recorded in the returned summary; no propagation.
        """
        doc_name_stem = doc_path.stem
        logger.info("--- Analyzing Document: %s ---", doc_path.name)

        doc_summary: Dict[str, Any] = {
            "document_path": str(doc_path.resolve()),
            "status": "Pending",
            "error_message": None,
            "extracted_images_details": [],
        }

        try:
            # --- Stage 1: Convert Document to Page Images ---
            page_images = utils.get_doc_page_images(
                doc_path, self.config["document_input"]["dpi"], self.config["document_input"].get("poppler_path")
            )

            # --- Stage 2: Batch Detection ---
            detection_results, viz_images = self.detector.detect_and_crop(
                page_images, visualize=self.config.get("results", {}).get("visualize", False)
            )

            if not detection_results:
                doc_summary["status"] = "Success - No Objects Found"
                return doc_summary

            # --- Stage 3: Save Visualizations (if created) ---
            if viz_images:
                viz_dir = output_dir / "visualizations" / doc_name_stem
                viz_dir.mkdir(parents=True, exist_ok=True)
                for i, img in enumerate(viz_images):
                    img.save(viz_dir / f"page_{i}_annotated.jpg")

            # --- Stage 4: Compile and Save Final Results ---
            extracted_details: List[Dict[str, Any]] = []
            crops_dir = output_dir / "cropped_images" / doc_name_stem
            if self.config.get("results", {}).get("save_cropped_images", True):
                crops_dir.mkdir(parents=True, exist_ok=True)

            for det_res in detection_results:
                crop_path_str = ""
                if self.config.get("results", {}).get("save_cropped_images", True):
                    crop_filename = f"page{det_res.source_page_index}_obj{det_res.object_index_on_page}_{det_res.class_name}.png"
                    crop_path = crops_dir / crop_filename
                    det_res.cropped_image.save(crop_path)
                    crop_path_str = str(crop_path.resolve())

                extracted_details.append({
                    "cropped_image_path": crop_path_str,
                    "detection_info": {
                        "source_page": det_res.source_page_index,
                        "class_name": det_res.class_name,
                        "confidence": round(det_res.confidence, 4),
                        "box_xyxy": det_res.box_xyxy,
                    },
                })

            doc_summary["extracted_images_details"] = extracted_details
            doc_summary["status"] = "Success"
        except Exception as e:
            logger.error("Failed to process %s: %s", doc_path.name, e, exc_info=True)
            doc_summary["status"] = "Failed"
            doc_summary["error_message"] = str(e)

        return doc_summary
