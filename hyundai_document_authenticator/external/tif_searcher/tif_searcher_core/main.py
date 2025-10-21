"""CLI entrypoint for the TIF text search workflow.

Runs OCR over images discovered under the configured input path, searches for
configured text with optional normalization, copies matched images to an output
folder, and optionally writes a CSV report.

This module is a CLI entrypoint; it intentionally uses print statements for
user-facing messages. Logging is available via the module-level logger for
programmatic consumption and debugging.
"""

from __future__ import annotations

import os
import logging
from typing import Any, Mapping, List, Dict
import shutil
from datetime import datetime
from tif_searcher.config import config
from tif_searcher.utils import get_image_paths, normalize_text, write_results_to_csv
from tif_searcher.ocr_engines import get_ocr_engine

logger = logging.getLogger(__name__)

def run_text_search_workflow(ocr_engine: Any, config: Mapping[str, Any]) -> None:
    """Run the text search workflow over discovered images.

    Discovers images, performs OCR using the provided engine, optionally
    normalizes text before matching, copies matched images to the output
    folder, and optionally writes a CSV report. This function is designed for
    CLI usage and prints user-facing progress and results.

    Args:
        ocr_engine (Any): OCR engine with an extract_text(image) -> str method.
        config (Mapping[str, Any]): Workflow configuration dictionary.

    Returns:
        None

    Raises:
        Exception: Unexpected errors from OCR engine or IO may propagate at
            runtime; per-file errors are caught and recorded to CSV when enabled.
    """
    print("\n==================================================")
    print("===   Starting Image Text Search Workflow    ===")
    print("==================================================\n")
    
    # --- STAGE 1: File Discovery and Preparation ---
    print("--- STAGE 1: Discovering and Preparing Files ---")
    image_paths = get_image_paths(config['input_path'], config['supported_formats'])
    if not image_paths:
        print("Workflow concluded: No images found to process.")
        return
    os.makedirs(config['output_folder'], exist_ok=True)
    print(f"✅ Output folder for matched images is ready at: {config['output_folder']}")

    # --- STAGE 2: Image Processing Loop ---
    print("\n--- STAGE 2: Processing Images and Searching for Text ---")
    found_count: int = 0
    csv_results: List[Dict[str, Any]] = []
    csv_header = ['timestamp', 'image_name', 'image_folder_path', 'status', 'search_text_found', 'ocr_engine_used', 'search_text_queried', 'extracted_text_snippet']

    for i, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        folder_path = os.path.dirname(img_path)
        row_data: Dict[str, Any] = {}
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {img_path}")
        try:
            extracted_text = ocr_engine.extract_text(img_path)
            
            if config.get('recognized_text_debug', False):
                print(f"  [DEBUG] Raw Extracted Text: \"{extracted_text}\"")

            # ==================================================================
            # === (THE FIX) CONDITIONAL NORMALIZATION LOGIC ====================
            # ==================================================================
            # Check the config to decide which comparison method to use.
            if config.get('allow_recognition_normalization', True):
                # Normalize both the search term and the OCR output.
                search_term = normalize_text(config['search_text'])
                text_to_search_in = normalize_text(extracted_text)
                print("  [INFO] Using normalized text for comparison.")
            else:
                # Use the original, raw text for a strict comparison.
                search_term = config['search_text']
                text_to_search_in = extracted_text
                print("  [INFO] Using strict, literal text for comparison.")
            
            # Now, perform the search using the chosen strings.
            is_found = search_term in text_to_search_in
            # ==================================================================
            
            if is_found:
                print(f"  ✔️ SUCCESS: Found '{config['search_text']}'.")

                # --- (THE FIX) CREATE A UNIQUE FILENAME TO PREVENT OVERWRITES ---
                output_filename = filename # Default to the original name

                # Only create a complex name if the input was a directory search
                if os.path.isdir(config['input_path']):
                    # Get the path of the image relative to the starting folder
                    relative_path = os.path.relpath(img_path, config['input_path'])
                    # Replace OS-specific path separators (\\ or /) with an underscore
                    output_filename = relative_path.replace(os.sep, '_')

                # Construct the full, safe path for the copy destination
                destination_path = os.path.join(config['output_folder'], output_filename)

                shutil.copy(img_path, destination_path)
                print(f"  ↪️  Image copied to '{destination_path}'.") # More descriptive log
                found_count += 1
            else:
                print(f"  ➖ Text not found.")
            
            row_data = {
                'status': 'Processed',
                'search_text_found': is_found,
                'extracted_text_snippet': extracted_text # Always log the raw text
            }
        except Exception as e:
            error_message = f"Error: {e}"
            print(f"  ❌ ERROR: Could not process file. Reason: {e}")
            row_data = {
                'status': 'Error',
                'search_text_found': False,
                'extracted_text_snippet': error_message
            }
        finally:
            if config.get('create_csv_report', False):
                base_data = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'image_name': filename,
                    'image_folder_path': folder_path,
                    'ocr_engine_used': config['ocr_engine'],
                    'search_text_queried': config['search_text']
                }
                row_data.update(base_data)
                csv_results.append(row_data)
    
    # --- STAGE 3: Final Reporting ---
    if config.get('create_csv_report', False) and csv_results:
        write_results_to_csv(config['csv_output_path'], csv_results, csv_header)
        
    print("\n==================================================")
    print("===             Workflow Complete            ===")
    print("==================================================\n")
    print(f"📊 Total images scanned: {len(image_paths)}")
    print(f"🎯 Images containing the text: {found_count}")
    print(f"📂 All found images are saved in: {config['output_folder']}")
    if config.get('create_csv_report', False):
        print(f"📋 A detailed report has been saved to: {config['csv_output_path']}")

if __name__ == '__main__':
    print("--- Initializing Global OCR Engine (this may take a moment)... ---")
    ocr_engine = get_ocr_engine(config)
    if ocr_engine:
        print("✅ Global OCR Engine initialized successfully and is ready.")
        run_text_search_workflow(ocr_engine, config)
