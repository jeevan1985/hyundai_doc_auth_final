# photo_extractor/cli.py
"""
Command-Line Interface (CLI) for the Photo Extractor application.

This module is the main entry point when the user runs the `photo-extractor` command.
Its responsibilities are:
- Parsing command-line arguments provided by the user.
- Setting up a robust logging system for the application run.
- Loading and validating the configuration.
- Instantiating the main `DocumentPipeline` class.
- Triggering the pipeline execution with the validated parameters.
- Handling top-level exceptions and providing clear exit statuses.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Import from within the package
from . import __version__
from .config import DEFAULT_CONFIG, load_config, deep_merge
from .pipeline import DocumentPipeline
# Shared log maintenance utility (defensive import path)
try:
    from core_engine.image_similarity_system.utils import maintain_log_files
except Exception:  # noqa: BLE001 - runtime safety
    maintain_log_files = None  # type: ignore

# --- Section 1: Logging Setup ---
def setup_logging(log_level: str, log_file: Path) -> None:
    """
    Configures a comprehensive logging setup for the application.

    This function sets up two log handlers:
    1. A StreamHandler to print logs to the console (stdout).
    2. A FileHandler to save all logs to a file for later review.

    It also quiets down overly verbose logs from third-party libraries.

    Args:
        log_level (str): The desired logging level (e.g., 'INFO', 'DEBUG').
        log_file (Path): The path to the file where logs should be saved.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Centralize logs under APP_LOG_DIR when available.
    env_log_dir = os.getenv("APP_LOG_DIR")
    if env_log_dir:
        try:
            central_dir = Path(env_log_dir) / "photo_extractor"
            central_dir.mkdir(parents=True, exist_ok=True)
            log_file = central_dir / log_file.name
        except Exception:
            # Fall back silently to provided log_file
            pass
    
    # Configure the root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)-20s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress excessive logs from noisy libraries to keep the output clean
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("ppocr").setLevel(logging.WARNING)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)


# --- Section 2: Main CLI Execution ---

# pylint: disable=too-many-statements
def main() -> None:
    """
    The main entry point for the DocProSight command-line interface.
    This function is called when the `docprosight` command is executed.
    """
    # --- Argument Parsing ---
    # Defines the commands, arguments, and options the user can provide.
    parser = argparse.ArgumentParser(
        prog="photo-extractor",
        description="Photo Extractor: Extract shop photos from TIF pages using a YOLO-based detector.",
        epilog="For more details and examples, see the project's README.md file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input",
        help="Path to the input document (e.g., doc.tif) or a folder of documents."
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Directory to save results. Overrides config.results.output_folder_path. "
            "If neither is provided, defaults to './photo_extractor_results'."
        ),
    )
    parser.add_argument(
        "--config",
        help=(
            "Path to a YAML configuration file. The final configuration is built as: "
            "DEFAULTS -> YAML overrides -> CLI overrides."
        )
    )
    # --- CLI overrides for config ---
    parser.add_argument("--log-level", dest="log_level", choices=["DEBUG","INFO","WARNING","ERROR"], help="Override project_meta.log_level")
    parser.add_argument("--dpi", dest="dpi", type=int, help="Override document_input.dpi (PDF rasterization)")
    parser.add_argument("--poppler-path", dest="poppler_path", help="Override document_input.poppler_path (Windows PDF support)")

    parser.add_argument("--model-path", dest="model_path", help="Override yolo_object_detection.model_path (.pt file)")
    parser.add_argument(
        "--confidence-threshold",
        dest="conf",
        type=float,
        help=(
            "Override yolo_object_detection.inference.confidence_threshold [0..1]"
        ),
    )
    parser.add_argument(
        "--iou-threshold",
        dest="iou",
        type=float,
        help=(
            "Override yolo_object_detection.inference.iou_threshold [0..1]"
        ),
    )
    parser.add_argument(
        "--target-object-names", dest="target_names", nargs="+",
        help="Override yolo_object_detection.inference.target_object_names (space-separated list)"
    )
    parser.add_argument("--imgsz", dest="imgsz", type=int, help="Override yolo_object_detection.inference.imgsz (e.g., 640)")

    parser.add_argument(
        "--save-cropped-images",
        dest="save_crops",
        type=lambda v: str(v).lower() in ["1", "true", "yes"],
        help="Override results.save_cropped_images (true/false)",
    )
    parser.add_argument("--summary-filename", dest="summary_filename", help="Override results.summary_filename")
    parser.add_argument(
        "--visualize",
        dest="visualize",
        type=lambda v: str(v).lower() in ["1", "true", "yes"],
        help="Override results.visualize (true/false)",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s v{__version__}"
    )

    args = parser.parse_args()

    # --- Initial Setup ---
    # Start logging to a temp default; will be reconfigured after config is resolved.
    temp_out = Path("./photo_extractor_results")
    temp_out.mkdir(parents=True, exist_ok=True)
    log_file = temp_out / "run.log"
    # Start with a default log level; this will be updated after config is loaded.
    setup_logging("INFO", log_file) 
    
    logger = logging.getLogger(__name__)
    logger.info(f"=================================================")
    logger.info(f" DocProSight v{__version__} | Starting New Analysis Run ")
    logger.info(f"=================================================")
    logger.info(f"Temporary output directory set to: {temp_out.resolve()}")

    try:
        # --- Configuration Loading ---
        # Load the default config and merge the user's config on top.
        if args.config:
            logger.info("Loading custom configuration from: %s", args.config)
            user_config = load_config(args.config)
            # Deep merge ensures nested dictionaries are updated correctly
            config = deep_merge(user_config, DEFAULT_CONFIG.copy())
        else:
            logger.warning("No '--config' file provided. Using default settings.")
            logger.error("A configuration file specifying 'yolo_object_detection.model_path' is mandatory for operation.")
            sys.exit(1)

        # Resolve effective output directory (CLI override > config > default)
        res = config.setdefault("results", {})
        effective_output = (
            args.output
            or res.get("output_folder_path")
            or "./photo_extractor_results"
        )
        output_dir = Path(effective_output)
        output_dir.mkdir(parents=True, exist_ok=True)
        res["output_folder_path"] = str(output_dir.resolve())

        # Re-setup logging based on the final, loaded configuration level
        log_file = output_dir / "run.log"
        setup_logging(config["project_meta"]["log_level"], log_file)

        # Housekeeping: maintain run logs in output_dir using module logging policy.
        try:
            if maintain_log_files is not None:
                maintain_log_files(
                    output_dir,
                    stem="run",
                    remove_logs_days=int(config.get("logging", {}).get("remove_logs_days", 7)),
                    backup_logs=bool(config.get("logging", {}).get("backup_logs", False)),
                )
        except Exception:
            # Never break CLI on housekeeping failures
            pass
        
        # --- Validation ---
        # Ensure the mandatory YOLO model path is present in the final config.
        if not config.get("yolo_object_detection", {}).get("model_path"):
            logger.critical("Configuration Error: The 'yolo_object_detection.model_path' must be specified in your config file.")
            sys.exit(1)
            
        # --- Pipeline Initialization and Execution ---
        # Instantiate the main pipeline class with the final configuration.
        logger.info("Initializing document processing pipeline...")
        pipeline = DocumentPipeline(config)
        
        # Trigger the main run method, which handles the entire workflow.
        pipeline.run(
            input_path=args.input,
            output_dir=str(output_dir)
        )

    except FileNotFoundError as err:
        logger.critical("File Not Found Error: %s", err)
        sys.exit(1)
    except ValueError as err:
        logger.critical("Value or Configuration Error: %s", err)
        sys.exit(1)
    except Exception as err:  # pylint: disable=broad-exception-caught
        logger.critical("An unexpected critical error occurred during execution: %s", err, exc_info=True)
        sys.exit(1)

# This standard Python construct ensures that the `main()` function is called
# only when this script is executed directly (e.g., via `python -m docprosight.cli`),
# not when it's imported as a module by another script.
if __name__ == '__main__':
    main()