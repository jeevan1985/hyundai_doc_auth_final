
# -* - coding: utf-8 -*-
"""
==============================================================================
Universal Model Downloader 📥
==============================================================================
Purpose:
--------
A production-ready, command-line tool to download and save deep learning models
from both TorchVision and the Hugging Face Hub. This script ensures that models
are correctly saved for offline use in the application.

Features:
---------
- 📥 Supports both TorchVision and Hugging Face models.
- 📦 Saves models in the correct format (state_dict for TorchVision, full directory for Hugging Face).
- ⚙️ Simple command-line interface.
- 📝 Richly commented and documented for clarity.
- ✨ Includes error handling and user-friendly logging.

Usage:
------
For TorchVision Models:
    python universal_model_downloader.py torchvision --model_name resnet50
    python universal_model_downloader.py torchvision --model_name efficientnet_b0

For Hugging Face Models:
    python universal_model_downloader.py hf --model_id microsoft/swin-base-patch4-window7-224
    python universal_model_downloader.py hf --model_id facebook/convnextv2-base-1k-224
"""

# ==============================================================================
# 1. Standard Library Imports
# ==============================================================================
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# ==============================================================================
# 2. Third-Party Library Imports
# ==============================================================================
import torch
import torchvision.models as torchvision_models
from transformers import AutoImageProcessor, AutoModel

# ==============================================================================
# 3. Logger Configuration
# ==============================================================================
# Configure a logger for clear and informative output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 4. TorchVision Model Definitions
# ==============================================================================
# A mapping of model names to their corresponding TorchVision loader and weights.
# This makes it easy to add new TorchVision models in the future.
TORCHVISION_MODELS: Dict[str, Dict[str, Any]] = {
    "resnet50": {
        "loader": torchvision_models.resnet50,
        "weights": torchvision_models.ResNet50_Weights.DEFAULT,
        "output_dir": "resnet",
        "output_filename": "resnet50-0676ba61.pth" # Official hash for ImageNet weights
    },
    "efficientnet_b0": {
        "loader": torchvision_models.efficientnet_b0,
        "weights": torchvision_models.EfficientNet_B0_Weights.DEFAULT,
        "output_dir": "efficientnet",
        "output_filename": "efficientnet_b0_weights.pth"
    },
    # Add other torchvision models here if needed, e.g.:
    # "efficientnet_b7": {
    #     "loader": torchvision_models.efficientnet_b7,
    #     "weights": torchvision_models.EfficientNet_B7_Weights.DEFAULT,
    #     "output_dir": "efficientnet",
    #     "output_filename": "efficientnet_b7_weights.pth"
    # },
}

# ==============================================================================
# 5. Core Functions
# ==============================================================================

def download_torchvision_model(model_name: str, output_path_override: Optional[Path] = None) -> None:
    """
    Downloads and saves a model from TorchVision.

    This function fetches a pre-trained model from TorchVision, extracts its
    state dictionary (the weights), and saves it to a local .pth file.

    Args:
        model_name (str): The name of the model to download (must be a key in
                          TORCHVISION_MODELS).
        output_path_override (Optional[Path]): If provided, saves the model to this
                                                exact path. Otherwise, uses the default.

    Raises:
        ValueError: If the requested model_name is not defined in TORCHVISION_MODELS.
        RuntimeError: If the download or save operation fails.
    """
    logger.info(f"🔍 Searching for TorchVision model: '{model_name}'")

    # Validate if the requested model is in our supported list.
    if model_name not in TORCHVISION_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not supported for TorchVision download. "
            f"Available models: {list(TORCHVISION_MODELS.keys())}"
        )

    # Determine the output path.
    if output_path_override:
        output_path = output_path_override
        output_dir = output_path.parent
    else:
        config = TORCHVISION_MODELS[model_name]
        output_dir = Path(config["output_dir"])
        output_path = output_dir / config["output_filename"]

    # Create the output directory if it doesn't exist.
    logger.info(f"📁 Ensuring directory exists: '{output_dir}'")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if the model already exists to avoid re-downloading.
    if output_path.exists():
        logger.info(f"✅ Model already exists at '{output_path}'. Skipping download.")
        return

    try:
        # Inform the user that the download is starting.
        logger.info(f"⬇️  Downloading '{model_name}' from TorchVision. This may take a moment...")

        # Load the pre-trained model from TorchVision (this triggers the download).
        config = TORCHVISION_MODELS[model_name]
        model = config["loader"](weights=config["weights"])

        # Save the model's state dictionary (the learned weights).
        logger.info(f"💾 Saving model weights to '{output_path}'...")
        torch.save(model.state_dict(), output_path)

        # Confirm successful download and save.
        logger.info(f"🎉 Successfully downloaded and saved '{model_name}'!")

    except Exception as e:
        # Catch any exceptions during download/save and report them.
        logger.error(f"❌ Failed to download or save '{model_name}'. Error: {e}", exc_info=True)
        raise RuntimeError(f"Operation failed for model '{model_name}'.") from e


def download_huggingface_model(model_id: str, output_path_override: Optional[Path] = None) -> None:
    """
    Downloads and saves a model and its processor from the Hugging Face Hub.

    This function downloads all necessary files for a model (weights, config, etc.)
    and its associated image processor from the Hugging Face Hub and saves them
    to a local directory.

    Args:
        model_id (str): The Hugging Face Hub ID of the model to download
                        (e.g., 'microsoft/swin-base-patch4-window7-224').
        output_path_override (Optional[Path]): If provided, saves the model to this
                                                directory. Otherwise, uses the model_id.

    Raises:
        RuntimeError: If the download or save operation fails.
    """
    # Determine the output directory.
    output_dir = output_path_override if output_path_override else Path(model_id)

    logger.info(f"🔍 Searching for Hugging Face model: '{model_id}'")

    # Create the output directory if it doesn't exist.
    logger.info(f"📁 Ensuring directory exists: '{output_dir}'")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # --- Download and save the Image Processor ---
        logger.info(f"⬇️  Downloading image processor for '{model_id}'...")
        processor = AutoImageProcessor.from_pretrained(model_id)
        processor.save_pretrained(output_dir)
        logger.info(f"✅ Saved image processor to '{output_dir}'.")

        # --- Download and save the Model ---
        logger.info(f"⬇️  Downloading model weights for '{model_id}'. This may be large...")
        model = AutoModel.from_pretrained(model_id)
        model.save_pretrained(output_dir)
        logger.info(f"✅ Saved model to '{output_dir}'.")

        # Confirm successful download and save.
        logger.info(f"🎉 Successfully downloaded and saved all components for '{model_id}'!")

    except Exception as e:
        # Catch any exceptions during download/save and report them.
        logger.error(f"❌ Failed to download or save '{model_id}'. Error: {e}", exc_info=True)
        raise RuntimeError(f"Operation failed for model '{model_id}'.") from e

# ==============================================================================
# 6. Command-Line Interface (CLI)
# ==============================================================================
def main():
    """
    Main function to parse command-line arguments and trigger the download.
    """
    # Create the top-level parser.
    parser = argparse.ArgumentParser(
        description="""
    Universal Model Downloader 📥
    A tool to download models from TorchVision or Hugging Face for offline use.
    """,
        formatter_class=argparse.RawTextHelpFormatter # Preserves formatting in help text.
    )

    # Create subparsers for the two main commands: 'torchvision' and 'hf'.
    subparsers = parser.add_subparsers(dest="source", required=True, help="The source to download from.")

    # --- Parser for TorchVision models ---
    parser_tv = subparsers.add_parser(
        "torchvision",
        help="Download a model from TorchVision.",
        description="Downloads a model from TorchVision and saves its weights as a .pth file."
    )
    parser_tv.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=list(TORCHVISION_MODELS.keys()),
        help="The name of the TorchVision model to download."
    )
    parser_tv.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="(Optional) The full path (including filename) to save the model file. Overrides the default."
    )

    # --- Parser for Hugging Face models ---
    parser_hf = subparsers.add_parser(
        "hf",
        help="Download a model from the Hugging Face Hub.",
        description="Downloads a model and its preprocessor from the Hugging Face Hub."
    )
    parser_hf.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="The Hugging Face Hub ID of the model (e.g., 'google/efficientnet-b0')."
    )
    parser_hf.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="(Optional) The directory to save the model files. Overrides the default (which is the model_id)."
    )

    # Parse the arguments provided by the user.
    args = parser.parse_args()

    # --- Execute the appropriate download function based on the sub-command ---
    try:
        if args.source == "torchvision":
            download_torchvision_model(model_name=args.model_name, output_path_override=args.output_path)
        elif args.source == "hf":
            download_huggingface_model(model_id=args.model_id, output_path_override=args.output_path)
    except (ValueError, RuntimeError) as e:
        # Log critical errors and exit with a non-zero status code to indicate failure.
        logger.critical(f"A critical error occurred: {e}")
        exit(1)

# ==============================================================================
# 7. Script Execution Guard
# ==============================================================================
if __name__ == "__main__":
    # This ensures the main() function is called only when the script is executed directly.
    main()
    
    
    
"""
# For EfficientNet-B0
python universal_model_downloader.py torchvision --model_name efficientnet_b0

# For ResNet-50
python universal_model_downloader.py torchvision --model_name resnet50

# For Swin Transformer
python universal_model_downloader.py hf --model_id microsoft/swin-base-patch4-window7-224

# For ConvNeXt
python universal_model_downloader.py hf --model_id facebook/convnextv2-base-1k-224
    
"""