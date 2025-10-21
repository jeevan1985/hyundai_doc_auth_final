"""CLI utility to test the ImageAuthenticityClassifier on a folder of images.

Loads optional environment variables from .env, initializes the classifier from
a YAML config, and prints per-image predictions and a summary.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

from PIL import Image
from dotenv import load_dotenv, find_dotenv

# Robust package-root resolution and environment loading
THIS_FILE: Path = Path(__file__).resolve()
# For: hyundai_document_authenticator/external/image_authenticity_classifier/test_classifier.py
# parents[0]=image_authenticity_classifier, [1]=external, [2]=hyundai_document_authenticator
PKG_ROOT: Path = THIS_FILE.parents[2]
REPO_ROOT: Path = PKG_ROOT.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

# Load .env and set working directory to package root so relative paths resolve
try:
    path_str = find_dotenv(filename=".env", usecwd=True)
    if path_str:
        load_dotenv(dotenv_path=path_str, override=False)
    else:
        candidate = REPO_ROOT / ".env"
        if candidate.exists():
            load_dotenv(dotenv_path=str(candidate), override=False)
except Exception:
    pass

try:
    os.chdir(str(PKG_ROOT))
except Exception:
    pass

from external.image_authenticity_classifier.classifier import ImageAuthenticityClassifier


def iter_images(folder: Path) -> List[Path]:
    """Return a list of image file paths in the given folder.

    Args:
        folder (Path): Folder to scan.

    Returns:
        List[Path]: Paths for files with common image extensions.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]


def main() -> None:
    """Parse CLI arguments, run inference on images in a folder, and print results.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Test ImageAuthenticityClassifier on a folder of images")
    parser.add_argument("--config", type=str, required=True, help="Path to classifier_config.yaml")
    parser.add_argument("--image-folder", type=str, required=True, help="Folder of images")
    args = parser.parse_args()

    clf = ImageAuthenticityClassifier(args.config)
    folder = Path(args.image_folder)
    images = iter_images(folder)
    if not images:
        print("No images found to test.")
        return

    preds = []
    for p in images:
        try:
            with Image.open(p) as img:
                r = clf.infer(img)
            print(f"{p.name}: class={r.get('class_name')}  score={r.get('score'):.4f}")
            preds.append((p.name, r))
        except Exception as e:
            print(f"{p.name}: error: {e}")

    # Simple summary
    counts = {}
    for _, r in preds:
        k = str(r.get("class_name"))
        counts[k] = counts.get(k, 0) + 1
    print("\nSummary counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    print("\nRaw JSON:")
    print(json.dumps({name: r for name, r in preds}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
