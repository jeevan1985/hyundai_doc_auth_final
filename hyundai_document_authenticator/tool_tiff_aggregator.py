"""Tool: TIFF Aggregator

A small, production-ready command-line utility to combine multiple single-page
TIFF files into a single multi-page TIFF. The tool supports combining files
that are located either directly in a parent folder or grouped per subfolder.

Usage examples:
- Combine all TIFFs directly under a folder (non-recursive), or if none exist at
  the parent level, combine per subfolder:

  python tool_tiff_aggregator.py \
      --tif-folder-path /path/to/parent_folder \
      --output-folder-path /path/to/output_folder

Notes:
- Multi-page TIFFs are created using Pillow's save_all/append_images.
- Images are normalized to a consistent mode for reliability; provide
  --target-mode to force a specific mode (e.g., "RGB").

This module is safe to import and provides a CLI entry point under
"if __name__ == '__main__'".
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from PIL import Image

# Global logger for this module
LOGGER = logging.getLogger(__name__)

# Supported TIFF file extensions
TIFF_EXTS: tuple[str, ...] = (".tif", ".tiff")


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with centralized APP_LOG_DIR/tools routing.

    When ``APP_LOG_DIR`` is set, also write to ``$APP_LOG_DIR/tools/tool_tiff_aggregator.log``
    in addition to console. On failure or when not set, fall back to console-only.

    Args:
        level: Logging level name (e.g., "DEBUG", "INFO").
    """
    root = logging.getLogger()
    root.handlers.clear()

    try:
        numeric_level = getattr(logging, level.upper())
    except AttributeError as exc:  # pragma: no cover
        raise ValueError(f"Invalid log level: {level}") from exc

    root.setLevel(numeric_level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Optional file handler
    try:
        import os
        from pathlib import Path
        env_log_dir = os.getenv("APP_LOG_DIR")
        if env_log_dir:
            tools_dir = Path(env_log_dir) / "tools"
            tools_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(tools_dir / "tool_tiff_aggregator.log"), encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
    except Exception:
        # Do not fail logging setup due to file handler issues
        pass


def _as_paths(items: Sequence[Union[str, Path]]) -> List[Path]:
    """Normalize a sequence of string/Path items to Path objects.

    Args:
        items (Sequence[Union[str, Path]]): A sequence of path-like values.

    Returns:
        List[Path]: Normalized Path objects.
    """
    return [Path(p) for p in items]


def _sorted_existing_tiffs(paths: Iterable[Path]) -> List[Path]:
    """Filter, normalize, and return sorted TIFF paths.

    Args:
        paths (Iterable[Path]): Candidate file paths.

    Returns:
        List[Path]: Sorted list of existing TIFF file paths.
    """
    tiffs = [p.resolve() for p in paths if p.suffix.lower() in TIFF_EXTS and p.exists()]
    return sorted(tiffs)


def combine_tiffs(
    tiffs: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    target_mode: Optional[str] = None,
) -> Optional[Path]:
    """Combine a list of TIFF files into a single multi-page TIFF.

    The function ensures a consistent image mode across pages. If ``target_mode``
    is not provided, the mode of the first image is used. All images are kept
    open until the combined TIFF is saved.

    Args:
        tiffs (Sequence[Union[str, Path]]): Paths to input TIFF files.
        output_path (Union[str, Path]): Destination file path for the multi-page TIFF.
        target_mode (Optional[str]): Optional Pillow image mode (e.g., "RGB", "L").
            If provided, all images are converted to this mode.

    Returns:
        Optional[Path]: The path to the created TIFF if successful; otherwise None
            when no input files are provided.

    Raises:
        FileNotFoundError: If any specified input path does not exist.
        OSError: On errors opening or saving image files.
        ValueError: If the ``tiffs`` sequence is empty.
    """
    tiff_paths = _as_paths(tiffs)
    if not tiff_paths:
        LOGGER.warning("No TIFF files provided to combine: %s", output_path)
        raise ValueError("No TIFF files provided to combine.")

    # Verify paths exist before opening
    missing = [str(p) for p in tiff_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Input TIFFs not found: {missing}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Sorting ensures page order is deterministic
    sorted_paths = sorted(tiff_paths)

    images: List[Image.Image] = []
    try:
        # Open first to determine mode
        first = Image.open(str(sorted_paths[0]))
        mode_to_use = target_mode or first.mode
        if first.mode != mode_to_use:
            first = first.convert(mode_to_use)
        images.append(first)

        # Remaining images
        for p in sorted_paths[1:]:
            img = Image.open(str(p))
            if img.mode != mode_to_use:
                img = img.convert(mode_to_use)
            images.append(img)

        # Save multi-page TIFF
        first_img, rest = images[0], images[1:]
        first_img.save(str(out), save_all=True, append_images=rest)
        LOGGER.info("Saved combined TIFF to %s (%d pages, mode=%s)", out, len(images), mode_to_use)
        return out
    finally:
        # Always close image handles to avoid resource leaks in large batches
        for img in images:
            try:
                img.close()
            except Exception:  # pragma: no cover - defensive cleanup
                pass


def discover_parent_tiffs(parent_dir: Union[str, Path]) -> List[Path]:
    """Return TIFF files directly under the given directory (non-recursive).

    Args:
        parent_dir (Union[str, Path]): Directory to scan.

    Returns:
        List[Path]: Sorted list of TIFF files directly within the directory.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    d = Path(parent_dir)
    if not d.exists():
        raise FileNotFoundError(f"Directory not found: {d}")
    if not d.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {d}")

    return _sorted_existing_tiffs(list(d.glob("*.tif")) + list(d.glob("*.tiff")))


def combine_per_directory(
    tif_folder_path: Union[str, Path],
    output_folder_path: Union[str, Path],
    target_mode: Optional[str] = None,
) -> int:
    """Combine TIFFs either at the parent folder level or per subfolder.

    If the parent folder contains TIFF files directly under it, this function
    combines them into a single multi-page TIFF named after the parent folder.
    Otherwise, it scans immediate subfolders and combines TIFFs per subfolder.

    Args:
        tif_folder_path (Union[str, Path]): Path to the parent folder containing
            TIFF files or subfolders of TIFF files.
        output_folder_path (Union[str, Path]): Destination folder where combined
            TIFFs are written.
        target_mode (Optional[str]): Optional Pillow image mode to standardize
            all pages (e.g., "RGB").

    Returns:
        int: Process exit code where 0 indicates success and 1 indicates that
            nothing was processed (e.g., no TIFFs found).

    Raises:
        FileNotFoundError: If the parent folder does not exist.
        NotADirectoryError: If the provided path is not a directory.
        OSError: For file I/O errors during image processing or saving.
    """
    output_dir = Path(output_folder_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    parent_dir = Path(tif_folder_path)
    if not parent_dir.exists() or not parent_dir.is_dir():
        raise NotADirectoryError(
            f"Provided TIFF folder path '{parent_dir}' does not exist or is not a directory."
        )

    # First try: combine files directly under parent
    parent_tiffs = discover_parent_tiffs(parent_dir)
    if parent_tiffs:
        output_file = output_dir / f"{parent_dir.name}.tif"
        LOGGER.info("Combining %d TIFF files directly under parent folder…", len(parent_tiffs))
        combine_tiffs(parent_tiffs, output_file, target_mode=target_mode)
        return 0

    # Otherwise, combine per child directory
    subfolders = [d for d in parent_dir.iterdir() if d.is_dir()]
    if not subfolders:
        LOGGER.warning("No TIFF files found in parent folder or subfolders: %s", parent_dir)
        return 1

    processed_any = False
    for sub in sorted(subfolders):
        sub_tiffs = discover_parent_tiffs(sub)
        if sub_tiffs:
            processed_any = True
            output_file = output_dir / f"{sub.name}.tif"
            LOGGER.info("Combining %d TIFF files in subfolder: %s", len(sub_tiffs), sub.name)
            combine_tiffs(sub_tiffs, output_file, target_mode=target_mode)
        else:
            LOGGER.info("No TIFF files found in subfolder: %s", sub.name)

    if not processed_any:
        LOGGER.warning("No TIFF files found in any subfolder under: %s", parent_dir)
        return 1

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the TIFF aggregator tool.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Combine multiple TIFF files in a folder or its subfolders into multi-"
            "page TIFFs. If the parent folder contains TIFFs directly, they are"
            " combined into a single file named after the parent folder; otherwise,"
            " each subfolder is combined into its own multi-page TIFF."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--tif-folder-path",
        type=Path,
        required=True,
        help="Path to the parent folder containing TIFF files or subfolders with TIFFs.",
    )
    parser.add_argument(
        "--output-folder-path",
        type=Path,
        required=True,
        help="Path where combined TIFF files will be saved.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    parser.add_argument(
        "--target-mode",
        type=str,
        default=None,
        help=(
            "Optional Pillow image mode to convert all pages to (e.g., 'RGB'). "
            "If omitted, the mode of the first image is used."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """CLI entry point for the TIFF aggregator tool.

    Returns:
        int: Process exit code, where 0 is success and non-zero indicates an
            error or no-op (no TIFF files found).
    """
    args = parse_args()
    setup_logging(args.log_level)

    try:
        return combine_per_directory(
            tif_folder_path=args.tif_folder_path,
            output_folder_path=args.output_folder_path,
            target_mode=args.target_mode,
        )
    except Exception as exc:
        # Provide a single, clear log entry for unexpected failures
        LOGGER.exception("Failed to combine TIFFs: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
