#!/usr/bin/env python3
"""Focused unit tests for core utilities (no external side effects). ✨

This script provides fast, deterministic unit-level tests for the most critical
utility helpers in the repository. It avoids network and database writes, uses
temporary directories/files, and demonstrates dependency injection for file-copy
operations. The goal is developer-speed feedback with professional clarity.

Tests covered:
- get_image_metadata: Missing and valid image files
- image_path_generator: Extension filtering
- save_similar_images_to_folder: Copier injection (no real IO)
- convert_numpy_to_native_python: JSON serializability

Excluded (by design):
- PostgreSQL interactions and full similarity workflow. These are integration
  concerns and should be validated in a dedicated environment-specific suite. 🛡️

Run:
  python -m hyundai_document_authenticator.tool_unit_test
  or
  python hyundai_document_authenticator/tool_unit_test.py

Exit code:
  0 if all tests pass; 1 otherwise.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path for absolute imports
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hyundai_document_authenticator.core_engine.image_similarity_system.utils import (  # noqa: E402
    convert_numpy_to_native_python,
    get_image_metadata,
    image_path_generator,
    save_similar_images_to_folder,
)

LOGGER = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger for test output.

    Args:
        level (str): Logging level name (e.g., "DEBUG", "INFO").
    """
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="[%(levelname)s] %(message)s")


def create_temp_image(
    path: Union[str, Path],
    size: Tuple[int, int] = (64, 48),
    color: Tuple[int, int, int] = (255, 0, 0),
) -> None:
    """Create a small RGB image for testing.

    Tries to use Pillow if available, otherwise falls back to OpenCV. This keeps
    the test resilient to varying environments while avoiding new dependencies.

    Args:
        path (Union[str, Path]): Target path for the image.
        size (Tuple[int, int]): (width, height) in pixels.
        color (Tuple[int, int, int]): RGB color tuple.
    """
    try:
        from PIL import Image  # type: ignore

        img = Image.new("RGB", size, color)
        img.save(str(path))
    except Exception:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        w, h = size
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        # Convert RGB to BGR for OpenCV
        arr[:, :] = (color[2], color[1], color[0])
        cv2.imwrite(str(path), arr)


def test_get_image_metadata_missing() -> bool:
    """Verify that missing image paths return safe defaults. 📐

    Returns:
        bool: True if expectations hold; False otherwise.
    """
    meta = get_image_metadata("this_file_does_not_exist.png")
    ok = (
        isinstance(meta, dict)
        and meta.get("dimensions_str") == "N/A"
        and meta.get("size_bytes") == -1
    )
    if not ok:
        LOGGER.error("Unexpected metadata for missing file: %s", meta)
    return ok


def test_get_image_metadata_valid() -> bool:
    """Verify that a valid image returns dimensions and size. 📐"""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        img = p / "img.png"
        create_temp_image(img, size=(80, 60))
        meta = get_image_metadata(str(img))
        dims = meta.get("dimensions_str")
        ok = (
            isinstance(meta, dict)
            and dims in ("80 x 60", "60 x 80")
            and isinstance(meta.get("size_bytes"), int)
            and meta.get("size_bytes", 0) > 0
        )
        if not ok:
            LOGGER.error("Unexpected metadata for valid file: %s", meta)
        return ok


def test_image_path_generator_filters_extensions() -> bool:
    """Ensure only allowed image extensions are yielded. 🖼️"""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        good1 = p / "a.jpg"
        good2 = p / "b.PNG"
        bad = p / "c.txt"
        create_temp_image(good1)
        create_temp_image(good2)
        bad.write_text("not an image", encoding="utf-8")

        found = list(image_path_generator(p, scan_subfolders=False))
        names = {f.name.lower() for f in found}
        ok = ("a.jpg" in names and "b.png" in names and "c.txt" not in names)
        if not ok:
            LOGGER.error("Unexpected generator results: %s", names)
        return ok


def test_save_similar_images_to_folder_copier_injection() -> bool:
    """Validate copier injection for save_similar_images_to_folder. 📦

    Uses a fake copier that records invocations but avoids real filesystem IO.

    Returns:
        bool: True if copier calls, outputs, and summary behave as expected.
    """
    copier_calls: List[Tuple[Path, Path]] = []

    def fake_copier(src: Union[str, Path], dst: Union[str, Path]) -> None:
        copier_calls.append((Path(src), Path(dst)))

    config: Dict[str, Any] = {
        "search_task": {
            "copy_query_image_to_output": True,
            "copy_similar_images_to_output": True,
            "save_search_summary_json": True,
            "top_k": 2,
            "save_query_in_separate_subfolder_if_copied": True,
        }
    }

    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        q = p / "query.png"
        s1 = p / "s1.png"
        s2 = p / "s2.png"
        out = p / "out"
        create_temp_image(q, (50, 40))
        create_temp_image(s1, (30, 30))
        create_temp_image(s2, (30, 30))

        results: List[Tuple[str, float]] = [(str(s1), 0.91), (str(s2), 0.89)]

        mapping, json_path, summary, missing = save_similar_images_to_folder(
            search_results=results,
            output_folder=out,
            query_image_path=q,
            config_for_summary=config,
            search_method_actually_used="unit-test",
            model_name_used="unit-model",
            total_search_time_seconds=0.01,
            json_filename="summary.json",
            copier=fake_copier,
        )

        conditions = [
            len(copier_calls) == 3,  # 1 query + 2 results
            json_path is not None and json_path.name == "summary.json",
            isinstance(summary, dict),
            missing == 0,
            any("query.png" in k for k in mapping.keys()),
            sum("s" in k for k in mapping.keys()) == 2,
        ]
        ok = all(conditions)
        if not ok:
            LOGGER.error(
                "Copier calls: %s, json_path=%s, missing=%s, mapping_keys=%s",
                copier_calls,
                json_path,
                missing,
                list(mapping.keys()),
            )
        return ok


def test_convert_numpy_to_native_python_json_serializable() -> bool:
    """Ensure converted payloads are JSON-serializable. 🔁"""
    import numpy as np  # local import to keep module load light

    payload: Dict[str, Any] = {
        "a": np.float32(1.5),
        "b": np.int32(7),
        "c": np.array([1, 2, 3], dtype=np.int16),
        "d": [np.float64(2.5), np.bool_(True)],
    }
    converted = convert_numpy_to_native_python(payload)
    try:
        json.dumps(converted)
        LOGGER.info("converted json:", converted)
    except TypeError as e:  # pragma: no cover - explicit log on failure
        LOGGER.error("JSON serialization failed: %s", e)
        return False

    conditions = [
        isinstance(converted["a"], float),
        isinstance(converted["b"], int),
        isinstance(converted["c"], list),
        isinstance(converted["d"][1], bool),
    ]
    ok = all(conditions)
    if not ok:
        LOGGER.error("Unexpected converted types: %s", {k: type(v) for k, v in converted.items()})
    return ok


def main() -> int:
    """Execute all unit tests and return an aggregated exit code.

    Returns:
        int: 0 if all tests pass; 1 otherwise.
    """
    setup_logging("INFO")
    LOGGER.info("CWD %s", Path.cwd())
    LOGGER.info("PY %s", sys.version.split()[0])

    ok = True
    ok &= test_get_image_metadata_missing()
    ok &= test_get_image_metadata_valid()
    ok &= test_image_path_generator_filters_extensions()
    ok &= test_save_similar_images_to_folder_copier_injection()
    ok &= test_convert_numpy_to_native_python_json_serializable()

    LOGGER.info("RESULT %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
