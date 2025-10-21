"""Public API exports for the Photo Extractor package.

Provides a simplified interface (PhotoExtractor) and direct access to detection
primitives (YOLODetector, DetectionResult) for advanced use cases.
"""
from __future__ import annotations

from .extractor import PhotoExtractor
from .detection import YOLODetector, DetectionResult

# Expose package version for CLI and consumers
__version__ = "0.1.0"

__all__ = ["PhotoExtractor", "YOLODetector", "DetectionResult", "__version__"]
