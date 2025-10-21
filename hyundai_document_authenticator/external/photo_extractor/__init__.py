"""
Wrapper to expose the vendored `photo_extractor` package.
"""
from .photo_extractor import PhotoExtractor, YOLODetector, DetectionResult

__all__ = ["PhotoExtractor", "YOLODetector", "DetectionResult"]