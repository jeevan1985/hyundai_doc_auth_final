"""Public API exports for the Image Authenticity Classifier package.

This subpackage exposes ImageAuthenticityClassifier as the public entry point.
The implementation lives in classifier.py. Import paths and behavior remain unchanged.
"""
from __future__ import annotations

from .classifier import ImageAuthenticityClassifier

__all__ = ["ImageAuthenticityClassifier"]
