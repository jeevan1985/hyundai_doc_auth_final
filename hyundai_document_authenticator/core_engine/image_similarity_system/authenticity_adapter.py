"""
Authenticity classifier adapter with a stable interface and graceful failure.

This module provides a thin adapter around the optional external
`external.image_authenticity_classifier.classifier.ImageAuthenticityClassifier`
so callers do not have to manage import errors or initialization failures.

Behavior parity
- When the classifier is disabled in config or unavailable/broken at runtime,
  callers treat the classifier as absent and skip authenticity checks. Outputs
  remain identical to previous behavior (image_authenticity={}, fraud logic only
  considers similarity evidence).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

logger = logging.getLogger(__name__)


class AuthenticityClassifierInterface:
    """Stable, minimal interface for image authenticity classifiers.

    Implementations should return a mapping that includes a 'class_name' key.
    The concrete adapter mirrors the external classifier's output.
    """

    def infer(self, image: Image.Image) -> Dict[str, Any]:  # pragma: no cover - interface
        """Infer authenticity class for a PIL image.

        Args:
            image: Pillow Image in RGB mode.

        Returns:
            Dict[str, Any]: A dictionary with at least {'class_name': str}.
        """
        raise NotImplementedError


class ExternalImageAuthenticityClassifierAdapter(AuthenticityClassifierInterface):
    """Adapter over the optional external authenticity classifier.

    The external classifier is imported lazily to avoid hard dependency. All
    failures are handled gracefully with clear logs.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize the external classifier if available.

        Args:
            config_path: Optional path string to the external classifier config.
        """
        try:
            # Local import to avoid hard dependency when feature is disabled
            from external.image_authenticity_classifier.classifier import ImageAuthenticityClassifier  # type: ignore
        except Exception as e_imp:  # pragma: no cover - environment dependent
            logger.warning(
                "Photo authenticity classifier import failed: %s. Feature will be disabled.",
                e_imp,
            )
            raise
        # Instantiate
        self._inner = ImageAuthenticityClassifier(config_path)

    def infer(self, image: Image.Image) -> Dict[str, Any]:
        """Run inference using the underlying external classifier.

        Exceptions are propagated to allow callers to decide fallback behavior,
        matching the current code path that sets class to 'unknown' on errors.

        Args:
            image: Pillow Image.
        Returns:
            A dict with at least a 'class_name' field.
        """
        return self._inner.infer(image)  # type: ignore[attr-defined]


def try_init_classifier(config_path: Optional[str]) -> Optional[AuthenticityClassifierInterface]:
    """Try to create the authenticity classifier adapter.

    Import and initialization failures are logged as warnings and result in a
    None return value, signaling callers to skip authenticity checks.

    Args:
        config_path: Optional string path to the external classifier config.

    Returns:
        An instance of AuthenticityClassifierInterface if successful, otherwise None.
    """
    try:
        # We pass through the path string unchanged; callers should resolve paths
        # before invoking this function if necessary.
        adapter = ExternalImageAuthenticityClassifierAdapter(config_path)
        logger.info("Photo authenticity classifier initialized (adapter): %s", config_path)
        return adapter
    except Exception as e:
        logger.warning(
            "Photo authenticity classifier unavailable or failed to initialize: %s. Continuing without authenticity checks.",
            e,
        )
        return None
