"""EfficientNet variant resolver.

This module provides a robust resolver for EfficientNet variants (B0–B7, V2-S/M/L)
so the pipeline can infer the correct torchvision loader, weights enum, and
input size based on a model identifier or local path. It also exposes a
feature-layer extractor that returns a 2D (B, D) tensor after global pooling.

The goal is to allow users to swap EfficientNet variants by changing only the
``pretrained_model_path_or_id`` field in the YAML config, while preserving the
current EfficientNet-B0 behavior as a safe fallback.

All public functions are documented with Google-style docstrings and include
precise type hints to support maintainability and static analysis.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torchvision import models as torchvision_models

logger = logging.getLogger(__name__)

# Type aliases
LoaderFn = Callable[..., torch.nn.Module]
ExtractorFn = Callable[[torch.nn.Module], torch.nn.Module]


def _make_global_pool_extractor() -> ExtractorFn:
    """Create a feature-layer extractor that returns pooled 2D outputs.

    Returns:
        ExtractorFn: A callable that, given a torchvision EfficientNet model,
            returns a module that excludes the classifier head and appends
            average pooling + flatten so the output tensor shape is (B, D).
    """
    def extractor(m: torch.nn.Module) -> torch.nn.Module:
        # Torchvision EfficientNet submodules order (v1/v2):
        #  - features (CNN trunk)
        #  - avgpool (sometimes named differently for v2, but present)
        #  - classifier (head)
        # We keep everything up to avgpool and add a Flatten, ensuring 2D output.
        children = list(m.children())
        # Remove classifier if present (last module is typically classifier)
        if len(children) >= 1 and hasattr(m, "classifier"):
            children = children[:-1]
        # Ensure global average pooling followed by flatten exists. Some
        # torchvision implementations already include avgpool before classifier,
        # but we add an extra check for safety.
        # If avgpool is already included (common case), the last non-head module
        # will output (B, C, 1, 1). Flatten will convert to (B, C).
        from torch import nn

        # If last child is not a pooling op, append AdaptiveAvgPool2d.
        if not children or not isinstance(children[-1], nn.AdaptiveAvgPool2d):
            children.append(nn.AdaptiveAvgPool2d((1, 1)))
        children.append(nn.Flatten(1))
        return nn.Sequential(*children)

    return extractor


def _variant_from_id(source: Optional[str]) -> str:
    """Infer EfficientNet variant from a model ID or file path.

    Args:
        source: The ``pretrained_model_path_or_id`` string from config. Can be
            a local file path, a local directory, a torchvision/huggingface
            identifier, or None.

    Returns:
        str: Canonical variant name (e.g., "efficientnet_b0", "efficientnet_b4",
        "efficientnet_v2_s", etc.). Defaults to "efficientnet_b0" if inference
        is not possible.
    """
    if not source:
        return "efficientnet_b0"

    s = Path(source).name.lower() if any(ch in source for ch in ("/", "\\")) else str(source).lower()

    # Try V2 first
    m = re.search(r"efficientnet[-_]?v2[-_]?([sml])", s)
    if m:
        return f"efficientnet_v2_{m.group(1)}"

    # Then V1 B0..B7
    m = re.search(r"efficientnet[-_]?b([0-7])", s)
    if m:
        return f"efficientnet_b{m.group(1)}"

    # HuggingFace style like google/efficientnet-b0
    m = re.search(r"efficientnet[-_]?([b][0-7])", s)
    if m:
        return f"efficientnet_{m.group(1)}"

    # Fallback to B0
    return "efficientnet_b0"


def resolve_efficientnet_variant(source: Optional[str]) -> Tuple[str, LoaderFn, Optional[Any], Tuple[int, int, int], ExtractorFn]:
    """Resolve an EfficientNet variant into loader, weights, input size, extractor.

    This function encapsulates all variant-specific details in one place.

    Args:
        source: The ``pretrained_model_path_or_id`` string used to infer the
            variant. May be None, in which case the resolver defaults to B0.

    Returns:
        Tuple[str, LoaderFn, Optional[Any], Tuple[int, int, int], ExtractorFn]:
            - variant name (canonical string)
            - torchvision loader function
            - default weights enum (or None if not available)
            - input size as (C, H, W)
            - feature-layer extractor that outputs pooled 2D (B, D)

    Notes:
        - If a particular weights enum is unavailable in the running
          torchvision version, the function returns None so the caller can fall
          back to local checkpoints.
        - Input sizes follow common EfficientNet defaults. If your weights were
          trained at a different resolution, rely on the model’s own
          transforms, or override upstream.
    """
    variant = _variant_from_id(source)

    # Canonical input sizes for EfficientNet variants.
    # Reference (typical defaults):
    #   B0:224, B1:240, B2:260, B3:300, B4:380, B5:456, B6:528, B7:600
    #   V2-S:384, V2-M:480, V2-L:512
    size_map: Dict[str, int] = {
        "efficientnet_b0": 224,
        "efficientnet_b1": 240,
        "efficientnet_b2": 260,
        "efficientnet_b3": 300,
        "efficientnet_b4": 380,
        "efficientnet_b5": 456,
        "efficientnet_b6": 528,
        "efficientnet_b7": 600,
        "efficientnet_v2_s": 384,
        "efficientnet_v2_m": 480,
        "efficientnet_v2_l": 512,
    }

    # Map to torchvision loader and weights enums (where available).
    loaders: Dict[str, LoaderFn] = {
        "efficientnet_b0": torchvision_models.efficientnet_b0,
        "efficientnet_b1": torchvision_models.efficientnet_b1,
        "efficientnet_b2": torchvision_models.efficientnet_b2,
        "efficientnet_b3": torchvision_models.efficientnet_b3,
        "efficientnet_b4": torchvision_models.efficientnet_b4,
        "efficientnet_b5": torchvision_models.efficientnet_b5,
        "efficientnet_b6": torchvision_models.efficientnet_b6,
        "efficientnet_b7": torchvision_models.efficientnet_b7,
        "efficientnet_v2_s": torchvision_models.efficientnet_v2_s,
        "efficientnet_v2_m": torchvision_models.efficientnet_v2_m,
        "efficientnet_v2_l": torchvision_models.efficientnet_v2_l,
    }

    weights_enums: Dict[str, Optional[Any]] = {}
    try:
        weights_enums = {
            "efficientnet_b0": torchvision_models.EfficientNet_B0_Weights.DEFAULT,
            "efficientnet_b1": torchvision_models.EfficientNet_B1_Weights.DEFAULT,
            "efficientnet_b2": torchvision_models.EfficientNet_B2_Weights.DEFAULT,
            "efficientnet_b3": torchvision_models.EfficientNet_B3_Weights.DEFAULT,
            "efficientnet_b4": torchvision_models.EfficientNet_B4_Weights.DEFAULT,
            "efficientnet_b5": torchvision_models.EfficientNet_B5_Weights.DEFAULT,
            "efficientnet_b6": torchvision_models.EfficientNet_B6_Weights.DEFAULT,
            "efficientnet_b7": torchvision_models.EfficientNet_B7_Weights.DEFAULT,
            "efficientnet_v2_s": torchvision_models.EfficientNet_V2_S_Weights.DEFAULT,
            "efficientnet_v2_m": torchvision_models.EfficientNet_V2_M_Weights.DEFAULT,
            "efficientnet_v2_l": torchvision_models.EfficientNet_V2_L_Weights.DEFAULT,
        }
    except Exception as exc:  # pragma: no cover - defensive for older torchvision
        logger.warning("Some EfficientNet weights enums are unavailable: %s", exc)

    variant = variant if variant in loaders else "efficientnet_b0"
    loader = loaders[variant]
    weights_enum = weights_enums.get(variant)
    input_size = (3, size_map.get(variant, 224), size_map.get(variant, 224))
    extractor = _make_global_pool_extractor()

    logger.info("Resolved EfficientNet variant: %s (input_size=%s)", variant, input_size)
    return variant, loader, weights_enum, input_size, extractor
