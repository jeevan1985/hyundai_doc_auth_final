# core_engine/image_similarity_system/constants.py
"""Core constants and default configuration values for the Image Similarity System. 🧭

This module defines canonical constants, default filenames/paths, logging levels,
model configuration descriptors, and search/index defaults used throughout the
image similarity system. It aims to be a single source of truth for defaults so
that behavior stays consistent across code paths and tools. ✨

Sphinx-ready documentation is provided using Google-style formatting, and
critical variables are annotated for clarity and type safety. Functionality is
unchanged.

References:
    - configs/image_similarity_config_TEMPLATE.yaml
    - configs/image_similarity_config.yaml

Note:
    - Model configuration values intentionally remain "loosely typed" (e.g.,
      Dict[str, Any]) to preserve flexibility and avoid introducing strong
      coupling to specific framework versions. A future improvement could
      introduce TypedDict or dataclasses for stricter schema without changing
      behavior. TODO: Evaluate adding a typed config layer if stability allows. 🧩
"""

from __future__ import annotations

# =============================================================================
# 1. Standard Library Imports
# =============================================================================
import logging
from typing import Any, Dict, List, Mapping, MutableMapping, Set

# =============================================================================
# 2. Third-Party Library Imports (for model definitions)
# =============================================================================
import torch
from torchvision import models as torchvision_models

# =============================================================================
# 3. Model Configurations
# =============================================================================
# This mapping is the single source of truth for supported models and how to
# extract features from them. Values include how to load the model and how to
# handle its outputs. The value structure is intentionally permissive (Dict[str, Any])
# to keep compatibility with existing patterns and to avoid behavior changes. ⚠️
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "convnext": {
        "type": "hf",
        "hf_default_id": "facebook/convnextv2-base-1k-224",
        "output_handler": lambda hf_model_outputs: hf_model_outputs.last_hidden_state.mean(
            dim=[2, 3]
        ),
        "input_size": (3, 224, 224),
    },
    "efficientnet": {
        "type": "torchvision",
        "torchvision_loader": torchvision_models.efficientnet_b0,
        "weights_enum": torchvision_models.EfficientNet_B0_Weights.DEFAULT,
        "feature_layer_extractor": lambda m: torch.nn.Sequential(*list(m.children())[:-1]),
        "output_handler": lambda tensor: torch.mean(tensor, dim=[2, 3]) if tensor.ndim == 4 else tensor,
        "input_size": (3, 224, 224),
    },
    "efficientnet_hf": {
        "type": "hf",
        "hf_default_id": "google/efficientnet-b0",
        "output_handler": lambda hf_model_outputs: hf_model_outputs.pooler_output,
        "input_size": (3, 224, 224),
    },
    "regnet": {
        "type": "hf",
        "hf_default_id": "facebook/regnet-y-16gf",
        "output_handler": lambda hf_model_outputs: hf_model_outputs.pooler_output,
        "input_size": (3, 224, 224),
    },
    "resnet": {
        "type": "torchvision",
        "torchvision_loader": torchvision_models.resnet50,
        "weights_enum": torchvision_models.ResNet50_Weights.DEFAULT,
        "feature_layer_extractor": lambda m: torch.nn.Sequential(*list(m.children())[:-1]),
        "output_handler": lambda tensor: torch.mean(tensor, dim=[2, 3]) if tensor.ndim == 4 else tensor,
        "input_size": (3, 224, 224),
    },
    "resnet_hf": {
        "type": "hf",
        "hf_default_id": "microsoft/resnet-50",
        "output_handler": lambda hf_model_outputs: hf_model_outputs.pooler_output,
        "input_size": (3, 224, 224),
    },
    "swin": {
        "type": "hf",
        "hf_default_id": "microsoft/swin-base-patch4-window7-224",
        "output_handler": lambda hf_model_outputs: hf_model_outputs.last_hidden_state.mean(dim=1),
        "input_size": (3, 224, 224),
    },
}

# =============================================================================
# 4. Application Settings & Default Names
# =============================================================================
# Filenames and folder names used consistently across the application.
DEFAULT_LOG_FOLDER: str = "logs"
DEFAULT_LOG_FILENAME: str = "image_similarity_system.log"
QUERY_SAVE_SUBFOLDER_NAME: str = "_query_image_source"
JSON_SUMMARY_DEFAULT_FILENAME: str = "search_results_summary.json"
DEFAULT_CONFIG_FILENAME: str = "image_similarity_config.yaml"

# Default columns to remove from results when not specified in YAML.
# For example: DEFAULT_REMOVE_COLUMNS_FROM_RESULTS = ['global_top_docs', 'fraud_doc_probability']
DEFAULT_REMOVE_COLUMNS_FROM_RESULTS: List[str] = []

# =============================================================================
# 5. Image Handling Constants
# =============================================================================
# Allowed image file extensions recognized by the system. Keeping a curated
# set avoids processing unintended files and reduces error handling overhead.
ALLOWED_IMAGE_EXTENSIONS: Set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".tiff",
    ".tif",
    ".webp",
}

# =============================================================================
# 6. FAISS Index Default Parameters
# =============================================================================
# Default values for FAISS index construction and search used as fallbacks when
# not explicitly provided by configuration. ⚙️

# For 'ivf' index type: Default number of clusters.
DEFAULT_IVF_NLIST: int = 100
# For 'ivf' index type: Default number of clusters to visit during a search.
DEFAULT_IVF_NPROBE: int = 10

# For 'hnsw' index type: Default number of connections per node during build.
DEFAULT_HNSW_M: int = 32
# For 'hnsw' index type: Build-time parameter affecting speed vs. quality.
DEFAULT_HNSW_EF_CONSTRUCTION: int = 40
# For 'hnsw' index type: Search-time parameter affecting speed vs. quality.
DEFAULT_HNSW_EF_SEARCH: int = 32

# =============================================================================
# 7. Search Operation Defaults
# =============================================================================
# General default parameters for search operations.
DEFAULT_TOP_K_SEARCH: int = 5

# Default threshold used for both per-query high-similarity counting and fraud
# probability decisions. This serves as a fallback for
# photo_authenticity_classifier_config.similar_doc_flag_threshold when not set
# in YAML.
SIMILAR_DOC_FLAG_THRESHOLD: float = 0.8

# =============================================================================
# 8. Logging Configuration
# =============================================================================
# Default logging level for the application. Can be overridden in the config.
DEFAULT_LOG_LEVEL: int = logging.INFO
