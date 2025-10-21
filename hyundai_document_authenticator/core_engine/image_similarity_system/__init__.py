"""core_engine.image_similarity_system

TIF document similarity focused package initializer.
Exposes only the components required for TIF workflows and their dependencies.
"""
import logging

# Prevent "No handler found" warnings if the application hasn't configured logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Public API: minimal components used by TIF workflows
from .feature_extractor import FeatureExtractor
from .faiss_manager import FaissIndexManager
from .searcher import ImageSimilaritySearcher
from .workflow import (
    build_index_from_tif_folder_workflow,
    execute_tif_batch_search_workflow,
)

__all__ = [
    # Core components
    "FeatureExtractor",
    "FaissIndexManager",
    "ImageSimilaritySearcher",
    # TIF workflows
    "build_index_from_tif_folder_workflow",
    "execute_tif_batch_search_workflow",
]

__version__ = "1.0.0"
