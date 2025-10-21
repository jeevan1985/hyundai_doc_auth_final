# core_engine/image_similarity_system/searcher.py
"""
This module contains the ImageSimilaritySearcher class, which orchestrates
the process of finding similar images. It uses a FeatureExtractor to process
query images and a vector database manager (like FaissIndexManager) to
perform the actual search. It also handles saving results.
"""
# =============================================================================
# 1. Standard Library Imports
# =============================================================================
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union, Any
import tqdm

# =============================================================================
# 2. Third-Party Library Imports
# =============================================================================
import numpy as np

# =============================================================================
# 3. Application-Specific Imports
# =============================================================================
from .feature_extractor import FeatureExtractor
from .utils import image_path_generator

# =============================================================================
# 4. Module-level Logger Setup
# =============================================================================
logger = logging.getLogger(__name__)

# =============================================================================
# 5. Helper Function for JSON Serialization
# =============================================================================
def convert_numpy_to_native_python(data: Any) -> Any:
    """
    Recursively traverses a data structure and converts NumPy types to native
    Python types to ensure JSON serializability.
    """
    if isinstance(data, list):
        return [convert_numpy_to_native_python(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_numpy_to_native_python(value) for key, value in data.items()}
    elif isinstance(data, (np.float16, np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    return data

# =============================================================================
# 6. ImageSimilaritySearcher Class Definition
# =============================================================================
class ImageSimilaritySearcher:
    """
    Orchestrates similarity searches using a feature extractor and a vector database manager.

    This searcher supports two explicit fallback modes when the vector database
    is unavailable or not ready:
    - 'bruteforce': perform a brute-force scan of a provided database folder
    - 'transient': return empty results; higher-level orchestration may merge
      transient indices or handle results differently.
    """
    def __init__(self,
                 feature_extractor: FeatureExtractor,
                 vector_db_manager: Optional[Any] = None,
                 fallback_mode: Optional[str] = None) -> None:
        """
        Initializes the ImageSimilaritySearcher.

        Args:
            feature_extractor (FeatureExtractor): An initialized feature extractor instance.
            vector_db_manager (Optional[Any]): An initialized vector database manager
                (e.g., FaissIndexManager or QdrantManager). This manager should be
                pre-loaded with an index before being passed here for searching.
            fallback_mode (Optional[str]): Controls behavior when the vector DB is not
                ready. Allowed values: 'bruteforce', 'transient', or None. When None,
                the legacy behavior applies: attempt brute-force if a folder is provided.

        Raises:
            ValueError: If fallback_mode is provided and not one of the supported values.
        """
        self.feature_extractor = feature_extractor
        self.vector_db_manager = vector_db_manager
        self.device = feature_extractor.device
        self.last_query_feature_extraction_time: Optional[float] = None
        self.fallback_mode: Optional[str] = (fallback_mode.lower() if isinstance(fallback_mode, str) else None)

        if self.fallback_mode not in {None, 'bruteforce', 'transient'}:
            raise ValueError("fallback_mode must be one of {'bruteforce','transient',None}")
        
        logger.debug("ImageSimilaritySearcher initialized.")
        if self.vector_db_manager:
            manager_type = type(self.vector_db_manager).__name__
            is_ready = self.vector_db_manager.is_index_loaded_and_ready()
            total_items = self.vector_db_manager.get_total_indexed_items()
            logger.info("  Vector DB Manager (%s) provided. Status: %s (Total items: %d).",
                        manager_type, "Ready" if is_ready else "Not Ready", total_items)
        else:
            logger.warning("  No Vector DB Manager provided. Only brute-force search is available when fallback_mode='bruteforce'.")

    def search_similar_images(self,
                              query_image_path: Union[str, Path],
                              top_k: int = 5,
                              db_folder_for_bruteforce: Optional[Path] = None,
                              bruteforce_batch_size: int = 32,
                              **kwargs: Any
                             ) -> Tuple[List[Tuple[str, float]], str, str, float]:
        """
        Finds top-k similar images for a given query image.

        This method first attempts to use the provided vector database manager. If the
        manager is not available or not ready, it will fall back to performing a
        brute-force search against the `db_folder_for_bruteforce` if provided.

        Args:
            query_image_path (Union[str, Path]): Path to the query image.
            top_k (int): Number of similar images to retrieve.
            db_folder_for_bruteforce (Optional[Path]): Folder to scan for images for
                brute-force search. This is a fallback if the vector DB is not ready.
            bruteforce_batch_size (int): Batch size for feature extraction during brute-force search.
            **kwargs (Any): Additional search-time parameters for the vector DB manager
                            (e.g., `ivf_nprobe_search` for FAISS).

        Returns:
            Tuple[List[Tuple[str, float]], str, str, float]: A tuple containing:
                - A list of (image_path, similarity_score) tuples.
                - A string indicating the search method used (e.g., "Faiss", "Qdrant", "Brute-Force").
                - A string with the model name used for feature extraction.
                - A float with the duration of the entire search operation in seconds.
        """
        query_path_obj = Path(query_image_path)
        if not query_path_obj.is_file():
            raise FileNotFoundError(f"Query image file not found: '{query_path_obj.resolve()}'")
        
        start_time = time.time()
        
        # Step 1: Extract features from the query image
        extraction_start_time = time.time()
        query_features_normalized = self.feature_extractor.extract_features(query_path_obj)
        self.last_query_feature_extraction_time = time.time() - extraction_start_time
        logger.debug("Query feature extraction took %.4f seconds.", self.last_query_feature_extraction_time)
        
        results: List[Tuple[str, float]] = []
        search_method_used = "Unspecified"
        
        # Step 2: Perform the search
        if self.vector_db_manager and self.vector_db_manager.is_index_loaded_and_ready():
            logger.info("Attempting search using the configured vector database.")
            search_method_used = type(self.vector_db_manager).__name__.replace("Manager", "")
            
            k_for_search = top_k + 1
            
            search_results_raw = self.vector_db_manager.search_similar_images(
                query_vector=query_features_normalized,
                top_k=k_for_search,
                **kwargs
            )
            
            resolved_query_path_str = str(query_path_obj.resolve())
            for path, score in search_results_raw:
                if str(Path(path).resolve()) != resolved_query_path_str:
                    results.append((path, score))
                if len(results) >= top_k:
                    break
            
            logger.info("Vector DB search found %d similar images (after self-filtering).", len(results))
        else:
            # Vector DB is not ready or not provided. Use explicit fallback behavior.
            if self.fallback_mode == 'transient':
                logger.warning("Vector DB not ready; fallback_mode='transient'. Returning empty results.")
                search_method_used = "Transient"
                results = []
            else:
                # Default to brute-force when fallback_mode is 'bruteforce' or None (legacy behavior)
                if self.fallback_mode not in {None, 'bruteforce'}:
                    logger.error("Unsupported fallback_mode '%s'. Defaulting to 'transient'.", self.fallback_mode)
                    search_method_used = "Transient"
                    results = []
                else:
                    logger.warning("Vector DB is not ready or not provided. Falling back to brute-force search.")
                    search_method_used = "Brute-Force"
                    if db_folder_for_bruteforce:
                        all_db_paths_gen = image_path_generator(db_folder_for_bruteforce)
                        db_paths_to_process = [p for p in all_db_paths_gen if p.resolve() != query_path_obj.resolve()]
                        
                        if not db_paths_to_process:
                            logger.warning("No images found in the specified folder for brute-force comparison.")
                        else:
                            logger.info("Performing brute-force comparison against %d images.", len(db_paths_to_process))
                            db_features = self._extract_features_for_bruteforce(db_paths_to_process, bruteforce_batch_size)
                            if db_features is not None:
                                similarities = np.dot(db_features, query_features_normalized)
                                top_indices = np.argsort(similarities)[::-1][:top_k]
                                results = [(str(db_paths_to_process[i]), float(similarities[i])) for i in top_indices]
                    else:
                        logger.error("Search cannot be performed: Vector DB is not ready and no brute-force folder was provided.")

        duration = time.time() - start_time
        model_name = self.feature_extractor.model_name
        
        return results, search_method_used, model_name, duration

    def search_similar_by_vector_bruteforce(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        db_folder_for_bruteforce: Optional[Path] = None,
        bruteforce_batch_size: int = 32,
        precomputed_db_features: Optional[np.ndarray] = None,
        precomputed_db_paths: Optional[List[Path]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Perform brute-force similarity search using a precomputed query vector.

        This avoids needing a query image path and is suitable for augmented workflows
        that already computed embeddings in Phase A.

        Args:
            query_vector: Normalized query feature vector (1D numpy array).
            top_k: Number of neighbors to return.
            db_folder_for_bruteforce: Folder containing candidate images.
            bruteforce_batch_size: Batch size for DB feature extraction when not precomputed.
            precomputed_db_features: Optional precomputed DB feature matrix (N, D).
            precomputed_db_paths: Optional list of image paths corresponding to rows in
                                  precomputed_db_features.
        Returns:
            List of (image_path, score) pairs sorted by score desc, length <= top_k.
        """
        if precomputed_db_features is None or precomputed_db_paths is None:
            if not db_folder_for_bruteforce:
                logger.error("Bruteforce vector search requires either precomputed DB features or a db_folder_for_bruteforce.")
                return []
            db_paths = [p for p in image_path_generator(db_folder_for_bruteforce)]
            if not db_paths:
                logger.warning("No images found in the specified folder for brute-force comparison.")
                return []
            db_features = self._extract_features_for_bruteforce(db_paths, bruteforce_batch_size)
        else:
            db_paths = precomputed_db_paths
            db_features = precomputed_db_features

        if db_features is None or db_features.shape[0] == 0:
            logger.warning("Database features are empty; cannot perform brute-force search.")
            return []

        qv = np.asarray(query_vector).reshape(1, -1).astype(db_features.dtype, copy=False)
        sims = np.dot(db_features, qv.T).ravel()
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(str(db_paths[i]), float(sims[i])) for i in top_indices]

    def _extract_features_for_bruteforce(self, image_paths: List[Path], batch_size: int) -> Optional[np.ndarray]:
        """Helper method to extract features in batches, specifically for brute-force search."""
        if not image_paths:
            return None
        
        all_features = []
        progress_bar = tqdm.tqdm(range(0, len(image_paths), batch_size), desc="Brute-Force Extraction", unit="batch")
        for i in progress_bar:
            batch_paths = image_paths[i:i + batch_size]
            batch_features = self.feature_extractor.extract_features(batch_paths)
            if batch_features is not None and batch_features.shape[0] > 0:
                all_features.append(batch_features)
        
        if not all_features:
            return None
        
        return np.concatenate(all_features, axis=0)
        
    

