"""Feature Extractor for Image Similarity.

This module defines the `FeatureExtractor` class, responsible for loading various
deep learning models (both from TorchVision and Hugging Face Transformers), 
preprocessing images, and extracting feature embeddings suitable for similarity search.

It handles:
- Model configuration via a `MODEL_CONFIGS` dictionary.
- Loading models from local paths or downloading from Hugging Face Hub.
- Automatic saving of downloaded Hugging Face models to a project-local directory.
- Image transformation pipelines appropriate for each model.
- Batch processing of images for feature extraction.
- GPU device selection and memory management considerations (via `gpu_utils`).
"""
# =============================================================================
# 1. Standard Library Imports
# =============================================================================
import logging
from pathlib import Path
from typing import List, Optional, Union, Any, Dict, Tuple

# =============================================================================
# 2. Third-Party Library Imports
# =============================================================================
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models as torchvision_models
from transformers import AutoImageProcessor, AutoModel

# =============================================================================
# 3. Application-Specific Imports
# =============================================================================
from .gpu_utils import get_device
from .constants import MODEL_CONFIGS
from .efficientnet_resolver import resolve_efficientnet_variant

# =============================================================================
# 4. Module-level Logger Setup
# =============================================================================
logger = logging.getLogger(__name__)

# =============================================================================
# 5. FeatureExtractor Class Definition
# =============================================================================
class FeatureExtractor:
    """
    Manages deep learning model loading, image preprocessing, and feature extraction.

    This class provides a unified interface for various image feature extraction
    models. It supports models from PyTorch's TorchVision library and Hugging Face
    Transformers. The primary goal is to take an image (or batch of images) and
    produce a dense vector representation (feature embedding).
    """
    
    # =========================================================================
    # 5.2. Initialization
    # =========================================================================
    def __init__(self, config: Dict[str, Any], project_root_path: Optional[Path] = None) -> None:
            """
            Initializes the FeatureExtractor.

            Args:
                config (Dict[str, Any]): The 'feature_extractor' section of the main config.
                project_root_path (Optional[Path]): The project's root path for resolving
                                                    relative model paths. Defaults to CWD.
            """
            self.config = config
            self.project_root_path = project_root_path or Path.cwd()

            self.model_name = self.config.get('model_name')
            if not self.model_name:
                raise ValueError("'model_name' is a required key in the feature_extractor config.")
            self.model_name = self.model_name.lower()
            
            if self.model_name not in MODEL_CONFIGS:
                raise ValueError(f"Unsupported model: '{self.model_name}'. Choose from {list(MODEL_CONFIGS.keys())}.")
            self.model_specific_config = MODEL_CONFIGS[self.model_name]

            self.pretrained_source = self.config.get('pretrained_model_path_or_id')
            self.device = get_device()

            self.image_processor: Optional[Any] = None # For Hugging Face models
            self.model: torch.nn.Module = self._load_model()
            self.transform: T.Compose = self._get_transform()

            self._feature_dim_cache: Optional[int] = None
            self.feature_dim: int = self._get_feature_dim()

            logger.debug("FeatureExtractor for model '%s' initialized. Feature dimension: %d.",
                        self.model_name, self.feature_dim)

    # =========================================================================
    # 5.3. Model Loading
    # =========================================================================
    def _load_model(self) -> torch.nn.Module:
        """
        Loads the specified model architecture and its pretrained weights.

        Handles both TorchVision and Hugging Face models. For Hugging Face models,
        it automatically caches downloaded models locally.

        Returns:
            torch.nn.Module: The loaded PyTorch model, moved to the correct device.

        Raises:
            ValueError: If the model configuration is invalid.
            RuntimeError: If model loading fails for any reason.
        """
        model_type = self.model_specific_config["type"]
        source_to_load = self.pretrained_source
        
        logger.debug("Loading model '%s' (type: %s) using source: '%s'",
                    self.model_name, model_type, source_to_load or "Default Weights/ID")
        
        model: torch.nn.Module

        # Resolve relative paths for local model files
        resolved_source = source_to_load
        if source_to_load and not Path(source_to_load).is_absolute():
            potential_path = self.project_root_path / source_to_load
            if potential_path.exists():
                resolved_source = str(potential_path.resolve())

        try:
            if model_type == "torchvision":
                # Allow EfficientNet variant auto-resolution based on the configured source.
                tv_loader = self.model_specific_config.get("torchvision_loader")
                weights_enum = self.model_specific_config.get("weights_enum")
                feature_extractor_layer = self.model_specific_config.get("feature_layer_extractor")

                if str(self.model_name).startswith("efficientnet"):
                    try:
                        variant, loader_fn, resolved_weights_enum, input_size, extractor = resolve_efficientnet_variant(self.pretrained_source)
                        tv_loader = loader_fn
                        weights_enum = resolved_weights_enum if resolved_weights_enum is not None else weights_enum
                        feature_extractor_layer = extractor
                        # Update dynamic model settings for downstream transform and feature dim detection
                        self.model_specific_config["input_size"] = input_size
                        self.model_specific_config["torchvision_loader"] = tv_loader
                        self.model_specific_config["weights_enum"] = weights_enum
                        self.model_specific_config["feature_layer_extractor"] = feature_extractor_layer
                        # Identity-safe output handler: accept 2D, otherwise mean pool
                        self.model_specific_config["output_handler"] = lambda tensor: tensor if getattr(tensor, 'ndim', 0) == 2 else torch.mean(tensor, dim=[2, 3])
                        logger.info("Using EfficientNet variant resolved via source: %s", variant)
                    except Exception as e_res:
                        logger.warning("EfficientNet resolver failed (%s). Falling back to static config for '%s'.", e_res, self.model_name)

                # Instantiate with no weights; we'll attach custom/default weights below
                base_model = tv_loader(weights=None)
                
                if resolved_source and Path(resolved_source).is_file():
                    logger.debug("Loading TorchVision weights from local file: %s", resolved_source)
                    # Warn early if filename suggests an architecture variant mismatch (e.g., B2 weights with B0 model)
                    try:
                        fname = Path(resolved_source).name.lower()
                        if self.model_name == "efficientnet" and any(tag in fname for tag in ["_b1", "_b2", "_b3", "_b4", "_b5", "_b6", "_b7"]):
                            logger.warning(
                                "Custom EfficientNet weights '%s' appear to target a non-B0 variant while model_name='efficientnet' (B0). "
                                "Consider using a matching model_name (e.g., 'efficientnet_b2') or a B0 checkpoint.",
                                fname,
                            )
                    except Exception:
                        pass

                    use_default_fallback = False
                    try:
                        checkpoint = torch.load(resolved_source, map_location=self.device, weights_only=True)
                        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
                        if state_dict is None:
                            state_dict = checkpoint
                        if isinstance(state_dict, dict) and state_dict and all(isinstance(k, str) for k in state_dict.keys()):
                            if all(k.startswith('module.') for k in state_dict.keys()):
                                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                        else:
                            raise RuntimeError("Checkpoint does not contain a valid state_dict mapping.")
                        # Prefer robust load: allow missing/unexpected keys; escalate on shape mismatches
                        try:
                            incompatible = base_model.load_state_dict(state_dict, strict=False)
                            missing_count = len(getattr(incompatible, 'missing_keys', []))
                            unexpected_count = len(getattr(incompatible, 'unexpected_keys', []))
                            if missing_count or unexpected_count:
                                logger.warning(
                                    "Custom weights loaded with missing_keys=%d, unexpected_keys=%d. Verify architecture compatibility.",
                                    missing_count, unexpected_count
                                )
                            logger.debug("Custom TorchVision weights loaded (non-strict).")
                        except Exception as e_ld:
                            raise RuntimeError(f"State dict load failed due to incompatible shapes or keys: {e_ld}") from e_ld
                    except Exception as e_ckpt:
                        logger.warning(
                            "Failed to load custom weights from '%s': %s. Falling back to default weights if available.",
                            resolved_source, e_ckpt
                        )
                        use_default_fallback = True

                    if use_default_fallback:
                        if weights_enum is not None:
                            logger.info("Loading default TorchVision weights (%s). May download if not cached.", str(weights_enum))
                            default_model = tv_loader(weights=weights_enum)
                            base_model.load_state_dict(default_model.state_dict())
                        else:
                            logger.warning("No default TorchVision weights available; proceeding with randomly initialized model.")
                elif weights_enum is not None:
                    logger.info("Loading default TorchVision weights (%s). May download if not cached.", str(weights_enum))
                    default_model = tv_loader(weights=weights_enum)
                    base_model.load_state_dict(default_model.state_dict())
                else:
                    logger.warning("No pretrained weights for TorchVision model '%s'. Using random init.", self.model_name)

                feature_extractor_layer = self.model_specific_config.get("feature_layer_extractor")
                model = feature_extractor_layer(base_model) if feature_extractor_layer else base_model

            elif model_type == "hf":
                load_id = resolved_source or self.model_specific_config.get("hf_default_id")
                if not load_id:
                    raise ValueError(f"Hugging Face model '{self.model_name}' requires a Hub ID or local path.")

                is_local_dir = Path(load_id).is_dir()
                try:
                    self.image_processor = AutoImageProcessor.from_pretrained(load_id, local_files_only=is_local_dir)
                    model = AutoModel.from_pretrained(load_id, local_files_only=is_local_dir)
                    logger.debug("Successfully loaded HF model and processor for '%s'.", self.model_name)
                except EnvironmentError as e:
                    if is_local_dir:
                        logger.warning("Strict local load failed from '%s'. Retrying with network access. Error: %s", load_id, e)
                        self.image_processor = AutoImageProcessor.from_pretrained(load_id)
                        model = AutoModel.from_pretrained(load_id)
                    else:
                        raise e

            else:
                raise ValueError(f"Unknown model type '{model_type}' in MODEL_CONFIGS.")

            model.to(self.device)
            model.eval()
            logger.info("Model '%s' loaded to device '%s' and set to evaluation mode.", self.model_name, self.device)
            return model

        except Exception as e:
            logger.critical("Fatal error during model loading for '%s': %s", self.model_name, e, exc_info=True)
            raise RuntimeError(f"Failed to load model '{self.model_name}'") from e

    # =========================================================================
    # 5.4. Image Transformation
    # =========================================================================
    def _get_transform(self) -> T.Compose:
        """
        Constructs the appropriate image transformation pipeline for the loaded model.
        """
        if self.model_specific_config["type"] == "torchvision":
            _, height, width = self.model_specific_config["input_size"]
            return T.Compose([
                T.ToPILImage(),
                T.Resize((height, width)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif self.model_specific_config["type"] == "hf":
            if not self.image_processor:
                raise RuntimeError("Hugging Face image processor not available for transformation.")
            # Wrap the HF processor call in a lambda to fit the Compose pipeline
            return T.Compose([
                T.ToPILImage(),
                lambda pil_img: self.image_processor(images=pil_img, return_tensors="pt")['pixel_values'].squeeze(0)
            ])
        else:
            raise NotImplementedError(f"Transformation for model type '{self.model_specific_config['type']}' not implemented.")

    # =========================================================================
    # 5.5. Feature Dimension and Extraction
    # =========================================================================
    def _get_feature_dim(self) -> int:
        """Determines the model's output feature dimension via a dummy forward pass."""
        # [FIXED] Corrected typo from `self_feature_dim_cache` to `self._feature_dim_cache`.
        if self._feature_dim_cache is not None:
            return self._feature_dim_cache

        logger.debug("Determining feature dimension for '%s' via dummy pass...", self.model_name)
        try:
            c, h, w = self.model_specific_config["input_size"]
            dummy_input = torch.randn(1, c, h, w).to(self.device)

            with torch.no_grad():
                if self.model_specific_config["type"] == "torchvision":
                    raw_output = self.model(dummy_input)
                elif self.model_specific_config["type"] == "hf":
                    raw_output = self.model(pixel_values=dummy_input)
                else:
                    raise RuntimeError(f"Unknown model type '{self.model_specific_config['type']}'")
                
                output_handler = self.model_specific_config["output_handler"]
                feature_tensor = output_handler(raw_output)

            if feature_tensor.ndim != 2:
                raise RuntimeError(f"Output handler produced a tensor with {feature_tensor.ndim} dimensions. Expected 2 (batch, dim).")
            
            self._feature_dim_cache = feature_tensor.shape[1]
            logger.debug("Determined feature dimension: %d", self._feature_dim_cache)
            return self._feature_dim_cache
        except Exception as e:
            logger.error("Error determining feature dimension for '%s': %s", self.model_name, e, exc_info=True)
            raise RuntimeError(f"Failed to determine feature dimension for '{self.model_name}'") from e

    def extract_features(self, 
                         image_input: Union[Path, str, np.ndarray, List[Union[Path, str, np.ndarray]]]
                         ) -> np.ndarray:
        """
        Extracts L2-normalized feature embeddings from one or more images.

        Args:
            image_input: A single image (path or NumPy array) or a list of images for batch processing.

        Returns:
            np.ndarray: A NumPy array of feature embeddings. Shape is `(feature_dim,)` for a single
                        input or `(batch_size, feature_dim)` for a batch input.

        Raises:
            FileNotFoundError: If an image path does not exist.
            ValueError: If an input type is invalid or an image cannot be read.
            RuntimeError: If model inference fails.
        """
        is_batch = isinstance(image_input, list)
        input_list = image_input if is_batch else [image_input]

        if not input_list:
            return np.array([])

        batch_tensors: List[torch.Tensor] = []
        for item in input_list:
            if isinstance(item, (Path, str)):
                img_path = Path(item)
                if not img_path.is_file():
                    raise FileNotFoundError(f"Image file not found: {img_path.resolve()}")
                image_bgr = cv2.imread(str(img_path.resolve()))
                if image_bgr is None:
                    raise ValueError(f"Could not read image from path: {img_path.resolve()}")
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                batch_tensors.append(self.transform(image_rgb))
            elif isinstance(item, np.ndarray):
                image_rgb = item if item.ndim == 3 else cv2.cvtColor(item, cv2.COLOR_GRAY2RGB)
                batch_tensors.append(self.transform(image_rgb))
            else:
                raise ValueError(f"Unsupported input type '{type(item)}'. Expected Path, str, or np.ndarray.")

        input_batch = torch.stack(batch_tensors).to(self.device)

        with torch.no_grad():
            if self.model_specific_config["type"] == "torchvision":
                raw_outputs = self.model(input_batch)
            elif self.model_specific_config["type"] == "hf":
                raw_outputs = self.model(pixel_values=input_batch)
            else:
                raise RuntimeError(f"Unknown model type '{self.model_specific_config['type']}' during feature extraction.")

            output_handler = self.model_specific_config["output_handler"]
            features = output_handler(raw_outputs)
        
        # The robust output handlers should always produce a 2D tensor.
        # An assertion is better than trying to patch the shape here.
        assert features.ndim == 2, f"Feature tensor must be 2D (batch, dim), but got shape {features.shape}"
        
        normalized_features = torch.nn.functional.normalize(features, p=2, dim=1)
        features_np = normalized_features.cpu().numpy()

        return features_np[0] if not is_batch and features_np.shape[0] == 1 else features_np