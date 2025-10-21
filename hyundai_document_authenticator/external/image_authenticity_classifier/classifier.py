"""Image authenticity classifier module.

Provides a robust wrapper over optional PyTorch/torchvision and timm backends to
load a classifier, preprocess inputs, and run single-image inference. Designed
to be resilient to missing dependencies and various checkpoint formats.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# Optional deep learning stack
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as tv_models
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except Exception as e:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    tv_models = None  # type: ignore
    T = None  # type: ignore
    logger.warning("Torch/torchvision not available for ImageAuthenticityClassifier: %s", e)

# Optional timm support (widely used for ResNet/EfficientNet)
try:  # pragma: no cover
    import timm  # type: ignore
    from timm.data import create_transform as timm_create_transform  # type: ignore
    from timm.data import resolve_model_data_config as timm_resolve_model_data_config  # type: ignore
    TIMM_AVAILABLE = True
except Exception as e:  # pragma: no cover
    TIMM_AVAILABLE = False
    timm = None  # type: ignore
    timm_create_transform = None  # type: ignore
    timm_resolve_model_data_config = None  # type: ignore
    logger.info("timm not available. Classifier will fall back to torchvision only. %s", e)

try:
    import yaml
except Exception as e:
    yaml = None  # type: ignore
    logger.warning("PyYAML not available, classifier config YAML loading may fail: %s", e)


@dataclass
class _ClassifierConfig:
    """Configuration for the classifier.

    model_name: Name of the backbone (e.g., resnet18, resnet50, efficientnet_b0, efficientnet_b2).
    weights_path: Path to checkpoint file containing weights/state_dict.
    device: Device selection policy. 'auto' will use CUDA if available else CPU.
    labels: Ordered class names. The model's output dimension will match len(labels).
    score_threshold: Global threshold (not applied in this simple wrapper, but kept for compatibility).
    per_class_thresholds: Optional per-class thresholds.
    transforms: Optional transform overrides dict with keys: image_size, mean, std.
    model_source: Preferred model source: 'auto'|'torchvision'|'timm'. If 'auto', tries torchvision then timm.
    """

    model_name: str = "resnet18"
    weights_path: Optional[str] = None
    device: str = "auto"  # auto|cuda|cpu
    labels: List[str] = None  # type: ignore
    score_threshold: float = 0.0
    per_class_thresholds: Dict[str, float] = None  # type: ignore
    transforms: Dict[str, Any] = None  # type: ignore
    model_source: str = "auto"  # auto|torchvision|timm


class ImageAuthenticityClassifier:
    """Robust image classifier for authenticity detection using Torch or timm backends.

    Features:
    - Device selection (auto/cpu/cuda) with safety fallbacks.
    - Model construction using torchvision or timm backbones.
    - Checkpoint loading tolerant to different formats and key prefixes.
    - Predicts a class_name and score, returning a default unknown result on failure.

    Args:
        config_path (Optional[str]): Optional path to a YAML config defining model_name,
            weights_path, device, labels, thresholds, and transforms.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize the classifier from an optional YAML configuration.

        Args:
            config_path (Optional[str]): Path to classifier_config.yaml. If None,
                sensible defaults are used.
        """
        self.config_path = Path(config_path) if config_path else None
        self.cfg = self._load_cfg(self.config_path)
        self.labels = list(self.cfg.labels or [])
        if not self.labels:
            # Ensure at least a fallback label set
            self.labels = ["original", "unknown"]
        self.device = self._select_device(self.cfg.device)

        self.model: Optional[nn.Module] = None if TORCH_AVAILABLE else None
        self.preprocess = None
        if TORCH_AVAILABLE:
            try:
                self.model, self.preprocess = self._build_model_and_transforms(self.cfg)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:  # pragma: no cover
                logger.warning("ImageAuthenticityClassifier model init failed: %s", e)
                self.model = None
                self.preprocess = None
        else:
            logger.warning("Torch unavailable: classifier will return unknown results.")

    # -----------------------
    # Public API
    # -----------------------
    def infer(self, image: Image.Image) -> Dict[str, Any]:
        """Run a single-image inference.

        Args:
            image (PIL.Image.Image): Input PIL image. Converted to RGB internally.

        Returns:
            Dict[str, Any]: {"class_name": str, "score": float in [0,1]}.
                On failure, returns {"class_name": "unknown", "score": 0.0}.
        """
        try:
            if image is None:
                return {"class_name": "unknown", "score": 0.0}
            if not TORCH_AVAILABLE or self.model is None or self.preprocess is None:
                return {"class_name": "unknown", "score": 0.0}

            img = image.convert("RGB")
            x = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                probs_t = F.softmax(logits, dim=1)[0]
                probs = probs_t.detach().cpu().numpy().tolist()
            if not probs:
                return {"class_name": "unknown", "score": 0.0}
            idx = int(max(range(len(probs)), key=lambda i: probs[i]))
            label = self.labels[idx] if idx < len(self.labels) else "unknown"
            score = float(probs[idx])
            return {"class_name": str(label), "score": score}
        except Exception as e:  # pragma: no cover
            logger.warning("ImageAuthenticityClassifier.infer failed: %s", e)
            return {"class_name": "unknown", "score": 0.0}

    # -----------------------
    # Internals
    # -----------------------
    def _load_cfg(self, path: Optional[Path]) -> _ClassifierConfig:
        """Load classifier configuration from YAML or return defaults.

        Args:
            path (Optional[pathlib.Path]): Optional path to a YAML configuration file.

        Returns:
            _ClassifierConfig: Parsed configuration or sensible defaults.
        """
        if path is None:
            return _ClassifierConfig(
                model_name="resnet18",
                weights_path=None,
                device="auto",
                labels=["original", "unknown"],
                score_threshold=0.0,
                per_class_thresholds={},
                transforms={"image_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                model_source="auto",
            )
        try:
            if yaml is None:
                raise RuntimeError("PyYAML is not available")
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return _ClassifierConfig(
                model_name=str(data.get("model_name", "resnet18")),
                weights_path=data.get("weights_path"),
                device=str(data.get("device", "auto")),
                labels=list(data.get("labels", []) or []),
                score_threshold=float(data.get("score_threshold", 0.0)),
                per_class_thresholds=dict(data.get("per_class_thresholds", {}) or {}),
                transforms=dict(data.get("transforms", {}) or {}),
                model_source=str(data.get("model_source", "auto")),
            )
        except Exception as e:
            logger.warning("Failed to load classifier config '%s': %s. Using defaults.", path, e)
            return _ClassifierConfig(
                model_name="resnet18",
                weights_path=None,
                device="auto",
                labels=["original", "unknown"],
                score_threshold=0.0,
                per_class_thresholds={},
                transforms={"image_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                model_source="auto",
            )

    def _select_device(self, device_str: str) -> str:
        """Select device string honoring requested policy and availability.

        Args:
            device_str (str): Requested device policy ('auto'|'cuda'|'cpu').

        Returns:
            str: 'cuda' or 'cpu'. If CUDA requested but unavailable, raises RuntimeError.
        """
        d = (device_str or "auto").lower()
        if not TORCH_AVAILABLE:
            return "cpu"
        if d == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if d == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            return "cuda"
        return "cpu"

    # --- Model building and transforms ---
    def _build_model_and_transforms(self, cfg: _ClassifierConfig) -> Tuple[nn.Module, Any]:
        """Construct the model and preprocessing transforms.

        Args:
            cfg (_ClassifierConfig): Configuration with model/backend and transforms.

        Returns:
            Tuple[nn.Module, Any]: (model, preprocess_transform)
        """
        model_name = (cfg.model_name or "resnet18").lower()
        num_classes = max(1, len(self.labels))

        preferred = (cfg.model_source or "auto").lower()
        model: Optional[nn.Module] = None
        backend_used = ""

        # Try to honor preferred backend but be resilient
        if preferred in ("auto", "torchvision"):
            model = self._try_create_torchvision_model(model_name, num_classes)
            backend_used = "torchvision" if model is not None else backend_used

        if model is None and preferred in ("auto", "timm") and TIMM_AVAILABLE:
            model = self._try_create_timm_model(model_name, num_classes)
            backend_used = "timm" if model is not None else backend_used

        if model is None:
            # Final safety fallback (torchvision resnet18)
            model = self._try_create_torchvision_model("resnet18", num_classes)
            backend_used = backend_used or "torchvision"
            logger.warning(
                "Falling back to default torchvision resnet18 due to model creation failure for '%s'", model_name
            )

        # Load weights if present
        if cfg.weights_path:
            self._load_weights(model, cfg.weights_path)

        # Build transforms
        preprocess = self._build_preprocess(cfg, model, backend_used)

        logger.info("Classifier initialized with backend=%s, model_name=%s, num_classes=%d", backend_used, model_name, num_classes)
        return model, preprocess

    def _try_create_torchvision_model(self, model_name: str, num_classes: int) -> Optional[nn.Module]:
        """Attempt to create a torchvision model with replaced classifier head.

        Args:
            model_name (str): Torchvision model name (e.g., 'resnet18').
            num_classes (int): Output classes to configure in final head.

        Returns:
            Optional[nn.Module]: Constructed model or None on failure.
        """
        if not TORCH_AVAILABLE:
            return None
        try:
            ctor = getattr(tv_models, model_name, None)
            model: Optional[nn.Module] = None
            if callable(ctor):
                # Support both modern "weights=None" and legacy "pretrained=False"
                try:
                    model = ctor(weights=None)
                except TypeError:
                    model = ctor(pretrained=False)  # type: ignore[arg-type]
            else:
                # Heuristic mapping for generic names like "resnet" or "efficientnet"
                if model_name.startswith("resnet"):
                    model = tv_models.resnet18(weights=None)
                elif model_name.startswith("efficientnet"):
                    # default to b0 if variant missing in this torchvision version
                    ef_ctor = getattr(tv_models, model_name, None)
                    if callable(ef_ctor):
                        try:
                            model = ef_ctor(weights=None)
                        except TypeError:
                            model = ef_ctor(pretrained=False)  # type: ignore[arg-type]
                    else:
                        model = tv_models.efficientnet_b0(weights=None)
                else:
                    model = tv_models.resnet18(weights=None)

            # Replace classification head to match num_classes
            self._replace_classifier_head(model, num_classes)
            return model
        except Exception as e:  # pragma: no cover
            logger.warning("Torchvision model creation failed for '%s': %s", model_name, e)
            return None

    def _try_create_timm_model(self, model_name: str, num_classes: int) -> Optional[nn.Module]:
        """Attempt to create a timm model with target num_classes.

        Args:
            model_name (str): timm model name.
            num_classes (int): Output classes.

        Returns:
            Optional[nn.Module]: Constructed model or None on failure.
        """
        if not TIMM_AVAILABLE:
            return None
        try:
            # timm will build the correct head when num_classes is provided
            model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
            return model
        except Exception as e:  # pragma: no cover
            logger.warning("timm model creation failed for '%s': %s", model_name, e)
            return None

    def _replace_classifier_head(self, model: nn.Module, num_classes: int) -> None:
        """Replace the final classification layer for common torchvision/timm models.

        Attempts to cover:
        - torchvision ResNet (model.fc)
        - torchvision EfficientNet (model.classifier[1]) and similar nets with classifier module
        - generic classifier attribute (Linear or Sequential with final Linear)
        - timm models that implement reset_classifier
        """
        # timm models generally support reset_classifier
        if hasattr(model, "reset_classifier"):
            try:
                model.reset_classifier(num_classes=num_classes)  # type: ignore[attr-defined]
                return
            except Exception:
                pass

        # torchvision ResNet-style models
        if hasattr(model, "fc") and isinstance(getattr(model, "fc"), nn.Linear):
            old: nn.Linear = getattr(model, "fc")
            setattr(model, "fc", nn.Linear(old.in_features, num_classes))
            return

        # torchvision EfficientNet and others with classifier
        if hasattr(model, "classifier"):
            classifier = getattr(model, "classifier")
            if isinstance(classifier, nn.Sequential):
                # Replace the last Linear layer inside the Sequential
                idx = None
                for i in range(len(classifier) - 1, -1, -1):
                    if isinstance(classifier[i], nn.Linear):
                        idx = i
                        break
                if idx is not None:
                    in_f = classifier[idx].in_features  # type: ignore[attr-defined]
                    new_seq = list(classifier.children())
                    new_seq[idx] = nn.Linear(in_f, num_classes)
                    setattr(model, "classifier", nn.Sequential(*new_seq))
                    return
            if isinstance(classifier, nn.Linear):
                in_f = classifier.in_features
                setattr(model, "classifier", nn.Linear(in_f, num_classes))
                return

        # Some models use 'head' as the classifier
        if hasattr(model, "head") and isinstance(getattr(model, "head"), nn.Linear):
            old_head: nn.Linear = getattr(model, "head")
            setattr(model, "head", nn.Linear(old_head.in_features, num_classes))
            return

        logger.warning("Could not automatically replace classifier head; model may have incorrect output dimension.")

    def _load_weights(self, model: nn.Module, weights_path: str) -> None:
        """Load weights from a variety of common checkpoint formats.

        Supports:
        - Plain state_dict
        - Dict with 'state_dict' or 'model' key
        - Keys prefixed with 'module.' (DataParallel) or 'model.' (some training frameworks)
        - Cross-backend loading with strict=False (partial load allowed)
        """
        try:
            wpath = Path(weights_path)
            if not wpath.is_file():
                logger.warning("Classifier weights not found at %s. Proceeding without weights.", wpath)
                return
            map_location = "cpu" if self.device == "cpu" else self.device
            state = torch.load(str(wpath), map_location=map_location)

            # Extract state_dict if needed
            if isinstance(state, dict):
                if "state_dict" in state and isinstance(state["state_dict"], dict):
                    state_dict = state["state_dict"]
                elif "model" in state and isinstance(state["model"], dict):
                    state_dict = state["model"]
                else:
                    # Might already be a state_dict
                    state_dict = state
            else:
                logger.warning("Unexpected checkpoint type (%s). Proceeding without weights.", type(state))
                return

            # Strip known prefixes
            def _strip_prefix(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
                """Return a copy of the mapping with a leading prefix removed from keys, if present.

                Args:
                    d (Dict[str, Any]): Original state-dict-like mapping.
                    prefix (str): Key prefix to remove (e.g., 'module.' or 'model.').

                Returns:
                    Dict[str, Any]: New mapping with prefixes stripped where applicable.
                """
                if any(k.startswith(prefix) for k in d.keys()):
                    return {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)} | {
                        k: v for k, v in d.items() if not k.startswith(prefix)
                    }
                return d

            state_dict = _strip_prefix(state_dict, "module.")
            state_dict = _strip_prefix(state_dict, "model.")

            # Attempt load with strict=False to allow cross-library compatibility
            incompatible = model.load_state_dict(state_dict, strict=False)
            missing = getattr(incompatible, "missing_keys", [])
            unexpected = getattr(incompatible, "unexpected_keys", [])
            if missing or unexpected:
                logger.info(
                    "Weights loaded with partial match. missing_keys=%d, unexpected_keys=%d",
                    len(missing), len(unexpected)
                )
        except Exception as e:
            logger.warning("Failed to load classifier weights '%s': %s. Proceeding without weights.", weights_path, e)

    def _build_preprocess(self, cfg: _ClassifierConfig, model: nn.Module, backend: str):
        """Create preprocessing transforms for model inference.

        Priority:
        1) If cfg.transforms provided, honor those (image_size/mean/std).
        2) If backend is timm and timm is available, use timm's recommended eval transform.
        3) Fallback to standard ImageNet normalization with 224 center-crop.
        """
        tconf = cfg.transforms or {}
        if tconf:
            size = int(tconf.get("image_size", 224))
            mean = tconf.get("mean", [0.485, 0.456, 0.406])
            std = tconf.get("std", [0.229, 0.224, 0.225])
            preprocess = T.Compose([
                T.Resize(size),
                T.CenterCrop(size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
            return preprocess

        if backend == "timm" and TIMM_AVAILABLE:
            try:
                cfg_d = timm_resolve_model_data_config(model)
                preprocess = timm_create_transform(**cfg_d, is_training=False)
                return preprocess
            except Exception as e:  # pragma: no cover
                logger.info("timm transform creation failed, falling back to default: %s", e)

        # Default torchvision-style transforms
        size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        preprocess = T.Compose([
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return preprocess
