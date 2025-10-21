
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import numpy as np

# Ensure the project root is on the Python path
PKG_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

# ---------------------------------------------------------------------------
# Lightweight stubs for heavy dependencies to keep tests self-contained.
# These stubs allow importing the feature_extractor without installing
# heavy frameworks, and provide just enough behavior for our tests.
# ---------------------------------------------------------------------------
class _Tensor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = np.array(arr)

    @property
    def shape(self):
        return self._arr.shape

    @property
    def ndim(self) -> int:
        return self._arr.ndim

    def to(self, *_args, **_kwargs):
        return self

    def permute(self, *dims: int) -> "_Tensor":
        return _Tensor(self._arr.transpose(dims))

    def cpu(self) -> "_Tensor":
        return self

    def numpy(self) -> np.ndarray:
        return np.array(self._arr)


def _from_numpy(arr: np.ndarray) -> _Tensor:
    return _Tensor(arr)


def _stack(tensors: list[_Tensor]) -> _Tensor:
    data = np.stack([t._arr for t in tensors], axis=0)
    return _Tensor(data)


def _randn(*shape: int) -> _Tensor:
    return _Tensor(np.random.randn(*shape).astype(np.float32))


def _mean(t: _Tensor, dim: list[int]) -> _Tensor:
    arr = t._arr
    for d in sorted(dim, reverse=True):
        arr = arr.mean(axis=d)
    return _Tensor(arr)


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


# torch.nn.functional.normalize replacement
class _NNFunc:
    @staticmethod
    def normalize(t: _Tensor, p: int = 2, dim: int = 1) -> _Tensor:
        arr = t._arr
        denom = np.linalg.norm(arr, ord=p, axis=dim, keepdims=True) + 1e-12
        return _Tensor(arr / denom)


class _NN:
    class Module:  # minimal placeholder
        pass

    functional = _NNFunc()


# Inject stubs into sys.modules if not present
if "torch" not in sys.modules:
    sys.modules["torch"] = SimpleNamespace(
        from_numpy=_from_numpy,
        stack=_stack,
        randn=_randn,
        no_grad=lambda: _NoGrad(),
        mean=_mean,
        nn=_NN(),
    )

# torchvision is imported for models and transforms but we bypass usage
if "torchvision" not in sys.modules:
    sys.modules["torchvision"] = MagicMock()
if "torchvision.transforms" not in sys.modules:
    sys.modules["torchvision.transforms"] = MagicMock()
if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()
# transformers are only used for HF models; not exercised here
if "transformers" not in sys.modules:
    sys.modules["transformers"] = MagicMock()

# Import after stubbing heavy deps
from hyundai_document_authenticator.core_engine.image_similarity_system.feature_extractor import FeatureExtractor  # noqa: E402


@pytest.fixture
def fe_config() -> dict:
    """Provides a default configuration for the FeatureExtractor.

    Returns:
        dict: Minimal config using a supported model key from constants.
    """
    return {
        "model_name": "resnet",  # Use supported key compatible with current implementation
        "pretrained": True,
        "device": "cpu",
        "image_size": 224,
    }


def test_feature_extractor_initialization(fe_config: dict, mocker: MagicMock):
    """Tests that the FeatureExtractor initializes with minimal dependencies.

    Patches heavy operations while validating key attributes are set.
    """
    # Patch heavy parts: model loading and feature-dim probing
    mock_model = MagicMock()
    mock_model.to.return_value = None
    mock_model.eval.return_value = None

    mocker.patch.object(FeatureExtractor, "_load_model", return_value=mock_model)
    mocker.patch(
        "hyundai_document_authenticator.core_engine.image_similarity_system.feature_extractor.get_device",
        return_value="cpu",
    )
    # Prevent dummy forward during feature-dim probing
    mocker.patch.object(FeatureExtractor, "_get_feature_dim", return_value=128)

    # Act
    feature_extractor = FeatureExtractor(config=fe_config, project_root_path=PKG_ROOT)

    # Assert
    assert feature_extractor.model_name == "resnet"
    # Feature dim patched to avoid real forward pass
    assert feature_extractor.feature_dim == 128


def test_extract_features(fe_config: dict, mocker: MagicMock):
    """Tests the feature extraction process for a batch of images using numpy arrays.

    The test uses a lightweight dummy model that returns a fixed 2D tensor via
    the minimal Tensor stub injected above, avoiding heavyweight dependencies.
    """
    # Arrange
    # Dummy model returns (batch, 128) tensor filled with ones
    class DummyModel:
        def to(self, *_args, **_kwargs):
            return None

        def eval(self):
            return None

        def __call__(self, input_batch):  # returns (batch, 128)
            b = input_batch.shape[0]
            return _Tensor(np.ones((b, 128), dtype=np.float32))

    mocker.patch.object(FeatureExtractor, "_load_model", return_value=DummyModel())
    mocker.patch(
        "hyundai_document_authenticator.core_engine.image_similarity_system.feature_extractor.get_device",
        return_value="cpu",
    )
    # Avoid probing the real model
    mocker.patch.object(FeatureExtractor, "_get_feature_dim", return_value=128)

    feature_extractor = FeatureExtractor(config=fe_config, project_root_path=PKG_ROOT)

    # Replace transform with a minimal numpy->Tensor converter to avoid torchvision
    def _simple_transform(np_img: np.ndarray):
        # Expect HxWxC uint8; convert to CHW float32 tensor
        arr = (np_img.astype(np.float32) / 255.0)
        if arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))
        return _Tensor(arr)

    feature_extractor.transform = _simple_transform  # type: ignore[assignment]

    # Create a batch of two dummy images as numpy arrays
    image_arrays = [
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.ones((64, 64, 3), dtype=np.uint8) * 255,
    ]

    # Act
    features = feature_extractor.extract_features(image_arrays)

    # Assert
    assert isinstance(features, np.ndarray)
    assert features.shape == (2, 128)
