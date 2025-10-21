
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
import numpy as np

# Ensure the project root is on the Python path
PKG_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from hyundai_document_authenticator.core_engine.image_similarity_system.faiss_manager import FaissIndexManager


@pytest.fixture
def mock_faiss(mocker: MagicMock) -> MagicMock:
    """Mocks the entire faiss library."""
    mock_faiss_lib = MagicMock()
    # Mock the Index class and its methods
    mock_index = MagicMock()
    mock_index.ntotal = 0
    mock_faiss_lib.IndexIVFFlat.return_value = mock_index
    mock_faiss_lib.IndexFlatL2.return_value = mock_index
    mock_faiss_lib.index_factory.return_value = mock_index
    
    mocker.patch.dict(sys.modules, {"faiss": mock_faiss_lib})
    return mock_faiss_lib


@pytest.fixture
def faiss_manager(tmp_path: Path) -> FaissIndexManager:
    """Provides a FaissIndexManager instance with a temporary output directory."""
    config = {
        "feature_dim": 128,
        "output_directory": str(tmp_path),
        "filename_stem": "test_index",
        "model_name": "test_model",
        "faiss_config": {
            "partition_capacity": 50 # Small capacity for testing
        },
        "index_type": "flat"
    }
    return FaissIndexManager(**config)


@pytest.fixture
def mock_feature_extractor(mocker: MagicMock) -> MagicMock:
    """Mocks the FeatureExtractor."""
    mock_fe = MagicMock()
    mock_fe.extract_features.return_value = np.random.rand(10, 128).astype(np.float32)
    return mock_fe


def test_faiss_manager_sharding_on_build(faiss_manager: FaissIndexManager, mock_faiss: MagicMock, mock_feature_extractor: MagicMock, mocker: MagicMock, tmp_path: Path):
    """
    Tests that build_index_from_folder creates multiple shards when capacity is exceeded.
    """
    # Arrange
    # Simulate 120 image paths, which should create 3 shards (50, 50, 20)
    image_paths = [tmp_path / f"img_{i}.jpg" for i in range(120)]
    mocker.patch("pathlib.Path.glob", return_value=image_paths)
    mocker.patch("hyundai_document_authenticator.core_engine.image_similarity_system.utils.image_path_generator", return_value=image_paths)
    
    # Mock faiss.write_index to check its calls
    mock_write_index = mocker.patch("faiss.write_index")

    # Act
    faiss_manager.build_index_from_folder(
        feature_extractor=mock_feature_extractor,
        image_folder=str(tmp_path),
        batch_size=10
    )

    # Assert
    # 1. Check that the feature extractor was called for all images
    # 120 images / 10 batch_size = 12 calls
    assert mock_feature_extractor.extract_features.call_count == 12

    # 2. Check that faiss.write_index was called 3 times for the 3 shards
    assert mock_write_index.call_count == 3

    # 3. Verify the filenames of the created shards
    expected_shard_names = [
        tmp_path / "test_index_test_model_flat_s0001.index",
        tmp_path / "test_index_test_model_flat_s0002.index",
        tmp_path / "test_index_test_model_flat_s0003.index",
    ]
    
    # Get the path from the second argument of each call to write_index
    called_paths = [call_args[0][1] for call_args in mock_write_index.call_args_list]
    
    assert str(expected_shard_names[0]) in called_paths
    assert str(expected_shard_names[1]) in called_paths
    assert str(expected_shard_names[2]) in called_paths

    # 4. Check that the manager's internal state tracks the shards
    assert len(faiss_manager.index_identifiers) == 3
