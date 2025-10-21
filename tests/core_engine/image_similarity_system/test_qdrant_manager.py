
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, ANY

import pytest
import numpy as np

# Ensure the project root is on the Python path
PKG_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from hyundai_document_authenticator.core_engine.image_similarity_system.qdrant_manager import QdrantManager


@pytest.fixture
def mock_qdrant_client(mocker) -> MagicMock:
    """Mocks the QdrantClient and its methods."""
    mock_client = MagicMock()
    # Mock the responses for collection listing and counting
    mock_client.get_collections.return_value = MagicMock(collections=[])
    mock_client.count.return_value = MagicMock(count=0)
    mocker.patch("hyundai_document_authenticator.core_engine.image_similarity_system.qdrant_manager.QdrantClient", return_value=mock_client)
    return mock_client


@pytest.fixture
def qdrant_config() -> dict:
    """Provides a default configuration for the QdrantManager tests."""
    return {
        "feature_dim": 128,
        "collection_name_stem": "test_collection",
        "model_name": "test_model",
        "qdrant_config": {
            "location": ":memory:",
            "partition_capacity": 100  # Set a small capacity for testing rollover
        }
    }


def test_qdrant_manager_initialization_sharded(mock_qdrant_client: MagicMock, qdrant_config: dict):
    """Test that the manager initializes in sharded mode and creates the first shard."""
    # Arrange
    # Act
    manager = QdrantManager(**qdrant_config)

    # Assert
    assert manager.partition_capacity == 100
    assert manager.collection_name_base == "test_collection_test_model"
    # Check that it created the first shard collection
    mock_qdrant_client.create_collection.assert_called_with(
        collection_name="test_collection_test_model_s0001",
        vectors_config=ANY,  # Use ANY for complex objects
        quantization_config=ANY,
        hnsw_config=ANY
    )
    assert manager.active_collection_name == "test_collection_test_model_s0001"


def test_upsert_with_shard_rollover(mock_qdrant_client: MagicMock, qdrant_config: dict):
    """
    Test the core logic of rolling over to a new shard when the current one is full.
    """
    # Arrange
    manager = QdrantManager(**qdrant_config)
    
    # Simulate the first shard being almost full
    mock_client = mock_qdrant_client
    mock_client.count.return_value = MagicMock(count=90)

    # Create a batch of 20 vectors, which should cause a rollover
    vectors = np.random.rand(20, qdrant_config["feature_dim"]).astype(np.float32)
    paths = [Path(f"/img_{i}.jpg") for i in range(20)]

    # Act
    manager._upsert_with_shard_rollover(vectors, paths)

    # Assert
    # 1. It should have upserted the first 10 vectors into the first shard
    assert mock_client.upsert.call_count == 2
    first_call = mock_client.upsert.call_args_list[0]
    assert first_call.kwargs["collection_name"] == "test_collection_test_model_s0001"
    assert len(first_call.kwargs["points"]) == 10

    # 2. It should have created a new shard (_s0002)
    # The manager creates the initial shard, so we expect a second create call for the new shard
    assert mock_client.create_collection.call_count == 2 
    second_create_call = mock_client.create_collection.call_args_list[1]
    assert second_create_call.kwargs["collection_name"] == "test_collection_test_model_s0002"

    # 3. It should have updated the active collection name
    assert manager.active_collection_name == "test_collection_test_model_s0002"

    # 4. It should have upserted the remaining 10 vectors into the new shard
    second_upsert_call = mock_client.upsert.call_args_list[1]
    assert second_upsert_call.kwargs["collection_name"] == "test_collection_test_model_s0002"
    assert len(second_upsert_call.kwargs["points"]) == 10
