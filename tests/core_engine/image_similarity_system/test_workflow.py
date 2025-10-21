
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import numpy as np

# Ensure the project root is on the Python path
PKG_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from hyundai_document_authenticator.core_engine.image_similarity_system.workflow import execute_tif_batch_search_workflow


@pytest.fixture
def workflow_config(tmp_path: Path) -> dict:
    """Provides a sample configuration for workflow tests."""
    # tmp_path is a built-in pytest fixture for a temporary directory
    return {
        "feature_extractor": {"model_name": "test_model"},
        "vector_database": {
            "provider": "faiss",
            "allow_fallback": False,
            "faiss": {"output_directory": str(tmp_path)}
        },
        "search_task": {
            "input_tif_folder_for_search": str(tmp_path),
            "output_folder_for_results": str(tmp_path),
            "top_k": 2,
            "top_doc": 2,
            "aggregation_strategy": "max",
            "privacy_mode": False, # Allow outputs for testing
            "index_query_doc_images": False # Use legacy path for this test
        },
        "photo_extractor_config": {
            "photo_extraction_mode": "bbox",
            "bbox_extraction": {"bbox_list": [[0, 0, 100, 100]]}
        },
        "searcher_config": {"search_text": "DUMMY"}
    }


@pytest.fixture
def mock_workflow_dependencies(mocker) -> dict:
    """Mocks all external dependencies for the TIF search workflow."""
    # Mock TifTextSearcher
    mock_tif_searcher = MagicMock()
    mock_tif_searcher.find_text_pages.return_value = [1] # Always find a page
    mocker.patch("hyundai_document_authenticator.core_engine.image_similarity_system.workflow.TifTextSearcher", return_value=mock_tif_searcher)

    # Mock PhotoExtractor
    mock_photo_extractor = MagicMock()
    # Return a single dummy PIL image
    mock_photo_extractor.extract_photos_from_bboxes.return_value = [MagicMock(spec=np.ndarray)]
    mocker.patch("hyundai_document_authenticator.core_engine.image_similarity_system.workflow.PhotoExtractor", return_value=mock_photo_extractor)

    # Mock FeatureExtractor
    mock_feature_extractor = MagicMock()
    mock_feature_extractor.feature_dim = 16
    mocker.patch("hyundai_document_authenticator.core_engine.image_similarity_system.workflow.FeatureExtractor", return_value=mock_feature_extractor)

    # Mock ImageSimilaritySearcher to return predictable results
    mock_searcher = MagicMock()
    def search_side_effect(*args, **kwargs):
        # Return different results based on the query to test aggregation
        query_path = kwargs.get("query_image_path")
        if "query1" in str(query_path):
            return [("doc_A_p1.jpg", 0.9), ("doc_B_p1.jpg", 0.8)], "mock", 0, 0
        else:
            return [("doc_B_p2.jpg", 0.95), ("doc_C_p1.jpg", 0.7)], "mock", 0, 0
    mock_searcher.search_similar_images.side_effect = search_side_effect
    mocker.patch("hyundai_document_authenticator.core_engine.image_similarity_system.workflow.ImageSimilaritySearcher", return_value=mock_searcher)
    
    # Mock FaissIndexManager to appear ready
    mock_faiss = MagicMock()
    mock_faiss.load_index.return_value = True
    mock_faiss.is_index_loaded_and_ready.return_value = True
    mock_faiss.get_total_indexed_items.return_value = 100
    mocker.patch("hyundai_document_authenticator.core_engine.image_similarity_system.workflow.FaissIndexManager", return_value=mock_faiss)

    # Mock filesystem interactions
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("pathlib.Path.glob", return_value=[
        Path("query1.tif"), Path("query2.tif")
    ])
    mocker.patch("PIL.Image.open") # Prevent file read attempts

    return {
        "tif_searcher": mock_tif_searcher,
        "photo_extractor": mock_photo_extractor,
        "searcher": mock_searcher,
        "faiss_manager": mock_faiss
    }


def test_execute_tif_batch_search_workflow_aggregation(workflow_config: dict, mock_workflow_dependencies: dict, tmp_path: Path):
    """
    Tests the end-to-end TIF search workflow with a focus on score aggregation.
    """
    # Arrange
    project_root = PKG_ROOT
    input_folder = tmp_path / "input"
    input_folder.mkdir()

    # Act
    result = execute_tif_batch_search_workflow(workflow_config, project_root, input_folder_override=input_folder)

    # Assert
    # 1. Check the overall status
    assert result["status"] == "success"
    assert len(result["per_query"]) == 2

    # 2. Check the global top documents, which should be aggregated with MAX score
    top_docs = result["top_documents"]
    assert len(top_docs) == 2 # top_doc is 2 in config
    assert top_docs[0]["document"] == "doc_B"
    assert top_docs[0]["score"] == pytest.approx(0.95) # From query2
    assert top_docs[1]["document"] == "doc_A"
    assert top_docs[1]["score"] == pytest.approx(0.9)  # From query1

    # 3. Check per-query results for query1
    query1_result = next(p for p in result["per_query"] if p["matched_query_document"] == "query1.tif")
    assert query1_result["top_docs"][0]["document"] == "doc_A"
    assert query1_result["top_docs"][0]["score"] == pytest.approx(0.9)
