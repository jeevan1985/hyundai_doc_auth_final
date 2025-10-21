
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

# Ensure the project root is on the Python path
PKG_ROOT: Path = Path(__file__).resolve().parents[4]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

# Ensure 'paddleocr' is available during import of the searcher; inject a stub if missing.
try:
    import paddleocr  # type: ignore  # noqa: F401
except Exception:
    import types
    sys.modules["paddleocr"] = types.SimpleNamespace(PaddleOCR=MagicMock())

from hyundai_document_authenticator.external.tif_searcher.tif_searcher_core.searcher import TifTextSearcher


@pytest.fixture
def mock_ocr_engine(mocker: MagicMock) -> MagicMock:
    """Mocks the PaddleOCR engine to return predefined text results."""
    mock_engine = MagicMock()
    
    # Simulate finding text on page 2 but not on page 1
    # The structure is [[(bbox, (text, confidence))]]
    page_1_result = []
    page_2_result = [
        ([[0, 0], [100, 0], [100, 20], [0, 20]], ("some other text", 0.99)),
        ([[0, 30], [100, 30], [100, 50], [0, 50]], ("target_text_here", 0.95)),
    ]
    
    # ocr() is called for each page
    mock_engine.ocr.side_effect = [page_1_result, page_2_result]
    
    mocker.patch("paddleocr.PaddleOCR", return_value=mock_engine)
    return mock_engine


@pytest.fixture
def tif_searcher(mock_ocr_engine: MagicMock) -> TifTextSearcher:
    """Provides a TifTextSearcher instance with a mocked OCR engine."""
    # Initialize with a specific search text
    return TifTextSearcher(search_text="target_text_here", ocr_backend="paddleocr")


@pytest.fixture
def mock_tif_file(mocker: MagicMock, tmp_path: Path) -> Path:
    """Creates a mock TIF file path and mocks PIL.Image to simulate a multi-page TIF."""
    mock_image = MagicMock(spec=Image.Image)
    mock_image.n_frames = 2 # Simulate a 2-page TIF
    mocker.patch("PIL.Image.open", return_value=mock_image)
    
    dummy_path = tmp_path / "test.tif"
    dummy_path.touch()
    return dummy_path


def test_find_text_pages_locates_correct_page(
    tif_searcher: TifTextSearcher, 
    mock_tif_file: Path, 
    mock_ocr_engine: MagicMock
):
    """
    Tests that find_text_pages correctly identifies the page containing the search text.
    """
    # Arrange
    # The tif_searcher is configured to search for "target_text_here"
    # The mock_ocr_engine is configured to find this text only on page 2

    # Act
    found_pages = tif_searcher.find_text_pages(mock_tif_file)

    # Assert
    # 1. Check that the OCR engine was called for each page of the TIF
    assert mock_ocr_engine.ocr.call_count == 2

    # 2. Check that the function returned only the page number where text was found
    assert found_pages == [2]


def test_find_text_pages_no_match(tif_searcher: TifTextSearcher, mock_tif_file: Path, mock_ocr_engine: MagicMock):
    """
    Tests that find_text_pages returns an empty list when the search text is not found.
    """
    # Arrange
    # Reconfigure the searcher to look for text that doesn't exist in the mock OCR output
    tif_searcher.search_text = "non_existent_text"

    # Act
    found_pages = tif_searcher.find_text_pages(mock_tif_file)

    # Assert
    assert found_pages == []
