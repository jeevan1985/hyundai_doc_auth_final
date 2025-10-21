
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

# Ensure the project root is on the Python path
PKG_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from hyundai_document_authenticator.external.photo_extractor.photo_extractor.extractor import PhotoExtractor


@pytest.fixture
def photo_extractor() -> PhotoExtractor:
    """Provides a default PhotoExtractor instance for bbox mode."""
    config_override = {
        "photo_extraction_mode": "bbox",
        "yolo_object_detection": None # Ensure YOLO is disabled
    }
    return PhotoExtractor(config_override=config_override)


@pytest.fixture
def sample_image() -> Image.Image:
    """Creates a simple 300x200 black PIL Image for testing."""
    return Image.new("RGB", (300, 200), "black")


def test_extract_photos_from_bboxes_xyxy(photo_extractor: PhotoExtractor, sample_image: Image.Image):
    """Tests cropping with a single bounding box in xyxy format."""
    # Arrange
    # Define a 100x50 box at position (10, 20)
    bboxes = [[10, 20, 110, 70]] # [x1, y1, x2, y2]
    bbox_format = "xyxy"

    # Act
    extracted_photos = photo_extractor.extract_photos_from_bboxes(
        image=sample_image,
        bboxes=bboxes,
        bbox_format=bbox_format,
        normalized=False
    )

    # Assert
    assert len(extracted_photos) == 1
    
    cropped_image = extracted_photos[0]
    assert cropped_image.size == (100, 50) # width, height


def test_extract_photos_from_bboxes_normalized(photo_extractor: PhotoExtractor, sample_image: Image.Image):
    """Tests cropping with a normalized bounding box."""
    # Arrange
    # Box should be 50% of width and 50% of height -> 150x100 pixels
    # Starts at 10% from left and 10% from top -> (30, 20)
    bboxes = [[0.1, 0.1, 0.6, 0.6]] # [x1, y1, x2, y2]
    bbox_format = "xyxy"

    # Act
    extracted_photos = photo_extractor.extract_photos_from_bboxes(
        image=sample_image,
        bboxes=bboxes,
        bbox_format=bbox_format,
        normalized=True
    )

    # Assert
    assert len(extracted_photos) == 1
    
    cropped_image = extracted_photos[0]
    # Check dimensions, allowing for rounding errors
    assert cropped_image.width == pytest.approx(150) # 300 * (0.6 - 0.1)
    assert cropped_image.height == pytest.approx(100) # 200 * (0.6 - 0.1)


def test_extract_photos_from_multiple_bboxes(photo_extractor: PhotoExtractor, sample_image: Image.Image):
    """Tests that the extractor can handle multiple bounding boxes in a single call."""
    # Arrange
    bboxes = [
        [0, 0, 50, 50],    # A 50x50 box at the top-left
        [200, 100, 300, 200] # A 100x100 box at the bottom-right
    ]
    bbox_format = "xyxy"

    # Act
    extracted_photos = photo_extractor.extract_photos_from_bboxes(
        image=sample_image,
        bboxes=bboxes,
        bbox_format=bbox_format,
        normalized=False
    )

    # Assert
    assert len(extracted_photos) == 2
    assert extracted_photos[0].size == (50, 50)
    assert extracted_photos[1].size == (100, 100)
