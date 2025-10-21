
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

# Ensure the project root is on the Python path
PKG_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from hyundai_document_authenticator.external.image_authenticity_classifier.classifier import ImageAuthenticityClassifier


@pytest.fixture(scope="module")
def classifier() -> ImageAuthenticityClassifier:
    """Pytest fixture to initialize the ImageAuthenticityClassifier from a test config."""
    # Construct path to the test config relative to this test file
    config_path = Path(__file__).parent.parent.parent / "fixtures" / "test_classifier_config.yaml"
    print(f"Loading test config from: {config_path.resolve()}")
    return ImageAuthenticityClassifier(config_path=str(config_path))


def test_classifier_inference(classifier: ImageAuthenticityClassifier, mocker) -> None:
    """
    Tests the infer method of the ImageAuthenticityClassifier.

    Args:
        classifier: The classifier instance from the pytest fixture.
        mocker: The pytest-mock fixture for mocking objects.
    """
    # 1. Arrange: Create a mock PIL Image object
    # We don't need a real image; we just need an object that has a .convert('RGB') method.
    mock_image = mocker.MagicMock(spec=Image.Image)
    mock_image.convert.return_value = mock_image  # .convert() returns itself

    # 2. Act: Run inference on the mock image
    # Since no real model is loaded, this tests the fallback/error-handling path.
    result = classifier.infer(mock_image)

    # 3. Assert: Check if the output is structured as expected
    assert isinstance(result, dict)
    assert "class_name" in result
    assert "score" in result
    assert isinstance(result["class_name"], str)
    assert isinstance(result["score"], float)

    # Because we didn't load a real model, we expect the 'unknown' default result
    assert result["class_name"] == "unknown"
    assert result["score"] == 0.0

def test_classifier_init_no_path() -> None:
    """Tests that the classifier can be initialized with no config path and not crash."""
    # Act & Assert: This should not raise an exception
    try:
        clf = ImageAuthenticityClassifier(config_path=None)
        assert clf is not None
        assert clf.model is not None  # A default model should be initialized
    except Exception as e:
        pytest.fail(f"ImageAuthenticityClassifier(config_path=None) raised an exception: {e}")
