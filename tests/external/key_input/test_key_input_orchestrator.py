
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

# Ensure the project root is on the Python path
PKG_ROOT: Path = Path(__file__).resolve().parents[3]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from hyundai_document_authenticator.external.key_input.key_input_orchestrator import KeyDrivenOrchestrator


@pytest.fixture
def sample_configs(tmp_path: Path) -> tuple[Path, Path]:
    """Creates temporary main and key_input config files for testing."""
    main_config_path = tmp_path / "image_similarity_config.yaml"
    key_input_config_path = tmp_path / "key_input_config.yaml"

    main_config = {
        "input_mode": {
            "doc_input_start": "key",
            "key_input_config_path": str(key_input_config_path)
        }
    }

    key_input_config = {
        "key_input": {
            "input_table_path": "/fake/table.csv",
            "batch_size": 10
        },
        "data_source": {
            "mode": "local",
            "local": {"search_roots": ["/fake/root"]}
        }
    }

    with open(main_config_path, "w") as f:
        yaml.dump(main_config, f)
    with open(key_input_config_path, "w") as f:
        yaml.dump(key_input_config, f)

    return main_config_path, key_input_config_path


@pytest.fixture
def mock_orchestrator_deps(mocker: MagicMock) -> dict:
    """Mocks the dependencies of the KeyDrivenOrchestrator."""
    # Mock the loader to return a single batch of filenames
    mock_loader = MagicMock()
    mock_loader.iter_filenames.return_value = iter([f"key_{i}.tif" for i in range(5)])
    mocker.patch("hyundai_document_authenticator.external.key_input.key_input_orchestrator.KeyInputLoader", return_value=mock_loader)

    # Mock the fetcher to return resolved paths
    mock_fetcher = MagicMock()
    fetched_paths = [Path(f"/fake/root/key_{i}.tif") for i in range(5)]
    mock_fetcher.fetch_batch.return_value = fetched_paths
    mocker.patch("hyundai_document_authenticator.external.key_input.key_input_orchestrator.LocalFolderFetcher", return_value=mock_fetcher)

    # Mock the final workflow function that gets called
    mock_workflow = mocker.patch("hyundai_document_authenticator.external.key_input.key_input_orchestrator.execute_tif_batch_search_workflow")

    return {
        "loader": mock_loader,
        "fetcher": mock_fetcher,
        "workflow": mock_workflow
    }


def test_orchestrator_run_local_mode(
    sample_configs: tuple[Path, Path],
    mock_orchestrator_deps: dict,
    tmp_path: Path
):
    """
    Tests the orchestrator's run method in 'local' fetch mode.
    """
    # Arrange
    main_config_path, _ = sample_configs
    orchestrator = KeyDrivenOrchestrator()

    # Act
    result = orchestrator.run(main_config_path)

    # Assert
    # 1. Check that the orchestrator reported success
    assert result is not None
    assert result["status"] == "success"
    assert result["total_files_requested"] == 5
    assert result["total_files_resolved"] == 5
    assert result["total_batches"] == 1

    # 2. Check that the final workflow was called
    mock_workflow = mock_orchestrator_deps["workflow"]
    mock_workflow.assert_called_once()

    # 3. Check the arguments passed to the workflow
    call_args, _ = mock_workflow.call_args
    workflow_config = call_args[0]
    workflow_input_override = call_args[2] # input_folder_override

    # The orchestrator should create a temporary directory for the batch
    # and pass it to the workflow.
    assert workflow_input_override is not None
    assert workflow_input_override.is_dir()
    # Check that the symlinks were created in the temp dir
    created_links = list(workflow_input_override.iterdir())
    assert len(created_links) == 5
    assert created_links[0].name == "key_0.tif"
