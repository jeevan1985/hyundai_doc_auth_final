
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

# Ensure the project root is on the Python path
PKG_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

# Also ensure the package source directory is importable for module-level relative imports in the CLI
PKG_SRC: Path = PKG_ROOT / "hyundai_document_authenticator"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))


def _inject_cli_import_stubs() -> None:
    """Inject stub modules for CLI's import-time dependencies to avoid sys.exit.

    The CLI imports from core_engine.image_similarity_system at module import time and exits on ImportError.
    These stubs make the imports succeed during tests to keep them hermetic.
    """
    # Create parent packages
    if "core_engine" not in sys.modules:
        sys.modules["core_engine"] = types.ModuleType("core_engine")
    if "core_engine.image_similarity_system" not in sys.modules:
        sys.modules["core_engine.image_similarity_system"] = types.ModuleType(
            "core_engine.image_similarity_system"
        )

    # Stub workflow module with required symbols
    wf_mod = types.ModuleType("core_engine.image_similarity_system.workflow")

    def _stub_build_index_from_tif_folder_workflow(*_args, **_kwargs):
        return {"status": "success", "exit_code": 0, "message": "stub"}

    def _stub_execute_tif_batch_search_workflow(*_args, **_kwargs):
        return {"status": "success", "per_query": []}

    wf_mod.build_index_from_tif_folder_workflow = _stub_build_index_from_tif_folder_workflow  # type: ignore[attr-defined]
    wf_mod.execute_tif_batch_search_workflow = _stub_execute_tif_batch_search_workflow  # type: ignore[attr-defined]
    sys.modules["core_engine.image_similarity_system.workflow"] = wf_mod

    # Stub config_loader module
    cl_mod = types.ModuleType("core_engine.image_similarity_system.config_loader")
    cl_mod.load_and_merge_configs = lambda _p: {}  # type: ignore[attr-defined]
    sys.modules["core_engine.image_similarity_system.config_loader"] = cl_mod


def test_cli_help_command():
    """Tests that the CLI app responds to the --help argument in-process.

    Uses Typer's CliRunner to avoid subprocess and keep tests hermetic.
    """
    _inject_cli_import_stubs()
    from hyundai_document_authenticator.doc_image_verifier import app  # import after stubbing

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    assert "TIF Document Similarity System - Local CLI" in result.stdout


def test_cli_search_doc_command_call(mocker: MagicMock):
    """Tests that the `search-doc` command calls the workflow function with parsed options.

    Patches the workflow and config loader, then runs the command in-process with CliRunner.
    """
    _inject_cli_import_stubs()
    # Import after stubbing; then patch the imported names in the module namespace
    import importlib
    cli_mod = importlib.import_module("hyundai_document_authenticator.doc_image_verifier")

    # Patch the function that does the actual work
    mock_workflow = mocker.patch.object(
        cli_mod,
        "execute_tif_batch_search_workflow",
        return_value={"per_query": []},  # Return a minimal valid result
    )

    # Mock the config loader to avoid filesystem dependency
    mocker.patch.object(cli_mod, "load_and_merge_configs", return_value={})

    runner = CliRunner()
    result = runner.invoke(cli_mod.app, [
        "search-doc",
        "--top-k",
        "15",
    ])

    # 1. Check that the CLI command ran successfully
    assert result.exit_code == 0

    # 2. Check that our mocked workflow function was called exactly once
    mock_workflow.assert_called_once()

    # 3. Check that the arguments passed to the workflow reflect the CLI options
    call_args, _ = mock_workflow.call_args
    passed_config = call_args[0]
    assert passed_config["search_task"]["top_k"] == 15
