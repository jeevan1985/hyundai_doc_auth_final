"""Public API exports for the key_input toolkit.

This package provides utilities for working with key-driven local image fetch
operations used by the mock API server and related workflows.

Typical usage:
    from external.key_input import LocalFolderFetcher, LocalFetchConfig, NameMappingConfig

The implementation lives in key_input_orchestrator.py. Import paths and behavior
remain unchanged for existing modules that import the module directly.
"""
from __future__ import annotations

# Prevent "No handler found" warnings if the application hasn't configured logging.
import logging as _logging
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

from .key_input_orchestrator import (
    LocalFolderFetcher,
    LocalFetchConfig,
    NameMappingConfig,
)

__all__ = [
    "LocalFolderFetcher",
    "LocalFetchConfig",
    "NameMappingConfig",
]
