"""Configuration loader and merger utilities for the Image Similarity System. 🧩

This module provides a robust configuration loading pipeline with clear order of
precedence, safe YAML parsing, deep dictionary merges, targeted override support,
and a helper to extract values via dot-separated key paths. The behavior is
kept identical to the original implementation while elevating documentation,
typing, and code style to production standards.

Order of precedence for loading configs (highest to lowest):
    1) config_dict overrides (in-memory)
    2) primary_config_path (YAML file)
    3) fallback_config_path (YAML file)

Design notes:
    - Deep dictionary merge ensures nested maps are merged rather than replaced. ⚙️
    - YAML parsing uses safe_load to avoid executing arbitrary code.
    - Intentional error handling: for primary/fallback files we log and continue
      in most cases to maintain a resilient experience for CLI/API users.

No functionality has been altered. Only clarity, safety, and documentation have
been improved. ✨
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

# Module logger
logger = logging.getLogger(__name__)


def _deep_merge_dicts(base_dict: Dict[str, Any], merge_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge merge_dict into base_dict and return a new dictionary. 🔀

    For keys where both values are dictionaries, a recursive merge is performed;
    otherwise the value from merge_dict overwrites the base.

    Args:
        base_dict (Dict[str, Any]): Base mapping to start from.
        merge_dict (Dict[str, Any]): Mapping whose values take precedence.

    Returns:
        Dict[str, Any]: A new dictionary representing the merged result.

    Raises:
        None
    """
    # Start with a deep copy to avoid mutating the original base mapping.
    result_dict = copy.deepcopy(base_dict)

    for key, value_to_merge in merge_dict.items():
        # When both sides are dicts, merge recursively to avoid losing nested values.
        if isinstance(value_to_merge, dict) and isinstance(result_dict.get(key), dict):
            result_dict[key] = _deep_merge_dicts(result_dict[key], value_to_merge)
        else:
            result_dict[key] = value_to_merge
    return result_dict


def load_config(
    primary_config_path: Optional[str] = None,
    fallback_config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load configuration using file(s) and/or in-memory overrides. 📜

    The function supports a clear order of precedence and a safe merge strategy,
    returning a single consolidated dictionary. Behavior is unchanged from the
    original implementation.

    Args:
        primary_config_path (Optional[str]): Path to the primary YAML configuration file.
        fallback_config_path (Optional[str]): Path to a fallback YAML file for defaults.
        config_dict (Optional[Dict[str, Any]]): Dictionary of overrides with highest precedence.

    Returns:
        Dict[str, Any]: The final merged configuration.

    Raises:
        FileNotFoundError: If primary_config_path is specified but not found and
            no fallback was successfully loaded.
        ValueError: If config_dict is not a dictionary, or no config source is given.
        yaml.YAMLError: If a YAML file is malformed and no usable config is available.
    """
    final_config: Dict[str, Any] = {}

    # 1. Load fallback config first (if provided)
    if fallback_config_path:
        fallback_file = Path(fallback_config_path)
        logger.debug(
            "Attempting to load fallback configuration from: %s", fallback_file.resolve()
        )
        if not fallback_file.is_file():
            logger.warning(
                "Fallback configuration file not found: %s. Skipping.",
                fallback_file.resolve(),
            )
        else:
            try:
                with open(fallback_file, "r", encoding="utf-8") as f:
                    fallback_data = yaml.safe_load(f)
                if isinstance(fallback_data, dict):
                    final_config = fallback_data
                    logger.info(
                        "Fallback configuration loaded from %s.", fallback_file.resolve()
                    )
                elif fallback_data is None:
                    logger.info(
                        "Fallback configuration file %s is empty.", fallback_file.resolve()
                    )
                    final_config = {}
                else:
                    logger.warning(
                        "Fallback config in '%s' is not a dictionary. Skipping.",
                        fallback_file.resolve(),
                    )
            except yaml.YAMLError as e_yaml:
                logger.error(
                    "Error parsing fallback YAML '%s': %s. Skipping.",
                    fallback_file,
                    e_yaml,
                )
            except Exception as e_load:  # noqa: BLE001 - log and continue, preserve behavior
                logger.error(
                    "Error loading fallback config '%s': %s. Skipping.",
                    fallback_file,
                    e_load,
                )

    # 2. Load primary config and merge it over the fallback
    if primary_config_path:
        primary_file = Path(primary_config_path)
        logger.debug(
            "Attempting to load primary configuration from: %s",
            primary_file.resolve(),
        )
        if not primary_file.is_file():
            if not final_config:
                raise FileNotFoundError(
                    f"Primary config not found: {primary_file.resolve()} and no fallback was loaded."
                )
            else:
                logger.warning(
                    "Primary config file %s not found. Using fallback.",
                    primary_file.resolve(),
                )
        else:
            try:
                with open(primary_file, "r", encoding="utf-8") as f:
                    primary_data = yaml.safe_load(f)
                if isinstance(primary_data, dict):
                    final_config = _deep_merge_dicts(final_config, primary_data)
                    logger.info(
                        "Primary configuration loaded and merged from %s.",
                        primary_file.resolve(),
                    )
                elif primary_data is None:
                    logger.info(
                        "Primary configuration file %s is empty.",
                        primary_file.resolve(),
                    )
                else:
                    logger.error(
                        "Primary config in '%s' is not a dictionary.",
                        primary_file.resolve(),
                    )
                    if not final_config:
                        raise ValueError(
                            f"Primary config {primary_file.resolve()} is invalid and no fallback was loaded."
                        )
            except yaml.YAMLError as e_yaml:
                logger.error("Error parsing primary YAML '%s': %s", primary_file, e_yaml)
                if not final_config:
                    raise
                logger.warning(
                    "Proceeding with fallback config due to primary config error."
                )
            except Exception as e_load:  # noqa: BLE001 - log and continue if fallback exists
                logger.error("Error loading primary config '%s': %s", primary_file, e_load)
                if not final_config:
                    raise
                logger.warning(
                    "Proceeding with fallback config due to primary config error."
                )

    # 3. Merge config_dict, which has the highest precedence
    if config_dict is not None:
        logger.debug(
            "Merging configuration from provided dictionary (highest precedence)."
        )
        if not isinstance(config_dict, dict):
            raise ValueError("Provided 'config_dict' must be a dictionary.")
        final_config = _deep_merge_dicts(final_config, config_dict)

    if not final_config and not primary_config_path and not fallback_config_path and config_dict is None:
        raise ValueError("No configuration source specified or all sources failed to load.")

    if not final_config and (primary_config_path or fallback_config_path):
        logger.warning(
            "All specified configuration files were empty or invalid, resulting in an empty config."
        )
        return {}

    logger.debug("Final merged configuration loaded.")
    return final_config


def save_config(config_data: Dict[str, Any], config_file_path: Union[str, Path]) -> bool:
    """Save the provided configuration mapping to a YAML file. 💾

    The parent directory is created if necessary. If `config_saving.create_backup_on_config_save`
    is True (default), a timestamped backup of the existing file is created prior to overwrite.

    Args:
        config_data (Dict[str, Any]): Configuration mapping to persist.
        config_file_path (Union[str, Path]): Destination path for the YAML file.

    Returns:
        bool: True if saving succeeded; False otherwise.

    Raises:
        None: Errors are logged and converted to a boolean outcome to maintain
        original behavior.
    """
    config_path = Path(config_file_path)
    try:
        # Ensure parent directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Honor nested 'config_saving.create_backup_on_config_save' flag (default True)
        should_create_backup = (
            config_data.get("config_saving", {}).get("create_backup_on_config_save", True)
        )

        if should_create_backup and config_path.is_file():
            try:
                import datetime
                import shutil

                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                backup_path = config_path.with_suffix(f".{timestamp}.bak")
                shutil.copy2(config_path, backup_path)
                logger.info("Configuration backup created at: %s", backup_path)
            except ImportError:
                logger.warning(
                    "Could not import modules for backup creation. Skipping backup."
                )
            except Exception as e_backup:  # noqa: BLE001 - log and continue per original behavior
                logger.error("Error creating backup for %s: %s", config_path, e_backup)

        # Write the new configuration to the file
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, indent=2, sort_keys=False, default_flow_style=False)
        logger.info("Configuration saved successfully to %s", config_path.resolve())
        return True

    except Exception as e:  # noqa: BLE001 - log and return False to preserve behavior
        logger.error(
            "Error saving configuration to %s: %s", config_path.resolve(), e, exc_info=True
        )
        return False


def load_and_merge_configs(
    primary_config_path: Union[str, Path],
    fallback_config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience wrapper to load a config with optional overrides. 🧰

    Args:
        primary_config_path (Union[str, Path]): Primary YAML configuration path.
        fallback_config_path (Optional[Union[str, Path]]): Fallback YAML path used for defaults.
        overrides (Optional[Dict[str, Any]]): Final precedence layer to merge onto the result.

    Returns:
        Dict[str, Any]: Final merged configuration dictionary.

    Raises:
        FileNotFoundError: If primary is missing and fallback not available.
        ValueError: If overrides is not a dictionary or no valid source provided.
        yaml.YAMLError: If YAML parsing fails and no fallback exists.
    """
    primary_path_str: Optional[str] = (
        str(primary_config_path)
        if isinstance(primary_config_path, (str, Path))
        else primary_config_path  # type: ignore[assignment]
    )
    fallback_path_str: Optional[str] = (
        str(fallback_config_path)
        if isinstance(fallback_config_path, (str, Path))
        else fallback_config_path  # type: ignore[assignment]
    )
    return load_config(
        primary_config_path=primary_path_str,
        fallback_config_path=fallback_path_str,
        config_dict=overrides or {},
    )


def get_config_value_by_key_path(
    config: Dict[str, Any],
    key_path: str,
    default: Any = None,
) -> Any:
    """Retrieve nested configuration value via dot-separated key path. 🔎

    Provides a safe and convenient way to access nested configuration values
    without writing multiple chained `.get()` calls.

    Args:
        config (Dict[str, Any]): Configuration mapping to search within.
        key_path (str): Dot-separated path (e.g., "vector_database.faiss.index_type").
        default (Any): Value to return if the key path is not found.

    Returns:
        Any: The value found at the key path, or the provided default.

    Example:
        >>> config = {"search_task": {"parameters": {"top_k": 10}}}
        >>> get_config_value_by_key_path(config, "search_task.parameters.top_k", default=5)
        10

    Raises:
        None
    """
    keys = key_path.split(".")
    current_level: Any = config
    for key in keys:
        if isinstance(current_level, dict) and key in current_level:
            current_level = current_level[key]
        else:
            logger.debug(
                "Key path '%s' not fully found. Returning default for key '%s'.",
                key_path,
                key,
            )
            return default

    return current_level
