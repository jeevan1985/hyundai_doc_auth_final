"""GPU utilities for device selection and CUDA cache management.

This module provides two helpers used by the TIF pipelines:
- get_device(): returns the best available torch.device (CUDA if available, otherwise CPU)
- clear_gpu_memory(): triggers Python GC and clears CUDA cache when available

Notes
- The import of torch is required by the feature extractor and FAISS flows. If torch is not
  installed in the current environment, static analyzers may report an import-error.
  That is an environment issue rather than a code defect. The code below keeps import
  statements explicit and adds minimal lint pragmas where appropriate.
"""

from __future__ import annotations

# Standard library
import gc
import logging

# Third-party
import torch  # pylint: disable=import-error

# Module logger
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Choose the most appropriate PyTorch device.

    Preference order
    1) If CUDA is available and multiple GPUs are present, choose the one with the most free memory.
    2) If a single CUDA device is present, choose it.
    3) Otherwise, fall back to CPU.

    Returns
        torch.device: e.g., torch.device('cuda:0') or torch.device('cpu')
    """
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        logger.info("NVIDIA CUDA is available. Found %d CUDA device(s).", num_devices)

        if num_devices == 1:
            device_index = 0
            device = torch.device(f"cuda:{device_index}")
            try:
                free_mem, total_mem = torch.cuda.mem_get_info(device_index)
                logger.info(
                    "Using single CUDA device %d: %s, Total Memory: %.2f GiB, Available Memory: %.2f GiB",
                    device_index,
                    torch.cuda.get_device_name(device_index),
                    total_mem / (1024**3),
                    free_mem / (1024**3),
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "Using single CUDA device %d. Could not get memory/name info: %s",
                    device_index,
                    exc,
                )
            return device

        max_free_memory = 0
        best_device_index = 0
        logger.info("Evaluating memory on multiple CUDA devices:")
        for i in range(num_devices):
            try:
                free_memory, total_memory = torch.cuda.mem_get_info(i)
                device_name = torch.cuda.get_device_name(i)
                logger.info(
                    "  Device %d (%s): Total Memory = %.2f GiB, Available Memory = %.2f GiB",
                    i,
                    device_name,
                    total_memory / (1024**3),
                    free_memory / (1024**3),
                )
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_device_index = i
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "Could not query memory/name for CUDA device %d: %s. Skipping for selection.",
                    i,
                    exc,
                )

        device = torch.device(f"cuda:{best_device_index}")
        selected_device_name = "N/A"
        try:
            selected_device_name = torch.cuda.get_device_name(best_device_index)
        except Exception:  # pylint: disable=broad-except
            # Already warned when the per-device query failed above
            pass
        logger.info(
            "Selected CUDA device %d (%s) with %.2f GiB free memory.",
            best_device_index,
            selected_device_name,
            max_free_memory / (1024**3),
        )
    else:
        device = torch.device("cpu")
        logger.info("No NVIDIA CUDA devices available. Using CPU.")
    return device


def clear_gpu_memory() -> None:
    """Attempt to free GPU memory by running GC and clearing CUDA cache.

    This can be used after large allocations or IVF training. If CUDA is not
    available, the function becomes a no-op beyond triggering Python GC.
    """
    logger.debug("Attempting to clear GPU memory...")
    gc.collect()
    logger.debug("Python garbage collection triggered.")

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            logger.debug("PyTorch CUDA cache cleared.")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Could not clear PyTorch CUDA cache: %s", exc)
    else:
        logger.debug("NVIDIA CUDA not available, skipping CUDA cache clearing.")
