"""PyTorch Dataset classes for loading and transforming images.

This module provides two types of PyTorch `Dataset` implementations for handling 
image data within the image similarity system:

1.  `IterableImageDataset`: Suited for very large collections of images where 
    loading all paths into memory is undesirable. It processes images one by one 
    from a provided generator of image paths. It skips problematic images with a 
    warning.

2.  `ImageDataset`: A standard map-style `Dataset` that takes a list of image 
    paths during initialization. It provides random access to images by index but 
    requires all paths to be in memory. It raises an error if an image cannot be 
    processed.

Both datasets use OpenCV for image loading (converting BGR to RGB) and can apply 
a user-defined torchvision transformation pipeline.
"""
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Generator, Union, Callable

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, IterableDataset

# Get a logger for this module
logger = logging.getLogger(__name__)

class IterableImageDataset(IterableDataset):
    """A PyTorch `IterableDataset` for memory-efficient loading of images from a path generator.

    This dataset is designed for scenarios with a very large number of images where 
    holding all image paths in memory might be infeasible. It consumes image paths 
    from a generator and processes them one by one.
    
    If an error occurs while loading or transforming an image, it logs a warning 
    and skips the problematic image, continuing with the next available one.

    Attributes:
        image_paths_generator (Generator[Path, None, None]): A generator that yields 
            `pathlib.Path` objects, each pointing to an image file.
        transform (Optional[Callable[[np.ndarray], Union[torch.Tensor, np.ndarray]]]): 
            An optional torchvision transform pipeline (e.g., `T.Compose([...])`) or 
            any callable that accepts an HWC RGB NumPy array and returns a processed 
            image (typically a PyTorch Tensor or a NumPy array).
    """
    def __init__(self, 
                 image_paths_generator: Generator[Path, None, None],
                 transform: Optional[Callable[[np.ndarray], Union[torch.Tensor, np.ndarray]]] = None) -> None:
        """Initializes the IterableImageDataset.

        Args:
            image_paths_generator: A generator yielding `pathlib.Path` objects 
                for the images to be loaded.
            transform: An optional callable (e.g., torchvision `T.Compose` pipeline) 
                to be applied to each image after loading and BGR-to-RGB conversion. 
                If None, images are returned as HWC RGB NumPy arrays. 
                Defaults to None.
        """
        super().__init__()
        self.image_paths_generator = image_paths_generator
        self.transform = transform
        logger.debug("IterableImageDataset initialized. Transform provided: %s.",
                     'Yes' if self.transform else 'No')

    def __iter__(self) -> Generator[Tuple[Union[torch.Tensor, np.ndarray], Path], None, None]:
        """Iterates over the dataset, loading, transforming, and yielding images with their paths.

        Images are loaded using OpenCV, converted from BGR to RGB, and then processed 
        by the optional `self.transform`.

        Yields:
            A tuple `(processed_image, original_path)`:
            - `processed_image` (Union[torch.Tensor, np.ndarray]): The transformed image. 
              Its type depends on the output of the `transform` pipeline (commonly a 
              PyTorch Tensor). If no transform is applied, it's an HWC RGB NumPy array.
            - `original_path` (Path): The `pathlib.Path` of the loaded image.

        Skips images that cause errors during loading or processing, logging a warning.
        """
        # Get worker information for potential multi-worker data loading contexts.
        # Basic sharding is commented out; for true iterable dataset sharding with
        # multiple workers, a more sophisticated approach might be needed depending on the
        # nature of image_paths_generator and how it can be partitioned or shared.
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # num_workers = worker_info.num_workers if worker_info else 1 

        for img_path in self.image_paths_generator:
            # Example sharding logic (if needed and generator supports it):
            # if num_workers > 1 and i % num_workers != worker_id:
            #     continue
            try:
                # Load image with OpenCV (default: BGR format)
                image_bgr = cv2.imread(str(img_path.resolve()))
                if image_bgr is None:
                    logger.warning("Worker %d: Failed to read image (cv2.imread returned None) from path: %s. Skipping this image.",
                                   worker_id, img_path.resolve())
                    continue # Skip to the next image

                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                
                processed_image: Union[torch.Tensor, np.ndarray] = image_rgb
                if self.transform:
                    processed_image = self.transform(image_rgb) # Apply transformations

                yield processed_image, img_path

            except (cv2.error, OSError, IOError, ValueError, RuntimeError) as e:
                # Robustness: we skip unreadable/corrupt images to keep streaming going.
                logger.warning(
                    "Worker %d: Skipping image due to processing error at %s: %s",
                    worker_id,
                    img_path.resolve(),
                    e,
                )
            except Exception as e:
                # Last-resort catch to prevent worker crash on unexpected errors.
                logger.error(
                    "Worker %d: Unexpected error for %s: %s. Skipping.",
                    worker_id,
                    img_path.resolve(),
                    e,
                    exc_info=False,
                )


class ImageDataset(Dataset):
    """A standard PyTorch `Dataset` for loading images from a list of paths.

    This dataset stores a list of all image paths in memory upon initialization, 
    making it suitable for datasets where this is feasible. It provides random 
    access to images via the `__getitem__` method.

    If an error occurs while loading or transforming an image at a specific index, 
    it logs the error and raises a `RuntimeError`, halting processing for that item.

    Attributes:
        image_paths (List[Path]): A list of `pathlib.Path` objects, each pointing to an image file.
        transform (Optional[Callable[[np.ndarray], Union[torch.Tensor, np.ndarray]]]): 
            An optional torchvision transform pipeline (e.g., `T.Compose([...])`) or 
            any callable that accepts an HWC RGB NumPy array and returns a processed 
            image (typically a PyTorch Tensor or a NumPy array).
    """
    def __init__(self, 
                 image_paths: List[Path],
                 transform: Optional[Callable[[np.ndarray], Union[torch.Tensor, np.ndarray]]] = None) -> None:
        """Initializes the ImageDataset.

        Args:
            image_paths: A list of `pathlib.Path` objects for all images in the dataset.
            transform: An optional callable (e.g., torchvision `T.Compose` pipeline) 
                to be applied to each image after loading and BGR-to-RGB conversion. 
                If None, images are returned as HWC RGB NumPy arrays. 
                Defaults to None.
        """
        self.image_paths = image_paths
        self.transform = transform
        logger.debug("ImageDataset initialized with %d image paths. Transform provided: %s.",
                     len(self.image_paths), 'Yes' if self.transform else 'No')

    def __len__(self) -> int:
        """Returns the total number of images (paths) in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Union[torch.Tensor, np.ndarray], Path]:
        """Retrieves an image by its index, applies transformations, and returns it with its path.

        Images are loaded using OpenCV, converted from BGR to RGB, and then processed 
        by the optional `self.transform`.

        Args:
            idx: The integer index of the image to retrieve from `self.image_paths`.

        Returns:
            A tuple `(processed_image, original_path)`:
            - `processed_image` (Union[torch.Tensor, np.ndarray]): The transformed image. 
              Its type depends on the output of the `transform` pipeline (commonly a 
              PyTorch Tensor). If no transform is applied, it's an HWC RGB NumPy array.
            - `original_path` (Path): The `pathlib.Path` of the loaded image.

        Raises:
            IndexError: If `idx` is out of bounds for `self.image_paths`.
            RuntimeError: If the image at the specified index cannot be read or processed. 
                The original exception is chained.
        """
        if not 0 <= idx < len(self.image_paths):
            err_msg = f"Index {idx} out of bounds for ImageDataset of length {len(self.image_paths)}."
            logger.error(err_msg)
            raise IndexError(err_msg)
            
        img_path = self.image_paths[idx]
        try:
            # Load image with OpenCV (default: BGR format)
            image_bgr = cv2.imread(str(img_path.resolve()))
            if image_bgr is None:
                err_msg = f"Failed to read image (cv2.imread returned None) from path: {img_path.resolve()} at index {idx}."
                logger.error(err_msg)
                raise RuntimeError(err_msg) # Critical error for map-style dataset

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            processed_image: Union[torch.Tensor, np.ndarray] = image_rgb
            if self.transform:
                processed_image = self.transform(image_rgb) # Apply transformations

            return processed_image, img_path
        
        except (cv2.error, OSError, IOError, ValueError, RuntimeError) as e:
            # Raise with context; these are expected I/O/decoding failures.
            logger.error(
                "Critical image processing error for %s at index %d: %s",
                img_path.resolve(),
                idx,
                e,
                exc_info=False,
            )
            raise RuntimeError(
                f"Error processing image {img_path.resolve()} at index {idx}. Original error: {e}"
            ) from e
        except Exception as e:
            # Preserve stack for unknown failures to aid investigation.
            logger.exception(
                "Unexpected error processing image %s at index %d: %s",
                img_path.resolve(),
                idx,
                e,
            )
            raise RuntimeError(
                f"Unexpected error processing image {img_path.resolve()} at index {idx}. Original error: {e}"
            ) from e 