from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image


def _to_grayscale_array(image: Image.Image | np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        array = np.mean(array, axis=2)
    if array.ndim != 2:
        raise ValueError("Expected a 2D grayscale image after preprocessing.")
    return array.astype(np.float32, copy=False)


def preprocess_image(
    image: Image.Image | np.ndarray,
    image_size: tuple[int, int] | None = None,
    normalize: bool = True,
    flatten: bool = True,
) -> np.ndarray:
    """Convert an image into a PCA-ready array."""
    array = _to_grayscale_array(image)

    if image_size is not None:
        resized = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
        resized = resized.resize(tuple(image_size), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32)

    if normalize and array.max(initial=0.0) > 1.0:
        array = array / 255.0

    if flatten:
        array = array.reshape(-1)

    return array


def preprocess_batch(
    images: Iterable[Image.Image | np.ndarray],
    image_size: tuple[int, int] | None = None,
    normalize: bool = True,
    flatten: bool = True,
) -> np.ndarray:
    """Apply the same preprocessing steps to a batch of images."""
    processed = [
        preprocess_image(
            image,
            image_size=image_size,
            normalize=normalize,
            flatten=flatten,
        )
        for image in images
    ]
    if not processed:
        raise ValueError("No images were provided for preprocessing.")
    return np.stack(processed, axis=0)
