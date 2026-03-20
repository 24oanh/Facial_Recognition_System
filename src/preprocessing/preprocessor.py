from __future__ import annotations

from typing import Any
from typing import Iterable

import numpy as np
from PIL import Image

from .face_detection import FaceBBox, crop_face_by_bbox, detect_and_crop_face


def _to_grayscale_array(image: Image.Image | np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        array = np.mean(array, axis=2)
    if array.ndim != 2:
        raise ValueError("Expected a 2D grayscale image after preprocessing.")
    return array.astype(np.float32, copy=False)


def _resize_grayscale_array(array: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    resized = Image.fromarray(array.astype(np.float32), mode="F")
    resized = resized.resize(tuple(image_size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def preprocess_image(
    image: Image.Image | np.ndarray,
    image_size: tuple[int, int] | None = None,
    normalize: bool = True,
    flatten: bool = True,
    face_detection: str | None = None,
    face_bbox: FaceBBox | None = None,
    face_padding_ratio: float = 0.25,
    face_crop_fallback: str = "original",
    face_square_crop: bool = True,
    face_scale_factor: float = 1.1,
    face_min_neighbors: int = 5,
    face_min_size: tuple[int, int] = (30, 30),
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray | None, dict[str, Any]]:
    """Convert an image into a PCA-ready array."""
    processing_metadata = {
        "face_detection_enabled": False,
        "face_detector": None,
        "face_detected": False,
        "face_crop_fallback": face_crop_fallback,
        "face_fallback_used": False,
        "face_bbox": None,
    }

    image_to_process: Image.Image | np.ndarray | None = image
    if face_bbox is not None:
        image_to_process, expanded_bbox = crop_face_by_bbox(
            image=image,
            bbox=face_bbox,
            padding_ratio=face_padding_ratio,
            square=face_square_crop,
        )
        processing_metadata = {
            "face_detection_enabled": True,
            "face_detector": face_detection or "precomputed",
            "face_detected": True,
            "face_crop_fallback": face_crop_fallback,
            "face_fallback_used": False,
            "face_bbox": expanded_bbox,
        }
    elif face_detection is not None:
        image_to_process, processing_metadata = detect_and_crop_face(
            image=image,
            detector=face_detection,
            padding_ratio=face_padding_ratio,
            square=face_square_crop,
            fallback=face_crop_fallback,
            scale_factor=face_scale_factor,
            min_neighbors=face_min_neighbors,
            min_size=face_min_size,
        )
        if image_to_process is None:
            if return_metadata:
                return None, processing_metadata
            raise ValueError("No face detected and face_crop_fallback='skip'.")

    array = _to_grayscale_array(image_to_process)

    if image_size is not None:
        array = _resize_grayscale_array(array, tuple(image_size))

    if normalize and array.max(initial=0.0) > 1.0:
        array = array / 255.0

    if flatten:
        array = array.reshape(-1)

    if return_metadata:
        return array, processing_metadata
    return array


def preprocess_batch(
    images: Iterable[Image.Image | np.ndarray],
    image_size: tuple[int, int] | None = None,
    normalize: bool = True,
    flatten: bool = True,
    face_detection: str | None = None,
    face_padding_ratio: float = 0.25,
    face_crop_fallback: str = "original",
    face_square_crop: bool = True,
    face_scale_factor: float = 1.1,
    face_min_neighbors: int = 5,
    face_min_size: tuple[int, int] = (30, 30),
) -> np.ndarray:
    """Apply the same preprocessing steps to a batch of images."""
    processed: list[np.ndarray] = []
    for image in images:
        processed_image = preprocess_image(
            image,
            image_size=image_size,
            normalize=normalize,
            flatten=flatten,
            face_detection=face_detection,
            face_bbox=None,
            face_padding_ratio=face_padding_ratio,
            face_crop_fallback=face_crop_fallback,
            face_square_crop=face_square_crop,
            face_scale_factor=face_scale_factor,
            face_min_neighbors=face_min_neighbors,
            face_min_size=face_min_size,
        )
        if processed_image is None:
            raise ValueError(
                "A face was not detected for one of the images and face_crop_fallback='skip'."
            )
        processed.append(processed_image)
    if not processed:
        raise ValueError("No images were provided for preprocessing.")
    return np.stack(processed, axis=0)
