from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - dependency failure is environment-specific
    cv2 = None

try:
    import torch
    from facenet_pytorch import MTCNN
except ModuleNotFoundError:  # pragma: no cover - dependency failure is environment-specific
    torch = None
    MTCNN = None


FaceBBox = tuple[int, int, int, int]


def _as_array(image: Image.Image | np.ndarray) -> np.ndarray:
    return np.asarray(image)


def _to_uint8_grayscale(image: Image.Image | np.ndarray) -> np.ndarray:
    array = _as_array(image)
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = array[:, :, :3]
        array = np.mean(array, axis=2)
    if array.ndim != 2:
        raise ValueError("Expected a 2D or 3D image for face detection.")
    if array.dtype != np.uint8:
        max_value = float(np.max(array)) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _to_rgb_pil_image(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    array = _as_array(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    elif array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected a 2D grayscale or 3D RGB image for MTCNN.")

    if array.dtype != np.uint8:
        max_value = float(np.max(array)) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _clip_bbox_to_image_shape(
    bbox: FaceBBox,
    image_shape: tuple[int, int],
) -> FaceBBox | None:
    image_height, image_width = image_shape
    x, y, width, height = bbox

    left = max(0, int(round(x)))
    top = max(0, int(round(y)))
    right = min(image_width, int(round(x + width)))
    bottom = min(image_height, int(round(y + height)))

    clipped_width = right - left
    clipped_height = bottom - top
    if clipped_width <= 0 or clipped_height <= 0:
        return None
    return left, top, clipped_width, clipped_height


@lru_cache(maxsize=1)
def _get_haar_cascade() -> Any:
    if cv2 is None:
        raise ImportError(
            "opencv-python is required for Haar-cascade face detection."
        )
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade from: {cascade_path}")
    return detector


@lru_cache(maxsize=1)
def _get_mtcnn_detector() -> Any:
    if MTCNN is None:
        raise ImportError(
            "facenet-pytorch is required for MTCNN face detection."
        )
    device = "cpu"
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        device = "cuda:0"
    return MTCNN(
        image_size=None,
        post_process=False,
        select_largest=True,
        keep_all=True,
        device=device,
    )


def detect_largest_face_bbox(
    image: Image.Image | np.ndarray,
    detector: str = "haar",
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: tuple[int, int] = (30, 30),
) -> FaceBBox | None:
    if detector == "haar":
        classifier = _get_haar_cascade()
        grayscale = _to_uint8_grayscale(image)
        faces = classifier.detectMultiScale(
            grayscale,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=tuple(min_size),
        )
        if len(faces) == 0:
            return None
        x, y, width, height = max(faces, key=lambda bbox: int(bbox[2]) * int(bbox[3]))
        return _clip_bbox_to_image_shape(
            (int(x), int(y), int(width), int(height)),
            grayscale.shape,
        )

    if detector == "mtcnn":
        model = _get_mtcnn_detector()
        rgb_image = _to_rgb_pil_image(image)
        boxes, _ = model.detect(rgb_image)
        if boxes is None or len(boxes) == 0:
            return None
        x1, y1, x2, y2 = max(
            boxes,
            key=lambda bbox: max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1])),
        )
        x = int(round(float(x1)))
        y = int(round(float(y1)))
        width = int(round(float(x2 - x1)))
        height = int(round(float(y2 - y1)))
        if width <= 0 or height <= 0:
            return None
        return _clip_bbox_to_image_shape((x, y, width, height), rgb_image.size[::-1])

    raise ValueError(f"Unsupported face detector: {detector}")


def _expand_bbox(
    bbox: FaceBBox,
    image_shape: tuple[int, int],
    padding_ratio: float = 0.0,
    square: bool = False,
) -> FaceBBox:
    image_height, image_width = image_shape
    x, y, width, height = bbox
    center_x = x + width / 2.0
    center_y = y + height / 2.0

    target_width = width * (1.0 + 2.0 * padding_ratio)
    target_height = height * (1.0 + 2.0 * padding_ratio)
    if square:
        side = max(target_width, target_height)
        target_width = side
        target_height = side

    left = int(round(center_x - target_width / 2.0))
    top = int(round(center_y - target_height / 2.0))
    right = int(round(center_x + target_width / 2.0))
    bottom = int(round(center_y + target_height / 2.0))

    left = max(0, left)
    top = max(0, top)
    right = min(image_width, right)
    bottom = min(image_height, bottom)

    if right <= left or bottom <= top:
        raise ValueError("Expanded face bounding box is empty.")
    return left, top, right - left, bottom - top


def _crop_bbox(
    image: Image.Image | np.ndarray,
    bbox: FaceBBox,
) -> np.ndarray:
    array = _as_array(image)
    x, y, width, height = bbox
    return array[y : y + height, x : x + width]


def crop_face_by_bbox(
    image: Image.Image | np.ndarray,
    bbox: FaceBBox,
    padding_ratio: float = 0.0,
    square: bool = False,
) -> tuple[np.ndarray, FaceBBox]:
    image_array = _as_array(image)
    image_shape = image_array.shape[:2]
    clipped_bbox = _clip_bbox_to_image_shape(bbox, image_shape)
    if clipped_bbox is None:
        raise ValueError("Bounding box is outside the image bounds.")
    expanded_bbox = _expand_bbox(
        bbox=clipped_bbox,
        image_shape=image_shape,
        padding_ratio=padding_ratio,
        square=square,
    )
    return _crop_bbox(image, expanded_bbox), expanded_bbox


def detect_and_crop_face(
    image: Image.Image | np.ndarray,
    detector: str = "haar",
    padding_ratio: float = 0.0,
    square: bool = False,
    fallback: str = "original",
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: tuple[int, int] = (30, 30),
) -> tuple[np.ndarray | Image.Image | None, dict[str, Any]]:
    if fallback not in {"original", "skip"}:
        raise ValueError("fallback must be either 'original' or 'skip'.")

    image_array = _as_array(image)
    image_shape = image_array.shape[:2]
    bbox = detect_largest_face_bbox(
        image=image,
        detector=detector,
        scale_factor=scale_factor,
        min_neighbors=min_neighbors,
        min_size=min_size,
    )
    metadata = {
        "face_detection_enabled": True,
        "face_detector": detector,
        "face_detected": bbox is not None,
        "face_crop_fallback": fallback,
        "face_fallback_used": False,
        "face_bbox": None,
    }

    if bbox is None:
        metadata["face_fallback_used"] = fallback == "original"
        if fallback == "skip":
            return None, metadata
        return image, metadata

    cropped_image, expanded_bbox = crop_face_by_bbox(
        bbox=bbox,
        image=image,
        padding_ratio=padding_ratio,
        square=square,
    )
    metadata["face_bbox"] = expanded_bbox
    return cropped_image, metadata
