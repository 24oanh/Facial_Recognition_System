from __future__ import annotations

from typing import Any
from typing import Iterable

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency in some environments
    cv2 = None

from .face_detection import FaceBBox, crop_face_by_bbox, detect_and_crop_face

_EPSILON = 1e-6

_PROCESSING_PROFILES: dict[str, dict[str, Any]] = {
    "standard": {
        "percentile_clip": None,
        "target_mean": None,
        "clahe_clip_limit": None,
        "clahe_grid_size": None,
        "unsharp_amount": 0.0,
        "unsharp_sigma": 1.0,
    },
    "orl_enhanced": {
        "percentile_clip": (1.0, 99.0),
        "target_mean": 0.45,
        "clahe_clip_limit": 1.8,
        "clahe_grid_size": (8, 8),
        "unsharp_amount": 0.2,
        "unsharp_sigma": 1.0,
    },
    "yale_b_strong": {
        "percentile_clip": (1.0, 99.5),
        "target_mean": 0.42,
        "clahe_clip_limit": 2.5,
        "clahe_grid_size": (8, 8),
        "unsharp_amount": 0.45,
        "unsharp_sigma": 1.0,
    },
    "custom_aligned": {
        "percentile_clip": (1.0, 99.2),
        "target_mean": 0.48,
        "clahe_clip_limit": 1.6,
        "clahe_grid_size": (8, 8),
        "unsharp_amount": 0.12,
        "unsharp_sigma": 0.9,
    },
}

_QUALITY_GATES: dict[str, dict[str, float]] = {
    "yale_b_strict": {
        "min_std": 0.11,
        "min_dynamic_range": 0.34,
        "min_entropy": 3.6,
        "max_shadow_ratio": 0.72,
        "max_highlight_ratio": 0.72,
        "shadow_threshold": 0.04,
        "highlight_threshold": 0.96,
    },
}


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


def _to_unit_range(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        return array
    if array.max(initial=0.0) > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def _clip_and_rescale_percentiles(
    array: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> tuple[np.ndarray, dict[str, float]]:
    low_value, high_value = np.percentile(array, [low_percentile, high_percentile])
    if high_value - low_value <= _EPSILON:
        return np.zeros_like(array, dtype=np.float32), {
            "clip_low_value": float(low_value),
            "clip_high_value": float(high_value),
            "clip_degenerate": True,
        }

    clipped = np.clip(array, low_value, high_value)
    clipped = (clipped - low_value) / (high_value - low_value)
    return clipped.astype(np.float32, copy=False), {
        "clip_low_value": float(low_value),
        "clip_high_value": float(high_value),
        "clip_degenerate": False,
    }


def _apply_gamma_to_target_mean(
    array: np.ndarray,
    target_mean: float | None,
) -> tuple[np.ndarray, dict[str, float | None]]:
    if target_mean is None:
        return array, {"gamma": None, "mean_before_gamma": float(array.mean())}

    mean_value = float(array.mean())
    if mean_value <= _EPSILON or mean_value >= 1.0 - _EPSILON:
        return array, {"gamma": 1.0, "mean_before_gamma": mean_value}

    gamma = np.log(target_mean) / np.log(mean_value)
    gamma = float(np.clip(gamma, 0.6, 1.8))
    adjusted = np.power(np.clip(array, 0.0, 1.0), gamma, dtype=np.float32)
    return adjusted.astype(np.float32, copy=False), {
        "gamma": gamma,
        "mean_before_gamma": mean_value,
    }


def _apply_clahe(
    array: np.ndarray,
    clip_limit: float | None,
    grid_size: tuple[int, int] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if clip_limit is None or grid_size is None or cv2 is None:
        return array, {
            "clahe_applied": False,
            "clahe_clip_limit": clip_limit,
            "clahe_grid_size": list(grid_size) if grid_size is not None else None,
        }

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(int(v) for v in grid_size),
    )
    uint8_array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    equalized = clahe.apply(uint8_array).astype(np.float32) / 255.0
    return equalized, {
        "clahe_applied": True,
        "clahe_clip_limit": float(clip_limit),
        "clahe_grid_size": [int(grid_size[0]), int(grid_size[1])],
    }


def _apply_unsharp_mask(
    array: np.ndarray,
    amount: float,
    sigma: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if amount <= 0.0 or cv2 is None:
        return array, {
            "unsharp_applied": False,
            "unsharp_amount": float(amount),
            "unsharp_sigma": float(sigma),
        }

    blurred = cv2.GaussianBlur(array, (0, 0), sigmaX=float(sigma))
    sharpened = np.clip(array * (1.0 + amount) - blurred * amount, 0.0, 1.0)
    return sharpened.astype(np.float32, copy=False), {
        "unsharp_applied": True,
        "unsharp_amount": float(amount),
        "unsharp_sigma": float(sigma),
    }


def _normalize_profile_name(processing_profile: str | None) -> str:
    return (processing_profile or "standard").strip().lower().replace("-", "_").replace(" ", "_")


def enhance_grayscale_array(
    array: np.ndarray,
    processing_profile: str | None = "standard",
) -> tuple[np.ndarray, dict[str, Any]]:
    profile_name = _normalize_profile_name(processing_profile)
    if profile_name not in _PROCESSING_PROFILES:
        raise ValueError(f"Unsupported processing_profile: {processing_profile}")

    config = _PROCESSING_PROFILES[profile_name]
    enhanced = _to_unit_range(array)
    metadata: dict[str, Any] = {
        "processing_profile": profile_name,
        "processing_profile_applied": profile_name != "standard",
    }

    percentile_clip = config["percentile_clip"]
    if percentile_clip is not None:
        enhanced, clip_metadata = _clip_and_rescale_percentiles(
            enhanced,
            low_percentile=float(percentile_clip[0]),
            high_percentile=float(percentile_clip[1]),
        )
        metadata.update(
            {
                "clip_percentiles": [float(percentile_clip[0]), float(percentile_clip[1])],
                **clip_metadata,
            }
        )
    else:
        metadata.update(
            {
                "clip_percentiles": None,
                "clip_low_value": None,
                "clip_high_value": None,
                "clip_degenerate": False,
            }
        )

    enhanced, gamma_metadata = _apply_gamma_to_target_mean(
        enhanced,
        target_mean=config["target_mean"],
    )
    metadata.update(gamma_metadata)
    metadata["target_mean"] = config["target_mean"]

    enhanced, clahe_metadata = _apply_clahe(
        enhanced,
        clip_limit=config["clahe_clip_limit"],
        grid_size=config["clahe_grid_size"],
    )
    metadata.update(clahe_metadata)

    enhanced, unsharp_metadata = _apply_unsharp_mask(
        enhanced,
        amount=float(config["unsharp_amount"]),
        sigma=float(config["unsharp_sigma"]),
    )
    metadata.update(unsharp_metadata)

    metadata["mean_after_enhancement"] = float(enhanced.mean())
    metadata["std_after_enhancement"] = float(enhanced.std())
    return np.clip(enhanced, 0.0, 1.0), metadata


def assess_processed_image_quality(
    array: np.ndarray,
    quality_gate: str | None = None,
) -> dict[str, Any]:
    unit_array = _to_unit_range(array)
    flat = unit_array.reshape(-1)
    p01, p99 = np.percentile(flat, [1.0, 99.0])
    hist, _ = np.histogram(flat, bins=32, range=(0.0, 1.0))
    probabilities = hist.astype(np.float64)
    probabilities = probabilities[probabilities > 0.0]
    if probabilities.size:
        probabilities = probabilities / probabilities.sum()
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
    else:
        entropy = 0.0

    metrics = {
        "quality_gate": quality_gate,
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min(initial=0.0)),
        "max": float(flat.max(initial=0.0)),
        "dynamic_range": float(p99 - p01),
        "entropy": entropy,
        "shadow_ratio": float(np.mean(flat <= 0.04)),
        "highlight_ratio": float(np.mean(flat >= 0.96)),
        "rejected": False,
        "rejection_reasons": [],
    }

    if quality_gate is None:
        return metrics

    gate_name = quality_gate.strip().lower().replace("-", "_").replace(" ", "_")
    if gate_name not in _QUALITY_GATES:
        raise ValueError(f"Unsupported quality_gate: {quality_gate}")

    config = _QUALITY_GATES[gate_name]
    shadow_threshold = float(config["shadow_threshold"])
    highlight_threshold = float(config["highlight_threshold"])
    shadow_ratio = float(np.mean(flat <= shadow_threshold))
    highlight_ratio = float(np.mean(flat >= highlight_threshold))
    metrics["quality_gate"] = gate_name
    metrics["shadow_ratio"] = shadow_ratio
    metrics["highlight_ratio"] = highlight_ratio

    rejection_reasons: list[str] = []
    if metrics["std"] < float(config["min_std"]):
        rejection_reasons.append("low_std")
    if metrics["dynamic_range"] < float(config["min_dynamic_range"]):
        rejection_reasons.append("low_dynamic_range")
    if metrics["entropy"] < float(config["min_entropy"]):
        rejection_reasons.append("low_entropy")
    if shadow_ratio > float(config["max_shadow_ratio"]):
        rejection_reasons.append("too_many_shadows")
    if highlight_ratio > float(config["max_highlight_ratio"]):
        rejection_reasons.append("too_many_highlights")

    metrics["rejected"] = bool(rejection_reasons)
    metrics["rejection_reasons"] = rejection_reasons
    return metrics


def preprocess_image(
    image: Image.Image | np.ndarray,
    image_size: tuple[int, int] | None = None,
    normalize: bool = True,
    flatten: bool = True,
    face_detection: str | None = None,
    face_bbox: FaceBBox | None = None,
    face_align: bool = False,
    face_padding_ratio: float = 0.0,
    face_crop_fallback: str = "original",
    face_square_crop: bool = False,
    face_scale_factor: float = 1.1,
    face_min_neighbors: int = 5,
    face_min_size: tuple[int, int] = (30, 30),
    processing_profile: str | None = "standard",
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
        "face_alignment_enabled": bool(face_align and image_size is not None and face_detection is not None),
        "face_aligned": False,
        "face_alignment_method": None,
        "processing_profile": _normalize_profile_name(processing_profile),
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
            "face_alignment_enabled": bool(face_align and image_size is not None and face_detection is not None),
            "face_aligned": False,
            "face_alignment_method": None,
            "processing_profile": _normalize_profile_name(processing_profile),
        }
        if face_align and image_size is not None and face_detection is not None:
            image_to_process, alignment_metadata = detect_and_crop_face(
                image=image_to_process,
                detector=face_detection,
                padding_ratio=0.0,
                square=False,
                fallback="original",
                scale_factor=face_scale_factor,
                min_neighbors=face_min_neighbors,
                min_size=face_min_size,
                align=True,
                alignment_output_size=tuple(image_size),
            )
            processing_metadata["face_alignment_enabled"] = bool(alignment_metadata.get("face_alignment_enabled"))
            processing_metadata["face_aligned"] = bool(alignment_metadata.get("face_aligned"))
            processing_metadata["face_alignment_method"] = alignment_metadata.get("face_alignment_method")
    elif face_detection is not None:
        image_to_process, face_metadata = detect_and_crop_face(
            image=image,
            detector=face_detection,
            padding_ratio=face_padding_ratio,
            square=face_square_crop,
            fallback=face_crop_fallback,
            scale_factor=face_scale_factor,
            min_neighbors=face_min_neighbors,
            min_size=face_min_size,
            align=face_align,
            alignment_output_size=tuple(image_size) if image_size is not None else None,
        )
        processing_metadata.update(face_metadata)
        if image_to_process is None:
            if return_metadata:
                return None, processing_metadata
            raise ValueError("No face detected and face_crop_fallback='skip'.")

    array = _to_grayscale_array(image_to_process)
    array, enhancement_metadata = enhance_grayscale_array(
        array,
        processing_profile=processing_profile,
    )

    if image_size is not None:
        array = _resize_grayscale_array(array, tuple(image_size))
        array = np.clip(array, 0.0, 1.0)

    if not normalize:
        array = array * 255.0

    if flatten:
        array = array.reshape(-1)

    processing_metadata.update(enhancement_metadata)
    if return_metadata:
        return array.astype(np.float32, copy=False), processing_metadata
    return array.astype(np.float32, copy=False)


def preprocess_batch(
    images: Iterable[Image.Image | np.ndarray],
    image_size: tuple[int, int] | None = None,
    normalize: bool = True,
    flatten: bool = True,
    face_detection: str | None = None,
    face_align: bool = False,
    face_padding_ratio: float = 0.0,
    face_crop_fallback: str = "original",
    face_square_crop: bool = False,
    face_scale_factor: float = 1.1,
    face_min_neighbors: int = 5,
    face_min_size: tuple[int, int] = (30, 30),
    processing_profile: str | None = "standard",
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
            face_align=face_align,
            face_padding_ratio=face_padding_ratio,
            face_crop_fallback=face_crop_fallback,
            face_square_crop=face_square_crop,
            face_scale_factor=face_scale_factor,
            face_min_neighbors=face_min_neighbors,
            face_min_size=face_min_size,
            processing_profile=processing_profile,
        )
        if processed_image is None:
            raise ValueError(
                "A face was not detected for one of the images and face_crop_fallback='skip'."
            )
        processed.append(processed_image)
    if not processed:
        raise ValueError("No images were provided for preprocessing.")
    return np.stack(processed, axis=0)
