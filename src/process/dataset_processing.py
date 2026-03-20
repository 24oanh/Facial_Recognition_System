from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import csv
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from ..configs import config as config_module
from ..preprocessing.face_detection import detect_largest_face_bbox
from ..preprocessing.preprocessor import preprocess_image


PROJECT_ROOT = Path(
    getattr(
        config_module,
        "PROJECT_ROOT",
        Path(__file__).resolve().parents[2],
    )
)
DATA_DIR = Path(
    getattr(
        config_module,
        "DATA_DIR",
        PROJECT_ROOT / "data" / "raw",
    )
)
PROCESSED_DIR = Path(
    getattr(
        config_module,
        "PROCESSED_DIR",
        PROJECT_ROOT / "data" / "processed",
    )
)
ORL_DATA_DIR = Path(
    getattr(
        config_module,
        "ORL_DATA_DIR",
        DATA_DIR / "ORL",
    )
)
EXTENDED_YALE_B_DIR = Path(
    getattr(
        config_module,
        "EXTENDED_YALE_B_DIR",
        DATA_DIR / "CroppedYale",
    )
)
LFW_DATA_DIR = Path(
    getattr(
        config_module,
        "LFW_DATA_DIR",
        DATA_DIR / "lfw",
    )
)
IMAGE_SIZE = tuple(
    getattr(
        config_module,
        "IMAGE_SIZE",
        (92, 112),
    )
)
IMAGE_SHAPE = tuple(
    getattr(
        config_module,
        "IMAGE_SHAPE",
        (IMAGE_SIZE[1], IMAGE_SIZE[0]),
    )
)
RANDOM_STATE = int(getattr(config_module, "RANDOM_STATE", 42))
TEST_SIZE = float(getattr(config_module, "TEST_SIZE", 0.2))


@dataclass(slots=True)
class FaceSample:
    path: Path
    subject_name: str
    sample_name: str
    face_bbox: tuple[int, int, int, int] | None = None
    face_area_ratio: float | None = None


PROCESSING_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "orl": {
        "balanced": {
            "min_images_per_subject": 10,
            "max_images_per_subject": 10,
            "balance_subjects": True,
            "target_images_per_subject": 10,
            "face_detection": "mtcnn",
            "face_crop_fallback": "skip",
            "min_face_area_ratio": 0.08,
        },
        "many_people_many_images": {
            "min_images_per_subject": 8,
            "max_images_per_subject": 10,
            "face_detection": "mtcnn",
            "face_crop_fallback": "skip",
            "min_face_area_ratio": 0.08,
        },
        "many_images_few_people": {
            "min_images_per_subject": 10,
            "max_images_per_subject": 10,
            "max_subjects": 20,
            "subject_selection": "original",
            "face_detection": "mtcnn",
            "face_crop_fallback": "skip",
            "min_face_area_ratio": 0.08,
        },
    },
    "extended_yale_b": {
        "balanced": {
            "min_images_per_subject": 50,
            "max_images_per_subject": 50,
            "balance_subjects": True,
            "target_images_per_subject": 50,
            "include_ambient": False,
            "face_detection": "mtcnn",
            "face_crop_fallback": "skip",
            "min_face_area_ratio": 0.08,
        },
        "many_people_many_images": {
            "min_images_per_subject": 50,
            "max_images_per_subject": 64,
            "include_ambient": False,
            "face_detection": "mtcnn",
            "face_crop_fallback": "skip",
            "min_face_area_ratio": 0.08,
        },
        "many_images_few_people": {
            "min_images_per_subject": 59,
            "max_images_per_subject": 59,
            "max_subjects": 20,
            "subject_selection": "most_images",
            "include_ambient": False,
            "face_detection": "mtcnn",
            "face_crop_fallback": "skip",
            "min_face_area_ratio": 0.08,
        },
    },
    "lfw": {
        "balanced": {
            "min_images_per_subject": 20,
            "max_images_per_subject": 20,
            "balance_subjects": True,
            "target_images_per_subject": 20,
            "face_detection": "mtcnn",
            "face_padding_ratio": 0.25,
            "face_crop_fallback": "skip",
            "min_face_area_ratio": 0.08,
        },
        "many_people_many_images": {
            "min_images_per_subject": 25,
            "max_images_per_subject": 25,
            "max_subjects": 40,
            "subject_selection": "most_images",
            "face_detection": "mtcnn",
            "face_padding_ratio": 0.25,
            "face_crop_fallback": "skip",
            "min_face_area_ratio": 0.08,
        },
        "many_images_few_people": {
            "min_images_per_subject": 10,
            "max_images_per_subject": 10,
            "max_subjects": 100,
            "subject_selection": "most_images",
            "face_detection": "mtcnn",
            "face_padding_ratio": 0.25,
            "face_crop_fallback": "skip",
            "min_face_area_ratio": 0.08,
        },
        "many_people_few_images": {
            "min_images_per_subject": 10,
            "max_images_per_subject": 10,
            "max_subjects": 100,
            "subject_selection": "most_images",
            "face_detection": "mtcnn",
            "face_padding_ratio": 0.25,
            "face_crop_fallback": "skip",
            "min_face_area_ratio": 0.08,
        },
    },
}


def _normalize_dataset_name(dataset_name: str) -> str:
    return dataset_name.strip().lower().replace("-", "_").replace(" ", "_")


def _resolve_orl_dir(raw_dir: str | Path | None) -> Path:
    base_dir = Path(raw_dir or ORL_DATA_DIR)
    candidates = [
        base_dir,
        base_dir.parent / "ORL",
        base_dir.parent / "orl",
        base_dir.parent,
    ]
    for candidate in candidates:
        if any(candidate.glob("s*/*.pgm")):
            return candidate
    return base_dir


def _resolve_extended_yale_b_dir(raw_dir: str | Path | None) -> Path:
    base_dir = Path(raw_dir or EXTENDED_YALE_B_DIR)
    candidates = [
        base_dir,
        base_dir.parent / "CroppedYale",
        base_dir.parent / "extended_yale_b" / "CroppedYale",
        base_dir.parent / "extended_yale_b",
        base_dir.parent,
    ]
    for candidate in candidates:
        if any(candidate.glob("yaleB*/*.pgm")):
            return candidate
    return base_dir


def _resolve_lfw_dir(raw_dir: str | Path | None) -> Path:
    base_dir = Path(raw_dir or LFW_DATA_DIR)
    candidates = [
        base_dir,
        base_dir.parent / "lfw",
        base_dir / "lfw",
        base_dir.parent,
    ]
    for candidate in candidates:
        if any(candidate.glob("*/*.jpg")):
            return candidate
    return base_dir


def _orl_sample_sort_key(path: Path) -> tuple[int, int]:
    subject_id = int(path.parent.name.lstrip("sS"))
    image_id = int(path.stem)
    return subject_id, image_id


def _extended_yale_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.parent.name)
    subject_id = int(match.group(1)) if match else 0
    return subject_id, path.stem


def _lfw_sort_key(path: Path) -> tuple[str, str]:
    return path.parent.name.lower(), path.stem.lower()


def _collect_orl_samples(raw_dir: str | Path | None = None) -> list[FaceSample]:
    base_dir = _resolve_orl_dir(raw_dir)
    image_paths = sorted(base_dir.glob("s*/*.pgm"), key=_orl_sample_sort_key)
    if not image_paths:
        raise ValueError(f"No ORL PGM images were found under: {base_dir}")
    return [
        FaceSample(
            path=path,
            subject_name=path.parent.name,
            sample_name=path.stem,
        )
        for path in image_paths
    ]


def _collect_extended_yale_b_samples(
    raw_dir: str | Path | None = None,
    include_ambient: bool = False,
) -> list[FaceSample]:
    base_dir = _resolve_extended_yale_b_dir(raw_dir)
    image_paths = sorted(base_dir.glob("yaleB*/*.pgm"), key=_extended_yale_sort_key)
    if not include_ambient:
        image_paths = [path for path in image_paths if "ambient" not in path.stem.lower()]
    if not image_paths:
        raise ValueError(f"No Extended Yale B PGM images were found under: {base_dir}")
    return [
        FaceSample(
            path=path,
            subject_name=path.parent.name,
            sample_name=path.stem,
        )
        for path in image_paths
    ]


def _collect_lfw_samples(raw_dir: str | Path | None = None) -> list[FaceSample]:
    base_dir = _resolve_lfw_dir(raw_dir)
    image_paths = sorted(base_dir.glob("*/*.jpg"), key=_lfw_sort_key)
    if not image_paths:
        raise ValueError(f"No LFW JPG images were found under: {base_dir}")
    return [
        FaceSample(
            path=path,
            subject_name=path.parent.name,
            sample_name=path.stem,
        )
        for path in image_paths
    ]


def _collect_samples(
    dataset_name: str,
    raw_dir: str | Path | None = None,
    include_ambient: bool = False,
) -> tuple[list[FaceSample], Path]:
    normalized = _normalize_dataset_name(dataset_name)
    if normalized in {"orl", "att_faces", "att"}:
        resolved_dir = _resolve_orl_dir(raw_dir)
        return _collect_orl_samples(resolved_dir), resolved_dir
    if normalized in {"extended_yale_b", "extended_yale", "cropped_yale"}:
        resolved_dir = _resolve_extended_yale_b_dir(raw_dir)
        return _collect_extended_yale_b_samples(
            resolved_dir,
            include_ambient=include_ambient,
        ), resolved_dir
    if normalized == "lfw":
        resolved_dir = _resolve_lfw_dir(raw_dir)
        return _collect_lfw_samples(resolved_dir), resolved_dir
    raise ValueError(f"Unsupported dataset name: {dataset_name}")


def _rank_subjects_for_selection(
    grouped: dict[str, list[FaceSample]],
    subject_selection: str = "original",
) -> list[tuple[str, list[FaceSample]]]:
    items = list(grouped.items())
    if subject_selection == "original":
        return items
    if subject_selection == "most_images":
        indexed_items = list(enumerate(items))
        ranked = sorted(
            indexed_items,
            key=lambda item: (-len(item[1][1]), item[0]),
        )
        return [item[1] for item in ranked]
    raise ValueError(f"Unsupported subject_selection: {subject_selection}")


def _filter_samples_by_subject_count(
    samples: list[FaceSample],
    min_images_per_subject: int = 1,
    max_images_per_subject: int | None = None,
    max_subjects: int | None = None,
    subject_selection: str = "original",
    balance_subjects: bool = False,
    target_images_per_subject: int | None = None,
) -> tuple[list[FaceSample], dict[str, Any]]:
    if min_images_per_subject < 1:
        raise ValueError("min_images_per_subject must be at least 1.")
    if max_images_per_subject is not None and max_images_per_subject < min_images_per_subject:
        raise ValueError("max_images_per_subject must be >= min_images_per_subject.")
    if max_subjects is not None and max_subjects < 1:
        raise ValueError("max_subjects must be at least 1.")
    if target_images_per_subject is not None and target_images_per_subject < 1:
        raise ValueError("target_images_per_subject must be at least 1.")
    if (
        target_images_per_subject is not None
        and balance_subjects
        and target_images_per_subject < min_images_per_subject
    ):
        raise ValueError("target_images_per_subject must be >= min_images_per_subject when balancing.")

    grouped: dict[str, list[FaceSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.subject_name].append(sample)

    eligible_grouped: dict[str, list[FaceSample]] = {}
    dropped_subjects_below_min: dict[str, int] = {}
    for subject_name, subject_samples in grouped.items():
        original_count = len(subject_samples)
        if original_count < min_images_per_subject:
            dropped_subjects_below_min[subject_name] = original_count
            continue
        eligible_grouped[subject_name] = subject_samples

    ranked_subjects = _rank_subjects_for_selection(
        eligible_grouped,
        subject_selection=subject_selection,
    )
    dropped_subjects_by_subject_limit: dict[str, int] = {}
    if max_subjects is not None and len(ranked_subjects) > max_subjects:
        kept_subjects = ranked_subjects[:max_subjects]
        dropped_subjects_by_subject_limit = {
            subject_name: len(subject_samples)
            for subject_name, subject_samples in ranked_subjects[max_subjects:]
        }
    else:
        kept_subjects = ranked_subjects

    pre_balance_counts = {
        subject_name: min(len(subject_samples), max_images_per_subject or len(subject_samples))
        for subject_name, subject_samples in kept_subjects
    }
    balance_images_per_subject: int | None = None
    if balance_subjects and pre_balance_counts:
        balance_images_per_subject = min(pre_balance_counts.values())
        if target_images_per_subject is not None:
            balance_images_per_subject = min(balance_images_per_subject, target_images_per_subject)

    filtered_samples: list[FaceSample] = []
    truncated_subjects_by_max_images: dict[str, dict[str, int]] = {}
    truncated_subjects_by_balancing: dict[str, dict[str, int]] = {}

    for subject_name, subject_samples in kept_subjects:
        keep_count = len(subject_samples)
        if max_images_per_subject is not None and keep_count > max_images_per_subject:
            truncated_subjects_by_max_images[subject_name] = {
                "original_count": keep_count,
                "kept_count": max_images_per_subject,
            }
            keep_count = max_images_per_subject
        if balance_images_per_subject is not None and keep_count > balance_images_per_subject:
            truncated_subjects_by_balancing[subject_name] = {
                "original_count": keep_count,
                "kept_count": balance_images_per_subject,
            }
            keep_count = balance_images_per_subject
        subject_samples = subject_samples[:keep_count]
        filtered_samples.extend(subject_samples)

    dropped_subjects = {
        **dropped_subjects_below_min,
        **dropped_subjects_by_subject_limit,
    }
    truncated_subjects = {
        **truncated_subjects_by_max_images,
        **truncated_subjects_by_balancing,
    }
    stats = {
        "subjects_before_filter": len(grouped),
        "subjects_after_filter": len({sample.subject_name for sample in filtered_samples}),
        "samples_before_filter": len(samples),
        "samples_after_filter": len(filtered_samples),
        "dropped_subjects": dropped_subjects,
        "dropped_subjects_below_min": dropped_subjects_below_min,
        "dropped_subjects_by_subject_limit": dropped_subjects_by_subject_limit,
        "truncated_subjects": truncated_subjects,
        "truncated_subjects_by_max_images": truncated_subjects_by_max_images,
        "truncated_subjects_by_balancing": truncated_subjects_by_balancing,
        "max_subjects": max_subjects,
        "subject_selection": subject_selection,
        "balance_subjects": balance_subjects,
        "target_images_per_subject": target_images_per_subject,
        "balance_images_per_subject": balance_images_per_subject,
    }
    return filtered_samples, stats


def _validate_samples_with_face_detection(
    samples: list[FaceSample],
    face_detection: str | None = None,
    min_face_area_ratio: float = 0.0,
    face_scale_factor: float = 1.1,
    face_min_neighbors: int = 5,
    face_min_size: tuple[int, int] = (30, 30),
) -> tuple[list[FaceSample], dict[str, Any]]:
    if face_detection is None:
        return samples, {
            "face_validation_enabled": False,
            "face_validation_samples_before": len(samples),
            "face_validation_samples_after": len(samples),
            "face_validation_valid_samples": len(samples),
            "face_validation_no_face_samples": 0,
            "face_validation_small_face_samples": 0,
            "face_validation_unreadable_samples": 0,
            "face_validation_detector": None,
            "face_validation_min_face_area_ratio": min_face_area_ratio,
        }

    valid_samples: list[FaceSample] = []
    no_face_files: list[str] = []
    small_face_files: list[str] = []
    unreadable_files: list[str] = []

    for sample in samples:
        try:
            with Image.open(sample.path) as image:
                bbox = detect_largest_face_bbox(
                    image=image,
                    detector=face_detection,
                    scale_factor=face_scale_factor,
                    min_neighbors=face_min_neighbors,
                    min_size=face_min_size,
                )
                image_width, image_height = image.size
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            unreadable_files.append(str(sample.path))
            continue

        if bbox is None:
            no_face_files.append(str(sample.path))
            continue

        _, _, bbox_width, bbox_height = bbox
        face_area_ratio = (bbox_width * bbox_height) / float(image_width * image_height)
        if face_area_ratio < min_face_area_ratio:
            small_face_files.append(str(sample.path))
            continue

        valid_samples.append(
            FaceSample(
                path=sample.path,
                subject_name=sample.subject_name,
                sample_name=sample.sample_name,
                face_bbox=bbox,
                face_area_ratio=face_area_ratio,
            )
        )

    return valid_samples, {
        "face_validation_enabled": True,
        "face_validation_samples_before": len(samples),
        "face_validation_samples_after": len(valid_samples),
        "face_validation_valid_samples": len(valid_samples),
        "face_validation_no_face_samples": len(no_face_files),
        "face_validation_small_face_samples": len(small_face_files),
        "face_validation_unreadable_samples": len(unreadable_files),
        "face_validation_no_face_files": no_face_files,
        "face_validation_small_face_files": small_face_files,
        "face_validation_unreadable_files": unreadable_files,
        "face_validation_detector": face_detection,
        "face_validation_min_face_area_ratio": min_face_area_ratio,
    }


def analyze_subject_count_thresholds(
    dataset_name: str,
    raw_dir: str | Path | None = None,
    thresholds: list[int] | None = None,
    include_ambient: bool = False,
) -> list[dict[str, Any]]:
    samples, resolved_raw_dir = _collect_samples(
        dataset_name=dataset_name,
        raw_dir=raw_dir,
        include_ambient=include_ambient,
    )
    grouped: dict[str, list[FaceSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.subject_name].append(sample)

    subject_counts = [len(subject_samples) for subject_samples in grouped.values()]
    if not subject_counts:
        raise ValueError("No subjects found to analyze thresholds.")

    candidate_thresholds = thresholds or sorted(
        {
            2,
            5,
            8,
            10,
            15,
            20,
            30,
            40,
            50,
            60,
            80,
            100,
            min(subject_counts),
            max(subject_counts),
        }
    )

    rows: list[dict[str, Any]] = []
    for threshold in candidate_thresholds:
        eligible_counts = [count for count in subject_counts if count >= threshold]
        rows.append(
            {
                "dataset_name": _normalize_dataset_name(dataset_name),
                "raw_dir": str(resolved_raw_dir),
                "threshold": threshold,
                "subjects_kept": len(eligible_counts),
                "samples_kept_without_cap": int(sum(eligible_counts)),
                "samples_kept_if_balanced": int(len(eligible_counts) * threshold),
                "max_images_available": int(max(eligible_counts)) if eligible_counts else 0,
            }
        )
    return rows


def get_dataset_processing_preset(
    dataset_name: str,
    preset_name: str,
) -> dict[str, Any]:
    normalized_dataset = _normalize_dataset_name(dataset_name)
    normalized_preset = _normalize_dataset_name(preset_name)
    if normalized_dataset not in PROCESSING_PRESETS:
        raise ValueError(f"Unsupported dataset name for presets: {dataset_name}")
    if normalized_preset not in PROCESSING_PRESETS[normalized_dataset]:
        raise ValueError(f"Unsupported preset name: {preset_name}")
    return dict(PROCESSING_PRESETS[normalized_dataset][normalized_preset])


def _load_and_preprocess_samples(
    samples: list[FaceSample],
    image_size: tuple[int, int] | None,
    normalize: bool,
    flatten: bool,
    face_detection: str | None = None,
    face_padding_ratio: float = 0.25,
    face_crop_fallback: str = "original",
    face_square_crop: bool = True,
    face_scale_factor: float = 1.1,
    face_min_neighbors: int = 5,
    face_min_size: tuple[int, int] = (30, 30),
    min_face_area_ratio: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    arrays: list[np.ndarray] = []
    valid_samples: list[FaceSample] = []
    unreadable_files: list[str] = []
    skipped_no_face_files: list[str] = []
    face_detected_samples = 0
    face_missed_samples = 0
    face_fallback_used_samples = 0

    for sample in samples:
        try:
            with Image.open(sample.path) as image:
                processed, processing_metadata = preprocess_image(
                    image,
                    image_size=image_size,
                    normalize=normalize,
                    flatten=flatten,
                    face_detection=face_detection if sample.face_bbox is None else None,
                    face_bbox=sample.face_bbox,
                    face_padding_ratio=face_padding_ratio,
                    face_crop_fallback=face_crop_fallback,
                    face_square_crop=face_square_crop,
                    face_scale_factor=face_scale_factor,
                    face_min_neighbors=face_min_neighbors,
                    face_min_size=face_min_size,
                    return_metadata=True,
                )
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            unreadable_files.append(str(sample.path))
            continue

        if processed is None:
            skipped_no_face_files.append(str(sample.path))
            continue
        if processing_metadata["face_detection_enabled"]:
            if processing_metadata["face_detected"]:
                face_detected_samples += 1
            else:
                face_missed_samples += 1
            if processing_metadata["face_fallback_used"]:
                face_fallback_used_samples += 1

        arrays.append(processed)
        valid_samples.append(sample)

    if not arrays:
        raise ValueError("No readable images remained after preprocessing.")

    X = np.stack(arrays, axis=0)
    stats = {
        "readable_samples": len(valid_samples),
        "unreadable_samples": len(unreadable_files),
        "unreadable_files": unreadable_files,
        "skipped_no_face_samples": len(skipped_no_face_files),
        "skipped_no_face_files": skipped_no_face_files,
        "face_detection_enabled": face_detection is not None,
        "face_detector": face_detection,
        "face_detected_samples": face_detected_samples,
        "face_missed_samples": face_missed_samples,
        "face_fallback_used_samples": face_fallback_used_samples,
        "face_crop_fallback": face_crop_fallback,
        "face_padding_ratio": face_padding_ratio,
        "face_square_crop": face_square_crop,
        "face_scale_factor": face_scale_factor,
        "face_min_neighbors": face_min_neighbors,
        "face_min_size": list(face_min_size),
        "min_face_area_ratio": min_face_area_ratio,
    }
    return X, {"samples": valid_samples, "stats": stats}


def create_model_inputs(
    samples: list[FaceSample],
    image_size: tuple[int, int] | None = IMAGE_SIZE,
    normalize: bool = True,
    flatten: bool = True,
    face_detection: str | None = None,
    face_padding_ratio: float = 0.25,
    face_crop_fallback: str = "original",
    face_square_crop: bool = True,
    face_scale_factor: float = 1.1,
    face_min_neighbors: int = 5,
    face_min_size: tuple[int, int] = (30, 30),
    min_face_area_ratio: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    X, payload = _load_and_preprocess_samples(
        samples=samples,
        image_size=image_size,
        normalize=normalize,
        flatten=flatten,
        face_detection=face_detection,
        face_padding_ratio=face_padding_ratio,
        face_crop_fallback=face_crop_fallback,
        face_square_crop=face_square_crop,
        face_scale_factor=face_scale_factor,
        face_min_neighbors=face_min_neighbors,
        face_min_size=face_min_size,
        min_face_area_ratio=min_face_area_ratio,
    )
    valid_samples: list[FaceSample] = payload["samples"]

    label_names = list(dict.fromkeys(sample.subject_name for sample in valid_samples))
    label_mapping = {subject_name: index for index, subject_name in enumerate(label_names)}
    y = np.asarray([label_mapping[sample.subject_name] for sample in valid_samples], dtype=int)

    metadata = {
        "file_paths": [str(sample.path) for sample in valid_samples],
        "sample_names": [sample.sample_name for sample in valid_samples],
        "subject_names": [sample.subject_name for sample in valid_samples],
        "label_names": label_names,
        "label_mapping": label_mapping,
        "processing_stats": payload["stats"],
    }
    return X, y, metadata


def _build_dataset_output_dir(
    dataset_name: str,
    output_root: str | Path,
    min_images_per_subject: int,
    max_images_per_subject: int | None,
    max_subjects: int | None,
    subject_selection: str,
    balance_subjects: bool,
    target_images_per_subject: int | None,
    image_size: tuple[int, int] | None,
    flatten: bool,
    include_ambient: bool,
    face_detection: str | None,
    face_padding_ratio: float,
    face_crop_fallback: str,
    min_face_area_ratio: float,
) -> Path:
    normalized = _normalize_dataset_name(dataset_name)
    width, height = image_size or IMAGE_SIZE
    max_token = "all" if max_images_per_subject is None else str(max_images_per_subject)
    flatten_token = "flat" if flatten else "image"
    config_parts = [
        f"min{min_images_per_subject}_max{max_token}_"
        f"size{width}x{height}_{flatten_token}"
    ]
    if max_subjects is not None:
        config_parts.append(f"subjects{max_subjects}_{subject_selection}")
    if balance_subjects:
        balance_token = "auto" if target_images_per_subject is None else str(target_images_per_subject)
        config_parts.append(f"balanced{balance_token}")
    if normalized in {"extended_yale_b", "extended_yale", "cropped_yale"}:
        config_parts.append("ambient" if include_ambient else "noambient")
    if face_detection is not None:
        padding_token = int(round(face_padding_ratio * 100))
        area_token = int(round(min_face_area_ratio * 100))
        config_parts.append(f"face{face_detection}_pad{padding_token}_area{area_token}_{face_crop_fallback}")
    config_id = "_".join(config_parts)
    return Path(output_root) / normalized / config_id


def _can_stratify_labels(y: np.ndarray, test_size: float) -> bool:
    label_counts = Counter(y.tolist())
    n_classes = len(label_counts)
    if n_classes < 2:
        return False
    if min(label_counts.values()) < 2:
        return False

    n_samples = int(y.shape[0])
    n_test = max(1, int(round(n_samples * test_size)))
    n_train = n_samples - n_test
    return n_test >= n_classes and n_train >= n_classes


def _build_split_indices(
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, bool]:
    indices = np.arange(y.shape[0])
    stratify_labels = y if stratify and _can_stratify_labels(y, test_size) else None
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )
    return train_indices, test_indices, stratify_labels is not None


def _subset_metadata(
    metadata: dict[str, Any],
    indices: np.ndarray | list[int],
) -> dict[str, Any]:
    selected_indices = [int(index) for index in np.asarray(indices, dtype=int).tolist()]
    return {
        "file_paths": [metadata["file_paths"][index] for index in selected_indices],
        "sample_names": [metadata["sample_names"][index] for index in selected_indices],
        "subject_names": [metadata["subject_names"][index] for index in selected_indices],
        "label_names": list(metadata["label_names"]),
        "label_mapping": dict(metadata["label_mapping"]),
    }


def _write_manifest_csv(
    output_path: Path,
    metadata: dict[str, Any],
    y: np.ndarray,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["index", "file_path", "subject_name", "label", "sample_name"],
        )
        writer.writeheader()
        for index, (file_path, subject_name, sample_name, label) in enumerate(
            zip(
                metadata["file_paths"],
                metadata["subject_names"],
                metadata["sample_names"],
                y.tolist(),
            )
        ):
            writer.writerow(
                {
                    "index": index,
                    "file_path": file_path,
                    "subject_name": subject_name,
                    "label": int(label),
                    "sample_name": sample_name,
                }
            )


def save_processed_dataset_bundle(
    X: np.ndarray,
    y: np.ndarray,
    metadata: dict[str, Any],
    summary: dict[str, Any],
    output_dir: str | Path,
    train_indices: np.ndarray | None = None,
    test_indices: np.ndarray | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_indices_array = np.asarray(train_indices if train_indices is not None else [], dtype=int)
    test_indices_array = np.asarray(test_indices if test_indices is not None else [], dtype=int)

    X_train = X[train_indices_array] if train_indices_array.size else np.empty((0, *X.shape[1:]), dtype=X.dtype)
    X_test = X[test_indices_array] if test_indices_array.size else np.empty((0, *X.shape[1:]), dtype=X.dtype)
    y_train = y[train_indices_array] if train_indices_array.size else np.empty((0,), dtype=y.dtype)
    y_test = y[test_indices_array] if test_indices_array.size else np.empty((0,), dtype=y.dtype)

    train_metadata = _subset_metadata(metadata, train_indices_array)
    test_metadata = _subset_metadata(metadata, test_indices_array)

    np.savez_compressed(
        output_path / "inputs.npz",
        X=X,
        y=y,
        train_indices=train_indices_array,
        test_indices=test_indices_array,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        file_paths=np.asarray(metadata["file_paths"], dtype=object),
        sample_names=np.asarray(metadata["sample_names"], dtype=object),
        subject_names=np.asarray(metadata["subject_names"], dtype=object),
        label_names=np.asarray(metadata["label_names"], dtype=object),
        train_file_paths=np.asarray(train_metadata["file_paths"], dtype=object),
        train_sample_names=np.asarray(train_metadata["sample_names"], dtype=object),
        train_subject_names=np.asarray(train_metadata["subject_names"], dtype=object),
        test_file_paths=np.asarray(test_metadata["file_paths"], dtype=object),
        test_sample_names=np.asarray(test_metadata["sample_names"], dtype=object),
        test_subject_names=np.asarray(test_metadata["subject_names"], dtype=object),
        image_size=np.asarray(summary["image_size"], dtype=int),
        image_shape=np.asarray(summary["image_shape"], dtype=int),
    )

    _write_manifest_csv(output_path / "manifest.csv", metadata, y)
    _write_manifest_csv(output_path / "manifest_train.csv", train_metadata, y_train)
    _write_manifest_csv(output_path / "manifest_test.csv", test_metadata, y_test)

    with (output_path / "label_mapping.json").open("w", encoding="utf-8") as json_file:
        json.dump(metadata["label_mapping"], json_file, indent=2, ensure_ascii=False)

    with (output_path / "summary.json").open("w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, indent=2, ensure_ascii=False)

    return output_path


def load_processed_dataset_bundle(processed_dir: str | Path) -> dict[str, Any]:
    processed_path = Path(processed_dir)
    with np.load(processed_path / "inputs.npz", allow_pickle=True) as npz_file:
        bundle = {
            "X": npz_file["X"],
            "y": npz_file["y"],
            "train_indices": npz_file["train_indices"],
            "test_indices": npz_file["test_indices"],
            "X_train": npz_file["X_train"],
            "X_test": npz_file["X_test"],
            "y_train": npz_file["y_train"],
            "y_test": npz_file["y_test"],
            "file_paths": npz_file["file_paths"].tolist(),
            "sample_names": npz_file["sample_names"].tolist(),
            "subject_names": npz_file["subject_names"].tolist(),
            "label_names": npz_file["label_names"].tolist(),
            "train_file_paths": npz_file["train_file_paths"].tolist(),
            "train_sample_names": npz_file["train_sample_names"].tolist(),
            "train_subject_names": npz_file["train_subject_names"].tolist(),
            "test_file_paths": npz_file["test_file_paths"].tolist(),
            "test_sample_names": npz_file["test_sample_names"].tolist(),
            "test_subject_names": npz_file["test_subject_names"].tolist(),
            "image_size": tuple(npz_file["image_size"].tolist()),
            "image_shape": tuple(npz_file["image_shape"].tolist()),
        }
    bundle["metadata"] = {
        "file_paths": bundle["file_paths"],
        "sample_names": bundle["sample_names"],
        "subject_names": bundle["subject_names"],
        "label_names": bundle["label_names"],
    }
    bundle["train_metadata"] = {
        "file_paths": bundle["train_file_paths"],
        "sample_names": bundle["train_sample_names"],
        "subject_names": bundle["train_subject_names"],
        "label_names": bundle["label_names"],
    }
    bundle["test_metadata"] = {
        "file_paths": bundle["test_file_paths"],
        "sample_names": bundle["test_sample_names"],
        "subject_names": bundle["test_subject_names"],
        "label_names": bundle["label_names"],
    }

    with (processed_path / "summary.json").open("r", encoding="utf-8") as json_file:
        bundle["summary"] = json.load(json_file)
    return bundle


def process_face_dataset(
    dataset_name: str,
    raw_dir: str | Path | None = None,
    output_root: str | Path = PROCESSED_DIR,
    min_images_per_subject: int = 1,
    max_images_per_subject: int | None = None,
    max_subjects: int | None = None,
    subject_selection: str = "original",
    balance_subjects: bool = False,
    target_images_per_subject: int | None = None,
    image_size: tuple[int, int] | None = IMAGE_SIZE,
    normalize: bool = True,
    flatten: bool = True,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
    include_ambient: bool = False,
    face_detection: str | None = None,
    face_padding_ratio: float = 0.25,
    face_crop_fallback: str = "original",
    face_square_crop: bool = True,
    face_scale_factor: float = 1.1,
    face_min_neighbors: int = 5,
    face_min_size: tuple[int, int] = (30, 30),
    min_face_area_ratio: float = 0.0,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    collected_samples, resolved_raw_dir = _collect_samples(
        dataset_name=dataset_name,
        raw_dir=raw_dir,
        include_ambient=include_ambient,
    )
    face_validated_samples, face_validation_stats = _validate_samples_with_face_detection(
        collected_samples,
        face_detection=face_detection,
        min_face_area_ratio=min_face_area_ratio,
        face_scale_factor=face_scale_factor,
        face_min_neighbors=face_min_neighbors,
        face_min_size=face_min_size,
    )
    filtered_samples, filter_stats = _filter_samples_by_subject_count(
        face_validated_samples,
        min_images_per_subject=min_images_per_subject,
        max_images_per_subject=max_images_per_subject,
        max_subjects=max_subjects,
        subject_selection=subject_selection,
        balance_subjects=balance_subjects,
        target_images_per_subject=target_images_per_subject,
    )
    X, y, metadata = create_model_inputs(
        samples=filtered_samples,
        image_size=image_size,
        normalize=normalize,
        flatten=flatten,
        face_detection=face_detection,
        face_padding_ratio=face_padding_ratio,
        face_crop_fallback=face_crop_fallback,
        face_square_crop=face_square_crop,
        face_scale_factor=face_scale_factor,
        face_min_neighbors=face_min_neighbors,
        face_min_size=face_min_size,
        min_face_area_ratio=min_face_area_ratio,
    )
    train_indices, test_indices, stratify_used = _build_split_indices(
        y=y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    train_metadata = _subset_metadata(metadata, train_indices)
    test_metadata = _subset_metadata(metadata, test_indices)

    label_distribution = Counter(y.tolist())
    summary = {
        "dataset_name": _normalize_dataset_name(dataset_name),
        "raw_dir": str(resolved_raw_dir),
        "min_images_per_subject": min_images_per_subject,
        "max_images_per_subject": max_images_per_subject,
        "max_subjects": max_subjects,
        "subject_selection": subject_selection,
        "balance_subjects": balance_subjects,
        "target_images_per_subject": target_images_per_subject,
        "include_ambient": include_ambient,
        "face_detection": face_detection,
        "face_padding_ratio": face_padding_ratio,
        "face_crop_fallback": face_crop_fallback,
        "face_square_crop": face_square_crop,
        "face_scale_factor": face_scale_factor,
        "face_min_neighbors": face_min_neighbors,
        "face_min_size": list(face_min_size),
        "min_face_area_ratio": min_face_area_ratio,
        "normalize": normalize,
        "flatten": flatten,
        "image_size": list(image_size or IMAGE_SIZE),
        "image_shape": list(IMAGE_SHAPE if image_size is None else (image_size[1], image_size[0])),
        "test_size": test_size,
        "random_state": random_state,
        "stratify_requested": stratify,
        "stratify_used": stratify_used,
        "samples_total": int(X.shape[0]),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "classes_total": len(metadata["label_names"]),
        "feature_shape": list(X.shape),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "label_distribution": {str(label): count for label, count in sorted(label_distribution.items())},
        "face_validation_stats": face_validation_stats,
        "filter_stats": filter_stats,
        "dropped_subject_count": len(filter_stats["dropped_subjects"]),
        "truncated_subject_count": len(filter_stats["truncated_subjects"]),
        "balance_images_per_subject": filter_stats["balance_images_per_subject"],
        "processing_stats": metadata["processing_stats"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    output_dir = _build_dataset_output_dir(
        dataset_name=dataset_name,
        output_root=output_root,
        min_images_per_subject=min_images_per_subject,
        max_images_per_subject=max_images_per_subject,
        max_subjects=max_subjects,
        subject_selection=subject_selection,
        balance_subjects=balance_subjects,
        target_images_per_subject=target_images_per_subject,
        image_size=image_size,
        flatten=flatten,
        include_ambient=include_ambient,
        face_detection=face_detection,
        face_padding_ratio=face_padding_ratio,
        face_crop_fallback=face_crop_fallback,
        min_face_area_ratio=min_face_area_ratio,
    )
    if save_artifacts:
        save_processed_dataset_bundle(
            X=X,
            y=y,
            metadata=metadata,
            summary=summary,
            output_dir=output_dir,
            train_indices=train_indices,
            test_indices=test_indices,
        )

    return {
        "X": X,
        "y": y,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metadata": metadata,
        "train_metadata": train_metadata,
        "test_metadata": test_metadata,
        "summary": summary,
        "output_dir": str(output_dir),
    }


def process_face_dataset_with_preset(
    dataset_name: str,
    preset_name: str,
    **overrides,
) -> dict[str, Any]:
    config = get_dataset_processing_preset(dataset_name, preset_name)
    config.update(overrides)
    return process_face_dataset(dataset_name=dataset_name, **config)


def process_orl_dataset(**kwargs) -> dict[str, Any]:
    return process_face_dataset_with_preset(
        dataset_name="orl",
        preset_name="balanced",
        raw_dir=kwargs.pop("raw_dir", ORL_DATA_DIR),
        **kwargs,
    )


def process_extended_yale_b_dataset(**kwargs) -> dict[str, Any]:
    return process_face_dataset_with_preset(
        dataset_name="extended_yale_b",
        preset_name="many_images_few_people",
        raw_dir=kwargs.pop("raw_dir", EXTENDED_YALE_B_DIR),
        **kwargs,
    )


def process_lfw_dataset(**kwargs) -> dict[str, Any]:
    return process_face_dataset_with_preset(
        dataset_name="lfw",
        preset_name="many_people_many_images",
        raw_dir=kwargs.pop("raw_dir", LFW_DATA_DIR),
        **kwargs,
    )


def process_lfw_many_people_few_images_dataset(**kwargs) -> dict[str, Any]:
    return process_face_dataset_with_preset(
        dataset_name="lfw",
        preset_name=kwargs.pop("preset_name", "many_people_few_images"),
        raw_dir=kwargs.pop("raw_dir", LFW_DATA_DIR),
        **kwargs,
    )


def process_all_face_datasets(
    output_root: str | Path = PROCESSED_DIR,
    dataset_configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    configs = dataset_configs or {
        "orl_balanced": {
            "dataset_name": "orl",
            **get_dataset_processing_preset("orl", "balanced"),
        },
        "lfw_many_people_many_images": {
            "dataset_name": "lfw",
            **get_dataset_processing_preset("lfw", "many_people_many_images"),
        },
        "lfw_many_people_few_images": {
            "dataset_name": "lfw",
            **get_dataset_processing_preset("lfw", "many_people_few_images"),
        },
    }
    results: dict[str, dict[str, Any]] = {}
    for result_name, config in configs.items():
        config = dict(config)
        dataset_name = config.pop("dataset_name", result_name)
        config.setdefault("output_root", output_root)
        results[result_name] = process_face_dataset(dataset_name=dataset_name, **config)
    return results
