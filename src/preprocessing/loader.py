from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from ..configs.config import (
    DATA_DIR,
    EXTENDED_YALE_B_DIR,
    IMAGE_SIZE,
    LFW_DATA_DIR,
    ORL_DATA_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from .preprocessor import preprocess_image


def _image_sort_key(path: Path) -> tuple[int, int]:
    subject_id = int(path.parent.name.lstrip("sS"))
    image_id = int(path.stem)
    return subject_id, image_id


def _normalize_dataset_name(dataset_name: str) -> str:
    return dataset_name.strip().lower().replace("-", "_").replace(" ", "_")


def _resolve_orl_dir(data_dir: str | Path) -> Path:
    base_path = Path(data_dir)
    candidates = [base_path, base_path.parent, base_path / "orl"]
    for candidate in candidates:
        if any(candidate.glob("s*/*.pgm")):
            return candidate
    parent_orl = base_path.parent / "orl"
    if any(parent_orl.glob("s*/*.pgm")):
        return parent_orl
    if (base_path / "orl").exists():
        return base_path / "orl"
    return base_path


def _resolve_extended_yale_b_dir(data_dir: str | Path) -> Path:
    base_path = Path(data_dir)
    candidates = [
        base_path,
        base_path / "CroppedYale",
        base_path / "extended_yale_b",
        base_path / "extended_yale_b" / "CroppedYale",
    ]
    for candidate in candidates:
        if any(candidate.glob("yaleB*/*.pgm")):
            return candidate
    if (base_path / "CroppedYale").exists():
        return base_path / "CroppedYale"
    if (base_path / "extended_yale_b" / "CroppedYale").exists():
        return base_path / "extended_yale_b" / "CroppedYale"
    return base_path


def _resolve_lfw_dir(data_dir: str | Path) -> Path:
    base_path = Path(data_dir)
    candidates = [
        base_path,
        base_path / "lfw",
        base_path / "lfw" / "lfw",
    ]
    for candidate in candidates:
        if any(candidate.glob("*/*.jpg")):
            return candidate
    if (base_path / "lfw" / "lfw").exists():
        return base_path / "lfw" / "lfw"
    if (base_path / "lfw").exists():
        return base_path / "lfw"
    return base_path


def _load_image_dataset(
    image_paths: list[Path],
    labels: list[int],
    image_size: tuple[int, int] | None,
    normalize: bool,
    flatten: bool,
    return_paths: bool,
    face_detection: str | None = None,
    face_padding_ratio: float = 0.25,
    face_crop_fallback: str = "original",
    face_square_crop: bool = True,
    face_scale_factor: float = 1.1,
    face_min_neighbors: int = 5,
    face_min_size: tuple[int, int] = (30, 30),
):
    if not image_paths:
        raise ValueError("No image files were found for the requested dataset.")

    samples: list[np.ndarray] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            processed = preprocess_image(
                image,
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
            )
        samples.append(processed)

    X = np.stack(samples, axis=0)
    y = np.asarray(labels, dtype=int)

    if return_paths:
        return X, y, image_paths
    return X, y


def load_orl_dataset(
    data_dir: str | Path = ORL_DATA_DIR,
    image_size: tuple[int, int] | None = IMAGE_SIZE,
    normalize: bool = True,
    flatten: bool = True,
    return_paths: bool = False,
    face_detection: str | None = None,
    face_padding_ratio: float = 0.25,
    face_crop_fallback: str = "original",
    face_square_crop: bool = True,
    face_scale_factor: float = 1.1,
    face_min_neighbors: int = 5,
    face_min_size: tuple[int, int] = (30, 30),
):
    """Load the ORL/AT&T face dataset from the local data folder."""
    base_path = _resolve_orl_dir(data_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {base_path}")

    image_paths = sorted(base_path.glob("s*/*.pgm"), key=_image_sort_key)
    if not image_paths:
        raise ValueError(f"No PGM images were found under: {base_path}")

    labels = [int(image_path.parent.name.lstrip("sS")) for image_path in image_paths]
    return _load_image_dataset(
        image_paths=image_paths,
        labels=labels,
        image_size=image_size,
        normalize=normalize,
        flatten=flatten,
        return_paths=return_paths,
        face_detection=face_detection,
        face_padding_ratio=face_padding_ratio,
        face_crop_fallback=face_crop_fallback,
        face_square_crop=face_square_crop,
        face_scale_factor=face_scale_factor,
        face_min_neighbors=face_min_neighbors,
        face_min_size=face_min_size,
    )


def _extended_yale_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.parent.name)
    subject_id = int(match.group(1)) if match else 0
    return subject_id, path.stem


def load_extended_yale_b_dataset(
    data_dir: str | Path = EXTENDED_YALE_B_DIR,
    image_size: tuple[int, int] | None = IMAGE_SIZE,
    normalize: bool = True,
    flatten: bool = True,
    return_paths: bool = False,
    include_ambient: bool = True,
    face_detection: str | None = None,
    face_padding_ratio: float = 0.25,
    face_crop_fallback: str = "original",
    face_square_crop: bool = True,
    face_scale_factor: float = 1.1,
    face_min_neighbors: int = 5,
    face_min_size: tuple[int, int] = (30, 30),
):
    """Load the local Extended Yale B / Cropped Yale dataset."""
    base_path = _resolve_extended_yale_b_dir(data_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {base_path}")

    image_paths = sorted(base_path.glob("yaleB*/*.pgm"), key=_extended_yale_sort_key)
    if not include_ambient:
        image_paths = [path for path in image_paths if "Ambient" not in path.name]
    if not image_paths:
        raise ValueError(f"No PGM images were found under: {base_path}")

    labels = []
    for image_path in image_paths:
        match = re.search(r"(\d+)", image_path.parent.name)
        if match is None:
            raise ValueError(f"Could not parse subject id from: {image_path.parent.name}")
        labels.append(int(match.group(1)))

    return _load_image_dataset(
        image_paths=image_paths,
        labels=labels,
        image_size=image_size,
        normalize=normalize,
        flatten=flatten,
        return_paths=return_paths,
        face_detection=face_detection,
        face_padding_ratio=face_padding_ratio,
        face_crop_fallback=face_crop_fallback,
        face_square_crop=face_square_crop,
        face_scale_factor=face_scale_factor,
        face_min_neighbors=face_min_neighbors,
        face_min_size=face_min_size,
    )


def _lfw_sort_key(path: Path) -> tuple[str, str]:
    return path.parent.name.lower(), path.stem.lower()


def load_lfw_dataset(
    data_dir: str | Path = LFW_DATA_DIR,
    image_size: tuple[int, int] | None = IMAGE_SIZE,
    normalize: bool = True,
    flatten: bool = True,
    return_paths: bool = False,
    min_images_per_subject: int = 5,
    max_people: int | None = None,
    face_detection: str | None = None,
    face_padding_ratio: float = 0.25,
    face_crop_fallback: str = "original",
    face_square_crop: bool = True,
    face_scale_factor: float = 1.1,
    face_min_neighbors: int = 5,
    face_min_size: tuple[int, int] = (30, 30),
):
    """Load the local LFW raw dataset from extracted JPG files."""
    base_path = _resolve_lfw_dir(data_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {base_path}")

    image_paths = sorted(base_path.glob("*/*.jpg"), key=_lfw_sort_key)
    if not image_paths:
        raise ValueError(f"No JPG images were found under: {base_path}")

    counts = Counter(path.parent.name for path in image_paths)
    if min_images_per_subject > 1:
        image_paths = [path for path in image_paths if counts[path.parent.name] >= min_images_per_subject]

    if max_people is not None:
        filtered_counts = Counter(path.parent.name for path in image_paths)
        ranked_people = sorted(
            filtered_counts.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )
        allowed_people = {name for name, _ in ranked_people[:max_people]}
        image_paths = [path for path in image_paths if path.parent.name in allowed_people]

    if not image_paths:
        raise ValueError("No LFW images remained after applying the subject filters.")

    subject_names = sorted({path.parent.name for path in image_paths}, key=str.lower)
    label_by_name = {name: index for index, name in enumerate(subject_names)}
    labels = [label_by_name[path.parent.name] for path in image_paths]

    return _load_image_dataset(
        image_paths=image_paths,
        labels=labels,
        image_size=image_size,
        normalize=normalize,
        flatten=flatten,
        return_paths=return_paths,
        face_detection=face_detection,
        face_padding_ratio=face_padding_ratio,
        face_crop_fallback=face_crop_fallback,
        face_square_crop=face_square_crop,
        face_scale_factor=face_scale_factor,
        face_min_neighbors=face_min_neighbors,
        face_min_size=face_min_size,
    )


def load_dataset(
    dataset_name: str = "orl",
    data_dir: str | Path | None = None,
    image_size: tuple[int, int] | None = IMAGE_SIZE,
    normalize: bool = True,
    flatten: bool = True,
    return_paths: bool = False,
    **dataset_kwargs,
):
    normalized = _normalize_dataset_name(dataset_name)
    if normalized in {"orl", "att_faces", "att"}:
        return load_orl_dataset(
            data_dir=data_dir or ORL_DATA_DIR,
            image_size=image_size,
            normalize=normalize,
            flatten=flatten,
            return_paths=return_paths,
        )
    if normalized in {"extended_yale_b", "extended_yale", "cropped_yale"}:
        return load_extended_yale_b_dataset(
            data_dir=data_dir or EXTENDED_YALE_B_DIR,
            image_size=image_size,
            normalize=normalize,
            flatten=flatten,
            return_paths=return_paths,
            **dataset_kwargs,
        )
    if normalized == "lfw":
        return load_lfw_dataset(
            data_dir=data_dir or LFW_DATA_DIR,
            image_size=image_size,
            normalize=normalize,
            flatten=flatten,
            return_paths=return_paths,
            **dataset_kwargs,
        )
    raise ValueError(f"Unsupported dataset name: {dataset_name}")


def load_and_split(
    data_dir: str | Path | None = None,
    image_size: tuple[int, int] | None = IMAGE_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    normalize: bool = True,
    flatten: bool = True,
    stratify: bool = True,
    dataset_name: str = "orl",
    **dataset_kwargs,
):
    """Load the dataset and return a train/test split."""
    X, y = load_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        image_size=image_size,
        normalize=normalize,
        flatten=flatten,
        **dataset_kwargs,
    )
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )
