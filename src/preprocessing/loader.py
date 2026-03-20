from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from ..configs.config import DATA_DIR, IMAGE_SIZE, RANDOM_STATE, TEST_SIZE
from .preprocessor import preprocess_image


def _image_sort_key(path: Path) -> tuple[int, int]:
    subject_id = int(path.parent.name.lstrip("sS"))
    image_id = int(path.stem)
    return subject_id, image_id


def load_orl_dataset(
    data_dir: str | Path = DATA_DIR,
    image_size: tuple[int, int] | None = IMAGE_SIZE,
    normalize: bool = True,
    flatten: bool = True,
    return_paths: bool = False,
):
    """Load the ORL/AT&T face dataset from the local data folder."""
    base_path = Path(data_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {base_path}")

    image_paths = sorted(base_path.glob("s*/*.pgm"), key=_image_sort_key)
    if not image_paths:
        raise ValueError(f"No PGM images were found under: {base_path}")

    samples: list[np.ndarray] = []
    labels: list[int] = []

    for image_path in image_paths:
        with Image.open(image_path) as image:
            processed = preprocess_image(
                image,
                image_size=image_size,
                normalize=normalize,
                flatten=flatten,
            )
        samples.append(processed)
        labels.append(int(image_path.parent.name.lstrip("sS")))

    X = np.stack(samples, axis=0)
    y = np.asarray(labels, dtype=int)

    if return_paths:
        return X, y, image_paths
    return X, y


def load_and_split(
    data_dir: str | Path = DATA_DIR,
    image_size: tuple[int, int] | None = IMAGE_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    normalize: bool = True,
    flatten: bool = True,
    stratify: bool = True,
):
    """Load the dataset and return a train/test split."""
    X, y = load_orl_dataset(
        data_dir=data_dir,
        image_size=image_size,
        normalize=normalize,
        flatten=flatten,
    )
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )
