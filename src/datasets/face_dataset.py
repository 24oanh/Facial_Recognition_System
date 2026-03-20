from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..configs.config import DATA_DIR, IMAGE_SIZE
from ..preprocessing.loader import load_orl_dataset


@dataclass(slots=True)
class FaceDataset:
    data: np.ndarray
    target: np.ndarray
    image_size: tuple[int, int]
    source_dir: str

    def __len__(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_classes(self) -> int:
        return int(np.unique(self.target).size)

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data, self.target


def load_face_dataset(
    data_dir: str | Path = DATA_DIR,
    image_size: tuple[int, int] | None = IMAGE_SIZE,
    normalize: bool = True,
    flatten: bool = True,
) -> FaceDataset:
    X, y = load_orl_dataset(
        data_dir=data_dir,
        image_size=image_size,
        normalize=normalize,
        flatten=flatten,
    )
    return FaceDataset(
        data=X,
        target=y,
        image_size=image_size or IMAGE_SIZE,
        source_dir=str(data_dir),
    )
