from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..configs.config import DATA_DIR, EXTENDED_YALE_B_DIR, IMAGE_SIZE, LFW_DATA_DIR, ORL_DATA_DIR
from ..preprocessing.loader import load_dataset


@dataclass
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
    data_dir: str | Path | None = None,
    image_size: tuple[int, int] | None = IMAGE_SIZE,
    normalize: bool = True,
    flatten: bool = True,
    dataset_name: str = "orl",
    **dataset_kwargs,
) -> FaceDataset:
    normalized_name = dataset_name.strip().lower().replace("-", "_").replace(" ", "_")
    default_source_dirs = {
        "orl": ORL_DATA_DIR,
        "att_faces": ORL_DATA_DIR,
        "att": ORL_DATA_DIR,
        "extended_yale_b": EXTENDED_YALE_B_DIR,
        "extended_yale": EXTENDED_YALE_B_DIR,
        "cropped_yale": EXTENDED_YALE_B_DIR,
        "lfw": LFW_DATA_DIR,
    }
    source_dir = str(data_dir or default_source_dirs.get(normalized_name, DATA_DIR))
    X, y = load_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        image_size=image_size,
        normalize=normalize,
        flatten=flatten,
        **dataset_kwargs,
    )
    return FaceDataset(
        data=X,
        target=y,
        image_size=image_size or IMAGE_SIZE,
        source_dir=source_dir,
    )
