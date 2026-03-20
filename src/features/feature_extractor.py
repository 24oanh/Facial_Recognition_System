from __future__ import annotations

import numpy as np

from .pca import PCAScratch


class PCAFeatureExtractor:
    """Thin wrapper around PCA used as a feature extractor."""

    def __init__(self, n_components: int):
        self.pca = PCAScratch(n_components=n_components)

    @property
    def components_(self) -> np.ndarray | None:
        return self.pca.components_

    @property
    def mean_(self) -> np.ndarray | None:
        return self.pca.mean_

    @property
    def explained_variance_ratio_(self) -> np.ndarray | None:
        return self.pca.explained_variance_ratio_

    def fit(self, X: np.ndarray) -> "PCAFeatureExtractor":
        self.pca.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.fit_transform(X)


FeatureExtractor = PCAFeatureExtractor
