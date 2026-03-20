from __future__ import annotations

import numpy as np


class PCAScratch:
    """Principal Component Analysis implemented with NumPy only."""

    def __init__(self, n_components: int):
        if n_components <= 0:
            raise ValueError("n_components must be a positive integer.")
        self.n_components = int(n_components)
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCAScratch":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("PCA expects a 2D array of shape (n_samples, n_features).")

        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError("PCA requires at least two samples.")

        max_components = min(n_samples, n_features)
        if self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} exceeds the maximum valid value "
                f"of {max_components} for input shape {X.shape}."
            )

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        covariance_small = X_centered @ X_centered.T / (n_samples - 1)
        eigenvalues_small, eigenvectors_small = np.linalg.eigh(covariance_small)

        order = np.argsort(eigenvalues_small)[::-1]
        eigenvalues = np.maximum(eigenvalues_small[order], 0.0)
        eigenvectors_small = eigenvectors_small[:, order]

        components = X_centered.T @ eigenvectors_small
        norms = np.linalg.norm(components, axis=0, keepdims=True)
        norms[norms == 0.0] = 1.0
        components = components / norms

        self.components_ = components[:, : self.n_components]
        self.explained_variance_ = eigenvalues[: self.n_components]

        total_variance = float(np.sum(eigenvalues))
        if total_variance > 0.0:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        else:
            self.explained_variance_ratio_ = np.zeros(self.n_components, dtype=np.float64)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("transform expects a 2D array.")
        return (X - self.mean_) @ self.components_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X_transformed = np.asarray(X_transformed, dtype=np.float64)
        return X_transformed @ self.components_.T + self.mean_

    def _check_is_fitted(self) -> None:
        if self.components_ is None or self.mean_ is None:
            raise ValueError("PCA has not been fitted yet.")


PCA_scratch = PCAScratch
