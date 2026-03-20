from __future__ import annotations

import numpy as np


class KNNScratch:
    """Simple KNN classifier with NumPy distance computations."""

    SUPPORTED_METRICS = {"euclidean", "manhattan", "cosine"}

    def __init__(self, k: int = 5, metric: str = "euclidean"):
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric '{metric}'. "
                f"Choose from {sorted(self.SUPPORTED_METRICS)}."
            )
        self.k = int(k)
        self.metric = metric
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNScratch":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("KNN expects a 2D training matrix.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if X.shape[0] < self.k:
            raise ValueError("k cannot be larger than the number of training samples.")

        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        distances = self._compute_distances(np.asarray(X, dtype=np.float64))
        neighbor_indices = np.argsort(distances, axis=1)[:, : self.k]
        predictions = [
            self._majority_vote(self.y_train[indices]) for indices in neighbor_indices
        ]
        return np.asarray(predictions)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        y = np.asarray(y)
        return float(np.mean(predictions == y))

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError("predict expects a 2D matrix.")

        if self.metric == "euclidean":
            X_sq = np.sum(X**2, axis=1, keepdims=True)
            X_train_sq = np.sum(self.X_train**2, axis=1, keepdims=True).T
            dist_sq = np.maximum(X_sq + X_train_sq - 2 * (X @ self.X_train.T), 0.0)
            return np.sqrt(dist_sq)

        if self.metric == "manhattan":
            return np.sum(np.abs(X[:, None, :] - self.X_train[None, :, :]), axis=2)

        X_norm = np.linalg.norm(X, axis=1, keepdims=True)
        train_norm = np.linalg.norm(self.X_train, axis=1, keepdims=True).T
        denominator = np.clip(X_norm * train_norm, 1e-12, None)
        cosine_similarity = (X @ self.X_train.T) / denominator
        return 1.0 - cosine_similarity

    @staticmethod
    def _majority_vote(labels: np.ndarray):
        unique, counts = np.unique(labels, return_counts=True)
        return unique[np.argmax(counts)]

    def _check_is_fitted(self) -> None:
        if self.X_train is None or self.y_train is None:
            raise ValueError("KNN has not been fitted yet.")


KNN_scratch = KNNScratch
