from __future__ import annotations

from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ..features.pca import PCAScratch
from ..models.knn import KNNScratch


def _ensure_2d_samples(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim < 2:
        raise ValueError("Expected input with shape (n_samples, ...).")
    if X.ndim == 2:
        return X
    return X.reshape(X.shape[0], -1)


class PCAKNNPipeline:
    def __init__(self, n_components: int, k: int = 5, metric: str = "euclidean"):
        self.n_components = n_components
        self.k = k
        self.metric = metric
        self.pca = PCAScratch(n_components=n_components)
        self.knn = KNNScratch(k=k, metric=metric)
        self.train_time_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PCAKNNPipeline":
        start = perf_counter()
        embeddings = self.pca.fit_transform(_ensure_2d_samples(X))
        self.knn.fit(embeddings, y)
        self.train_time_ = perf_counter() - start
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(_ensure_2d_samples(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        embeddings = self.transform(X)
        return self.knn.predict(embeddings)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_true = np.asarray(y)
        y_pred = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "report": classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "train_time": float(self.train_time_ or 0.0),
        }

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "PCAKNNPipeline":
        return joblib.load(path)
