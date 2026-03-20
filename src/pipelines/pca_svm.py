from __future__ import annotations

from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ..features.pca import PCAScratch
from ..models.svm import SVMClassifier


class PCASVMPipeline:
    def __init__(
        self,
        n_components: int,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        degree: int = 3,
    ):
        self.n_components = n_components
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.pca = PCAScratch(n_components=n_components)
        self.svm = SVMClassifier(C=C, kernel=kernel, gamma=gamma, degree=degree)
        self.train_time_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PCASVMPipeline":
        start = perf_counter()
        embeddings = self.pca.fit_transform(X)
        self.svm.fit(embeddings, y)
        self.train_time_ = perf_counter() - start
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        embeddings = self.transform(X)
        return self.svm.predict(embeddings)

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
    def load(cls, path: str | Path) -> "PCASVMPipeline":
        return joblib.load(path)
