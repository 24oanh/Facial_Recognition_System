from __future__ import annotations

import numpy as np
from sklearn.svm import SVC


class SVMClassifier:
    """Small wrapper around scikit-learn's SVC for the PCA+SVM pipeline."""

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        degree: int = 3,
        decision_function_shape: str = "ovr",
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.decision_function_shape = decision_function_shape
        self.model_: SVC | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.model_ = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            decision_function_shape=self.decision_function_shape,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model_.predict(np.asarray(X, dtype=np.float64))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        self._check_is_fitted()
        return float(self.model_.score(np.asarray(X, dtype=np.float64), np.asarray(y)))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self.model_.decision_function(np.asarray(X, dtype=np.float64))

    @property
    def support_vectors_(self) -> np.ndarray:
        self._check_is_fitted()
        return self.model_.support_vectors_

    def _check_is_fitted(self) -> None:
        if self.model_ is None:
            raise ValueError("SVM has not been fitted yet.")
