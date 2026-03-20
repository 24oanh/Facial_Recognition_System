from __future__ import annotations

import numpy as np

from ..configs.config import SVM_DEGREE, SVM_MAX_ITER, SVM_TOL
from .pca_knn import PCAKNNPipeline
from .pca_svm import PCASVMPipeline


def train_pca_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_components: int,
    k: int = 5,
    metric: str = "euclidean",
) -> PCAKNNPipeline:
    pipeline = PCAKNNPipeline(n_components=n_components, k=k, metric=metric)
    return pipeline.fit(X_train, y_train)


def train_pca_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_components: int,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    degree: int = SVM_DEGREE,
    decision_function_shape: str = "ovo",
    tol: float = SVM_TOL,
    max_iter: int = SVM_MAX_ITER,
) -> PCASVMPipeline:
    pipeline = PCASVMPipeline(
        n_components=n_components,
        C=C,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        decision_function_shape=decision_function_shape,
        tol=tol,
        max_iter=max_iter,
    )
    return pipeline.fit(X_train, y_train)
