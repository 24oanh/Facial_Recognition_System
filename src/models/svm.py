from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from ..configs.config import SVM_DEGREE, SVM_MAX_ITER, SVM_TOL

_EPSILON = 1e-8


def _resolve_gamma(gamma: str | float, X: np.ndarray) -> float:
    if isinstance(gamma, str):
        if gamma == "scale":
            variance = float(np.var(X))
            if variance <= 0.0:
                return 1.0 / X.shape[1]
            return 1.0 / (X.shape[1] * variance)
        if gamma == "auto":
            return 1.0 / X.shape[1]
        raise ValueError("gamma must be 'scale', 'auto', or a positive float.")

    gamma = float(gamma)
    if gamma <= 0.0:
        raise ValueError("gamma must be positive.")
    return gamma


def _compute_kernel(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str,
    gamma: float,
    degree: int,
    coef0: float,
) -> np.ndarray:
    if kernel == "linear":
        return X @ Y.T

    if kernel == "poly":
        return (gamma * (X @ Y.T) + coef0) ** degree

    if kernel == "rbf":
        X_sq = np.sum(X**2, axis=1, keepdims=True)
        Y_sq = np.sum(Y**2, axis=1, keepdims=True).T
        distances = np.maximum(X_sq + Y_sq - 2.0 * (X @ Y.T), 0.0)
        return np.exp(-gamma * distances)

    raise ValueError(f"Unsupported kernel '{kernel}'.")


@dataclass(slots=True)
class _BinarySVMModel:
    positive_label_: object
    negative_label_: object | None
    support_vectors_: np.ndarray
    dual_coef_: np.ndarray
    intercept_: float
    support_indices_: np.ndarray
    gamma_: float
    kernel_: str
    degree_: int
    coef0_: float

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.support_vectors_.size == 0:
            return np.full(X.shape[0], self.intercept_, dtype=np.float64)

        kernel_matrix = _compute_kernel(
            X,
            self.support_vectors_,
            kernel=self.kernel_,
            gamma=self.gamma_,
            degree=self.degree_,
            coef0=self.coef0_,
        )
        return kernel_matrix @ self.dual_coef_ + self.intercept_


class SVMClassifier:
    """Kernel SVM from scratch using dual optimization and multiclass OVO/OVR wrappers."""

    SUPPORTED_KERNELS = {"linear", "poly", "rbf"}

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        degree: int = SVM_DEGREE,
        decision_function_shape: str = "ovo",
        tol: float = SVM_TOL,
        max_iter: int = SVM_MAX_ITER,
        coef0: float = 1.0,
        class_weight: str | None = "balanced",
    ):
        if C <= 0.0:
            raise ValueError("C must be positive.")
        if kernel not in self.SUPPORTED_KERNELS:
            raise ValueError(
                f"Unsupported kernel '{kernel}'. Choose from {sorted(self.SUPPORTED_KERNELS)}."
            )
        if degree <= 0:
            raise ValueError("degree must be a positive integer.")
        if tol <= 0.0:
            raise ValueError("tol must be positive.")
        if max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if decision_function_shape not in {"ovr", "ovo"}:
            raise ValueError("decision_function_shape must be 'ovr' or 'ovo'.")
        if class_weight not in {None, "balanced"}:
            raise ValueError("class_weight must be None or 'balanced'.")

        self.C = float(C)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = int(degree)
        self.decision_function_shape = decision_function_shape
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.coef0 = float(coef0)
        self.class_weight = class_weight

        self.classes_: np.ndarray | None = None
        self.models_: list[_BinarySVMModel] = []
        self.n_features_in_: int | None = None
        self.gamma_: float | None = None
        self.n_iter_: list[int] | None = None
        self.class_to_index_: dict[object, int] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("SVM expects a 2D training matrix.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("SVM requires at least two classes.")

        self.classes_ = classes
        self.class_to_index_ = {label: index for index, label in enumerate(classes)}
        self.n_features_in_ = X.shape[1]
        self.gamma_ = _resolve_gamma(self.gamma, X)
        self.models_ = []
        self.n_iter_ = []

        if classes.size == 2:
            binary_targets = np.where(y == classes[1], 1.0, -1.0)
            model, n_iter = self._fit_binary_classifier(
                X,
                binary_targets,
                positive_label=classes[1],
                negative_label=classes[0],
            )
            self.models_.append(model)
            self.n_iter_.append(n_iter)
            return self

        if self.decision_function_shape == "ovr":
            for class_label in classes:
                binary_targets = np.where(y == class_label, 1.0, -1.0)
                model, n_iter = self._fit_binary_classifier(
                    X,
                    binary_targets,
                    positive_label=class_label,
                    negative_label=None,
                )
                self.models_.append(model)
                self.n_iter_.append(n_iter)
            return self

        for positive_index in range(1, classes.size):
            positive_label = classes[positive_index]
            for negative_index in range(positive_index):
                negative_label = classes[negative_index]
                pair_mask = (y == positive_label) | (y == negative_label)
                pair_targets = np.where(y[pair_mask] == positive_label, 1.0, -1.0)
                model, n_iter = self._fit_binary_classifier(
                    X[pair_mask],
                    pair_targets,
                    positive_label=positive_label,
                    negative_label=negative_label,
                )
                self.models_.append(model)
                self.n_iter_.append(n_iter)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("predict expects a 2D matrix.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, received {X.shape[1]}."
            )

        if self.classes_.size == 2:
            scores = self.models_[0].decision_function(X)
            return np.where(scores >= 0.0, self.classes_[1], self.classes_[0])

        if self.decision_function_shape == "ovo":
            votes, margins = self._collect_ovo_votes(X)
            predictions = np.empty(X.shape[0], dtype=self.classes_.dtype)
            for row_index in range(X.shape[0]):
                candidate_indices = np.flatnonzero(votes[row_index] == np.max(votes[row_index]))
                if candidate_indices.size == 1:
                    predictions[row_index] = self.classes_[candidate_indices[0]]
                    continue

                best_index = candidate_indices[np.argmax(margins[row_index, candidate_indices])]
                predictions[row_index] = self.classes_[best_index]
            return predictions

        scores = np.column_stack([m.decision_function(X) for m in self.models_])
        class_indices = np.argmax(scores, axis=1)
        return self.classes_[class_indices]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        y = np.asarray(y)
        return float(np.mean(predictions == y))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("decision_function expects a 2D matrix.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, received {X.shape[1]}."
            )

        if len(self.models_) == 1:
            return self.models_[0].decision_function(X)

        if self.decision_function_shape == "ovo":
            _, margins = self._collect_ovo_votes(X)
            return margins

        return np.column_stack([model.decision_function(X) for model in self.models_])

    @property
    def support_vectors_(self) -> np.ndarray:
        self._check_is_fitted()
        support_blocks = [model.support_vectors_ for model in self.models_ if model.support_vectors_.size]
        if not support_blocks:
            return np.empty((0, self.n_features_in_), dtype=np.float64)
        return np.vstack(support_blocks)

    def _fit_binary_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        positive_label: object,
        negative_label: object | None,
    ) -> tuple[_BinarySVMModel, int]:
        n_samples = X.shape[0]
        upper_bounds = self._compute_class_bounds(y)

        kernel_matrix = _compute_kernel(
            X,
            X,
            kernel=self.kernel,
            gamma=self.gamma_,
            degree=self.degree,
            coef0=self.coef0,
        )
        q_matrix = np.outer(y, y) * kernel_matrix
        ones = np.ones(n_samples, dtype=np.float64)

        def objective(alpha: np.ndarray) -> float:
            return 0.5 * alpha @ q_matrix @ alpha - alpha @ ones

        def gradient(alpha: np.ndarray) -> np.ndarray:
            return q_matrix @ alpha - ones

        result = minimize(
            objective,
            x0=np.zeros(n_samples, dtype=np.float64),
            jac=gradient,
            method="SLSQP",
            bounds=Bounds(np.zeros(n_samples, dtype=np.float64), upper_bounds),
            constraints=[
                LinearConstraint(y.reshape(1, -1), lb=np.array([0.0]), ub=np.array([0.0]))
            ],
            options={
                "maxiter": self.max_iter,
                "ftol": self.tol,
                "disp": False,
            },
        )

        alphas = np.clip(result.x, 0.0, upper_bounds)

        support_mask = alphas > _EPSILON
        support_indices = np.flatnonzero(support_mask)
        dual_coef = alphas[support_mask] * y[support_mask]

        if support_indices.size == 0:
            intercept = 0.0
        else:
            margin_mask = support_mask & (alphas < upper_bounds - _EPSILON)
            bias_indices = np.flatnonzero(margin_mask)
            if bias_indices.size == 0:
                bias_indices = support_indices

            intercept_values = []
            for index in bias_indices:
                intercept_values.append(
                    y[index] - np.dot(kernel_matrix[index, support_mask], dual_coef)
                )
            intercept = float(np.mean(intercept_values))

        model = _BinarySVMModel(
            positive_label_=positive_label,
            negative_label_=negative_label,
            support_vectors_=X[support_mask],
            dual_coef_=dual_coef,
            intercept_=intercept,
            support_indices_=support_indices,
            gamma_=float(self.gamma_),
            kernel_=self.kernel,
            degree_=self.degree,
            coef0_=self.coef0,
        )
        return model, int(result.nit)

    def _check_is_fitted(self) -> None:
        if self.classes_ is None or not self.models_:
            raise ValueError("SVM has not been fitted yet.")

    def _compute_class_bounds(self, y: np.ndarray) -> np.ndarray:
        if self.class_weight is None:
            return np.full(y.shape[0], self.C, dtype=np.float64)

        positive_mask = y > 0
        negative_mask = ~positive_mask
        n_positive = max(int(np.sum(positive_mask)), 1)
        n_negative = max(int(np.sum(negative_mask)), 1)

        positive_bound = self.C * y.shape[0] / (2.0 * n_positive)
        negative_bound = self.C * y.shape[0] / (2.0 * n_negative)

        bounds = np.empty(y.shape[0], dtype=np.float64)
        bounds[positive_mask] = positive_bound
        bounds[negative_mask] = negative_bound
        return bounds

    def _collect_ovo_votes(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        votes = np.zeros((X.shape[0], self.classes_.size), dtype=np.float64)
        margins = np.zeros_like(votes)

        for model in self.models_:
            decision_values = model.decision_function(X)
            positive_index = self.class_to_index_[model.positive_label_]
            negative_index = self.class_to_index_[model.negative_label_]
            positive_mask = decision_values >= 0.0

            votes[positive_mask, positive_index] += 1.0
            votes[~positive_mask, negative_index] += 1.0
            margins[:, positive_index] += decision_values
            margins[:, negative_index] -= decision_values

        return votes, margins
