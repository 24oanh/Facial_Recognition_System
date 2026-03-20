from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _resolve_component_matrix(components: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    components = np.asarray(components)
    expected_size = image_shape[0] * image_shape[1]

    if components.ndim != 2:
        raise ValueError("components must be a 2D array.")

    if components.shape[1] == expected_size:
        return components
    if components.shape[0] == expected_size:
        return components.T
    raise ValueError("components shape does not match the provided image shape.")


def plot_eigenfaces(
    components: np.ndarray,
    image_shape: tuple[int, int] = (112, 92),
    n_faces: int = 10,
):
    matrix = _resolve_component_matrix(components, image_shape)
    n_faces = min(n_faces, matrix.shape[0])
    n_cols = min(5, n_faces)
    n_rows = math.ceil(n_faces / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for idx in range(n_faces):
        axes[idx].imshow(matrix[idx].reshape(image_shape), cmap="gray")
        axes[idx].set_title(f"PC {idx + 1}")
        axes[idx].axis("off")

    for idx in range(n_faces, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Eigenfaces", fontsize=14)
    fig.tight_layout()
    return fig


def plot_explained_variance(explained_variance_ratio: np.ndarray):
    ratios = np.asarray(explained_variance_ratio, dtype=float)
    cumulative = np.cumsum(ratios)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(ratios) + 1), ratios, marker="o", label="Individual")
    ax.plot(range(1, len(cumulative) + 1), cumulative, marker="s", label="Cumulative")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("Explained Variance by PCA Components")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion_matrix(confusion_matrix_data, labels: list[str] | None = None):
    matrix = np.asarray(confusion_matrix_data)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_model_comparison(df, metric: str = "Accuracy"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df[metric].plot(kind="bar", ax=ax, color="#2f6db2")
    ax.set_ylabel(metric)
    ax.set_title(f"Model Comparison by {metric}")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_sample_images(
    images: np.ndarray,
    labels: np.ndarray | None = None,
    image_shape: tuple[int, int] = (112, 92),
    n_samples: int = 10,
):
    images = np.asarray(images)
    n_samples = min(n_samples, images.shape[0])
    n_cols = min(5, n_samples)
    n_rows = math.ceil(n_samples / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for idx in range(n_samples):
        axes[idx].imshow(images[idx].reshape(image_shape), cmap="gray")
        if labels is not None:
            axes[idx].set_title(f"Label {labels[idx]}")
        axes[idx].axis("off")

    for idx in range(n_samples, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Sample Images", fontsize=14)
    fig.tight_layout()
    return fig
