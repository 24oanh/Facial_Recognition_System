from __future__ import annotations

import numpy as np


def evaluate_pipeline(pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate any fitted pipeline exposing an `evaluate` method."""
    if not hasattr(pipeline, "evaluate"):
        raise TypeError("The provided pipeline does not implement `evaluate`.")
    return pipeline.evaluate(X_test, y_test)
