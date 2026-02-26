"""
metrics.py
----------
Tính toán và tổng hợp các chỉ số đánh giá mô hình.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def summarize_results(eval_dict: dict, model_name: str) -> pd.Series:
    """
    Rút trích các chỉ số chính từ kết quả evaluate().

    Parameters:
        eval_dict  : dict trả về từ model.evaluate()
        model_name : tên mô hình (để đặt tên cột)

    Returns:
        pd.Series với các chỉ số: accuracy, macro avg precision/recall/f1, train_time
    """
    report = eval_dict["report"]
    macro = report.get("macro avg", {})
    return pd.Series(
        {
            "Model":      model_name,
            "Accuracy":   round(eval_dict["accuracy"], 4),
            "Precision":  round(macro.get("precision", 0), 4),
            "Recall":     round(macro.get("recall", 0), 4),
            "F1-Score":   round(macro.get("f1-score", 0), 4),
            "Train Time": round(eval_dict.get("train_time", 0), 3),
        }
    )


def compare_models(*eval_tuples) -> pd.DataFrame:
    """
    So sánh nhiều mô hình.

    Usage:
        df = compare_models(
            (knn_eval, "PCA+KNN"),
            (svm_eval, "PCA+SVM"),
        )
    """
    rows = [summarize_results(d, name) for d, name in eval_tuples]
    df = pd.DataFrame(rows).set_index("Model")
    print(df.to_string())
    return df


def save_metrics(df: pd.DataFrame, path: str = "results/metrics/comparison.csv"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"[metrics] Đã lưu: {path}")
