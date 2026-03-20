import json
import pandas as pd
from pathlib import Path
from typing import Any


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


def _normalize_model_results(*eval_inputs: Any) -> list[tuple[dict, str]]:
    if len(eval_inputs) == 1 and isinstance(eval_inputs[0], dict):
        model_results = eval_inputs[0]
        if model_results and all(isinstance(value, dict) for value in model_results.values()):
            return [(eval_dict, model_name) for model_name, eval_dict in model_results.items()]

    normalized: list[tuple[dict, str]] = []
    for item in eval_inputs:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(
                "compare_models expects either a dict {model_name: eval_dict} "
                "or tuples in the form (eval_dict, model_name)."
            )
        eval_dict, model_name = item
        if not isinstance(eval_dict, dict):
            raise TypeError("Each evaluation result must be a dict returned by model.evaluate().")
        normalized.append((eval_dict, model_name))
    return normalized


def compare_models(*eval_tuples) -> pd.DataFrame:
    """
    So sánh nhiều mô hình.

    Usage:
        df = compare_models(
            {
                "PCA+KNN": knn_eval,
                "PCA+SVM": svm_eval,
            }
        )

        Hoặc:

        df = compare_models(
            (knn_eval, "PCA+KNN"),
            (svm_eval, "PCA+SVM"),
        )
    """
    normalized_results = _normalize_model_results(*eval_tuples)
    rows = [summarize_results(d, name) for d, name in normalized_results]
    df = pd.DataFrame(rows).set_index("Model")
    print(df.to_string())
    return df


def save_metrics(df: pd.DataFrame, path: str = "results/metrics/comparison.csv"):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".json":
        output_path.write_text(
            json.dumps(df.reset_index().to_dict(orient="records"), indent=2),
            encoding="utf-8",
        )
    else:
        df.to_csv(output_path)

    print(f"[metrics] Đã lưu: {output_path}")
