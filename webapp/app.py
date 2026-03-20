from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, abort, render_template, send_file


WEBAPP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEBAPP_DIR.parent
TEMPLATES_DIR = WEBAPP_DIR / "templates"
STATICS_DIR = WEBAPP_DIR / "statics"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
SAVED_MODELS_DIR = WEBAPP_DIR / "saved_models"

sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.configs import config as project_config
except Exception:
    project_config = None


app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATICS_DIR),
    static_url_path="/static",
)


DATASET_ORDER = ["orl", "extended_yale_b", "lfw"]
DATASET_LABELS = {
    "orl": "ORL/AT&T",
    "extended_yale_b": "Extended Yale B",
    "lfw": "LFW",
    "overview": "Notebook baseline",
    "custom": "Custom experiment",
}
DATASET_DESCRIPTIONS = {
    "orl": "Tap can bang nho, phu hop de quan sat do on dinh cua PCA va bo phan loai.",
    "extended_yale_b": "Tap anh anh sang phong phu, huu ich de so sanh do ben vung cua embedding PCA.",
    "lfw": "Tap du lieu thuc te, nhieu class va bien thien lon, phu hop de kiem tra kha nang tong quat hoa.",
}
PRESET_LABELS = {
    "balanced": "Balanced",
    "many_people_many_images": "Many people, many images",
    "many_images_few_people": "Many images, few people",
    "baseline": "Notebook baseline",
}
MODEL_LABELS = {
    "pca_knn": "PCA + KNN",
    "pca_svm": "PCA + SVM",
}
MODEL_ACCENTS = {
    "pca_knn": "primary",
    "pca_svm": "success",
}

PROJECT_GOALS = [
    "So sanh hai pipeline PCA + KNN va PCA + SVM tren bai toan nhan dang khuon mat.",
    "Danh gia trade-off giua do chinh xac, thoi gian huan luyen va kha nang mo rong theo tung dataset.",
    "Chuan hoa quy trinh xu ly du lieu, luu bundle trung gian va model artifact de tai lap thi nghiem.",
]
PIPELINE_STEPS = [
    {
        "step": "01",
        "title": "Nap va loc du lieu",
        "description": "Doc ORL, Extended Yale B va LFW; loc subject theo nguong anh va preset phu hop.",
    },
    {
        "step": "02",
        "title": "Tien xu ly anh",
        "description": "Resize, grayscale, normalize, flatten va tuy chon face detection cho cac tap can.",
    },
    {
        "step": "03",
        "title": "Giam chieu bang PCA",
        "description": "Trich xuat vector dac trung bang PCA scratch truoc khi dua vao bo phan loai.",
    },
    {
        "step": "04",
        "title": "Huan luyen va danh gia",
        "description": "Train PCA + KNN / PCA + SVM, sau do so sanh accuracy, F1, train time va artifact sinh ra.",
    },
]
COMPARISON_AXES = [
    "Accuracy theo so chieu PCA.",
    "Confusion matrix o cau hinh tot nhat cua moi mo hinh.",
    "Thoi gian huan luyen va do tre suy luan tren moi mau.",
    "Kiem dinh thong ke cho sai khac ve loi phan loai.",
    "Eigenfaces va explained variance de quan sat mat toan hoc cua PCA.",
]
STRUCTURE_CARDS = [
    {
        "path": "src/features/",
        "detail": "PCA scratch va wrapper trich xuat dac trung.",
    },
    {
        "path": "src/models/",
        "detail": "KNN, SVM va cac wrapper tuong thich import cu.",
    },
    {
        "path": "src/pipelines/",
        "detail": "Hai pipeline PCA -> KNN, PCA -> SVM va helper train/eval.",
    },
    {
        "path": "src/process/",
        "detail": "Tao processed bundle, manifest, label mapping va summary.",
    },
]
REPRO_COMMANDS = [
    {
        "title": "Train nhanh PCA + KNN",
        "command": (
            'python -c "from src.preprocessing import load_and_split; '
            "from src.pipelines import train_pca_knn; "
            "X_train, X_test, y_train, y_test = load_and_split(); "
            "model = train_pca_knn(X_train, y_train, n_components=20, k=3); "
            "print(model.evaluate(X_test, y_test)['accuracy'])\""
        ),
    },
    {
        "title": "Xu ly ORL preset balanced",
        "command": (
            'python -c "from src.process import process_orl_dataset; '
            "bundle = process_orl_dataset(); print(bundle['output_dir'])\""
        ),
    },
    {
        "title": "Xu ly Extended Yale B preset many_images_few_people",
        "command": (
            'python -c "from src.process import process_extended_yale_b_dataset; '
            "bundle = process_extended_yale_b_dataset(); print(bundle['output_dir'])\""
        ),
    },
    {
        "title": "Xu ly LFW preset many_people_many_images",
        "command": (
            'python -c "from src.process import process_lfw_dataset; '
            "bundle = process_lfw_dataset(); print(bundle['output_dir'])\""
        ),
    },
]
DOWNLOAD_ROOTS = {
    "saved-models": SAVED_MODELS_DIR,
    "metrics": METRICS_DIR,
    "processed": PROCESSED_DIR,
}


def _dataset_sort_index(dataset_key: str) -> int:
    try:
        return DATASET_ORDER.index(dataset_key)
    except ValueError:
        return len(DATASET_ORDER)


def _relative_path(path: Path, base: Path) -> str:
    return str(path.relative_to(base)).replace("\\", "/")


def _project_path(path: Path) -> str:
    try:
        return _relative_path(path, PROJECT_ROOT)
    except ValueError:
        return str(path)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _format_datetime(value: datetime | None) -> str:
    if value is None:
        return "N/A"
    return value.strftime("%d/%m/%Y %H:%M")


def _format_number(value: float | int | None, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.2f} MB"
    return f"{size_bytes / (1024**3):.2f} GB"


def _humanize_token(token: str) -> str:
    return token.replace("_", " ").replace("-", " ").strip().title()


def _humanize_preset(preset_key: str) -> str:
    return PRESET_LABELS.get(preset_key, _humanize_token(preset_key))


def _model_key_from_label(label: str) -> str:
    normalized = label.lower()
    if "knn" in normalized:
        return "pca_knn"
    if "svm" in normalized:
        return "pca_svm"
    return normalized.replace("+", "_").replace(" ", "_")


def _parse_dataset_and_preset(slug: str) -> tuple[str, str]:
    normalized = slug.strip("_")
    if not normalized:
        return "overview", "baseline"

    for dataset_key in sorted(DATASET_LABELS, key=len, reverse=True):
        if dataset_key in {"overview", "custom"}:
            continue
        if normalized == dataset_key:
            return dataset_key, "baseline"
        prefix = f"{dataset_key}_"
        if normalized.startswith(prefix):
            return dataset_key, normalized[len(prefix) :]

    return "custom", normalized


def _experiment_title(dataset_key: str, preset_key: str) -> str:
    if dataset_key == "overview":
        return "Notebook baseline"
    if dataset_key == "custom":
        return _humanize_preset(preset_key)
    return f"{DATASET_LABELS[dataset_key]} · {_humanize_preset(preset_key)}"


def _load_metrics_experiments() -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []

    for path in sorted(METRICS_DIR.glob("*.csv")):
        if path.name.startswith("."):
            continue

        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            records: list[dict[str, Any]] = []
            for row in reader:
                label = row.get("Model", "").strip()
                if not label:
                    continue

                records.append(
                    {
                        "model_key": _model_key_from_label(label),
                        "model_label": MODEL_LABELS.get(_model_key_from_label(label), label),
                        "accuracy": float(row.get("Accuracy", 0.0) or 0.0),
                        "precision": float(row.get("Precision", 0.0) or 0.0),
                        "recall": float(row.get("Recall", 0.0) or 0.0),
                        "f1_score": float(row.get("F1-Score", 0.0) or 0.0),
                        "train_time": float(row.get("Train Time", 0.0) or 0.0),
                    }
                )

        if not records:
            continue

        stem = path.stem
        slug = stem.removeprefix("notebook_comparison").strip("_")
        dataset_key, preset_key = _parse_dataset_and_preset(slug)
        ranked_records = sorted(records, key=lambda item: item["accuracy"], reverse=True)
        winner = ranked_records[0]
        runner_up = ranked_records[1] if len(ranked_records) > 1 else None
        modified_at = datetime.fromtimestamp(path.stat().st_mtime)

        experiments.append(
            {
                "name": stem,
                "title": _experiment_title(dataset_key, preset_key),
                "dataset_key": dataset_key,
                "dataset_label": DATASET_LABELS.get(dataset_key, _humanize_token(dataset_key)),
                "preset_key": preset_key,
                "preset_label": _humanize_preset(preset_key),
                "records": ranked_records,
                "winner": winner,
                "runner_up": runner_up,
                "accuracy_gap": winner["accuracy"] - runner_up["accuracy"] if runner_up else None,
                "file_name": path.name,
                "download_name": _relative_path(path, METRICS_DIR),
                "relative_path": _project_path(path),
                "modified_at": modified_at,
                "modified_label": _format_datetime(modified_at),
            }
        )

    return sorted(
        experiments,
        key=lambda item: (_dataset_sort_index(item["dataset_key"]), item["title"].lower()),
    )


def _infer_preset_key(summary: dict[str, Any]) -> str:
    if summary.get("balance_subjects"):
        return "balanced"
    if summary.get("max_subjects"):
        return "many_images_few_people"
    return "many_people_many_images"


def _load_processed_bundles() -> list[dict[str, Any]]:
    bundles: list[dict[str, Any]] = []

    for summary_path in PROCESSED_DIR.glob("*/*/summary.json"):
        with summary_path.open("r", encoding="utf-8") as json_file:
            summary = json.load(json_file)

        bundle_dir = summary_path.parent
        dataset_key = str(summary.get("dataset_name", bundle_dir.parent.name))
        preset_key = _infer_preset_key(summary)
        created_at = _parse_datetime(summary.get("created_at"))
        if created_at is None:
            created_at = datetime.fromtimestamp(summary_path.stat().st_mtime)

        processing_stats = summary.get("processing_stats", {})
        filter_stats = summary.get("filter_stats", {})
        inputs_path = bundle_dir / "inputs.npz"

        artifacts = []
        for label, filename in [
            ("Summary", "summary.json"),
            ("Inputs", "inputs.npz"),
            ("Manifest", "manifest.csv"),
            ("Labels", "label_mapping.json"),
        ]:
            artifact_path = bundle_dir / filename
            if artifact_path.exists():
                artifacts.append(
                    {
                        "label": label,
                        "download_name": _relative_path(artifact_path, PROCESSED_DIR),
                        "size_label": _format_file_size(artifact_path.stat().st_size),
                    }
                )

        face_detector = summary.get("face_detection") or processing_stats.get("face_detector")
        bundles.append(
            {
                "bundle_name": bundle_dir.name,
                "dataset_key": dataset_key,
                "dataset_label": DATASET_LABELS.get(dataset_key, _humanize_token(dataset_key)),
                "dataset_description": DATASET_DESCRIPTIONS.get(dataset_key, ""),
                "preset_key": preset_key,
                "preset_label": _humanize_preset(preset_key),
                "samples_total": int(summary.get("samples_total", 0)),
                "train_samples": int(summary.get("train_samples", 0)),
                "test_samples": int(summary.get("test_samples", 0)),
                "classes_total": int(summary.get("classes_total", 0)),
                "raw_subjects": int(filter_stats.get("subjects_before_filter", 0)),
                "raw_samples": int(filter_stats.get("samples_before_filter", 0)),
                "dropped_subject_count": int(summary.get("dropped_subject_count", 0)),
                "truncated_subject_count": int(summary.get("truncated_subject_count", 0)),
                "image_size_label": " x ".join(str(item) for item in summary.get("image_size", [])),
                "feature_shape": summary.get("feature_shape", []),
                "flatten": bool(summary.get("flatten", False)),
                "representation_label": "Flattened vector" if summary.get("flatten", False) else "2D image",
                "face_detection_label": face_detector.upper() if face_detector else "Khong dung",
                "face_detection_key": face_detector or "none",
                "face_detected_samples": int(processing_stats.get("face_detected_samples", 0)),
                "bundle_size_label": _format_file_size(inputs_path.stat().st_size) if inputs_path.exists() else "N/A",
                "created_at": created_at,
                "created_label": _format_datetime(created_at),
                "relative_path": _project_path(bundle_dir),
                "summary": summary,
                "artifacts": artifacts,
            }
        )

    def sort_key(item: dict[str, Any]) -> tuple[int, int, int, float]:
        created_at = item["created_at"].timestamp() if item["created_at"] else 0.0
        flatten_priority = 1 if item["flatten"] else 0
        face_priority = 1 if item["face_detection_key"] == "mtcnn" else 0
        return (_dataset_sort_index(item["dataset_key"]), -flatten_priority, -face_priority, -created_at)

    return sorted(bundles, key=sort_key)


def _select_featured_bundle(dataset_bundles: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not dataset_bundles:
        return None

    return max(
        dataset_bundles,
        key=lambda item: (
            1 if item["flatten"] else 0,
            1 if item["face_detection_key"] == "mtcnn" else 0,
            item["created_at"].timestamp() if item["created_at"] else 0.0,
        ),
    )


def _build_dataset_overviews(
    bundles: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    experiment_lookup = {
        item["dataset_key"]: item
        for item in experiments
        if item["dataset_key"] in DATASET_LABELS and item["dataset_key"] not in {"overview", "custom"}
    }

    overviews: list[dict[str, Any]] = []
    for dataset_key in DATASET_ORDER:
        dataset_bundles = [item for item in bundles if item["dataset_key"] == dataset_key]
        featured_bundle = _select_featured_bundle(dataset_bundles)
        experiment = experiment_lookup.get(dataset_key)

        raw_path = None
        if project_config is not None:
            if dataset_key == "orl":
                raw_path = Path(getattr(project_config, "ORL_DATA_DIR", ""))
            elif dataset_key == "extended_yale_b":
                raw_path = Path(getattr(project_config, "EXTENDED_YALE_B_DIR", ""))
            elif dataset_key == "lfw":
                raw_path = Path(getattr(project_config, "LFW_DATA_DIR", ""))

        overviews.append(
            {
                "dataset_key": dataset_key,
                "dataset_label": DATASET_LABELS[dataset_key],
                "description": DATASET_DESCRIPTIONS[dataset_key],
                "bundle_count": len(dataset_bundles),
                "featured_bundle": featured_bundle,
                "samples_total": featured_bundle["samples_total"] if featured_bundle else 0,
                "classes_total": featured_bundle["classes_total"] if featured_bundle else 0,
                "raw_subjects": featured_bundle["raw_subjects"] if featured_bundle else 0,
                "raw_samples": featured_bundle["raw_samples"] if featured_bundle else 0,
                "winner": experiment["winner"] if experiment else None,
                "accuracy_gap": experiment["accuracy_gap"] if experiment else None,
                "experiment": experiment,
                "raw_path": _project_path(raw_path) if raw_path else "N/A",
            }
        )

    return overviews


def _load_saved_models(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics_lookup = {
        (experiment["dataset_key"], record["model_key"]): record
        for experiment in experiments
        for record in experiment["records"]
    }

    saved_models: list[dict[str, Any]] = []
    for path in sorted(SAVED_MODELS_DIR.glob("*.joblib")):
        stem = path.stem.removesuffix("_notebook")
        model_key = "pca_knn" if stem.startswith("pca_knn") else "pca_svm"
        suffix = stem[len(model_key) :].strip("_")
        dataset_key, preset_key = _parse_dataset_and_preset(suffix)
        stat = path.stat()
        metric_record = metrics_lookup.get((dataset_key, model_key))
        modified_at = datetime.fromtimestamp(stat.st_mtime)

        saved_models.append(
            {
                "file_name": path.name,
                "relative_path": _project_path(path),
                "download_name": _relative_path(path, SAVED_MODELS_DIR),
                "model_key": model_key,
                "model_label": MODEL_LABELS[model_key],
                "accent": MODEL_ACCENTS[model_key],
                "dataset_key": dataset_key,
                "dataset_label": DATASET_LABELS.get(dataset_key, _humanize_token(dataset_key)),
                "preset_key": preset_key,
                "preset_label": _humanize_preset(preset_key),
                "size_label": _format_file_size(stat.st_size),
                "modified_at": modified_at,
                "modified_label": _format_datetime(modified_at),
                "accuracy": metric_record["accuracy"] if metric_record else None,
                "train_time": metric_record["train_time"] if metric_record else None,
            }
        )

    return sorted(
        saved_models,
        key=lambda item: (_dataset_sort_index(item["dataset_key"]), item["model_label"], item["file_name"]),
    )


def _build_model_parameters() -> list[dict[str, Any]]:
    if project_config is None:
        return []

    image_size = getattr(project_config, "IMAGE_SIZE", (92, 112))
    return [
        {
            "label": "Image size",
            "value": f"{image_size[0]} x {image_size[1]}",
            "detail": "Kich thuoc chuan cho buoc preprocess.",
        },
        {
            "label": "PCA components",
            "value": ", ".join(str(item) for item in getattr(project_config, "PCA_N_COMPONENTS", [])),
            "detail": "Danh sach so chieu PCA dung de quet thi nghiem.",
        },
        {
            "label": "KNN k",
            "value": ", ".join(str(item) for item in getattr(project_config, "KNN_K_VALUES", [])),
            "detail": "Tap gia tri k cho PCA + KNN.",
        },
        {
            "label": "SVM kernels",
            "value": ", ".join(str(item) for item in getattr(project_config, "SVM_KERNELS", [])),
            "detail": "Cac kernel duoc so sanh trong PCA + SVM.",
        },
        {
            "label": "SVM C",
            "value": ", ".join(str(item) for item in getattr(project_config, "SVM_C_VALUES", [])),
            "detail": "Day gia tri C de thu nghiem.",
        },
        {
            "label": "Train / test split",
            "value": f"{(1 - getattr(project_config, 'TEST_SIZE', 0.2)) * 100:.0f} / {getattr(project_config, 'TEST_SIZE', 0.2) * 100:.0f}",
            "detail": f"Random state = {getattr(project_config, 'RANDOM_STATE', 42)}.",
        },
    ]


def _build_home_chart(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    experiment_lookup = {
        experiment["dataset_key"]: experiment
        for experiment in experiments
        if experiment["dataset_key"] in DATASET_ORDER
    }

    labels: list[str] = []
    knn_accuracy: list[float] = []
    svm_accuracy: list[float] = []
    knn_train_time: list[float] = []
    svm_train_time: list[float] = []

    for dataset_key in DATASET_ORDER:
        experiment = experiment_lookup.get(dataset_key)
        if experiment is None:
            continue

        labels.append(DATASET_LABELS[dataset_key])
        records = {record["model_key"]: record for record in experiment["records"]}
        knn_record = records.get("pca_knn")
        svm_record = records.get("pca_svm")

        knn_accuracy.append(round((knn_record or {}).get("accuracy", 0.0) * 100, 2))
        svm_accuracy.append(round((svm_record or {}).get("accuracy", 0.0) * 100, 2))
        knn_train_time.append(round((knn_record or {}).get("train_time", 0.0), 3))
        svm_train_time.append(round((svm_record or {}).get("train_time", 0.0), 3))

    return {
        "labels": labels,
        "knn_accuracy": knn_accuracy,
        "svm_accuracy": svm_accuracy,
        "knn_train_time": knn_train_time,
        "svm_train_time": svm_train_time,
    }


def _build_dataset_chart(dataset_overviews: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "labels": [item["dataset_label"] for item in dataset_overviews],
        "samples": [item["samples_total"] for item in dataset_overviews],
        "classes": [item["classes_total"] for item in dataset_overviews],
    }


def _build_snapshot(
    dataset_overviews: list[dict[str, Any]],
    bundles: list[dict[str, Any]],
    saved_models: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
) -> dict[str, Any]:
    best_record = None
    best_dataset_label = None

    for experiment in experiments:
        winner = experiment["winner"]
        if best_record is None or winner["accuracy"] > best_record["accuracy"]:
            best_record = winner
            best_dataset_label = experiment["dataset_label"]

    total_processed_samples = sum(item["samples_total"] for item in dataset_overviews)
    return {
        "dataset_count": len(dataset_overviews),
        "processed_bundle_count": len(bundles),
        "saved_model_count": len(saved_models),
        "metric_count": len(experiments),
        "processed_sample_count": total_processed_samples,
        "best_accuracy_label": _format_percent(best_record["accuracy"]) if best_record else "N/A",
        "best_accuracy_dataset": best_dataset_label or "N/A",
        "best_accuracy_model": best_record["model_label"] if best_record else "N/A",
    }


def build_dashboard_context() -> dict[str, Any]:
    experiments = _load_metrics_experiments()
    bundles = _load_processed_bundles()
    dataset_overviews = _build_dataset_overviews(bundles, experiments)
    saved_models = _load_saved_models(experiments)
    snapshot = _build_snapshot(dataset_overviews, bundles, saved_models, experiments)

    return {
        "snapshot": snapshot,
        "experiments": experiments,
        "featured_experiments": [item for item in experiments if item["dataset_key"] in DATASET_ORDER],
        "processed_bundles": bundles,
        "dataset_overviews": dataset_overviews,
        "saved_models": saved_models,
        "home_chart": _build_home_chart(experiments),
        "dataset_chart": _build_dataset_chart(dataset_overviews),
        "model_parameters": _build_model_parameters(),
        "project_goals": PROJECT_GOALS,
        "pipeline_steps": PIPELINE_STEPS,
        "comparison_axes": COMPARISON_AXES,
        "structure_cards": STRUCTURE_CARDS,
        "repro_commands": REPRO_COMMANDS,
        "model_cards": [
            {
                "key": "pca_knn",
                "title": "PCA + KNN",
                "accent": "primary",
                "summary": "Pipeline nhe, de giai thich va phu hop cho baseline co toc do train nhanh.",
                "details": [
                    "PCA scratch de giam chieu ve khong gian eigenfaces.",
                    "KNN scratch de phan loai tren embedding sau PCA.",
                    "Hop voi bai toan benchmark nho va can baseline ro rang.",
                ],
            },
            {
                "key": "pca_svm",
                "title": "PCA + SVM",
                "accent": "success",
                "summary": "Pipeline manh hon cho tap du lieu kho, doi lai thoi gian huan luyen lon hon.",
                "details": [
                    "Dung PCA de giam nhieu va lam gon feature space.",
                    "SVM kiem soat bien phan tach tot hon tren tap nhieu class.",
                    "Thuong dat accuracy cao hon tren Extended Yale B va LFW.",
                ],
            },
        ],
    }


@app.template_filter("percent")
def percent_filter(value: float | None) -> str:
    return _format_percent(value)


@app.template_filter("number")
def number_filter(value: float | int | None) -> str:
    return _format_number(value, decimals=2)


@app.route("/")
def home() -> str:
    context = build_dashboard_context()
    return render_template(
        "home.html",
        active="home",
        page_title="Tong quan du an",
        **context,
    )


@app.route("/realtime")
def realtime() -> str:
    context = build_dashboard_context()
    return render_template(
        "realtime.html",
        active="realtime",
        page_title="Pipeline va quy trinh",
        **context,
    )


@app.route("/batch")
def batch() -> str:
    context = build_dashboard_context()
    return render_template(
        "batch.html",
        active="batch",
        page_title="Du lieu da xu ly",
        **context,
    )


@app.route("/database-builder")
def database_builder() -> str:
    context = build_dashboard_context()
    return render_template(
        "database_builder.html",
        active="database_builder",
        page_title="Artifacts va models",
        **context,
    )


@app.route("/downloads/<category>/<path:filename>")
def download_artifact(category: str, filename: str):
    base_dir = DOWNLOAD_ROOTS.get(category)
    if base_dir is None:
        abort(404)

    target = (base_dir / filename).resolve()
    try:
        target.relative_to(base_dir.resolve())
    except ValueError:
        abort(404)

    if not target.is_file():
        abort(404)

    return send_file(target, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
