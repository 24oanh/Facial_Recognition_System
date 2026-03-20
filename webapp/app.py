from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
from PIL import Image
from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

ROOT = Path(__file__).resolve().parent.parent
WEBAPP = ROOT / "webapp"
STATICS = WEBAPP / "statics"
TEMPLATES = WEBAPP / "templates"
UPLOADS = STATICS / "uploads"
TEMP = Path(tempfile.gettempdir()) / "math_for_ml_webapp"
PROCESSED = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"
MODELS = WEBAPP / "saved_models"
sys.path.insert(0, str(ROOT))

from src.configs.config import IMAGE_SIZE
from src.pipelines import train_pca_knn, train_pca_svm
from src.preprocessing import detect_largest_face_bbox, preprocess_image
from src.process.dataset_processing import (
    PROCESSING_PRESETS,
    build_face_database_from_directory,
    load_processed_dataset_bundle,
)

app = Flask(__name__, template_folder=str(TEMPLATES), static_folder=str(STATICS), static_url_path="/static")
app.secret_key = os.urandom(24)
for folder in (UPLOADS, TEMP, METRICS, MODELS):
    folder.mkdir(parents=True, exist_ok=True)

DATASET_ORDER = ["orl", "extended_yale_b", "lfw", "custom"]
PRESET_ORDER = ["balanced", "many_people_many_images", "many_images_few_people"]
DATASET_LABELS = {"orl": "ORL/AT&T", "extended_yale_b": "Extended Yale B", "lfw": "LFW", "custom": "Database"}
PRESET_LABELS = {
    "balanced": "Balanced",
    "many_people_many_images": "Many people, many images",
    "many_images_few_people": "Many images, few people",
}
MODEL_LABELS = {"pca_knn": "PCA + KNN", "pca_svm": "PCA + SVM"}
DOWNLOAD_ROOTS = {"processed": PROCESSED, "metrics": METRICS, "saved-models": MODELS}
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".pgm"}
MODEL_CACHE: dict[str, tuple[float, Any]] = {}

camera = None
camera_lock = threading.Lock()
realtime_running = False
realtime_processing = False
realtime_model_key = "pca_knn"
realtime_profile_key = ""
realtime_result: dict[str, Any] = {
    "identity": None,
    "confidence": 0.0,
    "bbox": None,
    "status": "idle",
    "model_label": MODEL_LABELS["pca_knn"],
    "profile_title": "",
}


@dataclass(slots=True)
class BuildJob:
    id: str
    job_type: str
    status: str = "queued"
    progress: int = 0
    message: str = "Dang cho..."
    logs: list[str] = field(default_factory=list)
    outputs: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None

    def log(self, message: str) -> None:
        self.message = message
        self.logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")


build_jobs: dict[str, BuildJob] = {}


def _dataset_label(key: str) -> str:
    return DATASET_LABELS.get(key, key.replace("_", " ").title())


def _preset_label(key: str) -> str:
    return PRESET_LABELS.get(key, key.replace("_", " ").title())


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def _size_label(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024**2:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024**2:.2f} MB"


def _rel(path: Path, base: Path) -> str:
    return str(path.relative_to(base)).replace("\\", "/")


def _proj(path: Path) -> str:
    try:
        return _rel(path, ROOT)
    except ValueError:
        return str(path)


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _profile_slug(value: str) -> str:
    return secure_filename(value).lower().replace("-", "_") or "database"


def _model_meta_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.name}.json")


def _read_model_meta(model_path: Path) -> dict[str, Any] | None:
    meta_path = _model_meta_path(model_path)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save_upload(file: FileStorage) -> tuple[Path, str]:
    name = secure_filename(file.filename or "upload.png")
    ext = Path(name).suffix.lower() or ".png"
    if ext not in ALLOWED_EXTS:
        ext = ".png"
    out_name = f"{uuid.uuid4().hex[:10]}{ext}"
    out_path = UPLOADS / out_name
    file.save(out_path)
    return out_path, out_name


def _infer_preset(summary: dict[str, Any]) -> str:
    if summary.get("balance_subjects"):
        return "balanced"
    if summary.get("max_subjects"):
        return "many_images_few_people"
    return "many_people_many_images"


def _parse_model_file(path: Path) -> tuple[str, str, str] | None:
    stem = path.stem.removesuffix("_notebook")
    for model_key in ("pca_knn", "pca_svm"):
        prefix = f"{model_key}_"
        if not stem.startswith(prefix):
            continue
        suffix = stem[len(prefix) :]
        for dataset_key in sorted(DATASET_ORDER, key=len, reverse=True):
            prefix2 = f"{dataset_key}_"
            if suffix.startswith(prefix2):
                preset_key = suffix[len(prefix2) :]
                if preset_key:
                    return model_key, dataset_key, preset_key
    return None


def _load_bundles() -> list[dict[str, Any]]:
    rows = []
    for summary_path in PROCESSED.glob("*/*/summary.json"):
        bundle_dir = summary_path.parent
        label_map_path = bundle_dir / "label_mapping.json"
        if not label_map_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
        inverse = {int(v): k for k, v in label_map.items()}
        dataset_key = _normalize_key(str(summary.get("dataset_name", bundle_dir.parent.name)))
        if dataset_key == "custom":
            preset_key = _profile_slug(str(summary.get("profile_slug") or bundle_dir.name))
            title = str(summary.get("profile_title") or summary.get("database_name") or bundle_dir.name)
            preset_label = "Folder database"
        else:
            preset_key = _normalize_key(str(summary.get("preset_name") or _infer_preset(summary)))
            title = f"{_dataset_label(dataset_key)} · {_preset_label(preset_key)}"
            preset_label = _preset_label(preset_key)
        rows.append(
            {
                "bundle_dir": bundle_dir,
                "bundle_relative": _proj(bundle_dir),
                "dataset_key": dataset_key,
                "dataset_label": _dataset_label(dataset_key),
                "preset_key": preset_key,
                "preset_label": preset_label,
                "profile_key": f"{dataset_key}__{preset_key}",
                "title": title,
                "summary": summary,
                "mtime": bundle_dir.stat().st_mtime,
                "inverse_label_mapping": inverse,
            }
        )
    return rows


def _match_bundle(bundles: list[dict[str, Any]], dataset_key: str, preset_key: str, model_mtime: float):
    cands = [
        b
        for b in bundles
        if b["dataset_key"] == dataset_key and b["preset_key"] == preset_key and b["summary"].get("flatten", False)
    ]
    if not cands:
        return None
    older = [b for b in cands if b["mtime"] <= model_mtime + 2]
    return max(older or cands, key=lambda x: x["mtime"])


def get_processed_bundles() -> list[dict[str, Any]]:
    bundles = _load_bundles()
    return sorted(
        bundles,
        key=lambda bundle: (
            DATASET_ORDER.index(bundle["dataset_key"]) if bundle["dataset_key"] in DATASET_ORDER else 99,
            bundle["title"].lower(),
        ),
    )


def _get_bundle_by_relative(bundle_relative: str) -> dict[str, Any] | None:
    normalized = bundle_relative.strip().replace("\\", "/")
    for bundle in get_processed_bundles():
        if bundle["bundle_relative"] == normalized:
            return bundle
    return None


def get_profiles() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    bundles = get_processed_bundles()
    bundle_by_relative = {bundle["bundle_relative"]: bundle for bundle in bundles}
    grouped: dict[str, dict[str, Any]] = {}
    for model_path in sorted(MODELS.glob("*.joblib")):
        meta = _read_model_meta(model_path)
        parsed = _parse_model_file(model_path)
        bundle = None
        if meta is not None:
            bundle = bundle_by_relative.get(str(meta.get("bundle_relative", "")))
        if bundle is None:
            if parsed is None:
                continue
            model_key, dataset_key, preset_key = parsed
            bundle = _match_bundle(bundles, dataset_key, preset_key, model_path.stat().st_mtime)
            if bundle is None:
                continue
        model_key = str((meta or {}).get("model_type") or (parsed[0] if parsed is not None else ""))
        if model_key not in MODEL_LABELS:
            stem = model_path.stem.removesuffix("_notebook")
            model_key = "pca_knn" if stem.startswith("pca_knn_") else "pca_svm" if stem.startswith("pca_svm_") else ""
        if model_key not in MODEL_LABELS:
            continue
        profile_key = bundle["profile_key"]
        profile = grouped.setdefault(
            profile_key,
            {
                "key": profile_key,
                "dataset_key": bundle["dataset_key"],
                "dataset_label": bundle["dataset_label"],
                "preset_key": bundle["preset_key"],
                "preset_label": bundle["preset_label"],
                "title": bundle["title"],
                "bundle_dir": bundle["bundle_dir"],
                "bundle_relative": bundle["bundle_relative"],
                "summary": bundle["summary"],
                "inverse_label_mapping": bundle["inverse_label_mapping"],
                "models": {},
            },
        )
        profile["models"][model_key] = {
            "path": model_path,
            "label": MODEL_LABELS[model_key],
            "relative_path": _proj(model_path),
            "size_label": _size_label(model_path.stat().st_size),
        }
    profiles = sorted(
        grouped.values(),
        key=lambda p: (
            DATASET_ORDER.index(p["dataset_key"]) if p["dataset_key"] in DATASET_ORDER else 99,
            PRESET_ORDER.index(p["preset_key"]) if p["preset_key"] in PRESET_ORDER else 99,
            p["title"].lower(),
        ),
    )
    return profiles, {p["key"]: p for p in profiles}


def _default_profile_key() -> str:
    profiles, _ = get_profiles()
    return profiles[0]["key"] if profiles else ""


def _cached_model(path: Path) -> Any:
    key = str(path.resolve())
    mtime = path.stat().st_mtime
    cached = MODEL_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    model = joblib.load(path)
    MODEL_CACHE[key] = (mtime, model)
    return model


def _preprocess(image_src: str | Path | Image.Image | np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    summary = profile["summary"]
    kwargs = {
        "image_size": tuple(summary.get("image_size", list(IMAGE_SIZE))),
        "normalize": bool(summary.get("normalize", True)),
        "flatten": bool(summary.get("flatten", True)),
        "face_detection": summary.get("face_detection"),
        "face_padding_ratio": float(summary.get("face_padding_ratio", 0.25)),
        "face_crop_fallback": summary.get("face_crop_fallback", "original"),
        "face_square_crop": bool(summary.get("face_square_crop", True)),
        "face_scale_factor": float(summary.get("face_scale_factor", 1.1)),
        "face_min_neighbors": int(summary.get("face_min_neighbors", 5)),
        "face_min_size": tuple(summary.get("face_min_size", [30, 30])),
    }
    if isinstance(image_src, (str, Path)):
        with Image.open(image_src) as image:
            pil = image.copy()
    elif isinstance(image_src, Image.Image):
        pil = image_src.copy()
    else:
        pil = Image.fromarray(image_src)
    try:
        arr = preprocess_image(pil, **kwargs)
    except Exception:
        kwargs["face_detection"] = None
        arr = preprocess_image(pil, **kwargs)
    arr = np.asarray(arr, dtype=np.float64)
    return np.expand_dims(arr, axis=0) if arr.ndim == 1 else arr


def _identity(profile: dict[str, Any], label: Any) -> str:
    try:
        return profile["inverse_label_mapping"].get(int(label), str(label))
    except Exception:
        return str(label)


def _predict_knn(profile: dict[str, Any], model: Any, X: np.ndarray) -> dict[str, Any]:
    emb = model.transform(X)
    dist = model.knn._compute_distances(emb)
    idx = np.argsort(dist, axis=1)[0, : model.knn.k]
    labels = model.knn.y_train[idx]
    uniq, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    pred = uniq[order[0]]
    top_k = []
    for i in order[:3]:
        label = uniq[i]
        mask = labels == label
        top_k.append(
            {
                "identity": _identity(profile, label),
                "confidence": float(counts[i] / len(idx)),
                "score": float(np.min(dist[0, idx[mask]])),
            }
        )
    return {
        "identity": _identity(profile, pred),
        "confidence": float(counts[order[0]] / len(idx)),
        "score": float(np.min(dist[0, idx])),
        "score_label": "Khoang cach gan nhat",
        "top_k": top_k,
    }


def _predict_svm(profile: dict[str, Any], model: Any, X: np.ndarray) -> dict[str, Any]:
    pred = model.predict(X)[0]
    decision = model.svm.decision_function(model.transform(X))
    if np.ndim(decision) == 1:
        margin = float(np.asarray(decision)[0])
        conf = float(1.0 / (1.0 + np.exp(-abs(margin))))
        top_k = [{"identity": _identity(profile, pred), "confidence": conf, "score": margin}]
        score = margin
    else:
        row = np.asarray(decision[0], dtype=np.float64)
        probs = np.exp(row - np.max(row))
        probs = probs / np.sum(probs)
        classes = list(model.svm.classes_)
        pred_idx = classes.index(pred)
        conf = float(probs[pred_idx])
        score = float(row[pred_idx])
        top_idx = np.argsort(probs)[::-1][:3]
        top_k = [
            {
                "identity": _identity(profile, classes[i]),
                "confidence": float(probs[i]),
                "score": float(row[i]),
            }
            for i in top_idx
        ]
    return {
        "identity": _identity(profile, pred),
        "confidence": conf,
        "score": score,
        "score_label": "Margin",
        "top_k": top_k,
    }


def predict(profile: dict[str, Any], model_key: str, image_src: str | Path | Image.Image | np.ndarray) -> dict[str, Any]:
    entry = profile["models"].get(model_key)
    if entry is None:
        return {"status": "error", "identity": "Model khong ton tai", "confidence": 0.0, "time_ms": 0.0}
    start = perf_counter()
    X = _preprocess(image_src, profile)
    model = _cached_model(entry["path"])
    data = _predict_knn(profile, model, X) if model_key == "pca_knn" else _predict_svm(profile, model, X)
    data.update({"status": "success", "model_label": MODEL_LABELS[model_key], "time_ms": round((perf_counter() - start) * 1000, 2)})
    return data


def compare_models(profile: dict[str, Any], image_src: str | Path | Image.Image | np.ndarray) -> dict[str, Any]:
    out = {}
    for model_key in ("pca_knn", "pca_svm"):
        out[model_key] = predict(profile, model_key, image_src)
        out[model_key]["ok"] = out[model_key]["status"] == "success"
    return out


def _bbox(profile: dict[str, Any], image: np.ndarray) -> list[int] | None:
    detector = profile["summary"].get("face_detection") or "haar"
    try:
        box = detect_largest_face_bbox(image, detector=detector)
    except Exception:
        try:
            box = detect_largest_face_bbox(image, detector="haar")
        except Exception:
            box = None
    if box is None:
        return None
    x, y, w, h = box
    return [int(x), int(y), int(x + w), int(y + h)]


def get_camera():
    global camera
    if cv2 is None:
        return None
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return camera


def release_camera() -> None:
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None


def recognize_frame(frame: np.ndarray) -> None:
    global realtime_processing, realtime_result
    if realtime_processing:
        return
    realtime_processing = True
    try:
        _, lookup = get_profiles()
        profile = lookup.get(realtime_profile_key)
        if profile is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if cv2 is not None else frame
        pred = predict(profile, realtime_model_key, rgb)
        realtime_result = {
            "identity": pred.get("identity"),
            "confidence": pred.get("confidence", 0.0),
            "bbox": _bbox(profile, rgb),
            "status": pred.get("status", "idle"),
            "model_label": pred.get("model_label", MODEL_LABELS[realtime_model_key]),
            "profile_title": profile["title"],
        }
    except Exception as exc:
        realtime_result = {
            "identity": None,
            "confidence": 0.0,
            "bbox": None,
            "status": "error",
            "error": str(exc),
            "model_label": MODEL_LABELS.get(realtime_model_key, realtime_model_key),
            "profile_title": "",
        }
    finally:
        realtime_processing = False


def generate_frames():
    global realtime_running
    if cv2 is None:
        return
    realtime_running = True
    last = 0.0
    while realtime_running:
        cam = get_camera()
        if cam is None:
            break
        ok, frame = cam.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        if time.time() - last >= 0.45 and not realtime_processing:
            threading.Thread(target=recognize_frame, args=(frame.copy(),), daemon=True).start()
            last = time.time()
        if realtime_result.get("bbox") is not None:
            x1, y1, x2, y2 = realtime_result["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (16, 185, 129), 2)
            label = realtime_result.get("identity") or "Dang tim"
            if realtime_result.get("identity"):
                label = f"{label}: {realtime_result.get('confidence', 0.0) * 100:.1f}%"
            cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (16, 185, 129), 2)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


def _write_metrics(path: Path, model_label: str, evaluation: dict[str, Any]) -> None:
    macro = evaluation.get("report", {}).get("macro avg", {})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Train Time"])
        writer.writeheader()
        writer.writerow(
            {
                "Model": model_label,
                "Accuracy": round(float(evaluation.get("accuracy", 0.0)), 4),
                "Precision": round(float(macro.get("precision", 0.0)), 4),
                "Recall": round(float(macro.get("recall", 0.0)), 4),
                "F1-Score": round(float(macro.get("f1-score", 0.0)), 4),
                "Train Time": round(float(evaluation.get("train_time", 0.0)), 3),
            }
        )


def _write_model_meta(model_path: Path, bundle: dict[str, Any], model_type: str) -> None:
    payload = {
        "bundle_relative": bundle["bundle_relative"],
        "dataset_key": bundle["dataset_key"],
        "preset_key": bundle["preset_key"],
        "profile_key": bundle["profile_key"],
        "profile_title": bundle["title"],
        "model_type": model_type,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _model_meta_path(model_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _job_output(job: BuildJob, category: str, path: Path, label: str) -> None:
    job.outputs.append({"label": label, "relative_path": _proj(path), "category": category, "filename": _rel(path, DOWNLOAD_ROOTS[category])})


def _run_build_database(job: BuildJob, payload: dict[str, Any]) -> None:
    face_detection = payload.get("face_detection")
    detector = "mtcnn" if face_detection == "auto" else None if face_detection == "none" else face_detection
    min_images = int(payload.get("min_images_per_person", 2))
    max_images_raw = str(payload.get("max_images_per_person", "")).strip()
    max_images = int(max_images_raw) if max_images_raw else None
    test_size = float(payload.get("test_size", 0.0))

    job.status = "running"
    job.progress = 10
    job.log("Dang doc thu muc nguon va kiem tra cau truc folder.")
    bundle = build_face_database_from_directory(
        source_dir=payload["source_dir"],
        database_name=payload.get("database_name") or Path(payload["source_dir"]).name,
        output_root=PROCESSED,
        min_images_per_subject=min_images,
        max_images_per_subject=max_images,
        test_size=test_size,
        face_detection=detector,
        face_crop_fallback="skip" if detector else "original",
        min_face_area_ratio=0.08 if detector else 0.0,
    )
    out = Path(bundle["output_dir"])
    summary = bundle["summary"]
    job.progress = 80
    job.log(
        f"Da tao database '{summary.get('profile_title', out.name)}' voi "
        f"{summary.get('classes_total', 0)} nguoi / {summary.get('samples_total', 0)} anh."
    )
    for name, label in [
        ("summary.json", "Summary"),
        ("inputs.npz", "Inputs"),
        ("manifest.csv", "Manifest"),
        ("manifest_train.csv", "Manifest train"),
        ("manifest_test.csv", "Manifest test"),
        ("label_mapping.json", "Label mapping"),
    ]:
        path = out / name
        if path.exists():
            _job_output(job, "processed", path, label)
    job.progress = 100
    job.status = "completed"
    job.log("Hoan tat build database tu thu muc folder ten nguoi.")


def _run_train(job: BuildJob, payload: dict[str, Any]) -> None:
    job.status = "running"
    job.progress = 10
    bundle = _get_bundle_by_relative(str(payload.get("bundle_relative", "")))
    if bundle is None:
        raise ValueError("Khong tim thay database da chon.")
    bundle_dir = bundle["bundle_dir"]
    job.log(f"Dang nap database: {bundle['title']}.")
    data = load_processed_dataset_bundle(bundle_dir)
    X_train = np.asarray(data["X_train"] if len(data["X_train"]) else data["X"], dtype=np.float64)
    y_train = np.asarray(data["y_train"] if len(data["y_train"]) else data["y"])
    X_test = np.asarray(data["X_test"], dtype=np.float64)
    y_test = np.asarray(data["y_test"])
    eval_split = "test"
    if X_test.size == 0 or y_test.size == 0:
        X_test = X_train
        y_test = y_train
        eval_split = "train"
        job.log("Khong co test split. Metrics se duoc tinh tren tap train/deployment.")
    n_components = int(payload.get("n_components", 20))
    model_type = payload["model_type"]
    job.progress = 40
    if model_type == "pca_knn":
        model = train_pca_knn(X_train, y_train, n_components=n_components, k=int(payload.get("k", 3)), metric=payload.get("metric", "euclidean"))
    else:
        model = train_pca_svm(X_train, y_train, n_components=n_components, C=float(payload.get("C", 1.0)), kernel=payload.get("kernel", "linear"), gamma=payload.get("gamma", "scale"))
    job.progress = 75
    evaluation = model.evaluate(X_test, y_test)
    stem = secure_filename(payload.get("output_name") or f"{model_type}_{bundle['dataset_key']}_{bundle['preset_key']}_{time.strftime('%Y%m%d_%H%M%S')}")
    if not stem.endswith(".joblib"):
        stem += ".joblib"
    model_path = MODELS / stem
    model.save(model_path)
    _write_model_meta(model_path, bundle, model_type)
    metrics_path = METRICS / f"{Path(stem).stem}_metrics.csv"
    _write_metrics(metrics_path, MODEL_LABELS[model_type], evaluation)
    _job_output(job, "saved-models", model_path, "Model")
    meta_path = _model_meta_path(model_path)
    if meta_path.exists():
        _job_output(job, "saved-models", meta_path, "Model metadata")
    _job_output(job, "metrics", metrics_path, "Metrics")
    _job_output(job, "processed", bundle_dir / "summary.json", "Bundle summary")
    job.progress = 100
    job.status = "completed"
    job.log(f"Hoan tat train. Accuracy={evaluation.get('accuracy', 0.0) * 100:.2f}% ({eval_split} split).")


def _start_job(job_type: str, payload: dict[str, Any]) -> BuildJob:
    job = BuildJob(id=uuid.uuid4().hex, job_type=job_type)
    build_jobs[job.id] = job

    def runner() -> None:
        try:
            if job_type == "build_database":
                _run_build_database(job, payload)
            else:
                _run_train(job, payload)
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.log(f"That bai: {exc}")

    threading.Thread(target=runner, daemon=True).start()
    return job


def _base_context(active: str, page_title: str) -> dict[str, Any]:
    profiles, lookup = get_profiles()
    return {
        "active": active,
        "page_title": page_title,
        "profiles": profiles,
        "profile_lookup": lookup,
        "selected_profile_key": _default_profile_key(),
    }


@app.template_filter("percent")
def percent_filter(value: float | None) -> str:
    return _percent(value)


@app.route("/", methods=["GET", "POST"])
def home():
    ctx = _base_context("home", "Nhan dang")
    selected = request.form.get("profile_key") or ctx["selected_profile_key"]
    profile = ctx["profile_lookup"].get(selected)
    if profile is None and ctx["profiles"]:
        profile = ctx["profiles"][0]
        selected = profile["key"]
    result = None
    image_name = None
    error_message = None
    if request.method == "POST":
        image = request.files.get("image")
        if image is None or not image.filename:
            error_message = "Vui long chon anh."
        elif profile is None:
            error_message = "Chua co profile nao san sang."
        else:
            image_path, image_name = _save_upload(image)
            try:
                result = compare_models(profile, image_path)
            except Exception as exc:
                error_message = str(exc)
    ctx.update({"selected_profile_key": selected, "selected_profile": profile, "result": result, "image_name": image_name, "error_message": error_message})
    return render_template("home.html", **ctx)


@app.route("/about")
def about():
    ctx = _base_context("about", "About")
    dataset_cards = []
    for dataset_key in DATASET_ORDER:
        presets = PROCESSING_PRESETS.get(dataset_key, {})
        profile_count = sum(1 for profile in ctx["profiles"] if profile["dataset_key"] == dataset_key)
        if not presets and not profile_count:
            continue
        dataset_cards.append(
            {
                "key": dataset_key,
                "label": _dataset_label(dataset_key),
                "profile_count": profile_count,
                "preset_count": len(presets),
                "presets": [_preset_label(key) for key in presets],
            }
        )
    ctx.update(
        {
            "model_cards": [
                {
                    "label": MODEL_LABELS["pca_knn"],
                    "description": "Dùng PCA để giảm chiều và KNN để tìm danh tính gần nhất trong embedding space.",
                },
                {
                    "label": MODEL_LABELS["pca_svm"],
                    "description": "Dùng PCA để nén đặc trưng và SVM để học biên phân lớp trên dữ liệu đã xử lý.",
                },
            ],
            "dataset_cards": dataset_cards,
            "profile_total": len(ctx["profiles"]),
            "artifact_total": sum(len(profile["models"]) for profile in ctx["profiles"]),
        }
    )
    return render_template("about.html", **ctx)


@app.route("/batch")
def batch_redirect():
    return redirect(url_for("about"))


@app.route("/realtime")
def realtime():
    global realtime_profile_key
    ctx = _base_context("realtime", "Realtime")
    if not realtime_profile_key:
        realtime_profile_key = ctx["selected_profile_key"]
    ctx.update({"selected_profile_key": realtime_profile_key or ctx["selected_profile_key"], "realtime_model_key": realtime_model_key})
    return render_template("realtime.html", **ctx)


@app.route("/video_feed")
def video_feed():
    if cv2 is None:
        abort(500)
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/realtime_result")
def get_realtime_result():
    return jsonify(realtime_result)


@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    global realtime_running
    realtime_running = False
    release_camera()
    return jsonify({"status": "stopped"})


@app.route("/set_realtime_model", methods=["POST"])
def set_realtime_model():
    global realtime_model_key
    data = request.get_json(silent=True) or {}
    model_key = data.get("model", "pca_knn")
    if model_key not in MODEL_LABELS:
        return jsonify({"status": "error", "message": "Invalid model"}), 400
    realtime_model_key = model_key
    return jsonify({"status": "success", "model": model_key})


@app.route("/set_realtime_profile", methods=["POST"])
def set_realtime_profile():
    global realtime_profile_key
    data = request.get_json(silent=True) or {}
    profile_key = data.get("profile_key", "")
    _, lookup = get_profiles()
    if profile_key not in lookup:
        return jsonify({"status": "error", "message": "Invalid profile"}), 400
    realtime_profile_key = profile_key
    return jsonify({"status": "success", "profile_key": profile_key})


@app.route("/database-builder")
def database_builder():
    ctx = _base_context("database_builder", "Database Builder")
    ctx.update(
        {
            "bundle_options": [
                {
                    "value": bundle["bundle_relative"],
                    "label": bundle["title"],
                    "hint": bundle["bundle_relative"],
                }
                for bundle in get_processed_bundles()
            ],
        }
    )
    return render_template("database_builder.html", **ctx)


@app.route("/database-builder/build", methods=["POST"])
def build_database():
    data = request.get_json(silent=True) or {}
    job_type = data.get("job_type")
    if job_type not in {"build_database", "train_model"}:
        return jsonify({"error": "Loai job khong hop le"}), 400
    if job_type == "build_database":
        source_dir = str(data.get("source_dir", "")).strip()
        database_name = str(data.get("database_name", "")).strip()
        if not source_dir:
            return jsonify({"error": "Vui long nhap thu muc nguon"}), 400
        if not database_name:
            return jsonify({"error": "Vui long nhap ten database"}), 400
        try:
            min_images = int(data.get("min_images_per_person", 2))
            max_images_raw = str(data.get("max_images_per_person", "")).strip()
            max_images = int(max_images_raw) if max_images_raw else None
            test_size = float(data.get("test_size", 0.0))
        except (TypeError, ValueError):
            return jsonify({"error": "Thong so build database khong hop le"}), 400
        if min_images < 1:
            return jsonify({"error": "So anh toi thieu moi nguoi phai >= 1"}), 400
        if max_images is not None and max_images < min_images:
            return jsonify({"error": "So anh toi da phai >= so anh toi thieu"}), 400
        if test_size < 0 or test_size >= 1:
            return jsonify({"error": "Ty le test phai nam trong [0, 1)"}), 400
    else:
        bundle_relative = str(data.get("bundle_relative", "")).strip()
        if _get_bundle_by_relative(bundle_relative) is None:
            return jsonify({"error": "Database duoc chon khong hop le"}), 400
        if data.get("model_type") not in MODEL_LABELS:
            return jsonify({"error": "Model khong hop le"}), 400
    job = _start_job(job_type, data)
    return jsonify({"status": "success", "job_id": job.id})


@app.route("/database-builder/status/<job_id>")
def get_build_status(job_id: str):
    job = build_jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Khong tim thay job"}), 404
    return jsonify(
        {
            "id": job.id,
            "job_type": job.job_type,
            "status": job.status,
            "progress": job.progress,
            "message": job.message,
            "logs": job.logs,
            "outputs": job.outputs,
            "error": job.error,
        }
    )


@app.route("/downloads/<category>/<path:filename>")
def download_artifact(category: str, filename: str):
    base = DOWNLOAD_ROOTS.get(category)
    if base is None:
        abort(404)
    target = (base / filename).resolve()
    try:
        target.relative_to(base.resolve())
    except ValueError:
        abort(404)
    if not target.is_file():
        abort(404)
    return send_file(target, as_attachment=True)


if __name__ == "__main__":
    if not realtime_profile_key:
        realtime_profile_key = _default_profile_key()
    app.run(debug=True)
