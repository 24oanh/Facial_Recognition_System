from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageDraw
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
BUILDER_UPLOADS = TEMP / "builder_uploads"
PROCESSED = ROOT / "data" / "processed"
METRICS = ROOT / "results" / "metrics"
MODELS = WEBAPP / "saved_models"
sys.path.insert(0, str(ROOT))

from src.configs.config import IMAGE_SIZE
from src.pipelines import train_pca_knn, train_pca_svm
from src.preprocessing import detect_face_bboxes, preprocess_image, warmup_face_detector
from src.process.dataset_processing import (
    PROCESSING_PRESETS,
    build_face_database_from_directory,
    load_processed_dataset_bundle,
    resolve_processed_dataset_bundle_dir,
)

app = Flask(__name__, template_folder=str(TEMPLATES), static_folder=str(STATICS), static_url_path="/static")
app.secret_key = os.urandom(24)
for folder in (UPLOADS, TEMP, BUILDER_UPLOADS, METRICS, MODELS):
    folder.mkdir(parents=True, exist_ok=True)

DATASET_ORDER = ["orl", "extended_yale_b", "custom", "lfw"]
PRESET_ORDER = [
    "balanced",
    "harsh_conditions",
    "enhanced_conditions",
    "many_people_many_images",
    "many_images_few_people",
]
DATASET_LABELS = {
    "orl": "ORL/AT&T",
    "extended_yale_b": "Extended Yale B",
    "lfw": "LFW",
    "custom": "Database",
}
PRESET_LABELS = {
    "balanced": "Balanced",
    "harsh_conditions": "Harsh conditions",
    "enhanced_conditions": "Enhanced conditions",
    "many_people_many_images": "Many people, many images",
    "many_images_few_people": "Many images, few people",
}
MODEL_LABELS = {"pca_knn": "PCA + KNN", "pca_svm": "PCA + SVM"}
DOWNLOAD_ROOTS = {"processed": PROCESSED, "metrics": METRICS, "saved-models": MODELS}
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".webp"}
MODEL_EXTENSIONS = {".joblib", ".pkl"}
MODEL_CACHE: dict[str, tuple[float, Any]] = {}
SVM_CLASS_REFERENCE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
KNN_REFERENCE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
UPLOAD_FACE_DETECTOR = "mtcnn"
REALTIME_FACE_DETECTOR = "haar"
CUSTOM_BUILD_PROCESSING_PROFILE = "custom_aligned"
CUSTOM_BUILD_FACE_ALIGN = True
CUSTOM_BUILD_FACE_PADDING_RATIO = 0.18
CUSTOM_BUILD_FACE_CROP_FALLBACK = "skip"
CUSTOM_BUILD_FACE_SQUARE_CROP = True
SVM_CONFIDENCE_TEMPERATURE = 1.75
SVM_CENTROID_REJECT_FACTOR = 1.15
DEFAULT_UNKNOWN_THRESHOLD = 0.35
INPUT_FACE_MIN_SIZE = (56, 56)
INPUT_FACE_MIN_AREA_RATIO = 0.0025
BUILDER_DEFAULTS = {
    "n_components": 20,
    "k": 3,
    "metric": "euclidean",
    "C": 1.0,
    "kernel": "linear",
    "gamma": "scale",
}
BUILDER_KNN_METRICS = {"euclidean", "cosine", "manhattan"}
BUILDER_SVM_KERNELS = {"linear", "rbf", "poly"}

camera = None
camera_lock = threading.Lock()
realtime_running = False
realtime_processing = False
realtime_model_key = "pca_knn"
realtime_profile_key = ""
realtime_unknown_threshold = DEFAULT_UNKNOWN_THRESHOLD
realtime_result: dict[str, Any] = {
    "identity": None,
    "confidence": 0.0,
    "raw_confidence": 0.0,
    "final_confidence": 0.0,
    "knn_vote_ratio": None,
    "knn_distance_confidence": None,
    "bbox": None,
    "faces": [],
    "face_count": 0,
    "display_state": None,
    "unknown_threshold": DEFAULT_UNKNOWN_THRESHOLD,
    "status": "idle",
    "model_key": realtime_model_key,
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
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def log(self, message: str) -> None:
        self.message = message
        self.logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def update_details(self, **kwargs: Any) -> None:
        self.details.update(kwargs)


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


def _parse_unknown_threshold(value: Any, default: float = DEFAULT_UNKNOWN_THRESHOLD) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        threshold = float(default)
    if threshold > 1.0:
        threshold = threshold / 100.0
    return float(np.clip(threshold, 0.0, 1.0))


def _resolve_builder_gamma(value: Any) -> str | float:
    raw_value = BUILDER_DEFAULTS["gamma"] if value in (None, "") else value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if not normalized:
            normalized = str(BUILDER_DEFAULTS["gamma"])
        if normalized in {"scale", "auto"}:
            return normalized
        raw_value = normalized
    try:
        gamma = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("SVM gamma khong hop le") from exc
    if gamma <= 0.0:
        raise ValueError("SVM gamma khong hop le")
    return gamma


def _resolve_builder_params(payload: dict[str, Any]) -> dict[str, Any]:
    def _coerce_int(name: str) -> int:
        value = payload.get(name, BUILDER_DEFAULTS[name])
        if value in (None, ""):
            value = BUILDER_DEFAULTS[name]
        return int(value)

    def _coerce_float(name: str) -> float:
        value = payload.get(name, BUILDER_DEFAULTS[name])
        if value in (None, ""):
            value = BUILDER_DEFAULTS[name]
        return float(value)

    metric = str(payload.get("metric", BUILDER_DEFAULTS["metric"]) or BUILDER_DEFAULTS["metric"]).strip().lower()
    if metric not in BUILDER_KNN_METRICS:
        raise ValueError("KNN metric khong hop le")

    kernel = str(payload.get("kernel", BUILDER_DEFAULTS["kernel"]) or BUILDER_DEFAULTS["kernel"]).strip().lower()
    if kernel not in BUILDER_SVM_KERNELS:
        raise ValueError("SVM kernel khong hop le")

    n_components = _coerce_int("n_components")
    k = _coerce_int("k")
    C = _coerce_float("C")
    if n_components < 2:
        raise ValueError("n_components khong hop le")
    if k < 1:
        raise ValueError("KNN k khong hop le")
    if C <= 0.0:
        raise ValueError("SVM C khong hop le")

    return {
        "n_components": n_components,
        "k": k,
        "metric": metric,
        "C": C,
        "kernel": kernel,
        "gamma": _resolve_builder_gamma(payload.get("gamma", BUILDER_DEFAULTS["gamma"])),
    }


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


def _open_image(image_src: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image_src, (str, Path)):
        with Image.open(image_src) as image:
            return image.copy()
    if isinstance(image_src, Image.Image):
        return image_src.copy()
    return Image.fromarray(np.asarray(image_src))


def _bbox_xyxy(bbox: tuple[int, int, int, int]) -> list[int]:
    x, y, w, h = bbox
    return [int(x), int(y), int(x + w), int(y + h)]


def _is_face_bbox_large_enough(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> bool:
    width = int(bbox[2])
    height = int(bbox[3])
    if width < INPUT_FACE_MIN_SIZE[0] or height < INPUT_FACE_MIN_SIZE[1]:
        return False

    image_width = max(int(image_size[0]), 1)
    image_height = max(int(image_size[1]), 1)
    area_ratio = (width * height) / float(image_width * image_height)
    return area_ratio >= float(INPUT_FACE_MIN_AREA_RATIO)


def _detect_faces(
    image_src: str | Path | Image.Image | np.ndarray,
    detector: str = UPLOAD_FACE_DETECTOR,
) -> list[tuple[int, int, int, int]]:
    image = _open_image(image_src)
    bboxes = detect_face_bboxes(
        image,
        detector=detector,
        min_size=INPUT_FACE_MIN_SIZE,
    )
    return [bbox for bbox in bboxes if _is_face_bbox_large_enough(bbox, image.size)]


def _recognize_faces(
    profile: dict[str, Any],
    image_src: str | Path | Image.Image | np.ndarray,
    detector: str = UPLOAD_FACE_DETECTOR,
    model_keys: tuple[str, ...] = ("pca_knn", "pca_svm"),
    unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
) -> list[dict[str, Any]]:
    faces = []
    for index, bbox in enumerate(_detect_faces(image_src, detector=detector), start=1):
        row = {
            "index": index,
            "bbox_xywh": [int(v) for v in bbox],
            "bbox": _bbox_xyxy(bbox),
        }
        for model_key in model_keys:
            try:
                item = predict(
                    profile,
                    model_key,
                    image_src,
                    face_bbox=bbox,
                    unknown_threshold=unknown_threshold,
                )
            except Exception as exc:
                item = _prediction_error(
                    str(exc),
                    model_label=MODEL_LABELS.get(model_key, model_key),
                )
                item["model_key"] = model_key
            item["ok"] = item.get("display_state") != "error"
            row[model_key] = item
        faces.append(row)
    return faces


def _save_annotated_upload(
    image_src: str | Path | Image.Image | np.ndarray,
    faces: list[dict[str, Any]],
) -> str:
    image = _open_image(image_src).convert("RGB")
    draw = ImageDraw.Draw(image)
    stroke = max(2, round(min(image.size) / 180))
    label_height = max(18, stroke * 8)

    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        label = str(face["index"])
        draw.rectangle((x1, y1, x2, y2), outline=(37, 99, 235), width=stroke)
        tag_right = x1 + max(24, label_height + 8)
        tag_bottom = y1 + label_height
        draw.rounded_rectangle(
            (x1, y1, tag_right, tag_bottom),
            radius=6,
            fill=(37, 99, 235),
        )
        draw.text((x1 + 8, y1 + 3), label, fill=(255, 255, 255))

    out_name = f"{uuid.uuid4().hex[:10]}_annotated.jpg"
    out_path = UPLOADS / out_name
    image.save(out_path, format="JPEG", quality=92)
    return out_name


def _resolve_uploaded_image(filename: str) -> Path | None:
    safe_name = secure_filename(filename or "")
    if not safe_name:
        return None
    path = (UPLOADS / safe_name).resolve()
    try:
        path.relative_to(UPLOADS.resolve())
    except ValueError:
        return None
    if not path.exists() or not path.is_file():
        return None
    return path


def _sanitize_uploaded_relative_path(filename: str) -> Path:
    normalized = filename.replace("\\", "/")
    raw_parts = [part for part in PurePosixPath(normalized).parts if part not in {"", ".", ".."}]
    parts = []
    for part in raw_parts:
        sanitized = re.sub(r'[<>:"/\\\\|?*\x00-\x1f]', "_", part).strip().strip(".")
        if sanitized:
            parts.append(sanitized)
    if len(parts) < 3:
        raise ValueError("Folder upload phai theo cau truc root_folder/person_name/image.ext")
    return Path(*parts)


def _scan_source_folder(source_dir: str | Path) -> dict[str, Any]:
    root = Path(source_dir).expanduser().resolve()
    subject_dirs = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name.lower())
    subjects_total = 0
    images_total = 0
    empty_subjects: list[str] = []
    for subject_dir in subject_dirs:
        image_count = sum(
            1
            for path in subject_dir.iterdir()
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTS
        )
        if image_count > 0:
            subjects_total += 1
            images_total += image_count
        else:
            empty_subjects.append(subject_dir.name)
    return {
        "root": root,
        "subjects_total": subjects_total,
        "images_total": images_total,
        "empty_subjects": empty_subjects,
    }


def _save_uploaded_source_folder(files: list[FileStorage]) -> tuple[Path, str]:
    if not files:
        raise ValueError("Vui long chon mot folder nguon.")
    temp_root = BUILDER_UPLOADS / uuid.uuid4().hex
    temp_root.mkdir(parents=True, exist_ok=True)
    folder_name = ""
    saved_count = 0
    try:
        for storage in files:
            relative_path = _sanitize_uploaded_relative_path(storage.filename or "")
            if not folder_name:
                folder_name = relative_path.parts[0]
            nested_path = Path(*relative_path.parts[1:])
            destination = temp_root / nested_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            storage.save(destination)
            saved_count += 1
        if saved_count == 0:
            raise ValueError("Folder da chon khong co file anh hop le.")
        return temp_root, folder_name or temp_root.name
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def _cleanup_payload_source_dir(payload: dict[str, Any]) -> None:
    cleanup_source_dir = str(payload.get("cleanup_source_dir", "")).strip()
    if cleanup_source_dir:
        shutil.rmtree(cleanup_source_dir, ignore_errors=True)


def _infer_preset(summary: dict[str, Any]) -> str:
    if summary.get("balance_subjects"):
        return "balanced"
    if summary.get("max_subjects"):
        return "many_images_few_people"
    return "many_people_many_images"


def _processing_label(profile_name: str | None) -> str:
    normalized = _normalize_key(profile_name or "standard")
    labels = {
        "standard": "Standard preprocessing",
        "orl_enhanced": "ORL enhanced preprocessing",
        "yale_b_strong": "Yale B strong preprocessing",
        "custom_aligned": "Aligned + normalized preprocessing",
    }
    return labels.get(normalized, normalized.replace("_", " ").title())


def _load_bundle_from_dir(bundle_dir: Path) -> dict[str, Any] | None:
    summary_path = bundle_dir / "summary.json"
    label_map_path = bundle_dir / "label_mapping.json"
    if not summary_path.exists() or not label_map_path.exists():
        return None

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    dataset_key = _normalize_key(str(summary.get("dataset_name", bundle_dir.parent.name)))
    if dataset_key == "custom":
        preset_key = _profile_slug(str(summary.get("profile_slug") or bundle_dir.name))
        title = str(summary.get("profile_title") or summary.get("database_name") or bundle_dir.name)
        preset_label = "Folder database"
    else:
        preset_key = ""
        for candidate in PROCESSING_PRESETS.get(dataset_key, {}):
            try:
                expected_dir = resolve_processed_dataset_bundle_dir(
                    dataset_name=dataset_key,
                    preset_name=candidate,
                    output_root=PROCESSED,
                    flatten=bool(summary.get("flatten", True)),
                    image_size=tuple(summary.get("image_size", list(IMAGE_SIZE))),
                )
            except ValueError:
                continue
            if expected_dir.resolve() == bundle_dir.resolve():
                preset_key = candidate
                break
        if not preset_key:
            preset_key = _normalize_key(str(summary.get("preset_name") or _infer_preset(summary)))
        title = f"{_dataset_label(dataset_key)} · {_preset_label(preset_key)}"
        preset_label = _preset_label(preset_key)

    inverse = {int(v): k for k, v in label_map.items()}
    face_detection = summary.get("face_detection")
    face_align = bool(summary.get("face_align", False))
    upload_mode_label = "Use full image as benchmark input"
    if face_detection:
        upload_mode_label = (
            "MTCNN detect + align"
            if face_align and str(face_detection).lower() == "mtcnn"
            else "Crop face automatically"
        )
    realtime_mode_label = f"{REALTIME_FACE_DETECTOR.upper()} bbox"
    if face_align and str(face_detection).lower() == "mtcnn":
        realtime_mode_label += " + MTCNN align"
    elif face_detection:
        realtime_mode_label += f" + {str(face_detection).upper()} crop"
    return {
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
        "subjects_total": int(summary.get("classes_total", 0)),
        "samples_total": int(summary.get("samples_total", 0)),
        "train_samples": int(summary.get("train_samples", 0)),
        "test_samples": int(summary.get("test_samples", 0)),
        "processing_label": _processing_label(summary.get("processing_profile")),
        "face_detection_label": str(face_detection).upper() if face_detection else "None",
        "upload_mode_label": upload_mode_label,
        "realtime_mode_label": realtime_mode_label,
    }


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
        bundle = _load_bundle_from_dir(summary_path.parent)
        if bundle is not None:
            rows.append(bundle)
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


def _resolve_bundle_for_model(
    bundle_by_relative: dict[str, dict[str, Any]],
    dataset_key: str,
    preset_key: str,
    model_mtime: float,
) -> dict[str, Any] | None:
    if dataset_key in PROCESSING_PRESETS and preset_key in PROCESSING_PRESETS[dataset_key]:
        try:
            expected_dir = resolve_processed_dataset_bundle_dir(
                dataset_name=dataset_key,
                preset_name=preset_key,
                output_root=PROCESSED,
                flatten=True,
            )
        except ValueError:
            expected_dir = None
        if expected_dir is not None and expected_dir.exists():
            bundle = bundle_by_relative.get(_proj(expected_dir))
            if bundle is not None:
                return bundle

    return _match_bundle(list(bundle_by_relative.values()), dataset_key, preset_key, model_mtime)


def _iter_model_files() -> list[Path]:
    return sorted(
        [path for path in MODELS.iterdir() if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS],
        key=lambda path: (path.name.lower(), path.stat().st_mtime),
    )


def _bundle_version_time(bundle: dict[str, Any]) -> float:
    bundle_dir = Path(bundle["bundle_dir"])
    summary_path = bundle_dir / "summary.json"
    try:
        return summary_path.stat().st_mtime if summary_path.exists() else bundle_dir.stat().st_mtime
    except OSError:
        return 0.0


def _model_matches_bundle_version(bundle: dict[str, Any], model_path: Path) -> bool:
    if bundle.get("dataset_key") != "custom":
        return True
    try:
        return model_path.stat().st_mtime + 2 >= _bundle_version_time(bundle)
    except OSError:
        return False


def _forget_model_caches(model_path: Path) -> None:
    try:
        resolved = str(model_path.resolve())
    except OSError:
        resolved = str(model_path)
    MODEL_CACHE.pop(resolved, None)
    for cache in (SVM_CLASS_REFERENCE_CACHE, KNN_REFERENCE_CACHE):
        for key in list(cache):
            if key == resolved or key.startswith(f"{resolved}::"):
                cache.pop(key, None)


def _svm_class_reference(profile: dict[str, Any], model: Any, model_path: Path) -> dict[str, Any]:
    bundle_dir = Path(profile["bundle_dir"])
    summary_path = bundle_dir / "summary.json"
    bundle_mtime = summary_path.stat().st_mtime if summary_path.exists() else bundle_dir.stat().st_mtime
    key = f"{model_path.resolve()}::{bundle_dir.resolve()}"
    version = max(model_path.stat().st_mtime, bundle_mtime)
    cached = SVM_CLASS_REFERENCE_CACHE.get(key)
    if cached and cached[0] == version:
        return cached[1]

    bundle = load_processed_dataset_bundle(bundle_dir)
    X_ref = np.asarray(bundle["X_train"] if np.asarray(bundle["X_train"]).size else bundle["X"], dtype=np.float64)
    y_ref = np.asarray(bundle["y_train"] if np.asarray(bundle["y_train"]).size else bundle["y"])
    embeddings = model.transform(X_ref)

    classes: dict[int, dict[str, Any]] = {}
    for raw_label in np.unique(y_ref):
        label = int(raw_label)
        class_embeddings = embeddings[y_ref == raw_label]
        centroid = class_embeddings.mean(axis=0)
        distances = np.linalg.norm(class_embeddings - centroid, axis=1)
        classes[label] = {
            "centroid": centroid,
            "distance_p50": float(np.percentile(distances, 50)),
            "distance_p95": float(np.percentile(distances, 95)),
            "distance_max": float(np.max(distances)),
        }

    payload = {"classes": classes}
    SVM_CLASS_REFERENCE_CACHE[key] = (version, payload)
    return payload


def _prune_profile_models(bundle: dict[str, Any], keep_paths: list[Path]) -> list[Path]:
    profile_key = str(bundle.get("profile_key", ""))
    bundle_relative = str(bundle.get("bundle_relative", ""))
    keep_resolved = set()
    for keep_path in keep_paths:
        try:
            keep_resolved.add(str(keep_path.resolve()))
        except OSError:
            keep_resolved.add(str(keep_path))

    removed: list[Path] = []
    for model_path in _iter_model_files():
        meta = _read_model_meta(model_path)
        if meta is None:
            continue
        if str(meta.get("profile_key", "")) != profile_key and str(meta.get("bundle_relative", "")) != bundle_relative:
            continue
        try:
            resolved = str(model_path.resolve())
        except OSError:
            resolved = str(model_path)
        if resolved in keep_resolved:
            continue
        _forget_model_caches(model_path)
        model_path.unlink(missing_ok=True)
        _model_meta_path(model_path).unlink(missing_ok=True)
        removed.append(model_path)
    return removed


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
    for model_path in _iter_model_files():
        meta = _read_model_meta(model_path)
        parsed = _parse_model_file(model_path)
        bundle = None
        if meta is not None:
            bundle = bundle_by_relative.get(str(meta.get("bundle_relative", "")))
        if bundle is None:
            if parsed is None:
                continue
            model_key, dataset_key, preset_key = parsed
            bundle = _resolve_bundle_for_model(
                bundle_by_relative,
                dataset_key,
                preset_key,
                model_path.stat().st_mtime,
            )
            if bundle is None:
                continue
        model_key = str((meta or {}).get("model_type") or (parsed[0] if parsed is not None else ""))
        if model_key not in MODEL_LABELS:
            stem = model_path.stem.removesuffix("_notebook")
            model_key = "pca_knn" if stem.startswith("pca_knn_") else "pca_svm" if stem.startswith("pca_svm_") else ""
        if model_key not in MODEL_LABELS:
            continue
        if not _model_matches_bundle_version(bundle, model_path):
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
                "subjects_total": bundle["subjects_total"],
                "samples_total": bundle["samples_total"],
                "train_samples": bundle["train_samples"],
                "test_samples": bundle["test_samples"],
                "processing_label": bundle["processing_label"],
                "face_detection_label": bundle["face_detection_label"],
                "upload_mode_label": bundle["upload_mode_label"],
                "realtime_mode_label": bundle["realtime_mode_label"],
                "models": {},
            },
        )
        current_model = profile["models"].get(model_key)
        next_model = {
            "path": model_path,
            "label": MODEL_LABELS[model_key],
            "relative_path": _proj(model_path),
            "size_label": _size_label(model_path.stat().st_size),
            "mtime": model_path.stat().st_mtime,
        }
        if current_model is None or next_model["mtime"] >= current_model["mtime"]:
            profile["models"][model_key] = next_model

    for profile in grouped.values():
        for model in profile["models"].values():
            model.pop("mtime", None)
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


def _default_realtime_profile_key() -> str:
    profiles, _ = get_profiles()
    for profile in profiles:
        if profile.get("dataset_key") == "custom":
            return profile["key"]
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


def _knn_reference_stats(profile: dict[str, Any], model: Any, model_path: Path) -> dict[str, Any]:
    bundle_dir = Path(profile["bundle_dir"])
    summary_path = bundle_dir / "summary.json"
    bundle_mtime = summary_path.stat().st_mtime if summary_path.exists() else bundle_dir.stat().st_mtime
    key = f"{model_path.resolve()}::{bundle_dir.resolve()}"
    version = max(model_path.stat().st_mtime, bundle_mtime)
    cached = KNN_REFERENCE_CACHE.get(key)
    if cached and cached[0] == version:
        return cached[1]

    bundle = load_processed_dataset_bundle(bundle_dir)
    X_ref = np.asarray(bundle["X_train"] if np.asarray(bundle["X_train"]).size else bundle["X"], dtype=np.float64)
    y_ref = np.asarray(bundle["y_train"] if np.asarray(bundle["y_train"]).size else bundle["y"])
    embeddings = model.transform(X_ref)
    distance_matrix = np.asarray(model.knn._compute_distances(embeddings), dtype=np.float64)
    np.fill_diagonal(distance_matrix, np.inf)
    nearest = distance_matrix.min(axis=1)
    finite = nearest[np.isfinite(nearest)]
    if finite.size == 0:
        finite = np.array([0.0], dtype=np.float64)

    global_stats = {
        "distance_p05": float(np.percentile(finite, 5)),
        "distance_p50": float(np.percentile(finite, 50)),
        "distance_p95": float(np.percentile(finite, 95)),
    }
    global_stats["reject_distance"] = max(float(global_stats["distance_p95"]), float(global_stats["distance_p50"]) + 1e-6)

    class_stats: dict[int, dict[str, float]] = {}
    for raw_label in np.unique(y_ref):
        label_embeddings = np.asarray(embeddings[y_ref == raw_label], dtype=np.float64)
        if label_embeddings.shape[0] < 2:
            continue
        same_label_distances = _pairwise_knn_metric_distances(label_embeddings, model.knn.metric)
        np.fill_diagonal(same_label_distances, np.inf)
        label_nearest = same_label_distances.min(axis=1)
        label_finite = label_nearest[np.isfinite(label_nearest)]
        if label_finite.size == 0:
            continue
        class_stats[int(raw_label)] = {
            "distance_p50": float(np.percentile(label_finite, 50)),
            "distance_p95": float(np.percentile(label_finite, 95)),
        }

    stats = {"global": global_stats, "classes": class_stats}
    KNN_REFERENCE_CACHE[key] = (version, stats)
    return stats


def _preprocess(
    image_src: str | Path | Image.Image | np.ndarray,
    profile: dict[str, Any],
    face_detection_override: str | None = None,
    face_bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    summary = profile["summary"]
    face_detection = face_detection_override if face_detection_override is not None else summary.get("face_detection")
    kwargs = {
        "image_size": tuple(summary.get("image_size", list(IMAGE_SIZE))),
        "normalize": bool(summary.get("normalize", True)),
        "flatten": bool(summary.get("flatten", True)),
        "face_detection": face_detection,
        "face_bbox": face_bbox,
        "face_align": bool(summary.get("face_align", False)),
        "face_padding_ratio": float(summary.get("face_padding_ratio", 0.0)),
        "face_crop_fallback": summary.get("face_crop_fallback", "original"),
        "face_square_crop": bool(summary.get("face_square_crop", False)),
        "face_scale_factor": float(summary.get("face_scale_factor", 1.1)),
        "face_min_neighbors": int(summary.get("face_min_neighbors", 5)),
        "face_min_size": tuple(summary.get("face_min_size", [30, 30])),
        "processing_profile": summary.get("processing_profile", "standard"),
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
        kwargs["face_bbox"] = None
        if face_detection_override is not None and face_detection_override != summary.get("face_detection"):
            kwargs["face_detection"] = summary.get("face_detection")
            try:
                arr = preprocess_image(pil, **kwargs)
            except Exception:
                kwargs["face_detection"] = None
                arr = preprocess_image(pil, **kwargs)
        else:
            kwargs["face_detection"] = None
            arr = preprocess_image(pil, **kwargs)
    arr = np.asarray(arr, dtype=np.float64)
    return np.expand_dims(arr, axis=0) if arr.ndim == 1 else arr


def _identity(profile: dict[str, Any], label: Any) -> str:
    try:
        return profile["inverse_label_mapping"].get(int(label), str(label))
    except Exception:
        return str(label)


def _label_sort_key(label: Any) -> tuple[int, int | str]:
    try:
        return (0, int(label))
    except (TypeError, ValueError):
        return (1, str(label))


def _clip_confidence(value: Any) -> float:
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def _softmax_confidences(scores: np.ndarray) -> np.ndarray:
    scaled = np.asarray(scores, dtype=np.float64) / max(float(SVM_CONFIDENCE_TEMPERATURE), 1e-6)
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        return np.full_like(scaled, 1.0 / max(len(scaled), 1), dtype=np.float64)
    return probs / total


def _pairwise_knn_metric_distances(X: np.ndarray, metric: str) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if metric == "euclidean":
        X_sq = np.sum(X**2, axis=1, keepdims=True)
        dist_sq = np.maximum(X_sq + X_sq.T - 2.0 * (X @ X.T), 0.0)
        return np.sqrt(dist_sq)
    if metric == "manhattan":
        return np.sum(np.abs(X[:, None, :] - X[None, :, :]), axis=2)

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    denominator = np.clip(norms * norms.T, 1e-12, None)
    cosine_similarity = (X @ X.T) / denominator
    return 1.0 - cosine_similarity


def _knn_distance_confidence(
    reference: dict[str, Any],
    raw_label: Any,
    raw_distance: float,
) -> float:
    label_stats = None
    try:
        label_stats = reference.get("classes", {}).get(int(raw_label))
    except (TypeError, ValueError):
        label_stats = None
    stats = label_stats or reference.get("global", {})
    accept_distance = float(stats.get("distance_p50", 0.0))
    reject_distance = max(float(stats.get("distance_p95", accept_distance)), accept_distance + 1e-6)
    if raw_distance <= accept_distance:
        return 1.0
    if raw_distance >= reject_distance:
        return 0.0
    return float(1.0 - ((raw_distance - accept_distance) / (reject_distance - accept_distance)))


def _candidate(
    profile: dict[str, Any],
    label: Any,
    score: float,
    raw_confidence: float,
    final_confidence: float | None = None,
    **extra: Any,
) -> dict[str, Any]:
    candidate = {
        "label": label,
        "identity": _identity(profile, label),
        "score": float(score),
        "raw_confidence": _clip_confidence(raw_confidence),
    }
    candidate["final_confidence"] = (
        candidate["raw_confidence"] if final_confidence is None else _clip_confidence(final_confidence)
    )
    candidate["confidence"] = candidate["final_confidence"]
    candidate.update(extra)
    return candidate


def _serialize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "identity": candidate.get("identity"),
        "confidence": _clip_confidence(candidate.get("final_confidence", candidate.get("confidence", 0.0))),
        "raw_confidence": _clip_confidence(candidate.get("raw_confidence", 0.0)),
        "final_confidence": _clip_confidence(candidate.get("final_confidence", candidate.get("confidence", 0.0))),
        "score": float(candidate.get("score", 0.0)),
    }
    for key in ("distance_to_centroid", "distance_confidence", "knn_vote_ratio", "knn_distance_confidence"):
        if key in candidate:
            payload[key] = candidate.get(key)
    return payload


def _build_prediction_from_candidates(
    candidates: list[dict[str, Any]],
    score_label: str,
) -> dict[str, Any]:
    if not candidates:
        return {
            "identity": "Unknown",
            "confidence": 0.0,
            "raw_confidence": 0.0,
            "final_confidence": 0.0,
            "score": 0.0,
            "score_label": score_label,
            "top_k": [],
            "threshold_triggered": False,
            "display_state": "unknown",
        }

    primary = candidates[0]
    prediction = {
        "identity": primary.get("identity"),
        "confidence": _clip_confidence(primary.get("final_confidence", 0.0)),
        "raw_confidence": _clip_confidence(primary.get("raw_confidence", 0.0)),
        "final_confidence": _clip_confidence(primary.get("final_confidence", 0.0)),
        "score": float(primary.get("score", 0.0)),
        "score_label": score_label,
        "top_k": [_serialize_candidate(item) for item in candidates[:3]],
        "threshold_triggered": False,
        "display_state": "recognized",
    }
    for key in ("distance_to_centroid", "distance_confidence", "knn_vote_ratio", "knn_distance_confidence"):
        if key in primary:
            prediction[key] = primary.get(key)
    return prediction


def _prediction_error(message: str, *, model_label: str | None = None, identity: str | None = None) -> dict[str, Any]:
    payload = {
        "status": "error",
        "display_state": "error",
        "identity": identity or message,
        "confidence": 0.0,
        "raw_confidence": 0.0,
        "final_confidence": 0.0,
        "score": 0.0,
        "score_label": "Score",
        "top_k": [],
        "threshold_triggered": False,
        "time_ms": 0.0,
    }
    if model_label is not None:
        payload["model_label"] = model_label
    if identity is not None:
        payload["error"] = message
    return payload


def _predict_knn(
    profile: dict[str, Any],
    model: Any,
    X: np.ndarray,
    model_path: Path,
) -> dict[str, Any]:
    emb = model.transform(X)
    dist_row = np.asarray(model.knn._compute_distances(emb)[0], dtype=np.float64)
    predict_neighbor_count = max(int(model.knn.k), 1)
    ranked_idx = np.argsort(dist_row)
    predict_idx = ranked_idx[:predict_neighbor_count]
    predict_total = max(len(predict_idx), 1)

    label_stats: dict[Any, dict[str, Any]] = {}
    for idx in predict_idx:
        label = model.knn.y_train[idx]
        distance = float(dist_row[idx])
        stat = label_stats.setdefault(
            label,
            {"label": label, "vote_count": 0, "min_distance": float("inf"), "weighted_vote": 0.0},
        )
        stat["vote_count"] += 1
        stat["min_distance"] = min(float(stat["min_distance"]), distance)
        stat["weighted_vote"] += 1.0 / max(distance, 1e-6)

    if len(label_stats) < 3:
        for idx in ranked_idx:
            label = model.knn.y_train[idx]
            stat = label_stats.setdefault(
                label,
                {"label": label, "vote_count": 0, "min_distance": float("inf"), "weighted_vote": 0.0},
            )
            stat["min_distance"] = min(float(stat["min_distance"]), float(dist_row[idx]))
            if len(label_stats) >= 3:
                break

    if profile["dataset_key"] == "custom":
        stats = _knn_reference_stats(profile, model, model_path)

        candidates = [
            _candidate(
                profile,
                item["label"],
                score=float(item["min_distance"]),
                raw_confidence=float(item["vote_count"] / predict_total),
                final_confidence=_knn_distance_confidence(stats, item["label"], float(item["min_distance"])),
                vote_count=int(item["vote_count"]),
                weighted_vote=float(item["weighted_vote"]),
                knn_vote_ratio=float(item["vote_count"] / predict_total),
                knn_distance_confidence=_knn_distance_confidence(stats, item["label"], float(item["min_distance"])),
            )
            for item in label_stats.values()
        ]
        candidates.sort(
            key=lambda item: (
                -float(item.get("weighted_vote", 0.0)),
                -int(item.get("vote_count", 0)),
                -float(item["final_confidence"]),
                float(item["score"]),
                _label_sort_key(item["label"]),
            )
        )
    else:
        candidates = [
            _candidate(
                profile,
                item["label"],
                score=float(item["min_distance"]),
                raw_confidence=float(item["vote_count"] / predict_total),
                vote_count=int(item["vote_count"]),
                knn_vote_ratio=float(item["vote_count"] / predict_total),
                knn_distance_confidence=float(item["vote_count"] / predict_total),
            )
            for item in label_stats.values()
        ]
        candidates.sort(
            key=lambda item: (
                -float(item["final_confidence"]),
                float(item["score"]),
                _label_sort_key(item["label"]),
            )
        )

    return _build_prediction_from_candidates(candidates, "Khoang cach gan nhat")


def _predict_svm(
    profile: dict[str, Any],
    model: Any,
    X: np.ndarray,
    model_path: Path,
) -> dict[str, Any]:
    embeddings = model.transform(X)
    decision = model.svm.decision_function(embeddings)
    classes = list(model.svm.classes_)
    candidates: list[dict[str, Any]]

    if np.ndim(decision) == 1:
        margin = float(np.asarray(decision)[0])
        score_row = np.asarray([-margin, margin], dtype=np.float64)
        probs = _softmax_confidences(score_row)
        candidates = [
            _candidate(
                profile,
                classes[index],
                score=float(score_row[index]),
                raw_confidence=float(probs[index]),
            )
            for index in range(len(classes))
        ]
        sort_key = lambda item: (
            -float(item["final_confidence"]),
            -float(item["score"]),
            _label_sort_key(item["label"]),
        )
    else:
        row = np.asarray(decision[0], dtype=np.float64)
        if getattr(model.svm, "decision_function_shape", "") == "ovo" and len(classes) > 2:
            votes, margins = model.svm._collect_ovo_votes(embeddings)
            vote_row = np.asarray(votes[0], dtype=np.float64)
            margin_row = np.asarray(margins[0], dtype=np.float64)
            probs = _softmax_confidences(margin_row)
            candidates = [
                _candidate(
                    profile,
                    classes[index],
                    score=float(margin_row[index]),
                    raw_confidence=float(probs[index]),
                    vote_count=int(vote_row[index]),
                )
                for index in range(len(classes))
            ]
            sort_key = lambda item: (
                -float(item["final_confidence"]),
                -float(item.get("vote_count", 0)),
                -float(item["score"]),
                _label_sort_key(item["label"]),
            )
        else:
            probs = _softmax_confidences(row)
            candidates = [
                _candidate(
                    profile,
                    classes[index],
                    score=float(row[index]),
                    raw_confidence=float(probs[index]),
                )
                for index in range(len(classes))
            ]
            sort_key = lambda item: (
                -float(item["final_confidence"]),
                -float(item["score"]),
                _label_sort_key(item["label"]),
            )

    if profile["dataset_key"] == "custom":
        class_reference = _svm_class_reference(profile, model, model_path)
        class_stats = class_reference.get("classes", {})

        def _distance_gate(raw_label: Any, base_confidence: float) -> tuple[float, float | None, float | None]:
            try:
                label_key = int(raw_label)
            except (TypeError, ValueError):
                return float(base_confidence), None, None
            stats = class_stats.get(label_key)
            if not stats:
                return float(base_confidence), None, None
            centroid = np.asarray(stats["centroid"], dtype=np.float64)
            distance = float(np.linalg.norm(embeddings[0] - centroid))
            accept_distance = float(stats["distance_p50"])
            reject_distance = max(
                float(stats["distance_p95"]) * float(SVM_CENTROID_REJECT_FACTOR),
                accept_distance + 1e-6,
            )
            if distance <= accept_distance:
                distance_conf = 1.0
            elif distance >= reject_distance:
                distance_conf = 0.0
            else:
                distance_conf = float(1.0 - ((distance - accept_distance) / (reject_distance - accept_distance)))
            return min(float(base_confidence), distance_conf), distance, distance_conf

        for item in candidates:
            gated_confidence, item_distance, item_distance_conf = _distance_gate(
                item["label"],
                float(item["raw_confidence"]),
            )
            item["final_confidence"] = gated_confidence
            item["confidence"] = gated_confidence
            item["distance_to_centroid"] = item_distance
            item["distance_confidence"] = item_distance_conf

    candidates.sort(key=sort_key)
    return _build_prediction_from_candidates(candidates, "Margin")


def _apply_unknown_threshold(prediction: dict[str, Any], unknown_threshold: float) -> dict[str, Any]:
    status = prediction.get("status")
    if status is not None and status != "success":
        return prediction

    threshold = _parse_unknown_threshold(unknown_threshold, default=0.0)
    adjusted = dict(prediction)
    adjusted["unknown_threshold"] = threshold
    adjusted["threshold_triggered"] = False
    if threshold <= 0.0 or adjusted.get("display_state") != "recognized":
        return adjusted

    confidence = float(adjusted.get("confidence", 0.0))
    if confidence >= threshold:
        return adjusted

    adjusted["identity"] = "Unknown"
    adjusted["confidence"] = 0.0
    adjusted["final_confidence"] = 0.0
    adjusted["top_k"] = []
    adjusted["threshold_triggered"] = True
    adjusted["display_state"] = "unknown"
    return adjusted


def predict(
    profile: dict[str, Any],
    model_key: str,
    image_src: str | Path | Image.Image | np.ndarray,
    face_detection_override: str | None = None,
    face_bbox: tuple[int, int, int, int] | None = None,
    unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
) -> dict[str, Any]:
    entry = profile["models"].get(model_key)
    if entry is None:
        return _prediction_error("Model khong ton tai", model_label=MODEL_LABELS.get(model_key, model_key))
    start = perf_counter()
    X = _preprocess(
        image_src,
        profile,
        face_detection_override=face_detection_override,
        face_bbox=face_bbox,
    )
    model = _cached_model(entry["path"])
    data = (
        _predict_knn(profile, model, X, entry["path"])
        if model_key == "pca_knn"
        else _predict_svm(profile, model, X, entry["path"])
    )
    data = _apply_unknown_threshold(data, unknown_threshold)
    data.update(
        {
            "status": "success",
            "model_key": model_key,
            "model_label": MODEL_LABELS[model_key],
            "time_ms": round((perf_counter() - start) * 1000, 2),
        }
    )
    return data


def compare_models(
    profile: dict[str, Any],
    image_src: str | Path | Image.Image | np.ndarray,
    face_detection_override: str | None = None,
    face_bbox: tuple[int, int, int, int] | None = None,
    unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
) -> dict[str, Any]:
    out = {}
    for model_key in ("pca_knn", "pca_svm"):
        out[model_key] = predict(
            profile,
            model_key,
            image_src,
            face_detection_override=face_detection_override,
            face_bbox=face_bbox,
            unknown_threshold=unknown_threshold,
        )
        out[model_key]["ok"] = out[model_key].get("display_state") != "error"
    return out


def _realtime_overlay_color(display_state: str | None) -> tuple[int, int, int]:
    if display_state == "recognized":
        return (16, 185, 129)
    if display_state == "unknown":
        return (148, 163, 184)
    return (239, 68, 68)


def _realtime_overlay_label(face: dict[str, Any]) -> str:
    index = face.get("index", "?")
    display_state = face.get("display_state")
    if display_state == "recognized":
        identity = face.get("identity") or "Unknown"
        return f"{index}. {identity}: {face.get('confidence', 0.0) * 100:.1f}%"
    if display_state == "unknown":
        return f"{index}. Unknown"
    if display_state == "error":
        return f"{index}. Error"
    identity = face.get("identity")
    return f"{index}. {identity}" if identity else f"Face {index}"


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
    global realtime_processing, realtime_profile_key, realtime_result
    if realtime_processing:
        return
    realtime_processing = True
    try:
        _, lookup = get_profiles()
        profile = lookup.get(realtime_profile_key)
        if profile is None and lookup:
            realtime_profile_key = _default_realtime_profile_key()
            profile = lookup.get(realtime_profile_key)
        if profile is None:
            realtime_result = {
                "identity": None,
                "confidence": 0.0,
                "raw_confidence": 0.0,
                "final_confidence": 0.0,
                "knn_vote_ratio": None,
                "knn_distance_confidence": None,
                "bbox": None,
                "faces": [],
                "face_count": 0,
                "display_state": None,
                "unknown_threshold": realtime_unknown_threshold,
                "status": "idle",
                "model_key": realtime_model_key,
                "model_label": MODEL_LABELS.get(realtime_model_key, realtime_model_key),
                "profile_title": "",
            }
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if cv2 is not None else frame
        bboxes = _detect_faces(rgb, detector=REALTIME_FACE_DETECTOR)
        if not bboxes:
            realtime_result = {
                "identity": None,
                "confidence": 0.0,
                "raw_confidence": 0.0,
                "final_confidence": 0.0,
                "knn_vote_ratio": None,
                "knn_distance_confidence": None,
                "bbox": None,
                "faces": [],
                "face_count": 0,
                "display_state": None,
                "unknown_threshold": realtime_unknown_threshold,
                "status": "no_face",
                "model_key": realtime_model_key,
                "model_label": MODEL_LABELS.get(realtime_model_key, realtime_model_key),
                "profile_title": profile["title"],
            }
            return
        faces = []
        for index, bbox in enumerate(bboxes, start=1):
            try:
                pred = predict(
                    profile,
                    realtime_model_key,
                    rgb,
                    face_bbox=bbox,
                    unknown_threshold=realtime_unknown_threshold,
                )
            except Exception as exc:
                pred = _prediction_error(
                    str(exc),
                    model_label=MODEL_LABELS.get(realtime_model_key, realtime_model_key),
                    identity="Error",
                )
                pred["model_key"] = realtime_model_key
            faces.append(
                {
                    "index": index,
                    "identity": pred.get("identity"),
                    "confidence": pred.get("confidence", 0.0),
                    "raw_confidence": pred.get("raw_confidence", 0.0),
                    "final_confidence": pred.get("final_confidence", 0.0),
                    "knn_vote_ratio": pred.get("knn_vote_ratio"),
                    "knn_distance_confidence": pred.get("knn_distance_confidence"),
                    "bbox": _bbox_xyxy(bbox),
                    "status": pred.get("status", "idle"),
                    "display_state": pred.get("display_state"),
                    "model_key": pred.get("model_key", realtime_model_key),
                }
            )
        candidates = [item for item in faces if item.get("display_state") in {"recognized", "unknown"}] or faces
        primary = max(candidates, key=lambda item: (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1]))
        realtime_result = {
            "identity": primary.get("identity"),
            "confidence": primary.get("confidence", 0.0),
            "raw_confidence": primary.get("raw_confidence", 0.0),
            "final_confidence": primary.get("final_confidence", 0.0),
            "knn_vote_ratio": primary.get("knn_vote_ratio"),
            "knn_distance_confidence": primary.get("knn_distance_confidence"),
            "bbox": primary.get("bbox"),
            "faces": faces,
            "face_count": len(faces),
            "display_state": primary.get("display_state"),
            "unknown_threshold": realtime_unknown_threshold,
            "status": "success" if any(item.get("status") == "success" for item in faces) else "error",
            "model_key": realtime_model_key,
            "model_label": MODEL_LABELS[realtime_model_key],
            "profile_title": profile["title"],
        }
    except Exception as exc:
        realtime_result = {
            "identity": None,
            "confidence": 0.0,
            "raw_confidence": 0.0,
            "final_confidence": 0.0,
            "knn_vote_ratio": None,
            "knn_distance_confidence": None,
            "bbox": None,
            "faces": [],
            "face_count": 0,
            "display_state": "error",
            "unknown_threshold": realtime_unknown_threshold,
            "status": "error",
            "error": str(exc),
            "model_key": realtime_model_key,
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
        for item in realtime_result.get("faces", []):
            x1, y1, x2, y2 = item["bbox"]
            color = _realtime_overlay_color(item.get("display_state"))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                _realtime_overlay_label(item),
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
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


def _selected_model_types(selection: str | None) -> list[str]:
    key = _normalize_key(selection or "both")
    if key in {"both", "all", "compare"}:
        return ["pca_knn", "pca_svm"]
    if key in MODEL_LABELS:
        return [key]
    raise ValueError("Lua chon model khong hop le.")


def _model_output_path(bundle: dict[str, Any], model_type: str, output_name: str | None) -> Path:
    base_name = Path(str(output_name or "").strip()).stem
    safe_base = secure_filename(base_name).replace("-", "_")
    if not safe_base:
        safe_base = f"{bundle['preset_key']}_{time.strftime('%Y%m%d_%H%M%S')}"
    return MODELS / f"{model_type}_{safe_base}.joblib"


def _job_output(job: BuildJob, category: str, path: Path, label: str) -> None:
    job.outputs.append({"label": label, "relative_path": _proj(path), "category": category, "filename": _rel(path, DOWNLOAD_ROOTS[category])})


def _run_build_and_train(job: BuildJob, payload: dict[str, Any]) -> None:
    builder_params = _resolve_builder_params(payload)
    source_dir = Path(payload["source_dir"]).expanduser().resolve()
    source_stats = _scan_source_folder(source_dir)
    detector_name = UPLOAD_FACE_DETECTOR.upper()
    detector_mode = "MTCNN + align 5-point" if CUSTOM_BUILD_FACE_ALIGN else detector_name

    job.status = "running"
    job.progress = 2
    job.update_details(
        current_stage="initializing",
        detector=detector_name,
        detector_mode=detector_mode,
        detector_status="dang_khoi_tao",
        subjects_total=int(source_stats["subjects_total"]),
        images_total=int(source_stats["images_total"]),
        processed_images=0,
        kept_images=0,
        no_face_images=0,
        unreadable_images=0,
        rejected_images=0,
        aligned_images=0,
        alignment_failed_images=0,
    )
    job.log("Dang doc thu muc nguon va kiem tra cau truc folder.")
    job.progress = 6
    job.log(
        f"Da nap {source_stats['images_total']} anh thuoc {source_stats['subjects_total']} nguoi tu folder nguon."
    )
    if source_stats["empty_subjects"]:
        job.log(f"Bo qua {len(source_stats['empty_subjects'])} folder khong co anh hop le.")

    job.progress = 10
    job.update_details(current_stage="warming_detector")
    job.log(f"Dang khoi tao {detector_name} cho buoc build database.")
    detector_info = warmup_face_detector(UPLOAD_FACE_DETECTOR)
    job.update_details(
        detector_status=detector_info.get("status", "ready"),
        detector_backend=detector_info.get("backend", detector_name.lower()),
        detector_device=detector_info.get("device_label", detector_info.get("device", "CPU")),
    )
    job.log(
        f"Khoi tao {detector_name} thanh cong "
        f"({detector_info.get('backend', detector_name.lower())}, {detector_info.get('device_label', 'CPU')})."
    )

    last_logged_progress = {"current": -1}

    def build_progress_callback(info: dict[str, Any]) -> None:
        total = max(int(info.get("total", 0)), 1)
        current = int(info.get("current", 0))
        percent = int(round((current / total) * 100))
        kept = int(info.get("kept_samples", 0))
        no_face = int(info.get("skipped_no_face_samples", 0))
        unreadable = int(info.get("unreadable_samples", 0))
        rejected = int(info.get("rejected_extreme_samples", 0))
        aligned = int(info.get("face_aligned_samples", 0))
        alignment_failed = int(info.get("face_alignment_failed_samples", 0))
        subject = str(info.get("current_subject") or "")
        sample = str(info.get("current_sample") or "")

        job.progress = max(job.progress, min(12 + int((current / total) * 43), 55))
        job.update_details(
            current_stage="preprocessing",
            processed_images=current,
            kept_images=kept,
            no_face_images=no_face,
            unreadable_images=unreadable,
            rejected_images=rejected,
            aligned_images=aligned,
            alignment_failed_images=alignment_failed,
            current_subject=subject,
            current_sample=sample,
        )
        job.message = (
            f"Tien xu ly {current}/{total} anh ({percent}%) • giu {kept} • "
            f"khong thay bbox {no_face} • loi doc {unreadable}"
        )

        checkpoint = max(total // 8, 1)
        should_log = current in {0, 1, total} or current - last_logged_progress["current"] >= checkpoint
        if should_log:
            suffix = f" • dang xu ly {subject}/{sample}" if subject and sample else ""
            job.log(
                f"[Build] {current}/{total} anh ({percent}%) • giu {kept} • "
                f"khong thay bbox {no_face} • reject {rejected} • align ok {aligned}{suffix}"
            )
            last_logged_progress["current"] = current

    bundle = build_face_database_from_directory(
        source_dir=source_dir,
        database_name=payload.get("database_name") or source_dir.name,
        source_label=str(payload.get("source_label") or payload.get("database_name") or source_dir.name),
        output_root=PROCESSED,
        face_detection=UPLOAD_FACE_DETECTOR,
        face_align=CUSTOM_BUILD_FACE_ALIGN,
        face_padding_ratio=CUSTOM_BUILD_FACE_PADDING_RATIO,
        face_square_crop=CUSTOM_BUILD_FACE_SQUARE_CROP,
        face_crop_fallback=CUSTOM_BUILD_FACE_CROP_FALLBACK,
        min_face_area_ratio=0.0,
        processing_profile=CUSTOM_BUILD_PROCESSING_PROFILE,
        progress_callback=build_progress_callback,
    )
    out = Path(bundle["output_dir"])
    summary = bundle["summary"]
    processing_stats = summary.get("processing_stats", {})
    bundle_meta = {
        "bundle_dir": out,
        "bundle_relative": _proj(out),
        "dataset_key": "custom",
        "preset_key": str(summary.get("profile_slug") or out.name),
        "profile_key": f"custom__{summary.get('profile_slug') or out.name}",
        "title": str(summary.get("profile_title") or out.name),
    }
    job.progress = 60
    job.update_details(
        current_stage="database_ready",
        kept_images=int(summary.get("samples_total", 0)),
        no_face_images=int(processing_stats.get("skipped_no_face_samples", 0)),
        unreadable_images=int(processing_stats.get("unreadable_samples", 0)),
        rejected_images=int(processing_stats.get("rejected_extreme_samples", 0)),
        aligned_images=int(processing_stats.get("face_aligned_samples", 0)),
        alignment_failed_images=int(processing_stats.get("face_alignment_failed_samples", 0)),
    )
    job.log(
        f"Da tao database '{summary.get('profile_title', out.name)}' voi "
        f"{summary.get('classes_total', 0)} nguoi / {summary.get('samples_total', 0)} anh."
    )
    job.log(
        "Thong ke build: "
        f"khong thay bbox {processing_stats.get('skipped_no_face_samples', 0)} anh • "
        f"loi doc {processing_stats.get('unreadable_samples', 0)} anh • "
        f"reject chat luong {processing_stats.get('rejected_extreme_samples', 0)} anh • "
        f"align thanh cong {processing_stats.get('face_aligned_samples', 0)} anh."
    )
    for name, label in [
        ("summary.json", "Summary"),
        ("inputs.npz", "Inputs"),
        ("manifest.csv", "Manifest"),
        ("label_mapping.json", "Label mapping"),
    ]:
        path = out / name
        if path.exists():
            _job_output(job, "processed", path, label)
    data = load_processed_dataset_bundle(out)
    X_train = np.asarray(data["X"], dtype=np.float64)
    y_train = np.asarray(data["y"])
    n_components = int(builder_params["n_components"])
    model_types = _selected_model_types(payload.get("model_scope"))
    total_models = len(model_types)
    saved_model_paths: list[Path] = []
    try:
        for index, model_type in enumerate(model_types, start=1):
            progress_base = 62 + int(((index - 1) / total_models) * 28)
            job.progress = progress_base
            job.update_details(
                current_stage="training",
                training_model=MODEL_LABELS[model_type],
                training_index=index,
                training_total=total_models,
            )
            if model_type == "pca_knn":
                job.log(
                    f"Dang train {MODEL_LABELS[model_type]} ({index}/{total_models}) "
                    f"voi n_components={n_components}, k={int(builder_params['k'])}, "
                    f"metric={builder_params['metric']}."
                )
            else:
                job.log(
                    f"Dang train {MODEL_LABELS[model_type]} ({index}/{total_models}) "
                    f"voi n_components={n_components}, C={float(builder_params['C'])}, "
                    f"kernel={builder_params['kernel']}, gamma={builder_params['gamma']}."
                )
            if model_type == "pca_knn":
                model = train_pca_knn(
                    X_train,
                    y_train,
                    n_components=n_components,
                    k=int(builder_params["k"]),
                    metric=builder_params["metric"],
                )
            else:
                model = train_pca_svm(
                    X_train,
                    y_train,
                    n_components=n_components,
                    C=float(builder_params["C"]),
                    kernel=builder_params["kernel"],
                    gamma=builder_params["gamma"],
                )
            model_path = _model_output_path(bundle_meta, model_type, payload.get("output_name"))
            model.save(model_path)
            _write_model_meta(model_path, bundle_meta, model_type)
            saved_model_paths.append(model_path)
            _job_output(job, "saved-models", model_path, "Model")
            meta_path = _model_meta_path(model_path)
            if meta_path.exists():
                _job_output(job, "saved-models", meta_path, "Model metadata")
            job.log(f"Da luu {MODEL_LABELS[model_type]}: {_proj(model_path)}")
            job.progress = min(progress_base + 14, 92)
    except Exception:
        for model_path in saved_model_paths:
            _forget_model_caches(model_path)
            model_path.unlink(missing_ok=True)
            _model_meta_path(model_path).unlink(missing_ok=True)
        raise

    removed_models = _prune_profile_models(bundle_meta, saved_model_paths)
    if removed_models:
        job.log(f"Da thay the {len(removed_models)} model cu trung ten cua database hien tai.")

    job.update_details(current_stage="completed", training_model=None)
    job.progress = 100
    job.status = "completed"
    job.log("Hoan tat build database va train model cho muc dich demo nhan dien truc tiep.")


def _start_job(job_type: str, payload: dict[str, Any]) -> BuildJob:
    job = BuildJob(id=uuid.uuid4().hex, job_type=job_type)
    build_jobs[job.id] = job

    def runner() -> None:
        try:
            _run_build_and_train(job, payload)
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.update_details(current_stage="failed")
            job.log(f"That bai: {exc}")
        finally:
            cleanup_source_dir = str(payload.get("cleanup_source_dir", "")).strip()
            if cleanup_source_dir:
                shutil.rmtree(cleanup_source_dir, ignore_errors=True)

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
        "default_unknown_threshold": DEFAULT_UNKNOWN_THRESHOLD,
    }


@app.template_filter("percent")
def percent_filter(value: float | None) -> str:
    return _percent(value)


@app.route("/", methods=["GET", "POST"])
def home():
    ctx = _base_context("home", "Nhan dang")
    selected = request.form.get("profile_key") or ctx["selected_profile_key"]
    unknown_threshold = _parse_unknown_threshold(request.form.get("unknown_threshold"), DEFAULT_UNKNOWN_THRESHOLD)
    profile = ctx["profile_lookup"].get(selected)
    if profile is None and ctx["profiles"]:
        profile = ctx["profiles"][0]
        selected = profile["key"]
    result = None
    image_name = None
    preview_image_name = None
    error_message = None
    if request.method == "POST":
        image = request.files.get("image")
        image_path = None
        original_image_name = request.form.get("original_image_name") or None
        if image is not None and image.filename:
            image_path, saved_name = _save_upload(image)
            image_name = saved_name
            original_image_name = image.filename
        else:
            existing_image_name = request.form.get("existing_image_name", "")
            image_path = _resolve_uploaded_image(existing_image_name)
            if image_path is not None:
                image_name = image_path.name
        if image_path is None:
            error_message = "Vui long chon anh."
        elif profile is None:
            error_message = "Chua co profile nao san sang."
        else:
            try:
                faces = _recognize_faces(
                    profile,
                    image_path,
                    detector=UPLOAD_FACE_DETECTOR,
                    unknown_threshold=unknown_threshold,
                )
                if not faces:
                    error_message = "Khong co khuon mat nao du lon de nhan dang trong anh."
                else:
                    preview_image_name = _save_annotated_upload(image_path, faces)
                    result = {"faces": faces, "face_count": len(faces)}
                    image_name = image_name or image_path.name
                    if original_image_name:
                        result["original_image_name"] = original_image_name
            except Exception as exc:
                error_message = str(exc)
    ctx.update(
        {
            "selected_profile_key": selected,
            "selected_profile": profile,
            "result": result,
            "image_name": image_name,
            "preview_image_name": preview_image_name or image_name,
            "unknown_threshold": unknown_threshold,
            "error_message": error_message,
        }
    )
    return render_template("home.html", **ctx)


@app.route("/about")
def about():
    ctx = _base_context("about", "About")
    dataset_descriptions = {
        "orl": "Bộ benchmark cân bằng 40 người x 10 ảnh, dùng để kiểm tra pipeline gốc trên ảnh test raw.",
        "extended_yale_b": "Hai profile 20 người x 50 ảnh: bản khắc nghiệt giữ điều kiện gốc và bản đã tăng cường tương phản mạnh.",
        "custom": "Database tự build cho demo triển khai. Builder dùng MTCNN + align + normalize, còn realtime lấy bbox bằng detector nhẹ của OpenCV rồi bám theo pipeline của bundle.",
    }
    grouped_profiles: dict[str, list[dict[str, Any]]] = {}
    for profile in ctx["profiles"]:
        grouped_profiles.setdefault(profile["dataset_key"], []).append(profile)

    dataset_cards = []
    for dataset_key in DATASET_ORDER:
        profiles = grouped_profiles.get(dataset_key, [])
        if not profiles:
            continue
        presets = sorted({_preset_label(profile["preset_key"]) for profile in profiles})
        dataset_cards.append(
            {
                "key": dataset_key,
                "label": _dataset_label(dataset_key),
                "profile_count": len(profiles),
                "presets": presets,
                "description": dataset_descriptions.get(dataset_key, ""),
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
            "benchmark_profile_total": sum(1 for profile in ctx["profiles"] if profile["dataset_key"] != "custom"),
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
    if not realtime_profile_key or realtime_profile_key not in ctx["profile_lookup"]:
        realtime_profile_key = _default_realtime_profile_key()
    ctx.update(
        {
            "selected_profile_key": realtime_profile_key or _default_realtime_profile_key() or ctx["selected_profile_key"],
            "realtime_model_key": realtime_model_key,
            "unknown_threshold": realtime_unknown_threshold,
        }
    )
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


@app.route("/set_realtime_unknown_threshold", methods=["POST"])
def set_realtime_unknown_threshold():
    global realtime_unknown_threshold, realtime_result
    data = request.get_json(silent=True) or {}
    realtime_unknown_threshold = _parse_unknown_threshold(data.get("unknown_threshold"), realtime_unknown_threshold)
    realtime_result["unknown_threshold"] = realtime_unknown_threshold
    return jsonify({"status": "success", "unknown_threshold": realtime_unknown_threshold})


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
    ctx["builder_defaults"] = dict(BUILDER_DEFAULTS)
    return render_template("database_builder.html", **ctx)


@app.route("/database-builder/build", methods=["POST"])
def build_database():
    if request.files:
        files = [file for file in request.files.getlist("source_files") if file and file.filename]
        data = request.form.to_dict(flat=True)
        try:
            source_dir, source_label = _save_uploaded_source_folder(files)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        data["source_dir"] = str(source_dir)
        data["source_label"] = source_label
        data["cleanup_source_dir"] = str(source_dir)
    else:
        data = request.get_json(silent=True) or {}
    job_type = data.get("job_type")
    if job_type != "build_and_train":
        _cleanup_payload_source_dir(data)
        return jsonify({"error": "Loai job khong hop le"}), 400
    source_dir = str(data.get("source_dir", "")).strip()
    database_name = str(data.get("database_name", "")).strip()
    if not source_dir:
        _cleanup_payload_source_dir(data)
        return jsonify({"error": "Vui long nhap thu muc nguon"}), 400
    if not database_name:
        _cleanup_payload_source_dir(data)
        return jsonify({"error": "Vui long nhap ten database"}), 400
    try:
        _selected_model_types(data.get("model_scope"))
        data.update(_resolve_builder_params(data))
    except ValueError as exc:
        _cleanup_payload_source_dir(data)
        return jsonify({"error": str(exc)}), 400
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
            "details": job.details,
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
        realtime_profile_key = _default_realtime_profile_key()
    app.run(debug=True)
