from __future__ import annotations

from pathlib import Path
import shutil
import tarfile
from typing import Iterable
from urllib.request import Request, urlopen
from zipfile import ZipFile

from ..configs.config import DATA_DIR

CHUNK_SIZE = 1024 * 1024
DOWNLOAD_HEADERS = {"User-Agent": "Mozilla/5.0"}

EXTENDED_YALE_B_URL = "https://www.dropbox.com/s/nqyqe406eh5m04k/CroppedYale.zip?dl=1"
LFW_URL = "https://ndownloader.figshare.com/files/5976018"


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cleanup_extract_artifacts(base_dir: Path) -> None:
    macosx_dir = base_dir / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir)


def _safe_extract_members(base_dir: Path, members: Iterable[str]) -> None:
    resolved_base = base_dir.resolve()
    for member in members:
        resolved_member = (resolved_base / member).resolve()
        if not str(resolved_member).startswith(str(resolved_base)):
            raise ValueError(f"Unsafe archive member path detected: {member}")


def _download_file(url: str, destination: Path, force: bool = False) -> Path:
    if destination.exists() and not force:
        return destination

    _ensure_directory(destination.parent)
    request = Request(url, headers=DOWNLOAD_HEADERS)

    with urlopen(request) as response, destination.open("wb") as output_file:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            output_file.write(chunk)

    return destination


def _extract_zip(archive_path: Path, extract_dir: Path) -> Path:
    with ZipFile(archive_path) as archive:
        members = archive.namelist()
        _safe_extract_members(extract_dir, members)
        _ensure_directory(extract_dir)
        archive.extractall(extract_dir)
    return extract_dir


def _extract_tar(archive_path: Path, extract_dir: Path) -> Path:
    with tarfile.open(archive_path) as archive:
        members = archive.getnames()
        _safe_extract_members(extract_dir, members)
        _ensure_directory(extract_dir)
        archive.extractall(extract_dir)
    return extract_dir


def download_extended_yale_b_raw(
    data_root: str | Path = DATA_DIR,
    force: bool = False,
) -> Path:
    """Download the Cropped Yale archive used as the local raw Extended Yale B dataset."""
    raw_root = _ensure_directory(Path(data_root))
    archive_path = raw_root / "CroppedYale.zip"
    extracted_root = raw_root / "CroppedYale"

    if extracted_root.exists() and not force:
        return extracted_root

    _download_file(EXTENDED_YALE_B_URL, archive_path, force=force)
    _extract_zip(archive_path, raw_root)
    _cleanup_extract_artifacts(raw_root)
    return extracted_root


def download_lfw_raw(
    data_root: str | Path = DATA_DIR,
    force: bool = False,
) -> Path:
    """Download the original LFW archive and extract it under the raw data directory."""
    raw_root = _ensure_directory(Path(data_root))
    archive_path = raw_root / "lfw.tgz"
    extracted_root = raw_root / "lfw"

    if extracted_root.exists() and not force:
        return extracted_root

    _download_file(LFW_URL, archive_path, force=force)
    _extract_tar(archive_path, raw_root)
    _cleanup_extract_artifacts(raw_root)
    return extracted_root


def ensure_dataset_downloaded(dataset_name: str, force: bool = False) -> Path:
    normalized = dataset_name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"extended_yale_b", "extended_yale", "cropped_yale"}:
        return download_extended_yale_b_raw(data_root=DATA_DIR, force=force)
    if normalized == "lfw":
        return download_lfw_raw(data_root=DATA_DIR, force=force)
    if normalized in {"orl", "att_faces", "att"}:
        orl_roots = [Path(DATA_DIR), Path(DATA_DIR) / "orl"]
        for orl_root in orl_roots:
            if any(orl_root.glob("s*/*.pgm")):
                return orl_root
        raise FileNotFoundError(
            "ORL is not auto-downloaded by this helper. Place it under data/raw/orl "
            "or keep the existing legacy layout under data/raw/."
        )
    raise ValueError(f"Unsupported dataset name: {dataset_name}")
