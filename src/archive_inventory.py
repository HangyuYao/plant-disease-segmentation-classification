from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, ZipFile, is_zipfile

METADATA_ROOTS = {"__macosx"}


@dataclass(frozen=True)
class ArchiveInventory:
    archive_name: str
    archive_size_bytes: int
    total_entries: int
    total_files: int
    class_counts: dict[str, int]
    extension_counts: dict[str, int]


def _valid_dataset_file_parts(filename: str) -> list[str] | None:
    parts = filename.replace("\\", "/").split("/")

    if len(parts) < 3:
        return None

    root, class_name = parts[0], parts[1]
    file_name = parts[-1]
    if not root or not class_name or not file_name:
        return None

    if root.lower() in METADATA_ROOTS:
        return None

    return parts


def inventory_zip(path: str | Path) -> ArchiveInventory:
    archive_path = Path(path)

    if not archive_path.exists():
        raise FileNotFoundError(str(archive_path))

    if not is_zipfile(archive_path):
        raise BadZipFile(f"Not a zip file: {archive_path}")

    class_counts: Counter[str] = Counter()
    extension_counts: Counter[str] = Counter()

    with ZipFile(archive_path) as archive:
        entries = archive.infolist()
        files = []

        for entry in entries:
            if entry.is_dir():
                continue

            parts = _valid_dataset_file_parts(entry.filename)
            if parts is None:
                continue

            files.append(entry)
            class_counts[parts[1]] += 1

            suffix = Path(entry.filename).suffix.lower()
            if suffix:
                extension_counts[suffix] += 1

    return ArchiveInventory(
        archive_name=archive_path.name,
        archive_size_bytes=archive_path.stat().st_size,
        total_entries=len(entries),
        total_files=len(files),
        class_counts=dict(sorted(class_counts.items())),
        extension_counts=dict(sorted(extension_counts.items())),
    )
