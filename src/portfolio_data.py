from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_csv_report(path: str | Path) -> list[dict[str, str]]:
    report_path = Path(path)

    if not report_path.exists():
        raise FileNotFoundError(str(report_path))

    with report_path.open("r", encoding="utf-8", newline="") as report_file:
        reader = csv.DictReader(report_file)
        if not reader.fieldnames:
            raise ValueError(f"CSV report is empty or missing a header: {report_path}")

        return list(reader)


def load_gallery_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)

    if not manifest_path.exists():
        return []

    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    if not isinstance(manifest, list):
        raise ValueError(f"Gallery manifest must be a list: {manifest_path}")

    for index, item in enumerate(manifest):
        if not isinstance(item, dict):
            raise ValueError(
                f"Gallery manifest {manifest_path} item at index {index} must be a dict"
            )

    return manifest
