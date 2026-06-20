from __future__ import annotations

import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.archive_inventory import inventory_zip


ARCHIVES = (
    ("Original images", "dataset2.zip"),
    ("Leaf-segmented images", "seg_dataset2.zip"),
    ("Ground-truth lesion masks", "ground truth.zip"),
)

FIELDNAMES = (
    "dataset",
    "archive",
    "class_name",
    "file_count",
    "archive_total_files",
    "size_mb",
)


def build_inventory_csv(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset, archive_name in ARCHIVES:
        archive_path = PROJECT_ROOT / archive_name
        inventory = inventory_zip(archive_path)
        size_mb = round(inventory.archive_size_bytes / (1024 * 1024), 2)

        for class_name, file_count in inventory.class_counts.items():
            rows.append(
                {
                    "dataset": dataset,
                    "archive": inventory.archive_name,
                    "class_name": class_name,
                    "file_count": file_count,
                    "archive_total_files": inventory.total_files,
                    "size_mb": f"{size_mb:.2f}",
                }
            )

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    build_inventory_csv(PROJECT_ROOT / "reports" / "dataset_inventory.csv")


if __name__ == "__main__":
    main()
