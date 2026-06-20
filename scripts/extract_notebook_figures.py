from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notebook_assets import extract_png_outputs


FIGURES_DIR = PROJECT_ROOT / "assets" / "figures"
MANIFEST_PATH = PROJECT_ROOT / "assets" / "gallery_manifest.json"
NOTEBOOK_PATH = PROJECT_ROOT / "plant_disease_project.ipynb"

CURATED_CLASS_LABELS = (
    "Apple Black Rot",
    "Apple Healthy",
    "Corn Northern Leaf Blight",
    "Corn Healthy",
    "Grape Black Rot",
    "Grape Leaf Blight",
    "Grape Healthy",
    "Peach Bacterial Spot",
    "Peach Healthy",
    "Tomato Bacterial Spot",
    "Tomato Late Blight",
    "Tomato Healthy",
)


def _manifest_item(image_path: Path, title: str, project_root: Path) -> dict[str, str]:
    return {
        "path": image_path.relative_to(project_root).as_posix(),
        "title": title,
        "category": "Leaf-segmented sample",
        "caption": (
            f"{title}: segmented/leaf-only sample from the original Colab output."
        ),
    }


def build_gallery_manifest(
    image_paths: list[Path], project_root: Path = PROJECT_ROOT
) -> list[dict[str, str]]:
    manifest = []
    for index, image_path in enumerate(image_paths):
        title = CURATED_CLASS_LABELS[index]
        manifest.append(_manifest_item(image_path, title, project_root))

    return manifest


def main() -> None:
    extracted = extract_png_outputs(NOTEBOOK_PATH, FIGURES_DIR, max_images=12)
    manifest = build_gallery_manifest(extracted)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")


if __name__ == "__main__":
    main()
