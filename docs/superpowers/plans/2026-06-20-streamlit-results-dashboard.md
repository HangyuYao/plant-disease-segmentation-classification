# Streamlit Results Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean GitHub portfolio structure and Streamlit results dashboard for the plant disease segmentation/classification project without committing datasets, weights, or large outputs.

**Architecture:** The app will run from small curated artifacts in `reports/` and `assets/figures/`. Utility modules will read zip metadata and notebook outputs during project preparation, while the Streamlit app itself will not depend on large archives, model weights, or Google Colab paths.

**Tech Stack:** Python, Streamlit, pandas, Pillow, pytest, standard-library `zipfile` and `json`.

---

## Planned File Structure

- Create: `.gitignore` - excludes datasets, archives, weights, caches, and generated outputs.
- Create: `README.md` - portfolio-facing project narrative, setup, app usage, data policy, and future inference notes.
- Create: `requirements.txt` - runtime and lightweight development dependencies.
- Create: `data/README.md` - explains dataset archives and how to obtain/place them locally.
- Create: `app/streamlit_app.py` - dashboard UI.
- Create: `src/__init__.py` - package marker.
- Create: `src/archive_inventory.py` - reads dataset archive metadata without extracting archives.
- Create: `src/notebook_assets.py` - extracts selected embedded notebook PNG outputs into curated assets.
- Create: `src/portfolio_data.py` - loads curated metrics and gallery metadata for the app.
- Create: `reports/dataset_inventory.csv` - small dataset class-count summary.
- Create: `reports/experiment1_summary.csv` - original vs leaf-segmented classification summary.
- Create: `reports/segmentation_summary.csv` - segmentation model mIoU/F1/time summary.
- Create: `reports/experiment2_summary.csv` - lesion-only masked classification summary.
- Create: `assets/figures/README.md` - explains curated figures.
- Create: `assets/gallery_manifest.json` - lists selected figures, captions, and categories.
- Create: `tests/test_archive_inventory.py` - unit tests for archive metadata helpers.
- Create: `tests/test_portfolio_data.py` - unit tests for report and manifest loading.

## Task 1: Add GitHub-Safe Ignore Rules

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Create ignore rules**

Create `.gitignore` with:

```gitignore
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/

# Jupyter / Colab
.ipynb_checkpoints/

# Archives and datasets
*.zip
dataset2/
seg_dataset2/
ground truth/
ground_truth/
dataset2_split/
exp1_split/
exp2_cls_data_strict/

# Model weights
*.pt
*.pth
*.ckpt
*.onnx

# Generated outputs
results/
results.zip
seg_vis/
baseline_models/
exp2_results_strict_50/
outputs/
runs/

# Local / OS
.DS_Store
Thumbs.db
```

- [ ] **Step 2: Verify ignored files**

Run:

```powershell
git check-ignore dataset2.zip seg_dataset2.zip "ground truth.zip"
```

Expected after git initialization: all three zip paths are printed.

## Task 2: Create Archive Inventory Utility

**Files:**
- Create: `src/__init__.py`
- Create: `src/archive_inventory.py`
- Create: `tests/test_archive_inventory.py`

- [ ] **Step 1: Write tests**

Create `tests/test_archive_inventory.py`:

```python
from pathlib import Path
from zipfile import ZipFile

from src.archive_inventory import inventory_zip


def test_inventory_zip_counts_classes_and_extensions(tmp_path: Path) -> None:
    archive = tmp_path / "sample.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("dataset/Class_A/image1.jpg", b"abc")
        zf.writestr("dataset/Class_A/image2.JPG", b"abc")
        zf.writestr("dataset/Class_B/image3.png", b"abc")
        zf.writestr("dataset/Class_B/", b"")

    result = inventory_zip(archive)

    assert result.archive_name == "sample.zip"
    assert result.total_files == 3
    assert result.class_counts == {"Class_A": 2, "Class_B": 1}
    assert result.extension_counts == {".jpg": 2, ".png": 1}


def test_inventory_zip_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.zip"

    try:
        inventory_zip(missing)
    except FileNotFoundError as exc:
        assert "missing.zip" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
pytest tests/test_archive_inventory.py -v
```

Expected: fail because `src.archive_inventory` does not exist.

- [ ] **Step 3: Implement utility**

Create `src/__init__.py` as an empty file.

Create `src/archive_inventory.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile, is_zipfile


@dataclass(frozen=True)
class ArchiveInventory:
    archive_name: str
    archive_size_bytes: int
    total_entries: int
    total_files: int
    class_counts: dict[str, int]
    extension_counts: dict[str, int]


def inventory_zip(path: str | Path) -> ArchiveInventory:
    archive_path = Path(path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    if not is_zipfile(archive_path):
        raise ValueError(f"Not a valid zip archive: {archive_path}")

    class_counts: dict[str, int] = {}
    extension_counts: dict[str, int] = {}

    with ZipFile(archive_path) as archive:
        entries = archive.infolist()
        file_entries = [entry for entry in entries if entry.file_size > 0]
        for entry in file_entries:
            suffix = Path(entry.filename).suffix.lower()
            if suffix:
                extension_counts[suffix] = extension_counts.get(suffix, 0) + 1

            parts = entry.filename.split("/")
            if len(parts) > 2:
                class_name = parts[1]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

    return ArchiveInventory(
        archive_name=archive_path.name,
        archive_size_bytes=archive_path.stat().st_size,
        total_entries=len(entries),
        total_files=len(file_entries),
        class_counts=dict(sorted(class_counts.items())),
        extension_counts=dict(sorted(extension_counts.items())),
    )
```

- [ ] **Step 4: Run tests**

Run:

```powershell
pytest tests/test_archive_inventory.py -v
```

Expected: both tests pass.

## Task 3: Generate Small Dataset Inventory Report

**Files:**
- Create: `scripts/build_dataset_inventory.py`
- Create: `reports/dataset_inventory.csv`

- [ ] **Step 1: Create report builder**

Create `scripts/build_dataset_inventory.py`:

```python
from __future__ import annotations

import csv
from pathlib import Path

from src.archive_inventory import inventory_zip


ARCHIVES = [
    ("Original images", Path("dataset2.zip")),
    ("Leaf-segmented images", Path("seg_dataset2.zip")),
    ("Ground-truth lesion masks", Path("ground truth.zip")),
]


def main() -> None:
    rows: list[dict[str, str | int]] = []
    for label, archive_path in ARCHIVES:
        inventory = inventory_zip(archive_path)
        for class_name, count in inventory.class_counts.items():
            rows.append(
                {
                    "dataset": label,
                    "archive": inventory.archive_name,
                    "class_name": class_name,
                    "file_count": count,
                    "total_files": inventory.total_files,
                    "size_mb": round(inventory.archive_size_bytes / 1024 / 1024, 2),
                }
            )

    output = Path("reports/dataset_inventory.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "archive", "class_name", "file_count", "total_files", "size_mb"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run builder**

Run:

```powershell
& 'C:\Users\姚航宇\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/build_dataset_inventory.py
```

Expected: `reports/dataset_inventory.csv` exists and contains 31 data rows: 12 original, 12 segmented, 7 mask classes.

## Task 4: Create Curated Metric Summary Files

**Files:**
- Create: `reports/experiment1_summary.csv`
- Create: `reports/segmentation_summary.csv`
- Create: `reports/experiment2_summary.csv`

- [ ] **Step 1: Create Experiment 1 summary**

Create `reports/experiment1_summary.csv` from notebook results:

```csv
model,dataset,best_val_acc,notes
mobilenet,original,0.952,No-seg best observed validation accuracy from notebook output
efficientnet,original,0.958,No-seg best observed validation accuracy from notebook output
mcffa,original,0.956,No-seg best observed validation accuracy from notebook output
mobilenet,leaf_segmented,0.935,Leaf-seg best observed validation accuracy from notebook output
efficientnet,leaf_segmented,0.947,Leaf-seg best observed validation accuracy from notebook output
mcffa,leaf_segmented,0.956,Leaf-seg best observed validation accuracy from notebook output
```

- [ ] **Step 2: Create segmentation summary**

Create `reports/segmentation_summary.csv`:

```csv
model,miou,f1,seg_time_seconds_per_image
mobilenet,0.276,0.392,0.0097
efficientnet,0.343,0.455,0.0147
mcffa,0.273,0.382,0.0115
```

- [ ] **Step 3: Create Experiment 2 summary**

Create `reports/experiment2_summary.csv`:

```csv
seg_dataset,miou,f1,seg_time_seconds_per_image,classifier,val_acc,cls_time_seconds_per_image
mobilenet_masked,0.276,0.392,0.0097,mobilenet,0.489,0.0093
mobilenet_masked,0.276,0.392,0.0097,efficientnet,0.596,0.0062
mobilenet_masked,0.276,0.392,0.0097,mcffa,0.629,0.0079
efficientnet_masked,0.343,0.455,0.0147,mobilenet,0.500,0.0058
efficientnet_masked,0.343,0.455,0.0147,efficientnet,0.579,0.0062
efficientnet_masked,0.343,0.455,0.0147,mcffa,0.550,0.0076
mcffa_masked,0.273,0.382,0.0115,mobilenet,0.568,0.0059
mcffa_masked,0.273,0.382,0.0115,efficientnet,0.621,0.0064
mcffa_masked,0.273,0.382,0.0115,mcffa,0.582,0.0061
```

## Task 5: Create Portfolio Data Loader

**Files:**
- Create: `src/portfolio_data.py`
- Create: `tests/test_portfolio_data.py`

- [ ] **Step 1: Write tests**

Create `tests/test_portfolio_data.py`:

```python
from pathlib import Path

from src.portfolio_data import load_csv_report


def test_load_csv_report_returns_rows(tmp_path: Path) -> None:
    report = tmp_path / "report.csv"
    report.write_text("model,score\nmobilenet,0.95\n", encoding="utf-8")

    rows = load_csv_report(report)

    assert rows == [{"model": "mobilenet", "score": "0.95"}]


def test_load_csv_report_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"

    try:
        load_csv_report(missing)
    except FileNotFoundError as exc:
        assert "missing.csv" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
pytest tests/test_portfolio_data.py -v
```

Expected: fail because `src.portfolio_data` does not exist.

- [ ] **Step 3: Implement loader**

Create `src/portfolio_data.py`:

```python
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_csv_report(path: str | Path) -> list[dict[str, str]]:
    report_path = Path(path)
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")
    with report_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_gallery_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return []
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Gallery manifest must be a list")
    return data
```

- [ ] **Step 4: Run tests**

Run:

```powershell
pytest tests/test_portfolio_data.py -v
```

Expected: both tests pass.

## Task 6: Extract Curated Notebook Figures

**Files:**
- Create: `src/notebook_assets.py`
- Create: `scripts/extract_notebook_figures.py`
- Create: `assets/figures/README.md`
- Create: `assets/gallery_manifest.json`

- [ ] **Step 1: Implement notebook image extractor**

Create `src/notebook_assets.py`:

```python
from __future__ import annotations

import base64
import json
from pathlib import Path


def extract_png_outputs(notebook_path: str | Path, output_dir: str | Path, max_images: int = 12) -> list[Path]:
    notebook = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for cell_index, cell in enumerate(notebook.get("cells", []), start=1):
        for output_index, output in enumerate(cell.get("outputs", []), start=1):
            image_data = output.get("data", {}).get("image/png")
            if not image_data:
                continue
            encoded = "".join(image_data) if isinstance(image_data, list) else image_data
            image_bytes = base64.b64decode(encoded)
            image_path = destination / f"notebook_cell_{cell_index:02d}_output_{output_index:02d}.png"
            image_path.write_bytes(image_bytes)
            written.append(image_path)
            if len(written) >= max_images:
                return written
    return written
```

- [ ] **Step 2: Implement extraction script**

Create `scripts/extract_notebook_figures.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from src.notebook_assets import extract_png_outputs


def main() -> None:
    output_dir = Path("assets/figures")
    images = extract_png_outputs("plant_disease_project.ipynb", output_dir, max_images=12)

    manifest = [
        {
            "path": str(path).replace("\\", "/"),
            "title": path.stem.replace("_", " ").title(),
            "category": "Notebook output",
            "caption": "Curated visual output extracted from the original Colab notebook.",
        }
        for path in images
    ]
    Path("assets/gallery_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add figure README**

Create `assets/figures/README.md`:

```markdown
# Curated Figures

This folder contains a small set of portfolio-friendly figures extracted from the original Colab notebook. These images are intentionally lightweight and are committed so the Streamlit dashboard can run without the full datasets, model weights, or generated output folders.
```

- [ ] **Step 4: Run extraction**

Run:

```powershell
& 'C:\Users\姚航宇\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/extract_notebook_figures.py
```

Expected: up to 12 PNG files in `assets/figures/` and a JSON manifest at `assets/gallery_manifest.json`.

## Task 7: Build Streamlit Dashboard

**Files:**
- Create: `app/streamlit_app.py`

- [ ] **Step 1: Implement app**

Create `app/streamlit_app.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.portfolio_data import load_gallery_manifest


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
ASSETS = ROOT / "assets"


def read_report(name: str) -> pd.DataFrame:
    path = REPORTS / name
    if not path.exists():
        st.warning(f"Missing report: {path.name}")
        return pd.DataFrame()
    return pd.read_csv(path)


def metric_card(label: str, value: str) -> None:
    st.metric(label=label, value=value)


def main() -> None:
    st.set_page_config(
        page_title="Plant Disease Segmentation Portfolio",
        page_icon="🌿",
        layout="wide",
    )

    st.title("Plant Disease Segmentation and Classification")
    st.caption("A research portfolio dashboard built from curated Colab experiment artifacts.")

    tabs = st.tabs(
        [
            "Overview",
            "Dataset",
            "Experiment 1",
            "Segmentation",
            "Experiment 2",
            "Visual Gallery",
            "Future Inference",
        ]
    )

    with tabs[0]:
        st.subheader("Research Question")
        st.write(
            "This project studies whether plant disease classification improves when models use "
            "original images, leaf-segmented images, or lesion-only masked images."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("Image classes", "12")
        with col2:
            metric_card("Original images", "2,400")
        with col3:
            metric_card("Ground-truth masks", "350")
        st.subheader("Pipeline")
        st.write(
            "Original plant images are compared with leaf-segmented inputs. Separate lesion "
            "segmentation models produce disease-region masks, which are then used to create "
            "masked datasets for downstream classifier evaluation."
        )

    with tabs[1]:
        st.subheader("Dataset Inventory")
        df = read_report("dataset_inventory.csv")
        if not df.empty:
            st.dataframe(df, use_container_width=True)
            summary = df.groupby("dataset", as_index=False)["file_count"].sum()
            st.bar_chart(summary, x="dataset", y="file_count")

    with tabs[2]:
        st.subheader("Experiment 1: Original vs Leaf-Segmented Classification")
        df = read_report("experiment1_summary.csv")
        if not df.empty:
            st.dataframe(df, use_container_width=True)
            chart_df = df.assign(best_val_acc=pd.to_numeric(df["best_val_acc"]))
            st.bar_chart(chart_df, x="model", y="best_val_acc", color="dataset")

    with tabs[3]:
        st.subheader("Segmentation Model Comparison")
        df = read_report("segmentation_summary.csv")
        if not df.empty:
            st.dataframe(df, use_container_width=True)
            st.bar_chart(df, x="model", y="miou")

    with tabs[4]:
        st.subheader("Experiment 2: Lesion-Only Masked Classification")
        df = read_report("experiment2_summary.csv")
        if not df.empty:
            st.dataframe(df, use_container_width=True)
            chart_df = df.assign(val_acc=pd.to_numeric(df["val_acc"]))
            st.bar_chart(chart_df, x="classifier", y="val_acc", color="seg_dataset")

    with tabs[5]:
        st.subheader("Visual Gallery")
        manifest = load_gallery_manifest(ASSETS / "gallery_manifest.json")
        if not manifest:
            st.info("No curated figures have been extracted yet.")
        for item in manifest:
            image_path = ROOT / item["path"]
            if image_path.exists():
                st.image(str(image_path), caption=item.get("caption", ""), use_container_width=True)

    with tabs[6]:
        st.subheader("Future Inference Demo")
        st.info(
            "Live inference is intentionally out of scope for version 1. It can be added later "
            "after model weights, preprocessing, class labels, and hosting policy are cleaned up."
        )
        st.write(
            "The future version can add image upload, disease prediction, segmentation mask "
            "generation, and Grad-CAM overlays as an optional tab."
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run Streamlit smoke test**

Run:

```powershell
streamlit run app/streamlit_app.py
```

Expected: app opens locally, all tabs render, and no dataset archive or model weight is required.

## Task 8: Add Documentation And Requirements

**Files:**
- Create: `README.md`
- Create: `requirements.txt`
- Create: `data/README.md`

- [ ] **Step 1: Create requirements**

Create `requirements.txt`:

```text
streamlit
pandas
pillow
pytest
```

- [ ] **Step 2: Create data README**

Create `data/README.md`:

```markdown
# Data Policy

The full datasets are not committed to GitHub.

Expected local archives:

- `dataset2.zip`: original plant disease images
- `seg_dataset2.zip`: leaf-segmented plant images
- `ground truth.zip`: lesion mask ground truth

The Streamlit dashboard uses small curated summaries and figures, so it can run without these archives after preparation.
```

- [ ] **Step 3: Create root README**

Create `README.md` with these sections:

```markdown
# Plant Disease Segmentation and Classification

This project compares plant disease classification using original images, leaf-segmented images, and lesion-only masked images generated by segmentation models.

## What This Repository Contains

- A Streamlit research dashboard
- Curated experiment summaries
- Lightweight notebook-derived figures
- Dataset inventory metadata
- Reproducibility notes from the original Colab workflow

## What Is Not Committed

Datasets, zip archives, trained model weights, extracted data folders, and large generated outputs are intentionally excluded from GitHub.

## Run The Dashboard

```powershell
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Future Work

Live inference can be added later once model weights, preprocessing, label mappings, and model hosting are cleaned up.
```

## Task 9: Verify Repository Health

**Files:**
- No new files

- [ ] **Step 1: Run tests**

Run:

```powershell
pytest -v
```

Expected: archive and report loader tests pass.

- [ ] **Step 2: Verify ignored large files**

Run:

```powershell
git status --short
```

Expected after git initialization: source/docs/reports/assets appear, while zip archives and model weights do not appear as staged or untracked files.

- [ ] **Step 3: Verify dashboard does not depend on large files**

Temporarily move only the three zip archives outside the project or run from a copy without them, then run:

```powershell
streamlit run app/streamlit_app.py
```

Expected: dashboard still loads because it uses committed summaries and curated figures.

## Self-Review

- The plan covers the approved dashboard-first scope.
- Live inference is documented as future work, not implemented.
- Large datasets, zips, weights, and generated outputs are excluded.
- Every created runtime file has a clear responsibility.
- The app can run from curated small artifacts without Google Colab paths.
