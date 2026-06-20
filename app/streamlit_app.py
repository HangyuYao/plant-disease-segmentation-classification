from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.portfolio_data import load_csv_report, load_gallery_manifest


REPORTS_DIR = ROOT / "reports"
ASSETS_DIR = ROOT / "assets"
GALLERY_MANIFEST = ASSETS_DIR / "gallery_manifest.json"


st.set_page_config(
    page_title="Plant Disease Research Dashboard",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def read_report(filename: str) -> tuple[pd.DataFrame, str | None]:
    path = REPORTS_DIR / filename
    try:
        records = load_csv_report(path)
    except FileNotFoundError:
        return pd.DataFrame(), f"Missing report: {path.relative_to(ROOT)}"
    except Exception as exc:
        return pd.DataFrame(), f"Could not read {path.relative_to(ROOT)}: {exc}"

    return pd.DataFrame(records), None


@st.cache_data(show_spinner=False)
def read_gallery_manifest() -> tuple[list[dict[str, Any]], str | None]:
    try:
        manifest = load_gallery_manifest(GALLERY_MANIFEST)
    except Exception as exc:
        return [], f"Could not read {GALLERY_MANIFEST.relative_to(ROOT)}: {exc}"

    if not manifest:
        return [], f"No gallery entries found in {GALLERY_MANIFEST.relative_to(ROOT)}"

    return manifest, None


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def show_report_warning(message: str | None) -> None:
    if message:
        st.warning(message)


def show_missing_columns(df: pd.DataFrame, columns: list[str], label: str) -> bool:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        st.info(f"{label} needs column(s): {', '.join(missing)}.")
        return True
    return False


def pct(value: float | int | None) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def seconds(value: float | int | None) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.4f}s"


def overview_tab(
    dataset_df: pd.DataFrame,
    exp1_df: pd.DataFrame,
    segmentation_df: pd.DataFrame,
    exp2_df: pd.DataFrame,
    manifest: list[dict[str, Any]],
) -> None:
    st.title("Plant Disease Research Dashboard")
    st.caption(
        "A curated portfolio view of dataset inventory, experimental summaries, "
        "segmentation results, and selected visual outputs."
    )

    metric_cols = st.columns(5)
    with metric_cols[0]:
        classes = dataset_df["class_name"].nunique() if "class_name" in dataset_df else 0
        st.metric("Classes", f"{classes:,}")
    with metric_cols[1]:
        image_count = (
            pd.to_numeric(dataset_df["file_count"], errors="coerce").sum()
            if "file_count" in dataset_df
            else 0
        )
        st.metric("Curated files", f"{int(image_count):,}" if image_count else "n/a")
    with metric_cols[2]:
        archive_count = dataset_df["archive"].nunique() if "archive" in dataset_df else 0
        st.metric("Archives tracked", f"{archive_count:,}")
    with metric_cols[3]:
        best_exp1 = (
            pd.to_numeric(exp1_df["best_val_acc"], errors="coerce").max()
            if "best_val_acc" in exp1_df
            else None
        )
        st.metric("Best Exp. 1 val acc", pct(best_exp1))
    with metric_cols[4]:
        st.metric("Gallery images", f"{len(manifest):,}" if manifest else "n/a")

    st.subheader("Version 1 Scope")
    st.write(
        "This dashboard is designed to run from curated reports and image artifacts. "
        "It does not load datasets, open archive files, fetch model weights, or call "
        "notebook/Colab paths at runtime."
    )

    if not exp2_df.empty and {"seg_dataset", "classifier", "val_acc"}.issubset(exp2_df.columns):
        st.subheader("Best Experiment 2 combinations")
        exp2_numeric = numeric(exp2_df, ["val_acc"])
        top = exp2_numeric.sort_values("val_acc", ascending=False).head(5)
        st.dataframe(
            top[["seg_dataset", "classifier", "val_acc"]],
            hide_index=True,
            use_container_width=True,
        )


def dataset_tab(df: pd.DataFrame, warning: str | None) -> None:
    st.header("Dataset")
    st.write(
        "Inventory of the curated dataset archives and class-level file counts used "
        "for the portfolio reports."
    )
    show_report_warning(warning)
    if df.empty:
        return

    df = numeric(df, ["file_count", "archive_total_files", "size_mb"])
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Datasets", df["dataset"].nunique() if "dataset" in df else "n/a")
    with metric_cols[1]:
        st.metric("Classes", df["class_name"].nunique() if "class_name" in df else "n/a")
    with metric_cols[2]:
        total_files = df["file_count"].sum() if "file_count" in df else None
        st.metric("Class file rows total", f"{int(total_files):,}" if total_files else "n/a")
    with metric_cols[3]:
        archive_total = (
            df.drop_duplicates("archive")["archive_total_files"].sum()
            if {"archive", "archive_total_files"}.issubset(df.columns)
            else None
        )
        st.metric("Archive total files", f"{int(archive_total):,}" if archive_total else "n/a")

    if not show_missing_columns(df, ["dataset", "file_count"], "Dataset chart"):
        chart_df = df.groupby("dataset", as_index=False)["file_count"].sum()
        st.bar_chart(chart_df, x="dataset", y="file_count")

    if {"archive", "archive_total_files", "size_mb"}.issubset(df.columns):
        st.subheader("Archive summary")
        archive_summary = (
            df.groupby("archive", as_index=False)
            .agg(archive_total_files=("archive_total_files", "max"), size_mb=("size_mb", "max"))
            .sort_values("archive")
        )
        st.dataframe(archive_summary, hide_index=True, use_container_width=True)

    st.subheader("Class inventory")
    st.dataframe(df, hide_index=True, use_container_width=True)


def experiment1_tab(df: pd.DataFrame, warning: str | None) -> None:
    st.header("Experiment 1")
    st.write("Validation accuracy comparison for baseline classifiers on original and leaf-segmented inputs.")
    show_report_warning(warning)
    if df.empty:
        return

    df = numeric(df, ["best_val_acc"])
    if not show_missing_columns(df, ["model", "dataset", "best_val_acc"], "Experiment 1 chart"):
        chart_df = df.pivot_table(index="model", columns="dataset", values="best_val_acc", aggfunc="max")
        st.bar_chart(chart_df)

        if df["best_val_acc"].notna().any():
            best = df.loc[df["best_val_acc"].idxmax()]
            st.metric(
                "Best observed validation accuracy",
                pct(best["best_val_acc"]),
                f"{best['model']} on {best['dataset']}",
            )
        else:
            st.info("No numeric validation accuracy values are available for Experiment 1.")

    st.dataframe(df, hide_index=True, use_container_width=True)


def segmentation_tab(df: pd.DataFrame, warning: str | None) -> None:
    st.header("Segmentation")
    st.write("Segmentation quality and per-image timing from the curated experiment summary.")
    show_report_warning(warning)
    if df.empty:
        return

    df = numeric(df, ["miou", "f1", "seg_time_seconds_per_image"])
    metric_cols = st.columns(3)
    if "miou" in df:
        metric_cols[0].metric("Best mIoU", f"{df['miou'].max():.3f}")
    if "f1" in df:
        metric_cols[1].metric("Best F1", f"{df['f1'].max():.3f}")
    if "seg_time_seconds_per_image" in df:
        metric_cols[2].metric("Fastest segmentation", seconds(df["seg_time_seconds_per_image"].min()))

    if not show_missing_columns(df, ["model", "miou", "f1"], "Segmentation quality chart"):
        quality = df.set_index("model")[["miou", "f1"]]
        st.bar_chart(quality)

    if not show_missing_columns(df, ["model", "seg_time_seconds_per_image"], "Segmentation timing chart"):
        st.bar_chart(df, x="model", y="seg_time_seconds_per_image")

    st.dataframe(df, hide_index=True, use_container_width=True)


def experiment2_tab(df: pd.DataFrame, warning: str | None) -> None:
    st.header("Experiment 2")
    st.write(
        "Classifier results on masked/segmented datasets, shown alongside segmentation quality and timing."
    )
    show_report_warning(warning)
    if df.empty:
        return

    df = numeric(df, ["miou", "f1", "seg_time_seconds_per_image", "val_acc", "cls_time_seconds_per_image"])
    if not show_missing_columns(df, ["seg_dataset", "classifier", "val_acc"], "Experiment 2 heatmap table"):
        pivot = df.pivot_table(index="seg_dataset", columns="classifier", values="val_acc", aggfunc="max")
        st.subheader("Validation accuracy by segmentation dataset")
        st.dataframe(pivot.style.format("{:.3f}"), use_container_width=True)
        st.bar_chart(pivot)

        if df["val_acc"].notna().any():
            best = df.loc[df["val_acc"].idxmax()]
            st.metric(
                "Best Experiment 2 validation accuracy",
                pct(best["val_acc"]),
                f"{best['classifier']} on {best['seg_dataset']}",
            )
        else:
            st.info("No numeric validation accuracy values are available for Experiment 2.")

    if not show_missing_columns(
        df,
        ["classifier", "cls_time_seconds_per_image"],
        "Classifier timing chart",
    ):
        timing = df.groupby("classifier", as_index=False)["cls_time_seconds_per_image"].mean()
        st.subheader("Average classifier time per image")
        st.bar_chart(timing, x="classifier", y="cls_time_seconds_per_image")

    st.subheader("Experiment 2 summary table")
    st.dataframe(df, hide_index=True, use_container_width=True)


def gallery_tab(manifest: list[dict[str, Any]], warning: str | None) -> None:
    st.header("Visual Gallery")
    st.write("Curated visual outputs from the notebook, displayed from the local assets folder.")
    show_report_warning(warning)
    if not manifest:
        return

    categories = sorted({str(item.get("category", "Uncategorized")) for item in manifest})
    selected = st.multiselect("Category", categories, default=categories)
    visible_items = [
        item
        for item in manifest
        if str(item.get("category", "Uncategorized")) in set(selected)
    ]

    if not visible_items:
        st.info("No gallery images match the selected category.")
        return

    columns = st.columns(3)
    for index, item in enumerate(visible_items):
        image_path = ROOT / str(item.get("path", ""))
        title = str(item.get("title", image_path.name or "Untitled image"))
        caption = str(item.get("caption", ""))
        category = str(item.get("category", ""))

        with columns[index % 3]:
            st.subheader(title)
            if image_path.is_file():
                st.image(str(image_path), caption=caption or category, use_container_width=True)
            else:
                st.warning(f"Missing image: {image_path.relative_to(ROOT)}")
            if category:
                st.caption(category)


def future_inference_tab() -> None:
    st.header("Future Inference")
    st.info("Live inference is intentionally out of scope for version 1.")
    st.write(
        "The first release is a reproducible portfolio dashboard backed by curated "
        "summary files and selected visuals. A later inference workflow should be "
        "added only after the runtime dependencies and model artifacts are packaged "
        "deliberately."
    )
    st.subheader("Needed for a later inference release")
    st.markdown(
        """
- Exported model weights with documented architecture and preprocessing.
- A small, versioned sample input set or an upload workflow with validation.
- Runtime dependency lockfile and model-loading smoke tests.
- Clear class labels, normalization steps, and segmentation/inference pipeline order.
- Resource limits and user-facing error handling for unsupported images.
"""
    )


def main() -> None:
    dataset_df, dataset_warning = read_report("dataset_inventory.csv")
    exp1_df, exp1_warning = read_report("experiment1_summary.csv")
    segmentation_df, segmentation_warning = read_report("segmentation_summary.csv")
    exp2_df, exp2_warning = read_report("experiment2_summary.csv")
    manifest, manifest_warning = read_gallery_manifest()

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
        for warning in [dataset_warning, exp1_warning, segmentation_warning, exp2_warning, manifest_warning]:
            show_report_warning(warning)
        overview_tab(dataset_df, exp1_df, segmentation_df, exp2_df, manifest)
    with tabs[1]:
        dataset_tab(dataset_df, dataset_warning)
    with tabs[2]:
        experiment1_tab(exp1_df, exp1_warning)
    with tabs[3]:
        segmentation_tab(segmentation_df, segmentation_warning)
    with tabs[4]:
        experiment2_tab(exp2_df, exp2_warning)
    with tabs[5]:
        gallery_tab(manifest, manifest_warning)
    with tabs[6]:
        future_inference_tab()


if __name__ == "__main__":
    main()
