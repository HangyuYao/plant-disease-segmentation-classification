from __future__ import annotations

import base64
import json

from scripts.extract_notebook_figures import build_gallery_manifest
from src.notebook_assets import extract_png_outputs
from src.portfolio_data import load_gallery_manifest


PNG_BYTES = b"\x89PNG\r\n\x1a\nminimal-test-png"


def test_extract_png_outputs_decodes_string_and_list_payloads(tmp_path):
    png_payload = base64.b64encode(PNG_BYTES).decode("ascii")
    notebook = {
        "cells": [
            {
                "outputs": [
                    {"data": {"image/png": png_payload}},
                    {"data": {"image/png": [png_payload[:8], png_payload[8:]]}},
                ]
            },
            {"outputs": [{"data": {"text/plain": "not an image"}}]},
            {"outputs": [{"data": {"image/png": png_payload}}]},
        ]
    }
    notebook_path = tmp_path / "notebook.ipynb"
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")

    extracted = extract_png_outputs(notebook_path, tmp_path / "figures", max_images=2)

    assert [path.name for path in extracted] == [
        "notebook_cell_01_output_01.png",
        "notebook_cell_01_output_02.png",
    ]
    assert [path.read_bytes() for path in extracted] == [PNG_BYTES, PNG_BYTES]


def test_gallery_manifest_uses_curated_labels_and_portfolio_contract(tmp_path):
    figures_dir = tmp_path / "assets" / "figures"
    figures_dir.mkdir(parents=True)
    image_paths = []
    for output_id in range(2, 14):
        image_path = figures_dir / f"notebook_cell_07_output_{output_id:02d}.png"
        image_path.write_bytes(PNG_BYTES)
        image_paths.append(image_path)

    manifest = build_gallery_manifest(image_paths, project_root=tmp_path)
    manifest_path = tmp_path / "assets" / "gallery_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded_manifest = load_gallery_manifest(manifest_path)

    assert loaded_manifest == manifest
    assert [item["title"] for item in manifest] == [
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
    ]
    for item in manifest:
        assert item["category"] == "Leaf-segmented sample"
        assert item["path"].startswith("assets/figures/")
        assert "\\" not in item["path"]
        assert (tmp_path / item["path"]).exists()
        assert "segmented/leaf-only sample from the original Colab output" in item[
            "caption"
        ]
