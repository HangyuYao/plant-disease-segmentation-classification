import json
from pathlib import Path

import pytest

from src.portfolio_data import load_csv_report, load_gallery_manifest


def test_load_csv_report_returns_rows_as_dict_strings(tmp_path):
    report_path = tmp_path / "report.csv"
    report_path.write_text("model,score\nmobilenet,0.95\n", encoding="utf-8")

    rows = load_csv_report(report_path)

    assert rows == [{"model": "mobilenet", "score": "0.95"}]


def test_load_csv_report_missing_file_raises_file_not_found():
    missing_path = Path("missing.csv")

    with pytest.raises(FileNotFoundError, match="missing.csv"):
        load_csv_report(missing_path)


def test_load_csv_report_empty_csv_raises_value_error(tmp_path):
    report_path = tmp_path / "empty.csv"
    report_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty.csv"):
        load_csv_report(report_path)


def test_load_gallery_manifest_valid_json_list_of_dicts_returns_list(tmp_path):
    manifest_path = tmp_path / "gallery.json"
    manifest = [{"title": "Leaf", "image": "leaf.png"}]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rows = load_gallery_manifest(manifest_path)

    assert rows == manifest


def test_load_gallery_manifest_missing_file_returns_empty_list(tmp_path):
    manifest_path = tmp_path / "missing_gallery.json"

    assert load_gallery_manifest(manifest_path) == []


def test_load_gallery_manifest_non_list_json_raises_value_error(tmp_path):
    manifest_path = tmp_path / "gallery.json"
    manifest_path.write_text('{"title": "Leaf"}', encoding="utf-8")

    with pytest.raises(ValueError, match="gallery.json"):
        load_gallery_manifest(manifest_path)


def test_load_gallery_manifest_non_dict_item_raises_value_error_with_index(tmp_path):
    manifest_path = tmp_path / "gallery.json"
    manifest_path.write_text('[{"title": "Leaf"}, "bad item"]', encoding="utf-8")

    with pytest.raises(ValueError, match=r"gallery\.json.*index 1"):
        load_gallery_manifest(manifest_path)


def test_load_gallery_manifest_malformed_json_raises_json_decode_error(tmp_path):
    manifest_path = tmp_path / "gallery.json"
    manifest_path.write_text("[", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_gallery_manifest(manifest_path)
