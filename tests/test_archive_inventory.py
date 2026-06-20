from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pytest

from src.archive_inventory import inventory_zip


def test_inventory_zip_counts_files_by_class_and_extension(tmp_path):
    archive_path = tmp_path / "sample_dataset.zip"

    with ZipFile(archive_path, "w") as archive:
        archive.writestr("dataset/Class_A/image1.jpg", "image")
        archive.writestr("dataset/Class_A/image2.JPG", "image")
        archive.writestr("dataset/Class_B/image3.png", "image")
        archive.writestr("dataset/Class_C/", "")

    inventory = inventory_zip(archive_path)

    assert inventory.archive_name == "sample_dataset.zip"
    assert inventory.archive_size_bytes == archive_path.stat().st_size
    assert inventory.total_entries == 4
    assert inventory.total_files == 3
    assert inventory.class_counts == {"Class_A": 2, "Class_B": 1}
    assert inventory.extension_counts == {".jpg": 2, ".png": 1}


def test_inventory_zip_counts_zero_byte_regular_files(tmp_path):
    archive_path = tmp_path / "zero_byte_dataset.zip"

    with ZipFile(archive_path, "w") as archive:
        archive.writestr("dataset/Class_A/empty.jpg", "")
        archive.writestr("dataset/Class_A/", "")

    inventory = inventory_zip(archive_path)

    assert inventory.total_entries == 2
    assert inventory.total_files == 1
    assert inventory.class_counts == {"Class_A": 1}
    assert inventory.extension_counts == {".jpg": 1}


def test_inventory_zip_skips_metadata_and_malformed_paths(tmp_path):
    archive_path = tmp_path / "messy_dataset.zip"

    with ZipFile(archive_path, "w") as archive:
        archive.writestr("__MACOSX/Class_A/image1.jpg", "metadata")
        archive.writestr("image_at_root.jpg", "image")
        archive.writestr("dataset//image2.png", "image")
        archive.writestr("/dataset/Class_A/image3.png", "image")
        archive.writestr("dataset/Class_B/image4.JPG", "image")

    inventory = inventory_zip(archive_path)

    assert inventory.total_entries == 5
    assert inventory.total_files == 1
    assert inventory.class_counts == {"Class_B": 1}
    assert inventory.extension_counts == {".jpg": 1}


def test_inventory_zip_allows_root_folder_with_space(tmp_path):
    archive_path = tmp_path / "ground_truth.zip"

    with ZipFile(archive_path, "w") as archive:
        archive.writestr("ground truth/Class_A/image1.png", "image")
        archive.writestr("ground truth/Class_B/image2.PNG", "image")

    inventory = inventory_zip(archive_path)

    assert inventory.total_files == 2
    assert inventory.class_counts == {"Class_A": 1, "Class_B": 1}
    assert inventory.extension_counts == {".png": 2}


def test_inventory_zip_missing_file_raises_file_not_found():
    missing_path = Path("missing_dataset.zip")

    with pytest.raises(FileNotFoundError, match="missing_dataset.zip"):
        inventory_zip(missing_path)


def test_inventory_zip_invalid_zip_raises_bad_zip_file(tmp_path):
    archive_path = tmp_path / "not_a_zip.zip"
    archive_path.write_text("not a zip archive")

    with pytest.raises(BadZipFile, match="Not a zip file"):
        inventory_zip(archive_path)
