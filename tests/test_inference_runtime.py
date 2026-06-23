from pathlib import Path

from src.inference.runtime import check_weight_files, optional_import_available


def test_optional_import_available_reports_missing_module() -> None:
    assert optional_import_available("definitely_missing_inference_dependency") is False


def test_check_weight_files_reports_missing_paths(tmp_path: Path) -> None:
    present = tmp_path / "present.pt"
    missing = tmp_path / "missing.pt"
    present.write_bytes(b"weights")

    result = check_weight_files({"present": present, "missing": missing})

    assert result["present"] is True
    assert result["missing"] is False
