from __future__ import annotations

import importlib.util
from pathlib import Path


def optional_import_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def check_weight_files(paths: dict[str, Path]) -> dict[str, bool]:
    return {name: path.is_file() for name, path in paths.items()}


def missing_inference_dependencies() -> list[str]:
    return [
        module
        for module in ["torch", "torchvision", "timm", "numpy", "PIL"]
        if not optional_import_available(module)
    ]
