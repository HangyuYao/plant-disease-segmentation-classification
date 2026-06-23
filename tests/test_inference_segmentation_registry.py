from __future__ import annotations

from src.inference.config import ROOT
from src.inference.segmentation import SEGMENTATION_MODELS, SEBlock, build_segmenter


def test_segmentation_registry_includes_all_local_checkpoints() -> None:
    assert set(SEGMENTATION_MODELS) == {"efficientnet", "mobilenet", "mcffa"}
    assert SEGMENTATION_MODELS["efficientnet"].weight_path == ROOT / "efficientnet_best.pt"
    assert SEGMENTATION_MODELS["mobilenet"].weight_path == ROOT / "mobilenet_best.pt"
    assert SEGMENTATION_MODELS["mcffa"].weight_path == ROOT / "mcffa_best.pt"


def test_build_segmenter_creates_registered_architectures() -> None:
    for model_key in SEGMENTATION_MODELS:
        model = build_segmenter(model_key)
        assert model.__class__.__name__.lower().startswith(model_key)


def test_se_block_uses_training_checkpoint_key_names() -> None:
    keys = set(SEBlock(256).state_dict())

    assert "se.1.weight" in keys
    assert "se.3.weight" in keys
