from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

CLASSIFIER_WEIGHTS = (
    ROOT
    / "baseline_models-20260620T213513Z-3-001"
    / "baseline_models"
    / "best_efficientnet_baseline.pth"
)
SEGMENTATION_WEIGHTS = ROOT / "efficientnet_best.pt"

CLASSIFIER_SIZE = 224
SEGMENTATION_SIZE = 384
