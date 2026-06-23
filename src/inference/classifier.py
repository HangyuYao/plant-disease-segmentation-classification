from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

from src.inference.config import CLASSIFIER_SIZE
from src.inference.labels import CLASS_LABELS, display_label


@dataclass(frozen=True)
class Prediction:
    label: str
    display: str
    confidence: float


def build_efficientnet_classifier() -> torch.nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        len(CLASS_LABELS),
    )
    return model


def load_classifier(weight_path: Path, device: str) -> torch.nn.Module:
    model = build_efficientnet_classifier()
    state = torch.load(weight_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def classify_image(
    model: torch.nn.Module,
    image: Image.Image,
    device: str,
    top_k: int = 3,
) -> list[Prediction]:
    transform = transforms.Compose(
        [
            transforms.Resize((CLASSIFIER_SIZE, CLASSIFIER_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu()

    values, indices = torch.topk(probs, k=min(top_k, len(CLASS_LABELS)))
    return [
        Prediction(
            label=CLASS_LABELS[int(index)],
            display=display_label(CLASS_LABELS[int(index)]),
            confidence=float(value),
        )
        for value, index in zip(values, indices)
    ]
