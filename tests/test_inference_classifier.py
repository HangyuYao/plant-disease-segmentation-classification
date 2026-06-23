from __future__ import annotations

from PIL import Image

from src.inference.classifier import classify_image
from src.inference.labels import CLASS_LABELS


class FakeClassifier:
    def __call__(self, batch):
        import torch

        scores = torch.zeros((batch.shape[0], len(CLASS_LABELS)), dtype=torch.float32)
        scores[:, 2] = 4.0
        scores[:, 4] = 2.0
        return scores


def test_classify_image_returns_sorted_display_labels() -> None:
    image = Image.new("RGB", (32, 32), color=(100, 120, 40))

    predictions = classify_image(FakeClassifier(), image, device="cpu", top_k=2)

    assert [prediction.label for prediction in predictions] == [
        "Corn___Northern_Leaf_Blight",
        "Grape___Black_rot",
    ]
    assert predictions[0].display == "Corn - Northern Leaf Blight"
    assert predictions[0].confidence > predictions[1].confidence
