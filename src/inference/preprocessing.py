from __future__ import annotations

from PIL import Image


def ensure_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")
