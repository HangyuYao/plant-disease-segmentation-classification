from __future__ import annotations

import numpy as np
from PIL import Image


def prepare_display_mask(mask: np.ndarray, response_mode: str) -> np.ndarray:
    if response_mode == "high":
        return mask
    if response_mode == "low":
        return 1.0 - mask
    raise ValueError("response_mode must be 'high' or 'low'")


def make_red_overlay(
    image: Image.Image,
    mask: np.ndarray,
    threshold: float,
    alpha: float = 0.45,
    response_mode: str = "high",
    suppress_white_background: bool = False,
) -> Image.Image:
    base = np.array(image.convert("RGB")).astype(np.float32)
    if response_mode == "high":
        mask_bool = mask >= threshold
    elif response_mode == "low":
        mask_bool = mask <= threshold
    else:
        raise ValueError("response_mode must be 'high' or 'low'")

    if suppress_white_background:
        foreground = np.any(base < 245, axis=2)
        mask_bool = mask_bool & foreground

    red = np.zeros_like(base)
    red[..., 0] = 255

    blended = base.copy()
    blended[mask_bool] = (1 - alpha) * base[mask_bool] + alpha * red[mask_bool]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
