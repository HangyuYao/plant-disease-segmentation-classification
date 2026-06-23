import numpy as np
from PIL import Image

from src.inference.overlay import make_red_overlay, prepare_display_mask
from src.inference.preprocessing import ensure_rgb


def test_ensure_rgb_converts_uploaded_image() -> None:
    image = Image.new("RGBA", (4, 3), (10, 20, 30, 255))

    result = ensure_rgb(image)

    assert result.mode == "RGB"
    assert result.size == (4, 3)


def test_make_red_overlay_marks_masked_pixels() -> None:
    image = Image.new("RGB", (2, 2), (100, 100, 100))
    mask = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)

    overlay = make_red_overlay(image, mask, threshold=0.5, alpha=0.5)

    arr = np.array(overlay)
    assert tuple(arr[0, 0]) == (100, 100, 100)
    assert arr[0, 1, 0] > arr[0, 1, 1]
    assert arr[1, 0, 0] > arr[1, 0, 1]


def test_make_red_overlay_can_suppress_white_background() -> None:
    image = Image.new("RGB", (3, 1), color=(255, 255, 255))
    image.putpixel((1, 0), (80, 120, 40))
    mask = np.array([[0.95, 0.95, 0.95]], dtype=np.float32)

    overlay = make_red_overlay(image, mask, threshold=0.5, suppress_white_background=True)

    arr = np.array(overlay)
    assert tuple(arr[0, 0]) == (255, 255, 255)
    assert arr[0, 1, 0] > arr[0, 1, 1]
    assert tuple(arr[0, 2]) == (255, 255, 255)


def test_make_red_overlay_can_use_low_mask_response() -> None:
    image = Image.new("RGB", (2, 1), color=(80, 120, 40))
    mask = np.array([[0.1, 0.9]], dtype=np.float32)

    overlay = make_red_overlay(image, mask, threshold=0.5, response_mode="low")

    arr = np.array(overlay)
    assert arr[0, 0, 0] > arr[0, 0, 1]
    assert tuple(arr[0, 1]) == (80, 120, 40)


def test_prepare_display_mask_matches_selected_response_mode() -> None:
    mask = np.array([[0.1, 0.9]], dtype=np.float32)

    high = prepare_display_mask(mask, response_mode="high")
    low = prepare_display_mask(mask, response_mode="low")

    np.testing.assert_allclose(high, mask)
    np.testing.assert_allclose(low, 1.0 - mask)
