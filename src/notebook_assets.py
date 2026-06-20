from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any


def _png_payload_to_text(payload: str | list[str]) -> str:
    if isinstance(payload, str):
        return payload

    return "".join(payload)


def extract_png_outputs(
    notebook_path: str | Path, output_dir: str | Path, max_images: int = 12
) -> list[Path]:
    notebook_file = Path(notebook_path)
    figures_dir = Path(output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with notebook_file.open("r", encoding="utf-8") as notebook_handle:
        notebook: dict[str, Any] = json.load(notebook_handle)

    extracted: list[Path] = []
    for cell_index, cell in enumerate(notebook.get("cells", []), start=1):
        for output_index, output in enumerate(cell.get("outputs", []), start=1):
            data = output.get("data", {})
            png_payload = data.get("image/png")
            if png_payload is None:
                continue

            payload_text = _png_payload_to_text(png_payload)
            image_bytes = base64.b64decode(payload_text)
            image_path = (
                figures_dir
                / f"notebook_cell_{cell_index:02d}_output_{output_index:02d}.png"
            )
            image_path.write_bytes(image_bytes)
            extracted.append(image_path)

            if len(extracted) >= max_images:
                return extracted

    return extracted
