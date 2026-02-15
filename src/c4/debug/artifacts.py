"""Debug artifact writer utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .draw import draw_bbox, draw_horizontal_marks


def ensure_debug_dir(debug_dir: str | Path) -> Path:
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_read_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        return np.zeros((256, 256, 3), dtype=np.uint8)
    return image


def _clip_bbox(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    x0_i = max(0, min(int(round(x0)), width - 1))
    y0_i = max(0, min(int(round(y0)), height - 1))
    x1_i = max(0, min(int(round(x1)), width - 1))
    y1_i = max(0, min(int(round(y1)), height - 1))
    if x1_i < x0_i:
        x0_i, x1_i = x1_i, x0_i
    if y1_i < y0_i:
        y0_i, y1_i = y1_i, y0_i
    return x0_i, y0_i, x1_i, y1_i


def write_overlay(
    image_path: str | Path,
    debug_dir: Path,
    filename: str,
    plot_bbox: list[float],
    axis_bbox: list[float],
    panel_bbox: list[float] | None = None,
    axis_band_bbox: list[float] | None = None,
) -> Path:
    image = _safe_read_image(image_path)
    if panel_bbox is not None:
        draw_bbox(image, panel_bbox, color=(255, 255, 0))
    if axis_band_bbox is not None:
        draw_bbox(image, axis_band_bbox, color=(255, 0, 255))
    draw_bbox(image, plot_bbox, color=(0, 255, 0))
    draw_bbox(image, axis_bbox, color=(0, 0, 255))
    cv2.putText(
        image,
        "panel/yellow band/magenta plot/green axis/red",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    out_path = debug_dir / filename
    cv2.imwrite(str(out_path), image)
    return out_path


def write_axis_debug(
    image_path: str | Path,
    debug_dir: Path,
    filename: str,
    axis_bbox: list[float],
    tick_rows: Iterable[int] | None = None,
) -> Path:
    image = _safe_read_image(image_path)
    h, w = image.shape[:2]
    x0, y0, x1, y1 = _clip_bbox(axis_bbox, w, h)
    crop = image[y0:y1, x0:x1].copy()
    if tick_rows is not None:
        draw_horizontal_marks(crop, tick_rows, color=(255, 0, 0))
    out_path = debug_dir / filename
    cv2.imwrite(str(out_path), crop)
    return out_path


def write_crop(
    image_path: str | Path,
    debug_dir: Path,
    filename: str,
    bbox: list[float],
) -> Path:
    image = _safe_read_image(image_path)
    h, w = image.shape[:2]
    x0, y0, x1, y1 = _clip_bbox(bbox, w, h)
    crop = image[y0:y1, x0:x1].copy()
    out_path = debug_dir / filename
    cv2.imwrite(str(out_path), crop)
    return out_path
