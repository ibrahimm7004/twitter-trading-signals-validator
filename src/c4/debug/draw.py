"""Drawing utilities for debug overlays."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def _to_int_bbox(bbox: list[float]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def draw_bbox(image: np.ndarray, bbox: list[float], color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    x0, y0, x1, y1 = _to_int_bbox(bbox)
    cv2.rectangle(image, (x0, y0), (x1, y1), color=color, thickness=2)
    return image


def draw_horizontal_marks(
    image: np.ndarray, ys: Iterable[int], color: tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    h, w = image.shape[:2]
    for y in ys:
        if 0 <= y < h:
            cv2.line(image, (0, y), (w - 1, y), color=color, thickness=1)
    return image


def draw_boxes_with_labels(
    image: np.ndarray,
    boxes: Iterable[tuple[int, int, int, int]],
    labels: Iterable[str],
    color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    for (x0, y0, x1, y1), label in zip(boxes, labels):
        cv2.rectangle(image, (x0, y0), (x1, y1), color=color, thickness=1)
        cv2.putText(
            image,
            label,
            (x0, max(0, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return image
