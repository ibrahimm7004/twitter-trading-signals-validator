"""Best-effort candle/time alignment stage."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..config import C4Config
from ..types import StageResult


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


def _smooth_scores(scores: np.ndarray, window: int = 9) -> np.ndarray:
    if scores.size == 0:
        return scores
    window = max(1, int(window))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(scores.astype(np.float32), kernel, mode="same")


def _pick_rightmost_run(scores: list[float], thresh: float, min_run: int) -> int | None:
    run_start: int | None = None
    run_len = 0
    best_idx: int | None = None
    for i, val in enumerate(scores):
        if val >= thresh:
            if run_start is None:
                run_start = i
                run_len = 1
            else:
                run_len += 1
            if run_len >= min_run:
                best_idx = i
        else:
            run_start = None
            run_len = 0
    return best_idx


def run(
    image_path: str | Path,
    plot_bbox: list[float],
    axis_bbox: list[float] | None,
    config: C4Config,
) -> StageResult[float | None]:
    image = cv2.imread(str(image_path))
    if image is None:
        return StageResult(data=None, confidence=0.0, abstain=False, reasons=[], debug={"image_read": False})

    h, w = image.shape[:2]
    px0, py0, px1, py1 = _clip_bbox(plot_bbox, w, h)
    if px1 <= px0 or py1 <= py0:
        return StageResult(data=None, confidence=0.0, abstain=False, reasons=[], debug={"plot_valid": False})

    plot_crop = image[py0:py1, px0:px1]
    plot_h, plot_w = plot_crop.shape[:2]
    if plot_h <= 0 or plot_w <= 0:
        return StageResult(data=None, confidence=0.0, abstain=False, reasons=[], debug={"plot_empty": True})

    x_right_limit_local = plot_w
    if axis_bbox is not None and len(axis_bbox) >= 1:
        axis_x0 = float(axis_bbox[0])
        if axis_x0 > px0:
            x_right_limit_local = min(plot_w, int(axis_x0 - px0))
    x_right_limit_local = max(1, x_right_limit_local)

    analysis_crop = plot_crop[:, :x_right_limit_local]
    gray = cv2.cvtColor(analysis_crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)

    col_counts = np.count_nonzero(edges > 0, axis=0).astype(np.float32)
    occupancy = col_counts / max(1.0, float(plot_h))
    col_counts[occupancy > 0.70] = 0.0

    smoothed = _smooth_scores(col_counts, window=9)
    max_smoothed = float(smoothed.max()) if smoothed.size else 0.0
    if max_smoothed <= 0.0:
        return StageResult(
            data=None,
            confidence=0.0,
            abstain=False,
            reasons=[],
            debug={"x_right_limit_local": x_right_limit_local, "max_smoothed": 0.0},
        )

    threshold = max(10.0, 0.18 * max_smoothed)
    idx = _pick_rightmost_run(smoothed.tolist(), threshold, min_run=6)
    if idx is None:
        return StageResult(
            data=None,
            confidence=0.0,
            abstain=False,
            reasons=[],
            debug={"x_right_limit_local": x_right_limit_local, "max_smoothed": max_smoothed, "threshold": threshold},
        )

    current_x_px = float(px0 + idx)
    confidence = float(np.clip(max_smoothed / max(1.0, plot_h * 0.25), 0.0, 1.0))
    return StageResult(
        data=current_x_px,
        confidence=confidence,
        abstain=False,
        reasons=[],
        debug={
            "x_right_limit_local": x_right_limit_local,
            "max_smoothed": max_smoothed,
            "threshold": threshold,
            "picked_index": int(idx),
        },
    )

