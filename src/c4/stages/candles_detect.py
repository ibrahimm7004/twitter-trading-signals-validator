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
    analysis_w = int(analysis_crop.shape[1])
    gray = cv2.cvtColor(analysis_crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges_clean = (edges > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(edges_clean, connectivity=8)
    removed_long_thin_count = 0
    for comp_id in range(1, num):
        comp_w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
        comp_area = int(stats[comp_id, cv2.CC_STAT_AREA])
        bbox_area = max(1.0, float(comp_w * comp_h))
        fill_ratio = float(comp_area) / bbox_area
        long_ratio = max(
            float(comp_w) / max(1.0, float(analysis_w)),
            float(comp_h) / max(1.0, float(plot_h)),
        )
        if long_ratio >= 0.35 and fill_ratio <= 0.05 and comp_area >= 60:
            edges_clean[labels == comp_id] = 0
            removed_long_thin_count += 1
    for comp_id in range(1, num):
        comp_w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
        comp_area = int(stats[comp_id, cv2.CC_STAT_AREA])
        h_ratio = float(comp_h) / max(1.0, float(plot_h))
        if h_ratio >= 0.60 and comp_w <= 8 and comp_area >= 30:
            edges_clean[labels == comp_id] = 0

    window = 9
    band_y0 = int(0.12 * float(plot_h))
    band_y1 = int(0.92 * float(plot_h))
    band_y0 = max(0, min(band_y0, plot_h))
    band_y1 = max(band_y0 + 1, min(band_y1, plot_h))
    hsv = cv2.cvtColor(analysis_crop, cv2.COLOR_BGR2HSV)
    teal_mask = cv2.inRange(hsv, (70, 61, 41), (110, 255, 255)).astype(np.uint8)
    teal_band = teal_mask[band_y0:band_y1, :]
    teal_frac = np.mean(teal_band > 0, axis=0) if teal_band.size > 0 else np.zeros((analysis_w,), dtype=np.float32)
    teal_cond = teal_frac > 0.50
    min_teal_run = int(max(40, 0.08 * float(analysis_w)))
    overlay_start_local: int | None = None
    overlay_teal_run_len = 0
    run_start: int | None = None
    best_run: tuple[int, int] | None = None
    for i, flag in enumerate(teal_cond.tolist()):
        if flag and run_start is None:
            run_start = i
        if (not flag) and run_start is not None:
            run_end = i - 1
            run_len = run_end - run_start + 1
            if run_len >= min_teal_run and (best_run is None or run_end > best_run[1]):
                best_run = (run_start, run_end)
            run_start = None
    if run_start is not None:
        run_end = len(teal_cond) - 1
        run_len = run_end - run_start + 1
        if run_len >= min_teal_run and (best_run is None or run_end > best_run[1]):
            best_run = (run_start, run_end)
    overlay_suppressed = False
    if best_run is not None:
        cand_start, cand_end = best_run
        if cand_start >= int(0.55 * float(analysis_w)):
            overlay_start_local = int(cand_start)
            overlay_teal_run_len = int(cand_end - cand_start + 1)
            edges_clean[:, overlay_start_local:] = 0
            overlay_suppressed = True

    band = edges_clean[band_y0:band_y1, :]

    col_counts = np.count_nonzero(band > 0, axis=0).astype(np.float32)
    occupancy = col_counts / max(1.0, float(band.shape[0]))
    col_counts[occupancy > 0.70] = 0.0
    if col_counts.size >= 1:
        pad = min(col_counts.size // 2, max(12, window * 2))
        if pad > 0:
            col_counts[:pad] = 0.0
            col_counts[-pad:] = 0.0

    smoothed = _smooth_scores(col_counts, window=window)
    if smoothed.size >= 1 and col_counts.size >= 1:
        pad = min(col_counts.size // 2, max(12, window * 2))
        if pad > 0:
            smoothed[:pad] = 0.0
            smoothed[-pad:] = 0.0
    max_smoothed = float(smoothed.max()) if smoothed.size else 0.0
    if max_smoothed <= 0.0:
        return StageResult(
            data=None,
            confidence=0.0,
            abstain=False,
            reasons=[],
            debug={
                "x_right_limit_local": x_right_limit_local,
                "max_smoothed": 0.0,
                "max_right": 0.0,
                "threshold": 0.0,
                "picked_index": None,
                "removed_long_thin_count": removed_long_thin_count,
                "band_y0": band_y0,
                "band_y1": band_y1,
                "analysis_w": analysis_w,
                "plot_h": plot_h,
                "overlay_start_local": overlay_start_local,
                "overlay_teal_run_len": overlay_teal_run_len,
                "overlay_suppressed": overlay_suppressed,
            },
        )

    n = int(smoothed.size)
    start = int(0.60 * n)
    max_right = float(smoothed[start:].max()) if n > 0 and start < n else 0.0
    if max_right <= 0.0:
        max_right = max_smoothed

    min_abs = max(3.0, 0.005 * float(plot_h))
    threshold = max(min_abs, 0.25 * max_right)
    idx = _pick_rightmost_run(smoothed.tolist(), threshold, min_run=4)
    if idx is None:
        return StageResult(
            data=None,
            confidence=0.0,
            abstain=False,
            reasons=[],
            debug={
                "x_right_limit_local": x_right_limit_local,
                "max_smoothed": max_smoothed,
                "max_right": max_right,
                "threshold": threshold,
                "picked_index": None,
                "removed_long_thin_count": removed_long_thin_count,
                "band_y0": band_y0,
                "band_y1": band_y1,
                "analysis_w": analysis_w,
                "plot_h": plot_h,
                "overlay_start_local": overlay_start_local,
                "overlay_teal_run_len": overlay_teal_run_len,
                "overlay_suppressed": overlay_suppressed,
            },
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
            "max_right": max_right,
            "threshold": threshold,
            "picked_index": int(idx),
            "removed_long_thin_count": removed_long_thin_count,
            "band_y0": band_y0,
            "band_y1": band_y1,
            "analysis_w": analysis_w,
            "plot_h": plot_h,
            "overlay_start_local": overlay_start_local,
            "overlay_teal_run_len": overlay_teal_run_len,
            "overlay_suppressed": overlay_suppressed,
        },
    )
