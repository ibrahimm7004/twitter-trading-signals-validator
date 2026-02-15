"""Frame localization stage (deterministic OpenCV)."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import C4Config
from ..schema import ChartFrame
from ..types import AbstainReason, ReasonCode, StageResult


def _largest_contour_bbox(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w <= 0 or h <= 0:
        return None
    return x, y, x + w, y + h


def _clean_text_mask(gray: np.ndarray, bright_thresh: int) -> np.ndarray:
    mask = (gray > bright_thresh).astype(np.uint8) * 255
    if mask.size == 0:
        return mask
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horiz_kernel)
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vert_kernel)
    cleaned = cv2.subtract(mask, horiz)
    cleaned = cv2.subtract(cleaned, vert)
    return cleaned


def _detect_red_badge_top(axis_bgr: np.ndarray, config: C4Config) -> int | None:
    h, w = axis_bgr.shape[:2]
    if h == 0 or w == 0:
        return None
    hsv = cv2.cvtColor(axis_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 120, 70], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 120, 70], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    bottom_start = int(h * 0.65)
    bottom_mask = red_mask[bottom_start:h, :]
    if int(bottom_mask.sum() / 255) < config.runtime.axis_red_badge_min_pixels:
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bottom_mask, connectivity=8)
    if num_labels <= 1:
        return None
    largest_idx = max(range(1, num_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    x, y, w_cc, h_cc, area = stats[largest_idx]
    if area < config.runtime.axis_red_badge_min_pixels:
        return None
    if (bottom_start + y + h_cc) < h - 3:
        return None
    if w_cc < int(0.40 * w):
        return None
    if h_cc < int(0.06 * h):
        return None
    return int(bottom_start + y)


def _tick_rows_from_text_components(axis_bgr: np.ndarray, config: C4Config) -> tuple[list[int], dict]:
    h, w = axis_bgr.shape[:2]
    if h == 0 or w == 0:
        return [], {"tick_rows_count": 0, "tick_rows_span": 0, "tick_thresh_used": None, "insufficient": True}
    gray = cv2.cvtColor(axis_bgr, cv2.COLOR_BGR2GRAY)
    thresholds = [
        config.runtime.axis_text_bright_thresh,
        150,
        140,
        130,
        120,
        110,
    ]
    badge_top = _detect_red_badge_top(axis_bgr, config)
    badge_cut = None
    if badge_top is not None:
        badge_cut = max(0, badge_top - config.runtime.axis_red_badge_margin_px)

    best_rows: list[int] = []
    best_score = -1.0
    best_thresh: int | None = None
    insufficient = True
    for thresh in thresholds:
        mask = _clean_text_mask(gray, int(thresh))
        roi_x0 = int(w * config.runtime.axis_digit_roi_left_ratio)
        if roi_x0 > 0:
            mask[:, :roi_x0] = 0
        if badge_cut is not None:
            mask[badge_cut:, :] = 0

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        centers: list[int] = []
        for idx in range(1, num_labels):
            x, y, w_cc, h_cc, area = stats[idx]
            if area < config.runtime.axis_cc_min_area or area > config.runtime.axis_cc_max_area:
                continue
            if h_cc < config.runtime.axis_cc_min_h or h_cc > config.runtime.axis_cc_max_h:
                continue
            if w_cc < config.runtime.axis_cc_min_w or w_cc > config.runtime.axis_cc_max_w:
                continue
            y_center = int(round(y + h_cc / 2.0))
            if badge_cut is not None and y_center >= badge_cut:
                continue
            centers.append(y_center)

        if not centers:
            continue

        centers.sort()
        groups: list[list[int]] = []
        tol = config.runtime.axis_row_merge_tol_px
        for y in centers:
            if not groups or abs(y - groups[-1][-1]) > tol:
                groups.append([y])
            else:
                groups[-1].append(y)

        rows = [int(round(sum(g) / len(g))) for g in groups]
        span = max(rows) - min(rows) if rows else 0
        span_ratio = span / float(max(1, h))
        score = len(rows) + span_ratio
        if len(rows) >= config.thresholds.min_ticks and span_ratio >= 0.35:
            insufficient = False
            score += 10.0
        if span_ratio < 0.20:
            score -= 5.0
        if score > best_score:
            best_score = score
            best_rows = rows
            best_thresh = int(thresh)

    span = (max(best_rows) - min(best_rows)) if best_rows else 0
    return best_rows, {
        "tick_rows_count": len(best_rows),
        "tick_rows_span": int(span),
        "tick_thresh_used": best_thresh,
        "insufficient": insufficient,
        "badge_top": badge_top,
    }


def _axis_bbox_from_components(
    band_gray: np.ndarray,
    band_offset_x: int,
    panel_bbox: tuple[int, int, int, int],
    config: C4Config,
    axis_right_edge: int,
) -> tuple[int, int, int, int, int, float]:
    mask = _clean_text_mask(band_gray, config.runtime.axis_text_bright_thresh)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept: list[tuple[int, int, int, int]] = []
    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        if area < config.runtime.axis_cc_min_area or area > config.runtime.axis_cc_max_area:
            continue
        if h < config.runtime.axis_cc_min_h or h > config.runtime.axis_cc_max_h:
            continue
        kept.append((x, y, w, h))

    band_h, band_w = band_gray.shape[:2]
    axis_cc_count = len(kept)
    axis_cc_density = axis_cc_count / float(max(1, band_h * band_w))

    if kept:
        x_min = min(x for x, _, _, _ in kept)
        padding = 6
        axis_x0 = band_offset_x + max(0, x_min - padding)
    else:
        axis_x0 = band_offset_x + max(0, band_w - config.runtime.axis_min_width_px)

    panel_x0, panel_y0, panel_x1, panel_y1 = panel_bbox
    axis_x1 = axis_right_edge
    axis_width = axis_x1 - axis_x0
    if axis_width < config.runtime.axis_min_width_px:
        axis_x0 = max(panel_x0, axis_x1 - config.runtime.axis_min_width_px)
    if axis_width > config.runtime.axis_max_width_px:
        axis_x0 = axis_x1 - config.runtime.axis_max_width_px

    axis_x0 = max(panel_x0, axis_x0)
    axis_y0, axis_y1 = panel_y0, panel_y1
    return axis_x0, axis_y0, axis_x1, axis_y1, axis_cc_count, axis_cc_density


def run(image_path: str, config: C4Config) -> StageResult[ChartFrame]:
    image = cv2.imread(str(image_path))
    if image is None:
        reason = AbstainReason(
            code=ReasonCode.IMAGE_READ_FAILED,
            stage="frame",
            message="Failed to read image.",
            details={"image_path": str(image_path)},
        )
        chart_frame = ChartFrame(
            plot_bbox=[0.0, 0.0, 0.0, 0.0],
            axis_bbox=[0.0, 0.0, 0.0, 0.0],
            scale="unknown",
            confidence=0.0,
        )
        return StageResult(data=chart_frame, confidence=0.0, abstain=True, reasons=[reason])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    panel_bbox = _largest_contour_bbox(closed)
    h, w = gray.shape[:2]
    confidence = 1.0
    reasons: list[AbstainReason] = []

    if panel_bbox is None:
        panel_bbox = (0, 0, w - 1, h - 1)
        confidence -= 0.6
    else:
        panel_area = (panel_bbox[2] - panel_bbox[0]) * (panel_bbox[3] - panel_bbox[1])
        if panel_area < 0.1 * (w * h):
            confidence -= 0.4

    panel_x0, panel_y0, panel_x1, panel_y1 = panel_bbox

    band_width = max(
        int(round(w * config.runtime.axis_right_band_ratio)),
        config.runtime.axis_min_width_px,
    )
    band_x1 = w - 1
    band_x0 = max(0, band_x1 - band_width)
    band_gray = gray[panel_y0:panel_y1, band_x0 : band_x1 + 1]

    axis_x0, axis_y0, axis_x1, axis_y1, axis_cc_count, axis_cc_density = _axis_bbox_from_components(
        band_gray, band_x0, panel_bbox, config, axis_right_edge=band_x1
    )

    axis_crop = image[axis_y0:axis_y1, axis_x0:axis_x1].copy()
    tick_rows, tick_debug = _tick_rows_from_text_components(axis_crop, config)
    if not tick_rows:
        confidence -= 0.2

    if axis_cc_count < config.runtime.axis_cc_min_count:
        confidence -= 0.35
    elif axis_cc_density < 0.0001:
        confidence -= 0.2

    if tick_rows:
        span_ratio = (max(tick_rows) - min(tick_rows)) / float(max(1, axis_crop.shape[0]))
        pad_top = config.runtime.axis_tick_pad_top_px
        pad_bot = config.runtime.axis_tick_pad_bot_px
        can_tighten = (
            len(tick_rows) >= config.thresholds.min_ticks
            and span_ratio >= 0.35
            and not tick_debug.get("insufficient", True)
        )
        if can_tighten:
            rel_y0 = max(0, min(tick_rows) - pad_top)
            rel_y1 = min(axis_crop.shape[0] - 1, max(tick_rows) + pad_bot)
            proposed_height = max(1, rel_y1 - rel_y0)
            if proposed_height >= int(0.60 * axis_crop.shape[0]):
                axis_y0 = max(panel_y0, axis_y0 + rel_y0)
                axis_y1 = min(panel_y1, axis_y0 + proposed_height)
                axis_crop = image[axis_y0:axis_y1, axis_x0:axis_x1].copy()
                tick_rows, tick_debug = _tick_rows_from_text_components(axis_crop, config)

    badge_top = tick_debug.get("badge_top")
    if badge_top is not None:
        if badge_top >= int(0.60 * axis_crop.shape[0]):
            badge_cut = max(0, badge_top - config.runtime.axis_red_badge_margin_px)
            axis_y1 = max(axis_y0 + 50, min(axis_y1, axis_y0 + badge_cut))
            axis_crop = image[axis_y0:axis_y1, axis_x0:axis_x1].copy()
            tick_rows, tick_debug = _tick_rows_from_text_components(axis_crop, config)

    if axis_x1 <= axis_x0 or axis_y1 <= axis_y0:
        confidence -= 0.6
    if (w - 1) - axis_x1 > config.runtime.axis_right_edge_max_gap:
        confidence -= 0.6

    plot_bbox = [float(panel_x0), float(panel_y0), float(axis_x0), float(panel_y1)]
    axis_bbox = [float(axis_x0), float(axis_y0), float(axis_x1), float(axis_y1)]

    if confidence < config.thresholds.frame_min_conf:
        reasons.append(
            AbstainReason(
                code=ReasonCode.FRAME_NOT_FOUND,
                stage="frame",
                message="Chart frame/axis confidence below threshold.",
                details={
                    "min_conf": config.thresholds.frame_min_conf,
                    "observed_conf": max(0.0, float(confidence)),
                },
            )
        )

    chart_frame = ChartFrame(
        plot_bbox=plot_bbox,
        axis_bbox=axis_bbox,
        scale="unknown",
        confidence=max(0.0, float(confidence)),
    )
    debug = {
        "panel_bbox": [float(panel_x0), float(panel_y0), float(panel_x1), float(panel_y1)],
        "axis_bbox": axis_bbox,
        "plot_bbox": plot_bbox,
        "axis_band_bbox": [float(band_x0), float(panel_y0), float(band_x1), float(panel_y1)],
        "tick_rows": tick_rows,
        "tick_rows_count": tick_debug.get("tick_rows_count"),
        "tick_rows_span": tick_debug.get("tick_rows_span"),
        "tick_thresh_used": tick_debug.get("tick_thresh_used"),
        "badge_top": tick_debug.get("badge_top"),
        "axis_cc_count": int(axis_cc_count),
        "axis_cc_density": float(axis_cc_density),
    }
    return StageResult(
        data=chart_frame,
        confidence=max(0.0, float(confidence)),
        abstain=bool(reasons),
        reasons=reasons,
        debug=debug,
    )
