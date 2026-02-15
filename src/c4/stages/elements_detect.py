"""Element detection stage for zones and lines in the plot area."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..models import ChartElement
from ..types import AbstainReason, ReasonCode, StageResult
from ..utils.price_mapper import make_price_mapper


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


def _annotation_mask(plot_crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(plot_crop, cv2.COLOR_BGR2HSV)
    sat_mask = cv2.inRange(hsv, (0, 80, 60), (180, 255, 255))

    gray = cv2.cvtColor(plot_crop, cv2.COLOR_BGR2GRAY)
    mean_gray = float(gray.mean())
    sat = hsv[:, :, 1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    if mean_gray < 140.0:
        stroke_mask = cv2.inRange(gray, 210, 255)
    else:
        dark_gray = cv2.inRange(gray, 0, 79)
        low_sat = cv2.inRange(sat, 0, 79)
        stroke_mask = cv2.bitwise_and(dark_gray, low_sat)

    stroke_mask = cv2.morphologyEx(stroke_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    stroke_mask = cv2.morphologyEx(stroke_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.bitwise_or(sat_mask, stroke_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _detect_zones(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    edges = cv2.Canny(mask, 60, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_h, mask_w = mask.shape[:2]
    max_area = 0.45 * float(mask_w * mask_h)
    zones: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if w < 40 or h < 12:
            continue
        area = float(w * h)
        if area > max_area:
            continue
        if w >= int(0.95 * mask_w) and h >= int(0.95 * mask_h):
            continue

        x0, y0, x1, y1 = x, y, x + w, y + h
        touches_left = x0 <= 2
        touches_right = x1 >= mask_w - 3
        touches_top = y0 <= 2
        touches_bottom = y1 >= mask_h - 3
        if touches_left and touches_right:
            continue
        if touches_top and touches_bottom:
            continue

        pts = approx.reshape(-1, 2)
        aligned = True
        for i in range(4):
            p0 = pts[i]
            p1 = pts[(i + 1) % 4]
            dx = float(p1[0] - p0[0])
            dy = float(p1[1] - p0[1])
            if dx == 0.0 and dy == 0.0:
                aligned = False
                break
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            angle = min(angle, 180.0 - angle)
            if not (angle <= 15.0 or abs(angle - 90.0) <= 15.0):
                aligned = False
                break
        if not aligned:
            continue
        zones.append((x, y, x + w, y + h))
    return zones


@dataclass
class _LineSeg:
    x1: int
    y1: int
    x2: int
    y2: int
    angle: float
    rho: float


def _segment_features(seg: tuple[int, int, int, int]) -> _LineSeg:
    x1, y1, x2, y2 = seg
    angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    if angle < 0:
        angle += 180.0
    rad = np.radians(angle)
    nx, ny = -np.sin(rad), np.cos(rad)
    rho = float(nx * x1 + ny * y1)
    return _LineSeg(x1, y1, x2, y2, angle, rho)


def _merge_collinear(lines: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not lines:
        return []
    feats = [_segment_features(s) for s in lines]
    clusters: list[list[_LineSeg]] = []
    for feat in feats:
        matched = False
        for cluster in clusters:
            a = float(np.mean([c.angle for c in cluster]))
            r = float(np.mean([c.rho for c in cluster]))
            if abs(feat.angle - a) <= 6.0 and abs(feat.rho - r) <= 18.0:
                cluster.append(feat)
                matched = True
                break
        if not matched:
            clusters.append([feat])

    merged: list[tuple[int, int, int, int]] = []
    for cluster in clusters:
        angle = float(np.mean([c.angle for c in cluster]))
        rad = np.radians(angle)
        ux, uy = np.cos(rad), np.sin(rad)
        pts = [(c.x1, c.y1) for c in cluster] + [(c.x2, c.y2) for c in cluster]
        ts = [px * ux + py * uy for px, py in pts]
        idx_min = int(np.argmin(ts))
        idx_max = int(np.argmax(ts))
        p0 = pts[idx_min]
        p1 = pts[idx_max]
        merged.append((int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1])))
    return merged


def _detect_lines(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    edges = cv2.Canny(mask, 60, 160)
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=40,
        minLineLength=50,
        maxLineGap=10,
    )
    if raw is None:
        return []
    lines: list[tuple[int, int, int, int]] = []
    for l in raw:
        x1, y1, x2, y2 = map(int, l[0].tolist())
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < 60:
            continue
        lines.append((x1, y1, x2, y2))
    return _merge_collinear(lines)


def _detect_lines_right_region(plot_crop_gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect long arrow-like lines in the rightmost ROI for light-theme charts."""
    h, w = plot_crop_gray.shape[:2]
    if h <= 0 or w <= 0:
        return []

    x0 = int(0.65 * w)
    if x0 >= w - 20:
        return []
    roi_gray = plot_crop_gray[:, x0:]
    if roi_gray.size == 0:
        return []

    roi_blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    roi = cv2.Canny(roi_blur, 10, 40)
    raw = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180.0,
        threshold=15,
        minLineLength=100,
        maxLineGap=60,
    )
    if raw is None:
        return []

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for l in raw:
        lx1, ly1, lx2, ly2 = map(int, l[0].tolist())
        x1 = lx1 + x0
        x2 = lx2 + x0
        y1 = ly1
        y2 = ly2
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < 140:
            continue
        angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
        if angle < 10.0 and length < 220.0:
            continue
        candidates.append((length, (x1, y1, x2, y2)))
    if not candidates:
        return []
    candidates.sort(key=lambda it: it[0], reverse=True)
    lines = [seg for _, seg in candidates[:3]]
    return lines


def _zone_invalid_global(
    bbox: tuple[int, int, int, int],
    plot_w: int,
    plot_h: int,
) -> bool:
    x0, y0, x1, y1 = bbox
    h = max(0, y1 - y0)
    touches_top = y0 <= 2
    touches_bottom = y1 >= (plot_h - 3)
    if touches_top and touches_bottom:
        return True
    if h >= int(0.92 * plot_h):
        return True
    if x1 <= x0 or y1 <= y0:
        return True
    if x0 < 0 or y0 < 0 or x1 > plot_w or y1 > plot_h:
        return True
    return False


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    a_area = float(max(0, ax1 - ax0) * max(0, ay1 - ay0))
    b_area = float(max(0, bx1 - bx0) * max(0, by1 - by0))
    denom = max(1e-6, a_area + b_area - inter)
    return inter / denom


def _dedupe_bboxes(boxes: list[tuple[int, int, int, int]], iou_thresh: float = 0.7) -> list[tuple[int, int, int, int]]:
    kept: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if any(_bbox_iou(box, prev) > iou_thresh for prev in kept):
            continue
        kept.append(box)
    return kept


def _detect_right_outline_zone(plot_gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect large right-side rectangular zone outlines on dark themes."""
    h_plot, w_plot = plot_gray.shape[:2]
    if h_plot <= 0 or w_plot <= 0:
        return []

    x_off = int(0.55 * w_plot)
    if x_off >= w_plot - 20:
        return []

    roi_gray = plot_gray[:, x_off:]
    edges = cv2.Canny(roi_gray, 10, 40)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k3, iterations=1)
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_h, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_v, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plot_area = float(w_plot * h_plot)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) != 4:
            # Fallback for faint translucent rectangle borders that break into >4 points.
            x, y, w_cc, h_cc = cv2.boundingRect(cnt)
            x += x_off
            x0, y0, x1, y1 = x, y, x + w_cc, y + h_cc
            if x0 < int(0.45 * w_plot):
                continue
            if w_cc < int(0.30 * w_plot) or h_cc < int(0.30 * h_plot):
                continue
            bbox_area = float(w_cc * h_cc)
            if bbox_area > 0.65 * plot_area:
                continue
            if w_cc >= int(0.95 * w_plot) or h_cc >= int(0.95 * h_plot):
                continue
            if _zone_invalid_global((x0, y0, x1, y1), w_plot, h_plot):
                continue
            candidates.append((bbox_area, (x0, y0, x1, y1)))
            continue
        x, y, w_cc, h_cc = cv2.boundingRect(approx)
        x += x_off
        x0, y0, x1, y1 = x, y, x + w_cc, y + h_cc

        if x0 < int(0.45 * w_plot):
            continue
        if w_cc < int(0.18 * w_plot) or h_cc < int(0.18 * h_plot):
            continue
        bbox_area = float(w_cc * h_cc)
        if bbox_area > 0.65 * plot_area:
            continue
        if w_cc >= int(0.95 * w_plot) or h_cc >= int(0.95 * h_plot):
            continue
        if _zone_invalid_global((x0, y0, x1, y1), w_plot, h_plot):
            continue

        pts = approx.reshape(-1, 2)
        aligned = True
        for i in range(4):
            p0 = pts[i]
            p1 = pts[(i + 1) % 4]
            dx = float(p1[0] - p0[0])
            dy = float(p1[1] - p0[1])
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            angle = min(angle, 180.0 - angle)
            if not (angle <= 15.0 or abs(angle - 90.0) <= 15.0):
                aligned = False
                break
        if not aligned:
            continue
        candidates.append((bbox_area, (x0, y0, x1, y1)))

    if not candidates:
        return []
    candidates.sort(key=lambda it: it[0], reverse=True)
    return [candidates[0][1]]


def run(
    image_path: str | Path,
    plot_bbox: list[float],
    axis_bbox: list[float],
    axis_ticks: list[object],
    scale: str,
    debug_dir: str | Path | None,
) -> StageResult[list[ChartElement]]:
    image = cv2.imread(str(image_path))
    if image is None:
        reason = AbstainReason(
            code=ReasonCode.IMAGE_READ_FAILED,
            stage="elements",
            message="Failed to read image for element detection.",
            details={"image_path": str(image_path)},
        )
        return StageResult(data=[], confidence=0.0, abstain=True, reasons=[reason], debug={})

    h, w = image.shape[:2]
    px0, py0, px1, py1 = _clip_bbox(plot_bbox, w, h)
    if px1 <= px0 or py1 <= py0:
        return StageResult(data=[], confidence=0.0, abstain=False, reasons=[], debug={})

    plot_crop = image[py0:py1, px0:px1].copy()
    plot_gray = cv2.cvtColor(plot_crop, cv2.COLOR_BGR2GRAY)
    mean_gray = float(plot_gray.mean())
    mask = _annotation_mask(plot_crop)
    zones = _detect_zones(mask)
    if not zones and mean_gray < 140.0:
        num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        h_plot, w_plot = mask.shape[:2]
        plot_area = float(h_plot * w_plot)
        fallback: list[tuple[int, int, int, int]] = []
        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w_cc = int(stats[i, cv2.CC_STAT_WIDTH])
            h_cc = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if w_cc < 80 or h_cc < 40:
                continue
            bbox_area = float(w_cc * h_cc)
            if bbox_area > 0.50 * plot_area:
                continue
            if w_cc > 0.92 * w_plot and h_cc > 0.92 * h_plot:
                continue
            rectangularity = float(area) / max(1.0, bbox_area)
            if rectangularity < 0.10:
                continue
            cand = (x, y, x + w_cc, y + h_cc)
            if _zone_invalid_global(cand, w_plot, h_plot):
                continue
            fallback.append(cand)
        zones = _dedupe_bboxes(fallback, iou_thresh=0.7)

    # Fallback: if mask-based zone detection finds nothing, try right-side outline detection.
    if not zones:
        right_zone = _detect_right_outline_zone(plot_gray)
        if right_zone:
            zones = _dedupe_bboxes(zones + right_zone, iou_thresh=0.7)

    # Apply global safety rule to all zone sources.
    h_plot, w_plot = plot_crop.shape[:2]
    zones = [z for z in zones if not _zone_invalid_global(z, w_plot, h_plot)]

    lines = _detect_lines(mask)
    if mean_gray >= 140.0:
        right_lines = _detect_lines_right_region(plot_gray)
        if right_lines:
            lines = _merge_collinear(lines + right_lines)

    mapper = None
    try:
        mapper = make_price_mapper(axis_ticks, scale=scale)
    except Exception:
        mapper = None
    axis_y0 = float(axis_bbox[1]) if axis_bbox else 0.0

    elements: list[ChartElement] = []
    for x0, y0, x1, y1 in zones:
        gx0, gy0, gx1, gy1 = x0 + px0, y0 + py0, x1 + px0, y1 + py0
        prices: dict = {}
        if mapper is not None:
            top = mapper(float(gy0) - axis_y0)
            bottom = mapper(float(gy1) - axis_y0)
            prices = {"low": float(min(top, bottom)), "high": float(max(top, bottom))}
        elements.append(
            ChartElement(
                kind="zone",
                geometry_px={"bbox": [float(gx0), float(gy0), float(gx1), float(gy1)]},
                prices=prices,
                confidence=0.72,
            )
        )

    for x0, y0, x1, y1 in lines:
        gx0, gy0, gx1, gy1 = x0 + px0, y0 + py0, x1 + px0, y1 + py0
        prices = {}
        if mapper is not None:
            p1 = mapper(float(gy0) - axis_y0)
            p2 = mapper(float(gy1) - axis_y0)
            prices = {"p1": float(p1), "p2": float(p2)}
        elements.append(
            ChartElement(
                kind="line",
                geometry_px={"p1": [float(gx0), float(gy0)], "p2": [float(gx1), float(gy1)]},
                prices=prices,
                confidence=0.65,
            )
        )

    if debug_dir is not None:
        dbg = plot_crop.copy()
        for x0, y0, x1, y1 in zones:
            cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 255), 2)
            if mapper is not None:
                gy0, gy1 = y0 + py0, y1 + py0
                top = mapper(float(gy0) - axis_y0)
                bottom = mapper(float(gy1) - axis_y0)
                label = f"{min(top,bottom):.2f}-{max(top,bottom):.2f}"
                cv2.putText(dbg, label, (x0, max(10, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        for x0, y0, x1, y1 in lines:
            cv2.line(dbg, (x0, y0), (x1, y1), (0, 0, 255), 2)
            if mapper is not None:
                p1 = mapper(float(y0 + py0) - axis_y0)
                p2 = mapper(float(y1 + py0) - axis_y0)
                label = f"{p1:.2f}->{p2:.2f}"
                cv2.putText(dbg, label, (x0, max(10, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path / "elements_debug.png"), dbg)

    abstain = False
    reasons: list[AbstainReason] = []
    if not elements:
        reasons.append(
            AbstainReason(
                code=ReasonCode.ELEMENTS_NOT_FOUND,
                stage="elements",
                message="No chart elements detected.",
                details={},
            )
        )

    confidence = float(np.clip(np.mean([e.confidence for e in elements]), 0.0, 1.0)) if elements else 0.0
    return StageResult(data=elements, confidence=confidence, abstain=abstain, reasons=reasons, debug={})
