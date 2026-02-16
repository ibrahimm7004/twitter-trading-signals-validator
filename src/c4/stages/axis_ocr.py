"""Axis OCR stage (PaddleOCR optional)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import inspect
import os
from pathlib import Path
import re
import statistics
import tempfile
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from ..config import C4Config
from ..schema import AxisTick
from ..types import AbstainReason, ReasonCode, StageResult


@dataclass
class OCRDetection:
    y_px: float
    value: float
    conf: float
    bbox: tuple[int, int, int, int]
    text: str


_NUM_RE = re.compile(r"(?P<num>[-+]?(?:\d[\d,]*\.?\d*|\.\d+))(?P<suffix>[kKmMbBtT])?")
_SUFFIX_MULTIPLIERS = {
    "K": 1_000,
    "M": 1_000_000,
    "B": 1_000_000_000,
    "T": 1_000_000_000_000,
}
class _OCRCache(dict[str, object]):
    def clear(self) -> None:  # pragma: no cover - test helper behavior
        super().clear()
        _get_paddle_ocr.cache_clear()


_OCR_CACHE: _OCRCache = _OCRCache()
_PADDLE_CACHE_HIT_LOGGED = False


def _set_paddle_env_defaults() -> None:
    # Paddle/PaddleX can be unstable on some CPU builds; set safe defaults.
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("DISABLE_AUTO_LOGGING_CONFIG", "1")
    os.environ.setdefault("PADDLE_DISABLE_ONEDNN", "1")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_onednn", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")


def _build_paddle_ocr() -> object:
    from paddleocr import PaddleOCR

    kwargs = {"use_angle_cls": False, "lang": "en"}
    try:
        sig = inspect.signature(PaddleOCR.__init__)
        if "show_log" in sig.parameters:
            kwargs["show_log"] = False
    except (TypeError, ValueError):
        pass
    try:
        return PaddleOCR(**kwargs)
    except Exception:
        return PaddleOCR(use_angle_cls=False, lang="en")


@lru_cache(maxsize=1)
def _get_paddle_ocr() -> object:
    _set_paddle_env_defaults()
    return _build_paddle_ocr()


def _get_or_create_paddle_ocr(config: C4Config) -> object:
    del config  # reserved for future backend keying/config-dependent options
    ocr = _get_paddle_ocr()
    _OCR_CACHE["paddle"] = ocr
    return ocr


def _parse_number(text: str) -> float | None:
    matched = _NUM_RE.search(text.replace(" ", ""))
    if not matched:
        return None
    raw = matched.group("num").replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    suffix = matched.group("suffix")
    if suffix:
        multiplier = _SUFFIX_MULTIPLIERS.get(suffix.upper(), 1)
        value *= multiplier
    return value


def _verbose(config: C4Config) -> bool:
    if os.getenv("C4_OCR_VERBOSE") == "1":
        return True
    runtime = getattr(config, "runtime", None)
    return bool(getattr(runtime, "ocr_verbose", False))


def _extract_paddle_detections(
    results: object,
    *,
    verbose: bool = False,
    vlog: Callable[[str], None] | None = None,
    meta: dict | None = None,
) -> list[dict[str, object]]:
    """Normalize PaddleOCR/PaddleX outputs into dicts with bbox/text/conf."""
    if vlog is None:
        vlog = lambda _msg: None

    def _vlog_lazy(factory: Callable[[], str]) -> None:
        if verbose:
            vlog(factory())

    def _poly_to_bbox(poly: object) -> tuple[int, int, int, int] | None:
        if isinstance(poly, np.ndarray):
            poly = poly.tolist()

        # Unwrap one extra nesting level commonly seen in pipeline outputs.
        if isinstance(poly, (list, tuple)) and len(poly) == 1:
            inner = poly[0]
            if isinstance(inner, np.ndarray):
                inner = inner.tolist()
            if isinstance(inner, (list, tuple)):
                poly = inner

        points: list[tuple[float, float]] = []
        if isinstance(poly, (list, tuple)):
            # Flat [x1,y1,x2,y2,x3,y3,x4,y4]
            if len(poly) == 8 and all(isinstance(v, (int, float)) for v in poly):
                points = [
                    (float(poly[0]), float(poly[1])),
                    (float(poly[2]), float(poly[3])),
                    (float(poly[4]), float(poly[5])),
                    (float(poly[6]), float(poly[7])),
                ]
            else:
                # [[x,y], [x,y], ...]
                for pt in poly:
                    if isinstance(pt, np.ndarray):
                        pt = pt.tolist()
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        try:
                            points.append((float(pt[0]), float(pt[1])))
                        except (TypeError, ValueError):
                            continue

        if not points:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (
            int(round(min(xs))),
            int(round(min(ys))),
            int(round(max(xs))),
            int(round(max(ys))),
        )

    def _collect_dt_poly_candidate_dicts(obj: object, out: list[dict]) -> None:
        if isinstance(obj, dict):
            if "dt_polys" in obj:
                out.append(obj)
            for value in obj.values():
                _collect_dt_poly_candidate_dicts(value, out)
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                _collect_dt_poly_candidate_dicts(item, out)

    def _normalize_rec_text(item: object) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="ignore")
        if isinstance(item, dict):
            for key in ("text", "label", "transcription", "value"):
                if key in item:
                    val = item.get(key)
                    if isinstance(val, str):
                        return val
                    if isinstance(val, bytes):
                        return val.decode("utf-8", errors="ignore")
                    if val is not None:
                        return str(val)
            return ""
        if isinstance(item, (list, tuple)):
            for sub in item:
                if isinstance(sub, (str, bytes, dict, list, tuple)):
                    text = _normalize_rec_text(sub)
                    if text != "":
                        return text
            return ""
        return "" if item is None else str(item)

    def _as_list(value: object) -> list[object]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if value is None:
            return []
        return [value]

    normalized: list[dict[str, object]] = []
    extracted_dt_polys = 0
    extracted_rec_texts = 0
    extracted_texts: list[str] = []
    rec_text_types: list[str] = []

    # PaddleOCR 3.x / PaddleX style
    candidate_dicts: list[dict] = []
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, dict):
            _collect_dt_poly_candidate_dicts(first, candidate_dicts)
        else:
            if hasattr(first, "json"):
                maybe_json = getattr(first, "json")
                if isinstance(maybe_json, dict):
                    _collect_dt_poly_candidate_dicts(maybe_json.get("res", maybe_json), candidate_dicts)
            if hasattr(first, "res"):
                maybe_res = getattr(first, "res")
                _collect_dt_poly_candidate_dicts(maybe_res, candidate_dicts)

    chosen_dt_polys: list[object] = []
    chosen_texts: list[str] = []
    chosen_raw_text_items: list[object] = []
    chosen_scores: list[object] = []
    chosen_keys: list[str] = []
    if candidate_dicts:
        best_score: tuple[int, int] | None = None
        text_keys = ("rec_texts", "rec_text", "texts", "rec_results", "rec_res")
        score_keys = ("rec_scores", "rec_score", "scores")
        for cand in candidate_dicts:
            dt_polys = _as_list(cand.get("dt_polys"))
            if not dt_polys:
                continue
            raw_text_items: list[object] = []
            text_key_used = ""
            for key in text_keys:
                if key in cand:
                    raw_text_items = _as_list(cand.get(key))
                    text_key_used = key
                    break
            raw_scores: list[object] = []
            score_key_used = ""
            for key in score_keys:
                if key in cand:
                    raw_scores = _as_list(cand.get(key))
                    score_key_used = key
                    break
            texts = [_normalize_rec_text(x) for x in raw_text_items]
            non_empty = sum(1 for t in texts if t.strip() != "")
            closeness = -abs(len(texts) - len(dt_polys))
            score = (non_empty, closeness)
            if best_score is None or score > best_score:
                best_score = score
                chosen_dt_polys = dt_polys
                chosen_texts = texts
                chosen_raw_text_items = raw_text_items
                chosen_scores = raw_scores
                chosen_keys = ["dt_polys", text_key_used, score_key_used]

    if meta is not None:
        meta["chosen_texts_preview"] = [repr(t) for t in chosen_texts[:5]]
    _vlog_lazy(
        lambda: f"[c4.axis_ocr] candidate_dt_poly_parents={len(candidate_dicts)} "
        f"chosen_keys={chosen_keys}"
    )
    _vlog_lazy(lambda: f"[c4.axis_ocr] chosen_texts_preview={[repr(t) for t in chosen_texts[:5]]}")
    _vlog_lazy(
        lambda: f"[c4.axis_ocr] chosen_text_types="
        f"{[type(x).__name__ for x in chosen_raw_text_items[:5]]}"
    )

    if chosen_dt_polys:
        first_poly = chosen_dt_polys[0]
        _vlog_lazy(lambda: f"[c4.axis_ocr] dt_poly0_type={type(first_poly).__name__}")
        if isinstance(first_poly, np.ndarray):
            _vlog_lazy(lambda: f"[c4.axis_ocr] dt_poly0_shape={first_poly.shape}")
        _vlog_lazy(lambda: f"[c4.axis_ocr] dt_poly0_preview={repr(first_poly)[:200]}")
        extracted_dt_polys = len(chosen_dt_polys)
        extracted_rec_texts = len(chosen_texts)
        pair_count = min(len(chosen_dt_polys), len(chosen_texts))
        for i in range(pair_count):
            poly = chosen_dt_polys[i]
            bbox = _poly_to_bbox(poly)
            if bbox is None:
                continue
            text = chosen_texts[i]
            raw_conf = chosen_scores[i] if i < len(chosen_scores) else 1.0
            try:
                conf = float(raw_conf)
            except (TypeError, ValueError):
                conf = 1.0
            normalized.append(
                {
                    "bbox": (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    "text": text,
                    "conf": float(conf),
                }
            )
            extracted_texts.append(text)
            rec_text_types.append(
                type(chosen_raw_text_items[i]).__name__ if i < len(chosen_raw_text_items) else "NoneType"
            )

    if meta is not None:
        meta["norm_len"] = len(normalized)
    _vlog_lazy(
        lambda: f"[c4.axis_ocr] extracted_dt_polys={extracted_dt_polys} "
        f"extracted_rec_texts={extracted_rec_texts}"
    )
    _vlog_lazy(
        lambda: f"[c4.axis_ocr] norm_len={len(normalized)} "
        f"norm_text_preview={[repr(d['text']) for d in normalized[:5]]}"
    )
    _vlog_lazy(lambda: f"[c4.axis_ocr] extracted_texts_preview={[repr(x) for x in extracted_texts[:5]]}")
    _vlog_lazy(lambda: f"[c4.axis_ocr] rec_text_types_preview={rec_text_types[:5]}")
    return normalized


def _preprocess_variants(axis_bgr: np.ndarray) -> list[tuple[np.ndarray, float]]:
    variants: list[tuple[np.ndarray, float]] = []
    variants.append((axis_bgr, 1.0))
    for scale in (2.0, 3.0):
        resized = cv2.resize(axis_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.append((resized, scale))

    gray = cv2.cvtColor(axis_bgr, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    sharpen = cv2.filter2D(gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32))
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10)
    for img in (gray, inv, sharpen, thr):
        variants.append((cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 1.0))
    return variants


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


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _expand_axis_roi_x(x0: int, x1: int, frame_w: int) -> tuple[int, int, int, int]:
    roi_w = max(1, int(x1 - x0))
    pad_left = _clamp_int(int(roi_w * 0.35), 40, 140)
    pad_right = _clamp_int(int(roi_w * 0.05), 10, 40)
    return max(0, x0 - pad_left), min(frame_w, x1 + pad_right), pad_left, pad_right


def _merge_by_y(detections: list[OCRDetection], tol: int) -> list[OCRDetection]:
    if not detections:
        return []
    detections.sort(key=lambda d: d.y_px)
    clusters: list[list[OCRDetection]] = []
    for det in detections:
        if not clusters or abs(det.y_px - clusters[-1][-1].y_px) > tol:
            clusters.append([det])
        else:
            clusters[-1].append(det)
    merged: list[OCRDetection] = []
    for cluster in clusters:
        best = max(cluster, key=lambda d: d.conf)
        merged.append(best)
    return merged


_CLEAN_ATTEMPTS: list[tuple[float, int, float | None] | None] = [
    (0.03, 6, 0.65),
    (0.08, 6, 0.9),
    (0.10, 6, None),
    None,
]
_RELAXED_CLEAN_ATTEMPTS = _CLEAN_ATTEMPTS[1:]


def _filter_candidates_by_bbox(
    parsed_ticks: Iterable[dict[str, Any]],
    crop_w: int,
    *,
    right_frac: float = 0.03,
    min_px: int = 6,
    width_frac: float | None = 0.65,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    if crop_w <= 0:
        return filtered
    right_limit = max(min_px, int(right_frac * crop_w))
    width_limit = int(width_frac * crop_w) if width_frac is not None else None
    for tick in parsed_ticks:
        bbox = tick.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        x0, _, x1, _ = bbox
        width = x1 - x0
        if crop_w - x1 > right_limit:
            continue
        if width_limit is not None and width > width_limit:
            continue
        filtered.append(dict(tick))
    return filtered


def _cluster_ticks_by_y(ticks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not ticks:
        return []
    sorted_ticks = sorted(ticks, key=lambda tick: float(tick.get("y_px", 0.0)))
    gaps: list[float] = []
    for index in range(1, len(sorted_ticks)):
        gap = sorted_ticks[index]["y_px"] - sorted_ticks[index - 1]["y_px"]
        if gap > 0:
            gaps.append(gap)
    med_gap = statistics.median(gaps) if gaps else 0.0
    threshold = max(15.0, 2.5 * med_gap) if med_gap > 0 else 15.0
    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for tick in sorted_ticks:
        if not current:
            current = [tick]
            clusters.append(current)
            continue
        prev_tick = current[-1]
        gap = tick["y_px"] - prev_tick["y_px"]
        if gap > threshold:
            current = [tick]
            clusters.append(current)
        else:
            current.append(tick)
    best = max(clusters, key=lambda cluster: (len(cluster), cluster[-1]["y_px"] - cluster[0]["y_px"]))
    return best


def _determine_direction(values: list[float]) -> str:
    if len(values) < 2:
        return "inc"
    pos = 0
    neg = 0
    for i in range(len(values) - 1):
        diff = values[i + 1] - values[i]
        if diff > 0:
            pos += 1
        if diff < 0:
            neg += 1
    return "inc" if pos >= neg else "dec"


def _monotonic_check_pair(
    prev: float | None,
    curr: float | None,
    direction: str,
    epsilon: float,
) -> bool:
    if direction == "inc":
        if prev is None or curr is None:
            return True
        return curr + epsilon >= prev
    if prev is None or curr is None:
        return True
    return curr <= prev + epsilon


def _longest_monotonic_subsequence(values: list[float], direction: str, epsilon: float) -> list[int]:
    n = len(values)
    if n == 0:
        return []
    dp = [1] * n
    prev_index = [-1] * n
    best_idx = 0
    for i in range(n):
        for j in range(i):
            if _monotonic_check_pair(values[j], values[i], direction, epsilon) and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                prev_index[i] = j
        if dp[i] > dp[best_idx]:
            best_idx = i
    sequence: list[int] = []
    while best_idx != -1:
        sequence.append(best_idx)
        best_idx = prev_index[best_idx]
    sequence.reverse()
    return sequence


def _compute_epsilon(values: list[float]) -> float:
    if not values:
        return 1e-9
    value_range = max(values) - min(values)
    return max(1e-9, 0.001 * value_range) if value_range > 0 else 1e-9


def _rescue_decimal(
    value: float,
    median_abs: float,
    direction: str,
    prev_value: float | None,
    next_value: float | None,
    epsilon: float,
) -> float | None:
    if median_abs <= 0:
        return None
    abs_value = abs(value)
    if abs_value <= 1:
        return None
    nearest = int(round(value))
    if abs(value - nearest) >= 1e-6 or nearest % 10 != 0:
        return None
    best_candidate = None
    best_diff = float("inf")
    for k in range(1, 7):
        candidate = value / (10 ** k)
        abs_candidate = abs(candidate)
        if abs_candidate == 0:
            continue
        if not (median_abs / 5 <= abs_candidate <= median_abs * 5):
            continue
        if not _monotonic_check_pair(prev_value, candidate, direction, epsilon):
            continue
        if not _monotonic_check_pair(candidate, next_value, direction, epsilon):
            continue
        diff = abs(abs_candidate - median_abs)
        if diff < best_diff:
            best_diff = diff
            best_candidate = candidate
    return best_candidate

def _clean_numeric_cluster(
    cluster: list[dict[str, Any]],
    median_abs: float,
    direction: str,
    epsilon: float,
) -> tuple[list[dict[str, Any]], list[float]]:
    cleaned_ticks: list[dict[str, Any]] = []
    cleaned_values: list[float] = []
    for index, tick in enumerate(cluster):
        tick_value = float(tick["value"])
        abs_value = abs(tick_value)
        if median_abs > 10 and abs_value < median_abs * 0.05:
            continue
        next_value = (
            float(cluster[index + 1]["value"])
            if index + 1 < len(cluster)
            else None
        )
        candidate = tick_value
        if median_abs > 0 and abs_value > median_abs * 20 and abs_value > 1:
            rescue = _rescue_decimal(tick_value, median_abs, direction, None, next_value, epsilon)
            if rescue is None:
                continue
            candidate = rescue
        cleaned_tick = dict(tick)
        cleaned_tick["value"] = float(candidate)
        cleaned_ticks.append(cleaned_tick)
        cleaned_values.append(float(candidate))
    return cleaned_ticks, cleaned_values


def _run_cleaning_attempt(
    candidates: list[dict[str, Any]],
    crop_w: int,
    min_ticks: int,
) -> list[dict[str, Any]] | None:
    if not candidates:
        return None
    clustered = _cluster_ticks_by_y(candidates)
    if not clustered:
        return None
    raw_values = [float(tick["value"]) for tick in clustered]
    direction = _determine_direction(raw_values)
    epsilon = _compute_epsilon(raw_values)
    values_for_median = [abs(val) for val in raw_values if abs(val) > 0]
    median_abs = float(statistics.median(values_for_median)) if values_for_median else 0.0
    cleaned_ticks, cleaned_values = _clean_numeric_cluster(clustered, median_abs, direction, epsilon)
    if not cleaned_ticks:
        return None
    sequence = _longest_monotonic_subsequence(cleaned_values, direction, epsilon)
    if len(sequence) >= min_ticks:
        return [cleaned_ticks[idx] for idx in sequence]
    return cleaned_ticks


def _apply_attempts(
    candidates0: list[dict[str, Any]],
    crop_w: int,
    min_ticks: int,
    attempts: list[tuple[float, int, float | None] | None],
) -> list[dict[str, Any]]:
    best_fallback: list[dict[str, Any]] = []
    best_len = 0
    for params in attempts:
        if params is None:
            filtered = [dict(tick) for tick in candidates0]
        else:
            right_frac, min_px, width_frac = params
            filtered = _filter_candidates_by_bbox(
                candidates0,
                crop_w,
                right_frac=right_frac,
                min_px=min_px,
                width_frac=width_frac,
            )
        result = _run_cleaning_attempt(filtered, crop_w, min_ticks) if filtered else None
        if result is None:
            continue
        if len(result) >= min_ticks:
            return result
        if len(result) > best_len:
            best_len = len(result)
            best_fallback = result
    if best_fallback:
        return best_fallback
    return [dict(tick) for tick in candidates0]


def _choose_final_ticks(
    primary: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
    min_ticks: int,
) -> list[dict[str, Any]]:
    if len(primary) >= min_ticks:
        return primary
    if len(fallback) >= min_ticks:
        return fallback
    return primary


def clean_axis_ticks(
    parsed_ticks: Iterable[dict[str, Any]],
    crop_w: int,
    min_ticks: int = 4,
    *,
    attempt_sequence: list[tuple[float, int, float | None] | None] | None = None,
) -> list[dict[str, Any]]:
    candidates0 = [dict(tick) for tick in parsed_ticks]
    if not candidates0:
        return []
    attempts = attempt_sequence if attempt_sequence is not None else _CLEAN_ATTEMPTS
    return _apply_attempts(candidates0, crop_w, min_ticks, attempts)


def _compute_vertical_roi_bounds(
    frame_h: int,
    axis_y0: int,
    axis_y1: int,
    plot_bbox: list[float] | None,
) -> tuple[int, int]:
    plot_y0 = axis_y0
    plot_y1 = axis_y1
    if plot_bbox and len(plot_bbox) == 4:
        _, py0, _, py1 = plot_bbox
        plot_y0 = max(plot_y0, int(round(py0)))
        plot_y1 = min(plot_y1, int(round(py1)))
    plot_height = max(plot_y1 - plot_y0, 0)
    fallback_threshold = max(350, int(round(0.35 * frame_h)))
    if plot_height < fallback_threshold:
        y0 = max(0, int(round(0.08 * frame_h)))
        y1 = min(frame_h, int(round(0.92 * frame_h)))
    else:
        y0 = max(axis_y0, plot_y0)
        y1 = min(axis_y1, plot_y1)
    if y1 <= y0:
        y0 = axis_y0
        y1 = axis_y1
    return y0, y1


def _preprocess_for_retry(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(scaled)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def _choose_best_roi(
    crop_a: np.ndarray | None,
    detections_a: list[OCRDetection],
    debug_a: list[dict[str, object]],
    crop_b: np.ndarray | None,
    detections_b: list[OCRDetection],
    debug_b: list[dict[str, object]],
) -> tuple[np.ndarray, list[OCRDetection], list[dict[str, object]], str]:
    if crop_b is None:
        assert crop_a is not None
        return crop_a, detections_a, debug_a, "axis"
    if crop_a is None:
        return crop_b, detections_b, debug_b, "plot"
    if len(detections_b) > len(detections_a):
        return crop_b, detections_b, debug_b, "plot"
    return crop_a, detections_a, debug_a, "axis"


_QUALITY_REASON_INSUFFICIENT_TICKS = "INSUFFICIENT_TICKS"
_QUALITY_REASON_LOW_NUMERIC_DENSITY = "LOW_NUMERIC_CANDIDATE_DENSITY"
_QUALITY_REASON_AXIS_CROP_STRIP_LIKE = "AXIS_CROP_STRIP_LIKE"
_LOW_NUMERIC_CANDIDATE_MAX = 4


def _bbox_w_h(bbox: tuple[int, int, int, int] | None) -> tuple[int | None, int | None]:
    if bbox is None:
        return None, None
    x0, y0, x1, y1 = bbox
    return max(0, int(x1 - x0)), max(0, int(y1 - y0))


def evaluate_input_quality_flags(
    *,
    frame_w: int,
    frame_h: int,
    min_ticks: int,
    axis_bbox: tuple[int, int, int, int] | None,
    plot_bbox: tuple[int, int, int, int] | None,
    axis_crop_w: int | None,
    axis_crop_h: int | None,
    raw_detections_count: int,
    parsed_numeric_candidates_count: int,
    merged_ticks_count: int,
) -> dict[str, Any]:
    insufficient_ticks = merged_ticks_count < min_ticks
    low_numeric_candidate_density = parsed_numeric_candidates_count <= _LOW_NUMERIC_CANDIDATE_MAX
    ocr_ran = raw_detections_count > 0
    crop_height_sane = axis_crop_h is not None and axis_crop_h >= 60
    crop_width_strip_like = axis_crop_w is not None and 120 <= axis_crop_w <= 320

    likely_faulty_input = bool(
        insufficient_ticks
        and low_numeric_candidate_density
        and ocr_ran
        and crop_height_sane
        and crop_width_strip_like
    )

    reasons: list[str] = []
    if insufficient_ticks:
        reasons.append(_QUALITY_REASON_INSUFFICIENT_TICKS)
    if likely_faulty_input:
        reasons.append(_QUALITY_REASON_LOW_NUMERIC_DENSITY)
        reasons.append(_QUALITY_REASON_AXIS_CROP_STRIP_LIKE)

    metrics: dict[str, float | int | bool | None] = {
        "frame_w": int(frame_w),
        "frame_h": int(frame_h),
        "min_ticks": int(min_ticks),
        "axis_crop_w": int(axis_crop_w) if axis_crop_w is not None else None,
        "axis_crop_h": int(axis_crop_h) if axis_crop_h is not None else None,
        "raw_detections_count": int(raw_detections_count),
        "parsed_numeric_candidates_count": int(parsed_numeric_candidates_count),
        "merged_ticks_count": int(merged_ticks_count),
        "condition_insufficient_ticks": insufficient_ticks,
        "condition_low_numeric_candidate_density": low_numeric_candidate_density,
        "condition_ocr_ran": ocr_ran,
        "condition_crop_height_sane": crop_height_sane,
        "condition_crop_width_strip_like": crop_width_strip_like,
    }
    return {
        "likely_faulty_input": likely_faulty_input,
        "reasons": reasons,
        "metrics": metrics,
    }


def _ocr_disabled_reason(message: str) -> AbstainReason:
    return AbstainReason(
        code=ReasonCode.OCR_NOT_IMPLEMENTED,
        stage="axis_ocr",
        message=message,
        details={"reason": "OCR_DISABLED"},
    )


def run(
    image_path: str | Path,
    axis_bbox: list[float] | None,
    config: C4Config,
    plot_bbox: list[float] | None = None,
    ocr_instance: object | None = None,
) -> StageResult[list[AxisTick]]:
    verbose = _verbose(config)
    _vlog = print if verbose else (lambda _msg: None)

    debug: dict = {
        "ocr_backend": config.runtime.ocr_backend,
        "ocr_import_ok": False,
        "ocr_method": None,
        "raw_detections_count": 0,
        "merged_ticks_count": 0,
        "avg_conf": 0.0,
    }
    if config.runtime.ocr_backend != "paddle":
        print(f"[c4.axis_ocr] backend={config.runtime.ocr_backend}")
        print("[c4.axis_ocr] paddle_import=skipped")
        print("[c4.axis_ocr] ocr_method=none")
        print(
            "[c4.axis_ocr] summary raw_detections="
            f"{debug['raw_detections_count']} merged_ticks={debug['merged_ticks_count']} "
            f"avg_conf={debug['avg_conf']:.4f}"
        )
        reason = _ocr_disabled_reason("OCR backend disabled.")
        return StageResult(data=[], confidence=0.0, abstain=True, reasons=[reason], debug=debug)

    print("[c4.axis_ocr] backend=paddle")
    global _PADDLE_CACHE_HIT_LOGGED
    if ocr_instance is not None:
        ocr = ocr_instance
        debug["ocr_import_ok"] = True
        debug["paddle_ocr_cached"] = True
        print("[c4.axis_ocr] paddle_import=ok")
    else:
        cached_before = _get_paddle_ocr.cache_info().currsize > 0
        try:
            ocr = _get_or_create_paddle_ocr(config)
        except Exception:
            print("[c4.axis_ocr] paddle_import=failed")
            print("[c4.axis_ocr] ocr_method=none")
            print(
                "[c4.axis_ocr] summary raw_detections="
                f"{debug['raw_detections_count']} merged_ticks={debug['merged_ticks_count']} "
                f"avg_conf={debug['avg_conf']:.4f}"
            )
            reason = _ocr_disabled_reason("PaddleOCR not available.")
            return StageResult(data=[], confidence=0.0, abstain=True, reasons=[reason], debug=debug)
        debug["ocr_import_ok"] = True
        debug["paddle_ocr_cached"] = cached_before
        print("[c4.axis_ocr] paddle_import=ok")
        if cached_before and not _PADDLE_CACHE_HIT_LOGGED:
            print("[c4.axis_ocr] paddle_ocr_cached=True")
            _PADDLE_CACHE_HIT_LOGGED = True

    image = cv2.imread(str(image_path))
    if image is None:
        reason = AbstainReason(
            code=ReasonCode.IMAGE_READ_FAILED,
            stage="axis_ocr",
            message="Failed to read image for OCR.",
            details={"image_path": str(image_path)},
        )
        return StageResult(data=[], confidence=0.0, abstain=True, reasons=[reason], debug={})

    if not axis_bbox:
        reason = AbstainReason(
            code=ReasonCode.AXIS_OCR_INSUFFICIENT_TICKS,
            stage="axis_ocr",
            message="Axis bbox missing for OCR.",
            details={},
        )
        return StageResult(data=[], confidence=0.0, abstain=True, reasons=[reason], debug={})

    h, w = image.shape[:2]
    axis_x0_i, axis_y0_i, axis_x1_i, axis_y1_i = _clip_bbox(axis_bbox, w, h)
    # Use [x0, y0, x1, y1) (x1/y1 exclusive) for slicing coordinates.
    axis_x0 = axis_x0_i
    axis_y0 = axis_y0_i
    axis_x1 = min(w, axis_x1_i + 1)
    axis_y1 = min(h, axis_y1_i + 1)
    axis_bbox_clipped = (axis_x0, axis_y0, axis_x1, axis_y1)
    plot_bbox_clipped: tuple[int, int, int, int] | None = None
    if axis_x1 <= axis_x0 or axis_y1 <= axis_y0:
        reason = AbstainReason(
            code=ReasonCode.AXIS_OCR_INSUFFICIENT_TICKS,
            stage="axis_ocr",
            message="Axis bbox invalid for OCR.",
            details={"axis_bbox": axis_bbox},
        )
        return StageResult(data=[], confidence=0.0, abstain=True, reasons=[reason], debug={})
    plot_y0 = axis_y0
    plot_y1 = axis_y1
    desired_width: int | None = None
    if plot_bbox and len(plot_bbox) == 4:
        px0_i, py0_i, px1_i, py1_i = _clip_bbox(plot_bbox, w, h)
        px0 = px0_i
        py0 = py0_i
        px1 = min(w, px1_i + 1)
        py1 = min(h, py1_i + 1)
        plot_bbox_clipped = (px0, py0, px1, py1)
        plot_y0 = max(plot_y0, py0)
        plot_y1 = min(plot_y1, py1)
        plot_width = max(px1 - px0, 1)
        desired_width = min(max(int(round(plot_width * 0.18)), 120), 260)
    crop_y0 = max(axis_y0, plot_y0)
    crop_y1 = min(axis_y1, plot_y1)
    if crop_y1 <= crop_y0:
        crop_y0, crop_y1 = axis_y0, axis_y1
    crop_x1 = axis_x1
    if desired_width is not None:
        crop_x0 = min(axis_x0, max(0, axis_x1 - desired_width))
    else:
        crop_x0 = axis_x0
    if crop_x1 <= crop_x0:
        crop_x0, crop_x1 = axis_x0, axis_x1
    crop_x0, crop_x1, roi_pad_left, roi_pad_right = _expand_axis_roi_x(crop_x0, crop_x1, w)
    axis_crop = image[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    axis_crop_h, axis_crop_w = axis_crop.shape[:2]

    detections: list[OCRDetection] = []
    debug_boxes_raw: list[dict] = []
    raw_ocr_candidates_count = 0
    method_logged = False
    last_extract_meta: dict = {"chosen_texts_preview": [], "norm_len": 0}
    def _results_empty(results: object) -> bool:
        if results is None:
            return True
        if isinstance(results, list):
            return len(results) == 0
        return False

    def _log_results(results: object) -> None:
        if not verbose:
            return
        _vlog(f"[c4.axis_ocr] ocr_results_type={type(results)}")
        if isinstance(results, list):
            _vlog(f"[c4.axis_ocr] ocr_results_len={len(results)}")
        _vlog(f"[c4.axis_ocr] ocr_results_preview={repr(results)[:300]}")

    def _has_usable_texts(results: object) -> bool:
        extracted = _extract_paddle_detections(results, verbose=verbose, vlog=_vlog, meta=last_extract_meta)
        if not extracted:
            return False
        texts = [str(item.get("text", "")) for item in extracted]
        return not all(t.strip() == "" for t in texts)

    def _run_ocr(img: np.ndarray):
        nonlocal method_logged
        try:
            if not method_logged:
                print("[c4.axis_ocr] ocr_method=ocr()")
                debug["ocr_method"] = "ocr()"
                method_logged = True
            if verbose:
                _vlog(f"[c4.axis_ocr] axis_crop_shape={img.shape} converted_bgr_to_rgb=False")
            results = ocr.ocr(img)
            _log_results(results)
            if (not _results_empty(results)) and _has_usable_texts(results):
                return results

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if verbose:
                _vlog(f"[c4.axis_ocr] axis_crop_shape={rgb.shape} converted_bgr_to_rgb=True")
            results = ocr.ocr(rgb)
            _log_results(results)
            if (not _results_empty(results)) and _has_usable_texts(results):
                return results

            tmp_dir = Path.cwd() / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".png",
                prefix="axis_ocr_",
                dir=str(tmp_dir),
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
            try:
                cv2.imwrite(str(tmp_path), img)
                if verbose:
                    _vlog(f"[c4.axis_ocr] axis_crop_shape={img.shape} converted_bgr_to_rgb=False")
                    _vlog(f"[c4.axis_ocr] ocr_temp_path={tmp_path}")
                results = ocr.ocr(str(tmp_path))
                _log_results(results)
                return results
            finally:
                tmp_path.unlink(missing_ok=True)
        except Exception:
            if hasattr(ocr, "predict"):
                if not method_logged:
                    print("[c4.axis_ocr] ocr_method=predict")
                    debug["ocr_method"] = "predict"
                    method_logged = True
                return ocr.predict(img)
            raise

    def _append_norm_results(
        results: list[dict[str, object]],
        scale: float,
        detections_acc: list[OCRDetection],
        debug_acc: list[dict[str, object]],
    ) -> None:
        nonlocal raw_ocr_candidates_count
        for item in results:
            bbox = item.get("bbox")
            text = str(item.get("text", ""))
            conf = float(item.get("conf", 0.0))
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            raw_ocr_candidates_count += 1
            value = _parse_number(text)
            if value is None:
                continue
            x_min = int(round(float(bbox[0]) / scale))
            y_min = int(round(float(bbox[1]) / scale))
            x_max = int(round(float(bbox[2]) / scale))
            y_max = int(round(float(bbox[3]) / scale))
            y_center = (y_min + y_max) / 2.0
            det = OCRDetection(
                y_px=float(y_center),
                value=float(value),
                conf=float(conf),
                bbox=(x_min, y_min, x_max, y_max),
                text=text,
            )
            detections_acc.append(det)
            debug_acc.append(
                {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "text": text,
                    "value": float(value),
                    "conf": float(conf),
                }
            )

    def _append_legacy_results(
        results: object,
        scale: float,
        detections_acc: list[OCRDetection],
        debug_acc: list[dict[str, object]],
    ) -> None:
        nonlocal raw_ocr_candidates_count
        if not isinstance(results, list):
            return
        for line in results:
            if not isinstance(line, list):
                continue
            for entry in line:
                if not (isinstance(entry, (list, tuple)) and len(entry) >= 2):
                    continue
                box = entry[0]
                txt_conf = entry[1]
                if not (isinstance(txt_conf, (list, tuple)) and len(txt_conf) >= 2):
                    continue
                if not isinstance(box, (list, tuple)):
                    continue
                xs = [p[0] for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
                ys = [p[1] for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
                if not xs or not ys:
                    continue
                text = str(txt_conf[0])
                try:
                    conf = float(txt_conf[1])
                except (TypeError, ValueError):
                    conf = 0.0
                raw_ocr_candidates_count += 1
                value = _parse_number(text)
                if value is None:
                    continue
                x_min = int(round(float(min(xs)) / scale))
                y_min = int(round(float(min(ys)) / scale))
                x_max = int(round(float(max(xs)) / scale))
                y_max = int(round(float(max(ys)) / scale))
                y_center = (y_min + y_max) / 2.0
                det = OCRDetection(
                    y_px=float(y_center),
                    value=float(value),
                    conf=float(conf),
                    bbox=(x_min, y_min, x_max, y_max),
                    text=text,
                )
                detections_acc.append(det)
                debug_acc.append(
                    {
                        "bbox": [x_min, y_min, x_max, y_max],
                        "text": text,
                        "value": float(value),
                        "conf": float(conf),
                    }
                )

    def _collect_detections_from_crop(crop: np.ndarray) -> tuple[list[OCRDetection], list[dict[str, object]]]:
        collected: list[OCRDetection] = []
        raw_boxes: list[dict[str, object]] = []
        for variant, scale in _preprocess_variants(crop):
            if verbose:
                _vlog(f"[c4.axis_ocr] axis_crop_shape={variant.shape} converted_bgr_to_rgb=False")
            results = _run_ocr(variant) or []
            norm = _extract_paddle_detections(results, verbose=verbose, vlog=_vlog, meta=last_extract_meta)
            if verbose:
                _vlog(
                    f"[c4.axis_ocr] norm_len={len(norm)} norm_text_preview={[repr(d.get('text', '')) for d in norm[:5]]}"
                )
            if len(norm) > 0:
                _append_norm_results(norm, scale, collected, raw_boxes)
                continue
            _append_legacy_results(results, scale, collected, raw_boxes)
        return collected, raw_boxes

    try:
        detections, debug_boxes_raw = _collect_detections_from_crop(axis_crop)
    except Exception as exc:
        debug["ocr_error"] = str(exc)
        debug["ocr_boxes"] = []
        if verbose:
            debug["ocr_boxes_raw"] = debug_boxes_raw
        reason = AbstainReason(
            code=ReasonCode.OCR_NOT_IMPLEMENTED,
            stage="axis_ocr",
            message="PaddleOCR failed to run.",
            details={"error": str(exc)},
        )
        insufficient = AbstainReason(
            code=ReasonCode.AXIS_OCR_INSUFFICIENT_TICKS,
            stage="axis_ocr",
            message="OCR failed before producing ticks.",
            details={"error": str(exc)},
        )
        print("[c4.axis_ocr] failure reason=OCR_EXCEPTION")
        print(f"[c4.axis_ocr] axis_crop_shape={axis_crop.shape}")
        print(f"[c4.axis_ocr] chosen_texts_preview={last_extract_meta.get('chosen_texts_preview', [])}")
        print(f"[c4.axis_ocr] norm_len={last_extract_meta.get('norm_len', 0)}")
        print(
            "[c4.axis_ocr] summary raw_detections="
            f"{debug['raw_detections_count']} merged_ticks={debug['merged_ticks_count']} "
            f"avg_conf={debug['avg_conf']:.4f}"
        )
        return StageResult(
            data=[],
            confidence=0.0,
            abstain=True,
            reasons=[reason, insufficient],
            debug=debug,
        )

    merged = _merge_by_y(detections, config.runtime.axis_row_merge_tol_px)
    parsed_ticks = [
        {
            "y_px": d.y_px,
            "value": d.value,
            "conf": d.conf,
            "bbox": d.bbox,
            "text": d.text,
        }
        for d in merged
    ]
    cleaned_ticks = clean_axis_ticks(
        parsed_ticks,
        axis_crop.shape[1],
        min_ticks=config.thresholds.min_ticks,
    )
    axis_ticks = [AxisTick(value=t["value"], y_px=t["y_px"], conf=t["conf"]) for t in cleaned_ticks]

    abstain = False
    reasons: list[AbstainReason] = []
    if len(axis_ticks) < config.thresholds.min_ticks:
        abstain = True
        reasons.append(
            AbstainReason(
                code=ReasonCode.AXIS_OCR_INSUFFICIENT_TICKS,
                stage="axis_ocr",
                message="Insufficient OCR ticks.",
                details={"min_ticks": config.thresholds.min_ticks, "observed": len(axis_ticks)},
            )
        )

    avg_conf = float(np.mean([t.conf for t in axis_ticks])) if axis_ticks else 0.0
    confidence = min(1.0, len(axis_ticks) / max(1, config.thresholds.min_ticks)) * avg_conf

    debug["raw_detections_count"] = raw_ocr_candidates_count
    debug["parsed_numeric_candidates_count"] = len(detections)
    debug["merged_ticks_count"] = len(axis_ticks)
    debug["avg_conf"] = avg_conf
    debug["axis_crop_shape"] = [axis_crop_w, axis_crop_h]
    debug["axis_crop_bbox"] = [crop_x0, crop_y0, crop_x1, crop_y1]
    debug["axis_crop_pad_left"] = roi_pad_left
    debug["axis_crop_pad_right"] = roi_pad_right
    debug["quality_flags"] = evaluate_input_quality_flags(
        frame_w=w,
        frame_h=h,
        min_ticks=config.thresholds.min_ticks,
        axis_bbox=axis_bbox_clipped,
        plot_bbox=plot_bbox_clipped,
        axis_crop_w=axis_crop_w,
        axis_crop_h=axis_crop_h,
        raw_detections_count=raw_ocr_candidates_count,
        parsed_numeric_candidates_count=len(detections),
        merged_ticks_count=len(axis_ticks),
    )

    # Keep only merged tick boxes for readable axis_debug overlay.
    merged_boxes: list[dict] = []
    for tick in cleaned_ticks:
        bbox = tick.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bx0, by0, bx1, by1 = bbox
        else:
            bx0 = by0 = bx1 = by1 = 0
        merged_boxes.append(
            {
                "bbox": [int(bx0), int(by0), int(bx1), int(by1)],
                "text": str(tick.get("text", "")),
                "value": float(tick["value"]),
                "conf": float(tick["conf"]),
            }
        )
    # Coordinate note:
    # - axis_crop_bbox is full-frame [x0,y0,x1,y1) (x1/y1 exclusive) used for slicing.
    # - ocr_boxes are axis-crop-local [x0,y0,x1,y1] used only for drawing on axis_crop/axis_debug.
    debug["ocr_boxes"] = merged_boxes
    if verbose:
        debug["ocr_boxes_raw"] = debug_boxes_raw

    if len(axis_ticks) < config.thresholds.min_ticks or len(detections) == 0:
        print(
            "[c4.axis_ocr] failure reason="
            + ("NO_RAW_DETECTIONS" if len(detections) == 0 else "INSUFFICIENT_TICKS")
        )
        print(f"[c4.axis_ocr] axis_crop_shape={axis_crop.shape}")
        print(f"[c4.axis_ocr] chosen_texts_preview={last_extract_meta.get('chosen_texts_preview', [])}")
        print(f"[c4.axis_ocr] norm_len={last_extract_meta.get('norm_len', 0)}")
    print(
        "[c4.axis_ocr] summary raw_detections="
        f"{debug['raw_detections_count']} merged_ticks={debug['merged_ticks_count']} "
        f"avg_conf={avg_conf:.4f}"
    )
    return StageResult(data=axis_ticks, confidence=confidence, abstain=abstain, reasons=reasons, debug=debug)
