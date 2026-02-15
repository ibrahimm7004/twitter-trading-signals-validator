"""Axis OCR stage (PaddleOCR optional)."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable

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


_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _parse_number(text: str) -> float | None:
    match = _NUM_RE.search(text.replace(" ", ""))
    if not match:
        return None
    raw = match.group(0).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _verbose(config: C4Config) -> bool:
    if os.getenv("C4_OCR_VERBOSE") == "1":
        return True
    runtime = getattr(config, "runtime", None)
    return bool(getattr(runtime, "ocr_verbose", False))


def _extract_paddle_detections(
    results: object,
    *,
    vlog: callable | None = None,
    meta: dict | None = None,
) -> list[dict[str, object]]:
    """Normalize PaddleOCR/PaddleX outputs into dicts with bbox/text/conf."""
    if vlog is None:
        vlog = lambda _msg: None

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
    vlog(
        f"[c4.axis_ocr] candidate_dt_poly_parents={len(candidate_dicts)} "
        f"chosen_keys={chosen_keys}"
    )
    vlog(f"[c4.axis_ocr] chosen_texts_preview={[repr(t) for t in chosen_texts[:5]]}")
    vlog(
        f"[c4.axis_ocr] chosen_text_types="
        f"{[type(x).__name__ for x in chosen_raw_text_items[:5]]}"
    )

    if chosen_dt_polys:
        first_poly = chosen_dt_polys[0]
        vlog(f"[c4.axis_ocr] dt_poly0_type={type(first_poly).__name__}")
        if isinstance(first_poly, np.ndarray):
            vlog(f"[c4.axis_ocr] dt_poly0_shape={first_poly.shape}")
        vlog(f"[c4.axis_ocr] dt_poly0_preview={repr(first_poly)[:200]}")
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
    vlog(
        f"[c4.axis_ocr] extracted_dt_polys={extracted_dt_polys} "
        f"extracted_rec_texts={extracted_rec_texts}"
    )
    vlog(f"[c4.axis_ocr] norm_len={len(normalized)} norm_text_preview={[repr(d['text']) for d in normalized[:5]]}")
    vlog(f"[c4.axis_ocr] extracted_texts_preview={[repr(x) for x in extracted_texts[:5]]}")
    vlog(f"[c4.axis_ocr] rec_text_types_preview={rec_text_types[:5]}")
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

    # Paddle/PaddleX can be unstable on some CPU builds; set safe defaults.
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("DISABLE_AUTO_LOGGING_CONFIG", "1")
    os.environ.setdefault("PADDLE_DISABLE_ONEDNN", "1")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_onednn", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    try:
        from paddleocr import PaddleOCR
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
    print("[c4.axis_ocr] paddle_import=ok")

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
    x0, y0, x1, y1 = _clip_bbox(axis_bbox, w, h)
    if x1 <= x0 or y1 <= y0:
        reason = AbstainReason(
            code=ReasonCode.AXIS_OCR_INSUFFICIENT_TICKS,
            stage="axis_ocr",
            message="Axis bbox invalid for OCR.",
            details={"axis_bbox": axis_bbox},
        )
        return StageResult(data=[], confidence=0.0, abstain=True, reasons=[reason], debug={})

    axis_crop = image[y0:y1, x0:x1].copy()
    try:
        kwargs = {"use_angle_cls": False, "lang": "en"}
        try:
            sig = inspect.signature(PaddleOCR.__init__)
            if "show_log" in sig.parameters:
                kwargs["show_log"] = False
        except (TypeError, ValueError):
            pass
        ocr = PaddleOCR(**kwargs)
    except Exception:
        ocr = PaddleOCR(use_angle_cls=False, lang="en")

    detections: list[OCRDetection] = []
    debug_boxes_raw: list[dict] = []
    method_logged = False
    last_extract_meta: dict = {"chosen_texts_preview": [], "norm_len": 0}
    def _results_empty(results: object) -> bool:
        if results is None:
            return True
        if isinstance(results, list):
            return len(results) == 0
        return False

    def _log_results(results: object) -> None:
        _vlog(f"[c4.axis_ocr] ocr_results_type={type(results)}")
        if isinstance(results, list):
            _vlog(f"[c4.axis_ocr] ocr_results_len={len(results)}")
        _vlog(f"[c4.axis_ocr] ocr_results_preview={repr(results)[:300]}")

    def _has_usable_texts(results: object) -> bool:
        extracted = _extract_paddle_detections(results, vlog=_vlog, meta=last_extract_meta)
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
            _vlog(f"[c4.axis_ocr] axis_crop_shape={img.shape} converted_bgr_to_rgb=False")
            results = ocr.ocr(img)
            _log_results(results)
            if (not _results_empty(results)) and _has_usable_texts(results):
                return results

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

    try:
        for variant, scale in _preprocess_variants(axis_crop):
            results = _run_ocr(variant) or []
            norm = _extract_paddle_detections(results, vlog=_vlog, meta=last_extract_meta)
            _vlog(f"[c4.axis_ocr] norm_len={len(norm)} norm_text_preview={[repr(d.get('text', '')) for d in norm[:5]]}")
            if len(norm) > 0:
                for item in norm:
                    bbox = item.get("bbox")
                    text = str(item.get("text", ""))
                    conf = float(item.get("conf", 0.0))
                    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                        continue
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
                    detections.append(det)
                    debug_boxes_raw.append(
                        {
                            "bbox": [x_min, y_min, x_max, y_max],
                            "text": text,
                            "value": float(value),
                            "conf": float(conf),
                        }
                    )
                continue

            # Classic PaddleOCR v2 fallback path.
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
                detections.append(det)
                debug_boxes_raw.append(
                    {
                        "bbox": [x_min, y_min, x_max, y_max],
                        "text": text,
                        "value": float(value),
                        "conf": float(conf),
                    }
                )
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
    axis_ticks = [AxisTick(value=d.value, y_px=d.y_px, conf=d.conf) for d in merged]

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

    debug["raw_detections_count"] = len(detections)
    debug["merged_ticks_count"] = len(axis_ticks)
    debug["avg_conf"] = avg_conf

    # Keep only merged tick boxes for readable axis_debug overlay.
    merged_boxes: list[dict] = []
    for tick, det in zip(axis_ticks, merged):
        bx0, by0, bx1, by1 = det.bbox
        merged_boxes.append(
            {
                "bbox": [int(bx0), int(by0), int(bx1), int(by1)],
                "text": det.text,
                "value": float(tick.value),
                "conf": float(tick.conf),
            }
        )
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
    debug["axis_crop_shape"] = [axis_crop.shape[1], axis_crop.shape[0]]
    return StageResult(data=axis_ticks, confidence=confidence, abstain=abstain, reasons=reasons, debug=debug)
