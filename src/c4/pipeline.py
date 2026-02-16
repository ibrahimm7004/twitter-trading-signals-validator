"""Pipeline entrypoint for Component 4 (M1)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import C4Config
from .debug.artifacts import ensure_debug_dir, write_axis_debug, write_crop, write_overlay
import cv2

from .debug.draw import draw_boxes_with_labels, draw_horizontal_marks
from .schema import OutputSchema, QualityFlags, empty_output
from .stages import axis_ocr, calibration, candles_detect, elements_detect, frame, scenario_reconstruct, signal_build
from .types import AbstainReason, StageResult


def _collect_reasons(results: Iterable[object]) -> list[AbstainReason]:
    reasons: list[AbstainReason] = []
    for result in results:
        reasons.extend(getattr(result, "reasons", []))
    return reasons


def _slice_bounds_from_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    already_exclusive: bool,
) -> tuple[int, int, int, int]:
    x0 = int(round(float(bbox[0])))
    y0 = int(round(float(bbox[1])))
    x1 = int(round(float(bbox[2])))
    y1 = int(round(float(bbox[3])))
    if already_exclusive:
        x0_i = max(0, min(x0, width))
        y0_i = max(0, min(y0, height))
        x1_i = max(0, min(x1, width))
        y1_i = max(0, min(y1, height))
    else:
        x0_i = max(0, min(x0, width - 1))
        y0_i = max(0, min(y0, height - 1))
        x1_i = min(width, max(0, min(x1, width - 1)) + 1)
        y1_i = min(height, max(0, min(y1, height - 1)) + 1)
    if x1_i <= x0_i:
        x0_i = max(0, min(x0_i, max(0, width - 1)))
        x1_i = min(width, x0_i + 1)
    if y1_i <= y0_i:
        y0_i = max(0, min(y0_i, max(0, height - 1)))
        y1_i = min(height, y0_i + 1)
    return x0_i, y0_i, x1_i, y1_i


def _expand_draw_box(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    pad: int = 1,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    x0_i = max(0, min(width - 1, x0 - pad))
    y0_i = max(0, min(height - 1, y0 - pad))
    x1_i = max(0, min(width - 1, x1 + pad))
    y1_i = max(0, min(height - 1, y1 + pad))
    if x1_i < x0_i:
        x0_i, x1_i = x1_i, x0_i
    if y1_i < y0_i:
        y0_i, y1_i = y1_i, y0_i
    return x0_i, y0_i, x1_i, y1_i


def run_pipeline(image_path: str | Path, config: C4Config, debug_dir: str | Path | None) -> OutputSchema:
    """Run the M1 pipeline with stub stages and abstain plumbing."""
    output = empty_output()

    stage_frame = frame.run(image_path=image_path, config=config)
    axis_bbox = stage_frame.data.axis_bbox if stage_frame.data is not None else None
    plot_bbox = stage_frame.data.plot_bbox if stage_frame.data is not None else None
    stage_axis = axis_ocr.run(image_path=image_path, axis_bbox=axis_bbox, plot_bbox=plot_bbox, config=config)
    stage_candles: StageResult[float | None] = StageResult(data=None, confidence=0.0, abstain=False, reasons=[], debug={})
    if stage_frame.data is not None:
        stage_candles = candles_detect.run(
            image_path=image_path,
            plot_bbox=stage_frame.data.plot_bbox,
            axis_bbox=stage_frame.data.axis_bbox,
            config=config,
        )

    stage_calib = calibration.run(stage_axis.data or [], config=config)
    if stage_calib.data is not None and stage_frame.data is not None:
        stage_frame.data.scale = stage_calib.data.scale

    stage_elements: StageResult[list] = StageResult(data=[], confidence=0.0, abstain=False, reasons=[], debug={})
    if stage_calib.data is not None and stage_frame.data is not None and stage_axis.data:
        stage_elements = elements_detect.run(
            image_path=image_path,
            plot_bbox=stage_frame.data.plot_bbox,
            axis_bbox=stage_frame.data.axis_bbox,
            axis_ticks=stage_axis.data,
            scale=stage_calib.data.scale,
            debug_dir=debug_dir,
            current_x_px=stage_candles.data,
        )

    stage_scenario: StageResult = StageResult(data=None, confidence=0.0, abstain=False, reasons=[], debug={})
    if stage_frame.data is not None:
        stage_scenario = scenario_reconstruct.run(
            plot_bbox=stage_frame.data.plot_bbox,
            elements=stage_elements.data or [],
            config=config,
            current_x_px=stage_candles.data,
            axis_ticks=stage_axis.data,
        )
    stage_signal: StageResult = signal_build.run(
        scenario=stage_scenario.data if stage_scenario.data is not None else None,
        elements=stage_elements.data or [],
        scenario_debug=stage_scenario.debug if isinstance(stage_scenario.debug, dict) else None,
    )

    if stage_frame.data is not None:
        output.chart_frame = stage_frame.data
    if stage_axis.data is not None:
        output.axis_ticks = stage_axis.data
    if stage_elements.data is not None:
        output.elements = stage_elements.data
    if stage_scenario.data is not None:
        output.scenario = stage_scenario.data
    if stage_signal.data is not None:
        output.signal = stage_signal.data
    if isinstance(stage_axis.debug, dict):
        quality_flags_payload = stage_axis.debug.get("quality_flags")
        if isinstance(quality_flags_payload, dict):
            output.quality_flags = QualityFlags.model_validate(quality_flags_payload)

    output.abstain_reasons = _collect_reasons([stage_frame, stage_axis, stage_calib, stage_elements, stage_scenario, stage_signal])
    output.abstain = any(
        stage.abstain for stage in [stage_frame, stage_axis, stage_calib, stage_elements, stage_scenario, stage_signal]
    ) or (
        len(output.abstain_reasons) > 0
    )

    if debug_dir is not None and stage_frame.data is not None:
        debug_path = ensure_debug_dir(debug_dir)
        panel_bbox = None
        axis_band_bbox = None
        if isinstance(stage_frame.debug, dict):
            panel_bbox = stage_frame.debug.get("panel_bbox")
            axis_band_bbox = stage_frame.debug.get("axis_band_bbox")
        overlay_path = write_overlay(
            image_path,
            debug_path,
            config.paths.debug_overlay_name,
            stage_frame.data.plot_bbox,
            stage_frame.data.axis_bbox,
            panel_bbox=panel_bbox,
            axis_band_bbox=axis_band_bbox,
        )
        tick_rows = None
        if isinstance(stage_frame.debug, dict):
            tick_rows = stage_frame.debug.get("tick_rows")
        axis_path = None
        axis_crop_bbox_debug = None
        if isinstance(stage_axis.debug, dict):
            maybe_axis_crop_bbox = stage_axis.debug.get("axis_crop_bbox")
            if isinstance(maybe_axis_crop_bbox, list) and len(maybe_axis_crop_bbox) == 4:
                axis_crop_bbox_debug = maybe_axis_crop_bbox
        if isinstance(stage_axis.debug, dict) and stage_axis.debug.get("ocr_boxes"):
            image = cv2.imread(str(image_path))
            if image is not None:
                if axis_crop_bbox_debug is not None:
                    x0_i, y0_i, x1_i, y1_i = _slice_bounds_from_bbox(
                        axis_crop_bbox_debug, image.shape[1], image.shape[0], already_exclusive=True
                    )
                else:
                    x0_i, y0_i, x1_i, y1_i = _slice_bounds_from_bbox(
                        stage_frame.data.axis_bbox, image.shape[1], image.shape[0], already_exclusive=False
                    )
                crop = image[y0_i:y1_i, x0_i:x1_i].copy()
                boxes = []
                labels = []
                for item in stage_axis.debug.get("ocr_boxes", []):
                    bx0, by0, bx1, by1 = item["bbox"]
                    raw_box = (
                        int(round(float(bx0))),
                        int(round(float(by0))),
                        int(round(float(bx1))),
                        int(round(float(by1))),
                    )
                    boxes.append(_expand_draw_box(raw_box, crop.shape[1], crop.shape[0], pad=1))
                    labels.append(str(item.get("value", item.get("text", ""))))
                draw_boxes_with_labels(crop, boxes, labels)
                axis_path = debug_path / config.paths.axis_debug_name
                cv2.imwrite(str(axis_path), crop)
        if axis_path is None:
            image = cv2.imread(str(image_path))
            if image is not None:
                if axis_crop_bbox_debug is not None:
                    x0_i, y0_i, x1_i, y1_i = _slice_bounds_from_bbox(
                        axis_crop_bbox_debug, image.shape[1], image.shape[0], already_exclusive=True
                    )
                else:
                    x0_i, y0_i, x1_i, y1_i = _slice_bounds_from_bbox(
                        stage_frame.data.axis_bbox, image.shape[1], image.shape[0], already_exclusive=False
                    )
                crop = image[y0_i:y1_i, x0_i:x1_i].copy()
                if tick_rows is not None:
                    draw_horizontal_marks(crop, tick_rows, color=(255, 0, 0))
                axis_path = debug_path / config.paths.axis_debug_name
                cv2.imwrite(str(axis_path), crop)
            else:
                axis_path = write_axis_debug(
                    image_path,
                    debug_path,
                    config.paths.axis_debug_name,
                    stage_frame.data.axis_bbox,
                    tick_rows=tick_rows,
                )
        write_crop(image_path, debug_path, "plot_crop.png", stage_frame.data.plot_bbox)
        if axis_crop_bbox_debug is not None:
            image = cv2.imread(str(image_path))
            if image is not None:
                x0_i, y0_i, x1_i, y1_i = _slice_bounds_from_bbox(
                    axis_crop_bbox_debug, image.shape[1], image.shape[0], already_exclusive=True
                )
                cv2.imwrite(str(debug_path / "axis_crop.png"), image[y0_i:y1_i, x0_i:x1_i].copy())
            else:
                write_crop(image_path, debug_path, "axis_crop.png", stage_frame.data.axis_bbox)
        else:
            write_crop(image_path, debug_path, "axis_crop.png", stage_frame.data.axis_bbox)
        # Keep JSON stable across runs regardless of debug directory location.
        output.debug_artifacts.overlay_path = Path(config.paths.debug_overlay_name).name if overlay_path is not None else None
        output.debug_artifacts.axis_debug_path = Path(config.paths.axis_debug_name).name if axis_path is not None else None

    return output
