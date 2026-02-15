"""Pipeline entrypoint for Component 4 (M1)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import C4Config
from .debug.artifacts import ensure_debug_dir, write_axis_debug, write_crop, write_overlay
import cv2

from .debug.draw import draw_boxes_with_labels
from .schema import OutputSchema, empty_output
from .stages import axis_ocr, calibration, candles_detect, elements_detect, frame, scenario_reconstruct, signal_build
from .types import AbstainReason, StageResult


def _collect_reasons(results: Iterable[object]) -> list[AbstainReason]:
    reasons: list[AbstainReason] = []
    for result in results:
        reasons.extend(getattr(result, "reasons", []))
    return reasons


def run_pipeline(image_path: str | Path, config: C4Config, debug_dir: str | Path | None) -> OutputSchema:
    """Run the M1 pipeline with stub stages and abstain plumbing."""
    output = empty_output()

    stage_frame = frame.run(image_path=image_path, config=config)
    axis_bbox = stage_frame.data.axis_bbox if stage_frame.data is not None else None
    stage_axis = axis_ocr.run(image_path=image_path, axis_bbox=axis_bbox, config=config)
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
        if isinstance(stage_axis.debug, dict) and stage_axis.debug.get("ocr_boxes"):
            image = cv2.imread(str(image_path))
            if image is not None:
                x0, y0, x1, y1 = stage_frame.data.axis_bbox
                x0_i = max(0, int(round(x0)))
                y0_i = max(0, int(round(y0)))
                x1_i = min(image.shape[1] - 1, int(round(x1)))
                y1_i = min(image.shape[0] - 1, int(round(y1)))
                crop = image[y0_i:y1_i, x0_i:x1_i].copy()
                boxes = []
                labels = []
                for item in stage_axis.debug.get("ocr_boxes", []):
                    bx0, by0, bx1, by1 = item["bbox"]
                    boxes.append((int(bx0), int(by0), int(bx1), int(by1)))
                    labels.append(str(item.get("value", item.get("text", ""))))
                draw_boxes_with_labels(crop, boxes, labels)
                axis_path = debug_path / config.paths.axis_debug_name
                cv2.imwrite(str(axis_path), crop)
        if axis_path is None:
            axis_path = write_axis_debug(
                image_path,
                debug_path,
                config.paths.axis_debug_name,
                stage_frame.data.axis_bbox,
                tick_rows=tick_rows,
            )
        write_crop(image_path, debug_path, "plot_crop.png", stage_frame.data.plot_bbox)
        write_crop(image_path, debug_path, "axis_crop.png", stage_frame.data.axis_bbox)
        output.debug_artifacts.overlay_path = str(overlay_path)
        output.debug_artifacts.axis_debug_path = str(axis_path)

    return output
