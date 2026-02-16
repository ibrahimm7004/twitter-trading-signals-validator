from __future__ import annotations

from c4.stages.axis_ocr import (
    _QUALITY_REASON_INSUFFICIENT_TICKS,
    _QUALITY_REASON_LOW_NUMERIC_DENSITY,
    evaluate_input_quality_flags,
)


def test_likely_faulty_input_true_for_bad_geometry_pattern():
    flags = evaluate_input_quality_flags(
        frame_w=1200,
        frame_h=900,
        min_ticks=4,
        axis_bbox=(1080, 100, 1120, 260),
        plot_bbox=(300, 120, 950, 280),
        axis_crop_w=180,
        axis_crop_h=160,
        raw_detections_count=6,
        parsed_numeric_candidates_count=3,
        merged_ticks_count=2,
    )

    assert flags["likely_faulty_input"] is True
    assert _QUALITY_REASON_INSUFFICIENT_TICKS in flags["reasons"]
    assert _QUALITY_REASON_LOW_NUMERIC_DENSITY in flags["reasons"]


def test_likely_faulty_input_false_for_plausible_geometry_with_ocr_failure():
    flags = evaluate_input_quality_flags(
        frame_w=1200,
        frame_h=900,
        min_ticks=4,
        axis_bbox=(980, 100, 1100, 760),
        plot_bbox=(120, 100, 970, 760),
        axis_crop_w=120,
        axis_crop_h=660,
        raw_detections_count=7,
        parsed_numeric_candidates_count=5,
        merged_ticks_count=2,
    )

    assert flags["likely_faulty_input"] is False
    assert _QUALITY_REASON_INSUFFICIENT_TICKS in flags["reasons"]


def test_likely_faulty_input_false_when_no_ocr_candidates():
    flags = evaluate_input_quality_flags(
        frame_w=1200,
        frame_h=900,
        min_ticks=4,
        axis_bbox=(1080, 100, 1120, 260),
        plot_bbox=(300, 120, 950, 280),
        axis_crop_w=180,
        axis_crop_h=160,
        raw_detections_count=0,
        parsed_numeric_candidates_count=0,
        merged_ticks_count=1,
    )

    assert flags["likely_faulty_input"] is False
    assert _QUALITY_REASON_INSUFFICIENT_TICKS in flags["reasons"]
