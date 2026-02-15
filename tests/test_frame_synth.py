from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c4.config import C4Config
from c4.pipeline import run_pipeline
from c4.stages import frame


def _make_synth_chart(path: Path) -> None:
    width, height = 800, 500
    image = np.full((height, width, 3), 30, dtype=np.uint8)
    panel_x0, panel_y0 = 50, 40
    panel_x1, panel_y1 = 750, 460
    cv2.rectangle(image, (panel_x0, panel_y0), (panel_x1, panel_y1), (210, 210, 210), -1)

    axis_width = 90
    axis_x0 = panel_x1 - axis_width
    cv2.rectangle(image, (axis_x0, panel_y0), (panel_x1, panel_y1), (235, 235, 235), -1)

    plot_x1 = axis_x0
    cv2.rectangle(image, (panel_x0, panel_y0), (plot_x1, panel_y1), (220, 220, 220), -1)

    tick_ys = list(range(panel_y0 + 30, panel_y1 - 20, 50))
    for idx, y in enumerate(tick_ys):
        cv2.line(image, (axis_x0 + 5, y), (axis_x0 + 30, y), (60, 60, 60), 2)
        cv2.putText(
            image,
            f"{100 + idx * 5}",
            (axis_x0 + 35, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(path), image)


def test_frame_stage_and_debug_outputs(tmp_path: Path) -> None:
    image_path = tmp_path / "synth.png"
    _make_synth_chart(image_path)

    config = C4Config()
    result = frame.run(str(image_path), config)
    assert result.data is not None

    plot_bbox = result.data.plot_bbox
    axis_bbox = result.data.axis_bbox

    assert axis_bbox[0] >= plot_bbox[2]
    assert plot_bbox[2] <= axis_bbox[0]
    assert plot_bbox[2] < axis_bbox[2]
    assert result.data.confidence >= config.thresholds.frame_min_conf

    debug_dir = tmp_path / "debug"
    output = run_pipeline(str(image_path), config, debug_dir)
    assert output.debug_artifacts.overlay_path is not None
    assert output.debug_artifacts.axis_debug_path is not None
    assert (debug_dir / "overlay.png").exists()
    assert (debug_dir / "axis_debug.png").exists()
    assert (debug_dir / "plot_crop.png").exists()
    assert (debug_dir / "axis_crop.png").exists()


def _make_projection_case(path: Path) -> tuple[int, int, int, int, int]:
    width, height = 1000, 520
    image = np.full((height, width, 3), 25, dtype=np.uint8)
    panel_x0, panel_y0 = 40, 30
    panel_x1, panel_y1 = 960, 490
    cv2.rectangle(image, (panel_x0, panel_y0), (panel_x1, panel_y1), (205, 205, 205), -1)

    axis_width = 60
    axis_x0 = panel_x1 - axis_width
    cv2.rectangle(image, (axis_x0, panel_y0), (panel_x1, panel_y1), (235, 235, 235), -1)

    plot_x1 = axis_x0
    cv2.rectangle(image, (panel_x0, panel_y0), (plot_x1, panel_y1), (220, 220, 220), -1)

    projection_x0 = plot_x1 - 260
    cv2.rectangle(image, (projection_x0, panel_y0 + 20), (plot_x1, panel_y1 - 20), (215, 215, 215), -1)
    cv2.arrowedLine(
        image,
        (projection_x0 + 20, panel_y0 + 60),
        (plot_x1 - 20, panel_y1 - 60),
        (240, 240, 240),
        8,
    )
    cv2.putText(
        image,
        "PROJ",
        (projection_x0 + 30, panel_y0 + 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (250, 250, 250),
        2,
        cv2.LINE_AA,
    )

    tick_ys = list(range(panel_y0 + 40, panel_y1 - 30, 60))
    for idx, y in enumerate(tick_ys):
        cv2.line(image, (axis_x0 + 5, y), (axis_x0 + 22, y), (60, 60, 60), 2)
        cv2.putText(
            image,
            f"{500 + idx * 10:.1f}",
            (axis_x0 + 25, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(path), image)
    return panel_x0, panel_y0, panel_x1, panel_y1, projection_x0


def test_axis_bbox_hugs_right_column(tmp_path: Path) -> None:
    image_path = tmp_path / "projection_case.png"
    panel_x0, panel_y0, panel_x1, panel_y1, projection_x0 = _make_projection_case(image_path)

    config = C4Config()
    result = frame.run(str(image_path), config)
    assert result.data is not None

    axis_bbox = result.data.axis_bbox
    axis_x0 = axis_bbox[0]
    expected_axis_x0 = panel_x1 - 60

    assert axis_x0 >= expected_axis_x0 - 8
    assert axis_x0 > projection_x0 + 80

    debug_dir = tmp_path / "debug_proj"
    output = run_pipeline(str(image_path), config, debug_dir)
    assert output.debug_artifacts.overlay_path is not None
    assert output.debug_artifacts.axis_debug_path is not None
    assert (debug_dir / "overlay.png").exists()
    assert (debug_dir / "axis_debug.png").exists()
    assert (debug_dir / "plot_crop.png").exists()
    assert (debug_dir / "axis_crop.png").exists()


def _make_tradingview_right_scale(path: Path) -> tuple[int, int, int, int, int]:
    width, height = 900, 520
    image = np.full((height, width, 3), 20, dtype=np.uint8)
    panel_x0, panel_y0 = 40, 30
    panel_x1, panel_y1 = 750, 490
    cv2.rectangle(image, (panel_x0, panel_y0), (panel_x1, panel_y1), (210, 210, 210), -1)

    gap_x0 = panel_x1
    gap_x1 = panel_x1 + 1
    cv2.rectangle(image, (gap_x0, panel_y0), (gap_x1, panel_y1), (30, 30, 30), -1)

    axis_x0 = panel_x1 + 1
    axis_x1 = 820
    cv2.rectangle(image, (axis_x0, panel_y0), (axis_x1, panel_y1), (235, 235, 235), -1)

    plot_x1 = panel_x1
    projection_x0 = plot_x1 - 240
    cv2.rectangle(image, (projection_x0, panel_y0 + 40), (plot_x1 - 5, panel_y1 - 40), (220, 220, 220), -1)
    cv2.arrowedLine(
        image,
        (projection_x0 + 30, panel_y0 + 70),
        (plot_x1 - 40, panel_y1 - 90),
        (245, 245, 245),
        8,
    )

    tick_ys = list(range(panel_y0 + 50, panel_y1 - 40, 70))
    for idx, y in enumerate(tick_ys):
        cv2.line(image, (axis_x0 + 5, y), (axis_x0 + 22, y), (60, 60, 60), 2)
        cv2.putText(
            image,
            f"{540 - idx * 7:.1f}",
            (axis_x1 - 60, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(path), image)
    return panel_x0, panel_y0, panel_x1, panel_y1, axis_x1


def test_axis_bbox_anchors_to_image_right_edge(tmp_path: Path) -> None:
    image_path = tmp_path / "tv_right_scale.png"
    panel_x0, panel_y0, panel_x1, panel_y1, axis_x1 = _make_tradingview_right_scale(image_path)

    config = C4Config()
    result = frame.run(str(image_path), config)
    assert result.data is not None

    axis_bbox = result.data.axis_bbox
    plot_bbox = result.data.plot_bbox

    assert abs(axis_bbox[2] - (900 - 1)) <= config.runtime.axis_right_edge_max_gap
    assert axis_bbox[0] >= 740
    assert plot_bbox[2] >= panel_x1
    assert plot_bbox[2] <= axis_bbox[0]


def _make_axis_only_with_badge(path: Path) -> tuple[int, int]:
    width, height = 220, 320
    image = np.full((height, width, 3), 120, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (10, height - 1), (30, 30, 30), -1)
    cv2.rectangle(image, (140, 0), (width - 1, height - 1), (150, 150, 150), -1)
    tick_ys = list(range(30, 220, 30))
    for idx, y in enumerate(tick_ys):
        cv2.putText(
            image,
            f"{500 - idx * 5}",
            (150, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    badge_top = 250
    cv2.rectangle(image, (0, badge_top), (width - 1, height - 1), (0, 0, 255), -1)
    cv2.putText(
        image,
        "BADGE",
        (20, badge_top + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), image)
    return height, badge_top


def test_tick_rows_ignore_red_badge(tmp_path: Path) -> None:
    image_path = tmp_path / "axis_only_badge.png"
    height, badge_top = _make_axis_only_with_badge(image_path)

    config = C4Config()
    result = frame.run(str(image_path), config)
    assert result.data is not None
    tick_rows = result.debug.get("tick_rows", [])
    assert len(tick_rows) >= 6
    assert max(tick_rows) < badge_top - config.runtime.axis_red_badge_margin_px
    assert result.data.axis_bbox[3] <= badge_top - config.runtime.axis_red_badge_margin_px


def _make_sparse_axis_labels(path: Path) -> tuple[int, int, int, int]:
    width, height = 260, 320
    image = np.full((height, width, 3), 140, dtype=np.uint8)
    panel_x0, panel_y0 = 10, 10
    panel_x1, panel_y1 = width - 1, height - 1
    cv2.rectangle(image, (panel_x0, panel_y0), (panel_x1, panel_y1), (200, 200, 200), -1)
    cv2.rectangle(image, (180, 0), (width - 1, height - 1), (150, 150, 150), -1)
    for y in [60, 160, 240]:
        cv2.putText(
            image,
            "123",
            (190, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), image)
    return panel_y0, panel_y1, width, height


def test_no_tighten_when_insufficient_ticks(tmp_path: Path) -> None:
    image_path = tmp_path / "sparse_ticks.png"
    panel_y0, panel_y1, width, height = _make_sparse_axis_labels(image_path)

    config = C4Config()
    result = frame.run(str(image_path), config)
    assert result.data is not None

    assert abs(result.data.axis_bbox[1] - panel_y0) <= 5
    assert abs(result.data.axis_bbox[3] - panel_y1) <= 5


def _make_axis_with_badge_and_many_ticks(path: Path) -> tuple[int, int]:
    width, height = 240, 360
    image = np.full((height, width, 3), 130, dtype=np.uint8)
    cv2.rectangle(image, (150, 0), (width - 1, height - 1), (150, 150, 150), -1)
    tick_ys = list(range(30, 260, 25))
    for idx, y in enumerate(tick_ys):
        cv2.putText(
            image,
            f"{60 - idx}",
            (190, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    badge_top = 280
    cv2.rectangle(image, (0, badge_top), (width - 1, height - 1), (0, 0, 255), -1)
    cv2.putText(
        image,
        "BADGE",
        (20, badge_top + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), image)
    return height, badge_top


def test_tradingview_style_ticks_detected_and_not_collapsed(tmp_path: Path) -> None:
    image_path = tmp_path / "axis_many_ticks_badge.png"
    height, badge_top = _make_axis_with_badge_and_many_ticks(image_path)

    config = C4Config()
    result = frame.run(str(image_path), config)
    assert result.data is not None
    tick_rows = result.debug.get("tick_rows", [])
    assert len(tick_rows) >= config.thresholds.min_ticks
    span = max(tick_rows) - min(tick_rows)
    assert span >= int(0.35 * height)
    assert result.data.axis_bbox[3] <= badge_top - config.runtime.axis_red_badge_margin_px
