"""Unit tests for the axis OCR tick cleaning helpers."""

from __future__ import annotations

from c4.stages.axis_ocr import clean_axis_ticks


DEFAULT_CROP_W = 120


def _make_tick(value: float, y_px: float, width: int = 30, x1_offset: int = 2) -> dict[str, float]:
    x1 = DEFAULT_CROP_W - x1_offset
    x0 = max(0, x1 - width)
    return {
        "y_px": y_px,
        "value": value,
        "conf": 0.5,
        "bbox": (x0, int(y_px - 3), x1, int(y_px + 3)),
    }


def _is_monotonic(values: list[float]) -> bool:
    if len(values) < 2:
        return True
    nondec = all(values[i + 1] >= values[i] for i in range(len(values) - 1))
    noninc = all(values[i + 1] <= values[i] for i in range(len(values) - 1))
    return nondec or noninc


def test_chart45_variance_rescued_and_monotonic():
    ticks = [
        _make_tick(1.4, 0.0),
        _make_tick(1.5, 10.0),
        _make_tick(1.6, 20.0),
        _make_tick(1.7, 30.0),
        _make_tick(1900.0, 40.0),
        _make_tick(526.0, 50.0),
        _make_tick(12.0, 60.0),
    ]
    cleaned = clean_axis_ticks(ticks, DEFAULT_CROP_W)
    values = [tick["value"] for tick in cleaned]
    assert all(abs(value) < 20 for value in values)
    assert _is_monotonic(values)
    assert any(abs(value - 1.9) < 0.05 for value in values)


def test_chart21_zero_and_small_removed():
    ticks = [
        _make_tick(186.0, 0.0),
        _make_tick(174.0, 10.0),
        _make_tick(0.0, 20.0),
        _make_tick(2.0, 30.0),
        _make_tick(162.0, 40.0),
        _make_tick(148.0, 50.0),
    ]
    cleaned = clean_axis_ticks(ticks, DEFAULT_CROP_W)
    values = [tick["value"] for tick in cleaned]
    assert 0.0 not in values
    assert 2.0 not in values
    assert _is_monotonic(values)


def test_multi_panel_cluster_keeps_main_panel():
    main_values = [0.05, 0.045, 0.04, 0.035, 0.03]
    tail_values = [16.0, 14.0, 12.0]
    ticks = [
        _make_tick(val, float(idx * 10)) for idx, val in enumerate(main_values)
    ]
    ticks += [
        _make_tick(val, 150.0 + float(idx * 10)) for idx, val in enumerate(tail_values)
    ]
    cleaned = clean_axis_ticks(ticks, DEFAULT_CROP_W)
    values = [tick["value"] for tick in cleaned]
    assert all(abs(val) < 0.1 for val in values)
    assert _is_monotonic(values)


def test_wide_bbox_price_tag_filtered():
    ticks = [
        _make_tick(1.0, 0.0),
        _make_tick(1.2, 10.0),
        _make_tick(1.4, 20.0),
        _make_tick(99.0, 30.0, width=int(DEFAULT_CROP_W * 0.75)),
        _make_tick(1.6, 40.0),
    ]
    cropped = clean_axis_ticks(ticks, DEFAULT_CROP_W)
    values = [tick["value"] for tick in cropped]
    assert 99.0 not in values
    assert len(values) >= 4
    assert _is_monotonic(values)


def test_right_shifted_ticks_use_relaxed_fallback():
    ticks = [
        _make_tick(2.0, 0.0, x1_offset=20),
        _make_tick(2.2, 10.0, x1_offset=20),
        _make_tick(2.4, 20.0, x1_offset=20),
        _make_tick(2.6, 30.0, x1_offset=20),
        _make_tick(2.8, 40.0, x1_offset=20),
    ]
    cleaned = clean_axis_ticks(ticks, DEFAULT_CROP_W)
    values = [tick["value"] for tick in cleaned]
    assert len(values) >= 4
    assert _is_monotonic(values)
