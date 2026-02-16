"""Tests for axis tick monotonicity evaluation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c4.eval_checks import evaluate_axis_tick_monotonicity


def _make_tick(y_px: float, value: float) -> dict[str, float]:
    return {"y_px": y_px, "value": value}


def test_monotonic_passes_when_strictly_increasing():
    ticks = [
        _make_tick(10, 100.0),
        _make_tick(20, 110.0),
        _make_tick(30, 120.0),
    ]
    result = evaluate_axis_tick_monotonicity(ticks)
    assert result.passed is True
    assert result.violations == 0
    assert result.drop_one_repaired is False


def test_monotonic_drop_one_repair_handles_single_bad_tick():
    ticks = [
        _make_tick(10, 10.0),
        _make_tick(20, 30.0),
        _make_tick(30, 20.0),
        _make_tick(40, 40.0),
    ]
    result = evaluate_axis_tick_monotonicity(ticks)
    assert result.passed is True
    assert result.drop_one_repaired is True
    assert result.violations == 0


def test_monotonic_fails_when_two_violations_remain():
    ticks = [
        _make_tick(10, 10.0),
        _make_tick(20, 30.0),
        _make_tick(30, 20.0),
        _make_tick(40, 15.0),
        _make_tick(50, 40.0),
    ]
    result = evaluate_axis_tick_monotonicity(ticks)
    assert result.passed is False
    assert result.drop_one_repaired is False
    assert result.violations >= 1
