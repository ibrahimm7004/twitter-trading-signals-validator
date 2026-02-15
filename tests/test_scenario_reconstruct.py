from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c4.config import C4Config
from c4.models import ChartElement
from c4.stages import scenario_reconstruct


def test_zone_waypoints_present_with_low_high_hints():
    cfg = C4Config()
    elements = [
        ChartElement(
            kind="zone",
            geometry_px={"bbox": [100.0, 100.0, 300.0, 200.0]},
            prices={"low": 90.0, "high": 110.0},
            confidence=0.8,
        )
    ]
    out = scenario_reconstruct.run([0.0, 0.0, 1000.0, 600.0], elements, cfg)
    wps = [w for w in out.data.waypoints if w.label == "zone"]
    assert len(wps) == 2
    hints = sorted([w.time_hint for w in wps])
    assert hints == ["high", "low"]


def test_long_line_sets_direct_and_current_target():
    cfg = C4Config()
    elements = [
        ChartElement(
            kind="line",
            geometry_px={"p1": [100.0, 400.0], "p2": [900.0, 100.0]},
            prices={"p1": 100.0, "p2": 200.0},
            confidence=0.8,
        ),
        ChartElement(
            kind="zone",
            geometry_px={"bbox": [200.0, 150.0, 400.0, 220.0]},
            prices={"low": 180.0, "high": 260.0},
            confidence=0.7,
        ),
    ]
    out = scenario_reconstruct.run([0.0, 0.0, 1000.0, 600.0], elements, cfg)
    assert out.data.movement_type == "direct"
    labels = [w.label for w in out.data.waypoints]
    assert "current" in labels
    assert "target" in labels


def test_confidence_positive_when_waypoints_present():
    cfg = C4Config()
    elements = [
        ChartElement(
            kind="line",
            geometry_px={"p1": [50.0, 300.0], "p2": [900.0, 120.0]},
            prices={"p1": 120.0, "p2": 180.0},
            confidence=0.7,
        )
    ]
    out = scenario_reconstruct.run([0.0, 0.0, 1000.0, 600.0], elements, cfg)
    assert 0.0 <= out.data.confidence <= 1.0
    assert out.data.confidence > 0.0


def test_current_price_uses_current_x_when_provided():
    cfg = C4Config()
    elements = [
        ChartElement(
            kind="line",
            geometry_px={"p1": [100.0, 300.0], "p2": [900.0, 100.0]},
            prices={"p1": 100.0, "p2": 200.0},
            confidence=0.8,
        )
    ]
    out = scenario_reconstruct.run([0.0, 0.0, 1000.0, 600.0], elements, cfg, current_x_px=500.0)
    current = next(w for w in out.data.waypoints if w.label == "current")
    assert abs(current.price - 150.0) < 1e-6


def test_projection_line_excluded_from_movement_type_when_current_x_provided():
    cfg = C4Config()
    elements = [
        ChartElement(
            kind="line",
            geometry_px={"p1": [900.0, 300.0], "p2": [1000.0, 320.0]},
            prices={"p1": 300.0, "p2": 50.0},  # strong bearish projection to the right
            confidence=0.8,
        ),
        ChartElement(
            kind="line",
            geometry_px={"p1": [650.0, 320.0], "p2": [820.0, 260.0]},
            prices={"p1": 100.0, "p2": 160.0},  # real bullish line near current
            confidence=0.8,
        ),
    ]
    out = scenario_reconstruct.run([0.0, 0.0, 1000.0, 600.0], elements, cfg, current_x_px=800.0)
    assert out.data.movement_type == "direct"
    assert out.debug["used_main_line_bucket"] == "near_current"
    assert out.debug["num_lines_right"] >= 1


def test_current_prefers_near_current_line_over_global_longest():
    cfg = C4Config()
    elements = [
        ChartElement(
            kind="line",
            geometry_px={"p1": [100.0, 300.0], "p2": [700.0, 300.0]},
            prices={"p1": 50.0, "p2": 200.0},  # longest, but far left of current
            confidence=0.8,
        ),
        ChartElement(
            kind="line",
            geometry_px={"p1": [780.0, 300.0], "p2": [860.0, 260.0]},
            prices={"p1": 100.0, "p2": 140.0},  # shorter near current
            confidence=0.8,
        ),
    ]
    out = scenario_reconstruct.run([0.0, 0.0, 1000.0, 600.0], elements, cfg, current_x_px=820.0)
    current = next(w for w in out.data.waypoints if w.label == "current")
    # Interpolate on near-current line: t=(820-780)/(860-780)=0.5 => 120
    assert abs(current.price - 120.0) < 1e-6
    assert out.debug["used_line_for_current"] == "near_current"


def test_prices_are_clamped_to_axis_tick_bounds():
    cfg = C4Config()
    elements = [
        ChartElement(
            kind="line",
            geometry_px={"p1": [650.0, 320.0], "p2": [820.0, 260.0]},
            prices={"p1": 80.0, "p2": -20.0},
            confidence=0.8,
        ),
        ChartElement(
            kind="zone",
            geometry_px={"bbox": [600.0, 250.0, 900.0, 380.0]},
            prices={"low": 40.0, "high": 1200.0},
            confidence=0.8,
        ),
    ]
    axis_ticks = [{"value": 50.0}, {"value": 1000.0}, {"text": "500"}]
    out = scenario_reconstruct.run(
        [0.0, 0.0, 1000.0, 600.0],
        elements,
        cfg,
        current_x_px=800.0,
        axis_ticks=axis_ticks,
    )
    for wp in out.data.waypoints:
        assert 50.0 <= wp.price <= 1000.0
    assert out.debug["bounds_source"] == "axis_ticks"
    assert out.debug["num_prices_clamped"] > 0


def test_bounds_source_axis_ticks_when_ticks_present():
    cfg = C4Config()
    elements = [
        ChartElement(
            kind="line",
            geometry_px={"p1": [100.0, 300.0], "p2": [900.0, 100.0]},
            prices={"p1": 100.0, "p2": 200.0},
            confidence=0.8,
        )
    ]
    axis_ticks = [{"value": 10.0}, {"value": 20.0}, {"value": 30.0}, {"value": 40.0}]
    out = scenario_reconstruct.run(
        [0.0, 0.0, 1000.0, 600.0],
        elements,
        cfg,
        current_x_px=500.0,
        axis_ticks=axis_ticks,
    )
    assert out.debug["bounds_source"] == "axis_ticks"
    assert out.debug["num_tick_values_used"] >= 2
