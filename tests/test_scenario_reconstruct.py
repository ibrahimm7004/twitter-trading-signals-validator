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
