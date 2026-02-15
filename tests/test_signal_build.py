from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c4.schema import Scenario, ScenarioWaypoint
from c4.stages import signal_build


def test_targets_within_bounds_and_no_duplicates():
    scenario = Scenario(
        movement_type="direct",
        patterns=[],
        waypoints=[
            ScenarioWaypoint(label="current", price=100.0, conf=0.6),
            ScenarioWaypoint(label="entry", price=95.0, conf=0.5),
            ScenarioWaypoint(label="target", price=1000.0, conf=0.5),
            ScenarioWaypoint(label="target", price=1000.0, conf=0.5),
        ],
        confidence=0.55,
    )
    res = signal_build.run(
        scenario=scenario,
        elements=[],
        scenario_debug={"price_min": 50.0, "price_max": 300.0, "bounds_source": "axis_ticks"},
    )
    assert res.data.targets
    assert all(50.0 <= t <= 300.0 for t in res.data.targets)
    assert len(res.data.targets) == len(set(res.data.targets))
    assert all(abs(t - 100.0) > 1e-6 for t in res.data.targets)


def test_confidence_increases_with_entry_and_target():
    base = Scenario(
        movement_type="direct",
        patterns=[],
        waypoints=[ScenarioWaypoint(label="current", price=100.0, conf=0.6)],
        confidence=0.5,
    )
    rich = Scenario(
        movement_type="direct",
        patterns=[],
        waypoints=[
            ScenarioWaypoint(label="current", price=100.0, conf=0.6),
            ScenarioWaypoint(label="entry", price=98.0, conf=0.45),
            ScenarioWaypoint(label="target", price=120.0, conf=0.5),
        ],
        confidence=0.5,
    )
    out_base = signal_build.run(base, elements=[])
    out_rich = signal_build.run(rich, elements=[])
    assert out_rich.data.confidence > out_base.data.confidence


def test_target_not_equal_current_unless_only_bound():
    scenario = Scenario(
        movement_type="direct",
        patterns=[],
        waypoints=[
            ScenarioWaypoint(label="current", price=150.0, conf=0.65),
            ScenarioWaypoint(label="target", price=150.0, conf=0.5),
            ScenarioWaypoint(label="target", price=180.0, conf=0.5),
        ],
        confidence=0.6,
    )
    out = signal_build.run(scenario=scenario, elements=[])
    assert 150.0 not in out.data.targets
    assert 180.0 in out.data.targets
