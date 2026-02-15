from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c4.schema import OutputSchema, empty_output
from c4.types import AbstainReason, ReasonCode


def test_output_schema_keys():
    output = empty_output()
    data = output.model_dump()
    assert set(data.keys()) == {
        "chart_frame",
        "axis_ticks",
        "elements",
        "scenario",
        "signal",
        "abstain",
        "abstain_reasons",
        "debug_artifacts",
    }


def test_abstain_reason_model():
    reason = AbstainReason(
        code=ReasonCode.NOT_IMPLEMENTED,
        stage="frame",
        message="stub",
        details={"a": 1},
    )
    payload = reason.model_dump()
    assert payload["code"] == ReasonCode.NOT_IMPLEMENTED
    assert payload["stage"] == "frame"
    assert payload["details"]["a"] == 1
