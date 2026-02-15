from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c4.stages.candles_detect import _pick_rightmost_run


def test_pick_rightmost_run_finds_rightmost_valid_run():
    scores = [0, 1, 2, 10, 11, 10, 11, 12, 1, 0, 9, 9, 9, 9, 9, 9]
    idx = _pick_rightmost_run(scores, thresh=8, min_run=6)
    assert idx == 15


def test_pick_rightmost_run_returns_none_when_run_too_short():
    scores = [0, 9, 9, 9, 0, 9, 9, 0]
    idx = _pick_rightmost_run(scores, thresh=8, min_run=4)
    assert idx is None


def test_pick_rightmost_run_prefers_rightmost_of_multiple_runs():
    scores = [9, 9, 9, 9, 0, 0, 8, 8, 8, 8]
    idx = _pick_rightmost_run(scores, thresh=8, min_run=4)
    assert idx == 9

