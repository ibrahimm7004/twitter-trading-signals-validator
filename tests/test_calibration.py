from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c4.config import C4Config
from c4.schema import AxisTick
from c4.stages import calibration


def test_linear_calibration_recovers_mapping():
    y_vals = np.linspace(0, 100, 10)
    slope = -2.5
    intercept = 300.0
    ticks = [AxisTick(value=float(slope * y + intercept), y_px=float(y), conf=0.9) for y in y_vals]
    cfg = C4Config()
    result = calibration.run(ticks, cfg)
    assert not result.abstain
    assert result.data is not None
    assert result.data.scale == "linear"
    assert math.isclose(result.data.params["slope"], slope, rel_tol=0.15, abs_tol=0.2)
    assert math.isclose(result.data.params["intercept"], intercept, rel_tol=0.1, abs_tol=1.0)


def test_log_calibration_selected():
    y_vals = np.linspace(0, 60, 12)
    a = -0.02
    b = math.log(1000.0)
    values = np.exp(a * y_vals + b)
    ticks = [AxisTick(value=float(v), y_px=float(y), conf=0.9) for y, v in zip(y_vals, values)]
    cfg = C4Config()
    result = calibration.run(ticks, cfg)
    assert not result.abstain
    assert result.data is not None
    assert result.data.scale in ("log", "linear")
    assert result.data.scale == "log"
