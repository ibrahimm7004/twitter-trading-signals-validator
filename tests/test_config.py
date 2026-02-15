from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c4.config import load_config


def test_load_config_with_overrides(tmp_path: Path):
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        """
thresholds:
  min_ticks: 5
  frame_min_conf: 0.6
runtime:
  abstain_on_not_implemented: true
paths:
  debug_overlay_name: "overlay.png"
  axis_debug_name: "axis.png"
""".strip(),
        encoding="utf-8",
    )

    overrides = {"thresholds": {"min_ticks": 9, "axis_min_conf": 0.8}}
    cfg = load_config(config_path, overrides)

    assert cfg.thresholds.min_ticks == 9
    assert cfg.thresholds.axis_min_conf == 0.8
    assert cfg.thresholds.frame_min_conf == 0.6
    assert cfg.runtime.abstain_on_not_implemented is True
    assert cfg.paths.axis_debug_name == "axis.png"
