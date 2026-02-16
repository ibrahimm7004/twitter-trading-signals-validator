from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c4.config import C4Config
from c4.pipeline import run_pipeline


def _fixture_image() -> Path:
    preferred = ROOT / "chart1.jpeg"
    if preferred.exists():
        return preferred
    fallback = ROOT / "test-images" / "chart1.jpeg"
    if fallback.exists():
        return fallback
    raise AssertionError("No chart1 fixture found for determinism test.")


def test_output_json_deterministic_across_debug_dirs(tmp_path: Path):
    image = _fixture_image()
    cfg = C4Config()
    cfg.runtime.ocr_backend = "disabled"

    out1 = run_pipeline(image_path=image, config=cfg, debug_dir=tmp_path / "run1_debug")
    out2 = run_pipeline(image_path=image, config=cfg, debug_dir=tmp_path / "run2_debug")

    d1 = out1.model_dump()
    d2 = out2.model_dump()

    s1 = json.dumps(d1, sort_keys=True, indent=2)
    s2 = json.dumps(d2, sort_keys=True, indent=2)
    assert s1 == s2

