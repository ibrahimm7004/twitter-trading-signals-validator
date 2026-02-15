from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c4.config import C4Config
from c4.pipeline import run_pipeline
from c4.schema import OutputSchema


def _fixture_images() -> list[Path]:
    candidates = [ROOT / "chart1.jpeg", ROOT / "chart2.jpeg"]
    available = [p for p in candidates if p.exists()]
    if available:
        return available[:2]
    fallback = [ROOT / "test-images" / "chart1.jpeg", ROOT / "test-images" / "chart2.jpeg"]
    return [p for p in fallback if p.exists()][:2]


def test_pipeline_e2e_signal_and_schema(tmp_path: Path):
    images = _fixture_images()
    assert images, "No fixture images found for e2e test."

    cfg = C4Config()
    cfg.runtime.ocr_backend = "disabled"

    for img in images:
        out = run_pipeline(image_path=img, config=cfg, debug_dir=tmp_path / img.stem)
        payload = out.model_dump()
        validated = OutputSchema.model_validate(payload)
        assert validated.signal.direction in {"long", "short", "unknown"}
        if not validated.abstain:
            assert validated.signal.confidence > 0.0
            # Bounds are currently available in scenario stage debug; enforce if present in output signal rationale.
            if "bounds:axis_ticks" in validated.signal.rationale:
                price_values = [t.value for t in validated.axis_ticks]
                if len(price_values) >= 2:
                    lo = min(price_values)
                    hi = max(price_values)
                    for t in validated.signal.targets:
                        assert lo <= t <= hi
