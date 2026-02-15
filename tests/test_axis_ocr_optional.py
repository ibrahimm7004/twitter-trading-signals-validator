from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c4.config import C4Config
from c4.stages import axis_ocr
from c4.types import ReasonCode


def _make_axis_image(path: Path) -> list[float]:
    width, height = 200, 300
    image = np.full((height, width, 3), 230, dtype=np.uint8)
    for idx, y in enumerate(range(40, 260, 30)):
        cv2.putText(
            image,
            f"{500 - idx * 5}",
            (80, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), image)
    return [0.0, 0.0, float(width - 1), float(height - 1)]


def test_axis_ocr_optional():
    pytest.importorskip("paddleocr")
    tmp_dir = Path(__file__).resolve().parent
    image_path = tmp_dir / "tmp_axis_ocr.png"
    axis_bbox = _make_axis_image(image_path)

    cfg = C4Config()
    cfg.runtime.ocr_backend = "paddle"
    result = axis_ocr.run(str(image_path), axis_bbox=axis_bbox, config=cfg)

    if result.abstain:
        assert any(r.code == ReasonCode.AXIS_OCR_INSUFFICIENT_TICKS for r in result.reasons)
    else:
        assert len(result.data) >= cfg.thresholds.min_ticks

    if image_path.exists():
        image_path.unlink()
