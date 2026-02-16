from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c4.config import C4Config
from c4.stages import axis_ocr


def test_paddle_ocr_cache_reuses_instance(monkeypatch):
    axis_ocr._OCR_CACHE.clear()
    created: list[object] = []

    def _fake_builder() -> object:
        obj = object()
        created.append(obj)
        return obj

    monkeypatch.setattr(axis_ocr, "_build_paddle_ocr", _fake_builder)

    cfg = C4Config()
    cfg.runtime.ocr_backend = "paddle"
    ocr1 = axis_ocr._get_or_create_paddle_ocr(cfg)
    ocr2 = axis_ocr._get_or_create_paddle_ocr(cfg)

    assert ocr1 is ocr2
    assert len(created) == 1
