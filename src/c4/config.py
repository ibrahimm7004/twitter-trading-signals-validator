"""Configuration system for Component 4."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_ticks: int = 6
    frame_min_conf: float = 0.7
    axis_min_conf: float = 0.7
    calibration_max_error: float = 0.15
    element_min_conf: float = 0.5


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    abstain_on_not_implemented: bool = True
    ocr_backend: str = "disabled"
    axis_width_ratio: float = 0.12
    axis_search_ratio: float = 0.25
    axis_right_band_ratio: float = 0.25
    axis_right_edge_max_gap: int = 2
    axis_text_bright_thresh: int = 160
    axis_min_width_px: int = 45
    axis_max_width_px: int = 160
    axis_cc_min_area: int = 15
    axis_cc_max_area: int = 2000
    axis_cc_min_h: int = 6
    axis_cc_max_h: int = 40
    axis_cc_min_count: int = 8
    axis_digit_roi_left_ratio: float = 0.12
    axis_row_merge_tol_px: int = 10
    axis_cc_min_w: int = 2
    axis_cc_max_w: int = 60
    axis_tick_pad_top_px: int = 18
    axis_tick_pad_bot_px: int = 18
    axis_red_badge_min_pixels: int = 600
    axis_red_badge_margin_px: int = 6


class PathConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    debug_overlay_name: str = "overlay.png"
    axis_debug_name: str = "axis_debug.png"


class C4Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thresholds: Thresholds = Field(default_factory=Thresholds)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    paths: PathConfig = Field(default_factory=PathConfig)


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> C4Config:
    """Load config from YAML and apply overrides."""
    config_path = Path(path)
    raw: dict[str, Any] = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if overrides:
        raw = _deep_update(raw, overrides)
    return C4Config.model_validate(raw)
