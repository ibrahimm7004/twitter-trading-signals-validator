"""Pydantic models for Component 4 JSON output schema (M1)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .models import ChartElement
from .types import AbstainReason


BBox = list[float]
Point = list[float]


class ChartFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plot_bbox: BBox
    axis_bbox: BBox
    scale: Literal["linear", "log", "unknown"]
    confidence: float

    @field_validator("plot_bbox", "axis_bbox")
    @classmethod
    def _validate_bbox(cls, value: BBox) -> BBox:
        if len(value) != 4:
            raise ValueError("bbox must have four values")
        return value


class AxisTick(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float
    y_px: float
    conf: float


class ElementKeypoints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: Optional[Point] = None
    end: Optional[Point] = None


class ElementPrices(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: Optional[float] = None
    end: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None


class Element(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal[
        "arrow",
        "box",
        "trendline",
        "channel",
        "fib",
        "handdrawn_path",
    ]
    bbox: BBox
    keypoints_px: ElementKeypoints = Field(default_factory=ElementKeypoints)
    prices: ElementPrices = Field(default_factory=ElementPrices)
    confidence: float
    notes: str = ""

    @field_validator("bbox")
    @classmethod
    def _validate_bbox(cls, value: BBox) -> BBox:
        if len(value) != 4:
            raise ValueError("bbox must have four values")
        return value


class ScenarioWaypoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Literal["current", "entry", "target", "invalidation", "zone"]
    price: float
    time_hint: Optional[str] = None
    conf: float


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    movement_type: Literal["direct", "conditional", "unknown"]
    patterns: list[
        Literal["sweep", "rejection", "break_continue", "pullback_continue"]
    ] = Field(default_factory=list)
    waypoints: list[ScenarioWaypoint] = Field(default_factory=list)
    confidence: float


class Signal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction: Literal["long", "short", "unknown"]
    entry: Optional[float] = None
    stop: Optional[float] = None
    targets: list[float] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    confidence: float


class DebugArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overlay_path: Optional[str] = None
    axis_debug_path: Optional[str] = None


class OutputSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chart_frame: ChartFrame
    axis_ticks: list[AxisTick]
    elements: list[ChartElement]
    scenario: Scenario
    signal: Signal
    abstain: bool
    abstain_reasons: list[AbstainReason]
    debug_artifacts: DebugArtifacts


def empty_output() -> OutputSchema:
    """Create a minimal valid output payload with abstain defaults."""
    chart_frame = ChartFrame(
        plot_bbox=[0.0, 0.0, 0.0, 0.0],
        axis_bbox=[0.0, 0.0, 0.0, 0.0],
        scale="unknown",
        confidence=0.0,
    )
    scenario = Scenario(movement_type="unknown", patterns=[], waypoints=[], confidence=0.0)
    signal = Signal(direction="unknown", entry=None, stop=None, targets=[], rationale=[], confidence=0.0)
    debug_artifacts = DebugArtifacts(overlay_path=None, axis_debug_path=None)
    return OutputSchema(
        chart_frame=chart_frame,
        axis_ticks=[],
        elements=[],
        scenario=scenario,
        signal=signal,
        abstain=True,
        abstain_reasons=[],
        debug_artifacts=debug_artifacts,
    )
