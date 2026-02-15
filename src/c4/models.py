"""Shared models for detected chart entities."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ChartElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["zone", "line"]
    geometry_px: dict
    prices: dict
    confidence: float

