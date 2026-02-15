"""Shared types and primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class ReasonCode(str, Enum):
    FRAME_NOT_FOUND = "FRAME_NOT_FOUND"
    AXIS_OCR_INSUFFICIENT_TICKS = "AXIS_OCR_INSUFFICIENT_TICKS"
    CALIBRATION_FAILED = "CALIBRATION_FAILED"
    ELEMENTS_NOT_FOUND = "ELEMENTS_NOT_FOUND"
    SCENARIO_INCOHERENT = "SCENARIO_INCOHERENT"
    IMAGE_READ_FAILED = "IMAGE_READ_FAILED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    FRAME_NOT_IMPLEMENTED = "FRAME_NOT_IMPLEMENTED"
    OCR_NOT_IMPLEMENTED = "OCR_NOT_IMPLEMENTED"
    CALIBRATION_NOT_IMPLEMENTED = "CALIBRATION_NOT_IMPLEMENTED"


class AbstainReason(BaseModel):
    """Structured abstain reason with stable code."""

    model_config = ConfigDict(extra="forbid")

    code: ReasonCode
    stage: str
    message: str
    details: dict = Field(default_factory=dict)


T = TypeVar("T")


@dataclass(frozen=True)
class StageResult(Generic[T]):
    """Result container for pipeline stages."""

    data: Optional[T] = None
    confidence: float = 0.0
    abstain: bool = False
    reasons: list[AbstainReason] = field(default_factory=list)
    debug: dict = field(default_factory=dict)
