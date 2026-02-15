"""Calibration stage (robust mapping fit)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..config import C4Config
from ..schema import AxisTick
from ..types import AbstainReason, ReasonCode, StageResult


@dataclass
class CalibrationResult:
    scale: str
    params: dict
    confidence: float
    fit_error: float
    num_ticks_used: int


def _theil_sen_fit(y: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    slopes: list[float] = []
    n = len(y)
    for i in range(n):
        for j in range(i + 1, n):
            dy = y[j] - y[i]
            if abs(dy) < 1e-6:
                continue
            slopes.append((v[j] - v[i]) / dy)
    if not slopes:
        return 0.0, float(np.median(v))
    slope = float(np.median(slopes))
    intercept = float(np.median(v - slope * y))
    return slope, intercept


def _median_abs_perc_error(true: np.ndarray, pred: np.ndarray) -> float:
    denom = np.maximum(1.0, np.abs(true))
    return float(np.median(np.abs(true - pred) / denom))


def _fit_linear(y: np.ndarray, v: np.ndarray) -> tuple[dict, float]:
    slope, intercept = _theil_sen_fit(y, v)
    pred = slope * y + intercept
    error = _median_abs_perc_error(v, pred)
    return {"slope": slope, "intercept": intercept}, error


def _fit_log(y: np.ndarray, v: np.ndarray) -> tuple[dict, float] | None:
    mask = v > 0
    if mask.sum() < 2:
        return None
    y_pos = y[mask]
    v_pos = v[mask]
    logv = np.log(v_pos)
    slope, intercept = _theil_sen_fit(y_pos, logv)
    pred_log = slope * y_pos + intercept
    pred = np.exp(pred_log)
    error = _median_abs_perc_error(v_pos, pred)
    return {"slope": slope, "intercept": intercept}, error


def run(axis_ticks: Iterable[AxisTick], config: C4Config) -> StageResult[CalibrationResult]:
    ticks = list(axis_ticks)
    if len(ticks) < config.thresholds.min_ticks:
        reason = AbstainReason(
            code=ReasonCode.CALIBRATION_FAILED,
            stage="calibration",
            message="Insufficient ticks for calibration.",
            details={"min_ticks": config.thresholds.min_ticks, "observed": len(ticks)},
        )
        return StageResult(data=None, confidence=0.0, abstain=True, reasons=[reason], debug={})

    y = np.array([t.y_px for t in ticks], dtype=np.float64)
    v = np.array([t.value for t in ticks], dtype=np.float64)

    lin_params, lin_err = _fit_linear(y, v)
    log_fit = _fit_log(y, v)
    log_err = None
    log_params = None
    if log_fit is not None:
        log_params, log_err = log_fit

    slope = lin_params["slope"]
    monotonic_ok = slope < 0
    if not monotonic_ok:
        lin_err *= 1.5

    chosen_scale = "linear"
    chosen_params = lin_params
    chosen_err = lin_err
    if log_err is not None and log_params is not None:
        if abs(lin_err - log_err) < 0.01:
            reason = AbstainReason(
                code=ReasonCode.CALIBRATION_FAILED,
                stage="calibration",
                message="Ambiguous scale selection.",
                details={"linear_err": lin_err, "log_err": log_err},
            )
            return StageResult(data=None, confidence=0.0, abstain=True, reasons=[reason], debug={})
        if log_err < lin_err:
            chosen_scale = "log"
            chosen_params = log_params
            chosen_err = log_err

    if chosen_err > config.thresholds.calibration_max_error:
        reason = AbstainReason(
            code=ReasonCode.CALIBRATION_FAILED,
            stage="calibration",
            message="Calibration error too high.",
            details={"max_error": config.thresholds.calibration_max_error, "observed": chosen_err},
        )
        return StageResult(data=None, confidence=0.0, abstain=True, reasons=[reason], debug={})

    confidence = max(0.0, 1.0 - chosen_err / max(1e-6, config.thresholds.calibration_max_error))
    confidence *= min(1.0, len(ticks) / max(1, config.thresholds.min_ticks))

    result = CalibrationResult(
        scale=chosen_scale,
        params=chosen_params,
        confidence=confidence,
        fit_error=chosen_err,
        num_ticks_used=len(ticks),
    )
    return StageResult(data=result, confidence=confidence, abstain=False, reasons=[], debug={})
