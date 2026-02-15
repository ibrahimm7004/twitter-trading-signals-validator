"""Utilities to map axis y-pixels to prices from OCR ticks."""

from __future__ import annotations

from bisect import bisect_left
import math
from typing import Callable, Iterable


def _tick_get(tick: object, key: str) -> float:
    if isinstance(tick, dict):
        return float(tick[key])
    return float(getattr(tick, key))


def make_price_mapper(axis_ticks: Iterable[object], scale: str) -> Callable[[float], float]:
    """Create a deterministic mapper from axis y-pixels to price values."""
    ticks = sorted(
        [{"y_px": _tick_get(t, "y_px"), "value": _tick_get(t, "value")} for t in axis_ticks],
        key=lambda t: t["y_px"],
    )
    if not ticks:
        raise ValueError("axis_ticks must not be empty")

    if len(ticks) == 1:
        only = float(ticks[0]["value"])
        return lambda _y: only

    if scale == "log":
        if any(t["value"] <= 0.0 for t in ticks):
            raise ValueError("log scale requires strictly positive prices")

    y_values = [float(t["y_px"]) for t in ticks]

    def _interp(y: float, a: dict[str, float], b: dict[str, float]) -> float:
        y0 = float(a["y_px"])
        y1 = float(b["y_px"])
        v0 = float(a["value"])
        v1 = float(b["value"])

        if y1 == y0:
            return v0
        t = (float(y) - y0) / (y1 - y0)
        if scale == "log":
            lv0 = math.log(v0)
            lv1 = math.log(v1)
            return float(math.exp(lv0 + t * (lv1 - lv0)))
        return float(v0 + t * (v1 - v0))

    def mapper(y_axis_px: float) -> float:
        y = float(y_axis_px)
        idx = bisect_left(y_values, y)

        if idx <= 0:
            return _interp(y, ticks[0], ticks[1])
        if idx >= len(ticks):
            return _interp(y, ticks[-2], ticks[-1])

        low = ticks[idx - 1]
        high = ticks[idx]
        return _interp(y, low, high)

    return mapper

