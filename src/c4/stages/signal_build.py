"""Build final trading signal summary from scenario + elements."""

from __future__ import annotations

from typing import Any

from ..models import ChartElement
from ..schema import Scenario, Signal, ScenarioWaypoint
from ..types import StageResult


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _dedupe_sorted(values: list[float], eps: float = 1e-6) -> list[float]:
    out: list[float] = []
    for v in sorted(values):
        if not out or abs(v - out[-1]) > eps:
            out.append(float(v))
    return out


def _extract_bounds(price_bounds: tuple[float, float] | None, scenario_debug: dict[str, Any] | None) -> tuple[float, float] | None:
    if price_bounds is not None and len(price_bounds) == 2:
        lo, hi = float(price_bounds[0]), float(price_bounds[1])
        return (min(lo, hi), max(lo, hi))
    if isinstance(scenario_debug, dict):
        lo = scenario_debug.get("price_min")
        hi = scenario_debug.get("price_max")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            lo_f, hi_f = float(lo), float(hi)
            if lo_f <= hi_f:
                return (lo_f, hi_f)
            return (hi_f, lo_f)
    return None


def run(
    scenario: Scenario | None,
    elements: list[ChartElement] | None,
    scenario_debug: dict[str, Any] | None = None,
    price_bounds: tuple[float, float] | None = None,
) -> StageResult[Signal]:
    if scenario is None:
        signal = Signal(direction="unknown", entry=None, stop=None, targets=[], rationale=["scenario_missing"], confidence=0.05)
        return StageResult(data=signal, confidence=signal.confidence, abstain=False, reasons=[], debug={})

    waypoints: list[ScenarioWaypoint] = list(scenario.waypoints or [])
    current_wp = next((w for w in waypoints if w.label == "current"), None)
    entry_wp = next((w for w in waypoints if w.label == "entry"), None)
    target_wps = [w for w in waypoints if w.label == "target"]
    zone_lows = [w.price for w in waypoints if w.label == "zone" and w.time_hint == "low"]
    zone_highs = [w.price for w in waypoints if w.label == "zone" and w.time_hint == "high"]

    current = float(current_wp.price) if current_wp is not None else None
    entry = float(entry_wp.price) if entry_wp is not None else None
    targets = [float(w.price) for w in target_wps]

    direction = "unknown"
    if scenario.movement_type != "unknown":
        if current is not None and targets:
            up = any(t > current + 1e-6 for t in targets)
            down = any(t < current - 1e-6 for t in targets)
            if up and not down:
                direction = "long"
            elif down and not up:
                direction = "short"
        if direction == "unknown" and current is not None and entry is not None:
            if entry > current + 1e-6:
                direction = "long"
            elif entry < current - 1e-6:
                direction = "short"

    bounds = _extract_bounds(price_bounds, scenario_debug)
    num_clamped = 0

    def _clamp(v: float) -> float:
        nonlocal num_clamped
        if bounds is None:
            return float(v)
        lo, hi = bounds
        c = min(max(float(v), lo), hi)
        if abs(c - float(v)) > 1e-9:
            num_clamped += 1
        return c

    if current is not None:
        current = _clamp(current)
    if entry is not None:
        entry = _clamp(entry)

    targets = [_clamp(t) for t in targets]
    if not targets and current is not None:
        if direction == "long":
            candidates = [_clamp(v) for v in zone_highs if v > current + 1e-6]
            if candidates:
                targets = [max(candidates)]
        elif direction == "short":
            candidates = [_clamp(v) for v in zone_lows if v < current - 1e-6]
            if candidates:
                targets = [min(candidates)]

    targets = _dedupe_sorted(targets)
    if current is not None:
        targets = [t for t in targets if abs(t - current) > 1e-6]
    if direction == "long":
        targets = [t for t in targets if current is None or t > current]
        targets = targets[:2]
    elif direction == "short":
        targets = [t for t in targets if current is None or t < current]
        targets = list(reversed(targets))[:2]
    else:
        targets = targets[:2]

    pivot = entry if entry is not None else current
    stop: float | None = None
    if pivot is not None:
        if direction == "long":
            lows = [_clamp(v) for v in zone_lows if v < pivot - 1e-6]
            if lows:
                stop = max(lows)
        elif direction == "short":
            highs = [_clamp(v) for v in zone_highs if v > pivot + 1e-6]
            if highs:
                stop = min(highs)

    rationale: list[str] = []
    if scenario.movement_type != "unknown":
        rationale.append(f"movement:{scenario.movement_type}")
    for p in scenario.patterns:
        rationale.append(f"pattern:{p}")
    if entry is not None:
        rationale.append("entry_from_scenario")
    if targets:
        rationale.append("target_from_scenario_or_zone")
    if stop is not None:
        rationale.append("stop_from_zone")
    if isinstance(scenario_debug, dict) and scenario_debug.get("bounds_source"):
        rationale.append(f"bounds:{scenario_debug['bounds_source']}")
    if not rationale:
        rationale = ["insufficient_signal_evidence"]

    confidence = float(scenario.confidence)
    if entry is not None and targets:
        confidence += 0.10
    if stop is not None:
        confidence += 0.05
    if direction == "unknown":
        confidence *= 0.8
    confidence = _clamp01(confidence)

    signal = Signal(
        direction=direction,
        entry=entry,
        stop=stop,
        targets=targets,
        rationale=rationale,
        confidence=confidence,
    )
    debug = {"num_prices_clamped_signal": num_clamped, "bounds": bounds}
    return StageResult(data=signal, confidence=confidence, abstain=False, reasons=[], debug=debug)
