"""Scenario reconstruction stage from detected chart elements."""

from __future__ import annotations

from math import hypot

from ..config import C4Config
from ..models import ChartElement
from ..schema import Scenario, ScenarioWaypoint
from ..types import StageResult


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _line_points_with_prices(elem: ChartElement) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if elem.kind != "line":
        return None
    p1 = elem.geometry_px.get("p1")
    p2 = elem.geometry_px.get("p2")
    price1 = elem.prices.get("p1")
    price2 = elem.prices.get("p2")
    if not (isinstance(p1, list) and isinstance(p2, list) and len(p1) >= 2 and len(p2) >= 2):
        return None
    if price1 is None or price2 is None:
        return None
    return (float(p1[0]), float(p1[1]), float(price1)), (float(p2[0]), float(p2[1]), float(price2))


def run(
    plot_bbox: list[float],
    elements: list[ChartElement],
    config: C4Config,
    current_x_px: float | None = None,
) -> StageResult[Scenario]:
    plot_x0, plot_y0, plot_x1, plot_y1 = [float(v) for v in plot_bbox]
    plot_w = max(1.0, plot_x1 - plot_x0)
    plot_h = max(1.0, plot_y1 - plot_y0)
    plot_diag = hypot(plot_w, plot_h)

    line_infos: list[dict] = []
    zone_vals: list[dict] = []

    for elem in elements:
        if elem.kind == "zone":
            low = elem.prices.get("low")
            high = elem.prices.get("high")
            if low is None or high is None:
                continue
            lo = float(min(low, high))
            hi = float(max(low, high))
            zone_vals.append({"low": lo, "high": hi, "conf": min(0.9, float(elem.confidence))})
            continue

        line_points = _line_points_with_prices(elem)
        if line_points is None:
            continue
        a, b = line_points
        length = hypot(b[0] - a[0], b[1] - a[1])
        if a[0] <= b[0]:
            left, right = a, b
        else:
            left, right = b, a
        line_infos.append(
            {
                "left": left,
                "right": right,
                "length": float(length),
                "dx": float(right[0] - left[0]),
                "dprice": float(right[2] - left[2]),
            }
        )

    waypoints: list[ScenarioWaypoint] = []
    patterns: list[str] = []
    movement_type = "unknown"

    # Current from line-evaluated price at current_x (when provided), else right-most endpoint.
    current_price: float | None = None
    if line_infos:
        main_for_current = max(line_infos, key=lambda ln: ln["length"])
        used_eval = False
        if current_x_px is not None:
            left = main_for_current["left"]
            right = main_for_current["right"]
            dx = float(right[0] - left[0])
            if abs(dx) > 1e-6:
                t = (float(current_x_px) - float(left[0])) / dx
                t = max(0.0, min(1.0, t))
                current_price = float(left[2] + t * (right[2] - left[2]))
                waypoints.append(ScenarioWaypoint(label="current", price=current_price, conf=0.60))
                used_eval = True
        if not used_eval:
            endpoints = []
            for ln in line_infos:
                endpoints.append(ln["left"])
                endpoints.append(ln["right"])
            current = max(endpoints, key=lambda p: p[0])
            current_price = float(current[2])
            waypoints.append(ScenarioWaypoint(label="current", price=current_price, conf=0.55))

    # Entry from zones around current.
    entry_price: float | None = None
    invalidation_price: float | None = None

    # Main line and movement type.
    main_line = max(line_infos, key=lambda ln: ln["length"]) if line_infos else None
    informative = False
    bullish = False
    if main_line is not None:
        informative = (main_line["dx"] >= 0.10 * plot_w) and (main_line["length"] >= 0.15 * plot_diag)
        if informative:
            movement_type = "direct"
            bullish = main_line["dprice"] > 0.0
            sign_main = 1 if main_line["dprice"] > 0 else -1
            for other in line_infos:
                if other is main_line:
                    continue
                if other["length"] < 0.6 * main_line["length"]:
                    continue
                sign_other = 1 if other["dprice"] > 0 else -1
                if sign_other != sign_main:
                    movement_type = "conditional"
                    patterns = ["sweep"]
                    break

    # Zone waypoints (sorted by price asc, low/high entries included).
    zone_points: list[ScenarioWaypoint] = []
    for z in zone_vals:
        zone_points.append(ScenarioWaypoint(label="zone", price=float(z["low"]), time_hint="low", conf=float(z["conf"])))
        zone_points.append(ScenarioWaypoint(label="zone", price=float(z["high"]), time_hint="high", conf=float(z["conf"])))
    zone_points.sort(key=lambda wp: wp.price)

    if informative and current_price is not None:
        line_prices = [ln["left"][2] for ln in line_infos] + [ln["right"][2] for ln in line_infos]
        zone_highs = [z["high"] for z in zone_vals]
        zone_lows = [z["low"] for z in zone_vals]

        target_candidates: list[float] = []
        if bullish:
            target_candidates.extend(line_prices)
            target_candidates.extend(zone_highs)
            if target_candidates:
                target = max(target_candidates)
                if abs(target - current_price) > 1e-6:
                    waypoints.append(ScenarioWaypoint(label="target", price=float(target), conf=0.50))
            eligible_entry = [z["high"] for z in zone_vals if z["high"] <= current_price]
            if eligible_entry:
                entry_price = min(eligible_entry, key=lambda p: abs(p - current_price))
                waypoints.append(ScenarioWaypoint(label="entry", price=float(entry_price), conf=0.45))
            if entry_price is not None:
                lows_below = [z["low"] for z in zone_vals if z["low"] < entry_price]
                if lows_below:
                    invalidation_price = min(lows_below)
                    waypoints.append(ScenarioWaypoint(label="invalidation", price=float(invalidation_price), conf=0.40))
        else:
            target_candidates.extend(line_prices)
            target_candidates.extend(zone_lows)
            if target_candidates:
                target = min(target_candidates)
                if abs(target - current_price) > 1e-6:
                    waypoints.append(ScenarioWaypoint(label="target", price=float(target), conf=0.50))
            eligible_entry = [z["low"] for z in zone_vals if z["low"] >= current_price]
            if eligible_entry:
                entry_price = min(eligible_entry, key=lambda p: abs(p - current_price))
                waypoints.append(ScenarioWaypoint(label="entry", price=float(entry_price), conf=0.45))
            if entry_price is not None:
                highs_above = [z["high"] for z in zone_vals if z["high"] > entry_price]
                if highs_above:
                    invalidation_price = max(highs_above)
                    waypoints.append(ScenarioWaypoint(label="invalidation", price=float(invalidation_price), conf=0.40))

    # Stable ordering: current, entry, zones(sorted), target, invalidation.
    ordered: list[ScenarioWaypoint] = []
    ordered.extend([wp for wp in waypoints if wp.label == "current"])
    ordered.extend([wp for wp in waypoints if wp.label == "entry"])
    ordered.extend(zone_points)
    ordered.extend([wp for wp in waypoints if wp.label == "target"])
    ordered.extend([wp for wp in waypoints if wp.label == "invalidation"])

    if ordered:
        conf = _clamp01(sum(wp.conf for wp in ordered) / len(ordered))
    else:
        conf = 0.0

    scenario = Scenario(
        movement_type=movement_type,
        patterns=patterns,
        waypoints=ordered,
        confidence=conf,
    )
    return StageResult(data=scenario, confidence=conf, abstain=False, reasons=[], debug={})
