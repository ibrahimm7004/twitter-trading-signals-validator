"""Scenario reconstruction stage from detected chart elements."""

from __future__ import annotations

from math import hypot
import re
from typing import Any

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


def _to_float_maybe(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        num = float(value)
        return num if num == num and num not in (float("inf"), float("-inf")) else None
    if isinstance(value, str):
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value.replace(",", ""))
        if m:
            try:
                num = float(m.group(0))
            except ValueError:
                return None
            return num if num == num and num not in (float("inf"), float("-inf")) else None
    return None


def _extract_tick_values(axis_ticks: list[object] | None) -> list[float]:
    if not axis_ticks:
        return []
    values: list[float] = []
    for tick in axis_ticks:
        candidates: list[Any] = []
        if isinstance(tick, dict):
            for key in ("value", "price", "tick_value", "text"):
                if key in tick:
                    candidates.append(tick.get(key))
        else:
            for key in ("value", "price", "tick_value", "text"):
                if hasattr(tick, key):
                    candidates.append(getattr(tick, key))
            if hasattr(tick, "model_dump"):
                try:
                    dumped = tick.model_dump()
                    if isinstance(dumped, dict):
                        for key in ("value", "price", "tick_value", "text"):
                            if key in dumped:
                                candidates.append(dumped.get(key))
                except Exception:
                    pass
        num = None
        for cand in candidates:
            num = _to_float_maybe(cand)
            if num is not None:
                break
        if num is not None:
            values.append(num)

    finite = [v for v in values if v == v and v not in (float("inf"), float("-inf"))]
    if len(finite) >= 8:
        finite_sorted = sorted(finite)
        cut = int(0.025 * len(finite_sorted))
        if cut > 0 and (2 * cut) < len(finite_sorted):
            finite = finite_sorted[cut:-cut]
    return finite


def run(
    plot_bbox: list[float],
    elements: list[ChartElement],
    config: C4Config,
    current_x_px: float | None = None,
    axis_ticks: list[object] | None = None,
    price_bounds: tuple[float, float] | None = None,
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

    window_px = max(25, int(0.06 * plot_w))
    lines_near_current: list[dict] = []
    lines_left_of_current: list[dict] = []
    lines_right_of_current: list[dict] = []
    used_main_line_bucket = "none"

    if current_x_px is not None:
        win_l = float(current_x_px) - float(window_px)
        win_r = float(current_x_px) + float(window_px)
        for ln in line_infos:
            x_min = float(ln["left"][0])
            x_max = float(ln["right"][0])
            if x_max < win_l:
                lines_left_of_current.append(ln)
            elif x_min > win_r:
                lines_right_of_current.append(ln)
            else:
                lines_near_current.append(ln)
        non_projection_lines = lines_left_of_current + lines_near_current
    else:
        non_projection_lines = list(line_infos)
        lines_left_of_current = list(line_infos)

    bounds_source = "none"
    num_tick_values_used = 0
    price_min: float | None = None
    price_max: float | None = None

    if price_bounds is not None and len(price_bounds) == 2:
        p0, p1 = float(price_bounds[0]), float(price_bounds[1])
        if p0 <= p1:
            price_min, price_max = p0, p1
            bounds_source = "explicit"
        else:
            price_min, price_max = p1, p0
            bounds_source = "explicit"
    else:
        tick_values = _extract_tick_values(axis_ticks)
        num_tick_values_used = len(tick_values)
        if len(tick_values) >= 2:
            price_min = float(min(tick_values))
            price_max = float(max(tick_values))
            bounds_source = "axis_ticks"
        else:
            fallback_values: list[float] = []
            for z in zone_vals:
                fallback_values.extend([float(z["low"]), float(z["high"])])
            for ln in lines_near_current:
                fallback_values.extend([float(ln["left"][2]), float(ln["right"][2])])
            if len(fallback_values) >= 2:
                price_min = float(min(fallback_values))
                price_max = float(max(fallback_values))
                bounds_source = "elements_fallback"

    num_prices_clamped = 0

    def _clamp_price(value: float) -> float:
        nonlocal num_prices_clamped
        if price_min is None or price_max is None:
            return float(value)
        clamped = min(max(float(value), price_min), price_max)
        if abs(clamped - float(value)) > 1e-9:
            num_prices_clamped += 1
        return clamped

    # Current from line-evaluated price at current_x (when provided), else right-most endpoint.
    current_price: float | None = None
    used_line_for_current = "endpoint_fallback"
    if line_infos:
        current_line = None
        if current_x_px is not None and lines_near_current:
            current_line = max(lines_near_current, key=lambda ln: ln["length"])
            used_line_for_current = "near_current"
        elif current_x_px is not None:
            current_line = max(line_infos, key=lambda ln: ln["length"])
            used_line_for_current = "global_longest"

        if current_x_px is not None and current_line is not None:
            left = current_line["left"]
            right = current_line["right"]
            dx = float(right[0] - left[0])
            if abs(dx) > 1e-6:
                t = (float(current_x_px) - float(left[0])) / dx
                t = max(0.0, min(1.0, t))
                current_price = _clamp_price(float(left[2] + t * (right[2] - left[2])))
                conf = 0.65 if used_line_for_current == "near_current" else 0.60
                waypoints.append(ScenarioWaypoint(label="current", price=current_price, conf=conf))
            else:
                used_line_for_current = "endpoint_fallback"
        if current_price is None:
            endpoints = []
            for ln in line_infos:
                endpoints.append(ln["left"])
                endpoints.append(ln["right"])
            current = max(endpoints, key=lambda p: p[0])
            current_price = _clamp_price(float(current[2]))
            waypoints.append(ScenarioWaypoint(label="current", price=current_price, conf=0.55))

    if bounds_source == "none" and current_price is not None:
        fallback_values: list[float] = [float(current_price)]
        for z in zone_vals:
            fallback_values.extend([float(z["low"]), float(z["high"])])
        for ln in lines_near_current:
            fallback_values.extend([float(ln["left"][2]), float(ln["right"][2])])
        if len(fallback_values) >= 2:
            price_min = float(min(fallback_values))
            price_max = float(max(fallback_values))
            bounds_source = "elements_fallback"
            for i, wp in enumerate(waypoints):
                if wp.label == "current":
                    clamped_cur = _clamp_price(float(wp.price))
                    waypoints[i] = ScenarioWaypoint(
                        label=wp.label,
                        price=clamped_cur,
                        time_hint=wp.time_hint,
                        conf=wp.conf,
                    )
                    current_price = clamped_cur
                    break

    # Entry from zones around current.
    entry_price: float | None = None
    invalidation_price: float | None = None

    # Main line and movement type.
    main_line = max(non_projection_lines, key=lambda ln: ln["length"]) if non_projection_lines else None
    informative = False
    bullish = False
    if main_line is not None:
        if current_x_px is not None:
            if main_line in lines_near_current:
                used_main_line_bucket = "near_current"
            elif main_line in lines_left_of_current:
                used_main_line_bucket = "left_of_current"
        informative = (main_line["dx"] >= 0.10 * plot_w) and (main_line["length"] >= 0.15 * plot_diag)
        if informative:
            movement_type = "direct"
            bullish = main_line["dprice"] > 0.0
            sign_main = 1 if main_line["dprice"] > 0 else -1
            for other in non_projection_lines:
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
        zone_points.append(
            ScenarioWaypoint(label="zone", price=_clamp_price(float(z["low"])), time_hint="low", conf=float(z["conf"]))
        )
        zone_points.append(
            ScenarioWaypoint(label="zone", price=_clamp_price(float(z["high"])), time_hint="high", conf=float(z["conf"]))
        )
    zone_points.sort(key=lambda wp: wp.price)

    if informative and current_price is not None:
        line_prices = [_clamp_price(float(ln["left"][2])) for ln in non_projection_lines] + [
            _clamp_price(float(ln["right"][2])) for ln in non_projection_lines
        ]
        near_line_prices = [_clamp_price(float(ln["left"][2])) for ln in lines_near_current] + [
            _clamp_price(float(ln["right"][2])) for ln in lines_near_current
        ]
        zone_highs = [_clamp_price(float(z["high"])) for z in zone_vals]
        zone_lows = [_clamp_price(float(z["low"])) for z in zone_vals]

        if bullish:
            target: float | None = max(near_line_prices) if near_line_prices else None
            if line_prices:
                target = max([target] + line_prices) if target is not None else max(line_prices)
            if zone_highs:
                target = max([target] + zone_highs) if target is not None else max(zone_highs)
            if target is not None:
                if abs(target - current_price) > 1e-6:
                    waypoints.append(ScenarioWaypoint(label="target", price=_clamp_price(float(target)), conf=0.50))
            eligible_entry = [z["high"] for z in zone_vals if z["high"] <= current_price]
            if eligible_entry:
                entry_price = _clamp_price(float(min(eligible_entry, key=lambda p: abs(p - current_price))))
                waypoints.append(ScenarioWaypoint(label="entry", price=entry_price, conf=0.45))
            if entry_price is not None:
                lows_below = [z["low"] for z in zone_vals if z["low"] < entry_price]
                if lows_below:
                    invalidation_price = _clamp_price(float(min(lows_below)))
                    waypoints.append(ScenarioWaypoint(label="invalidation", price=invalidation_price, conf=0.40))
        else:
            target = min(near_line_prices) if near_line_prices else None
            if line_prices:
                target = min([target] + line_prices) if target is not None else min(line_prices)
            if zone_lows:
                target = min([target] + zone_lows) if target is not None else min(zone_lows)
            if target is not None:
                if abs(target - current_price) > 1e-6:
                    waypoints.append(ScenarioWaypoint(label="target", price=_clamp_price(float(target)), conf=0.50))
            eligible_entry = [z["low"] for z in zone_vals if z["low"] >= current_price]
            if eligible_entry:
                entry_price = _clamp_price(float(min(eligible_entry, key=lambda p: abs(p - current_price))))
                waypoints.append(ScenarioWaypoint(label="entry", price=entry_price, conf=0.45))
            if entry_price is not None:
                highs_above = [z["high"] for z in zone_vals if z["high"] > entry_price]
                if highs_above:
                    invalidation_price = _clamp_price(float(max(highs_above)))
                    waypoints.append(ScenarioWaypoint(label="invalidation", price=invalidation_price, conf=0.40))

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
    debug = {
        "window_px": window_px,
        "num_lines_total": len(line_infos),
        "num_lines_near": len(lines_near_current),
        "num_lines_left": len(lines_left_of_current),
        "num_lines_right": len(lines_right_of_current),
        "used_line_for_current": used_line_for_current,
        "used_main_line_bucket": used_main_line_bucket,
        "price_min": price_min,
        "price_max": price_max,
        "num_tick_values_used": num_tick_values_used,
        "bounds_source": bounds_source,
        "num_prices_clamped": num_prices_clamped,
    }
    return StageResult(data=scenario, confidence=conf, abstain=False, reasons=[], debug=debug)
