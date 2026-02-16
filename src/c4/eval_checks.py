"""Utility helpers for Component 4 evaluation checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple


@dataclass
class TickMonotonicityCheck:
    """Result of applying the axis tick monotonicity validation."""

    passed: bool | None
    violations: int
    sorted_ticks: list[tuple[float, float]]
    direction: str | None
    drop_one_repaired: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": self.violations,
            "sorted_ticks": [[y, value] for y, value in self.sorted_ticks],
            "direction": self.direction,
            "drop_one_repaired": self.drop_one_repaired,
        }


VISUAL_ARTIFACT_REQUIREMENTS: tuple[tuple[str, bool], ...] = (
    ("overlay.png", True),
    ("axis_debug.png", True),
    ("axis_crop.png", True),
    ("plot_crop.png", True),
    ("elements_debug.png", False),
)


@dataclass
class VisualArtifactsCheck:
    missing_files: list[str]
    missing_required_files: list[str]


def evaluate_axis_tick_monotonicity(axis_ticks: Iterable[Any]) -> TickMonotonicityCheck:
    """Assess whether a list of axis ticks follows a monotonic trend."""

    sorted_ticks = _normalize_and_sort_ticks(axis_ticks)
    if len(sorted_ticks) < 2:
        return TickMonotonicityCheck(passed=None, violations=0, sorted_ticks=sorted_ticks, direction=None, drop_one_repaired=False)

    values = [value for _, value in sorted_ticks]
    violations, direction, _ = _count_violations(values)
    if violations == 0:
        return TickMonotonicityCheck(passed=True, violations=0, sorted_ticks=sorted_ticks, direction=direction, drop_one_repaired=False)

    drop_one_repaired = False
    best = violations
    if len(sorted_ticks) >= 4:
        for drop_idx in range(len(values)):
            reduced = values[:drop_idx] + values[drop_idx + 1 :]
            if len(reduced) < 2:
                continue
            reduced_violations, _, _ = _count_violations(reduced)
            if reduced_violations < best:
                best = reduced_violations
            if best == 0:
                drop_one_repaired = True
                break

    if drop_one_repaired:
        return TickMonotonicityCheck(passed=True, violations=0, sorted_ticks=sorted_ticks, direction=direction, drop_one_repaired=True)

    return TickMonotonicityCheck(passed=False, violations=best, sorted_ticks=sorted_ticks, direction=direction, drop_one_repaired=False)


def _normalize_and_sort_ticks(axis_ticks: Iterable[Any]) -> list[tuple[float, float]]:
    normalized: list[tuple[float, float]] = []
    for tick in axis_ticks:
        y_px = _extract_numeric(tick, "y_px")
        value = _extract_numeric(tick, "value")
        if y_px is None or value is None:
            continue
        normalized.append((y_px, value))
    return sorted(normalized, key=lambda pair: pair[0])


def _extract_numeric(obj: Any, attr: str) -> float | None:
    raw = obj.get(attr) if isinstance(obj, dict) else getattr(obj, attr, None)
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def _count_violations(values: Sequence[float]) -> Tuple[int, str, float]:
    first, last = values[0], values[-1]
    direction = "decreasing" if last < first else "increasing"
    value_range = max(values) - min(values)
    epsilon = max(1e-9, 0.001 * value_range)
    violations = 0
    for prev, curr in zip(values, values[1:]):
        satisfies_direction = curr >= prev if direction == "increasing" else curr <= prev
        if satisfies_direction or abs(curr - prev) <= epsilon:
            continue
        violations += 1
    return violations, direction, epsilon


def evaluate_visual_artifacts(artifacts_dir: Path) -> VisualArtifactsCheck:
    missing_files: list[str] = []
    missing_required_files: list[str] = []
    seen: set[str] = set()

    def _mark_missing(name: str, required: bool) -> None:
        if name in seen:
            return
        seen.add(name)
        missing_files.append(name)
        if required:
            missing_required_files.append(name)

    for name, required in VISUAL_ARTIFACT_REQUIREMENTS:
        if not (artifacts_dir / name).exists():
            _mark_missing(name, required)

    return VisualArtifactsCheck(missing_files=missing_files, missing_required_files=missing_required_files)
