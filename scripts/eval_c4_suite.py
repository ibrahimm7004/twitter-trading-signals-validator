from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import platform
import random
import shutil
import subprocess
import sys
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from c4.config import load_config  # noqa: E402
from c4.eval_checks import evaluate_axis_tick_monotonicity, evaluate_visual_artifacts  # noqa: E402
from c4.pipeline import run_pipeline  # noqa: E402
from c4.schema import OutputSchema  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
EPS = 1e-6


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    artifact_refs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "artifact_refs": self.artifact_refs,
        }


@dataclass
class EvalContext:
    root: Path
    run_dir: Path
    images_dir: Path
    images_root: Path
    flagged_root: Path
    determinism_root: Path
    thumbs: bool
    copy_original: bool
    quiet: bool

    def rel_from_root(self, path: Path | None) -> str | None:
        if path is None:
            return None
        return str(path.relative_to(self.root)).replace("\\", "/")

    def rel_from_run(self, path: Path | None) -> str | None:
        if path is None:
            return None
        return str(path.relative_to(self.run_dir)).replace("\\", "/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Component 4 on an image suite.")
    parser.add_argument("--images-dir", default="test-images", help="Input images directory. Default: test-images")
    parser.add_argument("--out-root", default="runs_eval", help="Output root for evaluation runs. Default: runs_eval")
    parser.add_argument("--run-name", default=None, help="Run folder name under out-root. Default: auto run_<timestamp>")
    parser.add_argument("--determinism", action="store_true", help="Run deterministic run1/run2 comparisons.")
    parser.add_argument("--determinism-subset", type=int, default=5, help="Number of images for determinism checks.")
    parser.add_argument("--determinism-stems", default=None, help="Optional comma-separated stems for determinism checks.")
    parser.add_argument("--determinism-seed", type=int, default=1337, help="Seed for determinism subset sampling.")
    parser.add_argument("--copy-original", action=argparse.BooleanOptionalAction, default=False, help="Copy source image to artifacts folder.")
    parser.add_argument("--thumbs", action=argparse.BooleanOptionalAction, default=True, help="Generate thumbs in eval/thumbs.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-image pipeline logs (errors still surfaced on exceptions).")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False, help="Overwrite run-name directory if it exists.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for smoke runs.")
    parser.add_argument("--stems", default=None, help="Comma-separated list of image stems to evaluate (subset mode).")
    parser.add_argument("--source-run", default=None, help="Run directory (or name) containing flagged/by_issue_code.json.")
    parser.add_argument("--issue-code", default=None, help="Issue code to target when selecting stems from a previous run.")
    parser.add_argument("--extra-stems", default=None, help="Additional comma-separated stems to include alongside the subset.")
    parser.add_argument("--dry-run", action="store_true", help="Show selection summary and exit without running evaluation.")
    return parser.parse_args()


_STEM_PATTERN = re.compile(r"(chart\\d+)", re.IGNORECASE)


def _parse_comma_separated(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _resolve_source_run_dir(value: str | None) -> Path | None:
    if not value:
        return None
    candidates = [Path(value)]
    if not Path(value).is_absolute():
        candidates.extend([ROOT / value, ROOT / "runs_eval" / value])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _extract_stems_from_entry(entry: Any) -> set[str]:
    stems: set[str] = set()
    if isinstance(entry, str):
        stems.update(match.lower() for match in _STEM_PATTERN.findall(entry))
        return stems
    if isinstance(entry, dict):
        for sub in entry.values():
            stems.update(_extract_stems_from_entry(sub))
        return stems
    if isinstance(entry, list):
        for item in entry:
            stems.update(_extract_stems_from_entry(item))
        return stems
    return stems


def _load_issue_code_stems(source_run_dir: Path, issue_code: str) -> set[str]:
    flagged_path = source_run_dir / "flagged" / "by_issue_code.json"
    if not flagged_path.exists():
        return set()
    try:
        data = json.loads(flagged_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    entry = data.get(issue_code)
    if entry is None:
        return set()
    stems = _extract_stems_from_entry(entry)
    if isinstance(entry, dict):
        images = entry.get("images")
        if isinstance(images, list):
            stems.update(str(item).strip().lower() for item in images if isinstance(item, str) and item.strip())
    return stems


def _resolve_images_from_stems(images_dir: Path, stems: set[str]) -> list[Path]:
    selected: list[Path] = []
    for stem in sorted(stems):
        matches = [
            p
            for p in images_dir.glob(f"{stem}.*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]
        matches_sorted = sorted(matches, key=lambda p: p.name.lower())
        if not matches_sorted:
            raise FileNotFoundError(f"No image files found for stem '{stem}' in {images_dir}")
        if len(matches_sorted) > 1:
            print(f"[subset] multiple candidates for {stem}; using {matches_sorted[0].name}")
        selected.append(matches_sorted[0])
    return selected


def _select_images(args: argparse.Namespace, images_dir: Path) -> tuple[list[Path], set[str]]:
    stems: set[str] = set()
    if args.issue_code:
        if not args.source_run:
            raise ValueError("--issue-code requires --source-run")
        source_run_dir = _resolve_source_run_dir(args.source_run)
        if source_run_dir is None:
            raise ValueError(f"Source run directory not found: {args.source_run}")
        issue_stems = _load_issue_code_stems(source_run_dir, args.issue_code)
        if not issue_stems:
            raise ValueError(f"Issue code '{args.issue_code}' had no stems in {source_run_dir}")
        stems.update(issue_stems)
    stems.update(_parse_comma_separated(args.stems))
    stems.update(_parse_comma_separated(args.extra_stems))
    if stems:
        images = _resolve_images_from_stems(images_dir, stems)
        if args.max_images is not None:
            images = images[: max(0, args.max_images)]
    else:
        images = discover_images(images_dir=images_dir, max_images=args.max_images)
    return images, stems


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _normalize_output_payload_for_hash(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload))
    debug_artifacts = normalized.get("debug_artifacts")
    if isinstance(debug_artifacts, dict):
        for key in list(debug_artifacts.keys()):
            if key.endswith("_path"):
                debug_artifacts.pop(key, None)
    return normalized


def normalized_output_sha256(payload: dict[str, Any]) -> str:
    normalized = _normalize_output_payload_for_hash(payload)
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _run_pipeline_with_optional_quiet(image_path: Path, cfg: Any, artifacts_dir: Path, quiet: bool):
    if not quiet:
        return run_pipeline(image_path=image_path, config=cfg, debug_dir=artifacts_dir)
    log_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(log_buffer):
            return run_pipeline(image_path=image_path, config=cfg, debug_dir=artifacts_dir)
    except Exception:
        captured = log_buffer.getvalue().strip()
        if captured:
            print(captured, file=sys.stderr)
        raise


def discover_images(images_dir: Path, max_images: int | None = None) -> list[Path]:
    images = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    images_sorted = sorted(images, key=lambda p: str(p.relative_to(images_dir)).lower())
    if max_images is not None:
        return images_sorted[: max(0, max_images)]
    return images_sorted


def bbox_in_bounds(bbox: list[float] | None, w: int, h: int, tol: float = 2.0) -> bool | None:
    if bbox is None or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return (-tol <= x0 <= w - 1 + tol) and (-tol <= x1 <= w - 1 + tol) and (-tol <= y0 <= h - 1 + tol) and (
        -tol <= y1 <= h - 1 + tol
    )



def in_bounds(v: float | None, lo: float | None, hi: float | None) -> bool | None:
    if v is None or lo is None or hi is None:
        return None
    return lo - EPS <= float(v) <= hi + EPS


def load_image(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path))


def write_thumb_if_present(src: Path, dst: Path, width: int = 700) -> None:
    if not src.exists():
        return
    img = load_image(src)
    if img is None:
        return
    h, w = img.shape[:2]
    if w > width:
        scale = width / float(w)
        img = cv2.resize(img, (width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img)


def artifact_hashes(artifacts_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not artifacts_dir.exists():
        return rows
    for p in sorted([x for x in artifacts_dir.rglob("*") if x.is_file()], key=lambda x: str(x).lower()):
        rows.append(
            {
                "rel_path": str(p.relative_to(artifacts_dir)).replace("\\", "/"),
                "sha256": sha256_file(p),
                "size": p.stat().st_size,
            }
        )
    return rows


def detect_magenta_current_x(elements_debug_path: Path) -> bool | None:
    if not elements_debug_path.exists():
        return None
    img = load_image(elements_debug_path)
    if img is None:
        return None
    b = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)
    mask = (b > 180) & (r > 180) & (g < 100) & (np.abs(b - r) < 40)
    return int(mask.sum()) > max(50, img.shape[0] // 2)


def detect_possible_giant_zone(elements_debug_path: Path) -> bool | None:
    if not elements_debug_path.exists():
        return None
    img = load_image(elements_debug_path)
    if img is None:
        return None
    b = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)
    yellow = (g > 160) & (r > 160) & (b < 120)
    ys, xs = np.where(yellow)
    if len(xs) < 200:
        return False
    x_span = xs.max() - xs.min() + 1
    y_span = ys.max() - ys.min() + 1
    h, w = img.shape[:2]
    return (x_span >= int(0.90 * w)) and (y_span >= int(0.90 * h))


def status_from_issues(issues: list[Issue]) -> str:
    if any(i.severity == "error" for i in issues):
        return "fail"
    if any(i.severity == "warn" for i in issues):
        return "warn"
    return "pass"


def compare_hash_sets(run1_dir: Path, run2_dir: Path) -> list[dict[str, Any]]:
    h1 = {row["rel_path"]: row for row in artifact_hashes(run1_dir)}
    h2 = {row["rel_path"]: row for row in artifact_hashes(run2_dir)}
    rels = sorted(set(h1.keys()) | set(h2.keys()))
    diffs: list[dict[str, Any]] = []
    for rel in rels:
        a = h1.get(rel)
        b = h2.get(rel)
        if a is None or b is None:
            diffs.append({"rel_path": rel, "type": "missing", "run1": a, "run2": b})
            continue
        if a["sha256"] != b["sha256"] or a["size"] != b["size"]:
            diffs.append({"rel_path": rel, "type": "content", "run1": a, "run2": b})
    return diffs


def unique_image_stems(images: list[Path]) -> dict[Path, str]:
    counts: dict[str, int] = {}
    mapping: dict[Path, str] = {}
    for path in images:
        stem = path.stem
        idx = counts.get(stem, 0)
        counts[stem] = idx + 1
        mapping[path] = stem if idx == 0 else f"{stem}_{idx+1}"
    return mapping


def evaluate_one(image_path: Path, image_id: str, cfg: Any, ctx: EvalContext) -> dict[str, Any]:
    out_img_root = ctx.images_root / image_id
    artifacts_dir = out_img_root / "artifacts"
    eval_dir = out_img_root / "eval"
    thumbs_dir = eval_dir / "thumbs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    if ctx.copy_original:
        shutil.copy2(image_path, artifacts_dir / image_path.name)

    output = _run_pipeline_with_optional_quiet(image_path=image_path, cfg=cfg, artifacts_dir=artifacts_dir, quiet=ctx.quiet)
    output_path = artifacts_dir / "output.json"
    output_payload = output.model_dump()
    stable_write_json(output_path, output_payload)
    output_json_normalized_sha256 = normalized_output_sha256(output_payload)

    img = load_image(image_path)
    if img is None:
        raise RuntimeError(f"Unable to read image: {image_path}")
    height, width = img.shape[:2]

    schema_valid = True
    try:
        validated = OutputSchema.model_validate(output.model_dump())
    except Exception:
        schema_valid = False
        validated = output

    axis_values = [float(t.value) for t in validated.axis_ticks]
    price_min = min(axis_values) if axis_values else None
    price_max = max(axis_values) if axis_values else None
    quality_flags_payload = validated.quality_flags.model_dump() if validated.quality_flags is not None else None
    likely_faulty_input = bool(validated.quality_flags.likely_faulty_input) if validated.quality_flags is not None else False
    quality_flag_reasons = list(validated.quality_flags.reasons) if validated.quality_flags is not None else []
    tick_monotonicity = evaluate_axis_tick_monotonicity(validated.axis_ticks)
    visual_artifacts = evaluate_visual_artifacts(artifacts_dir)
    visual_missing_files = list(visual_artifacts.missing_files)
    visual_missing_required = list(visual_artifacts.missing_required_files)

    zones_count = sum(1 for e in validated.elements if e.kind == "zone")
    lines_count = sum(1 for e in validated.elements if e.kind == "line")

    signal = validated.signal
    scenario = validated.scenario
    target0 = signal.targets[0] if signal.targets else None

    checks = {
        "schema_valid": schema_valid,
        "plot_bbox_in_bounds": bbox_in_bounds(validated.chart_frame.plot_bbox, width, height),
        "axis_bbox_in_bounds": bbox_in_bounds(validated.chart_frame.axis_bbox, width, height),
        "ticks_monotonic": tick_monotonicity.passed,
        "ticks_monotonic_details": tick_monotonicity.to_dict(),
        "likely_faulty_input": likely_faulty_input,
        "quality_flag_reasons": quality_flag_reasons,
        "quality_flags": quality_flags_payload,
        "vis_artifacts_missing_files": visual_missing_files,
        "scenario_waypoints_in_bounds": (
            all(in_bounds(w.price, price_min, price_max) is not False for w in scenario.waypoints)
            if (price_min is not None and price_max is not None)
            else None
        ),
        "signal_prices_in_bounds": (
            all(
                x is not False
                for x in [in_bounds(signal.entry, price_min, price_max), in_bounds(signal.stop, price_min, price_max)]
                + [in_bounds(t, price_min, price_max) for t in signal.targets]
            )
            if (price_min is not None and price_max is not None)
            else None
        ),
        "signal_vs_scenario_consistent": (
            True
            if scenario.movement_type == "unknown"
            else any(r == f"movement:{scenario.movement_type}" for r in signal.rationale)
        ),
    }

    metrics = {
        "axis_ticks_count": len(validated.axis_ticks),
        "price_min": price_min,
        "price_max": price_max,
        "elements_count": len(validated.elements),
        "zones_count": zones_count,
        "lines_count": lines_count,
        "scenario_movement_type": scenario.movement_type,
        "scenario_confidence": float(scenario.confidence),
        "signal_direction": signal.direction,
        "signal_entry": signal.entry,
        "signal_stop": signal.stop,
        "signal_targets_count": len(signal.targets),
        "signal_confidence": float(signal.confidence),
        "target_equals_min_bound": (
            (target0 is not None and price_min is not None and abs(float(target0) - float(price_min)) <= EPS)
            if (price_min is not None and target0 is not None)
            else None
        ),
        "target_equals_max_bound": (
            (target0 is not None and price_max is not None and abs(float(target0) - float(price_max)) <= EPS)
            if (price_max is not None and target0 is not None)
            else None
        ),
    }

    rel_out = ctx.rel_from_root(output_path)
    rel_elements_debug = ctx.rel_from_root(artifacts_dir / "elements_debug.png") if (artifacts_dir / "elements_debug.png").exists() else None
    rel_plot_crop = ctx.rel_from_root(artifacts_dir / "plot_crop.png") if (artifacts_dir / "plot_crop.png").exists() else None

    issues: list[Issue] = []
    if not schema_valid:
        issues.append(Issue("error", "SCHEMA_INVALID", "OutputSchema validation failed.", [rel_out or ""]))
    if metrics["axis_ticks_count"] == 0:
        issues.append(Issue("error", "NO_AXIS_TICKS", "No axis ticks extracted.", [rel_out or "", rel_elements_debug or rel_out or ""]))
    elif metrics["axis_ticks_count"] < 4:
        issues.append(Issue("warn", "NO_AXIS_TICKS", "Axis tick count is below 4.", [rel_out or ""]))
    if likely_faulty_input:
        reasons_joined = ", ".join(quality_flag_reasons) if quality_flag_reasons else "unspecified"
        issues.append(
            Issue(
                "info",
                "INPUT_GEOMETRY_SUSPECT",
                f"Input geometry likely faulty/non-calibratable ({reasons_joined}).",
                [rel_out or ""],
            )
        )
    if checks["plot_bbox_in_bounds"] is False or checks["axis_bbox_in_bounds"] is False:
        issues.append(Issue("error", "BBOX_OUT_OF_BOUNDS", "Frame bboxes are out of image bounds.", [rel_out or ""]))
    if tick_monotonicity.passed is False:
        sorted_pairs = ", ".join(f"({y:.1f},{value:.3f})" for y, value in tick_monotonicity.sorted_ticks)
        message = (
            f"Axis tick values are not monotonic (violations={tick_monotonicity.violations}). "
            f"Sorted ticks (y_px,value): [{sorted_pairs}]"
        )
        issues.append(Issue("warn", "TICKS_NOT_MONOTONIC", message, [rel_out or ""]))
    if checks["scenario_waypoints_in_bounds"] is False:
        issues.append(Issue("error", "WAYPOINT_OUT_OF_BOUNDS", "Scenario waypoint price outside tick bounds.", [rel_out or ""]))
    if checks["signal_prices_in_bounds"] is False:
        issues.append(Issue("error", "SIGNAL_PRICE_OUT_OF_BOUNDS", "Signal price outside tick bounds.", [rel_out or ""]))
    if checks["signal_vs_scenario_consistent"] is False:
        issues.append(Issue("warn", "SIGNAL_SCENARIO_MISMATCH", "Signal rationale does not mention scenario movement type.", [rel_out or ""]))
    if metrics["target_equals_min_bound"] is True:
        issues.append(Issue("warn", "TARGET_AT_MIN_BOUND", "First target equals minimum tick bound.", [rel_out or ""]))
    if metrics["target_equals_max_bound"] is True:
        issues.append(Issue("warn", "TARGET_AT_MAX_BOUND", "First target equals maximum tick bound.", [rel_out or ""]))
    if metrics["lines_count"] > 60:
        issues.append(Issue("warn", "TOO_MANY_LINES", "Line count above 60.", [rel_elements_debug or rel_out or ""]))
    if metrics["zones_count"] > 10:
        issues.append(Issue("warn", "TOO_MANY_ZONES", "Zone count above 10.", [rel_elements_debug or rel_out or ""]))

    elements_debug_path = artifacts_dir / "elements_debug.png"
    elements_debug_present = elements_debug_path.exists()
    has_magenta = detect_magenta_current_x(elements_debug_path) if elements_debug_present else None
    if elements_debug_present and has_magenta is None and "elements_debug.png" not in visual_missing_files:
        visual_missing_files.append("elements_debug.png")
    if has_magenta is False:
        issues.append(Issue("warn", "VIS_NO_CURRENT_X", "Magenta current_x marker not found in elements_debug.", [rel_elements_debug or rel_out or ""]))
    if visual_missing_required:
        message = (
            "Required visual artifacts missing: "
            + ", ".join(visual_missing_required)
            + "."
        )
        issues.append(Issue("info", "VIS_ARTIFACT_MISSING", message, [rel_out or ""]))

    possible_giant_zone = detect_possible_giant_zone(artifacts_dir / "elements_debug.png")
    if possible_giant_zone is True:
        issues.append(
            Issue(
                "warn",
                "VIS_POSSIBLE_GIANT_ZONE",
                "Possible giant zone rectangle spans most of the image.",
                [rel_elements_debug or rel_out or ""],
            )
        )

    status = status_from_issues(issues)

    if ctx.thumbs:
        write_thumb_if_present(artifacts_dir / "elements_debug.png", thumbs_dir / "elements_debug_thumb.jpg")
        write_thumb_if_present(artifacts_dir / "overlay.png", thumbs_dir / "overlay_thumb.jpg")

    hashes = artifact_hashes(artifacts_dir)
    stable_write_json(eval_dir / "hashes.json", hashes)

    image_info = {
        "path": str(image_path),
        "stem": image_id,
        "sha256": sha256_file(image_path),
        "output_json_normalized_sha256": output_json_normalized_sha256,
        "width": width,
        "height": height,
    }
    paths = {"output_json": rel_out, "elements_debug": rel_elements_debug, "plot_crop": rel_plot_crop}
    report = {
        "image": image_info,
        "status": status,
        "metrics": metrics,
        "checks": checks,
        "issues": [i.to_dict() for i in issues],
        "paths": paths,
    }
    stable_write_json(eval_dir / "report.json", report)

    md_lines = [
        f"# {image_id}",
        f"- Status: **{status}**",
        f"- Metrics: ticks={metrics['axis_ticks_count']}, elements={metrics['elements_count']} (zones={zones_count}, lines={lines_count}), scenario={metrics['scenario_movement_type']} ({metrics['scenario_confidence']:.3f})",
        f"- Signal: direction={metrics['signal_direction']}, entry={metrics['signal_entry']}, stop={metrics['signal_stop']}, targets={signal.targets}, conf={metrics['signal_confidence']:.3f}",
        "- Issues:",
    ]
    if issues:
        for i in issues:
            md_lines.append(f"  - [{i.severity}] `{i.code}`: {i.message}")
    else:
        md_lines.append("  - none")
    md_lines.append("- Artifacts:")
    md_lines.append(f"  - output: `{paths['output_json']}`")
    if paths["elements_debug"]:
        md_lines.append(f"  - elements_debug: `{paths['elements_debug']}`")
    if paths["plot_crop"]:
        md_lines.append(f"  - plot_crop: `{paths['plot_crop']}`")
    (eval_dir / "report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return report


def _timestamp_run_name() -> str:
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def resolve_run_dir(out_root: Path, run_name: str | None, overwrite: bool) -> Path:
    base_name = run_name or _timestamp_run_name()
    run_dir = out_root / base_name
    if overwrite:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        return run_dir
    if not run_dir.exists():
        return run_dir
    suffix = 2
    while True:
        candidate = out_root / f"{base_name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def update_latest_pointer(out_root: Path, run_dir: Path) -> str:
    out_root.mkdir(parents=True, exist_ok=True)
    latest_txt = out_root / "latest.txt"
    latest_txt.write_text(run_dir.name + "\n", encoding="utf-8")

    latest_link = out_root / "latest"
    if latest_link.exists():
        try:
            if latest_link.is_file():
                latest_link.unlink()
            else:
                shutil.rmtree(latest_link)
        except OSError:
            subprocess.run(["cmd", "/c", "rmdir", str(latest_link)], capture_output=True, text=True, check=False)
            if latest_link.exists():
                latest_link.unlink(missing_ok=True)
    try:
        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(latest_link), str(run_dir)],
            cwd=str(out_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return f"latest.txt updated ({latest_txt}); junction skipped: {proc.stderr.strip() or proc.stdout.strip()}"
        return f"latest junction created ({latest_link} -> {run_dir.name})"
    except Exception as exc:
        return f"latest.txt updated ({latest_txt}); junction skipped: {exc}"


def write_consolidated_reports(
    reports: list[dict[str, Any]],
    images: list[Path],
    cfg: Any,
    ctx: EvalContext,
    schema_or_bbox_fail_stems: list[str],
) -> tuple[Counter[str], Counter[str]]:
    status_counts = Counter(r["status"] for r in reports)
    issue_counter: Counter[str] = Counter()
    flagged: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "severity_distribution": Counter(), "images": [], "artifact_paths": []})

    for r in reports:
        stem = r["image"]["stem"]
        for i in r["issues"]:
            code = i["code"]
            issue_counter[code] += 1
            flagged[code]["count"] += 1
            flagged[code]["severity_distribution"][i["severity"]] += 1
            flagged[code]["images"].append(stem)
            flagged[code]["artifact_paths"].extend(i.get("artifact_refs", []))

    suite_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "num_images": len(images),
        "git_commit": None,
        "config": cfg.model_dump() if hasattr(cfg, "model_dump") else None,
        "run_dir": ctx.rel_from_root(ctx.run_dir),
        "images_dir": str(ctx.images_dir),
    }
    try:
        suite_meta["git_commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ctx.root, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        )
    except Exception:
        suite_meta["git_commit"] = None

    suite_summary = {
        "totals": {"images": len(images), "pass": status_counts["pass"], "warn": status_counts["warn"], "fail": status_counts["fail"]},
        "issues": dict(issue_counter),
        "images": reports,
    }
    stable_write_json(ctx.run_dir / "suite_meta.json", suite_meta)
    stable_write_json(ctx.run_dir / "suite_summary.json", suite_summary)

    csv_fields = [
        "stem",
        "status",
        "axis_ticks_count",
        "elements_count",
        "zones_count",
        "lines_count",
        "scenario_movement_type",
        "scenario_confidence",
        "signal_direction",
        "signal_confidence",
        "issue_codes",
    ]
    with (ctx.run_dir / "suite_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in reports:
            issue_codes = ",".join(sorted({i["code"] for i in r["issues"]}))
            writer.writerow(
                {
                    "stem": r["image"]["stem"],
                    "status": r["status"],
                    "axis_ticks_count": r["metrics"]["axis_ticks_count"],
                    "elements_count": r["metrics"]["elements_count"],
                    "zones_count": r["metrics"]["zones_count"],
                    "lines_count": r["metrics"]["lines_count"],
                    "scenario_movement_type": r["metrics"]["scenario_movement_type"],
                    "scenario_confidence": r["metrics"]["scenario_confidence"],
                    "signal_direction": r["metrics"]["signal_direction"],
                    "signal_confidence": r["metrics"]["signal_confidence"],
                    "issue_codes": issue_codes,
                }
            )

    top_issues = issue_counter.most_common(10)
    outlier_lines = [r for r in reports if r["metrics"]["lines_count"] > 50]
    red_flags = [r for r in reports if r["status"] == "fail" or r["metrics"]["lines_count"] > 60 or r["metrics"]["zones_count"] > 10]

    summary_md = [
        "# Component 4 Evaluation Suite",
        f"- Run dir: `{ctx.rel_from_root(ctx.run_dir)}`",
        f"- Images: {len(images)}",
        f"- Pass/Warn/Fail: {status_counts['pass']}/{status_counts['warn']}/{status_counts['fail']}",
        "## Top issue codes",
    ]
    if top_issues:
        for code, cnt in top_issues:
            summary_md.append(f"- `{code}`: {cnt}")
    else:
        summary_md.append("- none")
    summary_md.append("## Outlier metrics")
    summary_md.append(f"- lines_count > 50: {len(outlier_lines)} images")
    summary_md.append("## Red flag images")
    if red_flags:
        for r in red_flags:
            summary_md.append(f"- `{r['image']['stem']}` -> `{r['paths']['output_json']}`")
    else:
        summary_md.append("- none")
    (ctx.run_dir / "suite_summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")

    flagged_json = {}
    flagged_md = ["# Issues grouped by code"]
    for code in sorted(flagged.keys()):
        entry = flagged[code]
        sev_dist = dict(entry["severity_distribution"])
        imgs = sorted(set(entry["images"]))
        art_refs = sorted(set(entry["artifact_paths"]))
        flagged_json[code] = {
            "count": entry["count"],
            "severity_distribution": sev_dist,
            "images": imgs,
            "artifact_paths": art_refs,
        }
        flagged_md.append(f"## {code}")
        flagged_md.append(f"- count: {entry['count']}")
        flagged_md.append(f"- severity: {sev_dist}")
        if imgs:
            flagged_md.append("- images:")
            for s in imgs:
                out_ref = ctx.rel_from_root(ctx.images_root / s / "artifacts" / "output.json")
                dbg_ref = ctx.rel_from_root(ctx.images_root / s / "artifacts" / "elements_debug.png")
                flagged_md.append(f"  - `{s}`: `{out_ref}`, `{dbg_ref}`")
        if art_refs:
            flagged_md.append("- artifact refs:")
            for ar in art_refs[:40]:
                flagged_md.append(f"  - `{ar}`")
    stable_write_json(ctx.flagged_root / "by_issue_code.json", flagged_json)
    (ctx.flagged_root / "by_issue_code.md").write_text("\n".join(flagged_md) + "\n", encoding="utf-8")

    if schema_or_bbox_fail_stems:
        print("SCHEMA_INVALID/BBOX_OUT_OF_BOUNDS affected stems:")
        for stem in sorted(set(schema_or_bbox_fail_stems)):
            print(f"- {stem}")

    return status_counts, issue_counter


def run_determinism(
    images: list[Path],
    cfg: Any,
    ctx: EvalContext,
    subset_size: int,
    seed: int,
    quiet: bool,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if not images or subset_size <= 0:
        stable_write_json(ctx.determinism_root / "subset_list.json", [])
        stable_write_json(ctx.determinism_root / "comparisons.json", [])
        (ctx.determinism_root / "comparisons.md").write_text("# Determinism comparisons\n\n- no subset configured\n", encoding="utf-8")
        return []

    k = min(subset_size, len(images))
    sampled = sorted(rng.sample(images, k), key=lambda p: p.name.lower())
    stable_write_json(ctx.determinism_root / "subset_list.json", [p.name for p in sampled])

    comparisons: list[dict[str, Any]] = []
    for img in sampled:
        stem = img.stem
        run1 = ctx.determinism_root / stem / "run1"
        run2 = ctx.determinism_root / stem / "run2"
        run1.mkdir(parents=True, exist_ok=True)
        run2.mkdir(parents=True, exist_ok=True)

        out1 = _run_pipeline_with_optional_quiet(image_path=img, cfg=cfg, artifacts_dir=run1, quiet=quiet)
        out2 = _run_pipeline_with_optional_quiet(image_path=img, cfg=cfg, artifacts_dir=run2, quiet=quiet)
        stable_write_json(run1 / "output.json", out1.model_dump())
        stable_write_json(run2 / "output.json", out2.model_dump())

        diffs = compare_hash_sets(run1, run2)
        comparisons.append({"image": img.name, "identical": len(diffs) == 0, "differences": diffs})

    stable_write_json(ctx.determinism_root / "comparisons.json", comparisons)
    det_md = ["# Determinism comparisons"]
    for comp in comparisons:
        det_md.append(f"## {comp['image']}")
        det_md.append(f"- identical: {comp['identical']}")
        if comp["differences"]:
            for diff in comp["differences"]:
                det_md.append(f"  - `{diff['rel_path']}` type={diff['type']}")
                det_md.append(f"    - run1={diff['run1']}")
                det_md.append(f"    - run2={diff['run2']}")
    (ctx.determinism_root / "comparisons.md").write_text("\n".join(det_md) + "\n", encoding="utf-8")
    return comparisons


def main() -> int:
    args = parse_args()
    images_dir = (ROOT / args.images_dir).resolve()
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return 2

    try:
        images, selected_stems = _select_images(args, images_dir)
    except ValueError as exc:
        print(exc)
        return 2

    print(f"Images directory: {images_dir}")
    print(f"Selected {len(images)} image(s) for evaluation.")
    if selected_stems:
        stems_sorted = sorted(selected_stems)
        preview = ", ".join(stems_sorted[:10])
        if len(stems_sorted) > 10:
            preview += ", ..."
        print(f"Target stems ({len(selected_stems)}): {preview}")
    else:
        print("Target stems: (all images)")

    if not images:
        print(f"No images found in {images_dir}")
        return 2

    if args.determinism_stems and not args.determinism:
        print("--determinism-stems requires --determinism.")
        return 2

    if args.dry_run:
        print("Dry-run mode enabled; matched files:")
        for path in images:
            print(f"- {path}")
        return 0

    out_root = (ROOT / args.out_root).resolve()
    run_dir = resolve_run_dir(out_root=out_root, run_name=args.run_name, overwrite=args.overwrite)
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = EvalContext(
        root=ROOT,
        run_dir=run_dir,
        images_dir=images_dir,
        images_root=run_dir / "images",
        flagged_root=run_dir / "flagged",
        determinism_root=run_dir / "determinism",
        thumbs=args.thumbs,
        copy_original=args.copy_original,
        quiet=args.quiet,
    )
    ctx.images_root.mkdir(parents=True, exist_ok=True)
    ctx.flagged_root.mkdir(parents=True, exist_ok=True)
    if args.determinism:
        ctx.determinism_root.mkdir(parents=True, exist_ok=True)

    cfg = load_config(ROOT / "configs" / "default.yaml", {})
    id_map = unique_image_stems(images)

    reports: list[dict[str, Any]] = []
    schema_or_bbox_fail_stems: list[str] = []
    eval_started = time.perf_counter()
    for idx, image_path in enumerate(images, start=1):
        image_id = id_map[image_path]
        rel_img = image_path.relative_to(images_dir)
        print(f"[{idx}/{len(images)}] {rel_img}")
        image_started = time.perf_counter()
        report = evaluate_one(image_path=image_path, image_id=image_id, cfg=cfg, ctx=ctx)
        elapsed = time.perf_counter() - image_started
        reports.append(report)
        print(f"  elapsed={elapsed:.2f}s status={report['status']}")
        issue_codes = {i["code"] for i in report["issues"]}
        if "SCHEMA_INVALID" in issue_codes or "BBOX_OUT_OF_BOUNDS" in issue_codes:
            schema_or_bbox_fail_stems.append(report["image"]["stem"])
    eval_elapsed = time.perf_counter() - eval_started

    status_counts, issue_counter = write_consolidated_reports(
        reports=reports,
        images=images,
        cfg=cfg,
        ctx=ctx,
        schema_or_bbox_fail_stems=schema_or_bbox_fail_stems,
    )
    comparisons: list[dict[str, Any]] = []
    determinism_elapsed = 0.0
    if args.determinism:
        determinism_images = images
        det_stems = set(_parse_comma_separated(args.determinism_stems))
        if det_stems:
            deterministic_selected = [img for img in images if img.stem.lower() in det_stems]
            if not deterministic_selected:
                print("--determinism-stems did not match selected evaluation images.")
                return 2
            determinism_images = deterministic_selected
        det_started = time.perf_counter()
        comparisons = run_determinism(
            images=determinism_images,
            cfg=cfg,
            ctx=ctx,
            subset_size=args.determinism_subset,
            seed=args.determinism_seed,
            quiet=args.quiet,
        )
        determinism_elapsed = time.perf_counter() - det_started

    latest_note = update_latest_pointer(out_root=out_root, run_dir=run_dir)
    latest_txt = out_root / "latest.txt"
    latest_value = latest_txt.read_text(encoding="utf-8").strip() if latest_txt.exists() else "(missing)"

    top_issues = issue_counter.most_common(10)
    hash_rows = [
        {
            "stem": r["image"]["stem"],
            "output_json_normalized_sha256": r["image"]["output_json_normalized_sha256"],
            "output_json": r["paths"]["output_json"],
        }
        for r in sorted(reports, key=lambda item: item["image"]["stem"])
    ]
    stable_write_json(ctx.run_dir / "output_hashes_normalized.json", hash_rows)

    print("")
    print(f"Total images: {len(images)} | pass={status_counts['pass']} warn={status_counts['warn']} fail={status_counts['fail']}")
    print(f"Elapsed (main pass): {eval_elapsed:.2f}s")
    if args.determinism:
        non_identical = sum(1 for c in comparisons if not c["identical"])
        print(f"Determinism non-identical: {non_identical}/{len(comparisons)}")
        print(f"Elapsed (determinism): {determinism_elapsed:.2f}s")
    print("Top 10 issue codes:")
    for code, cnt in top_issues:
        print(f"- {code}: {cnt}")
    print("Normalized output.json SHA256 (debug_artifacts paths removed):")
    for row in hash_rows:
        print(f"- {row['stem']}: {row['output_json_normalized_sha256']}")
    print("Key reports:")
    print(ctx.rel_from_root(ctx.run_dir / "suite_summary.md"))
    print(ctx.rel_from_root(ctx.run_dir / "flagged" / "by_issue_code.md"))
    if args.determinism:
        print(ctx.rel_from_root(ctx.run_dir / "determinism" / "comparisons.md"))
    print(f"latest pointer: {ctx.rel_from_root(latest_txt)} -> {latest_value}")
    print(f"latest note: {latest_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
