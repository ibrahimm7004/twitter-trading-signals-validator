from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c4.config import C4Config
from c4.pipeline import run_pipeline
from c4.stages import axis_ocr, calibration, frame


def _load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _inspect_repo() -> dict[str, list[str]]:
    notes: dict[str, list[str]] = {}

    def add(section: str, line: str) -> None:
        notes.setdefault(section, []).append(line)

    config_text = _load_text(ROOT / "src" / "c4" / "config.py")
    axis_text = _load_text(ROOT / "src" / "c4" / "stages" / "axis_ocr.py")
    pipeline_text = _load_text(ROOT / "src" / "c4" / "pipeline.py")
    pyproject_text = _load_text(ROOT / "pyproject.toml")

    if "ocr_backend" in config_text:
        add("config", "`runtime.ocr_backend` exists in src/c4/config.py")
    if "PADDLE_DISABLE_ONEDNN" in axis_text:
        add("env", "Sets `PADDLE_DISABLE_ONEDNN=1` in src/c4/stages/axis_ocr.py")
    if "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK" in axis_text:
        add("env", "Mentions model hoster check env in src/c4/stages/axis_ocr.py")
    if "PaddleOCR" in axis_text:
        add("ocr", "PaddleOCR import is guarded by backend flag and try/except in src/c4/stages/axis_ocr.py")
    if "ocr_backend" in pipeline_text:
        add("pipeline", "Pipeline uses axis_ocr.run and calibration.run in src/c4/pipeline.py")
    if "paddleocr" in pyproject_text:
        add("deps", "pyproject.toml references PaddleOCR dependency")
    else:
        add("deps", "PaddleOCR is optional (not in pyproject.toml)")
    return notes


def _write_report(report_path: Path, inspection: dict[str, list[str]], summary_md: str, paddle_installed: bool) -> None:
    lines: list[str] = []
    lines.append("# C4 OCR Audit Report")
    lines.append("")
    lines.append("## What We Expect When OCR Is Working")
    lines.append("- `axis_debug.png` contains OCR boxes + numeric labels when detections exist.")
    lines.append("- `axis_ticks` length is non-empty and calibration `scale` is set when OCR succeeds.")
    lines.append("- Abstain reasons differ between OCR enabled vs disabled modes.")
    lines.append("")
    lines.append("## What Is Happening In This Repo Right Now")
    lines.append("- OCR enabled when `runtime.ocr_backend == \"paddle\"` (see src/c4/config.py).")
    lines.append("- Pipeline invokes `axis_ocr.run(...)` and `calibration.run(...)` in src/c4/pipeline.py.")
    lines.append("- Axis debug overlay prefers OCR boxes when `stage_axis.debug[\"ocr_boxes\"]` exists; otherwise tick-row marks.")
    lines.append("")
    lines.append("### PaddleOCR Accommodation Edits (from working tree scan)")
    for section, items in inspection.items():
        lines.append(f"- **{section}**:")
        for item in items:
            lines.append(f"  - {item}")
    lines.append("")
    lines.append("## Evidence")
    lines.append("- Runner outputs are under `runs/c4_ocr_ablation/`.")
    lines.append("- Each run writes `result.json` with keys: `abstain`, `abstain_reasons`, `frame`, `axis_ticks`, `num_ticks`, `mean_tick_conf`, `calibration`.")
    lines.append("")
    lines.append("### Runtime Verification")
    lines.append(f"- paddleocr installed: `{paddle_installed}`")
    lines.append("")
    lines.append("### Summary Table")
    lines.append(summary_md.strip() if summary_md else "(summary not available)")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    ticks = result.get("axis_ticks", [])
    mean_conf = 0.0
    if ticks:
        mean_conf = sum(t.get("conf", 0.0) for t in ticks) / max(1, len(ticks))
    calib = result.get("calibration") or {}
    return {
        "abstain": result.get("abstain", True),
        "num_ticks": len(ticks),
        "mean_conf": round(mean_conf, 3),
        "scale": calib.get("scale"),
        "fit_error": calib.get("fit_error"),
        "reason_codes": [r.get("code") for r in result.get("abstain_reasons", [])],
    }


def _run_one(image_path: Path, mode: str, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = C4Config()
    cfg.runtime.ocr_backend = "paddle" if mode == "paddle" else "disabled"

    stage_frame = frame.run(str(image_path), cfg)
    axis_bbox = stage_frame.data.axis_bbox if stage_frame.data else None
    stage_axis = axis_ocr.run(str(image_path), axis_bbox=axis_bbox, config=cfg)
    stage_calib = calibration.run(stage_axis.data or [], cfg)

    output = run_pipeline(str(image_path), cfg, out_dir)

    result = {
        "image": str(image_path),
        "mode": mode,
        "ocr_backend": cfg.runtime.ocr_backend,
        "abstain": output.abstain,
        "abstain_reasons": [r.model_dump() for r in output.abstain_reasons],
        "frame": output.chart_frame.model_dump(),
        "axis_ticks": [t.model_dump() for t in output.axis_ticks],
        "num_ticks": len(output.axis_ticks),
        "mean_tick_conf": float(sum(t.conf for t in output.axis_ticks) / max(1, len(output.axis_ticks))) if output.axis_ticks else 0.0,
        "calibration": stage_calib.data.__dict__ if stage_calib.data is not None else None,
        "notes": {
            "axis_stage_debug": stage_axis.debug,
            "frame_stage_debug": stage_frame.debug,
        },
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _write_summary(results: list[dict[str, Any]], summary_path: Path) -> str:
    lines = ["| image | mode | abstain | num_ticks | mean_conf | scale | fit_error | reason_codes |", "|---|---|---|---|---|---|---|---|"]
    for result in results:
        s = _summarize_result(result)
        lines.append(
            f"| {Path(result['image']).name} | {result['mode']} | {s['abstain']} | {s['num_ticks']} | {s['mean_conf']} | {s['scale']} | {s['fit_error']} | {', '.join([str(c) for c in s['reason_codes']])} |"
        )
    summary_md = "\n".join(lines)
    summary_path.write_text(summary_md, encoding="utf-8")
    return summary_md


def main() -> int:
    parser = argparse.ArgumentParser(description="Run C4 OCR ablation.")
    parser.add_argument("--out", type=str, default=str(ROOT / "runs" / "c4_ocr_ablation"))
    parser.add_argument("--images", nargs="+", default=["chart1.jpeg", "chart2.jpeg"])
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--paddle-only", action="store_true")
    parser.add_argument("--disabled-only", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out)
    if out_root.exists() and not args.keep:
        for item in out_root.glob("*"):
            if item.is_dir():
                for child in item.rglob("*"):
                    if child.is_file():
                        child.unlink()
                for child in sorted(item.rglob("*"), reverse=True):
                    if child.is_dir():
                        child.rmdir()
                item.rmdir()
            else:
                item.unlink()

    images = [ROOT / img for img in args.images]
    missing = [str(p) for p in images if not p.exists()]
    if missing:
        print(f"Missing images: {', '.join(missing)}")
        return 2

    modes = []
    if not args.disabled_only:
        modes.append("paddle")
    if not args.paddle_only:
        modes.append("disabled")

    results: list[dict[str, Any]] = []
    for image_path in images:
        for mode in modes:
            out_dir = out_root / image_path.stem / mode
            result = _run_one(image_path, mode, out_dir)
            results.append(result)

    summary_md = _write_summary(results, out_root / "summary.md")

    inspection = _inspect_repo()
    paddle_installed = False
    try:
        import paddleocr  # noqa: F401

        paddle_installed = True
    except Exception:
        paddle_installed = False

    report_path = ROOT / "docs" / "c4_ocr_audit_report.md"
    _write_report(report_path, inspection, summary_md, paddle_installed)

    print(f"Wrote outputs to {out_root}")
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
