"""Command-line interface for Component 4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .config import load_config
from .pipeline import run_pipeline
from .schema import OutputSchema

app = typer.Typer(add_completion=False)


@app.callback()
def main() -> None:
    """Component 4 CLI."""


def _build_overrides(
    min_ticks: Optional[int],
    frame_min_conf: Optional[float],
    axis_min_conf: Optional[float],
    calibration_max_error: Optional[float],
    element_min_conf: Optional[float],
) -> dict:
    overrides: dict = {"thresholds": {}}
    if min_ticks is not None:
        overrides["thresholds"]["min_ticks"] = min_ticks
    if frame_min_conf is not None:
        overrides["thresholds"]["frame_min_conf"] = frame_min_conf
    if axis_min_conf is not None:
        overrides["thresholds"]["axis_min_conf"] = axis_min_conf
    if calibration_max_error is not None:
        overrides["thresholds"]["calibration_max_error"] = calibration_max_error
    if element_min_conf is not None:
        overrides["thresholds"]["element_min_conf"] = element_min_conf
    if not overrides["thresholds"]:
        overrides = {}
    return overrides


@app.command()
def extract(
    image: Path = typer.Option(..., "--image", help="Path to input image."),
    out_json: Path = typer.Option(..., "--out_json", help="Path to output JSON."),
    debug_dir: Optional[Path] = typer.Option(None, "--debug_dir", help="Debug artifacts directory."),
    config: Path = typer.Option(
        "configs/default.yaml",
        "--config",
        help="Path to config YAML.",
        show_default=True,
    ),
    min_ticks: Optional[int] = typer.Option(None, "--min_ticks", help="Minimum OCR ticks."),
    frame_min_conf: Optional[float] = typer.Option(None, "--frame_min_conf", help="Min frame conf."),
    axis_min_conf: Optional[float] = typer.Option(None, "--axis_min_conf", help="Min axis conf."),
    calibration_max_error: Optional[float] = typer.Option(None, "--calibration_max_error", help="Max calibration error."),
    element_min_conf: Optional[float] = typer.Option(None, "--element_min_conf", help="Min element conf."),
) -> None:
    """Extract Component 4 structured output from a chart image."""
    overrides = _build_overrides(
        min_ticks=min_ticks,
        frame_min_conf=frame_min_conf,
        axis_min_conf=axis_min_conf,
        calibration_max_error=calibration_max_error,
        element_min_conf=element_min_conf,
    )
    cfg = load_config(config, overrides)
    output: OutputSchema = run_pipeline(image, cfg, debug_dir)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = output.model_dump()
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    app()
