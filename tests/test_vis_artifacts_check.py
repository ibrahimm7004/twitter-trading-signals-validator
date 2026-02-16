"""Tests for the visual artifact expectation helper used by the eval suite."""

from __future__ import annotations

from pathlib import Path

from c4.eval_checks import VISUAL_ARTIFACT_REQUIREMENTS, evaluate_visual_artifacts


def _touch(path: Path) -> None:
    path.write_bytes(b"")


def test_vis_artifacts_missing_optional_check(tmp_path: Path) -> None:
    for name, required in VISUAL_ARTIFACT_REQUIREMENTS:
        if required:
            _touch(tmp_path / name)

    result = evaluate_visual_artifacts(tmp_path)
    assert result.missing_required_files == []
    assert result.missing_files == ["elements_debug.png"]


def test_vis_artifacts_missing_required_files(tmp_path: Path) -> None:
    for name, required in VISUAL_ARTIFACT_REQUIREMENTS:
        if required and name != "axis_debug.png":
            _touch(tmp_path / name)

    result = evaluate_visual_artifacts(tmp_path)
    assert result.missing_files[0] == "axis_debug.png"
    assert result.missing_required_files == ["axis_debug.png"]
