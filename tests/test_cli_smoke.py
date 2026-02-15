from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image
from typer.testing import CliRunner

from c4.cli import app
from c4.schema import OutputSchema


def test_cli_extract_smoke(tmp_path: Path):
    image_path = tmp_path / "synth.png"
    Image.new("RGB", (128, 128), color=(10, 10, 10)).save(image_path)

    out_json = tmp_path / "out.json"
    debug_dir = tmp_path / "debug"

    runner = CliRunner()
    config_path = ROOT / "configs" / "default.yaml"
    result = runner.invoke(
        app,
        [
            "extract",
            "--image",
            str(image_path),
            "--out_json",
            str(out_json),
            "--debug_dir",
            str(debug_dir),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    assert out_json.exists()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    output = OutputSchema.model_validate(data)
    assert output.abstain is True
    assert len(output.abstain_reasons) > 0
    assert output.debug_artifacts.overlay_path is not None


def test_cli_help_shows_config_and_debug_dir():
    runner = CliRunner()
    result = runner.invoke(app, ["extract", "--help"])
    assert result.exit_code == 0
    output = result.stdout
    assert "--config" in output
    assert "configs/default.yaml" in output
    assert "--debug_dir" in output
