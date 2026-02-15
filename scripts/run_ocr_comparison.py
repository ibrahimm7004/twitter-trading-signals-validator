from __future__ import annotations

import argparse
import json
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from c4.config import C4Config  # noqa: E402
from c4.pipeline import run_pipeline  # noqa: E402


def _run_one(image_path: Path, mode: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    output_path = out_dir / "output.json"

    cfg = C4Config()
    cfg.runtime.ocr_backend = "paddle" if mode == "paddle" else "disabled"

    with log_path.open("w", encoding="utf-8") as log_fp:
        with redirect_stdout(log_fp):
            try:
                output = run_pipeline(image_path=image_path, config=cfg, debug_dir=out_dir)
                with output_path.open("w", encoding="utf-8") as fp:
                    json.dump(output.model_dump(), fp, indent=2)
            except Exception as exc:
                print(f"[run_ocr_comparison] failed: {exc!r}")
                with output_path.open("w", encoding="utf-8") as fp:
                    json.dump({"error": str(exc), "image": str(image_path), "mode": mode}, fp, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OCR disabled vs paddle comparison.")
    parser.add_argument("--out", default="runs", help="Output root directory.")
    args = parser.parse_args()

    images = [ROOT / "chart1.jpeg", ROOT / "chart2.jpeg"]
    missing = [p for p in images if not p.exists()]
    if missing:
        print("Missing required images:")
        for p in missing:
            print(f"  {p}")
        return 2

    out_root = Path(args.out)
    for image_path in images:
        for mode in ("disabled", "paddle"):
            run_dir = out_root / image_path.stem / mode
            if run_dir.exists():
                shutil.rmtree(run_dir)
            _run_one(image_path, mode, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

