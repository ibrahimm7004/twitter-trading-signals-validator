from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from c4.config import load_config  # noqa: E402
from c4.pipeline import run_pipeline  # noqa: E402


def _find_images(image: Path | None, directory: Path | None, pattern: str) -> list[Path]:
    if image is not None:
        return [image.resolve()]
    if directory is None:
        return []
    found = sorted({p.resolve() for p in directory.rglob(pattern) if p.is_file()})
    if found:
        return found
    # Useful fallback when users pass --dir runs but images are at repo root.
    fallback = sorted({p.resolve() for p in ROOT.glob(pattern) if p.is_file()})
    return fallback


def _safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Component-4 pipeline on single image or directory.")
    parser.add_argument("--image", type=Path, default=None, help="Single input image path.")
    parser.add_argument("--dir", type=Path, default=None, help="Directory to scan for images.")
    parser.add_argument("--glob", type=str, default="*.jpeg", help="Glob pattern for --dir mode.")
    parser.add_argument("--debug-dir", type=Path, default=Path("runs/_debug"), help="Debug artifacts output root.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/_out"), help="JSON output root.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Config YAML path.")
    args = parser.parse_args()

    if (args.image is None and args.dir is None) or (args.image is not None and args.dir is not None):
        print("Use exactly one of --image or --dir.")
        return 2

    images = _find_images(args.image, args.dir, args.glob)
    if not images:
        print("No images found.")
        return 2

    cfg = load_config(args.config, {})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.debug_dir.mkdir(parents=True, exist_ok=True)

    for img in images:
        stem = _safe_stem(img)
        out_dir = args.out_dir / stem
        dbg_dir = args.debug_dir / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        dbg_dir.mkdir(parents=True, exist_ok=True)

        output = run_pipeline(image_path=img, config=cfg, debug_dir=dbg_dir)
        out_json = out_dir / "output.json"
        out_json.write_text(json.dumps(output.model_dump(), indent=2, sort_keys=True), encoding="utf-8")

        signal = output.signal
        print(
            f"{img.name}: direction={signal.direction} entry={signal.entry} "
            f"stop={signal.stop} targets={signal.targets} conf={signal.confidence:.4f} abstain={output.abstain}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
