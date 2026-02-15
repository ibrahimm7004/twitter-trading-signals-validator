from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rename image files to chart<number>.<ext>, where <number> is "
            "extracted from the original filename."
        )
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing JPEG/PNG files to rename.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform renaming. Without this flag, the script runs as dry-run.",
    )
    return parser.parse_args()


def extract_number(filename_stem: str) -> str | None:
    match = re.search(r"(\d+)", filename_stem)
    return match.group(1) if match else None


def main() -> int:
    args = parse_args()
    folder = args.folder

    if not folder.exists() or not folder.is_dir():
        print(f"Error: '{folder}' is not a valid directory.")
        return 1

    # On Windows, globbing can be case-insensitive, so deduplicate paths.
    seen_paths: set[str] = set()
    image_files: list[Path] = []
    for pattern in ("*.jpeg", "*.jpg", "*.JPEG", "*.JPG", "*.png", "*.PNG"):
        for path in folder.glob(pattern):
            key = str(path.resolve()).lower()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            image_files.append(path)
    image_files.sort()

    if not image_files:
        print("No JPEG/PNG files found.")
        return 0

    planned: list[tuple[Path, Path]] = []
    used_numbers: dict[str, Path] = {}

    for src in image_files:
        number = extract_number(src.stem)
        if number is None:
            print(f"Skipping '{src.name}': no number found in filename.")
            continue

        if number in used_numbers:
            print(
                f"Skipping '{src.name}': duplicate number '{number}' also found in "
                f"'{used_numbers[number].name}'."
            )
            continue

        dst = src.with_name(f"chart{number}{src.suffix.lower()}")
        used_numbers[number] = src
        planned.append((src, dst))

    if not planned:
        print("No files eligible for renaming.")
        return 0

    print("Planned renames:")
    for src, dst in planned:
        print(f"  {src.name} -> {dst.name}")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to rename files.")
        return 0

    renamed_count = 0
    for src, dst in planned:
        if src == dst:
            continue
        if dst.exists():
            print(f"Skipping '{src.name}': target '{dst.name}' already exists.")
            continue
        src.rename(dst)
        renamed_count += 1

    print(f"\nDone. Renamed {renamed_count} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
