# Component 4 (C4)

Local-first, open-source extraction pipeline for TradingView-like chart screenshots.

## What is implemented

- Frame/axis localization, axis OCR (optional Paddle), calibration
- Elements detection, candle alignment, scenario reconstruction
- Final signal build (`direction`, `entry`, `stop`, `targets`, `confidence`)
- JSON output schema (Pydantic), YAML config, abstain plumbing
- Debug artifacts (`overlay.png`, `axis_debug.png`, `plot_crop.png`, `axis_crop.png`, `elements_debug.png`)

## Quickstart

```bash
python -m pip install -e .
pytest -q
c4 extract --help
```

## Single image run

```bash
c4 extract --image tests/assets/synth.png --out_json out.json --debug_dir debug --config configs/default.yaml
```

## Batch runner
Use the helper script to run one image or many images and write deterministic outputs:

```bash
python scripts/run_c4.py --image chart1.jpeg --debug-dir runs/_debug --out-dir runs/_out
python scripts/run_c4.py --dir runs --glob "*.jpeg" --debug-dir runs/_debug --out-dir runs/_out
```

Per-image outputs:
- `runs/_out/<image_stem>/output.json`
- `runs/_debug/<image_stem>/...` debug PNG artifacts

## Evaluation suite rerun (Windows-friendly)
Run the formal suite over `test-images` with one command. Each execution creates a fresh timestamped folder:

```bash
python scripts/eval_c4_suite.py
```

Outputs are written under:
- `runs_eval/run_<YYYYMMDD_HHMMSS>/suite_summary.{md,json,csv}`
- `runs_eval/run_<...>/flagged/by_issue_code.{md,json}`
- `runs_eval/run_<...>/determinism/comparisons.{md,json}`
- `runs_eval/run_<...>/images/<stem>/artifacts` and `eval`

Latest run pointer:
- `runs_eval/latest.txt`

Custom run name:

```bash
python scripts/eval_c4_suite.py --run-name run_after_determinism_fix
```

## Output schema (top-level)
- `chart_frame`
- `axis_ticks`
- `elements`
- `scenario`
- `signal`
- `abstain`
- `abstain_reasons`
- `debug_artifacts`
