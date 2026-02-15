# Component 4 (C4)

Local-first, open-source extraction pipeline scaffold for TradingView-like chart screenshots.

## What is implemented (M1)

- Repo scaffold and CLI skeleton
- JSON output schema (Pydantic models)
- YAML config loader with CLI overrides
- Abstain plumbing and structured reasons
- Debug artifact placeholder writer

## Quickstart

```bash
python -m pip install -e .
pytest -q
c4 extract --help
```

## Example run

```bash
c4 extract --image tests/assets/synth.png --out_json out.json --debug_dir debug --config configs/default.yaml
```

Even when stages abstain (M1 stubs), a valid JSON output is still emitted.
