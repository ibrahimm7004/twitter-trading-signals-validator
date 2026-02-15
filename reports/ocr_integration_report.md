# OCR Integration Report (Component 4)

## Where OCR Is Called
- `src/c4/pipeline.py`: `run_pipeline()` invokes `frame.run()`, then `axis_ocr.run(...)`, then `calibration.run(...)`.
- `src/c4/stages/axis_ocr.py`: `run()` performs OCR on the axis crop when `runtime.ocr_backend == "paddle"`.

## Backend Gating
- `src/c4/config.py`: `RuntimeConfig.ocr_backend` controls OCR backend (`"disabled"` or `"paddle"`).
- `configs/default.yaml`: default `ocr_backend` value.
- `src/c4/stages/axis_ocr.py`: early return + abstain when backend is not `"paddle"`.

## PaddleOCR Windows Support Edits (Working Tree)
- `src/c4/stages/axis_ocr.py`
  - Conditional import of `paddleocr.PaddleOCR` inside `run()` to keep Paddle optional.
  - Environment stability flags set before import:
    - `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK`
    - `PADDLE_DISABLE_ONEDNN`
    - `FLAGS_use_mkldnn`
    - `FLAGS_enable_mkldnn`
    - `FLAGS_enable_onednn`
    - `FLAGS_enable_pir_api`
  - Runtime logging of backend/method/metrics (backend selected, import success, OCR method, counts, avg_conf).
  - OCR call path compatibility across versions (`predict` preferred, fall back to `ocr`).
- `src/c4/pipeline.py`
  - Axis debug rendering uses OCR boxes (yellow boxes + labels) when available; falls back to tick rows otherwise.
- `src/c4/config.py`
  - `runtime.ocr_backend` config for enabling/disabling OCR.
- `configs/default.yaml`
  - default `ocr_backend` value for local runs.

## Harness Outputs
The comparison harness writes to:
```
runs/<chart_name>/<mode>/
```
Each run folder includes:
- `output.json`
- `overlay.png`
- `plot_crop.png`
- `axis_crop.png`
- `axis_debug.png`
- `run.log`

## How To Run
```
python scripts/run_ocr_comparison.py
```

## Common Failure Modes (Windows)
- PaddleOCR import fails (missing wheel or incompatible build). See `run.log` for import status.
- oneDNN / PIR runtime errors. The environment flags above are set to avoid these.
- Model download failures or network issues. `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` skips host checks.

