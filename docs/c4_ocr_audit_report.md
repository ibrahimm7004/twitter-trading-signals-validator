# C4 OCR Audit Report

## What We Expect When OCR Is Working
- `axis_debug.png` contains OCR boxes + numeric labels when detections exist.
- `axis_ticks` length is non-empty and calibration `scale` is set when OCR succeeds.
- Abstain reasons differ between OCR enabled vs disabled modes.

## What Is Happening In This Repo Right Now
- OCR enabled when `runtime.ocr_backend == "paddle"` (see src/c4/config.py).
- Pipeline invokes `axis_ocr.run(...)` and `calibration.run(...)` in src/c4/pipeline.py.
- Axis debug overlay prefers OCR boxes when `stage_axis.debug["ocr_boxes"]` exists; otherwise tick-row marks.

### PaddleOCR Accommodation Edits (from working tree scan)
- **config**:
  - `runtime.ocr_backend` exists in src/c4/config.py
- **env**:
  - Sets `PADDLE_DISABLE_ONEDNN=1` in src/c4/stages/axis_ocr.py
  - Mentions model hoster check env in src/c4/stages/axis_ocr.py
- **ocr**:
  - PaddleOCR import is guarded by backend flag and try/except in src/c4/stages/axis_ocr.py
- **deps**:
  - PaddleOCR is optional (not in pyproject.toml)

## Evidence
- Runner outputs are under `runs/c4_ocr_ablation/`.
- Each run writes `result.json` with keys: `abstain`, `abstain_reasons`, `frame`, `axis_ticks`, `num_ticks`, `mean_tick_conf`, `calibration`.

### Runtime Verification
- paddleocr installed: `True`

### Summary Table
| image | mode | abstain | num_ticks | mean_conf | scale | fit_error | reason_codes |
|---|---|---|---|---|---|---|---|
| chart1.jpeg | paddle | False | 15 | 1.0 | linear | 0.0017793594306050138 |  |
| chart1.jpeg | disabled | True | 0 | 0.0 | None | None | ReasonCode.OCR_NOT_IMPLEMENTED, ReasonCode.CALIBRATION_FAILED |
| chart2.jpeg | paddle | False | 12 | 1.0 | linear | 0.0021584589402795923 |  |
| chart2.jpeg | disabled | True | 0 | 0.0 | None | None | ReasonCode.OCR_NOT_IMPLEMENTED, ReasonCode.CALIBRATION_FAILED |
