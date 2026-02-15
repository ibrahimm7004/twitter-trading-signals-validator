# Component 4 Decisions (Non negotiables)

This repository implements Component 4 of the pipeline. These decisions are binding for all code and tests.

## 1. Local first and open source only

- Runtime must be local first: no paid APIs, no external services required.
- Use open source components only (Python first).
- Allowed core libs: OpenCV, PaddleOCR, SAM 2; optional GroundingDINO; optional local VLM (e.g., Qwen2.5-VL).

## 2. Deterministic backbone for numeric prices

- Numeric prices must be computed deterministically from geometry:
  1. localize chart frame and right axis
  2. OCR tick values on right axis only
  3. fit robust pixel to price mapping (linear or log)
  4. project detected element pixels onto the axis to get prices
- Never infer prices from freeform text descriptions.
- Never fabricate prices when OCR is weak or missing.

## 3. VLM usage restrictions

- VLMs are optional and must be local.
- VLMs are forbidden from producing numeric prices.
- VLMs may only assist with non numeric labeling and disambiguation after:
  - chart frame is detected
  - axis mapping is calibrated
  - element primitives are detected and priced by projection
- If a VLM output conflicts with geometry derived outputs, geometry wins.

## 4. Reliability and abstain policy

- The system must never guess silently.
- Every stage must return:
  - structured outputs
  - confidence score
  - debug info
  - structured abstain reasons when applicable
- If any critical upstream stage fails, the pipeline must abstain:
  - frame or axis not found with sufficient confidence
  - fewer than minimum required OCR ticks
  - calibration fit error too high or ambiguous (linear vs log)
  - element detection is too incomplete to price reliably
  - scenario reconstruction fails coherence checks
- Abstain must set:
  - `abstain=true`
  - `abstain_reasons=[...]` with stable reason codes and human readable messages
  - a valid JSON output must still be produced

## 5. Separation of stages (architecture contract)

Stages must be clearly separated and composable:

1. Frame and axis localization
2. Axis OCR
3. Calibration (pixel to price mapping)
4. Element detection (CV baseline)
5. Element pricing (axis projection)
6. Segmentation upgrade (SAM 2; optional GroundingDINO)
7. Scenario reconstruction (rules first; optional VLM tags only)
8. Optional metadata OCR (ticker/timeframe/indicators)
9. Evaluation harness

Each stage has a stable function boundary and a StageResult type.

## 6. Debug artifacts required

- CLI must accept `--debug_dir`.
- Pipeline must emit overlays and intermediate crops:
  - frame overlay, plot crop, axis crop
  - OCR boxes overlay on axis crop
  - calibration plot or diagnostics (at least JSON stats)
  - element overlays and priced annotations
- Debug outputs must be deterministic and reproducible.

## 7. Determinism and reproducibility

- No randomness unless explicitly seeded and documented.
- Config driven thresholds and toggles.
- Tests should generate synthetic charts; avoid committing binary assets.

## 8. Output schema stability

- Output must conform to the JSON schema defined in the plan.
- Additive schema changes require updating:
  - schema models
  - example JSON
  - evaluation harness
  - tests
