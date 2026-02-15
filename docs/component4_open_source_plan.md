# Component 4 extraction pipeline (open-source, local-first)

## Goal

Given a TradingView-like chart screenshot with user-added drawings and annotations, extract:

1. **Element-level prices**: start/end prices for arrows and lines; upper/lower bounds for zones; line price at a specific candle time when required.
2. **Scenario path**: reconstruct the expected price dynamics (direct vs conditional; sweep, rejection, break-and-continue, pullback-then-continue), plus an ordered list of waypoints.

## Scope constraints (MVP contract)

- Inputs are **TradingView-like** chart screenshots.
- **Price axis is on the right**; the pipeline may abstain if not found or unreadable.
- Text (ticker, timeframe, indicator names) is **secondary**; do not block pricing extraction on it.
- Reliability standard: **never guess silently**; return `abstain` with diagnostics when confidence is low.

---

## Outputs (define this first)

Create a single JSON output per image:

```json
{
  "chart_frame": {
    "plot_bbox": [x0, y0, x1, y1],
    "axis_bbox": [x0, y0, x1, y1],
    "scale": "linear|log",
    "confidence": 0.0
  },
  "axis_ticks": [
    { "value": 0.0, "y_px": 0, "conf": 0.0 }
  ],
  "elements": [
    {
      "id": "e1",
      "type": "arrow|box|trendline|channel|fib|handdrawn_path",
      "bbox": [x0, y0, x1, y1],
      "keypoints_px": { "start": [x, y], "end": [x, y] },
      "prices": { "start": 0.0, "end": 0.0, "low": 0.0, "high": 0.0 },
      "confidence": 0.0,
      "notes": ""
    }
  ],
  "scenario": {
    "movement_type": "direct|conditional|unknown",
    "patterns": ["sweep", "rejection", "break_continue", "pullback_continue"],
    "waypoints": [
      { "label": "current|entry|target|invalidation|zone", "price": 0.0, "time_hint": "optional", "conf": 0.0 }
    ],
    "confidence": 0.0
  },
  "abstain": false,
  "abstain_reasons": [],
  "debug_artifacts": {
    "overlay_path": "path/to/overlay.png",
    "axis_debug_path": "path/to/axis_debug.png"
  }
}
```

---

## Step-by-step build plan

### Step 1; Dataset and eval harness

- Collect 500 to 2,000 screenshots from your target distribution (Twitter/X posts, TradingView screenshots).
- Label minimal ground truth:
  - plot bbox; axis bbox
  - 8 to 15 right-axis tick values + y positions
  - element masks or keypoints (arrows, boxes, trendlines, hand-drawn paths)
  - scenario labels for a subset (direct/conditional; sweep/pullback/etc.)
- Build a local evaluation script that outputs:
  - per-element price error (ticks or percent)
  - scenario classification accuracy
  - abstain rate + abstain precision

Deliverable: `datasets/`, `labels/`, `eval.py`, and a small “gold” set (50 images) for quick regression tests.

### Step 2; Chart frame + axis localization (deterministic)

Objective: find the **plot area** and **right price axis region** robustly.

- Use OpenCV preprocessing: grayscale, denoise, adaptive threshold, morphological ops.
- Detect chart panel boundaries via contours and layout heuristics.
- Detect axis region on the right: look for repeated horizontal tick marks + numeric text density.

Deliverable: `locate_frame(image) -> (plot_bbox, axis_bbox, conf)` plus a debug overlay image.

### Step 3; Axis-only OCR and pixel→price calibration (deterministic math on top)

Objective: build a reliable mapping `price = f(y_px)`.

- Run OCR **only** inside `axis_bbox`:
  - multiple passes: scale up 2x/3x, invert, sharpen, binarize
  - parse floats; keep (value, y_px, conf)
- Fit a robust mapping:
  - linear scale: `price = a*y + b` using RANSAC / robust regression
  - log scale: fit in log-space; detect scale by tick spacing/value ratios
- Compute mapping uncertainty; if too few good ticks or fit error high, **abstain**.

Deliverable: `calibrate_axis(axis_crop) -> (mapping_fn, ticks, conf)`.

### Step 4; Baseline element detection with OpenCV (aligns with PDF strategy)

Objective: get a quick, debuggable baseline before deep segmentation.

- Use **Canny** edge detection to reveal strokes and candle edges.
- Use **Hough lines** / probabilistic Hough to extract straight segments.
- Use contours to extract rectangles/boxes.
- Heuristics for arrows: shaft + arrowhead pattern; fallback to “line with endpoint blob”.

Deliverable: `detect_elements_cv(image, plot_bbox) -> elements[]` with masks/bboxes/keypoints.

### Step 5; Upgrade element localization with open segmentation (reduce brittleness)

Objective: reliably isolate arrows, zones, and hand-drawn paths in cluttered charts.

- Use **SAM 2** for promptable segmentation on candidate regions.
- Use **GroundingDINO** (optional) to propose boxes for “arrow”, “rectangle zone”, “hand-drawn curve”; then feed to SAM 2 for masks.
- Post-process masks into primitives:
  - box: top/bottom y bounds
  - arrow: start/end keypoints
  - trendline/channel: line fit endpoints
  - hand-drawn path: skeletonize to polyline

Deliverable: `detect_elements_seg(image) -> elements[]` and a merged detector that prefers segmentation when available.

### Step 6; Price attribution for every element (core requirement)

Objective: compute prices by **projection onto the axis** and interpolation from OCR anchors.

- For each element primitive:
  - project key y-pixels vertically onto the axis (90° projection)
  - apply `price = f(y_px)`
- Zone bounds:
  - `low = f(y_bottom)`; `high = f(y_top)`
- Trendline at candle time:
  - map a candle’s x-position (nearest candle center) to y on the line equation; then `price = f(y)`

Deliverable: `assign_prices(elements, mapping_fn) -> priced_elements[]` plus per-element confidence.

### Step 7; Scenario reconstruction (rules first; VLM only as helper)

Objective: reconstruct the expected path and classify dynamics.

- Rules-based first:
  - if hand-drawn path exists; treat it as primary waypoint polyline
  - derive direction changes; infer targets/invalidation via proximity to zones/levels
  - classify direct vs conditional using ordering (e.g., lower low then up = conditional)
  - detect sweep/deviation patterns by path crossing then reversing around a level/zone
- Optional open VLM assistance:
  - use an open VLM (e.g., Qwen2.5-VL) strictly for **labeling and disambiguation**
  - do not allow it to produce numeric prices; it can only refer to existing elements/waypoints
- Coherence gate:
  - if scenario tags conflict with geometry or element ordering, downgrade confidence or abstain

Deliverable: `build_scenario(priced_elements) -> scenario`.

### Step 8; Secondary text extraction (only after pricing is solid)

Objective: extract ticker/timeframe/indicator names for metadata.

- Run OCR on known UI regions (top-left ticker, timeframe chips, indicator list).
- If UI varies, use VLM to propose ROIs, then OCR those ROIs.

Deliverable: `extract_metadata(image) -> {ticker,timeframe,indicators}`.

### Step 9; Reliability layer; confidence gating + abstain reasons

Objective: make the system “reliable” by refusing when the evidence is insufficient.

Abstain triggers (minimum set):

- axis bbox not found; or axis OCR produces < N valid ticks
- calibration fit error above threshold; or linear vs log ambiguous
- element segmentation overlap too high to separate; or primitive extraction fails
- scenario conflicts with extracted waypoints (incoherence rule)

Deliverable: global `confidence` score, per-stage confidences, and structured abstain reasons.

---

## Practical starting sequence (what to do first)

1. Implement **Step 2 + Step 3**; axis detection + axis-only OCR calibration.
2. Implement **Step 4 + Step 6** for boxes and straight lines; you should get zone bounds and line endpoint prices early.
3. Add **Step 5** (SAM 2; optional GroundingDINO) to handle hand-drawn paths and clutter.
4. Add **Step 7** scenario reconstruction with hard coherence gates.
5. Add **Step 8** metadata extraction last.

---

## Suggested open-source components (local-friendly)

- OpenCV; Canny + Hough lines for baseline geometry.
- PaddleOCR; right-axis tick OCR.
- SAM 2; segmentation of drawings and shapes.
- GroundingDINO; optional open-vocabulary detection to propose ROIs for SAM 2.
- Qwen2.5-VL; optional semantic helper for labeling scenario tags (not for numbers).

---

## References (implementation docs)

- OpenCV Canny edge detection: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
- OpenCV Hough line transform: https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- SAM 2: https://github.com/facebookresearch/sam2
- GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
- Qwen2.5-VL report: https://arxiv.org/abs/2502.13923
- DePlot (plot-to-table principle): https://arxiv.org/abs/2212.10505
