# TRACE v1.8 — Clean Prototype Package

This archive contains only the runnable TRACE prototype and the scientific material required to reproduce its key evidence. Submission-only files such as PPT/PDF decks, scorecards, judge reviews, SIH checklists and takeover notes are intentionally excluded.

## Folder layout

- `alert_api/` — FastAPI alert service, dashboard, tests, frozen BULBUL evidence/checkpoints and runtime data.
- `evidence_registry/` — runtime evidence used by the API, including sequential FANI tracking records and hazard-event targets.
- `validation_protocols/` — frozen protocols for NEPS-G, heat-wave and same-event Stage-2 validation.
- `scientific_reproduction/` — Stage-1 training/evaluation code, frozen samples/checkpoints, location-feature ablation artifacts, and Stage-2 HRRR controlled reconstruction code/data.
- `run_trace_demo.py` — one-command software/API replay.
- `validate_provider_input.py` — local provider-input preflight utility.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run the prototype verification

```bash
python verify_prototype.py
python run_trace_demo.py
```

Expected final line:

```text
TRACE_FINAL_DEMO_PASS_V1_8
```

## Launch the API/dashboard

```bash
cd alert_api
uvicorn trace_alert_api:app --host 0.0.0.0 --port 8000
```

Open `/` for the dashboard or `/api/v1/docs` for the API documentation.

## Reproduce key scientific evidence

```bash
cd scientific_reproduction
python reproduce_science.py
```

For the matched explicit-location ablation:

```bash
cd scientific_reproduction/stage1
python run_matched_location_ablation_seed1.py
```

The matched seed-1 no-message diagnostic compares the same model/split with and without explicit `sin_lat`, `cos_lat`, `sin_lon`, `cos_lon` input channels.

## Important claim boundaries

- GEFSv12 is a prototype substitute; direct NEPS-G skill is not bundled.
- BULBUL +72/+96/+120 are lead-sensitivity forecasts of one verification time, not three independent events.
- FANI temporal centroids are extreme-wind footprint centroids, not exact cyclone centers.
- HRRR Stage 2 is a controlled 12→5 km reconstruction benchmark, not operational India/NEPS-G forecast verification.
- The dashboard's 5 km ring is only a UI locator marker, not a predicted impact radius.
- LOW/MODERATE/SEVERE are prototype categories, not official warnings or calibrated probabilities.
- Human review is required before any public/official warning action.
