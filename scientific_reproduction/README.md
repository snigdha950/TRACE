# TRACE v1.8 scientific reproduction

The v1.8 release restores the scientific code/data that were omitted from the compact alert-only package.

## One-command replay

```bash
cd scientific_reproduction
python reproduce_science.py
```

This command currently verifies three things:

1. **Stage 1 frozen BULBUL inference**: recomputes the five-seed × three-lead GNN headline, reproducing mean Climate99 IoU **0.4297994244** and AP **0.6069505477**. All three lead forecasts verify the same 2019-11-09 00Z state, so this is lead sensitivity, not three independent weather cases.
2. **Location-feature sensitivity**: replaces explicit sin/cos latitude/longitude inputs with their training means at inference. Mean IoU changes from **0.4298 to 0.3599**. This shows geography contributes but the frozen model does not collapse. It is not a retrained no-location ablation.
3. **Stage 2 v2 train/eval replay**: reruns the original leakage-safe HRRR controlled 12→5 km trainer/evaluator from the bundled NPZ. The original scientific script is unchanged. A tiny wrapper only makes optional `eccodes` package-version logging nonfatal because raw GRIB decoding is not required for the prepared-NPZ replay.

A stronger matched retraining diagnostic is also included separately:

```bash
cd scientific_reproduction/stage1
python run_matched_location_ablation_seed1.py
```

Using the same frozen split, hyperparameters and seed 1, the no-message nonlinear branch changes from **IoU 0.4517 / AP 0.5893** with all features to **IoU 0.4366 / AP 0.5647** after removing only `sin_lat`, `cos_lat`, `sin_lon`, `cos_lon`. Spherical mesh geometry remains because it defines the architecture. This is a bounded single-seed diagnostic, not a five-seed full-GNN generalization claim.

The later four-event safety-diffusion v3 diagnostics are bundled under `scientific_reproduction/archived_v3/`. They remain archived executed evidence rather than being falsely presented as re-executed in this release.
