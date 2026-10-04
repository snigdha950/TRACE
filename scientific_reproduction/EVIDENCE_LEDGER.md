# TRACE v1.8 evidence ledger

This ledger separates fully regenerable evidence from archived executed evidence. It deliberately does not turn correlated forecasts into independent weather cases.

## BULBUL wind spherical GNN Climate99 area-weighted IoU
- Tier: **FULL_FROZEN_INFERENCE_REPLAY_INCLUDED**
- Source: `scientific_reproduction/stage1/BULBUL_GNN_REPRODUCTION.json`
- Generator: `scientific_reproduction/stage1/reproduce_bulbul_gnn.py`
- Value: `0.42979942438385327`
- Boundary: Mean over 5 seeds x 3 correlated lead forecasts sharing one valid time.

## BULBUL wind spherical GNN AP
- Tier: **FULL_FROZEN_INFERENCE_REPLAY_INCLUDED**
- Source: `scientific_reproduction/stage1/BULBUL_GNN_REPRODUCTION.json`
- Generator: `scientific_reproduction/stage1/reproduce_bulbul_gnn.py`
- Value: `0.6069505476562604`

## Stage-2 v2 held-out Ida deterministic projected MAE
- Tier: **FULL_TRAIN_EVAL_REPLAY_INCLUDED**
- Source: `scientific_reproduction/stage2_v2/stage2_run_v2/stage2_metrics.json`
- Generator: `scientific_reproduction/stage2_v2/run_stage2_weather_v2.py via run_portable_stage2_v2.py`
- Boundary: Original scientific runner is unchanged; wrapper only makes optional eccodes version logging nonfatal for prepared-NPZ replay.

## Stage-2 four-event event-equal deterministic MAE
- Tier: **ARCHIVED_EXECUTED_DIAGNOSTIC**
- Source: `scientific_reproduction/archived_v3/STAGE2_DIFFUSION_SAFETY_V3_DIAGNOSTICS.json`
- Value: `0.3178067875138786`
- Boundary: Later cross-event runner is documented in REPRODUCTION.md but is not bundled in this compact release.

## Stage-2 safety diffusion fair CRPS
- Tier: **ARCHIVED_EXECUTED_DIAGNOSTIC**
- Source: `scientific_reproduction/archived_v3/STAGE2_DIFFUSION_SAFETY_V3_DIAGNOSTICS.json`
- Value: `0.2602071614284556`

## Location-feature occlusion sensitivity
- Tier: **FULL_INFERENCE_SENSITIVITY_REPLAY_INCLUDED**
- Source: `scientific_reproduction/stage1/LOCATION_FEATURE_OCCLUSION_RESULTS.json`
- Generator: `scientific_reproduction/stage1/run_location_occlusion.py`
- Value: `{'original_iou': 0.4297994243838533, 'occluded_iou': 0.35990004823737604, 'delta': -0.06989937614647729}`
- Boundary: Inference-only occlusion, not a retrained no-location ablation.

## Matched retrained explicit-location ablation (seed 1)
- Tier: **FULL_RETRAINED_SINGLE_SEED_DIAGNOSTIC_INCLUDED**
- Source: `scientific_reproduction/stage1/RETRAINED_NO_EXPLICIT_LOCATION_NOMESSAGE_SEED1.json`
- Generator: `scientific_reproduction/stage1/run_matched_location_ablation_seed1.py`
- Model: no-message nonlinear ablation, chosen because the frozen evidence does not establish stable message-passing gain.
- Full features: IoU **0.45167879**, AP **0.58926652**.
- Without explicit `sin_lat/cos_lat/sin_lon/cos_lon`: IoU **0.43663445**, AP **0.56470688**.
- Boundary: same frozen event split and hyperparameters, seed 1 only; spherical mesh geometry is retained. This weakens the explicit-geography leakage concern but is not a five-seed full-GNN geographic-generalization proof.

## 5 km UI marker
- Tier: **SOFTWARE_DISPLAY_ONLY**
- Source: `alert_api/rebuild_alerts.py`
- Value: `5.0`
- Boundary: Explicitly not a predicted impact radius and not Stage-2 output.
