# TRACE Stage-2 v3 safety-diffusion review

## Why v3 exists
Stage-2 v2 produced severe false extremes, so the redesign was allowed by the frozen change rule for a **measured failure**. Before v3 results were opened, the safety protocol froze two sampling changes: deterministic DDIM (eta=0) and a training-only q99.5 symmetric clean-residual clip. Training objective, model family, 32-step cosine schedule, validation-only checkpointing, 16 samples, and exact coarse projection were kept.

## Result: catastrophic diffusion failure was removed

| Metric (event-equal over 4 cross-event folds) | Deterministic | Safety diffusion v3 |
|---|---:|---:|
| Speed MAE ↓ | 0.3178 | 0.3585 |
| q99 absolute error ↓ | 0.1673 | 0.1651 |
| Threshold IoU ↑ | 0.5846 | 0.4931 |
| Unresolved PSD log10 MAE ↓ | 0.4200 | 0.2351 |

### Probabilistic diagnostics

- Fair CRPS of the 16 conditional v3 samples: **0.2602 m/s**.
- Deterministic point-forecast MAE on the same folds: **0.3178 m/s**.
- The v3 conditional ensemble has lower CRPS than deterministic MAE in **4/4 event folds**.
- Mean spread-skill ratio: **1.52**, indicating over-dispersion rather than calibrated uncertainty.
- Diffusion sample/frame combinations with any far-field extreme beyond the tolerance: **13.9%**; average spurious far cells **0.36**.

## Interpretation
- **Deterministic + coarse projection remains the primary footprint/mean-error route** because it has lower MAE and higher threshold IoU.
- **Safety diffusion v3 earns a complementary role**: it substantially improves unresolved-scale spectral structure and is slightly better on mean q99 absolute error, while its conditional sample distribution has lower CRPS than the deterministic point forecast.
- Diffusion is therefore no longer presented as a universal winner. TRACE uses an evidence gate: deterministic for primary location/footprint fidelity, diffusion for stochastic fine-scale structure only when safety diagnostics pass.

## v2 → v3 failure correction
The original v2 Ida experiment had diffusion CRPS 16.27 m/s and huge false extremes. The v3 safety sampler reduces the event-equal four-fold CRPS to about 0.26 m/s. These numbers are not directly a single identical aggregate because v3 is reported across four cross-event folds; the F1 Ida fold uses the same Laura+Delta train / Sally validation / Ida test partition and shows the same qualitative correction.

## Claim boundary
This is a controlled HRRR reconstruction benchmark, not operational NEPS-G or India forecast validation. The four-fold analysis is broader cross-event robustness, not four pristine untouched holdouts. Conditional diffusion samples are not independent atmospheric forecast members.
