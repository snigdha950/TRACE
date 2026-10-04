# TRACE Stage-2 cross-event robustness review

## Scope
This is a **post-hoc frozen cross-event robustness analysis** of the already-defined Stage-2 v2 architecture. The four storms were already present in the development dataset, so this is broader resampling evidence, **not four pristine untouched holdouts**. Hyperparameters and architecture were frozen before these fold results.

## Four-event aggregate

| Method | Speed MAE m/s ↓ | Peak abs err m/s ↓ | q99 abs err m/s ↓ | Threshold IoU ↑ | Unresolved PSD log10 MAE ↓ |
|---|---:|---:|---:|---:|---:|
| bilinear_coarse_projected | 0.3255 | 0.5026 | 0.1986 | 0.4743 | 0.3218 |
| deterministic_coarse_projected | 0.3178 | 0.5500 | 0.1673 | 0.5846 | 0.4200 |
| diffusion_mean_coarse_projected | 6.8442 | 47.2204 | 22.2338 | 0.0382 | 2.1217 |

## Paired event result
- Deterministic has lower speed MAE than bilinear in **4/4** event folds.
- Bilinear has lower unresolved-scale PSD error than deterministic in **4/4** event folds and lower peak absolute error in **3/4** folds.
- Residual diffusion mean is worse than deterministic in **4/4** folds for speed MAE, peak error, q99 error, and unresolved PSD error.

## Engineering decision
- Keep **deterministic + exact coarse projection** as the primary Stage-2 reconstruction route for mean/tail footprint skill.
- Retain **projected bilinear** as a strong spectral/peak baseline and diagnostic comparator.
- Keep diffusion as an implemented experimental candidate behind a **skill/safety gate**; do not expose it as the default because it generates large false extremes in the tested small-data regime.
- Do not claim that diffusion generally fails; the failure applies to this compact implementation and dataset.

## Claim boundary
Allowed: controlled 12→5 km HRRR reconstruction evidence across four cross-event folds with a frozen architecture. Forbidden: operational NEPS-G/India forecast skill, independent four-event untouched generalization, or calibrated probabilistic weather uncertainty.
