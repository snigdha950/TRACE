from __future__ import annotations

from typing import Literal
import numpy as np


def _safe_ratio(num: float, den: float) -> float | None:
    return None if den == 0 else float(num / den)


def evaluate_percentile_footprint(
    ensemble_field: np.ndarray,
    reference_field: np.ndarray,
    climatology_threshold: np.ndarray,
    *,
    polarity: Literal["upper", "lower"] = "upper",
    support_threshold: float = 0.4,
) -> dict:
    """Hazard-generic spatial anomaly evaluator.

    Works for upper-tail hazards (e.g. heat) and lower-tail hazards (e.g. cold).
    This function evaluates supplied arrays; it does not claim any real-world hazard skill.
    """
    ens = np.asarray(ensemble_field, dtype=float)
    ref = np.asarray(reference_field, dtype=float)
    clim = np.asarray(climatology_threshold, dtype=float)
    if ens.ndim != 3:
        raise ValueError("ensemble_field must have shape [member, y, x]")
    if ref.shape != clim.shape or ens.shape[1:] != ref.shape:
        raise ValueError("reference_field, climatology_threshold, and ensemble spatial shape must match")
    if ens.shape[0] < 2:
        raise ValueError("at least two ensemble members are required")
    if not (0 < support_threshold <= 1):
        raise ValueError("support_threshold must be in (0,1]")
    if not np.isfinite(ens).all() or not np.isfinite(ref).all() or not np.isfinite(clim).all():
        raise ValueError("all arrays must be finite")

    if polarity == "upper":
        member_extreme = ens >= clim[None, :, :]
        truth = ref >= clim
    elif polarity == "lower":
        member_extreme = ens <= clim[None, :, :]
        truth = ref <= clim
    else:  # pragma: no cover - guarded by Literal in normal typed callers
        raise ValueError("polarity must be 'upper' or 'lower'")

    support = member_extreme.mean(axis=0)
    pred = support >= support_threshold
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    union = int(np.logical_or(pred, truth).sum())
    iou = _safe_ratio(tp, union)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    return {
        "schema": "trace-generic-hazard-footprint-evaluation/1.0",
        "polarity": polarity,
        "members": int(ens.shape[0]),
        "support_threshold": float(support_threshold),
        "metrics": {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "true_positive_cells": tp,
            "false_positive_cells": fp,
            "false_negative_cells": fn,
            "predicted_extreme_cells": int(pred.sum()),
            "reference_extreme_cells": int(truth.sum()),
            "mean_support_over_predicted_cells": float(support[pred].mean()) if pred.any() else None,
        },
        "scientific_skill_validation": False,
        "boundary": "This is a hazard-generic evaluation primitive. Scientific multi-hazard skill requires a frozen real-data event protocol and held-out results.",
    }
