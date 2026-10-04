"""TRACE Shift of Tails (SOT) diagnostic.

Implements the upper-tail SOT used by ECMWF-style extreme guidance:

    SOT90 = (Qf(90) - Qc(99)) / (Qc(99) - Qc(90))

where Qf is the forecast-ensemble quantile and Qc is the climatological
quantile. Positive values mean the forecast 90th percentile exceeds the
climatological 99th percentile.

TRACE caveat: with the ordinary 5-member GEFSv12 reforecast ensemble, the
forecast 90th percentile is very coarsely estimated. This module is therefore a
supplementary diagnostic, not a stand-alone calibrated probability product.
"""
from __future__ import annotations
import numpy as np


def compute_upper_sot(members, climate_q90, climate_q99, forecast_quantile=0.90):
    """Compute upper-tail SOT on an already aligned grid.

    Parameters
    ----------
    members : array (M, y, x)
    climate_q90, climate_q99 : arrays (y, x)
    forecast_quantile : float, default .90
    """
    m = np.asarray(members, dtype=float)
    q90 = np.asarray(climate_q90, dtype=float)
    q99 = np.asarray(climate_q99, dtype=float)
    if m.ndim != 3:
        raise ValueError("members must have shape (M,y,x)")
    if q90.shape != m.shape[1:] or q99.shape != m.shape[1:]:
        raise ValueError("climatology grid must match member grid")
    if not (0 < forecast_quantile < 1):
        raise ValueError("forecast_quantile must be in (0,1)")
    if not np.isfinite(m).all() or not np.isfinite(q90).all() or not np.isfinite(q99).all():
        raise ValueError("non-finite SOT input")

    denom = q99 - q90
    if np.any(denom <= 0):
        raise ValueError("climate q99 must exceed q90 at every grid point")
    qf = np.quantile(m, forecast_quantile, axis=0, method="linear")
    return (qf - q99) / denom


def climate_q90_q99(climate_quantiles, p_levels):
    """Extract/interpolate q90 and q99 from a climate-quantile cube."""
    q = np.asarray(climate_quantiles, dtype=float)
    p = np.asarray(p_levels, dtype=float)
    if q.ndim != 3 or q.shape[0] != len(p):
        raise ValueError("expected climate quantiles (P,y,x)")
    # Interpolate along quantile axis independently at every grid cell.
    flat = q.reshape(q.shape[0], -1)
    out = []
    for target in (0.90, 0.99):
        vals = np.array([np.interp(target, p, flat[:, j]) for j in range(flat.shape[1])])
        out.append(vals.reshape(q.shape[1:]))
    return out[0], out[1]


def self_test():
    rng = np.random.default_rng(11)
    climate = rng.normal(10, 2, size=(5000, 4, 5))
    q90 = np.quantile(climate, .90, axis=0)
    q99 = np.quantile(climate, .99, axis=0)
    normal = rng.normal(10, 2, size=(5, 4, 5))
    extreme = rng.normal(18, 1, size=(5, 4, 5))
    a = compute_upper_sot(normal, q90, q99)
    b = compute_upper_sot(extreme, q90, q99)
    assert b.mean() > a.mean()
    assert (b > 0).mean() > 0.9
    print("SOT_SELF_TEST_PASS", round(float(a.mean()), 3), round(float(b.mean()), 3))


if __name__ == "__main__":
    self_test()
