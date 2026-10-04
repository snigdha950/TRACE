"""Lightweight physical-consistency diagnostics for TRACE wind downscaling.

These are diagnostics/regularizers, not a claim of solving the full primitive
equations. They are suitable only when U/V wind components are generated.
"""
from __future__ import annotations
import numpy as np


def divergence_vorticity(u, v, dx_m, dy_m):
    u = np.asarray(u, dtype=float); v = np.asarray(v, dtype=float)
    if u.shape != v.shape or u.ndim != 2:
        raise ValueError("u and v must be same-shape 2-D arrays")
    du_dx = np.gradient(u, dx_m, axis=1)
    du_dy = np.gradient(u, dy_m, axis=0)
    dv_dx = np.gradient(v, dx_m, axis=1)
    dv_dy = np.gradient(v, dy_m, axis=0)
    return du_dx + dv_dy, dv_dx - du_dy


def gradient_consistency(pred_u, pred_v, truth_u, truth_v, dx_m, dy_m):
    pd, pv = divergence_vorticity(pred_u, pred_v, dx_m, dy_m)
    td, tv = divergence_vorticity(truth_u, truth_v, dx_m, dy_m)
    return {
        "divergence_mae_s-1": float(np.mean(np.abs(pd-td))),
        "vorticity_mae_s-1": float(np.mean(np.abs(pv-tv))),
        "pred_divergence_rms_s-1": float(np.sqrt(np.mean(pd**2))),
        "truth_divergence_rms_s-1": float(np.sqrt(np.mean(td**2))),
    }


def self_test():
    y, x = np.mgrid[-1:1:32j, -1:1:32j]
    # solid-body rotation: near-zero divergence, constant vorticity
    u = -y; v = x
    d, z = divergence_vorticity(u, v, dx_m=1.0, dy_m=1.0)
    assert abs(float(d.mean())) < 1e-10
    assert np.isfinite(z).all()
    print("PHYSICS_SELF_TEST_PASS", round(float(np.abs(d).max()), 8), round(float(z.mean()), 4))


if __name__ == "__main__":
    self_test()
