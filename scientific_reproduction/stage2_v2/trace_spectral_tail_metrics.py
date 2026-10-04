"""Tail, spectrum, and scale-consistency diagnostics for TRACE Stage 2.

These diagnostics are intended to test the PS's spectral-smoothing claim.
They are not substitutes for independent forecast verification.
"""
from __future__ import annotations
import numpy as np


def tail_metrics(prediction, truth, quantiles=(0.95, 0.99, 0.999), threshold=None):
    p = np.asarray(prediction, dtype=float)
    t = np.asarray(truth, dtype=float)
    if p.shape != t.shape:
        raise ValueError("prediction/truth shapes differ")
    mask = np.isfinite(p) & np.isfinite(t)
    if not mask.any():
        raise ValueError("no paired finite values")
    pv, tv = p[mask], t[mask]
    out = {
        "peak_bias": float(np.max(pv) - np.max(tv)),
        "mae": float(np.mean(np.abs(pv-tv))),
    }
    for q in quantiles:
        out[f"q{int(round(q*1000)):03d}_bias"] = float(np.quantile(pv, q) - np.quantile(tv, q))
    if threshold is not None:
        fp = pv >= threshold
        ft = tv >= threshold
        tp = np.sum(fp & ft); fa = np.sum(fp & ~ft); miss = np.sum(~fp & ft)
        out["threshold_recall"] = float(tp/(tp+miss)) if tp+miss else float("nan")
        out["threshold_precision"] = float(tp/(tp+fa)) if tp+fa else float("nan")
    return out


def radial_power_spectrum(field, dx_km=1.0, dy_km=1.0, bins=32):
    """Approximate isotropic 2-D power spectrum on a locally Cartesian patch."""
    x = np.asarray(field, dtype=float)
    if x.ndim != 2:
        raise ValueError("field must be 2-D")
    if not np.isfinite(x).all():
        raise ValueError("field contains non-finite values")
    x = x - x.mean()
    win_y = np.hanning(x.shape[0])
    win_x = np.hanning(x.shape[1])
    x = x * np.outer(win_y, win_x)
    fft = np.fft.rfft2(x)
    power = np.abs(fft)**2
    ky = np.fft.fftfreq(x.shape[0], d=dy_km)
    kx = np.fft.rfftfreq(x.shape[1], d=dx_km)
    kr = np.sqrt(ky[:, None]**2 + kx[None, :]**2)
    positive = kr > 0
    kmin = np.min(kr[positive]); kmax = np.max(kr)
    edges = np.geomspace(kmin, kmax, bins+1)
    centers = np.sqrt(edges[:-1]*edges[1:])
    psd = np.full(bins, np.nan)
    for i in range(bins):
        m = (kr >= edges[i]) & (kr < edges[i+1])
        if np.any(m):
            psd[i] = float(np.mean(power[m]))
    return centers, psd


def spectral_slope(field, dx_km=1.0, dy_km=1.0, min_fraction=0.15, max_fraction=0.70):
    k, p = radial_power_spectrum(field, dx_km, dy_km)
    finite = np.isfinite(p) & (p > 0)
    k, p = k[finite], p[finite]
    if len(k) < 6:
        return float("nan")
    lo = np.quantile(k, min_fraction); hi = np.quantile(k, max_fraction)
    m = (k >= lo) & (k <= hi)
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(k[m]), np.log(p[m]), 1)[0])


def _overlap_matrix(n_fine, n_coarse):
    """Area-overlap weights mapping uniform 1-D fine cells to coarse cells."""
    fine_edges = np.linspace(0.0, 1.0, n_fine+1)
    coarse_edges = np.linspace(0.0, 1.0, n_coarse+1)
    W = np.zeros((n_coarse, n_fine), dtype=float)
    for c in range(n_coarse):
        a, b = coarse_edges[c], coarse_edges[c+1]
        for f in range(n_fine):
            x, y = fine_edges[f], fine_edges[f+1]
            overlap = max(0.0, min(b, y)-max(a, x))
            if overlap:
                W[c, f] = overlap/(b-a)
    return W


def conservative_coarse_grain(fine, target_shape):
    """Conservative area-average from a uniform fine grid to arbitrary shape."""
    x = np.asarray(fine, dtype=float)
    if x.ndim != 2:
        raise ValueError("fine field must be 2-D")
    Wy = _overlap_matrix(x.shape[0], int(target_shape[0]))
    Wx = _overlap_matrix(x.shape[1], int(target_shape[1]))
    return Wy @ x @ Wx.T


def coarse_consistency_error(fine_prediction, coarse_input):
    down = conservative_coarse_grain(fine_prediction, np.asarray(coarse_input).shape)
    coarse = np.asarray(coarse_input, dtype=float)
    return {
        "coarse_mae": float(np.mean(np.abs(down-coarse))),
        "coarse_bias": float(np.mean(down-coarse)),
        "domain_mean_difference": float(down.mean()-coarse.mean()),
    }


def self_test():
    y, x = np.mgrid[0:64, 0:64]
    truth = np.sin(x/2.5) + 0.5*np.sin(y/1.7)
    smooth = np.sin(x/8.0) + 0.3*np.sin(y/7.0)
    assert spectral_slope(truth) != spectral_slope(smooth)
    coarse = conservative_coarse_grain(truth, (24, 24))
    err = coarse_consistency_error(truth, coarse)
    assert err["coarse_mae"] < 1e-12
    print("SPECTRAL_TAIL_SELF_TEST_PASS", round(spectral_slope(truth), 3), round(spectral_slope(smooth), 3))


if __name__ == "__main__":
    self_test()
