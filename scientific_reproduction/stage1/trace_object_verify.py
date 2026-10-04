"""Object-based verification helpers for TRACE.

This is a transparent, lightweight object matcher inspired by object-based
forecast verification practice. It is NOT an implementation of NOAA MET MODE.

It complements grid-cell CSI/IoU by reporting object hits, misses, false alarms,
matched centroid error, matched overlap, and area ratio.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    a1, a2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    h = math.sin(dlat/2)**2 + math.cos(a1)*math.cos(a2)*math.sin(dlon/2)**2
    return 2*r*math.asin(math.sqrt(max(0.0, min(1.0, h))))


def _spherical_centroid(iy, ix, lat, lon, weights):
    la = np.deg2rad(lat[iy])
    lo = np.deg2rad(lon[ix])
    x = float(np.sum(weights*np.cos(la)*np.cos(lo)))
    y = float(np.sum(weights*np.cos(la)*np.sin(lo)))
    z = float(np.sum(weights*np.sin(la)))
    return (
        float(np.rad2deg(np.arctan2(z, np.hypot(x, y)))),
        float(np.rad2deg(np.arctan2(y, x))),
    )


def objects_from_mask(mask, lat, lon, cell_area_km2, min_area_km2=5000.0):
    mask = np.asarray(mask, bool)
    area = np.asarray(cell_area_km2, float)
    raw, n = ndimage.label(mask, structure=np.ones((3, 3)))
    labels = np.zeros_like(raw, dtype=np.int32)
    objects = []
    for old_label in range(1, n+1):
        iy, ix = np.where(raw == old_label)
        a = float(area[iy, ix].sum())
        if a < min_area_km2:
            continue
        new_label = len(objects)+1
        labels[iy, ix] = new_label
        centroid = _spherical_centroid(iy, ix, lat, lon, area[iy, ix])
        objects.append({
            "label": new_label,
            "area_km2": a,
            "centroid_lat": centroid[0],
            "centroid_lon": centroid[1],
            "cell_count": int(len(iy)),
        })
    return labels, objects


def verify_objects(
    forecast_mask,
    reference_mask,
    lat,
    lon,
    cell_area_km2,
    *,
    min_area_km2=5000.0,
    centroid_gate_km=300.0,
):
    """
    One-to-one object matching.

    A forecast/reference pair is admissible if their footprints overlap OR their
    centroids are within centroid_gate_km. Among admissible pairs, Hungarian
    assignment maximizes a transparent interest score:
        0.7 * IoU + 0.3 * max(0, 1 - distance/gate)

    This avoids double-counting one reference object as several forecast hits.
    """
    area = np.asarray(cell_area_km2, float)
    flab, fobj = objects_from_mask(
        forecast_mask, lat, lon, area, min_area_km2=min_area_km2
    )
    rlab, robj = objects_from_mask(
        reference_mask, lat, lon, area, min_area_km2=min_area_km2
    )

    nf, nr = len(fobj), len(robj)
    if nf == 0 or nr == 0:
        hits = 0
        return {
            "forecast_object_count": nf,
            "reference_object_count": nr,
            "object_hits": hits,
            "object_false_alarms": nf,
            "object_misses": nr,
            "object_precision": (hits/nf if nf else float("nan")),
            "object_recall": (hits/nr if nr else float("nan")),
            "mean_matched_iou": float("nan"),
            "mean_matched_centroid_error_km": float("nan"),
            "mean_matched_area_ratio": float("nan"),
            "matches": [],
        }

    score = np.zeros((nf, nr), dtype=float)
    pair_meta = {}
    for i, fo in enumerate(fobj):
        fm = flab == fo["label"]
        for j, ro in enumerate(robj):
            rm = rlab == ro["label"]
            overlap = float(area[fm & rm].sum())
            union = fo["area_km2"] + ro["area_km2"] - overlap
            iou = overlap / union if union > 0 else 0.0
            dist = haversine_km(
                fo["centroid_lat"], fo["centroid_lon"],
                ro["centroid_lat"], ro["centroid_lon"]
            )
            admissible = overlap > 0 or dist <= centroid_gate_km
            interest = (
                0.7*iou + 0.3*max(0.0, 1.0-dist/centroid_gate_km)
                if admissible else 0.0
            )
            score[i, j] = interest
            pair_meta[(i, j)] = {
                "forecast_object": i+1,
                "reference_object": j+1,
                "iou": float(iou),
                "centroid_error_km": float(dist),
                "area_ratio_forecast_over_reference": float(
                    fo["area_km2"]/ro["area_km2"]
                ),
                "interest": float(interest),
                "admissible": bool(admissible),
            }

    # Dummy rows/cols allow objects to stay unmatched instead of forcing bad pairs.
    n = nf + nr
    cost = np.ones((n, n), dtype=float)
    cost[:nf, :nr] = 1.0 - score
    cost[:nf, nr:] = 1.0
    cost[nf:, :nr] = 1.0
    cost[nf:, nr:] = 0.0
    rows, cols = linear_sum_assignment(cost)

    matches = []
    for i, j in zip(rows, cols):
        if i < nf and j < nr:
            meta = pair_meta[(i, j)]
            if meta["admissible"] and meta["interest"] > 0:
                matches.append(meta)

    hits = len(matches)
    def _mean(k):
        return float(np.mean([m[k] for m in matches])) if matches else float("nan")

    return {
        "forecast_object_count": nf,
        "reference_object_count": nr,
        "object_hits": hits,
        "object_false_alarms": nf-hits,
        "object_misses": nr-hits,
        "object_precision": hits/nf if nf else float("nan"),
        "object_recall": hits/nr if nr else float("nan"),
        "mean_matched_iou": _mean("iou"),
        "mean_matched_centroid_error_km": _mean("centroid_error_km"),
        "mean_matched_area_ratio": _mean("area_ratio_forecast_over_reference"),
        "matches": matches,
    }
