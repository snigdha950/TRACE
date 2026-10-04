from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

MEMBER_ALIASES = ("member", "ensemble_member", "number", "realization", "ens", "perturbation")
LAT_ALIASES = ("lat", "latitude", "y")
LON_ALIASES = ("lon", "longitude", "x")
INIT_ALIASES = ("forecast_initialization_time", "initialization_time", "forecast_reference_time", "reference_time", "init_time")
VALID_ALIASES = ("valid_time", "time", "forecast_time")
LEAD_ALIASES = ("lead_hours", "forecast_period", "step", "lead")

MET_VARIABLE_HINTS = (
    "u10", "v10", "10m_u_component_of_wind", "10m_v_component_of_wind", "wind",
    "t2m", "2m_temperature", "temperature", "mslp", "sp", "q2m", "spfh_2m",
    "precip", "tp", "geopotential", "z",
)


def _lower_keys(mapping: dict[str, Any]) -> set[str]:
    return {str(k).lower() for k in mapping}


def _find_alias(keys: set[str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias.lower() in keys:
            return alias.lower()
    return None


def validate_provider_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Validate provider metadata for TRACE ingest.

    This is intentionally a schema/readiness check. It never returns a meteorological-skill
    claim and cannot convert a provider-compatible file into NEPS-G validation evidence.
    """
    provider = str(meta.get("provider_name") or meta.get("provider_hint") or "UNKNOWN").strip()
    dimensions = meta.get("dimensions") or {}
    variables = meta.get("variables") or {}
    coordinates = meta.get("coordinates") or {}
    lead_hours = meta.get("lead_hours") or []
    init_time = meta.get("initialization_time") or meta.get("forecast_reference_time")
    valid_times = meta.get("valid_times") or []

    if not isinstance(dimensions, dict):
        dimensions = {}
    if isinstance(variables, list):
        variables = {str(v): {} for v in variables}
    if not isinstance(variables, dict):
        variables = {}
    if isinstance(coordinates, list):
        coordinates = {str(v): {} for v in coordinates}
    if not isinstance(coordinates, dict):
        coordinates = {}

    dim_keys = _lower_keys(dimensions)
    var_keys = _lower_keys(variables)
    coord_keys = _lower_keys(coordinates)
    all_keys = dim_keys | coord_keys | var_keys

    member_name = _find_alias(dim_keys, MEMBER_ALIASES)
    lat_name = _find_alias(all_keys, LAT_ALIASES)
    lon_name = _find_alias(all_keys, LON_ALIASES)
    valid_name = _find_alias(all_keys, VALID_ALIASES)
    lead_name = _find_alias(all_keys, LEAD_ALIASES)
    init_name = _find_alias(all_keys, INIT_ALIASES)

    ensemble_size = None
    if member_name:
        try:
            ensemble_size = int(dimensions.get(member_name, dimensions.get(next(k for k in dimensions if str(k).lower() == member_name))))
        except Exception:
            ensemble_size = None

    recognized_met_vars = [name for name in variables if any(h in str(name).lower() for h in MET_VARIABLE_HINTS)]
    parsed_leads: list[float] = []
    for value in lead_hours:
        try:
            f = float(value)
            if np.isfinite(f):
                parsed_leads.append(f)
        except Exception:
            pass
    medium_range = [x for x in parsed_leads if 72 <= x <= 240]

    checks = {
        "latitude_axis_present": bool(lat_name),
        "longitude_axis_present": bool(lon_name),
        "ensemble_member_dimension_present": bool(member_name and ensemble_size and ensemble_size >= 2),
        "initialization_time_present": bool(init_time or init_name),
        "valid_time_or_lead_present": bool(valid_times or parsed_leads or valid_name or lead_name),
        "meteorological_variable_present": bool(recognized_met_vars or variables),
        "positive_dimensions": bool(dimensions) and all(isinstance(v, (int, np.integer)) and int(v) > 0 for v in dimensions.values()),
    }
    required_pass = all(checks.values())
    medium_range_status = (
        "PASS_EXPLICIT_3_TO_10_DAY_LEAD_PRESENT" if medium_range else
        "NOT_DEMONSTRATED_FROM_METADATA" if not parsed_leads else
        "FAIL_NO_3_TO_10_DAY_LEAD_IN_LIST"
    )

    normalized = {
        "provider_name": provider,
        "member_dimension": member_name,
        "ensemble_size": ensemble_size,
        "latitude_axis": lat_name,
        "longitude_axis": lon_name,
        "recognized_meteorological_variables": recognized_met_vars or list(variables.keys()),
        "lead_hours": parsed_leads,
        "medium_range_leads_72_240h": medium_range,
    }
    return {
        "schema": "trace-provider-ingest-validation/1.0",
        "status": "PASS_SCHEMA_COMPATIBLE" if required_pass else "FAIL_SCHEMA_INCOMPLETE",
        "provider": provider,
        "checks": checks,
        "normalized": normalized,
        "medium_range_status": medium_range_status,
        "trace_adapter_compatible": required_pass,
        "direct_nepsg_skill_claim_supported": False,
        "scientific_skill_validation": False,
        "boundary": "This validates input structure/readiness only. It does not establish forecast skill, calibrated probability, or direct NEPS-G validation.",
    }


def _xarray_metadata(path: Path) -> dict[str, Any]:
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - environment-dependent optional path
        return {"status": "BLOCKED_MISSING_XARRAY", "error": str(exc)}
    try:
        with xr.open_dataset(path) as ds:
            dims = {str(k): int(v) for k, v in ds.sizes.items()}
            variables = {str(k): {"dims": list(v.dims), "units": v.attrs.get("units")} for k, v in ds.data_vars.items()}
            coords = {str(k): {"dims": list(v.dims), "units": v.attrs.get("units")} for k, v in ds.coords.items()}
            attrs = {str(k): str(v) for k, v in ds.attrs.items()}
            lead_hours: list[float] = []
            for name in ds.coords:
                low = str(name).lower()
                if low in LEAD_ALIASES or low in {"step", "forecast_period"}:
                    vals = np.asarray(ds.coords[name].values).ravel()
                    for v in vals:
                        try:
                            if np.issubdtype(np.asarray(v).dtype, np.timedelta64):
                                lead_hours.append(float(v / np.timedelta64(1, "h")))
                            else:
                                lead_hours.append(float(v))
                        except Exception:
                            pass
            init = None
            for name in list(ds.coords) + list(ds.variables):
                if str(name).lower() in INIT_ALIASES:
                    try:
                        init = str(np.asarray(ds[name].values).ravel()[0])
                    except Exception:
                        init = str(ds[name].values)
                    break
            return {
                "provider_name": attrs.get("provider") or attrs.get("institution") or "UNKNOWN_NETCDF",
                "dimensions": dims,
                "variables": variables,
                "coordinates": coords,
                "lead_hours": lead_hours,
                "initialization_time": init,
                "source_format": "netcdf",
                "source_path": str(path),
            }
    except Exception as exc:
        return {"status": "FAIL_OPEN_NETCDF", "error": f"{type(exc).__name__}: {exc}", "source_path": str(path)}


def _npz_metadata(path: Path) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as z:
            keys = list(z.files)
            dimensions: dict[str, int] = {}
            coords: dict[str, Any] = {}
            variables: dict[str, Any] = {}
            for key in keys:
                arr = np.asarray(z[key])
                low = key.lower()
                if low in LAT_ALIASES or low in LON_ALIASES or low in LEAD_ALIASES:
                    coords[key] = {"shape": list(arr.shape)}
                    if arr.ndim == 1:
                        dimensions[key] = int(arr.shape[0])
                else:
                    variables[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            # NPZ does not carry named dimensions reliably. Honor conventional scalar metadata keys.
            for alias in MEMBER_ALIASES:
                if alias in z.files:
                    arr = np.asarray(z[alias])
                    dimensions[alias] = int(arr.size)
            lead_hours = []
            for alias in LEAD_ALIASES:
                if alias in z.files:
                    lead_hours = [float(x) for x in np.asarray(z[alias]).ravel() if np.isfinite(float(x))]
                    break
            return {
                "provider_name": "UNKNOWN_NPZ",
                "dimensions": dimensions,
                "variables": variables,
                "coordinates": coords,
                "lead_hours": lead_hours,
                "source_format": "npz",
                "source_path": str(path),
                "note": "NPZ has no standardized named-dimension metadata; explicit provider metadata may be required.",
            }
    except Exception as exc:
        return {"status": "FAIL_OPEN_NPZ", "error": f"{type(exc).__name__}: {exc}", "source_path": str(path)}


def inspect_local_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return {"schema": "trace-provider-file-inspection/1.0", "status": "FAIL_NOT_FOUND", "path": str(p)}
    suffix = p.suffix.lower()
    if suffix in {".nc", ".netcdf"}:
        raw = _xarray_metadata(p)
    elif suffix == ".npz":
        raw = _npz_metadata(p)
    elif suffix in {".grib", ".grb", ".grib2", ".grb2"}:
        try:
            import cfgrib  # noqa: F401
        except ImportError:
            return {
                "schema": "trace-provider-file-inspection/1.0",
                "status": "BLOCKED_OPTIONAL_GRIB_DEPENDENCY",
                "path": str(p),
                "required_optional_stack": ["cfgrib", "ecCodes"],
                "boundary": "File recognition succeeded, but GRIB decoding was not attempted without the optional decoder stack.",
            }
        raw = _xarray_metadata(p)
        raw["source_format"] = "grib2"
    else:
        return {"schema": "trace-provider-file-inspection/1.0", "status": "FAIL_UNSUPPORTED_FORMAT", "path": str(p), "supported": [".nc", ".npz", ".grib2/.grb2 with optional cfgrib/ecCodes"]}
    if raw.get("status", "").startswith(("FAIL", "BLOCKED")):
        return {"schema": "trace-provider-file-inspection/1.0", **raw}
    validation = validate_provider_metadata(raw)
    return {"schema": "trace-provider-file-inspection/1.0", "path": str(p), "source_metadata": raw, "validation": validation, "status": validation["status"]}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Validate a local weather-provider file against the TRACE ingest contract.")
    parser.add_argument("path")
    args = parser.parse_args()
    print(json.dumps(inspect_local_file(args.path), indent=2, default=str))


if __name__ == "__main__":
    main()
