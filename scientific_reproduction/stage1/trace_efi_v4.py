"""TRACE PS-aligned ERA5-climatology EFI, v4."""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np

P_LEVELS=np.linspace(0.01,0.99,99,dtype=float)

def _uv_names(ds):
    for u,v in (("u10","v10"),("10m_u_component_of_wind","10m_v_component_of_wind")):
        if u in ds and v in ds: return u,v
    raise ValueError("10-m U/V variables not found")

def _lat_lon_names(obj):
    lat=next((n for n in ("latitude","lat") if n in obj.coords),None)
    lon=next((n for n in ("longitude","lon") if n in obj.coords),None)
    if lat is None or lon is None: raise ValueError("lat/lon coordinates not found")
    return lat,lon

def normalize_grid(obj):
    lon_name=next((n for n in ("longitude","lon") if n in obj.coords),None)
    if lon_name is None: raise ValueError("longitude coordinate not found")
    obj=obj.assign_coords({lon_name:((obj[lon_name]+180)%360)-180}).sortby(lon_name)
    lat_name,_=_lat_lon_names(obj)
    return obj.sortby(lat_name)

def align_to_forecast(reference,forecast,method="linear"):
    ref=normalize_grid(reference); fc=normalize_grid(forecast)
    rlat,rlon=_lat_lon_names(ref); flat,flon=_lat_lon_names(fc)
    rename={}
    if rlat!=flat: rename[rlat]=flat
    if rlon!=flon: rename[rlon]=flon
    if rename: ref=ref.rename(rename)
    aligned=ref.interp({flat:fc[flat],flon:fc[flon]},method=method)
    lat_err=float(np.max(np.abs(aligned[flat].values-fc[flat].values)))
    lon_err=float(np.max(np.abs(aligned[flon].values-fc[flon].values)))
    if lat_err>1e-10 or lon_err>1e-10:
        raise ValueError(f"coordinate alignment failed lat={lat_err} lon={lon_err}")
    return aligned,fc,{"lat_error":lat_err,"lon_error":lon_err,"method":method}

def _noleap_doy(times):
    doy=times.dt.dayofyear
    leap_shift=(times.dt.is_leap_year & (times.dt.month>2)).astype(int)
    return doy-leap_shift

def build_era5_climatology(files_or_pattern,valid_time,window_days=7,p_levels=P_LEVELS):
    """
    Build quantiles conditioned on a no-leap seasonal day and exact UTC hour.
    Returns an xarray DataArray with coordinates preserved.
    """
    import xarray as xr, pandas as pd

    files=sorted(glob.glob(str(files_or_pattern))) if isinstance(files_or_pattern,(str,Path)) else [str(x) for x in files_or_pattern]
    if not files: raise FileNotFoundError("no climatology files")

    vt=pd.Timestamp(valid_time)
    target_doy=int(vt.dayofyear - (1 if vt.is_leap_year and vt.month>2 else 0))
    target_hour=int(vt.hour)

    chunks=[]; lat0=lon0=None; latn=lonn=None
    for f in files:
        ds=xr.open_dataset(f)
        try:
            u,v=_uv_names(ds)
            ws=np.hypot(ds[u],ds[v])
            tname="valid_time" if "valid_time" in ws.coords else "time"
            ws=normalize_grid(ws)
            latn,lonn=_lat_lon_names(ws)
            doy=_noleap_doy(ws[tname])
            diff=np.abs(doy-target_doy)
            # Seasonal window is nowhere near New Year for FANI; no wrap needed here.
            mask=(diff<=window_days)&(ws[tname].dt.hour==target_hour)
            sel=ws.where(mask,drop=True)
            if sel.sizes.get(tname,0):
                if lat0 is None:
                    lat0=sel[latn].values.copy(); lon0=sel[lonn].values.copy()
                elif not np.array_equal(lat0,sel[latn].values) or not np.array_equal(lon0,sel[lonn].values):
                    raise ValueError("ERA5 climatology grid changed between files")
                chunks.append(sel.values)
        finally:
            ds.close()

    if not chunks: raise ValueError("no climatology samples survived filtering")
    data=np.concatenate(chunks,axis=0)
    if data.shape[0]<100: raise ValueError(f"too few climate samples: {data.shape[0]}")
    if np.isfinite(data).mean()<0.99: raise ValueError("too many nonfinite climatology values")
    q=np.nanquantile(data,p_levels,axis=0)
    da=xr.DataArray(q,dims=("quantile",latn,lonn),
                    coords={"quantile":p_levels,latn:lat0,lonn:lon0},
                    name="era5_wind_climatology_quantiles")
    meta={"label":"PS-aligned ERA5-climatology EFI","samples":int(data.shape[0]),
          "valid_time":str(vt),"target_noleap_doy":target_doy,"hour_utc":target_hour,
          "window_days":window_days}
    return da,meta

def compute_efi(members,clim_q,p_levels=P_LEVELS):
    m=np.asarray(members,float); q=np.asarray(clim_q,float); p=np.asarray(p_levels,float)
    if m.ndim!=3 or q.ndim!=3: raise ValueError("expected (M,y,x) and (P,y,x)")
    if q.shape[0]!=len(p) or m.shape[1:]!=q.shape[1:]: raise ValueError("shape mismatch")
    if not np.isfinite(m).all() or not np.isfinite(q).all(): raise ValueError("nonfinite input")
    below=(m[None,...]<q[:,None,...]).mean(axis=1)
    pp=p[:,None,None]
    integrand=(pp-below)/np.sqrt(pp*(1-pp))
    trap=getattr(np,"trapezoid",None)
    val=trap(integrand,p,axis=0) if trap is not None else np.trapz(integrand,p,axis=0)
    return np.clip((2/np.pi)*val,-1,1)

def bias_diagnostic(forecast,reference):
    f=np.asarray(forecast,float); r=np.asarray(reference,float)
    if f.shape!=r.shape: raise ValueError("shape mismatch")
    mask=np.isfinite(f)&np.isfinite(r)
    d=f[mask]-r[mask]
    return {"mean_bias":float(d.mean()),"median_bias":float(np.median(d)),
            "mae":float(np.abs(d).mean()),"n":int(d.size)}

def self_test():
    rng=np.random.default_rng(42)
    climate=rng.gamma(4,1.5,size=(2500,5,6))
    q=np.quantile(climate,P_LEVELS,axis=0)
    normal=rng.gamma(4,1.5,size=(5,5,6))
    extreme=normal+8
    a=compute_efi(normal,q); b=compute_efi(extreme,q)
    assert a.shape==(5,6) and b.mean()>a.mean()
    assert np.max(np.abs(a))<=1 and np.max(np.abs(b))<=1
    print("EFI_SELF_TEST_PASS",round(float(a.mean()),3),round(float(b.mean()),3))

if __name__=="__main__":
    self_test()
