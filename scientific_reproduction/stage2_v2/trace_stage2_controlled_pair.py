"""Controlled 12 km -> 5 km pair constructor for Stage 2."""
from __future__ import annotations
import numpy as np
from trace_spectral_tail_metrics import conservative_coarse_grain

def _shape_for_extent(shape,src_km,target_km):
    ey=shape[-2]*src_km; ex=shape[-1]*src_km
    ny=round(ey/target_km); nx=round(ex/target_km)
    if abs(ny*target_km-ey)>1e-6 or abs(nx*target_km-ex)>1e-6:
        raise ValueError(f"Extent {ey}x{ex} km is not exactly divisible by {target_km} km.")
    return int(ny),int(nx)

def resample_channels(x,target_shape):
    x=np.asarray(x,float)
    if x.ndim==2: return conservative_coarse_grain(x,target_shape)
    if x.ndim!=3: raise ValueError("expected (C,Y,X) or (Y,X)")
    return np.stack([conservative_coarse_grain(ch,target_shape) for ch in x])

def make_controlled_pair(highres,source_dx_km=3.0,target_dx_km=5.0,coarse_dx_km=12.0):
    x=np.asarray(highres,float)
    target_shape=_shape_for_extent(x.shape,source_dx_km,target_dx_km)
    coarse_shape=_shape_for_extent(x.shape,source_dx_km,coarse_dx_km)
    target=resample_channels(x,target_shape)
    coarse=resample_channels(target,coarse_shape)
    source_mean=np.mean(x,axis=(-2,-1))
    coarse_mean=np.mean(coarse,axis=(-2,-1))
    drift=float(np.max(np.abs(np.atleast_1d(source_mean-coarse_mean))))
    meta={"source_dx_km":source_dx_km,"target_dx_km":target_dx_km,
          "coarse_dx_km":coarse_dx_km,"source_shape":list(x.shape),
          "target_shape":list(target.shape),"coarse_shape":list(coarse.shape),
          "max_domain_mean_drift":drift,
          "claim":"Controlled 12->5 km benchmark, not operational NEPS-G validation."}
    return coarse,target,meta

def self_test():
    y,x=np.mgrid[0:80,0:80]
    high=np.stack([np.sin(x/7)+0.2*np.cos(y/5),np.cos(x/9)-0.3*np.sin(y/6)])
    coarse,target,meta=make_controlled_pair(high,3,5,12)
    assert target.shape==(2,48,48) and coarse.shape==(2,20,20)
    assert meta["max_domain_mean_drift"]<1e-12
    print("CONTROLLED_12_TO_5_SELF_TEST_PASS",meta)

if __name__=="__main__": self_test()
