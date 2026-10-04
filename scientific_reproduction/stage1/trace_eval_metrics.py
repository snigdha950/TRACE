"""TRACE FANI evaluation metrics."""
from __future__ import annotations
import math, numpy as np

def binary_scores(forecast,reference):
    f=np.asarray(forecast,bool); r=np.asarray(reference,bool)
    if f.shape!=r.shape: raise ValueError("mask shape mismatch")
    tp=int((f&r).sum()); fp=int((f&~r).sum()); fn=int((~f&r).sum()); tn=int((~f&~r).sum())
    def div(a,b): return float(a/b) if b else float("nan")
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,
            "precision":div(tp,tp+fp),"recall":div(tp,tp+fn),
            "critical_success_index":div(tp,tp+fp+fn),
            "footprint_iou":div(tp,(f|r).sum())}

def haversine_km(lat1,lon1,lat2,lon2):
    R=6371.0088
    p1,p2=map(math.radians,[lat1,lat2])
    dp=math.radians(lat2-lat1); dl=math.radians(lon2-lon1)
    a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def peak_bias(forecast,reference,mask=None):
    f=np.asarray(forecast,float); r=np.asarray(reference,float)
    if mask is not None:
        m=np.asarray(mask,bool); f=f[m]; r=r[m]
    return float(np.nanmax(f)-np.nanmax(r)) if f.size else float("nan")
