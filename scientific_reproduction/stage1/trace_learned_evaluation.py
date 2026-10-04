"""Shared pre-result learned evaluation. Threshold-free AP handles tied scores."""
import numpy as np
from trace_object_verify import objects_from_mask,haversine_km

def average_precision(score,target):
 sc=np.asarray(score,float).ravel();y=np.asarray(target,bool).ravel()
 if not np.isfinite(sc).all():raise ValueError('nonfinite predictive score')
 if not y.any():return float('nan')
 order=np.argsort(-sc,kind='stable');s=sc[order];y=y[order]
 end=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1];tp=np.cumsum(y)[end].astype(float);precision=tp/(end+1)
 return float(np.sum(np.diff(np.r_[0.,tp])*precision)/y.sum())

def localization(mask,target,s):
 _,fo=objects_from_mask(mask,s['latitude'],s['longitude'],s['cell_area_km2'],5000.)
 _,ro=objects_from_mask(target,s['latitude'],s['longitude'],s['cell_area_km2'],5000.)
 dist=[haversine_km(f['centroid_lat'],f['centroid_lon'],r['centroid_lat'],r['centroid_lon']) for f in fo for r in ro]
 nearest=min(dist) if dist else float('nan')
 out={'minimum_object_to_reference_centroid_distance_km':nearest,'any_object_within_300km_reference':bool(nearest<=300) if np.isfinite(nearest) else None}
 # All frozen FANI lead samples verify 2019-05-03T00:00Z. Independent IMD supplied position.
 if str(s['event_id'][0])=='FANI_2019':
  imd=[haversine_km(f['centroid_lat'],f['centroid_lon'],19.1,85.5) for f in fo]
  out['minimum_wind_object_centroid_to_imd_km']=min(imd) if imd else float('nan')
 else:out['minimum_wind_object_centroid_to_imd_km']=float('nan')
 return out
