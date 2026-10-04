from pathlib import Path
import numpy as np, torch, json, hashlib
from scipy import ndimage
from trace_spherical_gnn import TraceSphericalGNN, apply_knn
from trace_learned_evaluation import average_precision
ROOT=Path('/mnt/data/trace_v17_work/TRACE_FINAL_PROTOTYPE/alert_api/source_evidence')
OUT=Path('/mnt/data/trace_v17_latlon_ablation')

def af(mask,area,min_area=5000):
 lab,n=ndimage.label(mask,structure=np.ones((3,3)));out=np.zeros_like(mask,bool)
 for i in range(1,n+1):
  m=lab==i
  if float(area[m].sum())>=min_area:out|=m
 return out
def iou(mask,target,area):
 u=mask|target
 return float(area[mask&target].sum()/area[u].sum()) if u.any() else None
rows=[]
for cp in sorted(ROOT.glob('gnn_seed*.pt')):
 ck=torch.load(cp,map_location='cpu',weights_only=False)
 model=TraceSphericalGNN(len(ck['feature_names']),hidden=32,blocks=2);model.load_state_dict(ck['state_dict']);model.eval()
 mean=np.asarray(ck['feature_mean'],np.float32);std=np.asarray(ck['feature_std'],np.float32);loc=[ck['feature_names'].index(n) for n in ['sin_lat','cos_lat','sin_lon','cos_lon']]
 for fp in sorted(ROOT.glob('BULBUL_2019_*.npz')):
  s={k:v for k,v in np.load(fp,allow_pickle=False).items()};x=s['node_features'].astype(np.float32);target=s['target_clim99_grid'].astype(bool);area=s['cell_area_km2']
  rec={'seed':int(ck['seed']),'lead_hours':int(s['lead_hours'][0])}
  for label,xx in [('original',x.copy()),('location_mean_occluded',x.copy())]:
   if label!='original': xx[:,loc]=mean[loc]
   xt=torch.from_numpy((xx-mean)/std);xyz=torch.from_numpy(s['mesh_xyz'].astype(np.float32));e=torch.from_numpy(s['edge_index'].astype(np.int64))
   with torch.no_grad(): p=torch.sigmoid(model(xt,xyz,e)).squeeze(-1).numpy()
   gp=apply_knn(p,s['mesh_to_grid_idx'],s['mesh_to_grid_weights']).reshape(target.shape);mask=af(gp>=.5,area)
   rec[label]={'iou':iou(mask,target,area),'ap':average_precision(gp,target),'pred_area_km2':float(area[mask].sum())}
  rows.append(rec)
summary={}
for label in ['original','location_mean_occluded']:
 vals_i=[r[label]['iou'] for r in rows if r[label]['iou'] is not None];vals_a=[r[label]['ap'] for r in rows]
 summary[label]={'mean_iou':float(np.mean(vals_i)),'sd_iou':float(np.std(vals_i,ddof=1)),'mean_ap':float(np.mean(vals_a)),'sd_ap':float(np.std(vals_a,ddof=1))}
summary['delta_occluded_minus_original']={'mean_iou':summary['location_mean_occluded']['mean_iou']-summary['original']['mean_iou'],'mean_ap':summary['location_mean_occluded']['mean_ap']-summary['original']['mean_ap']}
out={'status':'EXECUTED_CHECKPOINT_LOCATION_OCCLUSION_SENSITIVITY','meaning':'Inference-only sensitivity test. Explicit spherical position features are replaced by their training means; this is NOT a retrained no-location ablation.','rows':rows,'summary':summary}
(OUT/'LOCATION_FEATURE_OCCLUSION_RESULTS.json').write_text(json.dumps(out,indent=2));print(json.dumps(summary,indent=2))
