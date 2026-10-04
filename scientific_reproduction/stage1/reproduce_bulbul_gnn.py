from pathlib import Path
import json, numpy as np, torch
from scipy import ndimage
from trace_spherical_gnn import TraceSphericalGNN, apply_knn
from trace_learned_evaluation import average_precision
HERE=Path(__file__).resolve().parent; SRC=HERE/'bulbul_frozen'
def area_filter(mask,area,min_area=5000.0):
 lab,n=ndimage.label(mask,structure=np.ones((3,3)));out=np.zeros_like(mask,bool)
 for i in range(1,n+1):
  m=lab==i
  if float(area[m].sum())>=min_area:out|=m
 return out
def iou(mask,target,area):
 u=mask|target
 return float(area[mask&target].sum()/area[u].sum()) if u.any() else None
rows=[]
for seed in range(1,6):
 ck=torch.load(SRC/f'gnn_seed{seed}.pt',map_location='cpu',weights_only=False)
 names=list(ck.get('feature_names',[]));n=len(ck['feature_mean']); model=TraceSphericalGNN(n,32,2);model.load_state_dict(ck['state_dict']);model.eval()
 for lead in (72,96,120):
  s={k:v for k,v in np.load(SRC/f'BULBUL_2019_{lead}.npz',allow_pickle=False).items()};x=(s['node_features'].astype(np.float32)-ck['feature_mean'])/ck['feature_std']
  with torch.no_grad():p=torch.sigmoid(model(torch.from_numpy(x),torch.from_numpy(s['mesh_xyz'].astype(np.float32)),torch.from_numpy(s['edge_index'].astype(np.int64)))).squeeze(-1).numpy()
  gp=apply_knn(p,s['mesh_to_grid_idx'],s['mesh_to_grid_weights']).reshape(s['target_clim99_grid'].shape);mask=area_filter(gp>=.5,s['cell_area_km2']);target=s['target_clim99_grid'].astype(bool)
  rows.append({'seed':seed,'lead_hours':lead,'valid_time':str(s['valid_time'][0]),'area_weighted_iou':iou(mask,target,s['cell_area_km2']),'average_precision':average_precision(gp,target)})
summary={'mean_iou':float(np.mean([r['area_weighted_iou'] for r in rows])),'mean_ap':float(np.mean([r['average_precision'] for r in rows])),'lead_means':{str(l):float(np.mean([r['area_weighted_iou'] for r in rows if r['lead_hours']==l])) for l in (72,96,120)},'independent_verification_times':len(set(r['valid_time'] for r in rows)),'rows':rows}
out={'status':'PASS' if abs(summary['mean_iou']-0.42979942438385327)<1e-9 and abs(summary['mean_ap']-0.6069505476562604)<1e-9 else 'MISMATCH','expected':{'mean_iou':0.42979942438385327,'mean_ap':0.6069505476562604},'recomputed':summary,'interpretation':'5 frozen seeds x 3 correlated forecast leads, all verifying one BULBUL valid time; not 15 independent weather cases.'}
(HERE/'BULBUL_GNN_REPRODUCTION.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='recomputed'},indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2));raise SystemExit(0 if out['status']=='PASS' else 1)
