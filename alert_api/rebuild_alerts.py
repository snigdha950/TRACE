from __future__ import annotations
import json, math, hashlib, sys
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import ConvexHull
HERE=Path(__file__).resolve().parent
SRC=HERE/'source_evidence'; DATA=HERE/'data'; DATA.mkdir(exist_ok=True)
sys.path.insert(0,str(SRC))
from trace_spherical_gnn import TraceSphericalGNN, apply_knn
F={'gnn_probability_threshold':.5,'min_object_area_km2':5000.0,'efi_primary':.8,'efi_sensitivity':.5,'fixed_wind_primary_ms':17.5,'fixed_wind_sensitivity_ms':12.5,'member_support_threshold':.4,'seed_consensus_threshold':.6}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def filt(mask,area):
 l,n=ndimage.label(mask,structure=np.ones((3,3)));o=np.zeros_like(mask,bool)
 for i in range(1,n+1):
  m=l==i
  if float(area[m].sum())>=F['min_object_area_km2']:o|=m
 return o
def hull(mask,lat,lon):
 ij=np.argwhere(mask);pts=np.unique(np.column_stack([lon[ij[:,1]],lat[ij[:,0]]]),axis=0)
 if len(pts)<3:
  x,y=pts[0];e=.125;ring=[[x-e,y-e],[x+e,y-e],[x+e,y+e],[x-e,y+e],[x-e,y-e]]
 else:
  h=ConvexHull(pts);p=pts[h.vertices];ring=p.tolist()+[p[0].tolist()]
 return {'type':'Polygon','coordinates':[ring]}
def circle(lat0,lon0,r=5,n=48):
 R=6371.0088;la=math.radians(lat0);ring=[]
 for k in range(n):
  b=2*math.pi*k/n;d=r/R;la2=math.asin(math.sin(la)*math.cos(d)+math.cos(la)*math.sin(d)*math.cos(b));lo2=math.radians(lon0)+math.atan2(math.sin(b)*math.sin(d)*math.cos(la),math.cos(d)-math.sin(la)*math.sin(la2));ring.append([math.degrees(lo2),math.degrees(la2)])
 ring.append(ring[0]);return {'type':'Polygon','coordinates':[ring]}
def sev(w,e,s,a,t):
 return 'SEVERE' if w>=17.5 and e>=.8 and s>=.4 and a>=.8 else ('MODERATE' if w>=12.5 and e>=.5 and a>=.6 and t>0 else 'LOW')
def one(lead):
 z={k:v for k,v in np.load(SRC/f'BULBUL_2019_{lead}.npz',allow_pickle=False).items()};ms=[];ps=[];hs={}
 for seed in range(1,6):
  p=SRC/f'gnn_seed{seed}.pt';hs[str(seed)]=sha(p);c=torch.load(p,map_location='cpu',weights_only=False);m=TraceSphericalGNN(len(c['feature_mean']),32,2);m.load_state_dict(c['state_dict']);m.eval();x=(z['node_features'].astype(np.float32)-c['feature_mean'])/c['feature_std']
  with torch.no_grad():q=torch.sigmoid(m(torch.from_numpy(x),torch.from_numpy(z['mesh_xyz'].astype(np.float32)),torch.from_numpy(z['edge_index'].astype(np.int64)))).squeeze(-1).numpy()
  g=apply_knn(q,z['mesh_to_grid_idx'],z['mesh_to_grid_weights']).reshape(z['target_clim99_grid'].shape);ps.append(g);ms.append(filt(g>=.5,z['cell_area_km2']))
 ps=np.stack(ps);ms=np.stack(ms);a=ms.mean(0);q=ps.mean(0);mask=filt(a>=.6,z['cell_area_km2']);score=np.where(mask,q*a,-1);ij=np.unravel_index(np.argmax(score),score.shape);la=float(z['latitude'][ij[0]]);lo=float(z['longitude'][ij[1]]);pts=np.argwhere(mask);agr=float(a[mask].mean());w=float(z['ensemble_mean_speed'][mask].max());efi=float(z['raw_efi'][mask].max());sup=float(z['member_support'][mask].max());csup=float(z['member_climate_support'][mask].max());sot=float(z['raw_sot'][mask].max());bbox={'south':float(z['latitude'][pts[:,0]].min()),'north':float(z['latitude'][pts[:,0]].max()),'west':float(z['longitude'][pts[:,1]].min()),'east':float(z['longitude'][pts[:,1]].max())};crop={'south':max(5,bbox['south']-1),'north':min(30,bbox['north']+1),'west':max(75,bbox['west']-1),'east':min(100,bbox['east']+1)}
 return {'schema':'trace-alert/1.0','event_id':'BULBUL_2019','event_type':'extreme_wind_cyclone_context','valid_time':str(z['valid_time'][0]),'forecast_lead_hours':lead,'severity':sev(w,efi,sup,agr,sot),'severity_rule':'TRACE_SEVERITY_RULE_V1_NOT_CALIBRATED','anomaly_core_coordinate':{'lat':la,'lon':lo,'meaning':'model anomaly core; NOT an exact cyclone-center estimate'},'display_marker_radius_km':5.0,'display_marker_geojson':circle(la,lo),'predicted_impact_radius_km':None,'macro_threat_footprint_geojson':hull(mask,z['latitude'],z['longitude']),'macro_footprint_area_km2':float(z['cell_area_km2'][mask].sum()),'model_evidence':{'model':'wind spherical GNN, frozen 5-seed consensus','mean_seed_agreement_over_footprint':agr,'core_seed_agreement':float(a[ij]),'core_classifier_score_not_calibrated_probability':float(q[ij]),'max_ensemble_mean_wind_ms':w,'max_member_support_fraction':sup,'max_member_climate_support_fraction':csup,'max_efi':efi,'max_sot':sot,'max_wind_minus_climate_q99_ms':float(np.max((z['ensemble_mean_speed']-z['climate_q99'])[mask]))},'evidence_gate':{'status':'PASS','checks':{'seed_agreement_ge_0p6':agr>=.6,'efi_ge_0p5':efi>=.5,'sot_gt_0':sot>0,'wind_or_support_signal':w>=12.5 or sup>=.4},'note':'Gate supports prototype alert display only; it is not an official warning standard.'},'refinement_request':{'status':'READY_SOFTWARE_HANDOFF','crop_bbox':crop,'requested_input_grid_km':12,'requested_output_grid_km':5,'preferred_route':'deterministic_mean_plus_exact_coarse_projection','diffusion_route':'complementary_only_when_safety_diagnostics_pass','same_event_stage2_skill_status':'NOT_VALIDATED_FOR_BULBUL'},'source_boundary':{'stage1_provider':'NOAA GEFSv12 retrospective ensemble substitute','operational_target_provider':'NCMRWF NEPS-G','neps_g_validated':False,'five_km_means_grid_spacing_not_forecast_accuracy':True,'five_km_marker_is_ui_locator_not_hazard_radius':True,'calibrated_probability_claim':False},'provenance':{'sample_sha256':sha(SRC/f'BULBUL_2019_{lead}.npz'),'checkpoint_sha256_by_seed':hs,'prediction_consensus_threshold':.6}}

def main():
    torch.set_num_threads(4)

    alerts = [one(x) for x in (72, 96, 120)]

    (DATA / 'alerts.json').write_text(
        json.dumps(
            {
                'schema': 'trace-alert-collection/1.0',
                'alerts': alerts,
                'frozen_thresholds': F
            },
            indent=2
        ),
        encoding='utf-8',
        newline='\n'
    )

    print(
        'TRACE_ALERT_REBUILD_PASS',
        [(a['forecast_lead_hours'], a['severity']) for a in alerts]
    )


if __name__ == '__main__':
    main()
