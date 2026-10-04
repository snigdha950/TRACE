#!/usr/bin/env python3
from pathlib import Path
import json, shutil, subprocess, sys, numpy as np, hashlib
HERE=Path(__file__).resolve().parent
PROTO=json.loads((HERE/'STAGE1_FIVE_SEED_PROTOCOL.json').read_text())
MAN=json.loads((HERE/'PORTABLE_DATASET_MANIFEST.json').read_text())
REMOVE={'sin_lat','cos_lat','sin_lon','cos_lon'}
FULL=HERE/'retrained_fullfeature_nomessage_seed1'
NOLOC=HERE/'retrained_no_location_nomessage_seed1'
TMP=HERE/'_tmp_no_location_seed1'
OUT=HERE/'RETRAINED_NO_EXPLICIT_LOCATION_NOMESSAGE_SEED1.json'

def run(manifest,out):
    if out.exists(): shutil.rmtree(out)
    cmd=[sys.executable,str(HERE/'trace_gnn_train_eval.py'),str(manifest),'--epochs',str(PROTO['epochs']),'--patience',str(PROTO['patience']),'--hidden',str(PROTO['hidden']),'--blocks',str(PROTO['blocks']),'--lr',str(PROTO['learning_rate']),'--seed','1','--out',str(out),'--no-messages']
    cp=subprocess.run(cmd,cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    out.mkdir(exist_ok=True); (out/'training_stdout.log').write_text(cp.stdout)
    if cp.returncode: raise SystemExit(cp.returncode)

def metrics(out):
    d=json.loads((out/'gnn_metrics.json').read_text()); rows=[]
    for r in sorted(d['test_results'],key=lambda x:x['lead_hours']):
        m=r['methods']['no_message_ablation']['clim99']
        rows.append({'lead_hours':int(r['lead_hours']),'area_weighted_iou':float(m['area_weighted_iou']),'average_precision':float(m['average_precision_threshold_free'])})
    return {'mean_iou':float(np.mean([r['area_weighted_iou'] for r in rows])),'mean_ap':float(np.mean([r['average_precision'] for r in rows])),'rows':rows,'best_epoch':d['best_epoch'],'best_validation_bce':d['best_validation_bce']}

def main():
    run(HERE/'PORTABLE_DATASET_MANIFEST.json',FULL)
    if TMP.exists(): shutil.rmtree(TMP)
    TMP.mkdir()
    derived=[]
    for row in MAN['samples']:
        sp=HERE/row['path']
        with np.load(sp,allow_pickle=False) as z:d={k:z[k] for k in z.files}
        names=[str(x) for x in d['feature_names'].tolist()]; keep=[i for i,n in enumerate(names) if n not in REMOVE]
        d['node_features']=d['node_features'][:,keep].astype(np.float32); d['feature_names']=np.array([names[i] for i in keep])
        dp=TMP/sp.name; np.savez_compressed(dp,**d)
        derived.append({**row,'path':str(dp)})
    m={**MAN,'samples':derived,'derivation':'removed sin_lat, cos_lat, sin_lon, cos_lon only; spherical mesh geometry retained'}
    mp=TMP/'dataset_manifest.json';mp.write_text(json.dumps(m,indent=2))
    try: run(mp,NOLOC)
    finally: shutil.rmtree(TMP,ignore_errors=True)
    full,no=metrics(FULL),metrics(NOLOC)
    o={'status':'PASS_MATCHED_RETRAINED_NO_EXPLICIT_LOCATION_DIAGNOSTIC','seed':1,'model':'no_message_ablation','removed_features':sorted(REMOVE),'retained_geometry':'spherical mesh_xyz and relative edge geometry','full_features':full,'without_explicit_location_features':no,'delta_no_location_minus_full':{'mean_iou':no['mean_iou']-full['mean_iou'],'mean_ap':no['mean_ap']-full['mean_ap']},'scope':'single-seed matched diagnostic; not a five-seed full-GNN geographic-generalization proof'}
    OUT.write_text(json.dumps(o,indent=2)); print(json.dumps(o,indent=2))
if __name__=='__main__':main()
