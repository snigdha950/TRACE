#!/usr/bin/env python3
"""
TRACE same-feature non-graph learned baseline.

Purpose:
Test whether graph message passing adds value beyond the prepared feature set.

Uses:
- identical gnn_samples dataset
- identical train/validation/test event splits
- training-only normalization
- validation BCE only for model selection
- fixed probability threshold 0.5
- identical grid remapping, 5000 km2 area filter, and object verification
- FANI test-only guard

No sklearn dependency. This is a single linear sigmoid classifier in PyTorch.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.metadata, json, math, platform, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

from trace_spherical_gnn import apply_knn
from trace_eval_metrics import binary_scores
from trace_object_verify import verify_objects
from trace_gnn_train_eval import area_filter

FANI="FANI_2019"

def load(path):
    d=np.load(path,allow_pickle=False)
    return {k:d[k] for k in d.files}

def split_event(s):
    return str(s["split"][0]), str(s["event_id"][0])

def scaler(samples):
    x=np.concatenate([s["node_features"] for s in samples],axis=0).astype(np.float32)
    m=x.mean(0); sd=x.std(0); sd=np.where(sd<1e-6,1.0,sd)
    return m.astype(np.float32),sd.astype(np.float32)

def pos_weight(samples):
    y=np.concatenate([s["target_clim99_mesh"].reshape(-1) for s in samples]).astype(float)
    p=float(y.sum()); n=float(len(y)-p)
    return float(np.clip(n/p,1,20)) if p>0 else 1.0

from trace_learned_evaluation import average_precision,localization

def eval_one(model,s,m,sd):
    x=torch.from_numpy(((s["node_features"].astype(np.float32)-m)/sd))
    with torch.no_grad():
        p=torch.sigmoid(model(x)).squeeze(-1).numpy()
    prob=apply_knn(p,s["mesh_to_grid_idx"],s["mesh_to_grid_weights"]).reshape(s["target_clim99_grid"].shape)
    area=s["cell_area_km2"]; lat=s["latitude"]; lon=s["longitude"]
    pred=area_filter(prob>=0.5,area,5000)
    out={"event_id":str(s["event_id"][0]),"split":str(s["split"][0]),
         "lead_hours":int(s["lead_hours"][0]),"method":"linear_same_features","targets":{}}
    for name,target in [("clim99",s["target_clim99_grid"].astype(bool)),
                        ("fixed17p5",s["target_fixed_grid"].astype(bool))]:
        pix=binary_scores(pred,target)
        obj=verify_objects(pred,target,lat,lon,area,min_area_km2=5000,centroid_gate_km=300.0)
        union=pred|target
        aiou=float(area[pred&target].sum()/area[union].sum()) if union.any() else float("nan")
        out["targets"][name]={**pix,**{k:v for k,v in obj.items() if k!="matches"},**localization(pred,target,s),
                              "area_weighted_iou":aiou,
                              "average_precision_threshold_free":average_precision(prob,target),
                              "predicted_area_km2":float(area[pred].sum()),
                              "reference_area_km2":float(area[target].sum())}
    return out,prob,pred

def main():
    started=time.perf_counter()
    ap=argparse.ArgumentParser()
    ap.add_argument("dataset_manifest")
    ap.add_argument("--epochs",type=int,default=80)
    ap.add_argument("--patience",type=int,default=12)
    ap.add_argument("--lr",type=float,default=2e-3)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out",default="linear_run")
    ap.add_argument("--threads",type=int,default=4)
    a=ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads)
    manifest=json.loads(Path(a.dataset_manifest).read_text())
    samples=[load(x["path"]) for x in manifest["samples"]]
    train=[]; val=[]; test=[]
    for s in samples:
        sp,ev=split_event(s)
        if ev==FANI and sp!="test": raise SystemExit("FANI leakage detected")
        {"train":train,"validation":val,"test":test}[sp].append(s)
    if not train or not val or not test: raise SystemExit("need non-empty train/validation/test")
    ev_by_split={k:{split_event(x)[1] for x in v} for k,v in
                 {"train":train,"validation":val,"test":test}.items()}
    for a_,b_ in [("train","validation"),("train","test"),("validation","test")]:
        both=ev_by_split[a_]&ev_by_split[b_]
        if both: raise SystemExit(f"event leakage between {a_} and {b_}: {sorted(both)}")
    m,sd=scaler(train); pw=pos_weight(train)
    D=train[0]["node_features"].shape[1]
    model=nn.Linear(D,1)
    lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    best=math.inf; state=None; stale=0; hist=[]; best_epoch=-1
    for epoch in range(a.epochs):
        model.train(); losses=[]
        for i in np.random.permutation(len(train)):
            s=train[i]
            x=torch.from_numpy(((s["node_features"].astype(np.float32)-m)/sd))
            y=torch.from_numpy(s["target_clim99_mesh"].astype(np.float32))
            opt.zero_grad(); loss=lossfn(model(x),y); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        model.eval(); vv=[]
        with torch.no_grad():
            for s in val:
                x=torch.from_numpy(((s["node_features"].astype(np.float32)-m)/sd))
                y=torch.from_numpy(s["target_clim99_mesh"].astype(np.float32))
                vv.append(float(lossfn(model(x),y)))
        vl=float(np.mean(vv)); hist.append({"epoch":epoch,"train_bce":float(np.mean(losses)),"val_bce":vl})
        print(json.dumps(hist[-1]),flush=True)
        if math.isfinite(vl) and vl < best-1e-5:
            best=vl; best_epoch=epoch; stale=0
            state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=a.patience: break
    if state is None: raise SystemExit("validation loss never finite; refusing to evaluate untrained model")
    model.load_state_dict(state)
    out=Path(a.out); out.mkdir(exist_ok=True,parents=True)
    results=[]; rows=[]
    for s in test:
        r,prob,pred=eval_one(model,s,m,sd); results.append(r)
        sid=r["event_id"]+"_"+str(r["lead_hours"])
        np.savez_compressed(out/(sid+"_predictions.npz"),score=prob,mask=pred)
        for target,met in r["targets"].items():
            rows.append({"event_id":r["event_id"],"lead_hours":r["lead_hours"],
                         "method":"linear_same_features","target":target,**met})
    torch.save({"state_dict":model.state_dict(),"feature_mean":m,"feature_std":sd,
                "best_epoch":best_epoch,"seed":a.seed},out/"trace_linear_baseline.pt")
    summary={"status":"EXECUTED_REAL_WEATHER","model":"single linear sigmoid classifier",
             "runtime_seconds":time.perf_counter()-started,"same_features_as_gnn":True,"threshold":0.5,"seed":a.seed,
             "best_validation_bce":best,"best_epoch":best_epoch,
             "train_events":sorted({split_event(s)[1] for s in train}),
             "validation_events":sorted({split_event(s)[1] for s in val}),
             "test_events":sorted({split_event(s)[1] for s in test}),
             "fani_test_only":any(split_event(x)[1]==FANI for x in samples) and all(split_event(x)[0]=="test" for x in samples if split_event(x)[1]==FANI),"history":hist,"test_results":results}
    (out/"linear_metrics.json").write_text(json.dumps(summary,indent=2))
    with (out/"linear_comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    runman={"timestamp_utc":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),"config":vars(a),"dependency_code_sha256":{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ["trace_gnn_train_eval.py","trace_spherical_gnn.py","trace_learned_evaluation.py","trace_object_verify.py"]},"code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dataset_manifest_sha256":hashlib.sha256(Path(a.dataset_manifest).read_bytes()).hexdigest(),
            "packages":{p:importlib.metadata.version(p) for p in ["torch","numpy","scipy"]},
            "hardware":platform.platform(),"seed":a.seed,"pos_weight":pw,
            "n_train_samples":len(train),"n_val_samples":len(val),"n_test_samples":len(test),
            "sample_sha256":{x["path"]:hashlib.sha256(Path(x["path"]).read_bytes()).hexdigest()
                             for x in manifest["samples"]}}
    (out/"RUN_MANIFEST.json").write_text(json.dumps(runman,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ["history","test_results"]},indent=2))
if __name__=="__main__":
    main()
