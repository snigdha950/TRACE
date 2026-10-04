"""Train/evaluate TRACE spherical GNN on prebuilt weather samples.

No FANI tuning is allowed. Model selection uses validation BCE only; the final
probability threshold is fixed at 0.5. Test metrics include the frozen FANI case
but are never used for training or early stopping.
"""
from __future__ import annotations
import argparse, json, math, time, csv, hashlib, platform, importlib.metadata
from pathlib import Path
import numpy as np
import torch
from torch import nn

from trace_spherical_gnn import TraceSphericalGNN, apply_knn
from trace_eval_metrics import binary_scores
from trace_object_verify import verify_objects
from trace_learned_evaluation import average_precision,localization

FANI="FANI_2019"
NO_MESSAGES=False
OUTPUT_DIR=None


def load_sample(path):
    d=np.load(path,allow_pickle=False)
    return {k:d[k] for k in d.files}


def event_split(sample):
    return str(sample["split"][0]),str(sample["event_id"][0])


def fit_scaler(samples):
    xs=np.concatenate([s["node_features"] for s in samples],axis=0).astype(float)
    mean=xs.mean(axis=0); std=xs.std(axis=0)
    std=np.where(std<1e-6,1.0,std)
    return mean.astype(np.float32),std.astype(np.float32)


def tensorize(sample,mean,std,device):
    x=(sample["node_features"].astype(np.float32)-mean)/std
    y=sample["target_clim99_mesh"].astype(np.float32)
    return (
        torch.from_numpy(x).to(device),torch.from_numpy(y).to(device),
        torch.from_numpy(sample["mesh_xyz"].astype(np.float32)).to(device),
        torch.from_numpy(sample["edge_index"].astype(np.int64)[:, :0] if NO_MESSAGES else sample["edge_index"].astype(np.int64)).to(device),
    )


def calc_pos_weight(samples):
    ys=np.concatenate([s["target_clim99_mesh"].reshape(-1) for s in samples])
    pos=float(ys.sum()); neg=float(len(ys)-pos)
    if pos<=0: return 1.0
    return float(np.clip(neg/pos,1.0,20.0))


def eval_loss(model,samples,mean,std,device,loss_fn):
    if not samples: return float("nan")
    model.eval(); vals=[]
    with torch.no_grad():
        for s in samples:
            x,y,xyz,e=tensorize(s,mean,std,device)
            vals.append(float(loss_fn(model(x,xyz,e),y).cpu()))
    return float(np.mean(vals))


def area_filter(mask,area,min_area=5000.0):
    from scipy import ndimage
    labels,n=ndimage.label(mask,structure=np.ones((3,3)))
    out=np.zeros_like(mask,bool)
    for i in range(1,n+1):
        m=labels==i
        if float(area[m].sum())>=min_area: out|=m
    return out


def evaluate_sample(model,s,mean,std,device,threshold=0.5,min_area=5000.0):
    model.eval()
    with torch.no_grad():
        x,y,xyz,e=tensorize(s,mean,std,device)
        p=torch.sigmoid(model(x,xyz,e)).squeeze(-1).cpu().numpy()
    grid_prob=apply_knn(p,s["mesh_to_grid_idx"],s["mesh_to_grid_weights"]).reshape(s["target_clim99_grid"].shape)
    pred=area_filter(grid_prob>=threshold,s["cell_area_km2"],min_area)
    lat=s["latitude"]; lon=s["longitude"]; area=s["cell_area_km2"]
    masks={
        "fixed_support_0p4":area_filter(s["member_support"]>=0.4,area,5000),
        "ensemble_mean_fixed":area_filter(s["ensemble_mean_speed"]>=17.5,area,5000),
        "efi_0p8":area_filter(s["raw_efi"]>=0.8,area,5000),
        "sot_gt0":area_filter(s["raw_sot"]>0,area,5000),
        "ensemble_mean_climate99":area_filter(s["ensemble_mean_speed"]>=s["climate_q99"],area,5000),
        "member_climate_support_0p4":area_filter(s["member_climate_support"]>=.4,area,5000),
        "no_message_ablation" if NO_MESSAGES else "spherical_gnn":pred,
    }
    rank_scores={"fixed_support_0p4":s["member_support"],"ensemble_mean_fixed":s["ensemble_mean_speed"],"efi_0p8":s["raw_efi"],"sot_gt0":s["raw_sot"],"ensemble_mean_climate99":s["ensemble_mean_speed"]-s["climate_q99"],"member_climate_support_0p4":s["member_climate_support"]}
    out={"event_id":str(s["event_id"][0]),"split":str(s["split"][0]),"lead_hours":int(s["lead_hours"][0]),"threshold":threshold,"methods":{}}
    for method,mask in masks.items():
        scores={}
        for name,target in [("clim99",s["target_clim99_grid"].astype(bool)),("fixed17p5",s["target_fixed_grid"].astype(bool))]:
            pix=binary_scores(mask,target)
            obj=verify_objects(mask,target,lat,lon,area,min_area_km2=min_area,centroid_gate_km=300.0)
            union=mask|target; aiou=float(area[mask&target].sum()/area[union].sum()) if union.any() else float("nan")
            scores[name]={**pix,**{k:v for k,v in obj.items() if k!="matches"},"area_weighted_iou":aiou,**localization(mask,target,s),"average_precision_threshold_free":average_precision(grid_prob if method in ["spherical_gnn","no_message_ablation"] else rank_scores[method],target),"predicted_area_km2":float(area[mask].sum()),"reference_area_km2":float(area[target].sum())}
        out["methods"][method]=scores
    sample_id=out['event_id']+'_'+str(out['lead_hours'])
    np.savez_compressed(OUTPUT_DIR/(sample_id+'_predictions.npz'),gnn_score=grid_prob,gnn_mask=pred,**{k:v for k,v in masks.items() if k not in ['spherical_gnn','no_message_ablation']})

    return out


def main():
    global NO_MESSAGES,OUTPUT_DIR
    start=time.perf_counter()
    ap=argparse.ArgumentParser()
    ap.add_argument("dataset_manifest",help="gnn_samples/dataset_manifest.json")
    ap.add_argument("--epochs",type=int,default=80)
    ap.add_argument("--patience",type=int,default=12)
    ap.add_argument("--hidden",type=int,default=64)
    ap.add_argument("--blocks",type=int,default=3)
    ap.add_argument("--lr",type=float,default=2e-3)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out",default="gnn_run")
    ap.add_argument("--no-messages",action="store_true")
    args=ap.parse_args()
    NO_MESSAGES=args.no_messages;OUTPUT_DIR=Path(args.out);OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    manifest=json.loads(Path(args.dataset_manifest).read_text())
    samples=[load_sample(x["path"]) for x in manifest["samples"]]
    train=[]; val=[]; test=[]
    for s in samples:
        split,event=event_split(s)
        if event==FANI and split!="test": raise SystemExit("FANI leakage detected")
        {"train":train,"validation":val,"test":test}[split].append(s)
    if not train or not val or not test: raise SystemExit("need non-empty train, validation, and test samples")

    ev_by_split={k:{event_split(x)[1] for x in v} for k,v in {"train":train,"validation":val,"test":test}.items()}
    for a,b in [("train","validation"),("train","test"),("validation","test")]:
        if ev_by_split[a]&ev_by_split[b]:raise SystemExit(f"event leakage {a}/{b}: {ev_by_split[a]&ev_by_split[b]}")
    # Mesh topology must be identical across samples for one model.
    n=train[0]["mesh_xyz"].shape[0]; e0=train[0]["edge_index"]
    for s in samples:
        if s["mesh_xyz"].shape[0]!=n or s["edge_index"].shape!=e0.shape or not np.array_equal(s["edge_index"],e0):
            raise SystemExit("sample mesh/topology mismatch")

    mean,std=fit_scaler(train)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=TraceSphericalGNN(in_features=train[0]["node_features"].shape[1],hidden=args.hidden,blocks=args.blocks).to(device)
    pos_weight=calc_pos_weight(train)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight],device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)

    best=math.inf;best_state=None; best_epoch=-1; patience=0; history=[]
    for epoch in range(args.epochs):
        model.train(); train_losses=[]
        order=np.random.permutation(len(train))
        for i in order:
            x,y,xyz,e=tensorize(train[i],mean,std,device)
            opt.zero_grad(); logits=model(x,xyz,e); loss=loss_fn(logits,y); loss.backward(); opt.step()
            train_losses.append(float(loss.detach().cpu()))
        vl=eval_loss(model,val,mean,std,device,loss_fn)
        history.append({"epoch":epoch,"train_bce":float(np.mean(train_losses)),"val_bce":vl})
        print(json.dumps(history[-1]),flush=True)
        if math.isfinite(vl) and vl<best-1e-5:
            best=vl; best_epoch=epoch; patience=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            patience+=1
            if patience>=args.patience: break
    if best_state is None:raise SystemExit("NO_FINITE_VALIDATION: test evaluation prohibited")
    model.load_state_dict(best_state)

    results=[evaluate_sample(model,s,mean,std,device,0.5) for s in test]
    outdir=Path(args.out); outdir.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"feature_mean":mean,"feature_std":std,"feature_names":train[0]["feature_names"].tolist(),"best_epoch":best_epoch,"seed":args.seed},outdir/"trace_spherical_gnn.pt")
    summary={
        "status":"EXECUTED_REAL_WEATHER","no_messages":NO_MESSAGES,"runtime_seconds":time.perf_counter()-start,"model_parameter_count":sum(p.numel() for p in model.parameters()),"device":str(device),"seed":args.seed,"best_epoch":best_epoch,
        "best_validation_bce":best,"fixed_probability_threshold":0.5,
        "train_samples":len(train),"validation_samples":len(val),"test_samples":len(test),
        "fani_test_only":any(event_split(x)[1]==FANI for x in samples) and all(event_split(x)[0]=="test" for x in samples if event_split(x)[1]==FANI),"pos_weight":pos_weight,"history":history,"test_results":results,
        "claim_gate":"No superiority claim unless held-out metrics beat or materially complement frozen baselines."
    }
    (outdir/"gnn_metrics.json").write_text(json.dumps(summary,indent=2))
    rows=[]
    for r in results:
        for method,ms in r["methods"].items():
            for target,metrics in ms.items():rows.append({"event_id":r["event_id"],"lead_hours":r["lead_hours"],"method":method,"target":target,**metrics})
    with (outdir/"gnn_comparison.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    manifest={"sample_sha256":{x["path"]:hashlib.sha256(Path(x["path"]).read_bytes()).hexdigest() for x in manifest["samples"]},"timestamp_utc":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),"dependency_code_sha256":{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ["trace_spherical_gnn.py","trace_learned_evaluation.py","trace_object_verify.py"]},"seed":args.seed,"config":vars(args),"code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"dataset_manifest_sha256":hashlib.sha256(Path(args.dataset_manifest).read_bytes()).hexdigest(),"packages":{p:importlib.metadata.version(p) for p in ["torch","numpy","scipy"]},"hardware":platform.platform(),"training_events":sorted({str(s["event_id"][0]) for s in train}),"validation_events":sorted({str(s["event_id"][0]) for s in val}),"test_events":sorted({str(s["event_id"][0]) for s in test}),"uncertainty":"class-weighted discriminator score; NOT calibrated probability"}
    (outdir/"RUN_MANIFEST.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ["history","test_results"]},indent=2))

if __name__=="__main__": main()
