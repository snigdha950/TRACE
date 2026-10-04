"""Train/evaluate TRACE controlled 12->5 km Stage-2 benchmark.

Input dataset NPZ convention:
  coarse: [N,C,Hc,Wc]  exactly 12-km controlled inputs
  fine:   [N,C,Hf,Wf]  exactly 5-km controlled targets
  split:  [N] strings in {train,validation,test}
  sample_id/event_id optional strings

The script compares bilinear interpolation, a deterministic mean network, and
the tiny residual-diffusion prototype. It reports tail, footprint, PSD, scale
consistency, and divergence/vorticity diagnostics on held-out test samples.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from trace_resdiff_tiny import TinyResidualDiffusion
from trace_spectral_tail_metrics import tail_metrics, spectral_slope, coarse_consistency_error
from trace_physics_metrics import gradient_consistency


def load_dataset(path):
    d=np.load(path,allow_pickle=False)
    coarse=np.asarray(d["coarse"],np.float32); fine=np.asarray(d["fine"],np.float32)
    split=np.asarray(d["split"]).astype(str)
    if coarse.ndim!=4 or fine.ndim!=4 or len(coarse)!=len(fine) or len(split)!=len(coarse):
        raise ValueError("dataset shapes invalid")
    if coarse.shape[1]!=2 or fine.shape[1]!=2:
        raise ValueError("current Stage-2 benchmark expects U/V channels")
    ids=np.asarray(d["sample_id"]).astype(str) if "sample_id" in d.files else np.array([f"sample_{i}" for i in range(len(coarse))])
    return coarse,fine,split,ids


def speed(x):
    return np.hypot(x[:,0],x[:,1])


def fit_channel_scaler(coarse,fine,idx):
    # One physical scaling fitted only on training data.
    vals=np.concatenate([coarse[idx].transpose(1,0,2,3).reshape(2,-1),fine[idx].transpose(1,0,2,3).reshape(2,-1)],axis=1)
    mean=vals.mean(axis=1).astype(np.float32); std=vals.std(axis=1).astype(np.float32)
    std=np.where(std<1e-6,1.0,std).astype(np.float32)
    return mean,std


def norm(x,mean,std): return (x-mean[None,:,None,None])/std[None,:,None,None]
def denorm(x,mean,std): return x*std[None,:,None,None]+mean[None,:,None,None]


def train_model(cn,fn,train_idx,val_idx,epochs,patience,seed,device,lr):
    torch.manual_seed(seed); np.random.seed(seed)
    model=TinyResidualDiffusion(channels=2,steps=12).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
    best=float("inf"); best_state=None; best_epoch=-1; stale=0; history=[]
    target_size=fn.shape[-2:]
    for epoch in range(epochs):
        model.train(); losses=[]
        for i in np.random.permutation(train_idx):
            c=torch.from_numpy(cn[i:i+1]).to(device); f=torch.from_numpy(fn[i:i+1]).to(device)
            opt.zero_grad()
            # Train deterministic mean and diffusion jointly, but keep an explicit mean loss.
            dloss,mean=model.diffusion_loss(c,f)
            mloss=F.mse_loss(mean,f)
            # Scale consistency in normalized space, using area interpolation as a differentiable proxy.
            down=F.interpolate(mean,size=c.shape[-2:],mode="area")
            closs=F.l1_loss(down,c)
            loss=dloss+0.30*mloss+0.10*closs
            loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        model.eval(); vals=[]
        with torch.no_grad():
            for i in val_idx:
                c=torch.from_numpy(cn[i:i+1]).to(device); f=torch.from_numpy(fn[i:i+1]).to(device)
                mean=model.mean_model(c,target_size)
                vals.append(float(F.mse_loss(mean,f).cpu()))
        vl=float(np.mean(vals))
        history.append({"epoch":epoch,"train_objective":float(np.mean(losses)),"val_mean_mse":vl})
        if vl<best-1e-6:
            best=vl; best_epoch=epoch; stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=patience: break
    model.load_state_dict(best_state)
    return model,best,best_epoch,history


def method_metrics(pred_uv,truth_uv,coarse_uv,threshold):
    pred_sp=np.hypot(pred_uv[0],pred_uv[1]); truth_sp=np.hypot(truth_uv[0],truth_uv[1])
    out=tail_metrics(pred_sp,truth_sp,threshold=threshold)
    out["spectral_slope_pred"]=spectral_slope(pred_sp,dx_km=5.0,dy_km=5.0)
    out["spectral_slope_truth"]=spectral_slope(truth_sp,dx_km=5.0,dy_km=5.0)
    out["spectral_slope_abs_error"]=abs(out["spectral_slope_pred"]-out["spectral_slope_truth"])
    cc_u=coarse_consistency_error(pred_uv[0],coarse_uv[0]); cc_v=coarse_consistency_error(pred_uv[1],coarse_uv[1])
    out["coarse_u_mae"]=cc_u["coarse_mae"]; out["coarse_v_mae"]=cc_v["coarse_mae"]
    out.update(gradient_consistency(pred_uv[0],pred_uv[1],truth_uv[0],truth_uv[1],dx_m=5000.0,dy_m=5000.0))
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--epochs",type=int,default=60)
    ap.add_argument("--patience",type=int,default=10)
    ap.add_argument("--lr",type=float,default=2e-3)
    ap.add_argument("--samples",type=int,default=4,help="diffusion samples per test input")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out",default="stage2_run")
    args=ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(max(1,min(4,torch.get_num_threads())))

    coarse,fine,split,ids=load_dataset(args.dataset)
    train_idx=np.where(split=="train")[0]; val_idx=np.where(split=="validation")[0]; test_idx=np.where(split=="test")[0]
    if not len(train_idx) or not len(val_idx) or not len(test_idx): raise SystemExit("need train/validation/test splits")
    mean,std=fit_channel_scaler(coarse,fine,train_idx)
    cn=norm(coarse,mean,std).astype(np.float32); fn=norm(fine,mean,std).astype(np.float32)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,best,best_epoch,history=train_model(cn,fn,train_idx,val_idx,args.epochs,args.patience,args.seed,device,args.lr)

    # Threshold derived from TRAINING truth only, never from held-out test.
    train_speed=speed(fine[train_idx])
    threshold=float(np.quantile(train_speed,0.99))
    rows=[]
    model.eval()
    for i in test_idx:
        c=torch.from_numpy(cn[i:i+1]).to(device)
        with torch.no_grad():
            bil=F.interpolate(c,size=fn.shape[-2:],mode="bilinear",align_corners=False).cpu().numpy()
            det=model.mean_model(c,fn.shape[-2:]).cpu().numpy()
            samples=np.stack([model.sample(c,fn.shape[-2:],seed=int(args.seed+100*int(i)+j)).cpu().numpy()[0] for j in range(args.samples)],axis=0)
        bil=denorm(bil,mean,std)[0]; det=denorm(det,mean,std)[0]
        sample_uv=denorm(samples,mean,std)
        diff_mean=sample_uv.mean(axis=0)
        truth=fine[i]; c_phys=coarse[i]
        for method,pred in [("bilinear",bil),("deterministic_mean",det),("residual_diffusion_mean",diff_mean)]:
            rows.append({"sample_id":ids[i],"method":method,**method_metrics(pred,truth,c_phys,threshold)})
        # Sample spread is evidence of conditional variability, not a new NWP ensemble.
        rows[-1]["diffusion_sample_count"]=args.samples
        rows[-1]["diffusion_speed_spread_mean"]=float(np.std(np.hypot(sample_uv[:,0],sample_uv[:,1]),axis=0).mean())

    outdir=Path(args.out); outdir.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"channel_mean":mean,"channel_std":std,"seed":args.seed,"best_epoch":best_epoch},outdir/"trace_resdiff.pt")
    summary={"status":"EXECUTED","device":str(device),"seed":args.seed,"best_epoch":best_epoch,"best_validation_mean_mse":best,
             "training_q99_threshold_ms":threshold,"train_samples":len(train_idx),"validation_samples":len(val_idx),"test_samples":len(test_idx),
             "diffusion_samples_per_test":args.samples,"history":history,"test_metrics":rows,
             "guardrails":["Controlled 12->5 km benchmark only unless genuine operational inputs are used.","Diffusion samples from one parent input are not independent atmospheric initial-condition ensemble members.","Amplitude-preserving claim requires held-out tail/PSD advantage over deterministic baselines."]}
    (outdir/"stage2_metrics.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
