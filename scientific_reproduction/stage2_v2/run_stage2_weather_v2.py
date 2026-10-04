"""Leakage-safe controlled HRRR experiment with correct residual DDPM prior.
Not CorrDiff reproduction, not operational forecast validation.
"""
import json,time,hashlib,math,csv,platform,importlib.metadata
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from trace_resdiff_tiny import MeanDownscaler,ResidualDenoiser
from trace_spectral_tail_metrics import _overlap_matrix,radial_power_spectrum,spectral_slope
from trace_physics_metrics import gradient_consistency
HERE=Path(__file__).resolve().parent

class ResidualDDPM(nn.Module):
 def __init__(self,mean_model,residual_scale,steps=32):
  super().__init__();self.mean_model=mean_model;self.denoiser=ResidualDenoiser(2);self.steps=steps
  t=torch.linspace(0,1,steps+1);abar=torch.cos(((t+.008)/1.008)*math.pi/2)**2;abar=abar/abar[0]
  beta=(1-abar[1:]/abar[:-1]).clamp(.0001,.999);alpha=1-beta;ab=torch.cumprod(alpha,dim=0);prev=torch.cat([torch.ones(1),ab[:-1]])
  self.register_buffer('beta',beta);self.register_buffer('alpha',alpha);self.register_buffer('abar',ab);self.register_buffer('posterior_variance',beta*(1-prev)/(1-ab));self.register_buffer('residual_scale',torch.tensor(residual_scale,dtype=torch.float32).reshape(1,2,1,1))
  for p in self.mean_model.parameters():p.requires_grad_(False)
 def loss(self,c,f,t=None,eps=None):
  with torch.no_grad():mean=self.mean_model(c,f.shape[-2:]);r=(f-mean)/self.residual_scale
  if t is None:t=torch.randint(0,self.steps,(len(c),),device=c.device)
  if eps is None:eps=torch.randn_like(r)
  a=self.abar[t].reshape(-1,1,1,1);noisy=torch.sqrt(a)*r+torch.sqrt(1-a)*eps
  return F.mse_loss(self.denoiser(noisy,mean,t.float()/(self.steps-1)),eps)
 @torch.no_grad()
 def sample(self,c,size,seed):
  gen=torch.Generator(device=c.device).manual_seed(seed);mean=self.mean_model(c,size);x=torch.randn(mean.shape,generator=gen,device=c.device)
  for ti in reversed(range(self.steps)):
   eps=self.denoiser(x,mean,torch.full((len(c),),ti/(self.steps-1),device=c.device))
   x=(x-self.beta[ti]/torch.sqrt(1-self.abar[ti])*eps)/torch.sqrt(self.alpha[ti])
   if ti:x+=torch.sqrt(self.posterior_variance[ti])*torch.randn(x.shape,generator=gen,device=c.device)
  return mean+self.residual_scale*x

def operators():
 wy=_overlap_matrix(48,20);wx=wy.copy();return wy,wx,np.linalg.inv(wy@wy.T),np.linalg.inv(wx@wx.T)
WY,WX,IY,IX=operators()
def aggregate(x):return np.stack([WY@ch@WX.T for ch in x])
def project(x,c):
 return x+np.stack([WY.T@IY@(cc-WY@ff@WX.T)@IX@WX for ff,cc in zip(x,c)])
def metrics(p,t,c,threshold):
 ps=np.hypot(p[0],p[1]);ts=np.hypot(t[0],t[1]);err=ps-ts;a=ps>=threshold;b=ts>=threshold
 tp=int((a&b).sum());fp=int((a&~b).sum());fn=int((~a&b).sum());union=tp+fp+fn
 k,power=radial_power_spectrum(ps,5,5);_,truth=radial_power_spectrum(ts,5,5)
 mask=np.isfinite(power)&np.isfinite(truth)&(power>0)&(truth>0)
 unresolved=mask&(k>1/24)&(k<=1/10)
 out={'mae_ms':float(np.mean(np.abs(err))),'rmse_ms':float(np.sqrt(np.mean(err**2))),'uv_rmse_ms':float(np.sqrt(np.mean((p-t)**2))),'peak_bias_ms':float(ps.max()-ts.max()),'peak_abs_error_ms':float(abs(ps.max()-ts.max())),'q95_bias_ms':float(np.quantile(ps,.95)-np.quantile(ts,.95)),'q99_bias_ms':float(np.quantile(ps,.99)-np.quantile(ts,.99)),'q99_abs_error_ms':float(abs(np.quantile(ps,.99)-np.quantile(ts,.99))),'q999_bias_descriptive_ms':float(np.quantile(ps,.999)-np.quantile(ts,.999)),'threshold_iou':tp/union if union else None,'threshold_precision':tp/(tp+fp) if tp+fp else None,'threshold_recall':tp/(tp+fn) if tp+fn else None,'tp_cells':tp,'false_positive_cells':fp,'miss_cells':fn,'psd_log10_mae':float(np.mean(np.abs(np.log10(power[mask])-np.log10(truth[mask])))),'unresolved_psd_log10_mae':float(np.mean(np.abs(np.log10(power[unresolved])-np.log10(truth[unresolved])))),'spectral_slope_abs_error':abs(spectral_slope(ps,5,5)-spectral_slope(ts,5,5)),'coarse_uv_mae_ms':float(np.mean(np.abs(aggregate(p)-c))),'reference_peak_ms':float(ts.max()),'prediction_peak_ms':float(ps.max())}
 out.update(gradient_consistency(p[0],p[1],t[0],t[1],dx_m=5000,dy_m=5000))
 from trace_physics_metrics import divergence_vorticity
 pd,pv=divergence_vorticity(p[0],p[1],5000,5000);td,tv=divergence_vorticity(t[0],t[1],5000,5000);out['divergence_rmse_s-1']=float(np.sqrt(np.mean((pd-td)**2)));out['vorticity_rmse_s-1']=float(np.sqrt(np.mean((pv-tv)**2)));return out

def main():
 start=time.perf_counter();cp=HERE/'STAGE2_EXPERIMENT_CONFIG_v2.json';cfg=json.loads(cp.read_text());assert hashlib.sha256(cp.read_bytes()).hexdigest()==(HERE/'STAGE2_EXPERIMENT_CONFIG_v2.sha256').read_text().strip()
 torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 data_path=HERE/'data_stage2_v2/HRRR_CONTROLLED_12_TO_5_v2.npz';d=np.load(data_path);coarse=d['coarse'];fine=d['fine'];ids=d['sample_id'];events=d['event_id'];split=d['split'];indices={s:np.where(split==s)[0] for s in ['train','validation','test']}
 assert set(events[indices['train']])=={'LAURA_2020','DELTA_2020'} and set(events[indices['validation']])=={'SALLY_2020'} and set(events[indices['test']])=={'IDA_2021'}
 vals=fine[indices['train']].transpose(1,0,2,3).reshape(2,-1);mu=vals.mean(1);sigma=vals.std(1);assert (sigma>0).all();cn=(coarse-mu[None,:,None,None])/sigma[None,:,None,None];fn=(fine-mu[None,:,None,None])/sigma[None,:,None,None];c=torch.from_numpy(cn.astype('float32'));f=torch.from_numpy(fn.astype('float32'));out=HERE/'stage2_run_v2';out.mkdir(exist_ok=True)
 wy=torch.tensor(WY,dtype=torch.float32);wx=torch.tensor(WX,dtype=torch.float32)
 def agg_tensor(x):return torch.einsum('ij,bcjk,lk->bcil',wy,x,wx)
 mean=MeanDownscaler(2);opt=torch.optim.AdamW(mean.parameters(),lr=cfg['learning_rate']);best=math.inf;stale=0;history=[];state=None
 for epoch in range(cfg['mean_epochs']):
  mean.train();losses=[]
  for i in np.random.permutation(indices['train']):
   opt.zero_grad();pred=mean(c[i:i+1],(48,48));loss=F.mse_loss(pred,f[i:i+1])+.1*F.mse_loss(agg_tensor(pred),c[i:i+1]);loss.backward();opt.step();losses.append(float(loss.detach()))
  mean.eval()
  with torch.no_grad():vl=float(F.mse_loss(mean(c[indices['validation']],(48,48)),f[indices['validation']]))
  history.append({'phase':'mean','epoch':epoch,'train':float(np.mean(losses)),'validation_mse':vl});print(json.dumps(history[-1]),flush=True)
  if math.isfinite(vl) and vl<best-1e-6:best=vl;stale=0;state={k:v.detach().clone() for k,v in mean.state_dict().items()};best_mean_epoch=epoch
  else:
   stale+=1
   if stale>=cfg['patience']:break
 if state is None:raise RuntimeError("NO_FINITE_MEAN_VALIDATION: test evaluation prohibited")
 mean.load_state_dict(state);mean.eval()
 with torch.no_grad():res=(f[indices['train']]-mean(c[indices['train']],(48,48))).numpy();rs=np.maximum(res.transpose(1,0,2,3).reshape(2,-1).std(1),.001)
 model=ResidualDDPM(mean,rs,cfg['diffusion_steps']);opt=torch.optim.AdamW(model.denoiser.parameters(),lr=cfg['learning_rate']);best_diff=math.inf;stale=0;state=None
 # Fixed validation noise and timesteps, model selection never sees Ida test.
 gen=torch.Generator().manual_seed(31415);nv=len(indices['validation']);veps=torch.randn((nv,2,48,48),generator=gen);vt=torch.linspace(1,cfg['diffusion_steps']-1,nv).long()
 for epoch in range(cfg['diffusion_epochs']):
  model.train();losses=[]
  for i in np.random.permutation(indices['train']):
   opt.zero_grad();loss=model.loss(c[i:i+1],f[i:i+1]);loss.backward();opt.step();losses.append(float(loss.detach()))
  model.eval()
  with torch.no_grad():vl=float(model.loss(c[indices['validation']],f[indices['validation']],vt,veps))
  history.append({'phase':'diffusion','epoch':epoch,'train':float(np.mean(losses)),'validation_fixed_epsilon_mse':vl});print(json.dumps(history[-1]),flush=True)
  if math.isfinite(vl) and vl<best_diff-1e-6:best_diff=vl;stale=0;state={k:v.detach().clone() for k,v in model.state_dict().items()};best_diff_epoch=epoch
  else:
   stale+=1
   if stale>=cfg['patience']:break
 if state is None:raise RuntimeError("NO_FINITE_DIFFUSION_VALIDATION: test evaluation prohibited")
 model.load_state_dict(state);model.eval();torch.save({'state_dict':model.state_dict(),'mu':mu,'sigma':sigma,'residual_scale':rs,'config':cfg},out/'trace_residual_ddpm.pt')
 threshold=float(np.quantile(np.hypot(fine[indices['train'],0],fine[indices['train'],1]),.99));rows=[];arrays=[]
 def denorm(x):return x*sigma[None,:,None,None]+mu[None,:,None,None]
 for i in indices['test']:
  ci=c[i:i+1]
  with torch.no_grad():
   bil=denorm(F.interpolate(ci,size=(48,48),mode='bilinear',align_corners=False).numpy())[0];det=denorm(mean(ci,(48,48)).numpy())[0]
   sam=denorm(np.stack([model.sample(ci,(48,48),cfg['seed']+100*int(i)+j).numpy()[0] for j in range(cfg['diffusion_samples_per_parent'])]))
  methods={'bilinear':bil,'deterministic':det,'diffusion_mean':sam.mean(0)}
  for j,x in enumerate(sam):methods[f'diffusion_sample_{j}']=x
  for name,x in list(methods.items()):methods[name+'_coarse_projected']=project(x,coarse[i])
  for name,x in methods.items():rows.append({'sample_id':str(ids[i]),'event_id':str(events[i]),'method':name,**metrics(x,fine[i],coarse[i],threshold)})
  np.savez_compressed(out/(str(ids[i])+'_outputs.npz'),coarse=coarse[i],reference=fine[i],bilinear=bil,deterministic=det,diffusion_samples=sam,diffusion_mean=sam.mean(0),diffusion_projected_samples=np.stack([project(x,coarse[i]) for x in sam]),threshold=np.array(threshold));arrays.append(str(ids[i]))
 result={'status':'EXECUTED_REAL_HRRR_CONTROLLED_RECONSTRUCTION','claim_scope':'No operational NEPS-G/India validation; no independent forecast truth','seed':cfg['seed'],'device':'cpu','train_frames':len(indices['train']),'validation_frames':nv,'test_frames':len(indices['test']),'independent_test_event_count':1,'train_events':['LAURA_2020','DELTA_2020'],'validation_event':'SALLY_2020','test_event':'IDA_2021','training_q99_threshold_ms':threshold,'best_mean_epoch':best_mean_epoch,'best_diffusion_epoch':best_diff_epoch,'best_mean_validation_mse':best,'best_diffusion_validation_epsilon_mse':best_diff,'diffusion_terminal_alpha_bar':float(model.abar[-1]),'residual_scale_normalized':rs.tolist(),'diffusion_samples_per_parent':cfg['diffusion_samples_per_parent'],'runtime_seconds':time.perf_counter()-start,'history':history,'test_metrics':rows,'q999_limitation':'~2.3 tail cells per frame; descriptive only, not statistically supported extremes','physics':'.1 training coarse-aggregation MSE plus exact coarse aggregation projection ON/OFF ablation; not fluid/thermodynamic PDE constraint','approximate_gradients':'Projected-grid local Cartesian derivatives; same reference geometry, map-factor variation neglected'}
 (out/'stage2_metrics.json').write_text(json.dumps(result,indent=2));
 with (out/'stage2_comparison.csv').open('w') as fp:w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 manifest={'config_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'dataset_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'timestamp_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'packages':{p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','eccodes']},'hardware':platform.platform(),'models':'locally trained small convolutional mean and residual DDPM; no pretrained CorrDiff checkpoint','normalization':'training-only U/V channel mean/std; training-only post-mean residual std; all inference denormalized to m/s','test_outputs':arrays,'runtime_seconds':result['runtime_seconds']};(out/'RUN_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['history','test_metrics']}),flush=True)
if __name__=='__main__':main()
