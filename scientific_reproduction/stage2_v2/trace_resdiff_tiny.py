"""Tiny conditional residual diffusion prototype for TRACE Stage 2.

Architecture follows the *idea* of residual corrective diffusion:
1) deterministic conv net predicts a fine-grid conditional mean
2) diffusion denoiser models residual truth - predicted mean

Status: implementation + synthetic smoke test only. It is NOT evidence of
12->5 km weather skill and must not be presented as CorrDiff reproduction.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


class MeanDownscaler(nn.Module):
    def __init__(self, channels=2, hidden=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )
    def forward(self, coarse, target_size):
        up = F.interpolate(coarse, size=target_size, mode="bilinear", align_corners=False)
        return up + self.net(up)


class ResidualDenoiser(nn.Module):
    def __init__(self, channels=2, hidden=32):
        super().__init__()
        # noisy residual + conditioning mean + timestep channel
        self.net = nn.Sequential(
            nn.Conv2d(channels*2+1, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )
    def forward(self, noisy_residual, condition, t_norm):
        t = t_norm.view(-1,1,1,1).expand(-1,1,*noisy_residual.shape[-2:])
        return self.net(torch.cat([noisy_residual, condition, t], dim=1))


class TinyResidualDiffusion(nn.Module):
    def __init__(self, channels=2, steps=20):
        super().__init__()
        self.mean_model = MeanDownscaler(channels)
        self.denoiser = ResidualDenoiser(channels)
        beta = torch.linspace(1e-4, 0.02, steps)
        alpha = 1.0-beta
        abar = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("abar", abar)
        self.steps = steps

    def diffusion_loss(self, coarse, fine_truth):
        mean = self.mean_model(coarse, fine_truth.shape[-2:])
        residual = fine_truth - mean
        b = fine_truth.shape[0]
        t = torch.randint(0, self.steps, (b,), device=fine_truth.device)
        eps = torch.randn_like(residual)
        a = self.abar[t].view(b,1,1,1)
        noisy = torch.sqrt(a)*residual + torch.sqrt(1-a)*eps
        pred = self.denoiser(noisy, mean.detach(), t.float()/max(1,self.steps-1))
        return F.mse_loss(pred, eps), mean

    @torch.no_grad()
    def sample(self, coarse, target_size, seed=0):
        device=coarse.device
        gen=torch.Generator(device=device).manual_seed(seed)
        mean=self.mean_model(coarse,target_size)
        x=torch.randn(mean.shape,device=device,dtype=mean.dtype,generator=gen)
        b=coarse.shape[0]
        for ti in reversed(range(self.steps)):
            t=torch.full((b,),ti,device=device,dtype=torch.long)
            eps=self.denoiser(x,mean,t.float()/max(1,self.steps-1))
            alpha=self.alpha[ti]; abar=self.abar[ti]; beta=self.beta[ti]
            x=(x-(1-alpha)/torch.sqrt(1-abar)*eps)/torch.sqrt(alpha)
            if ti>0:
                noise=torch.randn(x.shape,device=device,dtype=x.dtype,generator=gen)
                x=x+torch.sqrt(beta)*noise
        return mean+x


def synthetic_smoke_train(steps=20, seed=0):
    """Deterministic toy optimization to verify mean + diffusion code paths."""
    torch.manual_seed(seed); torch.set_num_threads(1)
    yy,xx=torch.meshgrid(torch.linspace(-1,1,16),torch.linspace(-1,1,16),indexing="ij")
    u=torch.exp(-5*(xx**2+yy**2))*2 + 0.25*torch.sin(11*xx)
    v=torch.exp(-5*(xx**2+yy**2))*1.2 + 0.20*torch.cos(9*yy)
    fine=torch.stack([u,v],dim=0)[None]
    coarse=F.interpolate(fine,size=(7,7),mode="area")
    model=TinyResidualDiffusion(channels=2,steps=8)
    opt=torch.optim.Adam(model.parameters(),lr=2e-3)

    t_index=4
    eps=torch.randn_like(fine)
    losses=[]
    for _ in range(steps):
        opt.zero_grad()
        mean=model.mean_model(coarse,fine.shape[-2:])
        residual=fine-mean
        a=model.abar[t_index]
        noisy=torch.sqrt(a)*residual + torch.sqrt(1-a)*eps
        t_norm=torch.full((1,), t_index/max(1,model.steps-1), dtype=fine.dtype)
        pred=model.denoiser(noisy,mean.detach(),t_norm)
        loss=F.mse_loss(pred,eps) + 0.2*F.mse_loss(mean,fine)
        loss.backward(); opt.step(); losses.append(float(loss.detach()))

    sample=model.sample(coarse,(16,16),seed=1)
    if sample.shape!=fine.shape or not torch.isfinite(sample).all():
        raise AssertionError("residual diffusion sample failed")
    if not losses[-1] < losses[0]:
        raise AssertionError("synthetic residual-diffusion optimization did not reduce loss")
    return {"loss_first":losses[0],"loss_last":losses[-1],"sample_shape":list(sample.shape),
            "parameters":sum(p.numel() for p in model.parameters())}


if __name__ == "__main__":
    print("RESDIFF_SYNTHETIC_SMOKE_PASS", synthetic_smoke_train())
