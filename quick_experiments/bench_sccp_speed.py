# -*- coding: utf-8 -*-
"""
bench_sccp_speed.py — Pourquoi ConvScCP est-il plus lent à entraîner qu'un UNet ?

Mesure, pour la config ScCP du rapport (K=20, ic=64, kernel 9x9, prox l1) et des
variantes qui isolent chaque facteur (K, taille de kernel, checkpointing, LNO),
face aux baselines SmallUNet / UNetModel torchcfm :

  1. temps forward+backward par batch (B=128 comme a l'entrainement, + B=512
     pour diagnostiquer si on est limite par le calcul ou par le lancement
     sequentiel de petits kernels : un modele compute-bound voit son temps ~x4
     quand B x4 ; un modele latency-bound voit son temps a peine bouger) ;
  2. GFLOPs reels d'un forward+backward (torch.profiler, with_flops) ;
  3. nombre de kernels CUDA lances par forward+backward (proxy de la
     "sequentialite" effective : beaucoup de petits kernels = GPU sous-utilise).

Usage :  CUDA_VISIBLE_DEVICES=1 python bench_sccp_speed.py
Duree : ~1-2 min.
"""

import time
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

from models.architectures import ConvScCP_UNN, SmallUNet
from torchcfm.models.unet import UNetModel

DEVICE = torch.device("cuda")
DIM = 784


class TorchCFMWrapper(nn.Module):
    """Adapte UNetModel (t, x) -> interface xt_t (B, 785) des autres modeles."""
    def __init__(self):
        super().__init__()
        self.net = UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1)

    def forward(self, xt_t):
        x = xt_t[:, :DIM].view(-1, 1, 28, 28)
        t = xt_t[:, DIM]
        return self.net(t, x).view(-1, DIM)


def build_models():
    return {
        "ScCP L1 LFO K=20 k=9 (rapport)": lambda: ConvScCP_UNN(
            dim=784, K=20, internal_channel=64, use_Unet="l1", version="LFO"),
        "ScCP L1 LNO K=20 k=9 + ckpt   ": lambda: ConvScCP_UNN(
            dim=784, K=20, internal_channel=64, use_Unet="l1", version="LNO",
            use_checkpoint=True),
        "ScCP L1 LNO K=20 k=9 sans ckpt": lambda: ConvScCP_UNN(
            dim=784, K=20, internal_channel=64, use_Unet="l1", version="LNO"),
        "ScCP L1 LFO K=20 k=3          ": lambda: ConvScCP_UNN(
            dim=784, K=20, internal_channel=64, use_Unet="l1", version="LFO",
            kernel_size=3),
        "ScCP L1 LFO K=5  k=9          ": lambda: ConvScCP_UNN(
            dim=784, K=5, internal_channel=64, use_Unet="l1", version="LFO"),
        "SmallUNet base_ch=32          ": lambda: SmallUNet(base_ch=32),
        # B=512 OOM sur GPU partage -> scaling mesure a B=256 pour celui-ci
        "UNetModel torchcfm ch=32      ": TorchCFMWrapper,
    }


BIG_BATCH = {"UNetModel torchcfm ch=32      ": 256}      # defaut : 512


def fwd_bwd(model, xt_t, target):
    out = model(xt_t)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    model.zero_grad(set_to_none=True)


def bench_time(model, B, n_iter=30, n_warmup=5):
    xt_t = torch.randn(B, DIM + 1, device=DEVICE)
    xt_t[:, DIM] = torch.rand(B, device=DEVICE)          # t dans [0,1]
    target = torch.randn(B, DIM, device=DEVICE)
    for _ in range(n_warmup):
        fwd_bwd(model, xt_t, target)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fwd_bwd(model, xt_t, target)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1e3     # ms / batch


def profile_flops_kernels(model, B=128):
    xt_t = torch.randn(B, DIM + 1, device=DEVICE)
    xt_t[:, DIM] = torch.rand(B, device=DEVICE)
    target = torch.randn(B, DIM, device=DEVICE)
    fwd_bwd(model, xt_t, target)                          # warmup
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 with_flops=True) as prof:
        fwd_bwd(model, xt_t, target)
        torch.cuda.synchronize()
    gflops = sum(e.flops for e in prof.key_averages() if e.flops) / 1e9
    n_kernels = sum(e.count for e in prof.key_averages()
                    if e.device_type == torch.autograd.DeviceType.CUDA)
    return gflops, n_kernels


def main():
    print(f"GPU : {torch.cuda.get_device_name(0)}", flush=True)
    header = (f"{'modele':<32} {'params':>8} {'ms/batch':>9} {'ms/batch':>9} "
              f"{'scaling':>8} {'GFLOPs':>8} {'kernels':>8}")
    sub = (f"{'':<32} {'':>8} {'B=128':>9} {'B=grand':>9} "
           f"{'norm./4':>8} {'f+b':>8} {'CUDA':>8}")
    print(header); print(sub); print("-" * len(header))
    for name, build in build_models().items():
        model = build().to(DEVICE).train()
        n_params = sum(p.numel() for p in model.parameters())
        big = BIG_BATCH.get(name, 512)
        t128 = bench_time(model, 128)
        tbig = bench_time(model, big)
        gflops, n_kern = profile_flops_kernels(model, B=128)
        scaling = (tbig / t128) * (512 / big)            # ramene a un ratio "x4"
        print(f"{name:<32} {n_params/1e3:>7.0f}k {t128:>9.1f} {tbig:>9.1f} "
              f"{scaling:>8.2f} {gflops:>8.1f} {n_kern:>8}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
