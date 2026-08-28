# -*- coding: utf-8 -*-
"""
demo_gpu_util_lies.py — « GPU-Util 99% » ne veut PAS dire « GPU utilisé à 99% ».

nvidia-smi definit utilization.gpu comme : pourcentage du temps pendant lequel
AU MOINS UN kernel etait resident sur le GPU. C'est binaire occupe/idle dans le
temps ; ca ne dit RIEN du nombre de SMs actifs ni du debit atteint.

Quatre charges, toutes lues a 99-100% par `utilization.gpu`, avec des debits
reels separes par 4 ordres de grandeur (mesures, RTX 2080 Ti, ~13400 GFLOP/s
en pointe fp32) :

  A. petits kernels dos a dos (add sur 1000 floats)  sm 51%      0.1 GFLOP/s
  B. gros matmul 4096^3 (regime "gros UNet")         sm 77%   3800   GFLOP/s
  C. ScCP L1 LFO K=20, fwd+bwd, B=128                sm 80%    129   GFLOP/s
  D. SmallUNet base_ch=32, fwd+bwd, B=128            sm 82%    620   GFLOP/s

C vs D est le point cle : occupation IDENTIQUE (80 vs 82%), debit 4.8x plus
faible pour ScCP. Le GPU est bien occupe en permanence — il fait juste tres
peu de travail utile par unite de temps. Voir bench_sccp_speed.py pour la
decomposition (FLOPs, nb de kernels) et track_sccp_speed en memoire.

Usage :  CUDA_VISIBLE_DEVICES=1 python demo_gpu_util_lies.py   (~40 s)
La colonne sm% vient de `nvidia-smi pmon` filtre sur NOTRE PID : elle reste
valide meme si un autre job partage le GPU.
"""

import os
import subprocess
import threading
import time

import torch

DEVICE = torch.device("cuda")
GPU_INDEX = 1          # index PHYSIQUE pour nvidia-smi (CUDA_VISIBLE_DEVICES=1 -> cuda:0)
DURATION = 8.0         # secondes par phase
MY_PID = os.getpid()


def sample_util(stop, out):
    """sm% de NOTRE process seulement (nvidia-smi pmon), pour rester valide meme
    si un autre job partage le GPU — `--query-gpu=utilization.gpu` agrege tout."""
    while not stop.is_set():
        r = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-i", str(GPU_INDEX)],
                           capture_output=True, text=True)
        for line in r.stdout.splitlines():
            f = line.split()
            if len(f) > 3 and f[1] == str(MY_PID) and f[3].isdigit():
                out.append(int(f[3]))


def run_phase(name, work_fn, flops_per_call):
    """work_fn() lance un travail GPU ; mesure sm% (notre PID) + GFLOP/s reels."""
    stop, samples = threading.Event(), []
    th = threading.Thread(target=sample_util, args=(stop, samples))
    th.start()
    torch.cuda.synchronize()
    t0, n_calls = time.perf_counter(), 0
    while time.perf_counter() - t0 < DURATION:
        work_fn()
        n_calls += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set(); th.join()
    util = sum(samples) / max(len(samples), 1)
    gflops = n_calls * flops_per_call / elapsed / 1e9
    print(f"{name:<42} sm% (notre PID) = {util:5.1f}%   debit = {gflops:10.1f} GFLOP/s",
          flush=True)


def main():
    print(f"GPU : {torch.cuda.get_device_name(0)} "
          f"(~13400 GFLOP/s fp32 en pointe)", flush=True)

    print("sm% mesure par process (nvidia-smi pmon) -> insensible aux autres jobs.\n",
          flush=True)

    # A. chaine longue et fine : petits kernels dos a dos (regime ScCP)
    small = torch.randn(1000, device=DEVICE)
    def tiny():
        x = small
        for _ in range(200):           # 200 lancements minuscules
            x = x + 1.0
    run_phase("A. petits kernels dos a dos (~ScCP)", tiny,
              flops_per_call=200 * 1000)

    # B. pile courte et grosse : matmul dense (regime gros UNet)
    a = torch.randn(4096, 4096, device=DEVICE)
    b = torch.randn(4096, 4096, device=DEVICE)
    def big():
        torch.mm(a, b)
    run_phase("B. gros matmul 4096^3 (~gros UNet)", big,
              flops_per_call=2 * 4096 ** 3)

    # C/D. les VRAIS modeles, en conditions d'entrainement (B=128, fwd+bwd).
    # GFLOPs/step mesures par torch.profiler dans bench_sccp_speed.py.
    from bench_sccp_speed import fwd_bwd, DIM
    from models.architectures import ConvScCP_UNN, SmallUNet

    xt_t = torch.randn(128, DIM + 1, device=DEVICE)
    xt_t[:, DIM] = torch.rand(128, device=DEVICE)
    target = torch.randn(128, DIM, device=DEVICE)

    for label, model, gflops_step in [
        ("C. ScCP L1 LFO K=20 (fwd+bwd, B=128)",
         ConvScCP_UNN(dim=784, K=20, internal_channel=64,
                      use_Unet="l1", version="LFO"), 21.2),
        ("D. SmallUNet base_ch=32 (fwd+bwd, B=128)",
         SmallUNet(base_ch=32), 13.9),
    ]:
        model = model.to(DEVICE).train()
        for _ in range(5):
            fwd_bwd(model, xt_t, target)                       # warmup
        run_phase(label, lambda m=model: fwd_bwd(m, xt_t, target),
                  flops_per_call=gflops_step * 1e9)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
