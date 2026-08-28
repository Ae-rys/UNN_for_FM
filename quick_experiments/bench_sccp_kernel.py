# -*- coding: utf-8 -*-
"""
bench_sccp_kernel.py — Combien coute vraiment un ConvScCP AFHQ a gros kernel ?

Avant de lancer une run k=25 a l'aveugle (le monstre k25/K19/ic512 du run CIFAR
est mort a 514 h d'ETA), on mesure le temps reel par batch a la config AFHQ
exacte (K=10, ic=256, RGB 32x32, batch 128, use_checkpoint=True comme dans
build_experiments) pour k = 9, 15, 21, 25, et on extrapole l'ETA a 50k steps.

Ce que le chiffre repond
------------------------
Le cout d'une conv va naivement en k^2 : k=9 -> 25 serait ~7.7x, soit ~25 h pour
tes 3.25 h a k=9. MAIS bench_sccp_speed.py a montre que ScCP est limite par le
LANCEMENT de kernels CUDA (2000 petits kernels sequentiels), pas par les FLOPs.
De gros kernels amortissent cet overhead, donc le cout devrait scaler nettement
SOUS-lineairement en k^2. Ce script mesure lequel des deux regimes domine ici.

La colonne 'scaling B' (temps a B=512 / temps a B=128, ramene a un ratio x4)
diagnostique le regime : ~4 = compute-bound, ~1 = latency-bound.

Usage :  python bench_sccp_kernel.py --device cuda:1
Duree : ~2-4 min.
Sortie -> bench_sccp_kernel.txt (+ table a l'ecran)
"""

import argparse
import time

import torch

from compute_fid_cifar10 import ConvScCP_UNN, DIM, CHANNELS, IMG_SIZE

KERNELS = [9, 15, 21, 25]
K_ITER = 10
IC = 256
BATCH = 128                  # RECIPE["batch_size"]
BIG_BATCH = 512              # pour diagnostiquer compute- vs latency-bound
REF_STEPS = 50000            # budget de la run AFHQ actuelle


def build(kernel, device):
    """Config AFHQ exacte (run_cifar10_torchcfm_recipe.build_experiments)."""
    return ConvScCP_UNN(
        dim=DIM, K=K_ITER, internal_channel=IC, kernel_size=kernel,
        in_channels=CHANNELS, img_size=IMG_SIZE,
        use_Unet="l1", version="LFO", use_checkpoint=True, w_bias=True,
    ).to(device).train()


def fwd_bwd(model, xt_t, target):
    loss = ((model(xt_t) - target) ** 2).mean()
    loss.backward()
    model.zero_grad(set_to_none=True)


def bench_time(model, b, device, n_iter=20, n_warmup=5):
    """ms par forward+backward, batch b."""
    xt_t = torch.randn(b, DIM + 1, device=device)
    xt_t[:, DIM] = torch.rand(b, device=device)
    target = torch.randn(b, DIM, device=device)
    for _ in range(n_warmup):
        fwd_bwd(model, xt_t, target)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fwd_bwd(model, xt_t, target)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1e3


def main():
    p = argparse.ArgumentParser(description="Cout d'un ConvScCP AFHQ vs taille de kernel.")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--steps", type=int, default=REF_STEPS)
    args = p.parse_args()
    device = torch.device(args.device)

    print(f"GPU : {torch.cuda.get_device_name(device)}  |  config AFHQ "
          f"K={K_ITER} ic={IC} RGB {IMG_SIZE}x{IMG_SIZE} ckpt=True", flush=True)
    header = (f"{'kernel':>6} {'params':>9} {'ms/batch':>9} {'ms/batch':>9} "
              f"{'scaling B':>10} {'vs k=9':>7} {'k^2 naif':>9} {'ETA':>9}")
    sub = (f"{'':>6} {'':>9} {'B=128':>9} {'B=512':>9} {'(4=compute)':>10} "
           f"{'':>7} {'':>9} {args.steps//1000:>7}k st")
    print(header); print(sub); print("-" * len(header), flush=True)

    rows, t_ref = [], None
    for k in KERNELS:
        model = build(k, device)
        n_params = sum(q.numel() for q in model.parameters())
        try:
            t128 = bench_time(model, BATCH, device)
            tbig = bench_time(model, BIG_BATCH, device)
            scaling = (tbig / t128)
        except torch.cuda.OutOfMemoryError:
            print(f"{k:>6}  OOM a B={BIG_BATCH} — on garde B={BATCH} seul", flush=True)
            torch.cuda.empty_cache()
            t128, scaling = bench_time(model, BATCH, device), float("nan")
        if t_ref is None:
            t_ref = t128
        eta_h = t128 * args.steps / 1e3 / 3600
        naif = (k / KERNELS[0]) ** 2
        print(f"{k:>6} {n_params/1e6:>8.2f}M {t128:>9.1f} "
              f"{(tbig if scaling == scaling else float('nan')):>9.1f} "
              f"{scaling:>10.2f} {t128/t_ref:>7.2f} {naif:>9.2f} {eta_h:>7.1f} h",
              flush=True)
        rows.append((k, n_params, t128, scaling, t128 / t_ref, naif, eta_h))
        del model
        torch.cuda.empty_cache()

    with open("bench_sccp_kernel.txt", "w") as f:
        f.write(f"# config AFHQ K={K_ITER} ic={IC} RGB {IMG_SIZE} batch={BATCH} "
                f"ckpt=True | ETA pour {args.steps} steps\n")
        f.write("kernel\tparams\tms_b128\tscaling_B\tratio_vs_k9\tk2_naif\teta_h\n")
        for r in rows:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]:.2f}\t{r[3]:.3f}\t{r[4]:.3f}\t"
                    f"{r[5]:.3f}\t{r[6]:.2f}\n")

    r25 = next((r for r in rows if r[0] == 25), None)
    if r25:
        regime = ("SOUS-lineaire en k^2 : l'overhead de lancement domine, comme prevu"
                  if r25[4] < r25[5] * 0.7 else
                  "proche du k^2 naif : on est bien compute-bound a ces tailles")
        print(f"\nk=25 coute {r25[4]:.2f}x le k=9 (naif k^2 : {r25[5]:.2f}x) -> {regime}.")
        print(f"ETA k=25 sur {args.steps//1000}k steps : {r25[6]:.1f} h.", flush=True)
    print("\n-> bench_sccp_kernel.txt")


if __name__ == "__main__":
    main()
