# -*- coding: utf-8 -*-
"""
nmse_curves_afhq32.py

Courbes de NMSE des runs AFHQ-32 sous le regime t_max (loss x-pred sans clamp),
dans la convention du projet :

    nmse(t) = mse(t) / mse du predicteur trivial       (cf. denoise_probe.py)
              nmse = 1 -> le modele n'apprend rien ; nmse -> 0 -> parfait.

Trois panneaux :
  1. nmse_x1(t) = MSE(x1_pred, x1) / var(x1)          <- vue DEBRUITAGE
     Predicteur trivial = la moyenne du dataset. Reference tracee : le debruiteur
     lineaire OPTIMAL (Wiener gaussien, cf. wiener_view.py), calcule en forme
     fermee depuis mu et C des images. Un modele au-dessus de cette courbe n'a
     rien appris qu'un modele lineaire ne sache deja faire.
  2. nmse_v(t) = MSE(v_pred, ut) / var(ut)            <- vue OBJECTIF D'ENTRAINEMENT
     C'est la loss reelle des runs, renormalisee. Predicteur trivial = la vitesse
     moyenne (constante), d'ou un denominateur constant var(ut) = var(x1) + 1.
  3. nmse_v vs STEP : les courbes d'apprentissage, lues dans les loss_log des
     checkpoints et divisees par le meme var(ut).

ATTENTION — les runs AFHQ s'entrainent sur la TOTALITE du cache (get_afhq_loader
ne fait pas de split). Les images d'evaluation ont donc ete vues a l'entrainement,
alors que la reference Wiener est ajustee sur le complementaire : la comparaison
avantage legerement les reseaux. C'est un plancher de reference, pas un test.

Sorties -> nmse_curves_afhq32.png / .txt

Usage
-----
    source ~/.venvs/unn/bin/activate
    python nmse_curves_afhq32.py --device cuda:0 --ckpt ckpt_step_100000.pt
"""

import argparse
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_fid_cifar10 import build_from_name

IMG_SIZE, CHANNELS = 32, 3
DIM = CHANNELS * IMG_SIZE * IMG_SIZE
N_VAL, SEED = 512, 0                       # meme split que wiener_view.py

RUNS = [
    ("UNet_torchcfm_ch32", "v-pred", "#2b6cb0"),
    ("ConvScCP_UNN_rgb_k9_K10_ic128_L1_LFO", "x-pred", "#c05621"),
    ("MinimalUNetFM_kamb", "x-pred", "#2f855a"),
]
SHORT = {"UNet_torchcfm_ch32": "UNet torchcfm ch32",
         "ConvScCP_UNN_rgb_k9_K10_ic128_L1_LFO": "ConvScCP k9/K10/ic128",
         "MinimalUNetFM_kamb": "MinimalUNetFM (Kamb)"}


def wiener_nmse(xtr, xva, t_grid, var_va, seed=SEED + 1234):
    """Debruiteur lineaire optimal : E[x1|xt] sous un prior gaussien (mu, C) ajuste
    sur xtr. xt = (1-t) x0 + t x1 avec x0 ~ N(0, I), donc en base propre de C le
    gain vaut  t*lam / (t^2*lam + (1-t)^2)  (identique a wiener_view.py)."""
    mu = xtr.mean(dim=0, keepdim=True)
    xc = xtr - mu
    cov = xc.T @ xc / xc.shape[0]
    lam, U = torch.linalg.eigh(cov)
    lam = lam.clamp_min(0)
    gv = torch.Generator().manual_seed(seed)
    out = []
    for t in t_grid:
        x0 = torch.randn(xva.shape[0], xva.shape[1], generator=gv).double()
        xt = (1 - t) * x0 + t * xva
        gain = t * lam / (t ** 2 * lam + (1 - t) ** 2)
        pred = mu + ((xt - t * mu) @ U * gain) @ U.T
        out.append(float(((pred - xva) ** 2).mean()) / var_va)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--cache", type=str, default="./data/afhq_cat32_train.pt")
    p.add_argument("--results-dir", type=str, default="results_afhq32_tmax095")
    p.add_argument("--ckpt", type=str, default="ckpt_step_100000.pt")
    p.add_argument("--t-max", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    t_grid = [round(0.05 * i, 2) for i in range(1, 20)]          # 0.05 .. 0.95
    t_grid = [t for t in t_grid if t <= args.t_max + 1e-9]

    d = torch.load(args.cache, map_location="cpu", weights_only=False)
    x = d["data"].float().div_(127.5).sub_(1.0)
    g = torch.Generator().manual_seed(args.seed)
    x = x[torch.randperm(x.shape[0], generator=g)]
    flat = x.reshape(x.shape[0], -1).double()
    xva, xtr = flat[:N_VAL], flat[N_VAL:]
    var_x1 = float(((xva - xva.mean(dim=0, keepdim=True)) ** 2).mean())
    print(f"AFHQ : {flat.shape[0]} images | eval sur {N_VAL} | var(x1) = {var_x1:.4f}",
          flush=True)

    t0 = time.perf_counter()
    print("reference lineaire (Wiener)...", flush=True)
    wien = wiener_nmse(xtr, xva, t_grid, var_x1)

    # bruit FIXE, partage par tous les modeles et par la reference
    gv = torch.Generator().manual_seed(args.seed + 1234)
    x0_va = torch.randn(N_VAL, DIM, generator=gv).double()
    var_ut = float(((x0_va - xva) ** 2).mean())          # var du predicteur trivial en v
    print(f"var(ut) = {var_ut:.4f}   (temps ref : {time.perf_counter()-t0:.0f}s)",
          flush=True)

    x1 = xva.float().to(device)
    x0 = x0_va.float().to(device)
    ut = x1 - x0

    curves, learn = {}, {}
    for name, kind, _ in RUNS:
        path = os.path.join(args.results_dir, name, args.ckpt)
        if not os.path.exists(path):
            print(f"  [skip] {path} absent", flush=True); continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model, is_unet = build_from_name(ck["name"], device)
        model.load_state_dict(ck["ema_model"], strict=True)   # poids EMA = ceux qui comptent
        model.t_max = ck.get("t_max")
        model.eval() if is_unet else model.train()   # x-pred : .train() -> x1_pred brut
        nx, nv = [], []
        with torch.no_grad():
            for tv in t_grid:
                t = torch.full((N_VAL, 1), tv, device=device)
                xt = (1 - t) * x0 + t * x1
                if is_unet:
                    v = model(t.view(-1), xt.view(-1, CHANNELS, IMG_SIZE, IMG_SIZE))
                    v = v.reshape(N_VAL, -1)
                    x1p = xt + (1 - t) * v
                else:
                    x1p = model(torch.cat([xt, t], dim=-1))
                    v = (x1p - xt) / (1 - t)
                nx.append(float(((x1p - x1) ** 2).mean()) / var_x1)
                nv.append(float(((v - ut) ** 2).mean()) / var_ut)
        curves[name] = (nx, nv)
        ll = dict(ck["loss_log"])
        learn[name] = ([s for s in sorted(ll)], [ll[s] / var_ut for s in sorted(ll)])
        print(f"  {SHORT[name]:<24} step {ck['step']:>7,} | nmse_x1 moy "
              f"{np.mean(nx):.4f} | nmse_v moy {np.mean(nv):.4f}", flush=True)

    # ------------------------------------------------------------------ texte
    L = ["=" * 82,
         f"NMSE — {args.results_dir} / {args.ckpt} — eval sur {N_VAL} images, "
         f"bruit fixe (seed {args.seed})", "=" * 82, "",
         f"predicteur trivial x1 : moyenne du dataset,  var(x1) = {var_x1:.4f}",
         f"predicteur trivial v  : vitesse moyenne,     var(ut) = {var_ut:.4f}", "",
         "1. nmse_x1(t) = MSE(x1_pred, x1) / var(x1)   [vue debruitage]", "-" * 82,
         f"{'t':>6}{'lineaire opt.':>16}" + "".join(f"{SHORT[n][:20]:>22}" for n, _, _ in RUNS if n in curves)]
    for i, tv in enumerate(t_grid):
        L.append(f"{tv:>6.2f}{wien[i]:>16.4f}" +
                 "".join(f"{curves[n][0][i]:>22.4f}" for n, _, _ in RUNS if n in curves))
    L += ["", "2. nmse_v(t) = MSE(v_pred, ut) / var(ut)   [objectif d'entrainement]",
          "-" * 82,
          f"{'t':>6}{'':>16}" + "".join(f"{SHORT[n][:20]:>22}" for n, _, _ in RUNS if n in curves)]
    for i, tv in enumerate(t_grid):
        L.append(f"{tv:>6.2f}{'':>16}" +
                 "".join(f"{curves[n][1][i]:>22.4f}" for n, _, _ in RUNS if n in curves))
    L += ["", f"moyenne sur la grille t = {t_grid[0]}..{t_grid[-1]}", "-" * 82,
          f"{'modele':<26}{'nmse_x1':>10}{'nmse_v':>10}{'vs lineaire (x1)':>20}"]
    wm = float(np.mean(wien))
    L.append(f"{'lineaire optimal (Wiener)':<26}{wm:>10.4f}{'-':>10}{'-':>20}")
    for n, _, _ in RUNS:
        if n not in curves:
            continue
        mx, mv = float(np.mean(curves[n][0])), float(np.mean(curves[n][1]))
        L.append(f"{SHORT[n]:<26}{mx:>10.4f}{mv:>10.4f}{100*(mx/wm-1):>19.1f}%")
    L += ["", "ATTENTION : les runs AFHQ s'entrainent sur TOUT le cache, donc ces 512",
          "images ont ete vues a l'entrainement (la reference Wiener, elle, est ajustee",
          "sur le complementaire). Plancher de reference, pas test de generalisation."]
    txt = "\n".join(L)
    print("\n" + txt, flush=True)
    open("nmse_curves_afhq32.txt", "w").write(txt + "\n")

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))
    ax[0].plot(t_grid, wien, "k--", lw=1.6, label="lineaire optimal (Wiener)")
    for n, kind, col in RUNS:
        if n not in curves:
            continue
        ax[0].plot(t_grid, curves[n][0], "-o", ms=3.5, color=col,
                   label=f"{SHORT[n]} ({kind})")
        ax[1].plot(t_grid, curves[n][1], "-o", ms=3.5, color=col,
                   label=f"{SHORT[n]} ({kind})")
        st, va = learn[n]
        ax[2].plot(st, va, "-", color=col, label=SHORT[n])
    ax[0].set_title("nmse$_{x_1}$(t) — vue debruitage")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("MSE / var($x_1$)"); ax[0].set_yscale("log")
    ax[1].set_title("nmse$_v$(t) — objectif d'entrainement")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("MSE / var($u_t$)")
    ax[2].set_title("nmse$_v$ vs budget (loss du run / var($u_t$))")
    ax[2].set_xlabel("step"); ax[2].set_ylabel("MSE / var($u_t$)")
    ax[2].set_yscale("log"); ax[2].set_xscale("log")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=7.5)
    fig.suptitle(f"AFHQ-32, regime t_max = {args.t_max:g} (loss sans clamp) — "
                 f"{args.ckpt}", fontsize=11)
    fig.tight_layout()
    fig.savefig("nmse_curves_afhq32.png", dpi=130)
    print(f"\n-> nmse_curves_afhq32.png / .txt   ({time.perf_counter()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
