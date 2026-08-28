# -*- coding: utf-8 -*-
"""
compare_sccp_samples.py

Comparaison A OEIL des deux ScCP k9/K10/ic128, MEMES graines de bruit.

    results_afhq32/...            t ~ U(0, 1),    loss x-pred CLAMPEE
    results_afhq32_tmax095/...    t ~ U(0, 0.95), loss x-pred SANS clamp

Rangees adjacentes pour que l'oeil puisse comparer colonne par colonne, ce que
compare_tmax_ablation_afhq32.py ne permet pas (ses rangees sont separees par les
autres solveurs).

Solveur : Euler-20 pour tout le monde. C'est le seul que les DEUX acceptent — le
modele t_max=0.95 refuse Euler-100 et dopri5, qui evaluent t=0.99, hors de son
domaine d'entrainement (garde de fm_velocity_denom). Comparer a solveur egal est
de toute facon la seule lecture honnete.

Rangees
    1. ancien regime, budget apparie
    2. nouveau regime, meme budget          <- la comparaison
    3. |difference|, amplifiee              <- OU ils different
    4. ancien regime a 200k                 <- ce que l'ancien donne au maximum

Sorties -> compare_sccp_samples.png / .txt

Usage
-----
    source ~/.venvs/unn/bin/activate
    python compare_sccp_samples.py --device cuda:0 --n 12
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_fid_cifar10 import build_from_name

IMG_SIZE, CHANNELS = 32, 3
DIM = CHANNELS * IMG_SIZE * IMG_SIZE
RUN = "ConvScCP_UNN_rgb_k9_K10_ic128_L1_LFO"
AMP = 4.0                      # amplification de la carte de difference


@torch.no_grad()
def euler_sample(model, x0, steps=20):
    """Euler-N sur 0, 1/N, ..., 1-1/N ; dernier pas jusqu'a t=1. Le modele est en
    eval, il rend donc la VITESSE."""
    x = x0.clone()
    n = x.shape[0]
    grid = [i / steps for i in range(steps)]
    for i, tv in enumerate(grid):
        tn = grid[i + 1] if i + 1 < len(grid) else 1.0
        t = torch.full((n, 1), tv, device=x.device)
        v = model(torch.cat([x.view(n, -1), t], dim=-1)).view_as(x)
        x = x + v * (tn - tv)
    return x


def load(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m, is_unet = build_from_name(ck["name"], device)
    assert not is_unet
    m.load_state_dict(ck["ema_model"], strict=True)     # poids EMA
    m.t_max = ck.get("t_max")
    m.eval()
    return m, ck["step"], ck.get("t_max")


def to_img(x):
    return (x.detach().cpu().float() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--n", type=int, default=12, help="nombre de graines")
    p.add_argument("--step", type=int, default=120000, help="budget apparie")
    p.add_argument("--steps-euler", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = torch.device(args.device)

    # MEMES graines pour toutes les rangees
    g = torch.Generator().manual_seed(args.seed)
    x0 = torch.randn(args.n, CHANNELS, IMG_SIZE, IMG_SIZE, generator=g).to(device)

    specs = [
        ("ancien  t~U(0,1)",   f"results_afhq32/{RUN}/ckpt_step_{args.step}.pt"),
        ("nouveau t~U(0,.95)", f"results_afhq32_tmax095/{RUN}/ckpt_step_{args.step}.pt"),
        ("ancien  200k",       f"results_afhq32/{RUN}/ckpt_step_200000.pt"),
    ]
    imgs, labs = [], []
    for lab, path in specs:
        if not os.path.exists(path):
            print(f"  [skip] {path}"); continue
        m, step, tmax = load(path, device)
        out = euler_sample(m, x0, steps=args.steps_euler)
        imgs.append(out); labs.append(f"{lab}\n{step//1000}k · t_max={tmax}")
        print(f"  {lab:<20} step {step:,}  t_max={tmax}", flush=True)
        del m; torch.cuda.empty_cache()

    a, b = imgs[0], imgs[1]
    diff = (a - b).abs()
    per_col = [float(((a[i] - b[i]) ** 2).mean().sqrt()) for i in range(args.n)]
    rms = float(((a - b) ** 2).mean().sqrt())

    # echelle de reference : deux images AFHQ au hasard
    d = torch.load("./data/afhq_cat32_train.pt", map_location="cpu", weights_only=False)
    xd = d["data"].float().div_(127.5).sub_(1.0).reshape(-1, DIM)
    ref = float(((xd[:256] - xd[256:512]) ** 2).mean().sqrt())

    L = ["=" * 72,
         f"ScCP k9/K10/ic128 — memes graines, Euler-{args.steps_euler}, budget "
         f"{args.step:,}", "=" * 72, "",
         f"distance RMS ancien vs nouveau : {rms:.4f}",
         f"echelle de reference (2 images AFHQ au hasard) : {ref:.4f}",
         f"soit {100*rms/ref:.1f} % de l'echelle -> meme image, details differents", "",
         "par graine :",
         "  " + "  ".join(f"{i}:{v:.3f}" for i, v in enumerate(per_col)), "",
         f"le plus proche  : graine {int(np.argmin(per_col))} ({min(per_col):.4f})",
         f"le plus eloigne : graine {int(np.argmax(per_col))} ({max(per_col):.4f})"]
    txt = "\n".join(L)
    print("\n" + txt, flush=True)
    open("compare_sccp_samples.txt", "w").write(txt + "\n")

    rows = [to_img(imgs[0]), to_img(imgs[1]),
            np.clip(to_img(diff * AMP * 2 - 1), 0, 1)]
    row_labs = [labs[0], labs[1], f"|difference|\namplifiee x{AMP:g}"]
    if len(imgs) > 2:
        rows.append(to_img(imgs[2])); row_labs.append(labs[2])

    nr, nc = len(rows), args.n
    fig = plt.figure(figsize=(1.16 * nc + 3.2, 1.30 * nr + 1.0))
    gs = fig.add_gridspec(nr, nc + 3, wspace=0.05, hspace=0.06)
    for r, (row, lab) in enumerate(zip(rows, row_labs)):
        for c in range(nc):
            ax = fig.add_subplot(gs[r, c + 3])
            ax.imshow(row[c]); ax.set_xticks([]); ax.set_yticks([])
            if r == 2:
                ax.set_xlabel(f"{per_col[c]:.3f}", fontsize=6.5, labelpad=1.5,
                              color="#C05621")
        ax = fig.add_subplot(gs[r, 0:3]); ax.axis("off")
        ax.text(0.97, 0.5, lab, ha="right", va="center", fontsize=8.2,
                family="monospace", color="#C05621" if r == 2 else "#222222")
    fig.suptitle(
        f"Les deux ScCP sur les MEMES graines — Euler-{args.steps_euler}, "
        f"budget apparie {args.step//1000}k\n"
        f"RMS ancien/nouveau = {rms:.3f}, soit {100*rms/ref:.0f} % de l'ecart entre "
        f"deux images AFHQ au hasard", fontsize=10.5)
    fig.savefig("compare_sccp_samples.png", dpi=140, bbox_inches="tight")
    print("\n-> compare_sccp_samples.png / .txt", flush=True)


if __name__ == "__main__":
    main()
