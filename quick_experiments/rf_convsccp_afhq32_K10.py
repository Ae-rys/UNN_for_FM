# -*- coding: utf-8 -*-
"""
rf_convsccp_afhq32_K10.py — Champ de reception du ConvScCP a K=10 couches sur
AFHQ-chats 32x32, pour les deux kernels entraines (9x9 et 25x25).

Deux notions de portee, tracees ensemble :

  * RF NOMINAL   : ce que l'archi PEUT voir. Une iteration ConvScCP touche deux
    convolutions k x k en serie sur le chemin x -> x_next :
        u_next = prox( u + sigma * conv(y, W) )          -> +(k-1)/2
        x_next = ( x - tau * conv_transpose(u, V) + .. ) -> +(k-1)/2
    Le prox (L1ProxConv) est un clamp PONCTUEL (radius r(t) predit depuis t
    seul) : il n'ajoute aucune portee spatiale. Donc exactement
        R_nominal = K * (k - 1)   px de rayon.
    K=10, k=9  -> 80 px ;  K=10, k=25 -> 240 px. Les deux saturent tres
    largement une image de 32 px : la portee nominale n'est PAS la contrainte.

  * RF EFFECTIF  : ce que les poids APPRIS regardent vraiment,
        m(x') = | d x1_pred(pixel central) / d x_t(x') |
    moyenne sur des images AFHQ reelles bruitees, pour une grille de t.
    Rayon effectif = moyenne de |x' - centre| ponderee par m.
    Reference : R_unif = 12.25 px, le rayon qu'aurait un RF parfaitement
    uniforme sur 32x32 (= agregation reellement globale).

Convention FM du repo : t=0 = bruit, t=1 = data (Kamb & Ganguli Fig. 4a
predit du coarse-to-fine, donc un RF large pres de t=0 qui retrecit vers t=1).

Le noyau de mesure est celui de erf_afhq_ckpt.py ; ce script-ci passe les deux
checkpoints en une fois et produit UNE figure comparative.

Usage
-----
    source ~/.venvs/unn/bin/activate
    python rf_convsccp_afhq32_K10.py --device cuda:0

Duree : ~3 min (2 modeles x 7 valeurs de t x 16 backwards).
Sorties -> rf_convsccp_afhq32_K10.png , rf_convsccp_afhq32_K10_metrics.txt
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_fid_cifar10 import CHANNELS, IMG_SIZE
from sample_checkpoint import resolve_checkpoint

S = IMG_SIZE
TS = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]
N_SAMP = 16
K_LAYERS = 10

CKPTS = [
    ("kernel 9x9",  "results_afhq32/ConvScCP_UNN_rgb_k9_K10_ic256_L1_LFO/latest.pt"),
    ("kernel 25x25", "results_afhq32/ConvScCP_UNN_rgb_k25_K10_ic256_L1_LFO/latest.pt"),
]


def real_images(cache, n, device):
    if not os.path.exists(cache):
        raise FileNotFoundError(f"{cache} absent — lance prepare_afhq_cats.py")
    d = torch.load(cache)
    return d["data"][:n].float().div_(127.5).sub_(1.0).to(device)


def uniform_radius(s):
    c = s // 2
    ys, xs = np.mgrid[0:s, 0:s]
    return float(np.sqrt((ys - c) ** 2 + (xs - c) ** 2).mean())


def radial_stats(m, kernel):
    """(rayon effectif px, fraction de masse hors empreinte d'un seul kernel)."""
    c = S // 2
    ys, xs = np.mgrid[0:S, 0:S]
    d = np.sqrt((ys - c) ** 2 + (xs - c) ** 2)
    w = m / (m.sum() + 1e-12)
    r = kernel // 2
    outside = (np.abs(ys - c) > r) | (np.abs(xs - c) > r)
    return float((w * d).sum()), float(w[outside].sum())


def erf_at_t(model, x1, t, device):
    acc = torch.zeros(S, S)
    for i in range(x1.shape[0]):
        x0 = torch.randn(1, CHANNELS, S, S, device=device)
        xt = ((1 - t) * x0 + t * x1[i:i + 1]).detach().requires_grad_(True)
        inp = torch.cat([xt.view(1, -1), torch.full((1, 1), t, device=device)], dim=1)
        out = model(inp).view(1, CHANNELS, S, S)
        model.zero_grad(set_to_none=True)
        out[0, :, S // 2, S // 2].sum().backward()
        acc += xt.grad.detach().abs().sum(1).view(S, S).cpu()
    return (acc / x1.shape[0]).numpy()


def measure(ckpt_path, x1, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, is_unet, name, weight_keys = resolve_checkpoint(ckpt, device)
    if is_unet:
        raise SystemExit(f"{ckpt_path} n'est pas un ConvScCP.")
    model.load_state_dict(ckpt[weight_keys["raw"]], strict=True)
    # train() : le ConvScCP renvoie x1_pred (le debruiteur, la quantite mesuree
    # par Kamb) ; en eval() il renverrait la vitesse. La variante L1 n'a ni norm
    # ni dropout, donc ce mode ne change rien d'autre.
    model.train()
    for prm in model.parameters():
        prm.requires_grad_(False)

    kernel = int(name.split("_k")[1].split("_")[0])
    step = ckpt.get("step", "?")
    maps, stats = {}, {}
    for t in TS:
        m = erf_at_t(model, x1, t, device)
        maps[t] = m
        stats[t] = radial_stats(m, kernel)
        print(f"    t={t:.2f}  R_eff={stats[t][0]:5.2f} px   "
              f"masse hors kernel {kernel}x{kernel} = {100*stats[t][1]:5.1f} %", flush=True)
    del model
    torch.cuda.empty_cache()
    return name, step, kernel, maps, stats


def main():
    p = argparse.ArgumentParser(description="RF nominal + effectif du ConvScCP K=10 sur AFHQ 32x32.")
    p.add_argument("--cache", default="./data/afhq_cat32_train.pt")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device)
    r_unif = uniform_radius(S)
    x1 = real_images(args.cache, N_SAMP, device)

    print(f"image {S}x{S} | K = {K_LAYERS} couches | R_unif (agregation globale) "
          f"= {r_unif:.2f} px\n", flush=True)

    res = []
    for label, path in CKPTS:
        print(f"[{label}] {path}", flush=True)
        res.append((label,) + measure(path, x1, device))
        print("", flush=True)

    # ---------------- figure ----------------
    nrow, ncol = len(res) + 1, len(TS)
    fig = plt.figure(figsize=(2.3 * ncol, 2.55 * nrow))
    gs = fig.add_gridspec(nrow, ncol, height_ratios=[1] * len(res) + [1.25],
                          hspace=0.42, wspace=0.06)

    for i, (label, name, step, kernel, maps, stats) in enumerate(res):
        r_nom = K_LAYERS * (kernel - 1)
        for j, t in enumerate(TS):
            ax = fig.add_subplot(gs[i, j])
            mn = maps[t] / maps[t].max()
            ax.imshow(np.log10(mn + 1e-4), cmap="magma", vmin=-4, vmax=0)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"t={t}   R={stats[t][0]:.1f}px", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{label}\nRF nominal {r_nom} px", fontsize=9)

    ax = fig.add_subplot(gs[len(res), :])
    for (label, name, step, kernel, maps, stats), col in zip(res, ["tab:blue", "tab:orange"]):
        rs = [stats[t][0] for t in TS]
        ax.plot(TS, rs, "o-", color=col,
                label=f"{label} — R_eff moyen {np.mean(rs):.1f} px "
                      f"({100*np.mean(rs)/r_unif:.0f} % de R_unif)")
    ax.axhline(r_unif, color="crimson", ls="--", lw=1.3,
               label=f"agregation globale (R_unif = {r_unif:.1f} px)")
    ax.axhline(4.0, color="gray", ls=":", lw=1.2, label="1 seule conv 9x9 (rayon 4 px)")
    ax.set_xlabel("t   (0 = bruit, 1 = data)")
    ax.set_ylabel("rayon effectif (px)")
    ax.set_ylim(0, r_unif * 1.2)
    ax.set_title("Portee EFFECTIVE mesuree — le RF nominal (80 / 240 px) est hors echelle : "
                 "sature l'image 32x32 dans les deux cas", fontsize=9)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)

    fig.suptitle("ConvScCP, K = 10 couches, AFHQ-chats 32x32 — RF effectif "
                 "|d x1_pred(centre) / d x_t| vs t", fontsize=12, y=0.995)
    plt.savefig("rf_convsccp_afhq32_K10.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    with open("rf_convsccp_afhq32_K10_metrics.txt", "w") as f:
        f.write(f"# ConvScCP K={K_LAYERS} couches, AFHQ-chats {S}x{S}, poids raw\n")
        f.write(f"# R_unif({S}x{S}) = {r_unif:.4f} px  (RF parfaitement uniforme)\n\n")
        for label, name, step, kernel, maps, stats in res:
            r_nom = K_LAYERS * (kernel - 1)
            rs = [stats[t][0] for t in TS]
            f.write(f"## {name}  step={step}  kernel={kernel}\n")
            f.write(f"RF_nominal_radius_px\t{r_nom}\t(= K*(k-1), prox L1 ponctuel)\n")
            f.write("t\tR_eff_px\tpct_of_R_unif\tmass_outside_kernel\n")
            for t in TS:
                er, ok = stats[t]
                f.write(f"{t}\t{er:.4f}\t{100*er/r_unif:.2f}\t{ok:.4f}\n")
            f.write(f"mean_R_eff\t{np.mean(rs):.4f}\tpct_of_R_unif\t"
                    f"{100*np.mean(rs)/r_unif:.2f}\n\n")

    print("-> rf_convsccp_afhq32_K10.png , rf_convsccp_afhq32_K10_metrics.txt")


if __name__ == "__main__":
    main()
