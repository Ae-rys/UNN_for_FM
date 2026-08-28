# -*- coding: utf-8 -*-
"""
wiener_view.py
A quoi ressemble le "debruiteur lineaire optimal" qui sert de reference partout.

Definition
----------
Le meilleur estimateur de x1 parmi toutes les fonctions AFFINES de x_t. Avec
x_t = (1-t) x0 + t x1 et x0 ~ N(0, I), la solution LMMSE s'ecrit

    x1_hat = mu + t C (t^2 C + (1-t)^2 I)^-1 (x_t - t mu)

ou mu et C sont la moyenne et la covariance des images d'entrainement. Dans la
base propre de C, cela revient a multiplier chaque mode par le gain

    g_i(t) = t lambda_i / (t^2 lambda_i + (1-t)^2)

Les images naturelles ont un spectre en ~1/f^2 : les modes de grande variance
sont les BASSES frequences. Le gain les conserve et ecrase les autres — cet
estimateur est donc un flou adaptatif au niveau de bruit. C'est ce qui rend la
reference interessante : elle chiffre ce qu'un flou bien regle rapporte deja,
et tout ce qu'un reseau gagne au-dela vient de la structure non gaussienne.

Le script produit la grille des reconstructions et le profil des gains, sur les
memes images de validation et le meme bruit fixe que denoise_probe.py.

Usage
-----
    source ~/.venvs/unn/bin/activate
    python wiener_view.py

Sorties -> report_figs/fig_wiener.png
"""

import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = "./data/afhq_cat32_train.pt"
T_LIST = [0.2, 0.4, 0.6, 0.8]
N_SHOW = 6
N_VAL = 512
SEED = 0


def main():
    os.makedirs("report_figs", exist_ok=True)
    d = torch.load(CACHE)
    x = d["data"].float().div_(127.5).sub_(1.0)
    g = torch.Generator().manual_seed(SEED)
    x = x[torch.randperm(x.shape[0], generator=g)]
    C_, S = x.shape[1], x.shape[2]
    flat = x.reshape(x.shape[0], -1).double()
    xva, xtr = flat[:N_VAL], flat[N_VAL:]

    mu = xtr.mean(dim=0, keepdim=True)
    xc = xtr - mu
    n, dim = xc.shape
    cov = xc.T @ xc / n
    lam, U = torch.linalg.eigh(cov)
    lam = lam.clamp_min(0)
    print(f"covariance {dim}x{dim} sur {n} images ; "
          f"lambda max={lam.max():.4f} min={lam.min():.2e}", flush=True)

    gv = torch.Generator().manual_seed(SEED + 1234)
    var = float(((xva - xva.mean(dim=0, keepdim=True)) ** 2).mean())

    rows, labels, gains = [], [], {}
    for t in sorted(T_LIST):
        x0 = torch.randn(xva.shape[0], dim, generator=gv).double()
        xt = (1 - t) * x0 + t * xva
        gain = t * lam / (t ** 2 * lam + (1 - t) ** 2)
        pred = mu + ((xt - t * mu) @ U * gain) @ U.T
        mse = float(((pred - xva) ** 2).mean())
        gains[t] = gain.flip(0).numpy()            # du mode le plus energetique au moins
        rows += [xt[:N_SHOW], pred[:N_SHOW]]
        labels += [f"$x_t$   t={t:g}", f"Wiener  nmse {mse/var:.4f}"]
        print(f"  t={t:g}  nmse={mse/var:.4f}", flush=True)
    rows.append(xva[:N_SHOW]); labels.append("verite $x_1$")

    nr = len(rows)
    fig = plt.figure(figsize=(1.55 * N_SHOW + 5.2, 1.42 * nr))
    gs = fig.add_gridspec(nr, N_SHOW + 3, wspace=0.06, hspace=0.06)
    for r, (row, lab) in enumerate(zip(rows, labels)):
        imgs = (row.reshape(-1, C_, S, S).float() * 0.5 + 0.5).clamp(0, 1)
        for c in range(N_SHOW):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(imgs[c].permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(lab, fontsize=8.5, rotation=0, ha="right", va="center")

    ax = fig.add_subplot(gs[:, N_SHOW + 1:])
    for t in sorted(T_LIST):
        ax.plot(gains[t], lw=1.8, label=f"t={t:g}")
    ax.set_xscale("log")
    ax.set_xlabel("mode propre (du plus energetique au moins)", fontsize=9)
    ax.set_ylabel("gain appliqué", fontsize=9)
    ax.set_title("Le filtre garde les modes dominants\net écrase les autres", fontsize=9.5)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    fig.suptitle("Débruiteur linéaire optimal — la référence du banc", fontsize=11)
    plt.savefig("report_figs/fig_wiener.png", dpi=105, bbox_inches="tight")
    plt.close(fig)
    print("\n-> report_figs/fig_wiener.png", flush=True)


if __name__ == "__main__":
    main()
