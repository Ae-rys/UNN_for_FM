# -*- coding: utf-8 -*-
"""
make_sccp_trajectory_figs.py
Figures de trajectoire ScCP pour le rapport, mise en page 6 images par ligne.

Relit trajectory.pt (produit par trajectory_convsccp.py) : aucun modele n'est
re-execute, seule la mise en page change.

Sortie : une grille 6x6 par modele.
  LIGNES    = 6 temps de l'EDO, sous-echantillonnes parmi les 11 enregistres
              (t = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
  COLONNES  = 6 etats primaux x^(k) du deroule, sous-echantillonnes parmi les K+1
              (k = 0, 3, 6, 9, 12, 15 pour K=15)

Chaque vignette est normalisee INDIVIDUELLEMENT et annotee de son amplitude reelle
max|x^(k)| : sans ca les couches du milieu seraient saturees, le deroule sortant de
[-1,1] d'un facteur ~4 avant de se contracter.

Usage :  python make_sccp_trajectory_figs.py [--sample 0]
"""
import argparse, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "results/mnist_report_200ep_K15"
OUT = "internship_report/images"
NCOL = 6
MODELS = [("ConvScCP_UNN_L1_LFO", "LFO"), ("ConvScCP_UNN_L1_LNO", "LNO")]


def grid66(it, ts, s, path, n_t=6, n_k=6):
    """Grille n_t x n_k : lignes = temps de l'EDO, colonnes = etats primaux x^(k)."""
    T, K1 = it.shape[0], it.shape[1]
    ti = np.linspace(0, T - 1, n_t).round().astype(int)      # 6 temps parmi 11
    ki = np.linspace(0, K1 - 1, n_k).round().astype(int)     # 6 couches parmi K+1
    fig, axes = plt.subplots(n_t, n_k, figsize=(n_k * 1.30, n_t * 1.52))
    for r, j in enumerate(ti):
        for c, k in enumerate(ki):
            ax = axes[r, c]
            im = it[j, k, s, 0].numpy()
            ax.imshow(im, cmap="gray", vmin=im.min(), vmax=im.max())
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"$k={k}$", fontsize=11, pad=5)
            if c == 0:
                ax.set_ylabel(f"$t={float(ts[j]):.1f}$", fontsize=11, labelpad=6)
            ax.set_xlabel(f"{np.abs(im).max():.3g}", fontsize=7.5, labelpad=1.5, color="0.4")
    fig.subplots_adjust(wspace=0.05, hspace=0.30)
    fig.savefig(path, dpi=175, bbox_inches="tight"); plt.close(fig)
    print("ecrit :", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=0)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for sub, tag in MODELS:
        f = os.path.join(RUN, sub, "trajectory", "trajectory.pt")
        if not os.path.exists(f):
            print(f"MANQUANT : {f}"); continue
        d = torch.load(f, map_location="cpu", weights_only=False)
        grid66(d["iterates"], d["ts"], a.sample,
               os.path.join(OUT, f"{tag}_iterates_K15_200ep.png"))


if __name__ == "__main__":
    main()
