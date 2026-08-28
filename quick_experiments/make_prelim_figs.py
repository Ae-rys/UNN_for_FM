# -*- coding: utf-8 -*-
"""
make_prelim_figs.py
Illustrations de la section Preliminaries du rapport. Chaque figure porte sur une
notion REUTILISEE plus loin, pas sur une decoration :

  prox_soft_threshold.png   Def. "Proximal Operator" : le prox de gamma|.| est le
                            seuillage doux. C'est la non-linearite de LISTA.
  moreau_decomposition.png  Prop. "Moreau Decomposition" : soft(x,g) + clip(x,-g,g) = x.
                            C'est LA justification du prox dual de nos ScCP/DFB, qui
                            est un clamp — et donc du champ impair corrige par w_bias.
  continuity_equation.png   Def. "Continuity Equation" : p_t transportee par v_t, avec
                            des trajectoires d'echantillons. Le cadre de Flow Matching.
  unrolling_ista.png        Sec. "Unrolled Neural Networks" : ISTA (parametres fixes,
                            K -> infini) vs UNN (parametres appris, K fixe).

Conventions identiques aux autres figures du rapport : hauteur commune PAIR_H, pas de
titre interne (la legende LaTeX porte l'info), labels lisibles.

Usage :  python make_prelim_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "internship_report/images"
PAIR_H = 4.0
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12})


def fig_soft_threshold():
    x = np.linspace(-3, 3, 601)
    fig, ax = plt.subplots(figsize=(1.35 * PAIR_H, PAIR_H))
    ax.plot(x, x, "--", color="0.6", lw=1.2, label=r"identity ($\gamma=0$)")
    for g, c in zip([0.5, 1.0, 1.5], ["#1f77b4", "#2ca02c", "#d62728"]):
        ax.plot(x, np.sign(x) * np.maximum(np.abs(x) - g, 0), lw=2, color=c,
                label=rf"$\gamma={g}$")
        ax.axvspan(-g, g, color=c, alpha=0.045)
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("$x$"); ax.set_ylabel(r"$\mathrm{prox}_{\gamma|\cdot|}(x)$")
    ax.legend(fontsize=9.5, loc="upper left"); ax.grid(alpha=0.3)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
    fig.tight_layout(); p = os.path.join(OUT, "prox_soft_threshold.png")
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig); print("ecrit :", p)


def fig_moreau():
    """x = prox_{gamma g}(x) + gamma * prox_{g*/gamma}(x/gamma), pour g = |.| :
    le second terme est exactement clip(x, -gamma, gamma)."""
    x = np.linspace(-3, 3, 601); g = 1.0
    primal = np.sign(x) * np.maximum(np.abs(x) - g, 0)     # seuillage doux
    dual = np.clip(x, -g, g)                                # projection sur la boule l_inf
    fig, ax = plt.subplots(figsize=(1.35 * PAIR_H, PAIR_H))
    ax.plot(x, primal, lw=2.2, color="#1f77b4",
            label=r"$\mathrm{prox}_{\gamma g}(x)$  (soft-thresholding)")
    ax.plot(x, dual, lw=2.2, color="#d62728",
            label=r"$\gamma\,\mathrm{prox}_{g^*/\gamma}(x/\gamma)$  (clipping)")
    ax.plot(x, primal + dual, "--", lw=1.6, color="k", label=r"their sum $=x$")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("$x$"); ax.set_ylabel("output")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
    fig.tight_layout(); p = os.path.join(OUT, "moreau_decomposition.png")
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig); print("ecrit :", p)


def fig_continuity():
    """Chemin gaussien du Flow Matching : x_t = (1-t)x0 + t x1, x0~N(0,1), x1~N(2,0.3^2)
    independants => p_t = N(2t, (1-t)^2 + 0.09 t^2), exactement."""
    m1, s1 = 2.0, 0.3
    t = np.linspace(0, 1, 220); xs = np.linspace(-4, 4, 400)
    T, X = np.meshgrid(t, xs)
    mu = m1 * T; var = (1 - T) ** 2 + (s1 ** 2) * T ** 2
    P = np.exp(-((X - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)
    fig, ax = plt.subplots(figsize=(1.35 * PAIR_H, PAIR_H))
    ax.pcolormesh(T, X, P, shading="auto", cmap="Blues")
    rng = np.random.default_rng(0)
    for _ in range(9):                                   # trajectoires d'echantillons
        a, b = rng.normal(0, 1), rng.normal(m1, s1)
        ax.plot(t, (1 - t) * a + t * b, color="#d62728", lw=1.1, alpha=0.85)
    ax.set_xlabel("$t$"); ax.set_ylabel("$x$")
    ax.set_xlim(0, 1); ax.set_ylim(-4, 4)
    ax.text(0.03, 3.3, r"$p_0=\mathcal{N}(0,1)$", fontsize=10)
    ax.text(0.70, 3.3, r"$p_1$", fontsize=10)
    fig.tight_layout(); p = os.path.join(OUT, "continuity_equation.png")
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig); print("ecrit :", p)


def fig_unrolling():
    fig, ax = plt.subplots(figsize=(2.0 * PAIR_H, 0.92 * PAIR_H))
    ax.axis("off"); ax.set_xlim(0, 11.2); ax.set_ylim(0, 4.4)

    def block(cx, cy, label, sub, fc):
        ax.add_patch(FancyBboxPatch((cx - 0.72, cy - 0.42), 1.44, 0.84,
                    boxstyle="round,pad=0.06", fc=fc, ec="0.25", lw=1.2))
        ax.text(cx, cy + 0.11, label, ha="center", va="center", fontsize=11)
        ax.text(cx, cy - 0.20, sub, ha="center", va="center", fontsize=8.5, color="0.3")

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.2, color="0.3"))

    # --- ISTA : parametres FIXES, on itere jusqu'a convergence ---
    ax.text(0.15, 3.95, r"ISTA — fixed parameters, $k\to\infty$",
            fontsize=12, fontweight="bold")
    for i, cx in enumerate([1.5, 3.6, 5.7]):
        block(cx, 3.0, rf"$T_{{\Theta}}$", r"$W,\ \tau$  fixed", "#dfe7f2")
        if i: arrow(cx - 2.1 + 0.72, cx - 0.72, 3.0)
    arrow(0.35, 0.78, 3.0); ax.text(0.16, 3.28, "$z$", fontsize=11)
    arrow(6.42, 6.9, 3.0); ax.text(7.0, 3.0, r"$\cdots\ \to\ \hat{x}$",
                                   va="center", fontsize=11)

    # --- UNN : K FIXE, parametres APPRIS et distincts par couche ---
    ax.text(0.15, 1.75, r"Unrolled network — $K$ fixed, parameters learned",
            fontsize=12, fontweight="bold")
    for i, cx in enumerate([1.5, 3.6, 5.7]):
        block(cx, 0.8, rf"$T_{{\theta_{i+1}}}$", rf"$W_{i+1},\ \tau_{i+1}$  learned", "#f7dfd0")
        if i: arrow(cx - 2.1 + 0.72, cx - 0.72, 0.8)
    arrow(0.35, 0.78, 0.8); ax.text(0.16, 1.08, "$z$", fontsize=11)
    arrow(6.42, 6.9, 0.8); ax.text(7.0, 0.8, r"$\hat{x}$ after $K$ layers",
                                   va="center", fontsize=11)
    # fleche de transition placee A DROITE des blocs : au centre elle chevauchait
    # le titre de la seconde rangee.
    ax.annotate("", xy=(9.35, 1.30), xytext=(9.35, 2.50),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#d62728"))
    ax.text(9.62, 1.90, "truncate\n+ learn", fontsize=10, color="#d62728",
            va="center", ha="left", linespacing=1.3)

    fig.tight_layout(); p = os.path.join(OUT, "unrolling_ista.png")
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig); print("ecrit :", p)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_soft_threshold(); fig_moreau(); fig_continuity(); fig_unrolling()
