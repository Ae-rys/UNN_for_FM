# -*- coding: utf-8 -*-
"""
make_2moons_report_figs.py
Regenere les figures et les valeurs numeriques 2-moons du rapport, depuis les
checkpoints de results/2moons_report_50ep/ (produits par run_2moons.py --pairs 10:32).

Produit :
  internship_report/images/LNO_ot_2moons_x_pred_epochs_100.png
      generation par ScCP LNO, labels LISIBLES (repond au TODO du rapport).
  internship_report/images/2moons_tmax_check.png
      MEME modele echantillonne avec Euler-100 (ce que fait run_2moons) et avec
      Euler-20 (le seul grille coherente avec la plage d'entrainement t~U(0.05,0.95)).
  stdout : le tableau chiffre pour tab:quantitative.

Pourquoi la seconde figure : les modeles plats de 2-moons n'utilisent PAS le
mecanisme t_max. Ils divisent par `torch.clamp(1-t, 0.05)` en dur, et le sampler
est un Euler-100 sur [0,1] : il evalue donc t = 0.96 ... 1.00, hors de la plage
d'entrainement t ~ U(0.05, 0.95), la ou le clamp ecrase la vitesse d'un facteur
(1-t)/0.05 (x0 a t=1). La borne 0.95 correspond a un Euler-20, pas a un Euler-100.

Usage :  python make_2moons_report_figs.py
"""
import argparse
import os
import re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
from models import ScCP_UNN

RUN_DIR = "results/2moons_report_100ep"
OUT_DIR = "internship_report/images"
DIM, SEED, N_GEN = 2, 42, 2000


def target_moons(n=2000, seed=SEED):
    X, _ = make_moons(n_samples=n, noise=0.05, random_state=seed)
    X = torch.tensor(X, dtype=torch.float32)
    return ((X - X.mean(0)) / X.std(0)).numpy()


@torch.no_grad()
def euler_generate(model, n_steps, n=N_GEN, seed=SEED):
    """Euler explicite sur [0,1] en n_steps pas — la meme grille que torchdyn
    avec t_span=linspace(0,1,n_steps+1), mais ecrite a la main pour pouvoir
    inspecter les t reellement evalues."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, DIM, generator=g)
    ts, dt = [], 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((n, 1), i * dt)
        ts.append(i * dt)
        x = x + model(torch.cat([x, t], dim=-1)) * dt
    return x.numpy(), ts


def load(name, version):
    """K et dual_dim sont lus DANS le nom du run (ex. ..._K5_dual32), pas supposes :
    le meme script sert pour les balayages de profondeur."""
    m_ = re.match(r".*_K(\d+)_dual(\d+)$", name)
    K, dual = (int(m_.group(1)), int(m_.group(2))) if m_ else (10, 32)
    m = ScCP_UNN(dim=DIM, K=K, dual_dim=dual, version=version, prox_type="l1")
    m.load_state_dict(torch.load(os.path.join(RUN_DIR, name, "model.pt"),
                                 map_location="cpu", weights_only=False))
    return m.eval()


def scatter(ax, gen, ref, title):
    ax.scatter(ref[:, 0], ref[:, 1], s=5, alpha=0.25, c="gray", label="target (2-moons)")
    ax.scatter(gen[:, 0], gen[:, 1], s=5, alpha=0.55, c="steelblue", label="generated")
    if title:
        ax.set_title(title, fontsize=13)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.tick_params(labelsize=11)
    ax.set_xlabel("$x_1$", fontsize=12); ax.set_ylabel("$x_2$", fontsize=12)
    ax.legend(markerscale=3, fontsize=10, loc="upper right")


PAIR_H = 4.8   # hauteur (pouces) PARTAGEE par la courbe de loss et le scatter :
               # c'est ce qui permet un \includegraphics[height=...] identique des
               # deux cotes, donc des cadres d'axes alignes une fois cote a cote.


def loss_curve(out_path):
    """Courbe de loss par famille, echelle log, legende DANS le cadre, SANS titre
    interne (la legende LaTeX porte l'info). Meme hauteur que le scatter."""
    import plot_losses as pl
    orig_plot, orig_save = pl._plot_by_algo, pl._save_or_show

    def inside(ax, data, title, legend_outside=False, legend_fontsize=6.5, **kw):
        orig_plot(ax, data, "", legend_outside=False, legend_fontsize=9, **kw)
        ax.get_legend().get_frame().set_alpha(0.9)

    def log_save(fig, path):
        for ax in fig.get_axes():
            if ax.lines:
                ax.set_yscale("log")
                ax.grid(True, which="both", alpha=0.3)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print("ecrit :", path)

    pl._plot_by_algo, pl._save_or_show = inside, log_save
    try:
        pl.plot_losses_by_algo(os.path.abspath(RUN_DIR), output_path=out_path,
                               figsize=(6.4, PAIR_H))
    finally:
        pl._plot_by_algo, pl._save_or_show = orig_plot, orig_save


def main():
    global RUN_DIR
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default=RUN_DIR, help="dossier du run a depouiller")
    p.add_argument("--suffix", type=str, default="", help="suffixe des PNG produits, ex '_K5'")
    args = p.parse_args()
    RUN_DIR, sfx = args.run, args.suffix

    os.makedirs(OUT_DIR, exist_ok=True)
    ref = target_moons()
    lno = [d for d in sorted(os.listdir(RUN_DIR)) if d.startswith("ScCP") and "_LNO_" in d][0]
    model = load(lno, "LNO")

    loss_curve(os.path.join(OUT_DIR, f"2moons_algo{sfx}.png"))

    # --- figure du rapport : generation ScCP LNO, labels lisibles -------------
    gen100, ts = euler_generate(model, 100)
    fig, ax = plt.subplots(figsize=(PAIR_H, PAIR_H))
    scatter(ax, gen100, ref, "")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"LNO_ot_2moons_x_pred_epochs_100{sfx}.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    print("ecrit :", p)

    # --- controle t_max : Euler-100 (actuel) vs Euler-20 (coherent) ----------
    gen20, ts20 = euler_generate(model, 20)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    scatter(axes[0], gen100, ref, f"Euler-100 (actuel) — $t$ max evalue = {max(ts):.2f}")
    scatter(axes[1], gen20,  ref, f"Euler-20 ($t_{{max}}=0.95$) — $t$ max evalue = {max(ts20):.2f}")
    fig.suptitle("Effet de la grille d'echantillonnage : la plage d'entrainement est "
                 r"$t\sim U(0.05,\,0.95)$", fontsize=12)
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, f"2moons_tmax_check{sfx}.png")
    fig.savefig(p2, dpi=140); plt.close(fig)
    print("ecrit :", p2)

    n_out = sum(t > 0.95 for t in ts)
    print(f"\nEuler-100 : {n_out} des {len(ts)} pas evaluent t > 0.95 "
          f"(hors plage d'entrainement, clamp actif)")
    print(f"Euler-20  : {sum(t > 0.95 for t in ts20)} pas hors plage")

    # --- tableau chiffre pour tab:quantitative -------------------------------
    cfg = re.search(r"_K(\d+)_dual(\d+)$", lno)
    print(f"\n--- 2-Moons, 100 epochs, K={cfg.group(1)}, m={cfg.group(2)} "
          f"(tab:quantitative) ---")
    print(f"{'Model':<28}{'#Params':>9}{'Final loss':>12}{'W2 error':>11}")
    with open(os.path.join(RUN_DIR, "summary.txt")) as f:
        rows = [l.split("\t") for l in f.read().strip().split("\n")[1:]]
    for r in rows:
        print(f"{r[0]:<28}{int(r[1]):>9,}{float(r[2]):>12.4f}{float(r[3]):>11.4f}")
    n_iter = 39 * 100
    print(f"\nIterations reelles = 39 batches/epoch x 100 epochs = {n_iter:,} "
          f"(pas 2M : 2M = nombre d'ECHANTILLONS vus sur 200 epochs)")


if __name__ == "__main__":
    main()
