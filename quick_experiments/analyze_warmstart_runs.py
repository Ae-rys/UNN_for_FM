# -*- coding: utf-8 -*-
"""
analyze_warmstart_runs.py
Depouille la paire baseline/warmstart de run_warmstart_mnist.py et TRANCHE la cause de
la degradation, au lieu de la supposer.

Le tableau de run_warmstart_mnist.py donne warmstart nettement moins bon que baseline,
avec une loss d'entrainement quasi identique. Deux explications concurrentes :

  (a) EXPOSURE BIAS. A l'entrainement, u_prev vient d'une passe FROIDE sur x_{t-dt} du
      chemin conditionnel : une seule passe, jamais rechainee. A l'echantillonnage, u
      vient d'une passe DEJA CHAUDE au pas precedent, elle-meme chaude, etc. : une
      recursion de 100 pas que l'entrainement n'a jamais vue.
  (b) L'entrainement self-conditionne a simplement abime le modele.

Le discriminant est un troisieme echantillonneur, "sc1", qui reproduit EXACTEMENT la
condition d'entrainement : a chaque pas, u_prev est recalcule par une passe FROIDE sur
l'etat ODE precedent, sans aucun rechainage.

    cold : u = 0 a chaque pas                        (echantillonneur actuel)
    warm : u_n = u_K du pas n-1                      (recursion, ce que fait `sample`)
    sc1  : u_n = u_K d'une passe FROIDE sur x_{n-1}  (= la condition d'entrainement)

Si warmstart/sc1 rattrape baseline -> (a) : c'est la recursion qui casse, pas
l'entrainement. Si warmstart/sc1 reste mauvais -> (b).

Le script trace aussi le profil des rayons r(t) du prox par couche : c'est ce qui
distinguait le vieux ckpt MNIST (r ~ 1e-23, dual annihile, cf. diag_u0_forgetting.py)
d'un modele au canal dual vivant.

Usage
-----
    source ~/.venvs/unn/bin/activate
    CUDA_VISIBLE_DEVICES=1 python analyze_warmstart_runs.py --n-eval 2000

Sorties (--outdir, defaut results/warmstart_mnist/) :
    analysis_table.md    les 3 echantillonneurs x les 2 modeles
    analysis_radii.png   profil des rayons du prox par couche, baseline vs warmstart
    analysis_samples.png grilles comparees (meme bruit)
"""
import argparse
import os
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from run_warmstart_mnist import (N_STEPS, load_run, get_loader, grad_energy,
                                 prox_radii, sample, save_grid)


@torch.no_grad()
def sample_sc1(model, shape, device, n_steps=N_STEPS, x0=None):
    """Echantillonneur "self-conditionne a un pas" : u_prev est recalcule a chaque pas
    par une passe FROIDE sur l'etat ODE PRECEDENT. Aucune recursion -> c'est exactement
    la distribution de u vue a l'entrainement (une passe froide sur x_{t-dt}).
    Cout : 2 forwards par pas."""
    x = torch.randn(shape, device=device) if x0 is None else x0.clone()
    dt = 1.0 / n_steps
    x_prev, t_prev = None, None
    for n in range(n_steps):
        t = torch.full((shape[0], 1), n * dt, device=device)
        if x_prev is None:
            u_prev = model.cold_dual(x)                       # 1er pas : froid (comme a l'inference)
        else:
            _, u_prev = model(torch.cat([x_prev, t_prev], dim=-1),
                              u_init=model.cold_dual(x_prev), return_u=True)
        v, _ = model(torch.cat([x, t], dim=-1), u_init=u_prev, return_u=True)
        x_prev, t_prev = x, t
        x = x + v / n_steps
    return x


@torch.no_grad()
def generate_many(model, n, device, mode, batch, seed, n_steps=N_STEPS):
    """mode : 'cold' | 'warm' | 'sc1'. Meme graine -> memes x0 pour toutes les configs."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    outs, done = [], 0
    while done < n:
        b = min(batch, n - done)
        x0 = torch.randn(b, 784, generator=g).to(device)
        if mode == "sc1":
            xf = sample_sc1(model, (b, 784), device, n_steps=n_steps, x0=x0)
        else:
            xf, _ = sample(model, (b, 784), device, warm=(mode == "warm"),
                           n_steps=n_steps, x0=x0)
        outs.append(xf.cpu())
        done += b
    return torch.cat(outs, 0)[:n]


def main():
    p = argparse.ArgumentParser(description="Depouillement baseline/warmstart + test sc1.")
    p.add_argument("--outdir", type=str, default="results/warmstart_mnist")
    p.add_argument("--tags", type=str, default="baseline,warmstart")
    p.add_argument("--modes", type=str, default="cold,warm,sc1")
    p.add_argument("--n-eval", type=int, default=2000)
    p.add_argument("--n-grid", type=int, default=8, help="Echantillons affiches par ligne.")
    p.add_argument("--grid-only", action="store_true",
                   help="Ne calcule que les grilles d'images (pas de mini-FID) : quelques "
                        "secondes au lieu de ~9 min.")
    p.add_argument("--eval-batch", type=int, default=500)
    p.add_argument("--digit", type=int, default=0)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    tags = args.tags.split(",")
    modes = args.modes.split(",")

    models, losses, cfgs = {}, {}, {}
    for tag in tags:
        models[tag], losses[tag], cfgs[tag] = load_run(tag, args, device)

    from mnist_metrics import train_or_load_classifier, mini_fid
    _, full_dataset, dataset_f = get_loader(args.digit, args.batch, args.seed)
    clf = train_or_load_classifier(
        DataLoader(full_dataset, batch_size=256, shuffle=True, num_workers=2), device)
    n_ref = min(args.n_eval, len(dataset_f))
    g_ref = torch.Generator().manual_seed(args.seed)      # lot reel reproductible (cf. run_warmstart_mnist)
    real = next(iter(DataLoader(dataset_f, batch_size=n_ref, shuffle=True,
                                generator=g_ref)))[0][:n_ref]
    ge_real = grad_energy(real)
    std_real = float(real.view(n_ref, -1).std(dim=0).mean().item())
    print(f"[analyse] reference {n_ref} images  GE_reel={ge_real:.4f} std_reel={std_real:.4f}",
          flush=True)

    rows, grids = [], {}
    n_gen = args.n_grid if args.grid_only else args.n_eval
    for tag in tags:
        for mode in modes:
            t0 = time.perf_counter()
            s = generate_many(models[tag], n_gen, device, mode,
                              min(args.eval_batch, n_gen), args.seed)
            imgs = s.view(-1, 1, 28, 28).clamp(-1, 1)
            grids[(tag, mode)] = imgs[:args.n_grid]
            row = dict(tag=tag, mode=mode,
                       fid=float("nan") if args.grid_only else mini_fid(clf, imgs, real, device),
                       ge=grad_energy(imgs), std=float(s.std(dim=0).mean().item()),
                       loss=losses[tag][-1])
            rows.append(row)
            print(f"  [{tag}/{mode}] mini-FID={row['fid']:.2f}  GE={row['ge']:.4f}  "
                  f"std={row['std']:.4f}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- tableau
    lines = [f"# Depouillement warm-start — digit {args.digit}, K={cfgs[tags[0]]['K']} "
             f"ic={cfgs[tags[0]]['internal_channel']}, {cfgs[tags[0]]['epochs']} epochs, "
             f"N_STEPS={N_STEPS}, {args.n_eval} echantillons\n",
             "cold = u remis a zero (echantillonneur actuel) | warm = u rechaine (recursion) | "
             "sc1 = u recalcule par une passe FROIDE sur l'etat precedent (= condition "
             "d'entrainement, sans recursion)\n",
             f"| modele | echantillonneur | mini-FID | GE (reel {ge_real:.3f}) | "
             f"std_gen (reel {std_real:.3f}) | loss finale |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['tag']} | {r['mode']} | {r['fid']:.2f} | {r['ge']:.4f} | "
                     f"{r['std']:.4f} | {r['loss']:.4f} |")
    table = "\n".join(lines)
    if not args.grid_only:           # en grid-only les FID sont NaN : ne pas ecraser le vrai tableau
        with open(os.path.join(args.outdir, "analysis_table.md"), "w") as f:
            f.write(table + "\n")
        print("\n" + table, flush=True)

    # ---------------------------------------------------------------- grilles
    # Une ligne par (modele, echantillonneur) + une ligne de VRAIES images en bas :
    # sans reference, l'oeil compare des defauts entre eux au lieu de les comparer au reel.
    keys = list(grids)
    n_c = args.n_grid
    fig, axes = plt.subplots(len(keys) + 1, n_c,
                             figsize=(0.78 * n_c, 1.05 * (len(keys) + 1)), squeeze=False)
    fid_of = {(r["tag"], r["mode"]): r["fid"] for r in rows}
    for r, key in enumerate(keys):
        for c in range(n_c):
            axes[r, c].imshow(grids[key][c, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        f = fid_of[key]
        lbl = f"{key[0]}\n{key[1]}" + ("" if np.isnan(f) else f"\nFID {f:.1f}")
        axes[r, 0].set_ylabel(lbl, fontsize=7)
    perm = torch.randperm(real.shape[0], generator=torch.Generator().manual_seed(args.seed))
    for c in range(n_c):                                   # ligne de reference
        axes[-1, c].imshow(real[perm[c], 0].numpy(), cmap="gray", vmin=-1, vmax=1)
        axes[-1, c].set_xticks([]); axes[-1, c].set_yticks([])
    axes[-1, 0].set_ylabel("MNIST\nreel", fontsize=7)
    fig.suptitle("Meme bruit initial — modele x echantillonneur (derniere ligne : vraies images)",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(args.outdir, "analysis_samples.png"), dpi=140)
    plt.close(fig)

    # ------------------------------------------------- profil des rayons du prox
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    for tag in tags:
        for t_val, ls in [(0.1, ":"), (0.5, "-"), (0.9, "--")]:
            rs = prox_radii(models[tag], t_val)
            axs[0].semilogy(range(1, len(rs) + 1), rs, ls, marker="o", ms=3,
                            label=f"{tag}, t={t_val}")
    axs[0].axhline(1e-6, color="r", lw=1, ls="-", alpha=0.6, label="seuil couche morte")
    axs[0].set_xlabel("couche k"); axs[0].set_ylabel("rayon r(t) du prox l1")
    axs[0].set_title("Profil des rayons : canal dual vivant ou annihile ?", fontsize=10)
    axs[0].grid(alpha=0.3, which="both"); axs[0].legend(fontsize=7, ncol=2)

    xs = np.arange(len(rows))
    if args.grid_only:      # pas de FID calcule : ne pas ecrire une figure de barres vides
        plt.close(fig)
        print(f"\n[analyse] grid-only -> {args.outdir}/analysis_samples.png", flush=True)
        return
    axs[1].bar(xs, [r["fid"] for r in rows],
               color=["#4C78A8" if r["tag"] == "baseline" else "#F58518" for r in rows])
    axs[1].set_xticks(xs)
    axs[1].set_xticklabels([f"{r['tag']}\n{r['mode']}" for r in rows], fontsize=8)
    axs[1].set_ylabel("mini-FID (bas = mieux)")
    axs[1].set_title("Qualite par modele x echantillonneur", fontsize=10)
    axs[1].grid(alpha=0.3, axis="y")
    for x, r in zip(xs, rows):
        axs[1].text(x, r["fid"], f"{r['fid']:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "analysis_radii.png"), dpi=130)
    plt.close(fig)

    print(f"\n[analyse] -> {args.outdir}/analysis_table.md, analysis_radii.png, "
          f"analysis_samples.png", flush=True)


if __name__ == "__main__":
    main()
