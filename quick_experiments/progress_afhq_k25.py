# -*- coding: utf-8 -*-
"""
progress_afhq_k25.py
Repond a la question "est-ce que ca va converger vers de belles images avec plus
de pas ?" par deux mesures, sur le run AFHQ-chats k25 :

1. PROGRESSION (progress_afhq_k25.png)
   Meme bruit initial echantillonne depuis chaque checkpoint archive
   (10k, 20k, 30k, 40k...), poids BRUTS. Une ligne par checkpoint : on lit
   directement la vitesse a laquelle la qualite bouge encore. Si les lignes
   30k et 40k sont quasi identiques, le budget de steps n'est plus le facteur
   limitant (c'est l'archi), et prolonger ne servira a rien.

   On quantifie ce que l'oeil voit avec la distance L2 moyenne entre lignes
   consecutives (meme bruit -> la difference vient des poids seuls).

2. MEMORISATION (nn_afhq_k25.png)
   Avec 5 653 images d'entrainement, "de belles images" peut vouloir dire
   "des copies du train". Pour chaque echantillon du dernier checkpoint on
   affiche le plus proche voisin L2 dans le train, avec la distance. Une
   distance NN qui s'effondre = memorisation, pas generalisation
   (cf. serie ELS / Kamb).

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python progress_afhq_k25.py
    CUDA_VISIBLE_DEVICES=0 python progress_afhq_k25.py --n 8 --steps 200

Duree : ~40 s par checkpoint (n=6, 100 steps Euler) sur un 2080 Ti partage.
"""

import argparse
import glob
import os
import re

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_fid_cifar10 import CHANNELS, IMG_SIZE
from sample_checkpoint import resolve_checkpoint, generate_from_weights, _to_img

RUN_DIR = "results_afhq32/ConvScCP_UNN_rgb_k25_K10_ic256_L1_LFO"
TRAIN_CACHE = "./data/afhq_cat32_train.pt"


def find_checkpoints(run_dir):
    """Checkpoints archives tries par step, + latest.pt s'il est plus avance."""
    ckpts = []
    for path in glob.glob(os.path.join(run_dir, "ckpt_step_*.pt")):
        m = re.search(r"ckpt_step_(\d+)\.pt$", path)
        if m:
            ckpts.append((int(m.group(1)), path))
    ckpts.sort()

    latest = os.path.join(run_dir, "latest.pt")
    if os.path.exists(latest):
        # latest.pt bouge pendant l'entrainement : on ne l'ajoute que s'il est
        # STRICTEMENT plus avance que la derniere archive, sinon c'est un doublon.
        step = torch.load(latest, map_location="cpu", weights_only=False).get("step", -1)
        if not ckpts or step > ckpts[-1][0]:
            ckpts.append((step, latest))
    return ckpts


@torch.no_grad()
def nearest_neighbours(samples, train, device, chunk=512, skip_idx=None):
    """Pour chaque echantillon, (indice, distance L2) du plus proche voisin du train.

    Distances en pixels [-1,1], moyennees par pixel (RMSE) pour etre lisibles
    independamment de la resolution.

    skip_idx : indices train a masquer, un par echantillon. Indispensable quand
    les sondes SONT des images du train (baseline) : sinon chacune se retrouve
    elle-meme a distance 0 et la baseline ne veut rien dire.
    """
    s = samples.to(device).flatten(1)
    best_d = torch.full((len(s),), float("inf"), device=device)
    best_i = torch.zeros(len(s), dtype=torch.long, device=device)
    d = s.shape[1]
    for start in range(0, len(train), chunk):
        block = train[start:start + chunk].to(device).flatten(1)
        dist = torch.cdist(s, block) / (d ** 0.5)          # RMSE par pixel
        if skip_idx is not None:
            local = skip_idx.to(device) - start
            hit = (local >= 0) & (local < block.shape[0])
            if hit.any():
                dist[hit.nonzero(as_tuple=True)[0], local[hit]] = float("inf")
        m, idx = dist.min(dim=1)
        upd = m < best_d
        best_d[upd] = m[upd]
        best_i[upd] = idx[upd] + start
    return best_i.cpu(), best_d.cpu()


def main():
    p = argparse.ArgumentParser(description="Progression + memorisation du run AFHQ k25.")
    p.add_argument("--run-dir", type=str, default=RUN_DIR)
    p.add_argument("--n", type=int, default=6, help="Images par checkpoint.")
    p.add_argument("--steps", type=int, default=100, help="Steps Euler.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out-progress", type=str, default="progress_afhq_k25.png")
    p.add_argument("--out-nn", type=str, default="nn_afhq_k25.png")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpts = find_checkpoints(args.run_dir)
    if not ckpts:
        raise FileNotFoundError(f"aucun checkpoint dans {args.run_dir}")
    print(f"{len(ckpts)} checkpoints : {[s for s, _ in ckpts]}", flush=True)

    # meme bruit initial partout -> toute difference vient des poids
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    x0 = torch.randn(args.n, CHANNELS, IMG_SIZE, IMG_SIZE, generator=g).to(device)

    rows, model = [], None
    for step, path in ckpts:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if model is None:
            model, is_unet, name, keys = resolve_checkpoint(ckpt, device)
            print(f"{name} | {sum(q.numel() for q in model.parameters())/1e6:.2f}M params",
                  flush=True)
        print(f"  step {step:,} ...", flush=True)
        imgs, _ = generate_from_weights(ckpt, "raw", model, is_unet, keys, x0,
                                        "euler", args.steps, device)
        rows.append((step, imgs.cpu()))
        del ckpt

    # --- figure 1 : progression ---------------------------------------------
    nrow, ncol = len(rows), args.n
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.6 * ncol, 1.75 * nrow), squeeze=False)
    deltas = []
    for r, (step, imgs) in enumerate(rows):
        arr = _to_img(imgs)
        for c in range(ncol):
            axes[r, c].imshow(arr[c])
            axes[r, c].axis("off")
        label = f"step {step//1000}k"
        if r > 0:
            # ecart pixel moyen avec le checkpoint precedent (meme bruit)
            d = (imgs - rows[r - 1][1]).pow(2).mean().sqrt().item()
            deltas.append((rows[r - 1][0], step, d))
            label += f"\nΔ vs {rows[r-1][0]//1000}k : {d:.3f}"
        axes[r, 0].set_ylabel(label)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for sp in axes[r, 0].spines.values():
            sp.set_visible(False)
    fig.suptitle("AFHQ-chats k25 — meme bruit, poids bruts, a travers l'entrainement",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(args.out_progress, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> {args.out_progress}", flush=True)
    print("\n  ecart RMSE entre checkpoints consecutifs (meme bruit) :", flush=True)
    for a, b, d in deltas:
        print(f"    {a:>6,} -> {b:>6,} : {d:.4f}", flush=True)

    # --- figure 2 : plus proches voisins du train ---------------------------
    if not os.path.exists(TRAIN_CACHE):
        print(f"[!] {TRAIN_CACHE} absent -> check memorisation saute.", flush=True)
        return
    d = torch.load(TRAIN_CACHE)
    train = d["data"].float().div(127.5).sub(1.0)
    last_step, last_imgs = rows[-1]
    idx, dist = nearest_neighbours(last_imgs, train, device)

    # baseline : distance d'une VRAIE image de train a l'image de train la plus
    # proche AUTRE QU'ELLE-MEME (skip_idx) -> l'echelle de ce qu'est une distance
    # "normale" dans ce dataset. Sans ce masque la baseline vaut 0 (auto-match).
    n_base = min(256, len(train))
    pidx = torch.randperm(len(train))[:n_base]
    _, base = nearest_neighbours(train[pidx], train, device, skip_idx=pidx)

    fig, axes = plt.subplots(2, args.n, figsize=(1.7 * args.n, 4.0), squeeze=False)
    gen = _to_img(last_imgs)
    nn = _to_img(train[idx])
    for c in range(args.n):
        axes[0, c].imshow(gen[c])
        axes[1, c].imshow(nn[c])
        for r in (0, 1):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for sp in axes[r, c].spines.values():
                sp.set_visible(False)
        axes[1, c].set_xlabel(f"d={dist[c]:.3f}", fontsize=8)
    axes[0, 0].set_ylabel("genere", fontsize=9)
    axes[1, 0].set_ylabel("NN train", fontsize=9)
    fig.suptitle(f"Memorisation — step {last_step:,} | NN genere→train {dist.mean():.3f} "
                 f"vs baseline train→train(self exclu) {base.mean():.3f}  —  0 = copie exacte",
                 fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out_nn, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {args.out_nn}", flush=True)
    print(f"\n  NN dist genere->train : {dist.mean():.4f} "
          f"(min {dist.min():.4f}) | baseline train->train : {base.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
