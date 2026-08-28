# -*- coding: utf-8 -*-
"""
fid_dim_study.py
Faut-il reduire la dimension des features du mini-FID (128 -> 64) ?

Le diagnostic de fid_floor_mnist.py a montre 67/128 unites mortes et un rang
effectif de 61 : la covariance est singuliere et le plancher a N=2000 vaut ~4.
Tentation : reentrainer le classifieur avec fc1 en 64-d. Mais un plancher plus
bas ne vaut rien s'il ecrase aussi les ECARTS entre modeles -- le seul critere
utile est le rapport signal/plancher.

Le script compare donc plusieurs espaces de features :
  - clf128        : le classifieur actuel (results/mnist_classifier/clf.pt)
  - clf128-alive  : idem, en jetant les colonnes mortes (aucun reentrainement,
                    corrige juste la singularite de sqrtm)
  - clf64         : un classifieur identique mais fc1 en 64-d (reentraine)
  - pca-k         : projection PCA des features 128-d sur k axes (k = 8..61),
                    ajustee sur le train
et, dans chacun, mesure a N=2000 :
  - le PLANCHER   : FID(train, test) entre vraies images ;
  - trois SIGNAUX : FID(train, test degrade) pour trois modes d'echec typiques
                    d'un generateur -- flou, bruit, et mode-drop (seulement les
                    chiffres 0-4, i.e. diversite effondree) ;
  - le RAPPORT signal/plancher, qui dit combien la metrique discrimine.

Sortie : results/fid_floor_mnist/{fid_dim_study.png, fid_dim_study.md}

Usage :
    python fid_dim_study.py            # CPU, ~5-8 min (dont ~3 min pour clf64)
    python fid_dim_study.py --device cuda
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fid_floor_mnist import get_datasets, as_tensor, fid_from_feats, OUT_DIR
from mnist_metrics import MNISTClassifier, train_or_load_classifier, _embed_all


class MNISTClassifier64(MNISTClassifier):
    """Meme reseau, penultieme couche en `width` dimensions au lieu de 128."""

    def __init__(self, width=64):
        super().__init__()
        self.fc1 = nn.Linear(64 * 7 * 7, width)
        self.fc2 = nn.Linear(width, 10)


def train_clf64(train_ds, device, width=64, epochs=3,
                ckpt_path="results/mnist_classifier/clf64.pt"):
    """Meme recette que train_or_load_classifier, mais sur l'architecture etroite."""
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    clf = MNISTClassifier64(width).to(device)
    if os.path.exists(ckpt_path):
        clf.load_state_dict(torch.load(ckpt_path, map_location=device))
        clf.eval()
        print(f"[clf{width}] charge depuis {ckpt_path}", flush=True)
        return clf

    loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    clf.train()
    for ep in range(epochs):
        n_ok, n_tot = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = clf(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            n_ok += (logits.argmax(1) == y).sum().item(); n_tot += y.size(0)
        print(f"  [clf{width}] epoch {ep+1}/{epochs} acc={n_ok/n_tot:.4f}", flush=True)
    clf.eval()
    torch.save(clf.state_dict(), ckpt_path)
    return clf


@torch.no_grad()
def test_accuracy(clf, x, y, device, bs=1000):
    preds = [clf(x[i:i + bs].to(device)).argmax(1).cpu() for i in range(0, len(x), bs)]
    return (torch.cat(preds) == y).float().mean().item()


def gaussian_blur(x, sigma=1.0, ksize=5):
    """Flou gaussien separable, en conservant la plage [-1,1]."""
    ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
    k = torch.exp(-ax ** 2 / (2 * sigma ** 2)); k /= k.sum()
    pad = ksize // 2
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"), k.view(1, 1, -1, 1))
    return x.clamp(-1, 1)


def build_degradations(x_test, y_test, seed=0):
    """Trois modes d'echec generatifs classiques, appliques au TEST set."""
    g = torch.Generator().manual_seed(seed)
    return {
        "flou (sigma=1)": gaussian_blur(x_test, sigma=1.0),
        "bruit (sigma=0.3)": (x_test + 0.3 * torch.randn(x_test.shape, generator=g)).clamp(-1, 1),
        "mode-drop (0-4)": x_test[y_test <= 4],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--n", type=int, default=2000, help="Taille de lot (protocole standard).")
    p.add_argument("--n-seeds", type=int, default=5)
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.perf_counter()

    train_ds, test_ds = get_datasets()
    x_train, y_train = as_tensor(train_ds)
    x_test, y_test = as_tensor(test_ds)
    degraded = build_degradations(x_test, y_test)
    print(f"[data] degradations : " +
          ", ".join(f"{k} ({len(v)})" for k, v in degraded.items()), flush=True)

    # ------------------------------------------------------------ classifieurs
    clf128 = train_or_load_classifier(
        DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2), device)
    print(f"[clf128] acc test = {test_accuracy(clf128, x_test, y_test, device):.4f}",
          flush=True)
    clf64 = train_clf64(train_ds, device)
    acc64 = test_accuracy(clf64, x_test, y_test, device)
    print(f"[clf64]  acc test = {acc64:.4f}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    # -------------------------------------------------- embeddings (une fois)
    def embed_all_sets(clf):
        d = {"train": _embed_all(clf, x_train, device),
             "test": _embed_all(clf, x_test, device)}
        for name, imgs in degraded.items():
            d[name] = _embed_all(clf, imgs, device)
        return d

    E128, E64 = embed_all_sets(clf128), embed_all_sets(clf64)
    alive = E128["train"].max(axis=0) > 0
    print(f"[feat] clf128 : {alive.sum()}/128 vivantes | "
          f"clf64 : {(E64['train'].max(axis=0) > 0).sum()}/64 vivantes", flush=True)

    # PCA ajustee sur le train 128-d, appliquee a tous les lots
    mu = E128["train"].mean(0)
    _, _, Vt = np.linalg.svd(E128["train"] - mu, full_matrices=False)

    def pca_space(k):
        return {name: (f - mu) @ Vt[:k].T for name, f in E128.items()}

    spaces = {
        "clf128": (128, E128),
        "clf128-alive": (int(alive.sum()), {n: f[:, alive] for n, f in E128.items()}),
        f"clf64": (64, E64),
    }
    for k in (8, 16, 32, 61):
        spaces[f"pca-{k}"] = (k, pca_space(k))

    # ------------------------------------------------------------- evaluation
    rng = np.random.default_rng(0)
    # memes indices pour tous les espaces -> comparaison a bruit d'echantillonnage egal
    idx = [(rng.choice(len(x_train), args.n, replace=False),
            rng.choice(len(x_test), args.n, replace=False),
            {n: rng.choice(len(v), min(args.n, len(v)), replace=False)
             for n, v in degraded.items()})
           for _ in range(args.n_seeds)]

    rows = []
    for space, (dim, E) in spaces.items():
        floors, sigs = [], {k: [] for k in degraded}
        for i_tr, i_te, i_dg in idx:
            ref = E["train"][i_tr]
            floors.append(fid_from_feats(ref, E["test"][i_te]))
            for name in degraded:
                sigs[name].append(fid_from_feats(ref, E[name][i_dg[name]]))
        row = dict(space=space, dim=dim,
                   floor=float(np.mean(floors)), floor_std=float(np.std(floors)))
        for name in degraded:
            row[name] = float(np.mean(sigs[name]))
            row[name + "_ratio"] = row[name] / row["floor"]
        rows.append(row)
        print(f"[{space:13s} d={dim:3d}] plancher={row['floor']:6.2f} +/- {row['floor_std']:.2f}"
              + "".join(f" | {n}={row[n]:7.1f} (x{row[n+'_ratio']:5.1f})" for n in degraded),
              flush=True)

    # ---------------------------------------------------------------- sorties
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(degraded)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    xs = np.arange(len(rows))
    axs[0].bar(xs, [r["floor"] for r in rows],
               yerr=[r["floor_std"] for r in rows], capsize=3, color="tab:red")
    axs[0].set_ylabel(f"plancher train-vs-test a N={args.n}")
    axs[0].set_title("Plancher (bas = mieux)")
    for w, name in enumerate(names):
        axs[1].bar(xs + (w - 1) * 0.27, [r[name + "_ratio"] for r in rows],
                   width=0.27, label=name)
    axs[1].set_ylabel("signal / plancher")
    axs[1].set_yscale("log")
    axs[1].set_title("Pouvoir discriminant (haut = mieux)")
    axs[1].legend(fontsize=8)
    for ax in axs:
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{r['space']}\n(d={r['dim']})" for r in rows],
                           fontsize=8, rotation=30, ha="right")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Choix de l'espace de features du mini-FID MNIST", fontsize=12)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fid_dim_study.png")
    fig.savefig(fig_path, dpi=140)

    md = [f"# Dimension des features du mini-FID : 128 vs 64 vs PCA\n",
          f"N={args.n} par lot, {args.n_seeds} tirages (memes indices pour tous les espaces). "
          f"clf64 acc test = {acc64:.4f}.\n",
          "| espace | d | plancher | " + " | ".join(f"{n} (ratio)" for n in names) + " |",
          "|---|---|---|" + "---|" * len(names)]
    for r in rows:
        md.append(f"| {r['space']} | {r['dim']} | {r['floor']:.2f} +/- {r['floor_std']:.2f} | "
                  + " | ".join(f"{r[n]:.1f} (x{r[n+'_ratio']:.1f})" for n in names) + " |")
    md.append("\nLecture : baisser la dimension baisse mecaniquement le plancher, mais ce "
              "n'est un gain que si le RATIO signal/plancher monte. Comparer la colonne "
              "ratio, pas la colonne plancher.\n")
    md_path = os.path.join(OUT_DIR, "fid_dim_study.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"\n[out] {fig_path}\n[out] {md_path}\n[done] {time.perf_counter()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
