# -*- coding: utf-8 -*-
"""
fid_floor_mnist.py
Plancher du mini-FID de mnist_metrics.py : que vaut la metrique entre DEUX
lots de vraies images MNIST (train vs test, tous les chiffres) ?

C'est la valeur de reference qui manquait pour lire les mini-FID des runs
(warm-start, sweeps, ...) : un modele parfait ne descend pas a 0, il descend
a ce plancher-la, et le plancher DEPEND DU NOMBRE D'ECHANTILLONS (la covariance
128x128 est mal estimee a petit N, ce qui gonfle la distance de Frechet).

Le script fait trois choses :
  1. verifie que le classifieur cache (results/mnist_classifier/clf.pt) est bon
     -- accuracy train et test, matrice de confusion resumee ;
  2. mesure le mini-FID train-vs-test a N croissant (plusieurs seeds) ;
  3. mesure un CONTROLE train-vs-train (deux sous-ensembles disjoints du meme
     jeu) : la difference avec train-vs-test isole le biais de taille finie
     du vrai ecart de distribution entre les deux splits.

Sortie : results/fid_floor_mnist/{fid_floor_vs_n.png, fid_floor_mnist.md}

Usage :
    python fid_floor_mnist.py                 # CPU, ~2 min
    python fid_floor_mnist.py --device cuda
"""
import argparse
import os
import time

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from mnist_metrics import train_or_load_classifier, frechet_distance, _embed_all

OUT_DIR = "results/fid_floor_mnist"


def get_datasets():
    """MNIST train + test, memes transforms que le pipeline FM ([-1,1])."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train = torchvision.datasets.MNIST(root="./data", train=True,
                                       download=True, transform=transform)
    test = torchvision.datasets.MNIST(root="./data", train=False,
                                      download=True, transform=transform)
    return train, test


def as_tensor(dataset):
    """(N,1,28,28) float dans [-1,1] + labels, en memoire."""
    x = dataset.data.float().div(255.0).sub(0.5).div(0.5).unsqueeze(1)
    return x, dataset.targets.clone()


@torch.no_grad()
def accuracy(clf, x, y, device, batch_size=1000):
    preds = []
    for i in range(0, x.shape[0], batch_size):
        preds.append(clf(x[i:i + batch_size].to(device)).argmax(1).cpu())
    preds = torch.cat(preds)
    acc = (preds == y).float().mean().item()
    per_class = [((preds == y) & (y == c)).sum().item() / max((y == c).sum().item(), 1)
                 for c in range(10)]
    return acc, per_class, preds


def fid_from_feats(fa, fb):
    mu_a, sig_a = fa.mean(0), np.cov(fa, rowvar=False)
    mu_b, sig_b = fb.mean(0), np.cov(fb, rowvar=False)
    return frechet_distance(mu_a, sig_a, mu_b, sig_b)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="CPU par defaut : le job est petit et les GPU servent aux runs.")
    p.add_argument("--n-seeds", type=int, default=5,
                   help="Repetitions du sous-echantillonnage par valeur de N.")
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[500, 1000, 2000, 5000, 10000],
                   help="Tailles de lot testees (bornees par 10k = taille du test set).")
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.perf_counter()

    train_ds, test_ds = get_datasets()
    x_train, y_train = as_tensor(train_ds)
    x_test, y_test = as_tensor(test_ds)
    print(f"[data] train {tuple(x_train.shape)}  test {tuple(x_test.shape)}  "
          f"range [{x_train.min():.2f}, {x_train.max():.2f}]", flush=True)

    # ------------------------------------------------------- 1. sanite du CNN
    clf = train_or_load_classifier(
        DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2), device)

    acc_tr, _, _ = accuracy(clf, x_train, y_train, device)
    acc_te, per_class, preds_te = accuracy(clf, x_test, y_test, device)
    print(f"\n[clf] accuracy train = {acc_tr:.4f}   test = {acc_te:.4f}", flush=True)
    print("[clf] accuracy test par chiffre : " +
          "  ".join(f"{c}:{a:.3f}" for c, a in enumerate(per_class)), flush=True)
    worst = int(np.argmin(per_class))
    print(f"[clf] pire chiffre : {worst} ({per_class[worst]:.3f})", flush=True)
    if acc_te < 0.97:
        print("[clf] ATTENTION : accuracy test < 0.97, les features sont douteuses "
              "comme espace de mini-FID.", flush=True)

    # ------------------------------------------------- 2. embeddings une fois
    feat_train = _embed_all(clf, x_train, device)
    feat_test = _embed_all(clf, x_test, device)
    print(f"\n[feat] train {feat_train.shape}  test {feat_test.shape}  "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

    # Diagnostic de rang : les features sortent d'un ReLU, des unites peuvent etre
    # mortes -> covariance singuliere -> sqrtm bruite et FID gonfle a petit N.
    dead = int((feat_train.max(axis=0) == 0).sum())
    sv = np.linalg.svd(feat_train - feat_train.mean(0), compute_uv=False)
    rank = int((sv > sv[0] * 1e-6).sum())
    print(f"[feat] unites mortes : {dead}/128   rang effectif : {rank}/128", flush=True)

    fid_full = fid_from_feats(feat_train, feat_test)
    print(f"\n[FID] train COMPLET (60000) vs test COMPLET (10000) = {fid_full:.3f}",
          flush=True)

    # ------------------------------------------- 3. plancher en fonction de N
    rng = np.random.default_rng(0)
    rows = []
    for n in args.sizes:
        if n > feat_test.shape[0]:
            continue
        tt, tv = [], []
        for _ in range(args.n_seeds):
            i_tr = rng.choice(feat_train.shape[0], n, replace=False)
            i_te = rng.choice(feat_test.shape[0], n, replace=False)
            tt.append(fid_from_feats(feat_train[i_tr], feat_test[i_te]))
            # controle : deux lots disjoints du MEME split (train) -> pur biais
            j = rng.choice(feat_train.shape[0], 2 * n, replace=False)
            tv.append(fid_from_feats(feat_train[j[:n]], feat_train[j[n:]]))
        rows.append(dict(n=n,
                         tt_mean=float(np.mean(tt)), tt_std=float(np.std(tt)),
                         tv_mean=float(np.mean(tv)), tv_std=float(np.std(tv))))
        print(f"[FID] N={n:6d}  train-vs-test = {rows[-1]['tt_mean']:6.3f} "
              f"+/- {rows[-1]['tt_std']:.3f}   train-vs-train = "
              f"{rows[-1]['tv_mean']:6.3f} +/- {rows[-1]['tv_std']:.3f}", flush=True)

    # ------------------- 3b. reference reelle COMPLETE au lieu d'un lot apparie
    # Le protocole actuel compare N generes a N reels. Mais les features reelles
    # sont deja toutes calculees : prendre les 60k du train comme reference ne
    # coute RIEN et supprime la moitie du bruit d'estimation (celle qui vient du
    # cote reel). Seul le cote genere reste limite par N, qui lui coute cher.
    print("\n[FID] reference reelle appariee (N reels) vs complete (60000 reels) :",
          flush=True)
    rows_ref = []
    for n in args.sizes:
        if n > feat_test.shape[0]:
            continue
        matched, full = [], []
        for _ in range(args.n_seeds):
            i_te = rng.choice(feat_test.shape[0], n, replace=False)
            i_tr = rng.choice(feat_train.shape[0], n, replace=False)
            matched.append(fid_from_feats(feat_train[i_tr], feat_test[i_te]))
            full.append(fid_from_feats(feat_train, feat_test[i_te]))
        rows_ref.append(dict(n=n,
                             m_mean=float(np.mean(matched)), m_std=float(np.std(matched)),
                             f_mean=float(np.mean(full)), f_std=float(np.std(full))))
        r = rows_ref[-1]
        print(f"[FID] N_gen={n:6d}  appariee = {r['m_mean']:6.3f} +/- {r['m_std']:.3f}"
              f"   complete = {r['f_mean']:6.3f} +/- {r['f_std']:.3f}", flush=True)

    # ------------------------------------------------------------- 4. sorties
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(ns, [r["tt_mean"] for r in rows], yerr=[r["tt_std"] for r in rows],
                marker="o", capsize=3, label="train vs test (vrais deux splits)")
    ax.errorbar(ns, [r["tv_mean"] for r in rows], yerr=[r["tv_std"] for r in rows],
                marker="s", capsize=3, ls="--",
                label="train vs train (controle : biais de taille finie seul)")
    ax.errorbar([r["n"] for r in rows_ref], [r["f_mean"] for r in rows_ref],
                yerr=[r["f_std"] for r in rows_ref], marker="^", capsize=3, ls="-.",
                color="tab:green",
                label="N generes vs reference reelle COMPLETE (60k, gratuit)")
    ax.axhline(fid_full, color="k", ls=":", lw=1,
               label=f"train 60k vs test 10k = {fid_full:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("N echantillons par lot")
    ax.set_ylabel("mini-FID (features du CNN MNIST)")
    ax.set_title(f"Plancher du mini-FID sur MNIST (tous chiffres)\n"
                 f"CNN test acc = {acc_te:.4f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "fid_floor_vs_n.png")
    fig.savefig(fig_path, dpi=140)
    print(f"\n[out] figure -> {fig_path}", flush=True)

    md = [f"# Plancher du mini-FID MNIST (tous les chiffres)\n",
          f"Classifieur : `results/mnist_classifier/clf.pt` — accuracy "
          f"train {acc_tr:.4f}, test {acc_te:.4f} "
          f"(pire chiffre : {worst}, {per_class[worst]:.3f}). "
          f"Features 128-d : {dead} unites mortes, rang effectif {rank}/128.\n",
          f"**train complet (60000) vs test complet (10000) : "
          f"mini-FID = {fid_full:.3f}**\n",
          f"Lots de meme taille, {args.n_seeds} tirages, moyenne +/- ecart-type :\n",
          "| N | train vs test | train vs train (controle) |",
          "|---|---|---|"]
    for r in rows:
        md.append(f"| {r['n']} | {r['tt_mean']:.3f} +/- {r['tt_std']:.3f} "
                  f"| {r['tv_mean']:.3f} +/- {r['tv_std']:.3f} |")
    md.append("\n## Reference reelle appariee (N) vs complete (60000)\n")
    md.append("| N generes | reference appariee | reference complete |")
    md.append("|---|---|---|")
    for r in rows_ref:
        md.append(f"| {r['n']} | {r['m_mean']:.3f} +/- {r['m_std']:.3f} "
                  f"| {r['f_mean']:.3f} +/- {r['f_std']:.3f} |")
    md.append("\nPrendre TOUT le train comme reference ne coute rien (features deja "
              "calculees, mises en cache) et retire la part du bruit qui vient du cote "
              "reel. A ne changer qu'une fois, en re-evaluant tout : les deux colonnes "
              "ne sont pas comparables entre elles.\n")
    md.append("\nLecture : a N fixe, le plancher train-vs-test est la valeur qu'un "
              "modele PARFAIT atteindrait avec ce protocole. Tout mini-FID rapporte "
              "ailleurs doit etre lu comme un ecart AU-DESSUS de ce plancher, a N egal. "
              "Le controle train-vs-train mesure la part du plancher qui n'est que du "
              "bruit d'estimation (aucun ecart de distribution) ; ce qui reste entre "
              "les deux courbes est le vrai decalage train/test de MNIST.\n")
    md_path = os.path.join(OUT_DIR, "fid_floor_mnist.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"[out] tableau -> {md_path}", flush=True)
    print(f"[done] {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
