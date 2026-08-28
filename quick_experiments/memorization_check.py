# -*- coding: utf-8 -*-
"""
memorization_check.py
Diagnostic de MEMORISATION d'un checkpoint FM (ConvScCP_UNN ou SmallUNet) sur MNIST.

Idee : un modele qui MEMORISE recrache des images du TRAIN. On genere N samples, on
cherche pour chacun son plus proche voisin (PPV) dans le TRAIN MNIST (L2 pixel), et on
compare la DISTRIBUTION de ces distances a une REFERENCE calibree : les images de TEST
held-out ont elles aussi un PPV dans le train, a une distance "normale" (ni copie, ni
bruit). C'est cette reference qui donne un sens a la distance brute.

  dist(gen->train) ~ dist(test->train)  => generalisation saine
  dist(gen->train) << dist(test->train) => copie / memorisation

/!\ Piege (archis localement equivariantes type ScCP) : la L2 pixel n'est PAS robuste
aux translations. Un sample peut etre une recombinaison/translation locale du train,
loin en L2 brut mais "copie" localement -- et inversement. D'ou deux garde-fous :
  (1) --shift S : PPV robuste a de petites translations (min de la distance sur les
      decalages entiers de -S..S en x et y) ;
  (2) TOUJOURS regarder la planche sample|PPV a l'oeil (memorization_pairs.png) : c'est
      le juge de paix, la metrique n'est qu'un tri.

Sorties (dans --outdir, defaut <dir du ckpt>/memorization/) :
  memorization_pairs.png    ligne gen / ligne PPV-train, distance annotee, triees du + proche
  memorization_hist.png     histogramme dist(gen->train) vs dist(test->train)
  memorization_metrics.txt  medianes, ratio, %gen sous la mediane test, #PPV uniques

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python memorization_check.py \
        --ckpt results/OT-CFM/ConvScCP_UNN_L1_LNO/model.pt --n 64

    # PPV robuste a +-2 px de translation (recommande pour ScCP)
    python memorization_check.py --ckpt <path> --n 64 --shift 2
"""
import argparse
import math
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T

from models.architectures import ConvScCP_UNN, SmallUNetX1
from generate_digits import infer_config, generate

S = 28
DIM = S * S


def build_model(ckpt, device):
    """Charge le state_dict et reconstruit le bon modele (config auto-detectee)."""
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if "layers.0.W_weight" in sd:                            # ConvScCP_UNN
        cfg = infer_config(sd)
        model = ConvScCP_UNN(dim=cfg["dim"], K=cfg["K"], internal_channel=cfg["internal_channel"],
                             use_Unet=cfg["use_Unet"], version=cfg["version"],
                             w_bias=cfg["w_bias"], in_channels=cfg["in_channels"],
                             img_size=cfg["img_size"], kernel_size=cfg["kernel"],
                             prox_w=cfg["prox_w"]).to(device)
    elif "inc.conv1.weight" in sd:                           # baseline SmallUNet
        base_ch, in_ch = sd["inc.conv1.weight"].shape[0], sd["inc.conv1.weight"].shape[1]
        cfg = dict(dim=in_ch * 28 * 28, internal_channel=base_ch, img_size=28,
                   in_channels=in_ch, K="-", version="SmallUNet")
        model = SmallUNetX1(in_channels=in_ch, out_channels=in_ch, base_ch=base_ch).to(device)
    else:
        raise ValueError(f"Type de modele non reconnu depuis les cles : {list(sd)[:5]}")
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, cfg


def load_mnist(train, n, device):
    """n premieres images MNIST (train ou test), normalisees en [-1,1] -> (n, DIM)."""
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    ds = torchvision.datasets.MNIST("./data", train=train, download=True, transform=tf)
    n = min(n, len(ds))
    X = torch.stack([ds[i][0] for i in range(n)]).view(n, DIM)
    return X.to(device)


@torch.no_grad()
def nn_to_train(queries, train, shift=0, chunk=64):
    """Plus proche voisin de chaque `query` dans `train` (L2). Renvoie (dist_min, idx_min).

    shift>0 : distance ROBUSTE aux translations -> pour chaque decalage (dx,dy) entier
    dans [-shift, shift]^2 on decale la QUERY (bord rempli a -1 = fond MNIST) et on garde
    le min. Neutralise une "copie translatee" que la L2 brute croirait eloignee.
    """
    Q = queries.view(-1, 1, S, S)
    offsets = [(0, 0)] if shift == 0 else [(dx, dy)
               for dx in range(-shift, shift + 1) for dy in range(-shift, shift + 1)]
    best_d = torch.full((Q.shape[0],), float("inf"), device=Q.device)
    best_i = torch.zeros(Q.shape[0], dtype=torch.long, device=Q.device)
    for dx, dy in offsets:
        Qs = torch.roll(Q, shifts=(dy, dx), dims=(2, 3))
        if dx or dy:                                          # remet le bord decale au fond (-1)
            if dy > 0:   Qs[:, :, :dy, :] = -1
            elif dy < 0: Qs[:, :, dy:, :] = -1
            if dx > 0:   Qs[:, :, :, :dx] = -1
            elif dx < 0: Qs[:, :, :, dx:] = -1
        Qf = Qs.reshape(Q.shape[0], DIM)
        for a in range(0, Qf.shape[0], chunk):                # chunk : evite une grosse matrice
            d, i = torch.cdist(Qf[a:a + chunk], train).min(dim=1)
            m = d < best_d[a:a + chunk]
            best_d[a:a + chunk][m] = d[m]
            best_i[a:a + chunk][m] = i[m]
    return best_d, best_i


def _sq(ax, img):
    ax.imshow(img.view(S, S).clamp(-1, 1).cpu(), cmap="gray", vmin=-1, vmax=1)
    ax.axis("off")


def pairs_figure(gen, dist, nn_idx, train, path, n_show):
    """Ligne du haut = samples generes (les + proches du train a gauche) ; ligne du bas =
    leur PPV dans le train, distance annotee. Un sample quasi identique a son PPV = copie."""
    order = torch.argsort(dist)[:n_show]
    m = len(order)
    fig, axes = plt.subplots(2, m, figsize=(1.0 * m, 2.2), squeeze=False)
    for c, j in enumerate(order.tolist()):
        _sq(axes[0, c], gen[j]);            axes[0, c].set_title(f"{dist[j]:.1f}", fontsize=7)
        _sq(axes[1, c], train[nn_idx[j]])
    axes[0, 0].set_ylabel("gen",  fontsize=8, rotation=0, ha="right", va="center")
    axes[1, 0].set_ylabel("PPV\ntrain", fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle("Sample genere (haut) vs son plus proche voisin du TRAIN (bas)\n"
                 "trie du plus proche au moins proche — chiffre = distance L2", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  -> {path}")


def hist_figure(d_gen, d_test, path, shift):
    """Distributions superposees : si dist(gen->train) est decalee vers 0 par rapport a
    la reference test->train, le modele colle au train (memorisation)."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    lo = 0.0
    hi = float(max(d_gen.max(), d_test.max()) * 1.02)
    bins = torch.linspace(lo, hi, 40).tolist()
    ax.hist(d_test.cpu().numpy(), bins=bins, alpha=0.55, label="test->train (reference)", color="C0")
    ax.hist(d_gen.cpu().numpy(),  bins=bins, alpha=0.55, label="gen->train", color="C3")
    ax.axvline(float(d_test.median()), color="C0", ls="--", lw=1)
    ax.axvline(float(d_gen.median()),  color="C3", ls="--", lw=1)
    tag = "" if shift == 0 else f" (PPV robuste +-{shift}px)"
    ax.set_xlabel(f"distance L2 au plus proche voisin du train{tag}")
    ax.set_ylabel("nb d'images"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Memorisation : gen->train decale vers 0 vs reference test->train ?", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  -> {path}")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Diagnostic de memorisation d'un checkpoint FM sur MNIST.")
    p.add_argument("--ckpt", type=str, required=True, help="checkpoint a diagnostiquer")
    p.add_argument("--n", type=int, default=64, help="nb de samples generes a tester")
    p.add_argument("--ntrain", type=int, default=60000, help="taille du dictionnaire TRAIN (defaut: tout)")
    p.add_argument("--ntest", type=int, default=1000, help="nb d'images TEST pour la reference")
    p.add_argument("--shift", type=int, default=0,
                   help="PPV robuste aux translations +-shift px (recommande ~2 pour ScCP)")
    p.add_argument("--n-show", type=int, default=12, help="nb de paires affichees dans la planche")
    p.add_argument("--batch", type=int, default=128, help="taille de lot pour la generation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="", help="defaut : <dir du ckpt>/memorization/")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg = build_model(args.ckpt, device)
    print(f"Config auto-detectee : {cfg}")
    print(f"Checkpoint : {args.ckpt}  ({sum(q.numel() for q in model.parameters()):,} params)")
    if cfg["in_channels"] != 1:
        raise SystemExit("Ce diagnostic est ecrit pour MNIST mono-canal (in_channels=1).")

    train = load_mnist(True,  args.ntrain, device)
    test  = load_mnist(False, args.ntest,  device)
    print(f"Train (dictionnaire) : {tuple(train.shape)} | Test (reference) : {tuple(test.shape)}")

    # generation
    print(f"Generation de {args.n} samples (dopri5)...")
    gen = generate(model, args.n, cfg["dim"], device, args.batch, args.seed).to(device)

    # PPV au train pour les samples ET pour la reference test
    d_gen,  i_gen  = nn_to_train(gen,  train, shift=args.shift)
    d_test, _      = nn_to_train(test, train, shift=args.shift)

    med_gen, med_test = float(d_gen.median()), float(d_test.median())
    ratio = med_gen / med_test
    # fraction de samples plus proches du train que l'image test mediane : > ~0.5 suspect
    frac_below = float((d_gen < med_test).float().mean())
    n_unique = len(torch.unique(i_gen))                       # #images train distinctes touchees

    print("\n===== RESULTATS =====")
    print(f"  mediane dist(gen ->train)  = {med_gen:.2f}")
    print(f"  mediane dist(test->train)  = {med_test:.2f}   (reference held-out)")
    print(f"  ratio gen/test             = {ratio:.2f}   (~1 sain, <<1 memorisation)")
    print(f"  % samples sous la med. test = {100*frac_below:.0f}%   (>~50% suspect)")
    print(f"  PPV train distincts        = {n_unique}/{args.n}   (petit => copie des memes images)")

    outdir = args.outdir or os.path.join(os.path.dirname(args.ckpt), "memorization")
    os.makedirs(outdir, exist_ok=True)
    pairs_figure(gen.cpu(), d_gen.cpu(), i_gen.cpu(), train.cpu(),
                 os.path.join(outdir, "memorization_pairs.png"), min(args.n_show, args.n))
    hist_figure(d_gen, d_test, os.path.join(outdir, "memorization_hist.png"), args.shift)

    with open(os.path.join(outdir, "memorization_metrics.txt"), "w") as f:
        f.write(f"# memorisation {args.ckpt}\n")
        f.write(f"# n={args.n} ntrain={args.ntrain} ntest={args.ntest} shift={args.shift}\n")
        f.write(f"median_dist_gen_train\t{med_gen:.4f}\n")
        f.write(f"median_dist_test_train\t{med_test:.4f}\n")
        f.write(f"ratio_gen_over_test\t{ratio:.4f}\n")
        f.write(f"frac_gen_below_test_median\t{frac_below:.4f}\n")
        f.write(f"n_unique_train_neighbors\t{n_unique}\t/{args.n}\n")
    print(f"  -> {os.path.join(outdir, 'memorization_metrics.txt')}")
    print(f"\nTermine. Tout est dans {outdir}/")


if __name__ == "__main__":
    main()
