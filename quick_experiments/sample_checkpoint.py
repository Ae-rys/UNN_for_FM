# -*- coding: utf-8 -*-
"""
sample_checkpoint.py
Genere des images depuis un de NOS checkpoints
(run_cifar10_torchcfm_recipe.py / run_imagenet32.py), au choix depuis les poids
BRUTS ou les poids EMA.

Les deux jeux de poids sont stockes dans chaque checkpoint :
    state_dict  -> poids bruts (net)  : la vraie qualite courante du modele.
    ema_model   -> poids EMA          : meilleurs en fin d'entrainement, mais
                                        FLOUS tot (EMA pas encore chaude, cf.
                                        diag_ema_vs_raw.py). Absent des vieux
                                        checkpoints run_imagenet32 -> repli auto.

L'archi est reconstruite automatiquement depuis le champ 'name' du checkpoint
(build_from_name), donc rien a repasser (--K/--ic/...).

Usage
-----
    # poids bruts (recommande pour juger la qualite AVANT ~40k steps)
    python sample_checkpoint.py --ckpt .../latest.pt --weights raw

    # poids EMA (recommande en fin d'entrainement)
    python sample_checkpoint.py --ckpt .../latest.pt --weights ema

    # les deux cote a cote, meme bruit
    python sample_checkpoint.py --ckpt .../latest.pt --weights both

    # plus d'images, integration fine
    python sample_checkpoint.py --ckpt .../ckpt_step_50000.pt --n 16 --steps 200

Conseil GPU : si un entrainement tourne sur le GPU 1 (CUDA_VISIBLE_DEVICES=1),
lance ceci sur le GPU 0 libre :  CUDA_VISIBLE_DEVICES=0 python sample_checkpoint.py ...

Sortie -> <dir du ckpt>/sample_<weights>_step_<N>.png
"""

import argparse
import math
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchcfm.models.unet import UNetModel

from compute_fid_cifar10 import build_from_name, _VelocityWrapper, sample_batch, CHANNELS, IMG_SIZE
from sample_otcfm_pretrained import REF_CFG


def resolve_checkpoint(ckpt, device):
    """Reconstruit l'archi et localise les jeux de poids, quel que soit le format.

    Deux formats de checkpoint coexistent :
      - NOS checkpoints (run_cifar10_torchcfm_recipe / run_imagenet32) : champ 'name'
        (archi via build_from_name), poids bruts sous 'state_dict', EMA sous 'ema_model'.
      - Checkpoint OTCFM PREENTRAINE officiel (otcfm_cifar10_weights_step_*.pt) : pas
        de 'name', archi = UNetModel(**REF_CFG), poids bruts sous 'net_model', EMA sous
        'ema_model'.

    Retourne (model, is_unet, name, weight_keys) ou weight_keys mappe 'raw'/'ema' vers
    la vraie cle du checkpoint.
    """
    if "name" in ckpt:                                        # nos checkpoints
        model, is_unet = build_from_name(ckpt["name"], device)
        return model, is_unet, ckpt["name"], {"raw": "state_dict", "ema": "ema_model"}
    if "net_model" in ckpt:                                   # OTCFM preentraine officiel
        model = UNetModel(**REF_CFG).to(device)
        return model, True, "OTCFM_pretrained", {"raw": "net_model", "ema": "ema_model"}
    raise KeyError("Checkpoint non reconnu : ni 'name' (nos runs) ni 'net_model' "
                   f"(OTCFM preentraine). Cles presentes : {list(ckpt)[:6]}")


def _to_img(x):
    """(B,3,32,32) dans [-1,1] -> (B,32,32,3) dans [0,1]."""
    return (x.detach().cpu() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()


@torch.no_grad()
def generate_from_weights(ckpt, which, model, is_unet, weight_keys, x0, solver, steps, device):
    """Charge le jeu de poids `which` ('raw'|'ema') et echantillonne depuis x0.

    weight_keys : mapping 'raw'/'ema' -> vraie cle du checkpoint (cf. resolve_checkpoint).
    Retourne (images, cle_utilisee) ; None si le jeu demande est absent.
    """
    key = weight_keys[which]
    if key not in ckpt:
        if which == "ema":
            raw_key = weight_keys["raw"]
            print("  [!] pas de poids EMA dans ce checkpoint (vieux run ?) "
                  f"-> repli sur les poids bruts ('{raw_key}').", flush=True)
            key = raw_key
        else:
            raise KeyError(f"'{key}' absent du checkpoint.")

    model.load_state_dict(ckpt[key], strict=True)
    model.eval()
    vf = _VelocityWrapper(model, is_unet).to(device).eval()

    # meme bruit initial fourni -> les differences viennent des poids, pas du tirage
    x = x0.clone()
    if solver == "euler":
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((1,), i * dt, device=device)
            x = x + vf(t, x) * dt
    else:
        from torchdyn.core import NeuralODE
        node = NeuralODE(vf, solver="dopri5", atol=1e-5, rtol=1e-5)
        x = node.trajectory(x, t_span=torch.linspace(0, 1, 2, device=device))[-1]
    print(f"  genere depuis '{key}' ({which})", flush=True)
    return x, key


def _grid_shape(n):
    """Nb de (lignes, colonnes) pour disposer n images en grille la plus CARREE
    possible : n=4 -> 2x2, n=9 -> 3x3, n=8 -> 2x4. Plus compact qu'une seule ligne."""
    rows = int(math.floor(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    return rows, cols


def plot_rows(rows, title, save_path, n):
    """rows : liste de (label, images). Chaque jeu de poids est affiche en grille
    carree (gr x gc). Si plusieurs jeux (both), les blocs sont poses cote a cote."""
    gr, gc = _grid_shape(n)
    nsets = len(rows)
    fig, axes = plt.subplots(gr, gc * nsets, figsize=(1.9 * gc * nsets, 1.9 * gr),
                             squeeze=False)
    for s, (label, imgs) in enumerate(rows):
        arr = _to_img(imgs[:n])
        for k in range(gr * gc):
            r, c = divmod(k, gc)
            ax = axes[r, s * gc + c]
            if k < len(arr):
                ax.imshow(arr[k])
            ax.axis("off")
        # label du bloc, au-dessus de sa premiere vignette
        axes[0, s * gc].set_title(label, loc="left", fontsize=9)
    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> {save_path}", flush=True)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Genere depuis un checkpoint (poids bruts ou EMA).")
    p.add_argument("--ckpt", required=True, help="latest.pt ou ckpt_step_N.pt.")
    p.add_argument("--weights", type=str, default="raw", choices=["raw", "ema", "both"],
                   help="raw=poids bruts (defaut), ema=EMA, both=les deux cote a cote.")
    p.add_argument("--n", type=int, default=4, help="Nb d'images (dispose en grille carree, ex. 4 -> 2x2).")
    p.add_argument("--solver", type=str, default="euler", choices=["euler", "dopri5"])
    p.add_argument("--steps", type=int, default=100, help="Steps Euler (100 = recette FID).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="", help="Chemin figure (defaut: a cote du ckpt).")
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, is_unet, name, weight_keys = resolve_checkpoint(ckpt, device)
    step = ckpt.get("step", ckpt.get("epoch", "?"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name} | step {step} | {n_params/1e6:.2f}M params | solver {args.solver}-{args.steps}",
          flush=True)

    # meme bruit initial pour tous les jeux de poids
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    x0 = torch.randn(args.n, CHANNELS, IMG_SIZE, IMG_SIZE, generator=g).to(device)

    which_list = ["raw", "ema"] if args.weights == "both" else [args.weights]
    label_of = {"raw": "raw", "ema": "EMA"}
    rows = []
    for which in which_list:
        imgs, _ = generate_from_weights(ckpt, which, model, is_unet, weight_keys, x0,
                                        args.solver, args.steps, device)
        rows.append((label_of[which], imgs))

    out = args.out or os.path.join(os.path.dirname(args.ckpt),
                                   f"sample_{args.weights}_step_{step}.png")
    plot_rows(rows, f"{name} — step {step}", out, args.n)


if __name__ == "__main__":
    main()
