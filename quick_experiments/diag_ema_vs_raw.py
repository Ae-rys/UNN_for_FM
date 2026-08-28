# -*- coding: utf-8 -*-
"""
diag_ema_vs_raw.py
Diagnostic : compare les images generees par les poids BRUTS (net) vs les poids
EMA d'un meme checkpoint, avec le MEME bruit initial.

Motivation : l'EMA (decay 0.9999) a une constante de temps ~1/(1-0.9999)=10k
steps. Tot dans l'entrainement (< quelques constantes de temps), l'EMA traine
derriere les poids bruts car encore contaminee par l'init aleatoire. Les images
d'echantillon (generees depuis l'EMA) paraissent alors pires que le modele reel.
Ce script rend l'ecart visible.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python diag_ema_vs_raw.py \
        --ckpt results_cifar10_torchcfm_recipe/ConvScCP_.../latest.pt

Sortie : <dir du ckpt>/ema_vs_raw_step_<N>.png
"""

import argparse
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_fid_cifar10 import build_from_name, _VelocityWrapper, sample_batch, CHANNELS, IMG_SIZE


def _to_img(x):
    return (x.detach().cpu() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    name, step = ck["name"], ck.get("step", "?")
    model, is_unet = build_from_name(name, device)

    # meme bruit initial pour les deux jeux de poids
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    x0 = torch.randn(args.n, CHANNELS, IMG_SIZE, IMG_SIZE, generator=g).to(device)

    rows = []
    for label, key in [("brut (net)", "state_dict"), ("EMA", "ema_model")]:
        if key not in ck:
            print(f"  [skip] '{key}' absent du checkpoint", flush=True)
            continue
        model.load_state_dict(ck[key], strict=True)
        model.eval()
        vf = _VelocityWrapper(model, is_unet).to(device).eval()
        # on reutilise x0 en repositionnant la graine dans sample_batch : ici on
        # passe par un tirage identique -> on force x0 en amont.
        x = x0.clone()
        dt = 1.0 / args.steps
        for i in range(args.steps):
            t = torch.full((1,), i * dt, device=device)
            x = x + vf(t, x) * dt
        rows.append((label, x))
        print(f"  {label:12s} genere ({key})", flush=True)

    fig, axes = plt.subplots(len(rows), args.n, figsize=(1.7 * args.n, 1.9 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for r, (label, imgs) in enumerate(rows):
        arr = _to_img(imgs[:args.n])
        for c in range(args.n):
            axes[r, c].imshow(arr[c]); axes[r, c].axis("off")
        axes[r, 0].text(-0.1, 0.5, label, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=9)
    fig.suptitle(f"{name}\nstep {step} — poids bruts vs EMA (meme bruit)", fontsize=9)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(args.ckpt), f"ema_vs_raw_step_{step}.png")
    plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
    print(f"\n  -> {out}", flush=True)


if __name__ == "__main__":
    main()
