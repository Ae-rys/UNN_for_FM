# -*- coding: utf-8 -*-
"""
make_afhq_report_figs.py
Deux grilles 2x2 AFHQ-32 pour le rapport : le meilleur ScCP et le UNet torchcfm ch64.

  internship_report/images/afhq32_sccp_2x2.png
  internship_report/images/afhq32_unet_ch64_2x2.png

Echantillonnage : euler_sample() de la recette, qui TRONQUE la grille a t <= t_max.
C'est indispensable ici parce que les deux runs n'ont PAS le meme domaine :
  - UNet_torchcfm_ch64                    t_max = 0.95  (entraine sur t ~ U(0, 0.95))
  - ConvScCP_UNN_rgb_k15_K20_ic256_L1_LFO t_max = None  (ancien regime, clamp 0.05)
Interroger le ch64 au-dela de 0.95 le sortirait de son domaine d'entrainement.
Les deux utilisent le MEME nombre de pas et la MEME graine, donc la comparaison
visuelle ne depend ni du budget de solveur ni du bruit initial.

Poids EMA : ce sont eux qui portent les resultats publies (cf. train_one).

Usage :  python make_afhq_report_figs.py [--n-steps 100] [--seed 0]
"""
import argparse, os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_cifar10_torchcfm_recipe import euler_sample, CHANNELS, IMG_SIZE
from compute_fid_cifar10 import build_from_name

OUT = "internship_report/images"
RUNS = [
    ("ConvScCP_UNN_rgb_k15_K20_ic256_L1_LFO", "afhq32_sccp_2x2.png"),
    ("UNet_torchcfm_ch64",                    "afhq32_unet_ch64_2x2.png"),
]


def grid2x2(imgs, path):
    """4 images RGB dans [-1,1] -> PNG 2x2 carre, sans titre interne
    (la legende LaTeX porte l'info) et sans marge."""
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 4.8))
    for a, im in zip(axes.flat, imgs):
        a.imshow(((im.permute(1, 2, 0).clamp(-1, 1) + 1) / 2).cpu().numpy())
        a.axis("off")
    fig.subplots_adjust(wspace=0.02, hspace=0.02, left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("ecrit :", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:1")
    args = p.parse_args()
    dev = torch.device(args.device)
    os.makedirs(OUT, exist_ok=True)

    for name, out_png in RUNS:
        ck = torch.load(f"results_afhq32/{name}/latest.pt", map_location="cpu",
                        weights_only=False)
        model, is_unet = build_from_name(ck["name"], dev)
        model.load_state_dict(ck["ema_model"])          # EMA = poids publies
        t_max = ck.get("t_max")
        imgs = euler_sample(model, is_unet, dev, n=4, n_steps=args.n_steps,
                            t_max=t_max, seed=args.seed)
        if not is_unet:                                  # ScCP sort a plat
            imgs = imgs.view(-1, CHANNELS, IMG_SIZE, IMG_SIZE)
        print(f"{name:<42} step={ck.get('step'):>7}  t_max={t_max}")
        grid2x2(imgs, os.path.join(OUT, out_png))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
