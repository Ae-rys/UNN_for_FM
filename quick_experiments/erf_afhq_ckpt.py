# -*- coding: utf-8 -*-
"""
erf_afhq_ckpt.py — Le champ de reception EFFECTIF du ConvScCP AFHQ est-il deja
global, ou la portee est-elle vraiment la contrainte ?

Question posee
--------------
A k=9 / K=10, le RF *nominal* vaut deja ~161 px, soit 5x l'image 32x32 : la
portee nominale est saturee, on ne peut pas en acheter davantage avec un plus
gros kernel. La seule facon qu'un kernel plus large aide, c'est que le RF
*effectif* des poids APPRIS soit, lui, bien plus petit que l'image.

Ce script le mesure sur le checkpoint entraine : carte
    |d x1_pred(pixel central) / d x_t(x')|
moyennee sur des images AFHQ reelles bruitees, pour une grille de t.
(Convention FM du repo : t=0 = bruit, t=1 = data. Kamb & Ganguli Fig. 4a :
coarse-to-fine, gros RF pres de t=0 qui retrecit vers t=1.)

Le chiffre qui tranche
----------------------
On compare le rayon effectif mesure R(t) au rayon d'un RF UNIFORME sur 32x32
(R_unif ~ 12.2 px, calcule exactement ici) :
    R(t) ~ R_unif       -> le modele agrege deja globalement. Un kernel plus
                           large n'achete PAS de portee : la run k=25 testerait
                           en pratique surtout la capacite.
    R(t) << R_unif      -> la portee effective est bien la contrainte. Le
                           kernel plus large est l'intervention pertinente.

Adapte de erf_vs_time_checkpoint.py (MNIST 28x28 1 canal) au cas AFHQ
32x32 RGB, avec reconstruction auto de l'archi depuis le champ 'name' du
checkpoint (resolve_checkpoint) — rien a repasser en --K/--ic/--kernel.

Usage
-----
    python erf_afhq_ckpt.py \
        --ckpt results_afhq32/ConvScCP_UNN_rgb_k9_K10_ic256_L1_LFO/latest.pt \
        --device cuda:1

Duree : ~2-5 min (7 valeurs de t x 16 backwards chacune).
Sorties -> erf_afhq_maps.png, erf_afhq_radius.png, erf_afhq_metrics.txt
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_fid_cifar10 import CHANNELS, IMG_SIZE
from sample_checkpoint import resolve_checkpoint

S = IMG_SIZE
TS = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]
N_SAMP = 16          # images reelles moyennees par valeur de t


def real_images(cache, n, device):
    """n images AFHQ reelles, remises dans [-1,1] comme a l'entrainement."""
    if not os.path.exists(cache):
        raise FileNotFoundError(f"{cache} absent — lance prepare_afhq_cats.py")
    d = torch.load(cache)
    x = d["data"][:n].float().div_(127.5).sub_(1.0)
    return x.to(device)


def uniform_radius(s):
    """Rayon effectif d'un RF parfaitement UNIFORME sur s x s : la reference
    'agregation globale'. Sert de plafond auquel comparer le R(t) mesure."""
    c = s // 2
    ys, xs = np.mgrid[0:s, 0:s]
    return float(np.sqrt((ys - c) ** 2 + (xs - c) ** 2).mean())


def radial_stats(m, kernel):
    """(rayon effectif en px, fraction de masse HORS de l'empreinte d'un seul
    kernel k x k centre). La 2e mesure dit combien de la reponse vient de
    au-dela de ce qu'une conv unique peut voir."""
    c = S // 2
    ys, xs = np.mgrid[0:S, 0:S]
    d = np.sqrt((ys - c) ** 2 + (xs - c) ** 2)
    w = m / (m.sum() + 1e-12)
    r = kernel // 2
    outside = (np.abs(ys - c) > r) | (np.abs(xs - c) > r)
    return float((w * d).sum()), float(w[outside].sum())


def erf_at_t(model, x1, t, device):
    """Carte |d out(pixel central, somme sur les 3 canaux) / d x_t| moyennee sur
    les images de x1, sommee sur les canaux d'entree -> (S, S)."""
    acc = torch.zeros(S, S)
    dim = CHANNELS * S * S
    for i in range(x1.shape[0]):
        x0 = torch.randn(1, CHANNELS, S, S, device=device)
        xt = ((1 - t) * x0 + t * x1[i:i + 1]).detach().requires_grad_(True)
        inp = torch.cat([xt.view(1, -1), torch.full((1, 1), t, device=device)], dim=1)
        out = model(inp).view(1, CHANNELS, S, S)
        model.zero_grad(set_to_none=True)
        out[0, :, S // 2, S // 2].sum().backward()
        acc += xt.grad.detach().abs().sum(1).view(S, S).cpu()
    return (acc / x1.shape[0]).numpy()


def main():
    p = argparse.ArgumentParser(description="RF effectif d'un ckpt ConvScCP AFHQ 32x32 RGB.")
    p.add_argument("--ckpt", default="results_afhq32/ConvScCP_UNN_rgb_k9_K10_ic256_L1_LFO/latest.pt")
    p.add_argument("--cache", default="./data/afhq_cat32_train.pt")
    p.add_argument("--weights", default="raw", choices=["raw", "ema"])
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--tag", default="afhq")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, is_unet, name, weight_keys = resolve_checkpoint(ckpt, device)
    if is_unet:
        raise SystemExit("Ce script mesure le RF d'un ConvScCP, pas d'un UNet.")
    model.load_state_dict(ckpt[weight_keys[args.weights]], strict=True)

    # train() : ConvScCP renvoie x1_pred (le debruiteur, quantite mesuree par
    # Kamb) ; en eval() il renverrait la vitesse. Pas de norm/dropout dans la
    # variante L1, donc ce mode ne change rien d'autre.
    model.train()
    for prm in model.parameters():
        prm.requires_grad_(False)

    kernel = int(name.split("_k")[1].split("_")[0])
    step = ckpt.get("step", "?")
    r_unif = uniform_radius(S)
    print(f"{name} | step {step} | poids {args.weights} | kernel {kernel} "
          f"| image {S}x{S}", flush=True)
    print(f"reference RF uniforme (agregation globale) : R_unif = {r_unif:.2f} px\n", flush=True)

    x1 = real_images(args.cache, N_SAMP, device)
    maps, stats = {}, {}
    for t in TS:
        m = erf_at_t(model, x1, t, device)
        er, out_k = radial_stats(m, kernel)
        maps[t], stats[t] = m, (er, out_k)
        print(f"t={t:.2f} :  R_eff = {er:5.2f} px  ({100*er/r_unif:5.1f} % de R_unif)"
              f"   masse hors kernel {kernel}x{kernel} = {100*out_k:5.1f} %", flush=True)

    rs = [stats[t][0] for t in TS]
    frac = 100 * float(np.mean(rs)) / r_unif
    verdict = ("RF effectif DEJA quasi global -> un kernel plus large n'achete pas de "
               "portee ; la run k=25 testerait surtout la capacite."
               if frac > 85 else
               "RF effectif nettement PLUS PETIT que l'image -> la portee est bien la "
               "contrainte ; le kernel plus large est l'intervention pertinente.")
    print(f"\nmoyenne sur t : R_eff = {np.mean(rs):.2f} px = {frac:.1f} % de R_unif")
    print(f"VERDICT : {verdict}", flush=True)

    with open(f"erf_{args.tag}_metrics.txt", "w") as f:
        f.write(f"# {name} step={step} weights={args.weights} kernel={kernel}\n")
        f.write(f"# R_unif({S}x{S}) = {r_unif:.4f} px\n")
        f.write("t\tR_eff_px\tpct_of_R_unif\tmass_outside_kernel\n")
        for t in TS:
            er, ok = stats[t]
            f.write(f"{t}\t{er:.4f}\t{100*er/r_unif:.2f}\t{ok:.4f}\n")
        f.write(f"\nmean_R_eff\t{np.mean(rs):.4f}\npct_of_R_unif\t{frac:.2f}\n")
        f.write(f"verdict\t{verdict}\n")

    fig, ax = plt.subplots(1, len(TS), figsize=(2.6 * len(TS), 3.0))
    for j, t in enumerate(TS):
        mn = maps[t] / maps[t].max()
        ax[j].imshow(np.log10(mn + 1e-4), cmap="magma", vmin=-4, vmax=0)
        er, ok = stats[t]
        ax[j].set_title(f"t={t}\nR={er:.1f}px ({100*er/r_unif:.0f}%)", fontsize=8)
        ax[j].set_xticks([]); ax[j].set_yticks([])
    fig.suptitle(f"RF effectif vs t — {name} (t=0 bruit, t=1 data)", fontsize=10)
    plt.tight_layout(); plt.savefig(f"erf_{args.tag}_maps.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(5.2, 3.4))
    ax2.plot(TS, rs, "o-", label="R effectif mesure")
    ax2.axhline(r_unif, color="crimson", ls="--", lw=1.2,
                label=f"R uniforme = agregation globale ({r_unif:.1f} px)")
    ax2.axhline(kernel / 2, color="gray", ls=":", lw=1.2,
                label=f"1 seul kernel {kernel}x{kernel}")
    ax2.set_ylim(0, r_unif * 1.15)
    ax2.set_xlabel("t  (0 = bruit, 1 = data)"); ax2.set_ylabel("rayon effectif (px)")
    ax2.set_title(f"Portee effective — {name}", fontsize=9)
    ax2.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"erf_{args.tag}_radius.png", dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"\n-> erf_{args.tag}_maps.png , erf_{args.tag}_radius.png , erf_{args.tag}_metrics.txt")


if __name__ == "__main__":
    main()
