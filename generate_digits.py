# -*- coding: utf-8 -*-
"""
generate_digits.py
Genere plein de chiffres MNIST avec un checkpoint ConvScCP_UNN (.pt = state_dict).

La config (K, internal_channel, version, w_bias, prox l1/scalaire...) est
AUTO-DETECTEE depuis les formes du state_dict -> marche pour n'importe quel run
ConvScCP_UNN sauve avec --save-model, pas seulement temp-4.

Usage
-----
    # GPU 0 souvent occupe -> forcer le GPU libre
    CUDA_VISIBLE_DEVICES=1 python generate_digits.py \
        --ckpt results/temp-4/ConvScCP_UNN_L1_LNO/model.pt --n 256

    python generate_digits.py --ckpt <path> --n 100 --out mes_chiffres.png
"""
import argparse
import math
import os
import re

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper

from models.architectures import ConvScCP_UNN, SmallUNet, SmallUNetX1


def infer_config(sd):
    """Deduit les hyperparametres de ConvScCP_UNN depuis les formes du state_dict."""
    K = 1 + max(int(m.group(1)) for k in sd for m in [re.match(r"layers\.(\d+)\.", k)] if m)
    W = sd["layers.0.W_weight"]                       # (internal_channel, in_channels, k, k)
    ic, in_ch, ksize = W.shape[0], W.shape[1], W.shape[2]
    # LFO <-> V_weight appris par couche ; LNO <-> V=W + buffer _sigma_u (power iteration).
    # Critere STRUCTUREL : log_tau est present dans les deux versions, il ne discrimine pas.
    version = "LFO" if any(k.endswith(".V_weight") for k in sd) else "LNO"
    w_bias  = any(k.endswith(".W_bias") for k in sd)
    is_l1   = any(".prox.time_scaling." in k for k in sd)   # L1ProxConv vs DoubleConvTime
    use_Unet = "l1" if is_l1 else False
    # largeur du MLP r(t) du prox l1 : le defaut a change (8 -> 32) au fil des runs,
    # donc on la lit dans le checkpoint au lieu de la supposer.
    prox_w = sd["layers.0.prox.time_scaling.0.weight"].shape[0] if is_l1 else 32
    img_size = 28 if in_ch == 1 else 32                # MNIST 28 mono / RGB 32
    dim = in_ch * img_size * img_size
    return dict(dim=dim, K=K, internal_channel=ic, in_channels=in_ch, img_size=img_size,
                version=version, use_Unet=use_Unet, w_bias=w_bias, kernel=ksize, prox_w=prox_w)


@torch.no_grad()
def generate(model, n, dim, device, batch, seed):
    """n echantillons via l'ODE FM (dopri5, t: 0->1). Renvoie (n, dim) dans ~[-1,1]."""
    torch.manual_seed(seed)
    outs = []
    node = NeuralODE(torch_wrapper(model), solver="dopri5", atol=1e-5, rtol=1e-5)
    t_span = torch.linspace(0, 1, 2, device=device)
    done = 0
    while done < n:
        b = min(batch, n - done)
        x0 = torch.randn(b, dim, device=device)
        traj = node.trajectory(x0, t_span=t_span)
        outs.append(traj[-1].cpu())
        done += b
        print(f"  genere {done}/{n}")
    return torch.cat(outs, 0)[:n]


def save_grid(imgs, img_size, save_path, title, cols=None):
    """imgs: (n, dim) -> grille PNG (gris, [-1,1]).

    cols : nombre de colonnes. None -> grille la plus carree possible (defaut).
           cols=n -> les n chiffres COTE A COTE sur une seule ligne de carres.
    """
    n = imgs.shape[0]
    imgs = imgs.view(n, img_size, img_size).clamp(-1, 1)
    cols = math.ceil(math.sqrt(n)) if cols is None else max(1, min(cols, n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.9, rows * 0.9))
    axes = axes.reshape(-1) if hasattr(axes, "reshape") else [axes]
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(imgs[i], cmap="gray", vmin=-1, vmax=1)
    #fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Generer des chiffres avec un ConvScCP_UNN entraine.")
    p.add_argument("--ckpt", type=str, default="results/OT-CFM/SmallUNetX1_baseline/model.pt")
    p.add_argument("--n",     type=int, default=256, help="nombre de chiffres a generer")
    p.add_argument("--cols",  type=int, default=None,
                   help="nb de colonnes de la grille (defaut: grille carree). "
                        "--n 4 --cols 4 -> 4 chiffres cote a cote sur une ligne")
    p.add_argument("--batch", type=int, default=128, help="taille de lot pour la generation")
    p.add_argument("--seed",  type=int, default=0)
    p.add_argument("--out",   type=str, default="", help="chemin PNG (defaut: <dir du ckpt>/generated_<n>.png)")
    p.add_argument("--save-individual", action="store_true", help="sauve aussi chaque chiffre en .png separe")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if "layers.0.W_weight" in sd:                           # ConvScCP_UNN
        cfg = infer_config(sd)
        model = ConvScCP_UNN(dim=cfg["dim"], K=cfg["K"], internal_channel=cfg["internal_channel"],
                             use_Unet=cfg["use_Unet"], version=cfg["version"],
                             w_bias=cfg["w_bias"], in_channels=cfg["in_channels"],
                             img_size=cfg["img_size"], kernel_size=cfg["kernel"],
                             prox_w=cfg["prox_w"]).to(device)
    elif "inc.conv1.weight" in sd:                          # SmallUNet baseline
        base_ch = sd["inc.conv1.weight"].shape[0]           # (base_ch, in_channels, 3, 3)
        in_ch   = sd["inc.conv1.weight"].shape[1]
        cfg = dict(dim=in_ch * 28 * 28, internal_channel=base_ch, img_size=28,
                   in_channels=in_ch, K="-", version="SmallUNet")
        model = SmallUNetX1(in_channels=in_ch, out_channels=in_ch, base_ch=base_ch).to(device)
    else:
        raise ValueError(f"Type de modele non reconnu depuis les cles: {list(sd)[:5]}")
    print(f"Config auto-detectee : {cfg}")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [warn] missing={missing}  unexpected={unexpected}")
    model.eval()
    print(f"Checkpoint charge : {args.ckpt}  ({sum(p.numel() for p in model.parameters()):,} params)")

    imgs = generate(model, args.n, cfg["dim"], device, args.batch, args.seed)

    out = args.out or os.path.join(os.path.dirname(args.ckpt), f"generated_{args.n}.png")
    save_grid(imgs, cfg["img_size"], out,
              title=""#f"{os.path.basename(os.path.dirname(args.ckpt))} — {args.n} echantillons "
                #    f"(K={cfg['K']}, ic={cfg['internal_channel']}, {cfg['version']})",
             , cols=args.cols)
    print(f"\nGrille sauvee -> {out}")

    if args.save_individual:
        d = os.path.join(os.path.dirname(args.ckpt), f"samples_{args.n}")
        os.makedirs(d, exist_ok=True)
        arr = imgs.view(args.n, cfg["img_size"], cfg["img_size"]).clamp(-1, 1)
        for i in range(args.n):
            plt.imsave(os.path.join(d, f"{i:04d}.png"), arr[i], cmap="gray", vmin=-1, vmax=1)
        print(f"Chiffres individuels -> {d}/")


if __name__ == "__main__":
    main()
