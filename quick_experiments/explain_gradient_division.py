# -*- coding: utf-8 -*-
"""
explain_gradient_division.py

« Si la loss est la meme, comment les gradients peuvent-ils differer ? »

Parce que ce n'est pas la meme fonction de la SORTIE DU RESEAU. Les deux modeles
representent le meme champ de vitesse, mais par des variables de sortie differentes :

    v-pred :  v = f_theta(x_t, t)                      (sortie = la vitesse)
    x-pred :  v = (f_theta(x_t, t) - x_t) / (1 - t)     (sortie = x1_pred)

La seconde contient une DIVISION par (1-t). Retropropager a travers elle multiplie
le gradient entrant par 1/(1-t) : c'est la jacobienne de la reparametrisation.

Soit r = v_theta - u_t le residu en vitesse, commun aux deux. Avec L = mean(r^2) :

    dL/d(sortie_v) = 2r / B                    (v-pred)
    dL/d(sortie_x) = 2r / ((1-t) B)            (x-pred)   <- facteur 1/(1-t)

    car  x1_pred - x1 = (1-t) r  : une erreur de 0.01 sur x1 a t=0.95 vaut deja
    0.20 d'erreur en vitesse. Le poids 1/(1-t)^2 de la loss compte ce facteur deux
    fois ; le residu x-espace, lui-meme (1-t) fois plus petit, en rend un. Reste 1.

Puis dL/dtheta = dL/d(sortie) . d(sortie)/dtheta, ou d(sortie)/dtheta depend de
l'ARCHITECTURE. D'ou l'ecart observe sur les gradients-parametres (x10 a t=0.95 dans
check_gradient_geometry.py) et non x20 : le facteur exact 1/(1-t) porte sur la sortie,
la jacobienne du reseau module le reste.

Ce script verifie le facteur 1/(1-t) a l'exactitude machine, sur un vrai checkpoint.

Sortie -> explain_gradient_division.txt

Usage
-----
    source ~/.venvs/unn/bin/activate
    python explain_gradient_division.py --device cuda:0
"""

import argparse
import os

import torch

from compute_fid_cifar10 import build_from_name

IMG_SIZE, CHANNELS = 32, 3
DIM = CHANNELS * IMG_SIZE * IMG_SIZE
CKPT = "results_afhq32/ConvScCP_UNN_rgb_k9_K10_ic128_L1_LFO/latest.pt"
T_GRID = [0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--ckpt", type=str, default=CKPT)
    p.add_argument("--cache", type=str, default="./data/afhq_cat32_train.pt")
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    B = args.batch

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model, is_unet = build_from_name(ck["name"], device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.train()                       # x-pred : le forward renvoie x1_pred
    assert not is_unet

    d = torch.load(args.cache, map_location="cpu", weights_only=False)
    data = d["data"].float().div_(127.5).sub_(1.0)
    x1 = data[torch.randperm(data.shape[0])[:B]].reshape(B, -1).to(device)
    x0 = torch.randn_like(x1)
    ut = x1 - x0

    lines = ["=" * 74,
             f"Facteur de gradient introduit par la division  v = (x1p - x_t)/(1-t)",
             f"checkpoint : {args.ckpt}  |  batch {B}", "=" * 74,
             f"{'t':>7}{'1/(1-t)':>12}{'|dL/dx1p| / |dL/dv|':>24}{'ecart rel':>14}",
             "-" * 74]
    worst = 0.0
    for tv in T_GRID:
        t = torch.full((B, 1), tv, device=device)
        xt = (1 - t) * x0 + t * x1

        x1p = model(torch.cat([xt, t], dim=-1))
        x1p.retain_grad()
        v = (x1p - xt) / (1 - t)        # LA division : la sortie x-pred devient vitesse
        v.retain_grad()
        loss = torch.mean((v - ut) ** 2)
        model.zero_grad(set_to_none=True)
        loss.backward()

        gx = float(x1p.grad.norm())
        gv = float(v.grad.norm())
        ratio, theo = gx / gv, 1.0 / (1 - tv)
        rel = abs(ratio - theo) / theo
        worst = max(worst, rel)
        lines.append(f"{tv:>7.2f}{theo:>12.2f}{ratio:>24.4f}{rel:>14.1e}")

    lines += ["-" * 74,
              f"ecart relatif max a 1/(1-t) : {worst:.1e}",
              "",
              "Meme loss (meme scalaire, meme residu en vitesse), mais le gradient qui",
              "ARRIVE sur la sortie du reseau est 1/(1-t) fois plus grand cote x-pred.",
              "Le UNet, qui sort la vitesse directement, n'a pas cette division : son",
              "facteur vaut 1 a tous les t.",
              "",
              "Ce n'est donc pas un artefact du clamp ni du tirage de t : c'est la",
              "parametrisation elle-meme. La retirer demanderait de faire sortir la",
              "vitesse au ScCP (predicts_x1=False), pas de retoucher la loss."]

    txt = "\n".join(lines)
    print(txt, flush=True)
    with open("explain_gradient_division.txt", "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
