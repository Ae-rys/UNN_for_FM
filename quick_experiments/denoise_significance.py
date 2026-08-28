# -*- coding: utf-8 -*-
"""
denoise_significance.py
Barres d'erreur sur les nmse de denoise_probe.py, par bootstrap sur les images
de validation. Repond a : "l'ecart entre ces deux configs depasse-t-il le bruit ?"

Pourquoi c'est necessaire
-------------------------
Les nmse sont mesurees sur 512 images seulement. Les ecarts qu'on cherche a
interpreter sont petits (2 a 15 %). Sans intervalle de confiance, "la capacite
est inerte" et "la capacite aide un peu" sont indiscernables.

Deux precautions :
  * BOOTSTRAP APPARIE. Tous les modeles voient les MEMES 512 images et le MEME
    bruit fixe. On reechantillonne donc les indices d'images UNE fois par tirage
    et on les applique a tous les modeles : l'incertitude commune (quelles images
    sont dans la validation) s'annule dans les differences, et l'IC sur l'ecart
    est bien plus serre que ce que suggereraient deux IC separes. Comparer des
    IC qui se chevauchent est un test trop conservateur — c'est l'IC de la
    DIFFERENCE qu'il faut lire.
  * La variabilite mesuree est celle du TIRAGE DES IMAGES uniquement. L'alea de
    l'entrainement (init, ordre des batchs) n'est PAS couvert : il faudrait
    plusieurs graines par config. Un ecart significatif ici veut donc dire
    "pas un artefact du jeu de validation", pas "reproductible a coup sur".

Usage
-----
    source ~/.venvs/unn/bin/activate
    python denoise_significance.py --runs \\
        results_denoise_probe_20k/unet_ref_ch64_b1_m1-2-2 \\
        results_denoise_probe_20k/unet_ref_ch32_b1_m1-2-2 \\
        results_denoise_probe_20k/unet_kamb

Sortie : tableau des nmse +/- IC95, puis IC95 des ecarts deux a deux.
"""

import argparse
import gc
import os

import numpy as np
import torch

from denoise_probe import (build_model, forward_x1, load_data, make_val_set,
                           name_to_config, remap_state_dict)


def read_selected(run_dir):
    path = os.path.join(run_dir, "metrics.txt")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.startswith("selected="):
                    return line.strip().split("=", 1)[1]
    return "raw"


@torch.no_grad()
def per_image_se(model, val_sets, t_list, channels, img_size, is_unet_ref, batch=256):
    """Erreur quadratique MOYENNE PAR IMAGE (moyenne sur les pixels et sur les t
    de t_list) -> vecteur (n_val,). C'est la statistique que l'on bootstrappe."""
    model.eval()
    acc = None
    for t in t_list:
        xt, x1 = val_sets[t]
        errs = []
        for i in range(0, xt.shape[0], batch):
            xb, yb = xt[i:i + batch], x1[i:i + batch]
            tb = torch.full((xb.shape[0], 1), float(t), device=xb.device)
            pred = forward_x1(model, xb, tb, channels, img_size, is_unet_ref)
            errs.append(((pred - yb) ** 2).mean(dim=1))
        e = torch.cat(errs)
        acc = e if acc is None else acc + e
    return (acc / len(t_list)).cpu().numpy()


def main():
    p = argparse.ArgumentParser(description="IC bootstrap sur les nmse.")
    p.add_argument("--runs", nargs="+", required=True, help="Dossiers de run.")
    p.add_argument("--cache", type=str, default="./data/afhq_cat32_train.pt")
    p.add_argument("--t", type=str, default="0.2,0.4,0.6,0.8")
    p.add_argument("--n-val", type=int, default=512)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:1")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and (device.index or 0) >= torch.cuda.device_count():
        device = torch.device("cuda:0")
    t_list = [float(s) for s in args.t.split(",") if s.strip()]

    x_train, x_val = load_data(args.cache, args.n_val, device, seed=args.seed)
    channels, img_size = x_train.shape[1], x_train.shape[2]
    del x_train
    val_sets = make_val_set(x_val, t_list, seed=args.seed + 1234)
    x1 = x_val.reshape(x_val.shape[0], -1)
    var = float(((x1 - x1.mean(dim=0, keepdim=True)) ** 2).mean())

    names, se = [], []
    for run in args.runs:
        name = os.path.basename(run.rstrip("/"))
        ckpt = torch.load(os.path.join(run, "model.pt"), map_location=device,
                          weights_only=False)
        cfg = name_to_config(name)
        model = build_model(cfg, device, channels, img_size)
        which = read_selected(run)
        model.load_state_dict(remap_state_dict(
            ckpt["ema_model" if which == "ema" else "state_dict"]))
        se.append(per_image_se(model, val_sets, t_list, channels, img_size,
                               cfg["arch"] == "unet_ref"))
        names.append(name)
        print(f"  charge {name} [{which}]", flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()

    se = np.stack(se)                                    # (n_modeles, n_val)
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, se.shape[1], size=(args.n_boot, se.shape[1]))
    boot = se[:, idx].mean(axis=2) / var                 # (n_modeles, n_boot)

    print(f"\nnmse +/- IC95 (bootstrap {args.n_boot} tirages sur {se.shape[1]} images, "
          f"t={t_list})")
    print("-" * 74)
    order = np.argsort(se.mean(axis=1))
    for i in order:
        lo, hi = np.percentile(boot[i], [2.5, 97.5])
        print(f"  {names[i]:<34}{se[i].mean()/var:>8.4f}   [{lo:.4f}, {hi:.4f}]"
              f"   +/-{(hi-lo)/2*100/(se[i].mean()/var):>5.1f} %")

    print(f"\nEcarts APPARIES (memes images rechantillonnees) — IC95 de la difference")
    print("-" * 74)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            i, j = order[a], order[b]
            d = boot[j] - boot[i]                        # j est le moins bon
            lo, hi = np.percentile(d, [2.5, 97.5])
            rel = 100 * d.mean() / boot[i].mean()
            verdict = "SIGNIFICATIF" if lo > 0 or hi < 0 else "dans le bruit"
            print(f"  {names[j]:<30} - {names[i]:<30}")
            print(f"      {d.mean():+.4f} ({rel:+.1f} %)  IC95 [{lo:+.4f}, {hi:+.4f}]"
                  f"   -> {verdict}")


if __name__ == "__main__":
    main()
