# -*- coding: utf-8 -*-
"""
generalization_gap.py
"Faut-il plus d'images ?" — la reponse se mesure, elle ne se devine pas.

On evalue le MEME modele sur deux ensembles disjoints, avec le meme protocole
de bruit fixe : des images d'ENTRAINEMENT (vues des milliers de fois) et des
images de VALIDATION (jamais vues). L'ecart tranche :

  train ~= val   SOUS-APPRENTISSAGE. Le modele n'a meme pas fini d'exploiter les
                 images qu'il a. Ajouter des donnees ne servira A RIEN ; ce qu'il
                 faut, c'est plus de steps (ou une meilleure architecture).

  train << val   MEMORISATION. Le modele exploite des details propres aux images
                 d'entrainement et ne generalise pas. La, plus de donnees (ou de
                 l'augmentation) attaque directement le probleme.

Le meme bruit fixe est applique aux deux ensembles (generateur dedie, meme ordre
de t), et les deux ensembles ont la MEME taille : l'ecart mesure ne peut pas
venir d'un desequilibre d'echantillonnage. Un bootstrap apparie sur les images
donne l'intervalle de confiance de l'ecart.

Usage
-----
    source ~/.venvs/unn/bin/activate
    python generalization_gap.py --ckpt \\
        results_afhq32/ConvScCP_UNN_rgb_k9_K10_ic128_L1_LFO/latest.pt

    # plusieurs checkpoints : l'ecart grandit-il avec l'entrainement ?
    python generalization_gap.py --ckpt .../ckpt_step_{30000,100000}.pt
"""

import argparse
import gc
import os

import numpy as np
import torch

from denoise_probe import forward_x1, make_val_set


def split_data(cache, n_val, n_probe, device, seed=0):
    """Reproduit le split de denoise_probe.py, puis prend n_probe images de
    CHAQUE cote. Tailles egales -> l'ecart n'est pas un artefact de taille."""
    d = torch.load(cache)
    x = d["data"].float().div_(127.5).sub_(1.0)
    g = torch.Generator().manual_seed(seed)
    x = x[torch.randperm(x.shape[0], generator=g)]
    x_val, x_train = x[:n_val], x[n_val:]
    n = min(n_probe, x_val.shape[0], x_train.shape[0])
    return x_train[:n].to(device), x_val[:n].to(device)


@torch.no_grad()
def per_image_se(model, imgs, t_list, is_unet, seed=1234):
    """Erreur quadratique moyenne par image, moyennee sur les t. Le bruit est
    tire du meme generateur pour les deux ensembles -> comparaison appariee."""
    model.eval()
    B = imgs.shape[0]
    C, S = imgs.shape[1], imgs.shape[2]
    x1 = imgs.reshape(B, -1)
    g = torch.Generator().manual_seed(seed)
    acc = torch.zeros(B, device=imgs.device)
    for t in t_list:
        x0 = torch.randn(B, x1.shape[1], generator=g).to(imgs.device)
        xt = (1 - t) * x0 + t * x1
        tb = torch.full((B, 1), float(t), device=imgs.device)
        errs = []
        for i in range(0, B, 256):
            pred = forward_x1(model, xt[i:i + 256], tb[i:i + 256], C, S, is_unet)
            errs.append(((pred - x1[i:i + 256]) ** 2).mean(dim=1))
        acc += torch.cat(errs)
    return (acc / len(t_list)).cpu().numpy()


def main():
    p = argparse.ArgumentParser(description="Sous-apprentissage ou memorisation ?")
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--cache", type=str, default="./data/afhq_cat32_train.pt")
    p.add_argument("--t", type=str, default="0.2,0.4,0.6,0.8")
    p.add_argument("--weights", type=str, default="ema", choices=["ema", "raw"])
    p.add_argument("--n-val", type=int, default=512)
    p.add_argument("--n-probe", type=int, default=512)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:1")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and (device.index or 0) >= torch.cuda.device_count():
        device = torch.device("cuda:0")
    t_list = [float(s) for s in args.t.split(",") if s.strip()]

    xtr, xva = split_data(args.cache, args.n_val, args.n_probe, device, args.seed)
    var = float(((xva.reshape(xva.shape[0], -1)
                  - xva.reshape(xva.shape[0], -1).mean(0, keepdim=True)) ** 2).mean())
    print(f"{xtr.shape[0]} images train vs {xva.shape[0]} images val | t={t_list} | "
          f"var={var:.4f}\n", flush=True)

    from sample_checkpoint import resolve_checkpoint
    rng = np.random.default_rng(args.seed)
    print(f"{'checkpoint':<46}{'steps':>9}{'train':>9}{'val':>9}{'ecart':>9}   IC95")
    print("-" * 96)
    for path in args.ckpt:
        model = None
        try:
            ck = torch.load(path, map_location=device, weights_only=False)
            model, is_unet, name, keys = resolve_checkpoint(ck, device)
            key = keys.get(args.weights)
            model.load_state_dict(ck[key if key in ck else keys["raw"]])
            step = int(ck.get("step", 0))
            se_tr = per_image_se(model, xtr, t_list, is_unet) / var
            se_va = per_image_se(model, xva, t_list, is_unet) / var
            idx = rng.integers(0, len(se_tr), size=(args.n_boot, len(se_tr)))
            d = se_va[idx].mean(1) - se_tr[idx].mean(1)
            lo, hi = np.percentile(d, [2.5, 97.5])
            rel = 100 * d.mean() / se_tr.mean()
            print(f"{name[:44]:<46}{step:>9,}{se_tr.mean():>9.4f}{se_va.mean():>9.4f}"
                  f"{d.mean():>+9.4f}   [{lo:+.4f}, {hi:+.4f}]  ({rel:+.1f} %)", flush=True)
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"  [ECHEC] {path} : {exc}", flush=True)
        finally:
            model = None; gc.collect(); torch.cuda.empty_cache()

    print("\nLecture : ecart ~ 0 -> sous-apprentissage, plus de donnees ne servira a rien.")
    print("          ecart >> 0 -> memorisation, plus de donnees / d'augmentation aide.")


if __name__ == "__main__":
    main()
