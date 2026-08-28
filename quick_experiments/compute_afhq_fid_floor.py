# -*- coding: utf-8 -*-
"""
compute_afhq_fid_floor.py
Le PLANCHER du FID AFHQ-32, et les FID des modeles recalcules contre la MEME
reference — l'equivalent de la ligne "train vs. test reference" de tab:quantitative.

Pourquoi c'est necessaire : un FID entre deux echantillons FINIS de la MEME loi
n'est pas nul. Sans ce plancher, un FID de 10 ou de 46 n'a pas d'echelle. Le cache
AFHQ contient TOUS les chats (5153 train + 500 val = 5653), il n'y a donc pas de
split de test sur disque : on en fabrique un.

Protocole, identique pour les trois mesures :
    reference R = 3653 chats reels
    plancher    = FID(R, 2000 chats reels DISJOINTS de R)
    modele      = FID(R, 2000 images generees)
Les 2000 reels tenus a l'ecart ne sont jamais dans R, donc le plancher mesure
exactement ce que couterait un generateur PARFAIT a ce nombre d'echantillons.

Les FID annonces precedemment utilisaient R = 5653 (tous les reels) : ils ne sont
pas comparables a ce plancher, d'ou le recalcul complet ici.

Usage :  python compute_afhq_fid_floor.py
"""
import os, shutil
import numpy as np
import torch
from PIL import Image

CACHE = "./data/afhq_cat32_train.pt"
WORK = "/tmp/claude-829643295/-home-ec4036/ac843e58-587d-4be8-96f8-ea5eaaade836/scratchpad/afhq_fid"
GEN = {"ScCP LFO k15/K20/ic256": "ConvScCP_UNN_rgb_k15_K20_ic256_L1_LFO",
       "UNet torchcfm ch64":     "UNet_torchcfm_ch64"}
N_HELD = 2000
SEED = 0


def dump(arr, folder):
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)
    for i, im in enumerate(arr):
        Image.fromarray(im).save(os.path.join(folder, f"{i:06d}.png"))
    return folder


def main():
    d = torch.load(CACHE, map_location="cpu", weights_only=False)
    x = d["data"]                                   # uint8 (N,3,32,32)
    n = x.shape[0]
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n, generator=g)
    held, ref = perm[:N_HELD], perm[N_HELD:]        # DISJOINTS par construction
    assert len(set(held.tolist()) & set(ref.tolist())) == 0

    to_np = lambda t: x[t].permute(0, 2, 3, 1).numpy()
    ref_dir = dump(to_np(ref), os.path.join(WORK, "ref_3653"))
    held_dir = dump(to_np(held), os.path.join(WORK, "held_2000"))
    print(f"reference : {len(ref)} reels | tenus a l'ecart : {len(held)} reels")

    from cleanfid import fid
    floor = fid.compute_fid(ref_dir, held_dir, mode="clean", num_workers=2)
    print(f"\n{'Jeu compare a la reference':<34}{'N':>7}{'FID':>9}")
    print("-" * 50)
    print(f"{'2000 chats REELS (plancher)':<34}{N_HELD:>7}{floor:>9.2f}")
    for label, folder in GEN.items():
        p = os.path.join(WORK, folder)
        if not os.path.isdir(p):
            print(f"{label:<34}{'?':>7}{'ABSENT':>9}"); continue
        s = fid.compute_fid(ref_dir, p, mode="clean", num_workers=2)
        print(f"{label:<34}{len(os.listdir(p)):>7}{s:>9.2f}   "
              f"(x{s/floor:.1f} le plancher)")


if __name__ == "__main__":
    main()
