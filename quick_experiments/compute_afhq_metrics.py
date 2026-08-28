# -*- coding: utf-8 -*-
"""
compute_afhq_metrics.py
De quoi remplir la ligne AFHQ-32 de tab:quantitative pour deux modeles :
le meilleur ConvScCP et le UNet torchcfm ch64.

Deux metriques, et une precaution.

  FID (clean-fid, mode "clean") entre N images generees et les 5653 chats AFHQ-32
  du cache d'entrainement. Il n'existe pas de stats de reference publiees pour
  AFHQ-32 : la reference est donc le jeu d'entrainement lui-meme, ce qui donne un
  FID comparable ENTRE CES DEUX MODELES mais pas a un FID publie.

  MSE-vitesse commune : ||v(x_t,t) - u_t||^2 sur le MEME lot, les MEMES (x0,x1) et
  les MEMES t ~ U(0, 0.95), pour les deux modeles.
  C'EST LA SEULE FACON DE REMPLIR LA COLONNE "Val. loss" HONNETEMENT : les loss
  journalisees pendant l'entrainement ne sont PAS comparables ici, parce que les
  deux runs n'ont pas le meme regime temporel --
      ConvScCP ... t_max=None -> ancien clamp min=0.05, qui mord des t>0.776 et
                   fait sortir une loss x-pred mecaniquement trop basse ;
      UNet ch64 .. t_max=0.95 -> pas de clamp, la loss EST la MSE-vitesse.
  Comparer 0.1238 a 0.1346 n'a donc aucun sens. On recalcule les deux ici.

L'echantillonnage tronque chaque modele a SON propre t_max (euler_sample).

Usage :  python compute_afhq_metrics.py [--num-gen 2000] [--device cuda:1]
"""
import argparse, os, shutil
import numpy as np
import torch
from PIL import Image

from run_cifar10_torchcfm_recipe import euler_sample, CHANNELS, IMG_SIZE, RECIPE
from compute_fid_cifar10 import build_from_name

RUNS = ["ConvScCP_UNN_rgb_k15_K20_ic256_L1_LFO", "UNet_torchcfm_ch64"]
CACHE = "./data/afhq_cat32_train.pt"
WORK = "/tmp/claude-829643295/-home-ec4036/ac843e58-587d-4be8-96f8-ea5eaaade836/scratchpad/afhq_fid"


def to_uint8(x):
    """(N,3,32,32) dans [-1,1] -> (N,32,32,3) uint8."""
    return (((x.clamp(-1, 1) + 1) / 2) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def dump(arr, folder):
    os.makedirs(folder, exist_ok=True)
    for i, im in enumerate(arr):
        Image.fromarray(im).save(os.path.join(folder, f"{i:06d}.png"))
    return folder


def load_real():
    d = torch.load(CACHE, map_location="cpu", weights_only=False)
    x = d["data"]
    return x.float().div_(127.5).sub_(1.0)          # uint8 [0,255] -> [-1,1], comme run_afhq32


@torch.no_grad()
def velocity_mse(model, is_unet, real, dev, n=512, seed=0, t_max=0.95):
    """MSE-vitesse sur un lot fixe : memes (x0,x1,t) pour tous les modeles."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(real.shape[0], generator=g)[:n]
    x1 = real[idx].to(dev)
    x0 = torch.randn(n, CHANNELS, IMG_SIZE, IMG_SIZE, generator=g).to(dev)
    t = (torch.rand(n, generator=g) * t_max).to(dev)
    tc = t.view(-1, 1, 1, 1)
    xt = (1 - tc) * x0 + tc * x1
    ut = x1 - x0
    if is_unet:
        v = model(t, xt)
    else:
        v = model(torch.cat([xt.reshape(n, -1), t.view(n, 1)], dim=-1)).view_as(xt)
    return float(((v - ut) ** 2).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-gen", type=int, default=2000)
    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()
    dev = torch.device(args.device)

    real = load_real()
    print(f"AFHQ-32 reel : {tuple(real.shape)}")
    real_dir = os.path.join(WORK, "real")
    if not os.path.isdir(real_dir) or len(os.listdir(real_dir)) != real.shape[0]:
        shutil.rmtree(real_dir, ignore_errors=True)
        dump(to_uint8(real), real_dir)
    print(f"reference : {len(os.listdir(real_dir))} images dans {real_dir}")

    from cleanfid import fid
    rows = []
    for name in RUNS:
        ck = torch.load(f"results_afhq32/{name}/latest.pt", map_location="cpu", weights_only=False)
        model, is_unet = build_from_name(ck["name"], dev)
        model.load_state_dict(ck["ema_model"]); model.eval()
        t_max = ck.get("t_max")

        mse = velocity_mse(model, is_unet, real, dev)

        gen_dir = os.path.join(WORK, name)
        shutil.rmtree(gen_dir, ignore_errors=True)
        outs, done = [], 0
        while done < args.num_gen:
            b = min(250, args.num_gen - done)
            im = euler_sample(model, is_unet, dev, n=b, n_steps=args.steps,
                              t_max=t_max, seed=1000 + done)
            outs.append(to_uint8(im.view(-1, CHANNELS, IMG_SIZE, IMG_SIZE)))
            done += b
            print(f"  {name}: {done}/{args.num_gen}", flush=True)
        dump(np.concatenate(outs, 0), gen_dir)
        score = fid.compute_fid(real_dir, gen_dir, mode="clean", num_workers=2)

        rows.append(dict(name=name, params=sum(q.numel() for q in model.parameters()),
                         step=ck.get("step"), hours=ck.get("train_time_s", 0) / 3600,
                         t_max=t_max, logged=(ck.get("loss_log") or [[None, None]])[-1][1],
                         mse=mse, fid=score))
        del model; torch.cuda.empty_cache()

    print(f"\n--- AFHQ-32 ({args.num_gen} generees vs {real.shape[0]} reelles) ---")
    h = f"{'Model':<40}{'#Params':>11}{'Steps':>9}{'Temps':>8}{'MSE-vit':>10}{'FID':>9}"
    print(h); print("-" * len(h))
    for r in rows:
        print(f"{r['name']:<40}{r['params']:>11,}{r['step']:>9,}{r['hours']:>7.1f}h"
              f"{r['mse']:>10.4f}{r['fid']:>9.2f}")
    print("\nLoss JOURNALISEES (NON comparables, regimes t_max differents) :")
    for r in rows:
        print(f"  {r['name']:<40} t_max={str(r['t_max']):<5} loss={r['logged']:.4f}")


if __name__ == "__main__":
    main()
