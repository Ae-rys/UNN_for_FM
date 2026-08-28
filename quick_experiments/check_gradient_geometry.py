# -*- coding: utf-8 -*-
"""
check_gradient_geometry.py

Depuis le passage a t ~ U(0, t_max) sans clamp, UNet (v-pred) et ScCP (x-pred)
optimisent la MEME quantite : ||v_pred - ut||^2 sur le meme domaine en t
(verifie a l'exactitude machine par check_tmax_equivalence.py).

Meme loss ne veut pas dire meme probleme d'optimisation. Pour un modele x-pred,
    dL/dtheta = dL/dx1p . dx1p/dtheta   avec  dL/dx1p ~ 1/(1-t)^2,
donc un echantillon a t proche de t_max pese jusqu'a 1/(1-t_max)^2 fois plus dans
le gradient que la meme erreur a t=0 — alors que pour un v-pred tous les t pesent
1 dans l'espace de sortie du reseau. Avec grad_clip=1.0 partage, le clipping ne
mord donc pas de la meme facon sur les deux familles.

Ce script mesure ce qui reste asymetrique :
  A. sur des batchs realistes (t ~ U(0, t_max)) : loss, norme du gradient AVANT
     clipping, et frequence a laquelle grad_clip=1.0 mord ;
  B. a t fixe : norme du gradient en fonction de t -> ou est la masse.

Sorties -> check_gradient_geometry.png / .txt

Usage
-----
    source ~/.venvs/unn/bin/activate
    python check_gradient_geometry.py --device cuda:0
"""

import argparse
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

from compute_fid_cifar10 import build_from_name
from run_cifar10_torchcfm_recipe import RECIPE, t_max_for

IMG_SIZE, CHANNELS = 32, 3
DIM = CHANNELS * IMG_SIZE * IMG_SIZE
RES = "results_afhq32"

RUNS = [
    ("UNet_torchcfm_ch32 (v-pred)", f"{RES}/UNet_torchcfm_ch32/ckpt_step_200000.pt"),
    ("ConvScCP k9/K10/ic128 (x-pred)",
     f"{RES}/ConvScCP_UNN_rgb_k9_K10_ic128_L1_LFO/latest.pt"),
]
T_GRID = [0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]


def loss_of(model, is_unet, xt, ut, t):
    """La loss TELLE QU'ELLE EST MAINTENANT dans train_one, branche par branche.
    Les deux expressions sont egales terme a terme ; seul le chemin de gradient differe."""
    B = xt.shape[0]
    tc = t.view(-1, 1)
    if is_unet:
        xt_img = xt.view(B, CHANNELS, IMG_SIZE, IMG_SIZE)
        ut_img = ut.view(B, CHANNELS, IMG_SIZE, IMG_SIZE)
        return torch.mean((model(t, xt_img) - ut_img) ** 2)
    out = model(torch.cat([xt, tc], dim=-1))       # .train() -> x1_pred
    x1_tgt = xt + ut * (1 - tc)
    return torch.mean((out - x1_tgt) ** 2 / (1 - tc) ** 2)


def grad_norm(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward()
    sq = sum(float(p.grad.pow(2).sum()) for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    return sq ** 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--cache", type=str, default="./data/afhq_cat32_train.pt")
    p.add_argument("--euler-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-batch", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)              # le plan OT tire via le RNG numpy global
    t_max = t_max_for(args.euler_steps)
    clip = RECIPE["grad_clip"]
    B, NB = args.batch_size, args.n_batch

    obj = torch.load(args.cache, map_location="cpu", weights_only=False)
    data = obj["data"].float().div_(127.5).sub_(1.0)
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    models = []
    for label, path in RUNS:
        if not os.path.exists(path):
            print(f"  [skip] {path} absent"); continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m, is_unet = build_from_name(ckpt["name"], device)
        m.load_state_dict(ckpt["state_dict"], strict=True)   # poids bruts : ce sont
        m.t_max = t_max                                      # eux qui recoivent le grad
        m.train()
        models.append((label, m, is_unet))
        print(f"  {label:<32} step {ckpt.get('step')} | t_max={t_max}", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(data.shape[0], generator=g)[: B * NB]
    x1_all = data[idx].reshape(NB, B, -1).to(device)
    x0_all = torch.randn(NB, B, DIM, generator=g).to(device)
    pairs = [FM.ot_sampler.sample_plan(x0_all[b], x1_all[b]) for b in range(NB)]

    t0 = time.perf_counter()
    lines = ["=" * 76,
             f"A. Batchs realistes : t ~ U(0, {t_max:g}), OT, {NB} x {B} images, "
             f"grad_clip={clip}", "=" * 76,
             f"{'modele':<34}{'loss':>9}{'|grad| med':>12}{'|grad| p95':>12}"
             f"{'clip mord':>11}"]
    stats = {}
    for label, model, is_unet in models:
        ls, gs = [], []
        for b, (x0, x1) in enumerate(pairs):
            tg = torch.Generator(device="cpu").manual_seed(1000 + b)
            t = (torch.rand(B, generator=tg) * t_max).to(device)
            xt = (1 - t.view(-1, 1)) * x0 + t.view(-1, 1) * x1
            ut = x1 - x0
            loss = loss_of(model, is_unet, xt, ut, t)
            ls.append(float(loss.detach())); gs.append(grad_norm(model, loss))
        gs = np.array(gs)
        stats[label] = gs
        lines.append(f"{label:<34}{np.mean(ls):>9.4f}{np.median(gs):>12.3f}"
                     f"{np.percentile(gs, 95):>12.3f}"
                     f"{100*np.mean(gs > clip):>10.0f}%")
    lines += ["",
              "loss : identique par construction (meme t, meme quantite). |grad| : norme",
              "AVANT clipping. « clip mord » = part des steps ou grad_clip la rabote,",
              "donc ou la DIRECTION est conservee mais l'amplitude perdue.", ""]

    lines += ["=" * 76, "B. Norme du gradient a t FIXE (meme batch pour les deux)",
              "=" * 76,
              f"{'t':>7}" + "".join(f"{lab[:24]:>26}" for lab, _, _ in models)]
    curves = {lab: [] for lab, _, _ in models}
    for tv in T_GRID:
        row = f"{tv:>7.2f}"
        for label, model, is_unet in models:
            gs = []
            for x0, x1 in pairs[:8]:
                t = torch.full((B,), tv, device=device)
                xt = (1 - tv) * x0 + tv * x1
                gs.append(grad_norm(model, loss_of(model, is_unet, xt, x1 - x0, t)))
            curves[label].append(float(np.median(gs)))
            row += f"{curves[label][-1]:>26.3f}"
        lines.append(row)
    lines += ["",
              "Un v-pred a un gradient d'amplitude comparable a tous les t. Un x-pred le",
              f"voit croitre comme 1/(1-t)^2, borne a {1/(1-t_max)**2:.0f} par t_max : c'est",
              "la meme loss, mais pas le meme probleme d'optimisation.", ""]

    # ---- C. part du gradient portee par la queue en t -------------------------
    T_CUT = 0.9
    lines += ["=" * 76,
              f"C. Part du gradient du batch portee par les echantillons t > {T_CUT}",
              "=" * 76,
              f"{'modele':<34}{'% du batch':>12}{'||g_queue||/||g||':>20}"]
    for label, model, is_unet in models:
        fr, share = [], []
        for b, (x0, x1) in enumerate(pairs[:12]):
            tg = torch.Generator(device="cpu").manual_seed(1000 + b)
            t = (torch.rand(B, generator=tg) * t_max).to(device)
            hi = t > T_CUT
            if int(hi.sum()) == 0:
                continue
            xt = (1 - t.view(-1, 1)) * x0 + t.view(-1, 1) * x1
            ut = x1 - x0
            g_full = grad_norm(model, loss_of(model, is_unet, xt, ut, t))
            # meme normalisation (/B) que la loss complete : on isole la CONTRIBUTION
            # de la queue au gradient du batch, pas sa loss moyenne.
            scale = int(hi.sum()) / B
            g_hi = grad_norm(model,
                             loss_of(model, is_unet, xt[hi], ut[hi], t[hi]) * scale)
            fr.append(100 * scale); share.append(g_hi / max(g_full, 1e-12))
        lines.append(f"{label:<34}{np.mean(fr):>11.1f}%{np.mean(share):>20.2f}")
    lines += ["",
              f"Si la queue portait tout, le ratio vaudrait ~1 ; s'il vaut ~ sa part en",
              f"nombre ({np.mean(fr):.0f} %), aucun regime ne domine.", ""]

    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    with open("check_gradient_geometry.txt", "w") as f:
        f.write(txt + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for (lab, _, _), st in zip(models, ["-o", "-s"]):
        ax[0].plot(T_GRID, curves[lab], st, ms=4, label=lab)
    ax[0].set_yscale("log"); ax[0].set_xlabel("t"); ax[0].set_ylabel("|grad| (mediane)")
    ax[0].set_title("B. Norme du gradient a t fixe"); ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    ax[1].boxplot([stats[lab] for lab, _, _ in models],
                  tick_labels=[lab.split(" (")[0] for lab, _, _ in models], showfliers=False)
    ax[1].axhline(clip, color="r", ls="--", lw=1, label=f"grad_clip={clip}")
    ax[1].set_yscale("log"); ax[1].set_ylabel("|grad| avant clipping")
    ax[1].set_title(f"A. Batchs t ~ U(0, {t_max:g})"); ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.suptitle("Meme loss, meme domaine en t — mais pas la meme geometrie de gradient")
    fig.tight_layout(); fig.savefig("check_gradient_geometry.png", dpi=130)
    print(f"-> check_gradient_geometry.png / .txt   ({time.perf_counter()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
