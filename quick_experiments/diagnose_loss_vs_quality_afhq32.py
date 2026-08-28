# -*- coding: utf-8 -*-
"""
diagnose_loss_vs_quality_afhq32.py

POURQUOI le ConvScCP affiche une loss PLUS BASSE que UNet_torchcfm_ch32 sur AFHQ-32
tout en produisant de moins bons echantillons ?

Hypothese testee : les deux nombres ne mesurent PAS la meme chose.

    run_cifar10_torchcfm_recipe.train_one, l.367 et l.388-393

      UNet   (is_unet=True, v-pred) :
          loss = mean( (v_pred - ut)^2 )                       <- MSE vitesse PURE

      ConvScCP / MinimalUNetFM (predicts_x1=True, x-pred) :
          loss = mean( (x1_pred - x1)^2 / clamp((1-t)^2, min=0.05) )

    Or  ||x1_pred - x1||^2 / (1-t)^2  ==  ||v_pred - ut||^2  EXACTEMENT.
    Le clamp min=0.05 mord des que (1-t)^2 < 0.05, i.e. t > 0.7764 : sur les
    22.4 % SUPERIEURS de l'axe des temps, la loss x-pred DIVISE l'erreur vitesse
    par (1-t)^2/0.05 -> elle sous-estime la vraie MSE-vitesse, sans borne
    (facteur 0.02 a t=0.99). Le 0.130 du ScCP et le 0.171 du UNet ne sont donc
    pas comparables.

Deuxieme anomalie testee : les seuils de clamp ne coincident PAS.
      - loss d'entrainement (x-pred) : clamp((1-t)^2, 0.05)  -> mord a t > 0.776
      - conversion en vitesse a l'eval (architectures.py, forward) :
                                       clamp(1-t, 0.05)      -> mord a t > 0.95
    Donc sur t in (0.776, 0.95) le modele est entraine avec un poids RABAISSE
    alors que le sampler amplifie son erreur par 1/(1-t) (jusqu'a x20).
    Et sur t > 0.95 le sampler renvoie une vitesse TROP PETITE d'un facteur
    (1-t)/0.05 -> la fin de trajectoire n'arrive jamais.

Ce script recalcule, sur les MEMES paires (x0,x1) couplees OT et les MEMES t :
    L_logged   la loss telle qu'elle est journalisee (specifique au modele)
    L_v_true   MSE vitesse SANS clamp                    <- metrique commune
    L_v_sampl  MSE de la vitesse REELLEMENT integree par l'EDO (avec le clamp
               d'eval du modele) -> ce que paie la qualite d'echantillon
    L_x1       MSE en espace x1, non ponderee            <- metrique commune bis

Sorties -> diagnose_loss_vs_quality_afhq32.png / .txt

Usage
-----
    source ~/.venvs/unn/bin/activate
    python diagnose_loss_vs_quality_afhq32.py --device cuda:0
"""

import argparse
import os
import time

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher)

from compute_fid_cifar10 import build_from_name

IMG_SIZE, CHANNELS = 32, 3
DIM = CHANNELS * IMG_SIZE * IMG_SIZE
RES = "results_afhq32"

# (nom lisible, chemin du checkpoint)  -- runs a 200k steps, meme recette
def runs_for(res_dir, ckpt):
    """ckpt : nom du fichier a charger dans chaque run_dir (ckpt_step_N.pt / latest.pt)."""
    return [
        ("UNet_torchcfm_ch32 (v-pred)", f"{res_dir}/UNet_torchcfm_ch32/{ckpt}"),
        ("ConvScCP k9/K10/ic128 (x-pred)",
         f"{res_dir}/ConvScCP_UNN_rgb_k9_K10_ic128_L1_LFO/{ckpt}"),
        ("MinimalUNetFM_kamb (x-pred)", f"{res_dir}/MinimalUNetFM_kamb/{ckpt}"),
    ]

T_GRID = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
          0.776, 0.85, 0.9, 0.95, 0.97, 0.99]


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    name = ckpt["name"]
    model, is_unet = build_from_name(name, device)
    key = "ema_model" if "ema_model" in ckpt else "state_dict"
    model.load_state_dict(ckpt[key], strict=True)
    step = ckpt.get("step", "?")
    # x-pred : .train() fait sortir x1_pred brut (pas de dropout/BN dans ces archis,
    # donc aucun effet de bord) ; le UNet torchcfm a du dropout -> .eval().
    model.eval() if is_unet else model.train()
    # t_max du run : None = ckpt d'avant le passage a la loss sans clamp. C'est lui
    # qui decide de la formule journalisee, sinon on compare a un objectif fantome.
    model.t_max = ckpt.get("t_max", None)
    return model, is_unet, name, step, key


@torch.no_grad()
def predict_x1(model, is_unet, xt_flat, t_col):
    """Retourne (x1_pred, v_sampler) aplatis.
    x1_pred : prediction de x1 SANS aucun clamp (metrique commune).
    v_sampler : la vitesse que l'EDO integrerait reellement pour ce modele."""
    B = xt_flat.shape[0]
    if is_unet:
        xt_img = xt_flat.view(B, CHANNELS, IMG_SIZE, IMG_SIZE)
        v = model(t_col.view(-1), xt_img).reshape(B, -1)
        x1p = xt_flat + (1.0 - t_col) * v
        return x1p, v                      # pas de clamp cote UNet
    out = model(torch.cat([xt_flat, t_col], dim=-1))   # .train() -> x1_pred
    v_sampl = (out - xt_flat) / torch.clamp(1.0 - t_col, min=0.05)
    return out, v_sampl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--cache", type=str, default="./data/afhq_cat32_train.pt")
    p.add_argument("--batch-size", type=int, default=128,
                   help="doit valoir 128 : le plan OT est minibatch-dependant.")
    p.add_argument("--n-batch", type=int, default=32, help="batches par point de t")
    p.add_argument("--coupling", type=str, default="ot", choices=["ot", "indep"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default=RES)
    p.add_argument("--ckpt", type=str, default="latest.pt",
                   help="fichier a charger dans chaque run_dir (ckpt_step_N.pt pour "
                        "comparer a budget apparie).")
    p.add_argument("--t-max", type=float, default=1.0,
                   help="restreint t au domaine d'entrainement du run (0.95 pour les "
                        "runs --euler-steps 20). Le tirage U(0,1) et la grille en t "
                        "sont tronques d'autant.")
    args = p.parse_args()
    runs = runs_for(args.results_dir, args.ckpt)
    t_grid = [t for t in T_GRID if t <= args.t_max + 1e-9]

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    # OTPlanSampler.sample_plan tire l'appariement via le RNG numpy GLOBAL : sans
    # cette graine, deux executions ne voient pas les memes paires et la moyenne
    # L_v_true bouge beaucoup (queue lourde en t->1, dominee par quelques
    # echantillons ou 1/(1-t)^2 vaut 10^4).
    import numpy as np
    np.random.seed(args.seed)

    obj = torch.load(args.cache, map_location="cpu", weights_only=False)
    # meme normalisation que get_afhq_loader : uint8 [0,255] -> [-1,1]
    data = obj["data"].float().div_(127.5).sub_(1.0)
    print(f"AFHQ cache : {tuple(data.shape)}  [{data.min():.2f}, {data.max():.2f}]",
          flush=True)

    FM = (ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
          if args.coupling == "ot" else ConditionalFlowMatcher(sigma=0.0))

    models = []
    for label, path in runs:
        if not os.path.exists(path):
            print(f"  [skip] {path} absent"); continue
        m, is_unet, name, step, key = load_model(path, device)
        n = sum(q.numel() for q in m.parameters())
        print(f"  {label:<32} step {step:>7} | {n/1e6:.2f}M | poids {key} | t_max {m.t_max}", flush=True)
        models.append((label, m, is_unet))

    B, NB = args.batch_size, args.n_batch
    # paires fixes, identiques pour TOUS les modeles et TOUS les t
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(data.shape[0], generator=g)[: B * NB]
    x1_all = data[idx].reshape(NB, B, -1).to(device)
    x0_all = torch.randn(NB, B, DIM, generator=g).to(device)

    # couplage OT applique une fois par batch (identique pour tous les t/modeles)
    pairs = []
    for b in range(NB):
        if args.coupling == "ot":
            a, c = FM.ot_sampler.sample_plan(x0_all[b], x1_all[b])
        else:
            a, c = x0_all[b], x1_all[b]
        pairs.append((a, c))

    t0 = time.perf_counter()
    # ---- A. reproduction de la loss journalisee (t ~ U(0,1), comme a l'entrainement)
    print("\n[A] loss journalisee reproduite (t ~ U(0,1))", flush=True)
    rep = {}
    for label, model, is_unet in models:
        tot_log = tot_v = 0.0
        for b, (x0, x1) in enumerate(pairs):
            tg = torch.Generator(device="cpu").manual_seed(1000 + b)
            t = (torch.rand(B, generator=tg) * args.t_max).to(device).view(-1, 1)
            xt = (1 - t) * x0 + t * x1
            ut = x1 - x0
            x1p, _ = predict_x1(model, is_unet, xt, t)
            if is_unet:
                v = (x1p - xt) / (1 - t)
                tot_log += torch.mean((v - ut) ** 2).item()
            elif model.t_max is None:
                tot_log += torch.mean((x1p - x1) ** 2          # runs legacy : clampee
                                      / torch.clamp((1 - t) ** 2, min=0.05)).item()
            else:
                tot_log += torch.mean((x1p - x1) ** 2          # runs t_max : sans clamp
                                      / (1 - t) ** 2).item()
            tot_v += torch.mean(((x1p - xt) / (1 - t) - ut) ** 2).item()
        rep[label] = (tot_log / NB, tot_v / NB)
        print(f"  {label:<32} L_logged={tot_log/NB:.4f}   L_v_true={tot_v/NB:.4f}",
              flush=True)

    # ---- B. profil par t, metriques communes
    print("\n[B] profil par t (metriques communes)", flush=True)
    curves = {lab: {k: [] for k in ("logged", "v_true", "v_sampl", "x1")}
              for lab, _, _ in models}
    for ti, tv in enumerate(t_grid):
        for label, model, is_unet in models:
            acc = dict(logged=0.0, v_true=0.0, v_sampl=0.0, x1=0.0)
            for x0, x1 in pairs:
                t = torch.full((B, 1), tv, device=device)
                xt = (1 - t) * x0 + t * x1
                ut = x1 - x0
                x1p, v_sampl = predict_x1(model, is_unet, xt, t)
                v_true = (x1p - xt) / (1 - t)
                acc["v_true"] += torch.mean((v_true - ut) ** 2).item()
                acc["v_sampl"] += torch.mean((v_sampl - ut) ** 2).item()
                acc["x1"] += torch.mean((x1p - x1) ** 2).item()
                w_log = ((1 - tv) ** 2 if model.t_max is not None
                         else max((1 - tv) ** 2, 0.05))
                acc["logged"] += (torch.mean((v_true - ut) ** 2).item() if is_unet
                                  else torch.mean((x1p - x1) ** 2).item() / w_log)
            for k in acc:
                curves[label][k].append(acc[k] / NB)
        el = time.perf_counter() - t0
        print(f"  t={tv:<6.3f}  ({ti+1}/{len(t_grid)})  ecoule {el:6.1f}s  "
              f"ETA {el/(ti+1)*(len(t_grid)-ti-1):6.1f}s", flush=True)

    # ---- sorties
    lines = []
    lines.append("=" * 78)
    lines.append("A. Loss journalisee reproduite (t~U(0,1), couplage %s, %d x %d images)"
                 % (args.coupling, NB, B))
    lines.append(f"   run_dir={args.results_dir}  ckpt={args.ckpt}  t~U(0,{args.t_max:g})")
    lines.append("=" * 78)
    lines.append(f"{'modele':<34}{'L_logged':>11}{'L_v_true':>11}{'ecart':>10}")
    for lab, (lg, vt) in rep.items():
        lines.append(f"{lab:<34}{lg:>11.4f}{vt:>11.4f}{vt/lg:>9.2f}x")
    lines.append("")
    lines.append("L_logged = ce qui est ecrit dans loss.txt / summary.txt.")
    lines.append("L_v_true = MSE vitesse sans clamp, la SEULE metrique commune.")
    lines.append("")
    for k, title in (("v_true", "B1. MSE vitesse SANS clamp (metrique commune)"),
                     ("v_sampl", "B2. MSE de la vitesse REELLEMENT integree par l'EDO"),
                     ("x1", "B3. MSE en espace x1 (non ponderee)"),
                     ("logged", "B4. loss journalisee, par t")):
        lines.append("=" * 78)
        lines.append(title)
        lines.append("=" * 78)
        hdr = f"{'t':>7}" + "".join(f"{lab[:26]:>28}" for lab, _, _ in models)
        lines.append(hdr)
        for i, tv in enumerate(t_grid):
            row = f"{tv:>7.3f}"
            for lab, _, _ in models:
                row += f"{curves[lab][k][i]:>28.4f}"
            lines.append(row)
        lines.append("")
    # ---- C. ou se joue l'ecart : integrale en t de part et d'autre du clamp
    import numpy as np
    tz = getattr(np, "trapezoid", None) or np.trapz
    tg = np.array(t_grid); lo = tg <= 0.7764

    def seg(y, m):
        y = np.array(y)
        return tz(y[m], tg[m]) / (tg[m][-1] - tg[m][0])

    lines.append("=" * 78)
    lines.append("C. Moyenne en t de part et d'autre du seuil de clamp (t = 0.7764)")
    lines.append("=" * 78)
    lines.append(f"{'modele':<34}{'metrique':<12}{'t<=0.776':>11}{'t>0.776':>11}")
    for lab, _, _ in models:
        for k, nm in (("v_true", "v-MSE"), ("logged", "journalisee")):
            lines.append(f"{lab:<34}{nm:<12}"
                         f"{seg(curves[lab][k], lo):>11.4f}"
                         f"{seg(curves[lab][k], ~lo):>11.4f}")
    lines.append("")
    lines.append("Sur t<=0.776 les deux colonnes sont IDENTIQUES par construction ;")
    lines.append("tout l'ecart de loss vient de t>0.776, ou le clamp remplace la vraie")
    lines.append("MSE vitesse des modeles x-pred par une valeur ~15x plus petite.")
    lines.append("")

    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    with open("diagnose_loss_vs_quality_afhq32.txt", "w") as f:
        f.write(txt + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    styles = ["-o", "-s", "-^"]
    for ax, k, title in zip(
            axes, ("logged", "v_true", "v_sampl"),
            ("loss JOURNALISEE (non comparable)",
             "MSE vitesse sans clamp (comparable)",
             "MSE de la vitesse integree par l'EDO")):
        for (lab, _, _), st in zip(models, styles):
            ax.plot(t_grid, curves[lab][k], st, ms=4, label=lab)
        ax.axvline(0.7764, color="k", ls=":", lw=1)
        ax.axvline(0.95, color="r", ls=":", lw=1)
        ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("MSE")
    axes[0].text(0.78, 0.03, "clamp loss\nt>0.776", transform=axes[0].transAxes,
                 fontsize=7)
    axes[2].text(0.80, 0.90, "clamp eval\nt>0.95", color="r",
                 transform=axes[2].transAxes, fontsize=7)
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("AFHQ-32 : pourquoi une loss plus basse ne veut pas dire "
                 "de meilleurs echantillons")
    fig.tight_layout()
    fig.savefig("diagnose_loss_vs_quality_afhq32.png", dpi=130)
    print("\n-> diagnose_loss_vs_quality_afhq32.png", flush=True)
    print(f"-> diagnose_loss_vs_quality_afhq32.txt", flush=True)
    print(f"Total {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
