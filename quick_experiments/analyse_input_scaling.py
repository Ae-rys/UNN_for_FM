# -*- coding: utf-8 -*-
"""
analyse_input_scaling.py

Faut-il donner x_t/t (ou une version mise a l'echelle) au ScCP plutot que x_t ?

Le ScCP resout  min_x  1/2||z - x||^2 + g(Wx).  Le terme d'attache suppose donc
une observation de la forme  z = x + bruit,  a GAIN UNITE sur le signal. Or le
code passe z = x_t = t.x1 + (1-t).eps : la composante signal vaut t.x1, contractee
d'un facteur t. D'ou la proposition :

    sigma(t) = (1-t)/t                     bruit du probleme de debruitage
    s(t)     = sqrt(sigma_data^2 + sigma^2) echelle naturelle de x_t/t
    z_tilde  = (x_t/t)/s(t) = x_t / sqrt(t^2 sigma_data^2 + (1-t)^2)
    x1_pred  = s(t) . ScCP(z_tilde, t)

Le rescaling entree/sortie par le MEME s est gratuit en expressivite (il se
reporte sur g, qui est appris et conditionne en t), donc on peut choisir s pour
normaliser l'entree. Reste a savoir ce que ca coute ailleurs.

Ce script produit deux choses
-----------------------------
A. CONDITIONNEMENT — variance d'entree, variance de la cible, et facteur de
   gradient sur la sortie du reseau, avant et apres, sur une grille en t.
   La loss x-pred vaut ||x1p - x1||^2/(1-t)^2 et x1p - x1 = (1-t).r avec r le
   residu en vitesse, donc  dL/d(sortie) ∝ s(t)/(1-t)  au lieu de  1/(1-t).

B. LE DEFAUT EST-IL REEL ? — on mesure le gain empirique
       a(t) = <x1_pred, x1> / ||x1||^2
   du ScCP entraine, compare au UNet (temoin, sans probleme de gain) et au
   debruiteur lineaire OPTIMAL (Wiener gaussien). Toute E[x1|x_t] est contractee
   vers la moyenne — c'est la regression vers la moyenne, pas un defaut. La
   question est donc : le ScCP est-il contracte PLUS QUE l'optimum ? Si oui, le
   probleme de gain est reel et non compense. Sinon, la proposition ne sert a rien
   et il ne faut pas depenser 20 h de GPU dessus.

Sorties -> analyse_input_scaling.png / .txt

Usage
-----
    source ~/.venvs/unn/bin/activate
    python analyse_input_scaling.py --device cuda:0
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_fid_cifar10 import build_from_name

IMG_SIZE, CHANNELS = 32, 3
DIM = CHANNELS * IMG_SIZE * IMG_SIZE
N_VAL, SEED = 512, 0

MODELS = [
    ("ConvScCP (x-pred)", "results_afhq32/ConvScCP_UNN_rgb_k9_K10_ic128_L1_LFO/ckpt_step_40000.pt", "#C05621"),
    ("UNet ch32 (temoin)", "results_afhq32_tmax095/UNet_torchcfm_ch32/ckpt_step_200000.pt", "#1B5E8C"),
]
T_GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--cache", type=str, default="./data/afhq_cat32_train.pt")
    p.add_argument("--sccp", type=str, default=None,
                   help="checkpoint ScCP a analyser (defaut : le plus avance trouve)")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    d = torch.load(args.cache, map_location="cpu", weights_only=False)
    x = d["data"].float().div_(127.5).sub_(1.0)
    g = torch.Generator().manual_seed(args.seed)
    x = x[torch.randperm(x.shape[0], generator=g)]
    flat = x.reshape(x.shape[0], -1).double()
    xva, xtr = flat[:N_VAL], flat[N_VAL:]
    sd2 = float(xva.var())
    sd = sd2 ** 0.5

    lines = ["=" * 84,
             f"sigma_data mesure sur AFHQ-chats 32 : {sd:.4f}   (EDM prend 0.5)",
             "=" * 84, "",
             "A. CONDITIONNEMENT — ce que le rescaling change",
             "-" * 84,
             f"{'t':>6}{'sigma(t)':>10}{'s(t)':>9}"
             f"{'var(z) act.':>13}{'var(z) prop.':>14}"
             f"{'var cible':>11}{'grad act.':>11}{'grad prop.':>12}"]
    rows = []
    for t in T_GRID:
        sig = (1 - t) / t
        s = (sd2 + sig ** 2) ** 0.5
        var_now = t ** 2 * sd2 + (1 - t) ** 2          # var(x_t)
        var_new = 1.0                                   # par construction
        var_tgt = sd2 / s ** 2                          # var(x1/s), cible du reseau
        gr_now = 1.0 / (1 - t)
        gr_new = s / (1 - t)
        rows.append((t, sig, s, var_now, var_new, var_tgt, gr_now, gr_new))
        lines.append(f"{t:>6.2f}{sig:>10.2f}{s:>9.3f}{var_now:>13.3f}"
                     f"{var_new:>14.3f}{var_tgt:>11.4f}{gr_now:>11.2f}{gr_new:>12.2f}")
    gn = np.array([r[6] for r in rows]); gp = np.array([r[7] for r in rows])
    lines += ["",
              f"amplitude du facteur de gradient : actuel x{gn.max()/gn.min():.1f}  "
              f"(monotone croissant),  propose x{gp.max()/gp.min():.1f}  (en U)",
              f"a grand t le facteur passe de {gn[-1]:.1f} a {gp[-1]:.1f} (divise par "
              f"{gn[-1]/gp[-1]:.1f}) ; a petit t il monte de {gn[0]:.2f} a {gp[0]:.1f}.",
              ""]

    # ---------------------------------------------------------------- partie B
    sccp = args.sccp
    if sccp is None:
        cands = ["results_afhq32_tmax095/ConvScCP_UNN_rgb_k9_K10_ic128_L1_LFO/latest.pt",
                 "results_afhq32/ConvScCP_UNN_rgb_k9_K10_ic128_L1_LFO/latest.pt"]
        sccp = next(c for c in cands if os.path.exists(c))
    runs = [("ConvScCP", sccp, "#C05621"), ("UNet ch32", MODELS[1][1], "#1B5E8C")]

    gv = torch.Generator().manual_seed(args.seed + 1234)
    x0 = torch.randn(N_VAL, DIM, generator=gv).double()

    # --- reference : gain du debruiteur lineaire optimal (Wiener gaussien) ---
    mu = xtr.mean(dim=0, keepdim=True)
    xc = xtr - mu
    lam, U = torch.linalg.eigh(xc.T @ xc / xc.shape[0])
    lam = lam.clamp_min(0)
    xva_c = xva - mu
    denom = float((xva_c ** 2).sum())

    gains = {"lineaire optimal": []}
    alphas, dmse = {}, {}
    for t in T_GRID:
        xt = (1 - t) * x0 + t * xva
        gain = t * lam / (t ** 2 * lam + (1 - t) ** 2)
        pred = mu + ((xt - t * mu) @ U * gain) @ U.T
        gains["lineaire optimal"].append(float(((pred - mu) * xva_c).sum()) / denom)

    x1_d = xva.to(device).float()
    x0_d = x0.to(device).float()
    mu_d = mu.to(device).float()
    denom_d = float(((x1_d - mu_d) ** 2).sum())

    lines += ["=" * 84,
              "B. LE DEFAUT EST-IL REEL ? gain empirique a(t) = <x1p-mu, x1-mu>/||x1-mu||^2",
              "=" * 84,
              "a(t) = 1 : aucune contraction. a(t) < optimum : le modele contracte TROP.",
              "-" * 84,
              f"{'t':>6}{'lineaire opt.':>16}" + "".join(f"{n:>16}" for n, _, _ in runs)
              + f"{'ScCP/opt':>11}"]
    for nm, path, _ in runs:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m, is_unet = build_from_name(ck["name"], device)
        m.load_state_dict(ck["ema_model"], strict=True)
        m.t_max = ck.get("t_max")
        m.eval() if is_unet else m.train()
        gains[nm], alphas[nm], dmse[nm] = [], [], []
        with torch.no_grad():
            for t in T_GRID:
                tc = torch.full((N_VAL, 1), t, device=device)
                xt = (1 - tc) * x0_d + tc * x1_d
                if is_unet:
                    v = m(tc.view(-1), xt.view(-1, CHANNELS, IMG_SIZE, IMG_SIZE)).reshape(N_VAL, -1)
                    x1p = xt + (1 - tc) * v
                else:
                    x1p = m(torch.cat([xt, tc], dim=-1))
                num = float(((x1p - mu_d) * (x1_d - mu_d)).sum())
                gains[nm].append(num / denom_d)
                # alpha* = rescaling scalaire OPTIMAL de la sortie du modele.
                # C'est LE test du biais de gain : un modele plus faible doit
                # contracter davantage (c'est optimal), mais un modele bien calibre
                # a alpha* = 1. alpha* > 1 => sortie systematiquement trop petite.
                den_p = float(((x1p - mu_d) ** 2).sum())
                alphas[nm].append(num / den_p)
                mse0 = float(((x1p - x1_d) ** 2).mean())
                a = num / den_p
                mse1 = float(((mu_d + a * (x1p - mu_d) - x1_d) ** 2).mean())
                dmse[nm].append(100.0 * (1.0 - mse1 / mse0))
        print(f"  {nm:<12} {path}  step {ck.get('step')}", flush=True)

    for i, t in enumerate(T_GRID):
        opt = gains["lineaire optimal"][i]
        row = f"{t:>6.2f}{opt:>16.4f}" + "".join(f"{gains[n][i]:>16.4f}" for n, _, _ in runs)
        lines.append(row + f"{gains['ConvScCP'][i]/opt:>11.3f}")

    r = np.array([gains["ConvScCP"][i] / gains["lineaire optimal"][i]
                  for i in range(len(T_GRID))])
    ru = np.array([gains["UNet ch32"][i] / gains["lineaire optimal"][i]
                   for i in range(len(T_GRID))])
    lines += ["",
              f"ratio au lineaire optimal — ScCP : {r.min():.3f} a {r.max():.3f} "
              f"(moyenne {r.mean():.3f})",
              f"                            UNet : {ru.min():.3f} a {ru.max():.3f} "
              f"(moyenne {ru.mean():.3f})", ""]
    lines += ["=" * 84,
              "C. TEST DECISIF — alpha*(t), rescaling scalaire optimal de la sortie",
              "=" * 84,
              "alpha* = 1 : la sortie est bien calibree, rien a gagner a la rescaler.",
              "alpha* > 1 : sortie systematiquement trop PETITE = biais de gain reel.",
              "La derniere colonne donne le gain de MSE qu'on obtiendrait en appliquant",
              "ce rescaling optimal — c'est le PLAFOND de ce que ma proposition peut rapporter.",
              "-" * 84,
              f"{'t':>6}" + "".join(f"{n + ' alpha*':>20}" for n, _, _ in runs)
              + "".join(f"{n + ' gain MSE':>22}" for n, _, _ in runs)]
    for i, t in enumerate(T_GRID):
        lines.append(f"{t:>6.2f}"
                     + "".join(f"{alphas[n][i]:>20.4f}" for n, _, _ in runs)
                     + "".join(f"{dmse[n][i]:>21.3f}%" for n, _, _ in runs))
    asc = np.array(alphas["ConvScCP"]); dsc = np.array(dmse["ConvScCP"])
    lines += ["",
              f"ScCP : alpha* de {asc.min():.4f} a {asc.max():.4f} (moyenne {asc.mean():.4f})",
              f"       gain de MSE maximal atteignable par rescaling : {dsc.max():.3f} %",
              ""]

    verdict = (
        f"alpha* moyen = {asc.mean():.4f}, gain de MSE plafonne a {dsc.max():.3f} %. "
        + ("La sortie du ScCP est deja calibree : le biais de gain que ma proposition "
           "corrige N'EXISTE PAS en pratique, le reseau l'a appris."
           if abs(asc.mean() - 1) < 0.03 else
           "La sortie du ScCP est mal calibree : le biais de gain est reel."))
    lines += ["VERDICT : " + verdict, ""]

    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    open("analyse_input_scaling.txt", "w").write(txt + "\n")

    fig, ax = plt.subplots(1, 4, figsize=(20.5, 4.6))
    ax[0].plot(T_GRID, [r_[3] for r_ in rows], "-o", ms=4, color="#C05621", label="actuel : var($x_t$)")
    ax[0].axhline(1.0, color="#1B5E8C", ls="--", lw=1.5, label="propose : variance 1")
    ax[0].plot(T_GRID, [r_[5] for r_ in rows], "-s", ms=4, color="#2F855A",
               label="cible du reseau, $x_1/s(t)$")
    ax[0].set_yscale("log"); ax[0].set_title("Variances vues par le reseau")
    ax[1].plot(T_GRID, gn, "-o", ms=4, color="#C05621", label="actuel  $1/(1-t)$")
    ax[1].plot(T_GRID, gp, "-o", ms=4, color="#1B5E8C", label="propose  $s(t)/(1-t)$")
    ax[1].set_yscale("log"); ax[1].set_title("Facteur de gradient sur la sortie")
    ax[2].plot(T_GRID, gains["lineaire optimal"], "k--", lw=1.6, label="lineaire optimal")
    for n, _, col in runs:
        ax[2].plot(T_GRID, gains[n], "-o", ms=4, color=col, label=n)
    ax[2].set_title("$a(t)$ — le MAUVAIS test")

    # panneau decisif : alpha*, rescaling scalaire optimal de la sortie.
    ax[3].axhline(1.0, color="#444444", ls="--", lw=1.5, label="calibration parfaite")
    for n, _, col in runs:
        ax[3].plot(T_GRID, alphas[n], "-o", ms=4, color=col, label=n)
    ax[3].fill_between(T_GRID, 0.97, 1.03, color="#2F855A", alpha=0.12,
                       label="+/- 3 %")
    ax[3].set_ylim(0.6, 1.2)
    ax[3].set_title(r"$\alpha^*(t)$ — LE test : la sortie est-elle mal calibree ?")
    for a in ax:
        a.set_xlabel("t"); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle("Donner $x_t$ ou $x_t/t$ au ScCP — ce que le rescaling changerait "
                 "(1, 2) et pourquoi il ne sert a rien (3, 4)", fontsize=11)
    fig.tight_layout(); fig.savefig("analyse_input_scaling.png", dpi=130)
    print("-> analyse_input_scaling.png / .txt", flush=True)


if __name__ == "__main__":
    main()
