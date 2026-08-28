# -*- coding: utf-8 -*-
"""
budget_matched.py
Compare les architectures A BUDGET DE STEPS APPARIE, en n'utilisant que des
checkpoints deja sur disque.

Le probleme qu'il resout
------------------------
Le plan factoriel du banc de debruitage a ete mesure a 3 000 steps, budget
auquel TOUTES les configurations sont moins bonnes qu'un simple filtre lineaire :
ses exposants decrivent une vitesse de convergence, pas une qualite atteignable.
Relancer le plan a 20 000 steps couterait ~33 h de GPU.

Mais les runs Flow Matching de results_afhq32 archivent leurs poids tous les
10 000 steps. On dispose donc deja de plusieurs architectures evaluees aux MEMES
budgets — il suffit de les lire. Aucun entrainement, seulement de l'evaluation.

Ce que le script produit
------------------------
  * la trajectoire nmse(steps) de chaque configuration ;
  * les tableaux a budget APPARIE, pour chaque budget ou au moins deux
    configurations existent ;
  * le budget a partir duquel chaque configuration passe sous le debruiteur
    lineaire optimal — c'est-a-dire a partir duquel elle devient comparable.

Toutes les nmse sont mesurees avec le protocole de denoise_probe.py : memes 512
images de validation tenues a l'ecart, meme bruit fixe, memes niveaux de bruit.

Usage
-----
    source ~/.venvs/unn/bin/activate
    python budget_matched.py
    python budget_matched.py --min-configs 3     # ne garder que les budgets denses

Sorties -> budget_matched.csv, budget_matched.png, budget_matched.txt
"""

import argparse
import gc
import glob
import os
import re

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from denoise_probe import evaluate, load_data, make_val_set, reference_mse


def short_name(name):
    m = re.match(r"ConvScCP_UNN_rgb_k(\d+)_K(\d+)_ic(\d+)", name)
    return f"k{m.group(1)}/K{m.group(2)}/ic{m.group(3)}" if m else name


def main():
    p = argparse.ArgumentParser(description="Comparaison a budget de steps apparie.")
    p.add_argument("--results-dir", type=str, default="results_afhq32")
    p.add_argument("--cache", type=str, default="./data/afhq_cat32_train.pt")
    p.add_argument("--t", type=str, default="0.2,0.4,0.6,0.8")
    p.add_argument("--weights", type=str, default="ema", choices=["ema", "raw"])
    p.add_argument("--n-val", type=int, default=512)
    p.add_argument("--min-configs", type=int, default=2,
                   help="Budget retenu s'il compte au moins ce nombre de configs.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:1")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and (device.index or 0) >= torch.cuda.device_count():
        device = torch.device("cuda:0")
    t_list = [float(s) for s in args.t.split(",") if s.strip()]

    _, x_val = load_data(args.cache, args.n_val, device, seed=args.seed)
    C, S = x_val.shape[1], x_val.shape[2]
    val_sets = make_val_set(x_val, t_list, seed=args.seed + 1234)
    mse_mean, _ = reference_mse(val_sets, x_val)

    # reference lineaire, sur le MEME tirage de bruit que les modeles
    from analyze_denoise_probe import wiener_floor
    emp, _ana, var = wiener_floor(args.cache, args.n_val, t_list, seed=args.seed)
    linear = float(np.mean([emp[t] / var for t in t_list]))

    from sample_checkpoint import resolve_checkpoint
    ckpts = sorted(glob.glob(os.path.join(args.results_dir, "*", "ckpt_step_*.pt")))
    print(f"{len(ckpts)} checkpoints a evaluer | reference lineaire = {linear:.4f}\n",
          flush=True)

    data = {}          # nom court -> {step: nmse}
    params = {}
    for c in ckpts:
        try:
            ck = torch.load(c, map_location=device, weights_only=False)
            model, is_unet, name, keys = resolve_checkpoint(ck, device)
            key = keys.get(args.weights)
            model.load_state_dict(ck[key if key in ck else keys["raw"]])
            step = int(ck.get("step", int(re.search(r"step_(\d+)", c).group(1))))
            v = float(np.mean(list(evaluate(model, val_sets, C, S, is_unet).values())) / mse_mean)
            sn = short_name(name)
            data.setdefault(sn, {})[step] = v
            params[sn] = sum(q.numel() for q in model.parameters())
            print(f"  {sn:<18} step {step:>7,}  nmse {v:.4f}", flush=True)
        except Exception as exc:
            print(f"  [saute] {os.path.basename(os.path.dirname(c))} : {exc}", flush=True)
        finally:
            model = None; gc.collect(); torch.cuda.empty_cache()

    if not data:
        return
    out = []

    def say(s=""):
        print(s, flush=True); out.append(s)

    # ---- budgets apparies ----
    all_steps = sorted({s for d in data.values() for s in d})
    say("\n" + "=" * 74)
    say(f"COMPARAISONS A BUDGET APPARIE  (reference lineaire optimale : {linear:.4f})")
    say("=" * 74)
    for step in all_steps:
        present = [(n, d[step]) for n, d in data.items() if step in d]
        if len(present) < args.min_configs:
            continue
        present.sort(key=lambda x: x[1])
        say(f"\n{step:,} steps — {len(present)} configuration(s)")
        say(f"  {'config':<18}{'params':>11}{'nmse':>9}   {'regime'}")
        for n, v in present:
            reg = "valide" if v < linear else f"SOUS-ENTRAINE (+{100*(v/linear-1):.1f} % vs lineaire)"
            say(f"  {n:<18}{params[n]:>11,}{v:>9.4f}   {reg}")

    # ---- franchissement de la reference lineaire ----
    say("\n" + "=" * 74)
    say("BUDGET A PARTIR DUQUEL LA CONFIGURATION DEVIENT COMPARABLE")
    say("=" * 74)
    for n, d in sorted(data.items(), key=lambda kv: params[kv[0]]):
        steps = sorted(d)
        cross = next((s for s in steps if d[s] < linear), None)
        best = min(d.values())
        say(f"  {n:<18}{params[n]:>11,}  "
            + (f"passe sous {linear:.4f} des {cross:,} steps" if cross
               else f"jamais (meilleur : {best:.4f} a {max(steps):,} steps)"))

    # ---- figure : vue d'ensemble + zoom sur la zone ou le classement se joue ----
    # (echelle lineaire, tout est ecrase apres 50k ; d'ou le second panneau)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = sorted(data.items(), key=lambda kv: -params[kv[0]])
    for ax, zoom in zip(axes, (False, True)):
        for n, d in order:
            steps = [s for s in sorted(d) if not zoom or s >= 40000]
            if not steps:
                continue
            ax.plot(steps, [d[s] for s in steps], "o-", ms=4, lw=1.6,
                    label=f"{n}  ({params[n]/1e6:.2f}M)")
        ax.axhline(linear, color="#B03A2E", lw=2, ls="--")
        ax.set_xlabel("steps d'entrainement")
        ax.set_ylabel("nmse (validation, bruit fixe)")
        if zoom:
            ax.set_ylim(0.192, 0.232)
            ax.set_title("Zoom : a partir de 40 000 steps, la ou le classement se joue",
                         fontsize=10)
            ax.text(42000, linear + 0.0007, "debruiteur lineaire optimal",
                    fontsize=8.5, color="#B03A2E")
        else:
            ax.set_yscale("log")
            ax.set_title("Vue d'ensemble (echelle log)", fontsize=10)
            ax.text(all_steps[0], linear * 1.03, "debruiteur lineaire optimal",
                    fontsize=8.5, color="#B03A2E")
            ax.legend(fontsize=7.5)
    plt.tight_layout(); plt.savefig("budget_matched.png", dpi=115); plt.close(fig)

    with open("budget_matched.csv", "w") as f:
        f.write("config,n_params,step,nmse\n")
        for n, d in data.items():
            for s in sorted(d):
                f.write(f"{n},{params[n]},{s},{d[s]:.6f}\n")
    with open("budget_matched.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n-> budget_matched.csv / .png / .txt", flush=True)


if __name__ == "__main__":
    main()
