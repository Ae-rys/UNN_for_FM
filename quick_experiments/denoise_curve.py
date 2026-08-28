# -*- coding: utf-8 -*-
"""
denoise_curve.py
Trace la courbe nmse(t) sur une grille DENSE de niveaux de bruit, A POSTERIORI,
depuis les `model.pt` deja ecrits par denoise_probe.py. Aucun reentrainement.

A quoi ca sert
--------------
1. Les runs lances avec une version de denoise_probe.py qui n'evaluait qu'aux
   quelques t d'entrainement gardent quand meme leurs poids : on peut donc en
   extraire la courbe complete apres coup.
2. denoise_probe.py ecrit summary.txt / mse_vs_t.png A LA RACINE du results-dir :
   un second balayage ECRASE le resume du premier. Ici on relit tous les runs
   presents sur disque, k=9 et k=15 melanges, et on les trace ENSEMBLE — ce que
   le script d'entrainement ne peut pas faire puisqu'il ne connait que ses
   propres configs.

Ce que la courbe dit : la qualite de reconstruction a chaque niveau de bruit, et
si le modele interpole entre les t vus (marques par des points). Une bosse entre
deux t vus voudrait dire que la liste --t est trop clairsemee.

Usage
-----
    source ~/.venvs/unn/bin/activate

    # tous les runs presents
    python denoise_curve.py

    # comparer deux familles, filtrees par sous-chaine
    python denoise_curve.py --only k15 --out curve_k15.png
    python denoise_curve.py --only ic256

Sorties -> <results-dir>/curve_all.png + curve_all.csv (noms distincts de ceux
de denoise_probe.py : rien n'est ecrase, meme si un balayage tourne encore).
"""

import argparse
import gc
import os

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from denoise_probe import (
    build_model, evaluate, load_data, make_val_set, name_to_config,
    plot_summary, reference_mse,
)


def read_metrics(run_dir):
    """metrics.txt -> dict (sert a recuperer `selected` et les t d'entrainement)."""
    path = os.path.join(run_dir, "metrics.txt")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            if "=" in line and "\t" not in line:
                k, v = line.strip().split("=", 1)
                out[k] = v
    return out


def main():
    p = argparse.ArgumentParser(
        description="Courbe nmse(t) dense a posteriori, depuis les model.pt.")
    p.add_argument("--results-dir", type=str, default="results_denoise_probe")
    p.add_argument("--cache", type=str, default="./data/afhq_cat32_train.pt")
    p.add_argument("--eval-t", type=str, default="",
                   help="Defaut : 0.05 a 0.95 par pas de 0.05.")
    p.add_argument("--t", type=str, default="0.2,0.4,0.6,0.8",
                   help="t d'entrainement, pour les marqueurs et la moyenne du "
                        "classement (lu dans metrics.txt si present).")
    p.add_argument("--weights", type=str, default="auto", choices=["auto", "raw", "ema"],
                   help="auto (defaut) = ce que le run avait retenu (metrics.txt).")
    p.add_argument("--only", type=str, default="")
    p.add_argument("--skip", type=str, default="")
    p.add_argument("--n-val", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="curve_all.png")
    p.add_argument("--device", type=str, default="cuda:1")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and (device.index or 0) >= torch.cuda.device_count():
        device = torch.device("cuda:0")
    t_train = [float(s) for s in args.t.split(",") if s.strip()]
    if args.eval_t:
        eval_t = [float(s) for s in args.eval_t.split(",") if s.strip()]
    else:
        eval_t = [round(0.05 * i, 2) for i in range(1, 20)]
    eval_t = sorted(set(eval_t) | set(t_train))

    # MEMES seeds que denoise_probe.py -> meme split et meme bruit de validation,
    # donc les chiffres recalcules ici sont directement comparables aux metrics.txt.
    x_train, x_val = load_data(args.cache, args.n_val, device, seed=args.seed)
    channels, img_size = x_train.shape[1], x_train.shape[2]
    del x_train
    val_sets = make_val_set(x_val, eval_t, seed=args.seed + 1234)
    mse_mean, copy_ref = reference_mse(val_sets, x_val)

    runs = sorted(d for d in os.listdir(args.results_dir)
                  if os.path.exists(os.path.join(args.results_dir, d, "model.pt")))
    if args.only:
        runs = [r for r in runs if args.only in r]
    if args.skip:
        runs = [r for r in runs if args.skip not in r]
    if not runs:
        print(f"Aucun model.pt dans {args.results_dir} (filtres : only='{args.only}' "
              f"skip='{args.skip}').", flush=True)
        return
    print(f"{len(runs)} run(s) a evaluer sur {len(eval_t)} niveaux de bruit "
          f"[{eval_t[0]:g} .. {eval_t[-1]:g}] — device {device}\n", flush=True)

    results = []
    for name in runs:
        run_dir = os.path.join(args.results_dir, name)
        model = None
        try:
            meta = read_metrics(run_dir)
            ckpt = torch.load(os.path.join(run_dir, "model.pt"), map_location=device,
                              weights_only=False)
            which = args.weights if args.weights != "auto" else meta.get("selected", "raw")
            cfg = name_to_config(name)
            model = build_model(cfg, device, channels, img_size)
            model.load_state_dict(ckpt["ema_model" if which == "ema" else "state_dict"])
            mse = evaluate(model, val_sets, channels, img_size, cfg["arch"] == "unet_ref")
            nmse = {t: v / mse_mean for t, v in mse.items()}
            tt = [t for t in t_train if t in nmse]
            results.append(dict(name=name, mse=mse, nmse=nmse,
                                n_params=sum(p.numel() for p in model.parameters()),
                                nmse_mean=sum(nmse[t] for t in tt) / len(tt)))
            print(f"  {name:<44} [{which}] nmse {results[-1]['nmse_mean']:.4f}", flush=True)
        except Exception as exc:
            print(f"  [SAUTE] {name} : {exc}", flush=True)
        finally:
            model = None; gc.collect(); torch.cuda.empty_cache()

    if not results:
        return
    results.sort(key=lambda r: r["nmse_mean"])
    out_png = os.path.join(args.results_dir, args.out)
    plot_summary(results, mse_mean, copy_ref, out_png, t_train)

    out_csv = os.path.splitext(out_png)[0] + ".csv"
    with open(out_csv, "w") as f:
        f.write("name,n_params,nmse_mean," + ",".join(f"nmse_t{t:g}" for t in eval_t) + "\n")
        for r in results:
            f.write(f"{r['name']},{r['n_params']},{r['nmse_mean']:.6f}," +
                    ",".join(f"{r['nmse'][t]:.6f}" for t in eval_t) + "\n")
        f.write("[ref]copie_de_xt,,," +
                ",".join(f"{copy_ref[t]/mse_mean:.6f}" for t in eval_t) + "\n")
    print(f"\n-> {out_png}\n-> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
