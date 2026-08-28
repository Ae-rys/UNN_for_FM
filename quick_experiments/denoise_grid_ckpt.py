# -*- coding: utf-8 -*-
"""
denoise_grid_ckpt.py
Fait passer les modeles entraines en FLOW MATCHING COMPLET (results_afhq32/...)
sur le banc de debruitage a bruit fixe, et sort la grille x_t | prediction |
verite plus la nmse par niveau de bruit.

Pourquoi c'est le bon pont entre les deux experiences
-----------------------------------------------------
Ces modeles ont ete entraines sur t ~ U[0,1] : debruiter a un t fixe est un
SOUS-PROBLEME de ce qu'ils ont appris, aucune adaptation n'est necessaire. On
peut donc les comparer directement aux modeles de denoise_probe.py — memes 512
images de validation tenues a l'ecart, meme bruit fixe, meme metrique — alors
que les uns ont vu 45 000 a 165 000 steps de FM et les autres 3 000 a 20 000
steps de debruitage.

C'est aussi la seule facon de confronter la nmse aux ECHANTILLONS GENERES par
le meme poids (les step_N.png du run) : si un modele a une bonne nmse et sort
des pates, le proxy est invalide comme critere de selection, et ca se voit ici
sur une seule figure.

L'archi est reconstruite depuis le champ 'name' du checkpoint (build_from_name),
donc rien a repasser. Les poids EMA sont pris par defaut, comme pour les
step_N.png du run.

Usage
-----
    source ~/.venvs/unn/bin/activate

    # tous les runs d'un dossier de resultats
    python denoise_grid_ckpt.py --results-dir results_afhq32

    # un ou plusieurs checkpoints precis, poids bruts, autres niveaux de bruit
    python denoise_grid_ckpt.py --weights raw --t 0.3,0.6,0.9 \\
        --ckpt results_afhq32/ConvScCP_UNN_rgb_k9_K20_ic256_L1_LFO/latest.pt

    # une archive intermediaire, pour voir l'effet du budget
    python denoise_grid_ckpt.py --ckpt \\
        results_afhq32/ConvScCP_UNN_rgb_k15_K20_ic256_L1_LFO/ckpt_step_50000.pt

Sorties -> <dossier du checkpoint>/denoise_grid_<weights>_step_<N>.png
           + denoise_ckpt_summary.txt dans --out-dir
"""

import argparse
import gc
import glob
import os

import torch

from denoise_probe import evaluate, load_data, make_val_set, reference_mse, save_grid


def load_generative(ckpt_path, device, weights="ema"):
    """Reconstruit l'archi depuis le nom et charge les poids demandes.

    Retourne (model, is_unet, name, step). `is_unet` distingue les UNet
    torchcfm (appeles model(t, x_image)) des modeles vectoriels (x_t aplati),
    ce que forward_x1 / evaluate attendent."""
    from sample_checkpoint import resolve_checkpoint
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model, is_unet, name, keys = resolve_checkpoint(ck, device)   # prend le ckpt CHARGE
    key = keys.get(weights)
    if key is None or key not in ck:
        key = keys.get("raw", "state_dict")
        print(f"    [warn] poids '{weights}' absents -> repli sur '{key}'", flush=True)
    model.load_state_dict(ck[key])
    model.eval()
    return model, is_unet, name, int(ck.get("step", 0))


def main():
    p = argparse.ArgumentParser(
        description="Banc de debruitage applique aux checkpoints FM.")
    p.add_argument("--ckpt", nargs="*", default=[], help="Checkpoints .pt.")
    p.add_argument("--results-dir", type=str, default="",
                   help="Dossier de runs : prend le latest.pt de chaque sous-dossier.")
    p.add_argument("--cache", type=str, default="./data/afhq_cat32_train.pt")
    p.add_argument("--t", type=str, default="0.2,0.4,0.6,0.8")
    p.add_argument("--weights", type=str, default="ema", choices=["ema", "raw"])
    p.add_argument("--n-val", type=int, default=512)
    p.add_argument("--n-show", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="",
                   help="Defaut : a cote de chaque checkpoint.")
    p.add_argument("--device", type=str, default="cuda:1")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and (device.index or 0) >= torch.cuda.device_count():
        print(f"[warn] {device} invalide -> repli sur cuda:0", flush=True)
        device = torch.device("cuda:0")

    ckpts = list(args.ckpt)
    if args.results_dir:
        ckpts += sorted(glob.glob(os.path.join(args.results_dir, "*", "latest.pt")))
    if not ckpts:
        print("Rien a faire : donner --ckpt et/ou --results-dir.", flush=True)
        return

    t_list = [float(s) for s in args.t.split(",") if s.strip()]
    # MEMES seeds que denoise_probe.py -> meme split, meme bruit : les nmse
    # obtenues ici sont directement comparables a celles du banc.
    x_train, x_val = load_data(args.cache, args.n_val, device, seed=args.seed)
    channels, img_size = x_train.shape[1], x_train.shape[2]
    del x_train
    val_sets = make_val_set(x_val, t_list, seed=args.seed + 1234)
    mse_mean, copy_ref = reference_mse(val_sets, x_val)
    print(f"Device {device} | poids {args.weights} | t={t_list} | "
          f"var(donnees)={mse_mean:.4f}\n", flush=True)

    rows = []
    for c in ckpts:
        if not os.path.exists(c):
            print(f"  [absent] {c}", flush=True)
            continue
        model = None
        try:
            model, is_unet, name, step = load_generative(c, device, args.weights)
            n_params = sum(q.numel() for q in model.parameters())
            mse = evaluate(model, val_sets, channels, img_size, is_unet)
            nmse = {t: v / mse_mean for t, v in mse.items()}
            out_dir = args.out_dir or os.path.dirname(os.path.abspath(c))
            os.makedirs(out_dir, exist_ok=True)
            png = os.path.join(
                out_dir, f"denoise_grid_{args.weights}_step_{step}.png")
            save_grid(model, val_sets, png,
                      f"{name} — step {step:,} ({args.weights}) — "
                      f"nmse {sum(nmse.values())/len(nmse):.4f}",
                      channels, img_size, is_unet, n=args.n_show, t_show=t_list)
            rows.append((name, step, n_params, nmse, png))
            print(f"  {name:<40} step {step:>7,}  nmse "
                  f"{sum(nmse.values())/len(nmse):.4f}  -> {png}", flush=True)
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"  [ECHEC] {c} : {exc}", flush=True)
        finally:
            model = None; gc.collect(); torch.cuda.empty_cache()

    if not rows:
        return
    rows.sort(key=lambda r: sum(r[3].values()) / len(r[3]))
    header = (f"{'checkpoint':<42}{'steps':>9}{'params':>11}{'nmse':>8}  " +
              "".join(f"{'t='+format(t,'g'):>9}" for t in t_list))
    lines = [header, "-" * len(header)]
    for name, step, n_params, nmse, _ in rows:
        lines.append(f"{name:<42}{step:>9,}{n_params:>11,}"
                     f"{sum(nmse.values())/len(nmse):>8.4f}  " +
                     "".join(f"{nmse[t]:>9.3f}" for t in t_list))
    lines.append("-" * len(header))
    lines.append(f"{'[ref] copie de x_t':<42}{'':>9}{'':>11}{'':>8}  " +
                 "".join(f"{copy_ref[t]/mse_mean:>9.3f}" for t in t_list))
    lines.append(f"{'[ref] predicteur constant':<42}{'':>9}{'':>11}{'':>8}  " +
                 "".join(f"{1.0:>9.3f}" for t in t_list))
    table = "\n".join(lines)
    print("\n" + table, flush=True)
    out = os.path.join(args.out_dir or ".", "denoise_ckpt_summary.txt")
    with open(out, "w") as f:
        f.write(f"weights={args.weights} t={t_list} n_val={args.n_val} "
                f"var_donnees={mse_mean:.6f}\n\n{table}\n")
    print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
