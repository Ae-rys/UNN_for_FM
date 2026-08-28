# -*- coding: utf-8 -*-
"""
sampler_diagnosis.py
"Il debruite parfaitement mais il genere des pates" : ce script separe les deux
causes possibles.

  A. ERREUR DE DISCRETISATION — le solveur n'est pas assez fin.
     Test : meme bruit initial, nombre de pas d'Euler croissant, plus dopri5
     adaptatif. Si les images ne changent plus a partir d'un certain nombre de
     pas, l'integration est convergee et le solveur est HORS DE CAUSE.

  B. ERREUR DU CHAMP — v_theta est faux la ou ca compte.
     Test du DEPART RETARDE : au lieu de partir de t=0 avec du bruit pur, on
     part de t0 > 0 avec un VRAI x_t0 = (1-t0) x0 + t0 x1 construit sur une
     image de validation, et on integre de t0 a 1. Le modele recoit alors une
     trajectoire exacte jusqu'a t0 et ne doit la continuer que sur [t0, 1].
       - si le resultat est net des t0 petit  -> le champ est bon partout,
         le probleme serait ailleurs ;
       - s'il faut t0 grand pour que ce soit net -> le champ est faux sur
         [0, t0] : c'est la que la trajectoire s'engage vers un pate, et tout
         ce qui suit ne fait qu'affiner ce pate fidelement.
     Le balayage en t0 LOCALISE la panne sur l'axe du temps.

Pourquoi ce diagnostic est necessaire : une grille de debruitage a t=0.8 est
trompeuse. x_0.8 contient deja 80 % de l'image, donc la nettete de la prediction
est en grande partie HERITEE de l'entree, pas creee par le modele. Seule la
generation libre teste ce que le modele sait inventer.

Usage
-----
    source ~/.venvs/unn/bin/activate

    python sampler_diagnosis.py --ckpt \\
        results_afhq32/ConvScCP_UNN_rgb_k9_K20_ic256_L1_LFO/latest.pt

    # comparer le run inacheve et le run a 165k steps
    python sampler_diagnosis.py --ckpt A/latest.pt B/latest.pt --n 6

Sorties -> sampler_diag_<nom>_solver.png   (test A)
           sampler_diag_<nom>_start.png    (test B)
"""

import argparse
import gc
import os

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from denoise_probe import load_data


def _grid(rows, labels, path, title, channels, img_size):
    nr, n = len(rows), rows[0].shape[0]
    fig, axes = plt.subplots(nr, n, figsize=(1.35 * n, 1.45 * nr), squeeze=False)
    fig.suptitle(title, fontsize=10)
    for r, (row, lab) in enumerate(zip(rows, labels)):
        imgs = (row.view(-1, channels, img_size, img_size) * 0.5 + 0.5).clamp(0, 1)
        for c in range(n):
            ax = axes[r][c]
            im = imgs[c].permute(1, 2, 0).cpu().numpy()
            ax.imshow(im.squeeze() if channels == 1 else im,
                      cmap="gray" if channels == 1 else None)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(lab, fontsize=7, rotation=0, ha="right", va="center")
    plt.tight_layout(rect=(0.04, 0, 1, 0.96))
    plt.savefig(path, dpi=95); plt.close(fig)


@torch.no_grad()
def integrate(vf, x, t0, steps, device):
    """Euler de t0 a 1 en `steps` pas, depuis l'etat x (deja au temps t0)."""
    dt = (1.0 - t0) / steps
    for i in range(steps):
        t = torch.full((1,), t0 + i * dt, device=device)
        x = x + vf(t, x) * dt
    return x


def main():
    p = argparse.ArgumentParser(description="Solveur ou champ : qui fait les pates ?")
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--cache", type=str, default="./data/afhq_cat32_train.pt")
    p.add_argument("--weights", type=str, default="ema", choices=["ema", "raw"])
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--euler-steps", type=str, default="10,50,200,1000")
    p.add_argument("--t0", type=str, default="0.0,0.1,0.2,0.3,0.5,0.7")
    p.add_argument("--start-steps", type=int, default=200)
    p.add_argument("--n-val", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="",
                   help="Defaut : a cote du checkpoint (comme denoise_grid_ckpt.py).")
    p.add_argument("--device", type=str, default="cuda:1")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and (device.index or 0) >= torch.cuda.device_count():
        device = torch.device("cuda:0")
    steps_list = [int(s) for s in args.euler_steps.split(",")]
    t0_list = [float(s) for s in args.t0.split(",")]

    _, x_val = load_data(args.cache, args.n_val, device, seed=args.seed)
    C, S = x_val.shape[1], x_val.shape[2]
    x1 = x_val[:args.n]                                   # vraies images (B,C,S,S)

    from sample_checkpoint import resolve_checkpoint
    from compute_fid_cifar10 import _VelocityWrapper
    from torchdyn.core import NeuralODE

    for path in args.ckpt:
        ck = torch.load(path, map_location=device, weights_only=False)
        model, is_unet, name, keys = resolve_checkpoint(ck, device)
        key = keys.get(args.weights) or keys.get("raw")
        model.load_state_dict(ck[key if key in ck else keys["raw"]])
        model.eval()
        step = int(ck.get("step", 0))
        vf = _VelocityWrapper(model, is_unet)
        tag = f"{name}_step{step}"
        out_dir = args.out_dir or os.path.dirname(os.path.abspath(path))
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, f"sampler_diag_{tag}")
        print(f"\n{name} (step {step:,}, poids {args.weights})", flush=True)

        # bruit initial COMMUN a toutes les variantes : les differences observees
        # ne viennent que du solveur / du point de depart, jamais du tirage.
        g = torch.Generator(device="cpu").manual_seed(args.seed + 7)
        x0 = torch.randn(args.n, C, S, S, generator=g).to(device)

        # ---- A. le solveur est-il en cause ? ----
        rows, labels = [], []
        for ns in steps_list:
            rows.append(integrate(vf, x0.clone(), 0.0, ns, device).cpu())
            labels.append(f"Euler {ns}")
            print(f"  Euler {ns} pas", flush=True)
        with torch.no_grad():
            node = NeuralODE(vf, solver="dopri5", atol=1e-5, rtol=1e-5)
            traj = node.trajectory(x0.clone(),
                                   t_span=torch.linspace(0, 1, 2, device=device))
        rows.append(traj[-1].cpu()); labels.append("dopri5 1e-5")
        print("  dopri5 (celui des step_N.png)", flush=True)
        _grid(rows, labels, f"{base}_solver.png",
              f"{name} step {step:,} — A. raffinement du solveur (meme bruit initial)\n"
              "si les lignes se ressemblent, l'integration est convergee",
              C, S)

        # ---- B. ou la trajectoire derape-t-elle ? ----
        rows, labels = [], []
        gv = torch.Generator(device="cpu").manual_seed(args.seed + 1234)
        noise = torch.randn(args.n, C, S, S, generator=gv).to(device)
        for t0 in t0_list:
            xt0 = (1 - t0) * noise + t0 * x1               # vrai point de la trajectoire
            rows.append(integrate(vf, xt0.clone(), t0, max(args.start_steps, 1),
                                  device).cpu())
            labels.append(f"depart t={t0:g}")
            print(f"  depart retarde t0={t0:g}", flush=True)
        rows.append(x1.cpu()); labels.append("$x_1$ (cible)")
        _grid(rows, labels, f"{base}_start.png",
              f"{name} step {step:,} — B. depart retarde depuis un VRAI x_t0\n"
              "la ligne ou ca devient net borne l'intervalle de temps fautif",
              C, S)

        del model, vf; gc.collect(); torch.cuda.empty_cache()
        print(f"  -> {base}_solver.png\n  -> {base}_start.png", flush=True)


if __name__ == "__main__":
    main()
