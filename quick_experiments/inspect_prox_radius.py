# -*- coding: utf-8 -*-
"""
inspect_prox_radius.py
Trace les rayons appris r_k(t) du prox l1, couche par couche.

Pourquoi ca compte
------------------
Dans ConvScCP_UNN(use_Unet="l1"), le temps n'entre NULLE PART ailleurs : W, V,
tau, sigma, alpha sont tous independants de t. Toute la dependance temporelle du
champ de vitesses passe par les K scalaires

    r_k(t) = softplus(MLP_k(t)),     prox_k(u) = clamp(u, -r_k(t), +r_k(t))

Avec K=10, le conditionnement en temps du reseau entier, c'est DIX NOMBRES.

Ce graphe repond a la question "faut-il un encodage sinusoidal du temps ?" :

  * r_k(t) lisses et lentement variables -> un MLP scalaire les represente sans
    peine, un encodage sinusoidal (qui sert a vaincre le biais spectral, donc a
    representer des dependances a HAUTE frequence) n'apporterait rien. Le
    probleme serait alors la LARGEUR du canal (1 scalaire/couche), pas la
    parametrisation de t.
  * r_k(t) qui saturent, se coupent, ou tentent des variations abruptes ->
    la parametrisation bride le modele, et l'encodage vaut le detour.

On trace aussi dr/dt normalise : c'est lui qui revele une eventuelle demande de
haute frequence, invisible a l'oeil sur r(t).

Usage
-----
    source ~/.venvs/unn/bin/activate
    python inspect_prox_radius.py --ckpt \\
        results_afhq32/ConvScCP_UNN_rgb_k9_K10_ic128_L1_LFO/latest.pt

Sortie -> <dir du ckpt>/prox_radius_step<N>.png + .txt
"""

import argparse
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description="Rayons r_k(t) du prox l1.")
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--weights", type=str, default="ema", choices=["ema", "raw"])
    p.add_argument("--n-t", type=int, default=401)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    from sample_checkpoint import resolve_checkpoint

    for path in args.ckpt:
        ck = torch.load(path, map_location=device, weights_only=False)
        model, is_unet, name, keys = resolve_checkpoint(ck, device)
        key = keys.get(args.weights)
        model.load_state_dict(ck[key if key in ck else keys["raw"]])
        model.eval()
        step = int(ck.get("step", 0))
        if not hasattr(model, "layers"):
            print(f"  [saute] {name} : pas un ScCP deroule", flush=True)
            continue

        t = torch.linspace(0, 1, args.n_t, device=device).view(-1, 1)
        rs, lines = [], []
        with torch.no_grad():
            for k, layer in enumerate(model.layers):
                prox = layer.prox
                if not hasattr(prox, "time_scaling"):
                    print(f"  [saute] couche {k} : prox sans conditionnement en t")
                    continue
                out = prox.time_scaling(t)
                full = torch.nn.functional.softplus(out).cpu().numpy()   # (n_t, out_dim)
                r = full.mean(axis=1) if full.shape[1] > 1 else full[:, 0]
                rs.append(r)
                amp = (r.max() - r.min()) / max(abs(r.mean()), 1e-12)
                extra = ""
                if full.shape[1] > 1:
                    # prox l1c : la moyenne sur les canaux cache l'essentiel.
                    # Ce qui compte est la DISPERSION entre canaux (sinon le
                    # per-canal ne sert a rien) et le nombre de canaux eteints.
                    spread = full.std(axis=1).mean() / max(abs(full.mean()), 1e-12)
                    dead = int((full.max(axis=0) < 1e-3).sum())
                    extra = (f" | dispersion_inter_canaux={spread:.3f}"
                             f" canaux_eteints={dead}/{full.shape[1]}")
                lines.append(f"couche {k:>2} : r(0)={r[0]:.4f} r(1)={r[-1]:.4f} "
                             f"min={r.min():.4f} max={r.max():.4f} "
                             f"amplitude_relative={amp:.3f}{extra}")
        if not rs:
            continue
        rs = np.stack(rs)
        tt = t[:, 0].cpu().numpy()
        # derivee normalisee : revele une demande de haute frequence invisible sur r(t)
        d = np.gradient(rs, tt, axis=1) / np.maximum(np.abs(rs).mean(axis=1, keepdims=True), 1e-12)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        for k in range(rs.shape[0]):
            c = plt.cm.viridis(k / max(rs.shape[0] - 1, 1))
            axes[0].plot(tt, rs[k], color=c, lw=1.5, label=f"k={k}")
            axes[1].plot(tt, d[k], color=c, lw=1.2)
        axes[0].set_xlabel("t"); axes[0].set_ylabel("rayon r(t) du clamp")
        axes[0].set_title(f"{name}\nstep {step:,} — rayons appris par couche", fontsize=9)
        axes[0].legend(fontsize=6, ncol=2)
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].set_xlabel("t"); axes[1].set_ylabel("(dr/dt) / |r|")
        axes[1].set_title("derivee normalisee\nplate = aucune demande de haute frequence",
                          fontsize=9)
        plt.tight_layout()
        out_dir = os.path.dirname(os.path.abspath(path))
        png = os.path.join(out_dir, f"prox_radius_step{step}.png")
        plt.savefig(png, dpi=110); plt.close(fig)

        amp_all = (rs.max(axis=1) - rs.min(axis=1)) / np.maximum(np.abs(rs).mean(axis=1), 1e-12)
        summary = (f"{name} step {step:,} — {rs.shape[0]} couches\n"
                   f"amplitude relative de r(t) : mediane {np.median(amp_all):.3f}, "
                   f"max {amp_all.max():.3f}\n"
                   f"|dr/dt|/|r| max sur tout t et toutes couches : {np.abs(d).max():.3f}\n\n"
                   + "\n".join(lines))
        with open(os.path.splitext(png)[0] + ".txt", "w") as f:
            f.write(summary + "\n")
        print(f"\n{summary}\n-> {png}", flush=True)


if __name__ == "__main__":
    main()
