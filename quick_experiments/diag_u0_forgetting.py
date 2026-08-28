# -*- coding: utf-8 -*-
"""
diag_u0_forgetting.py
Pourquoi le transfert du dual (sample_sccp_utransfer.py) ne change RIEN a K=20 ?

Question posee : le modele a ete entraine avec u^(0)=0 partout, on s'attendrait a ce
qu'un u^(0) != 0 le casse. Il ne se passe rien du tout (delta_v ~ 1e-3, images
identiques au pixel pres). Ce script mesure la cause, en trois diagnostics :

  (A) OUBLI  : divergence relative entre le deroule FROID (u^(0)=0) et le deroule CHAUD
      (u^(0)=c*u_prev), iteration par iteration. Si elle DECROIT avec k, le deroule
      contracte : l'initialisation duale est oubliee, et le modele est de fait
      insensible a u^(0). C'est la propriete de convergence de Chambolle-Pock, mesuree.

  (B) REPONSE : delta_v = ||v(u^(0)=c*u_dir) - v(u^(0)=0)|| / ||v(u^(0)=0)|| en fonction
      de l'amplitude c, balayee sur plusieurs decades, pour deux directions :
        - "warm" : le vrai u_prev du pas d'ODE precedent (la direction utile) ;
        - "rand" : un bruit gaussien de meme norme (direction quelconque).
      Une reponse LINEAIRE en c = regime de perturbation infinitesimale (rien de
      non-lineaire n'est declenche). Le c auquel la courbe sature/decroche dit combien
      il faut pousser pour reellement casser le modele.

  (C) SATURATION : fraction des coordonnees duales collees a la borne du prox l1
      (clamp(u, -r(t), r(t))). Sur ces coordonnees la sortie du prox est CONSTANTE en
      son entree : l'information de u^(0) y est detruite en une iteration, derivee nulle.
      C'est le second mecanisme d'oubli, en plus de la contraction.

Usage
-----
    source ~/.venvs/unn/bin/activate
    python diag_u0_forgetting.py \
        --ckpt results/temp-13-200-epochs-ot/ConvScCP_UNN_L1_LFO/model.pt

Sorties (--outdir, defaut results/sccp_utransfer/) :
    u0_forgetting.png   les trois panneaux
    u0_forgetting.txt   les valeurs chiffrees
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.architectures import ConvScCP_UNN
from generate_digits import infer_config
from sample_sccp_utransfer import build_model


@torch.no_grad()
def unroll(model, z, t, u0, keep_all=True):
    """Re-deroule EXACTEMENT ConvScCP_UNN.forward (branche LNO ou LFO), en gardant
    tous les iteres. Retourne xs [(K+1) tenseurs], us [(K+1)], sat [K] (fraction de
    coordonnees duales collees a la borne du prox, par iteration).

    Duplique la boucle du modele au lieu de l'instrumenter : le diagnostic reste un
    outil d'analyse externe, architectures.py n'a pas a porter de code de mesure.
    """
    x, u = z.clone(), u0.clone()
    xs, us, sat = [x.clone()], [u.clone()], []
    if model.version == "LNO":
        taus = F.softplus(model.log_tau)
    else:
        tau_k = F.softplus(model.log_tau0)
    for k, layer in enumerate(model.layers):
        if model.version == "LNO":
            tau_k = taus[k]
            sigma_k = 0.99 / (tau_k * layer.spectral_norm() ** 2)
        else:
            sigma_k = torch.tensor(1.0, device=z.device)
        alpha_k = (1.0 + 2.0 * tau_k).pow(-0.5)
        x, u = layer(x, u, z, t, tau_k, sigma_k, alpha_k)
        if model.version == "LFO":
            tau_k = alpha_k * tau_k
        # fraction saturee : |u| a la borne r(t) du prox l1 (clamp actif)
        r = F.softplus(layer.prox.time_scaling(t))
        r = r.view(r.shape[0], *([1] * (u.dim() - 1)))
        sat.append((u.abs() >= r * (1 - 1e-5)).float().mean().item())
        if keep_all:
            xs.append(x.clone()); us.append(u.clone())
        else:                          # ic=512/1024 : garder les K+1 duals sature la VRAM
            xs, us = [x], [u]
    return xs, us, sat


@torch.no_grad()
def dual_step_ratio(model, z, t, u0):
    """||u^(0)|| / ||sigma_1 * W y^(1)|| a la premiere iteration.

    C'est LE discriminant entre "u^(0) est oublie" et "u^(0) commande tout". Le prox l1
    est un clamp : quand il sature, u^(1) = r(t)*signe(u^(0) + sigma*W y). Seul le SIGNE
    de la somme survit. Si le pas dual domine (ratio << 1), le signe est celui du pas et
    u^(0) est efface ; s'il est domine (ratio >~ 1), le signe est celui de u^(0), qui
    dicte alors tout le deroule.
    """
    layer = model.layers[0]
    tau = F.softplus(model.log_tau)[0] if model.version == "LNO" else F.softplus(model.log_tau0)
    sigma = (0.99 / (tau * layer.spectral_norm() ** 2) if model.version == "LNO"
             else torch.tensor(1.0, device=z.device))
    alpha = (1.0 + 2.0 * tau).pow(-0.5)
    V = layer.V_weight if model.version == "LFO" else layer.W_weight
    x1 = (z - tau * F.conv_transpose2d(u0, V, padding=layer.pad) + tau * z) / (1 + tau)
    y = x1 + alpha * (x1 - z)
    step = sigma * F.conv2d(y, layer.W_weight, bias=layer.W_bias, padding=layer.pad)
    r = F.softplus(layer.prox.time_scaling(t)).view(t.shape[0], *([1] * (u0.dim() - 1)))
    ratio = (u0.flatten(1).norm(dim=1) / step.flatten(1).norm(dim=1).clamp(min=1e-12)).mean().item()
    return ratio, (u0.abs() / r.expand_as(u0)).mean().item()


@torch.no_grad()
def velocity(model, z, t, u0):
    """v = (x^(K) - z) / clamp(1-t, 0.05), la sortie eval du modele."""
    xs, _, _ = unroll(model, z, t, u0, keep_all=False)
    return (xs[-1] - z) / torch.clamp(1 - t, min=0.05).view(-1, 1, 1, 1)


@torch.no_grad()
def collect_states(model, cfg, n, device, steps, t_targets, seed):
    """Fait tourner l'echantillonneur FROID (celui du modele) et preleve, aux t demandes,
    l'etat x_t reel de la trajectoire ET le u^(K) du pas PRECEDENT = le u_prev qu'on
    voudrait transferer. On travaille donc sur les vrais points visites, pas sur des
    x_t synthetiques."""
    torch.manual_seed(seed)
    dim, C, S = cfg["dim"], cfg["in_channels"], cfg["img_size"]
    x = torch.randn(n, dim, device=device)
    u_prev = torch.zeros(n, model.internal_channel, S, S, device=device)
    picks = {}
    idx_targets = {int(round(tt * steps)): tt for tt in t_targets}
    for i in range(steps):
        t_val = i / steps
        t_col = torch.full((n, 1), t_val, device=device)
        z = x.view(n, C, S, S)
        if i in idx_targets:
            picks[idx_targets[i]] = (z.clone(), t_col.clone(), u_prev.clone())
        v, u_K = model(torch.cat([x, t_col], dim=-1),
                       u_init=u_prev, return_u=True)      # trajectoire froide : u_prev
        x = x + v / steps                                  # non utilise comme init (voir ci-dessous)
        u_prev = u_K
    return picks


def main():
    p = argparse.ArgumentParser(description="Pourquoi u^(0) n'a aucun effet a K=20.")
    p.add_argument("--ckpt", type=str,
                   default="results/temp-13-200-epochs-ot/ConvScCP_UNN_L1_LFO/model.pt")
    p.add_argument("--n", type=int, default=64, help="Taille du lot de diagnostic.")
    p.add_argument("--steps", type=int, default=100, help="Pas d'Euler (meme grille que l'expe).")
    p.add_argument("--t-list", type=str, default="0.1,0.5,0.9")
    p.add_argument("--scales", type=str, default="0.1,0.25,0.5,1,2,5,10,50,100,1000",
                   help="Amplitudes c de u^(0) balayees (panneau B).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="results/sccp_utransfer")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)
    t_list = [float(s) for s in args.t_list.split(",")]
    scales = [float(s) for s in args.scales.split(",")]

    model, cfg = build_model(args.ckpt, device)
    print(f"[u0-forgetting] {args.ckpt}\n  K={cfg['K']} ic={cfg['internal_channel']} "
          f"kernel={cfg['kernel']} {cfg['version']} | device={device}", flush=True)

    picks = collect_states(model, cfg, args.n, device, args.steps, t_list, args.seed)
    lines = [f"# {args.ckpt}  K={cfg['K']} ic={cfg['internal_channel']} {cfg['version']}  "
             f"n={args.n} steps={args.steps}"]

    # ---------------- (A) oubli de u^(0) le long du deroule ----------------
    forget = {}
    for t_val, (z, t_col, u_prev) in picks.items():
        xs_c, us_c, sat = unroll(model, z, t_col, torch.zeros_like(u_prev))
        xs_w, us_w, _ = unroll(model, z, t_col, u_prev)
        # divergence relative primale et duale, iteration par iteration
        dx = [((a - b).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1).clamp(min=1e-12)).mean().item()
              for a, b in zip(xs_w, xs_c)]
        du = [((a - b).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1).clamp(min=1e-12)).mean().item()
              for a, b in zip(us_w[1:], us_c[1:])]
        ratio, u_over_r = dual_step_ratio(model, z, t_col, u_prev)
        forget[t_val] = dict(dx=dx, du=du, sat=sat, ratio=ratio, u_over_r=u_over_r)
        lines.append(f"\n[D] t={t_val}  ||u^(0)|| / ||pas dual sigma*W y|| = {ratio:.3f}   "
                     f"(|u^(0)|/r(t) moyen = {u_over_r:.3f})")
        lines.append("     ratio << 1 -> le pas dual impose le signe, u^(0) est efface ; "
                     "ratio >~ 1 -> u^(0) impose le signe et commande le deroule.")
        lines.append(f"\n[A] t={t_val}  divergence primale ||x_chaud-x_froid||/||x_froid|| par iteration :")
        lines.append("     " + "  ".join(f"k={k}:{v:.2e}" for k, v in enumerate(dx)))
        lines.append(f"[A] t={t_val}  divergence duale par iteration :")
        lines.append("     " + "  ".join(f"k={k+1}:{v:.2e}" for k, v in enumerate(du)))
        lines.append(f"[C] t={t_val}  fraction duale saturee (clamp actif) par iteration :")
        lines.append("     " + "  ".join(f"k={k+1}:{v:.3f}" for k, v in enumerate(sat)))

    # ---------------- (B) reponse en fonction de l'amplitude de u^(0) ----------------
    resp = {}
    for t_val, (z, t_col, u_prev) in picks.items():
        v0 = velocity(model, z, t_col, torch.zeros_like(u_prev))
        n0 = v0.flatten(1).norm(dim=1).clamp(min=1e-12)
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        rand_dir = torch.randn(u_prev.shape, generator=g).to(device)
        rand_dir = rand_dir / rand_dir.flatten(1).norm(dim=1).view(-1, 1, 1, 1) \
            * u_prev.flatten(1).norm(dim=1).view(-1, 1, 1, 1)      # meme norme que u_prev
        curves = {}
        for name, direction in (("warm (u_prev)", u_prev), ("rand (meme norme)", rand_dir)):
            ys = []
            for c in scales:
                v = velocity(model, z, t_col, c * direction)
                ys.append(((v - v0).flatten(1).norm(dim=1) / n0).mean().item())
            curves[name] = ys
        resp[t_val] = curves
        for name, ys in curves.items():
            lines.append(f"\n[B] t={t_val}  delta_v vs amplitude c, direction {name} :")
            lines.append("     " + "  ".join(f"c={c:g}:{y:.2e}" for c, y in zip(scales, ys)))

    # ---------------------------------- figure ----------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for t_val in t_list:
        d = forget[t_val]
        axs[0].semilogy(range(len(d["dx"])), np.maximum(d["dx"], 1e-16), "o-",
                        ms=3, label=f"primal x, t={t_val}")
        axs[0].semilogy(range(1, len(d["du"]) + 1), np.maximum(d["du"], 1e-16), "s--",
                        ms=3, alpha=0.6, label=f"dual u, t={t_val}")
    axs[0].set_xlabel("iteration k du deroule CP"); axs[0].set_ylabel("divergence relative chaud/froid")
    axs[0].set_title("(A) oubli de u$^{(0)}$ : le dual se resynchronise\n"
                     "en ~2 iterations, le primal garde un offset constant", fontsize=9)
    axs[0].grid(alpha=0.3, which="both"); axs[0].legend(fontsize=7)

    for t_val in t_list:
        for name, ys in resp[t_val].items():
            axs[1].loglog(scales, np.maximum(ys, 1e-16), "o-", ms=3, label=f"{name}, t={t_val}")
    axs[1].loglog(scales, [s * resp[t_list[0]]["warm (u_prev)"][0] / scales[0] for s in scales],
                  "k:", lw=1, label="pente 1 (lineaire)")
    axs[1].set_xlabel("amplitude c de u$^{(0)}$"); axs[1].set_ylabel(r"$\delta_v$ relatif")
    axs[1].set_title("(B) reponse du champ a l'amplitude de u$^{(0)}$", fontsize=10)
    axs[1].grid(alpha=0.3, which="both"); axs[1].legend(fontsize=7)

    for t_val in t_list:
        axs[2].plot(range(1, len(forget[t_val]["sat"]) + 1), forget[t_val]["sat"],
                    "o-", ms=3, label=f"t={t_val}")
    axs[2].set_xlabel("iteration k"); axs[2].set_ylabel("fraction de coordonnees a la borne")
    axs[2].set_ylim(0, 1.02)
    axs[2].set_title("(C) saturation du prox l1 : entree effacee", fontsize=10)
    axs[2].grid(alpha=0.3); axs[2].legend(fontsize=8)

    fig.suptitle(f"Insensibilite a u$^{{(0)}}$ — ConvScCP K={cfg['K']} ic={cfg['internal_channel']} "
                 f"{cfg['version']} (ckpt du rapport)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_png = os.path.join(args.outdir, "u0_forgetting.png")
    plt.savefig(out_png, dpi=130)
    plt.close(fig)

    out_txt = os.path.join(args.outdir, "u0_forgetting.txt")
    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\n[u0-forgetting] -> {out_png}\n[u0-forgetting] -> {out_txt}", flush=True)


if __name__ == "__main__":
    main()
