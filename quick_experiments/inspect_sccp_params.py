# -*- coding: utf-8 -*-
"""
inspect_sccp_params.py
Visualise les pas APPRIS du deroule ConvScCP, couche par couche : tau_k, sigma_k,
alpha_k, ||W_k||, et le produit tau_k*sigma_k*||W_k||^2 (condition de Chambolle-Pock).

Les formules reproduisent EXACTEMENT ConvScCP_UNN.forward() :

  LNO : tau_k   = softplus(log_tau[k])            (K pas primaux appris, un par couche)
        alpha_k = (1 + 2*tau_k)^(-1/2)
        ||W_k|| = spectral_norm() (power iteration sur W_k)
        sigma_k = 0.99 / (tau_k * ||W_k||^2)      -> produit CP = 0.99 par construction

  LFO : tau_0   = softplus(log_tau0)              (UN scalaire appris)
        tau_{k+1} = alpha_k * tau_k               (recursion geometrique)
        sigma_k = 1.0                             -> produit CP = tau_k * ||W_k||^2, LIBRE
                                                     (rien ne garantit <= 1)

Condition de convergence de Chambolle-Pock : tau*sigma*||W||^2 <= 1. En LNO elle est
imposee a 0.99 ; en LFO elle n'est pas contrainte, d'ou l'interet du graphe.

Sorties (dans <dir du ckpt>/params/) : sccp_params.png , sccp_params.txt

Usage
-----
    python inspect_sccp_params.py --ckpt results/temp-4/ConvScCP_UNN_L1_LNO/model.pt
"""
import argparse
import os

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trajectory_convsccp import build_model


@torch.no_grad()
def extract(model):
    """Renvoie un dict de listes (une valeur par couche k) : tau, sigma, alpha, sn, prod."""
    K = model.K
    # power iteration : plusieurs passes pour converger avant de LIRE la valeur
    # (spectral_norm() raffine le buffer _sigma_u a chaque appel).
    def _norm(W):                               # meme convention que sigma_max_power_iter :
        return float(torch.linalg.matrix_norm(W.reshape(W.shape[0], -1), ord=2))

    sns, vns = [], []
    for layer in model.layers:
        if model.version == "LNO":
            for _ in range(50):
                sn = layer.spectral_norm()
            sns.append(float(sn))
            vns.append(float(sn))               # LNO : V = W -> ||V|| = ||W|| (transposee)
        else:                                   # LFO : V appris, W sert quand meme d'operateur
            sns.append(_norm(layer.W_weight))
            vns.append(_norm(layer.V_weight))   # V est LIBRE : ||V|| != ||W|| en general

    tau, sigma, alpha = [], [], []
    if model.version == "LNO":
        taus = F.softplus(model.log_tau)
        for k in range(K):
            t_k = float(taus[k])
            tau.append(t_k)
            alpha.append(float((1.0 + 2.0 * t_k) ** -0.5))
            sigma.append(0.99 / (t_k * sns[k] ** 2))
    else:
        t_k = float(F.softplus(model.log_tau0))
        for k in range(K):
            a_k = float((1.0 + 2.0 * t_k) ** -0.5)
            tau.append(t_k); alpha.append(a_k); sigma.append(1.0)
            t_k = a_k * t_k                     # recursion, identique au forward
    prod = [tau[k] * sigma[k] * sns[k] ** 2 for k in range(K)]
    return dict(tau=tau, sigma=sigma, alpha=alpha, sn=sns, vn=vns, prod=prod)


def plot(p, cfg, path, title):
    ks = range(len(p["tau"]))
    fig, axs = plt.subplots(2, 4, figsize=(19, 7.5))
    panels = [
        ("tau", r"$\mu_k$  (pas primal)", True),
        ("sigma", r"$\tau_k$  (pas dual)", True),
        ("alpha", r"$\alpha_k$  (momentum)", False),
        ("sn", r"$\|W_k\|$  (norme spectrale)", False),
        ("vn", r"$\|V_k\|$  (adjoint ; $=\|W_k\|$ en LNO)", False),
        ("prod", r"$\mu_k\,\tau_k\,\|W_k\|^2$  (condition CP)", True),
    ]
    for ax, (key, lab, logy) in zip(axs.flat, panels):
        ax.plot(ks, p[key], "o-", ms=4)
        ax.set_xlabel("couche k"); ax.set_title(lab, fontsize=11)
        ax.grid(alpha=0.3)
        if logy and min(p[key]) > 0:
            ax.set_yscale("log")
        if key == "prod":                       # seuil de convergence CP
            ax.axhline(1.0, color="r", ls="--", lw=1.2, label="limite CP = 1")
            ax.legend(fontsize=8)
    for ax in axs.flat[len(panels):]:           # cases restantes (dont l'encart texte)
        ax.axis("off")
    axs.flat[-1].text(0.0, 0.5, "\n".join([
        f"version   : {cfg['version']}",
        f"K         : {cfg['K']}",
        f"ic        : {cfg['internal_channel']}",
        f"kernel    : {cfg['kernel']}",
        f"prox_w    : {cfg['prox_w']}",
        "",
        f"mu   : {min(p['tau']):.4g} -> {max(p['tau']):.4g}",
        f"tau : {min(p['sigma']):.4g} -> {max(p['sigma']):.4g}",
        f"prod  : {min(p['prod']):.4g} -> {max(p['prod']):.4g}",
    ]), fontsize=10, family="monospace", va="center")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser(description="Visualise tau_k / sigma_k appris d'un ConvScCP.")
    ap.add_argument("--ckpt", type=str, default="results/temp-4/ConvScCP_UNN_L1_LNO/model.pt")
    ap.add_argument("--outdir", type=str, default="", help="defaut : <dir du ckpt>/params/")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = build_model(args.ckpt, device)
    print(f"Config : {cfg}")

    p = extract(model)
    outdir = args.outdir or os.path.join(os.path.dirname(args.ckpt), "params")
    os.makedirs(outdir, exist_ok=True)
    name = os.path.basename(os.path.dirname(args.ckpt))
    plot(p, cfg, os.path.join(outdir, "sccp_params.png"),
         f"pas appris du deroule ScCP — {name} — {cfg['version']} K={cfg['K']} ic={cfg['internal_channel']}")

    txt = os.path.join(outdir, "sccp_params.txt")
    with open(txt, "w") as f:
        f.write(f"# {args.ckpt}  version={cfg['version']} K={cfg['K']} ic={cfg['internal_channel']}\n")
        f.write("k\tmu\ttau\talpha\tsn_W\tsn_V\tprod_CP\n")
        for k in range(len(p["tau"])):
            f.write(f"{k}\t{p['tau'][k]:.6g}\t{p['sigma'][k]:.6g}\t{p['alpha'][k]:.6g}\t"
                    f"{p['sn'][k]:.6g}\t{p['vn'][k]:.6g}\t{p['prod'][k]:.6g}\n")
    print(f"  -> {txt}")

    print("\n k |    mu_k |   tau_k |   alpha |   ||W|| |   ||V|| |  prod CP")
    for k in range(len(p["tau"])):
        print("%2d | %7.4g | %7.4g | %7.4g | %7.4g | %7.4g | %8.4g"
              % (k, p["tau"][k], p["sigma"][k], p["alpha"][k], p["sn"][k], p["vn"][k], p["prod"][k]))


if __name__ == "__main__":
    main()
