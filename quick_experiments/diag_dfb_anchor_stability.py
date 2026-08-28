# -*- coding: utf-8 -*-
"""
diag_dfb_anchor_stability.py
Pourquoi le deroule ConvDFB ne "debruite" pas, alors que l'ancrage a l'entree y est ?

Dans les deux archis le primal est ancre sur z, mais pas de la meme facon :
  DFB   x^(k)   = z - W^T u^(k-1)                (ancrage RIGIDE, primal esclave du dual)
  ScCP  x^(k+1) = (x^(k) - tau W^T u + tau z)/(1+tau)   (ancrage relache + contraction 1/(1+tau))

Toute la dynamique du DFB vit donc dans le dual. En substituant x_next dans le pas dual :

    u^(k+1) = clamp( (I - tau A A^T) u^(k) + tau A z ,  +-r(t) )      avec A = conv2d(., W)

non-expansive SSI  tau * ||A||^2 < 2. Or le code choisit  tau = 1.99 / sigma^2  avec sigma la
norme spectrale de W REMIS A PLAT en matrice (C, k*k) (sigma_max_power_iter), qui n'est PAS la
norme d'operateur de la convolution : pour un noyau passe-bas, ||A|| = sup_xi ||W_hat(xi)||
peut valoir plusieurs fois sigma_proxy. Si c'est le cas, le pas depasse le seuil de stabilite,
u diverge et vient saturer le clamp du prox l1 -> plateau plat + oscillations de signe.

Ce script mesure, couche par couche, sur les poids ENTRAINES :
  sigma_proxy   norme utilisee par le code pour fixer le pas
  ||A||         vraie norme d'operateur de la convolution (iteration de puissance sur
                conv2d / conv_transpose2d, qui sont exactement adjoints a stride 1)
  tau ||A||^2   produit de stabilite du forward-backward dual   (seuil : 2)
  r(t)          rayon appris du clamp l1
  saturation    fraction des coordonnees de u^(k) collees a +-r(t), sur de VRAIS x_t
  ||u^(k)||     amplitude du dual le long du deroule

et, pour le ScCP en regard, le facteur de contraction primal 1/(1+tau_k) et la condition
de Chambolle-Pock sigma*tau*||A||^2 <= 1.

Sorties (dans --outdir) : dfb_anchor_stability.png, dfb_anchor_stability.txt

Usage
-----
    source ~/.venvs/unn/bin/activate
    CUDA_VISIBLE_DEVICES=1 python diag_dfb_anchor_stability.py
    python diag_dfb_anchor_stability.py --dfb <ckpt> --sccp <ckpt> --outdir <dir>
"""
import argparse
import os

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.architectures import ConvDFB_UNN, ConvScCP_UNN, sigma_max_power_iter
from trajectory_convsccp import build_model


@torch.no_grad()
def conv_operator_norm(W, pad, img_size=28, n_iter=200, tol=1e-7, device="cpu"):
    """Vraie norme d'operateur de x -> conv2d(x, W, padding=pad) sur des images
    img_size x img_size, par iteration de puissance sur A^T A. conv_transpose2d est
    l'adjoint exact de conv2d a stride 1, donc l'iteration est licite.

    A comparer a sigma_max(W.view(C, -1)), qui est la norme du seul motif de patch et
    ignore completement la structure de convolution (donc le gain en frequence)."""
    C, in_ch = W.shape[0], W.shape[1]
    v = torch.randn(1, in_ch, img_size, img_size, device=device)
    v /= v.norm()
    lam_prev = 0.0
    for _ in range(n_iter):
        Av = F.conv2d(v, W, padding=pad)
        v_new = F.conv_transpose2d(Av, W, padding=pad)
        lam = v_new.norm()
        if lam == 0:
            return 0.0
        v = v_new / lam
        if abs(lam.item() - lam_prev) < tol * max(1.0, lam.item()):
            lam_prev = lam.item()
            break
        lam_prev = lam.item()
    return lam_prev ** 0.5                      # ||A|| = sqrt(||A^T A||)


@torch.no_grad()
def dfb_unroll_trace(model, xt_t):
    """Rejoue le deroule DFB a la main pour capturer le dual (le forward ne le rend pas).

    Renvoie, par couche : ||u^(k)|| (moyenne batch), la fraction de coordonnees saturees
    contre le clamp, le rayon r_k(t) et ||x^(k)||_inf.
    """
    B = xt_t.shape[0]
    z = xt_t[:, :model.dim].contiguous().view(B, 1, 28, 28)
    t = xt_t[:, model.dim:]
    u = torch.zeros(B, model.internal_channel, 28, 28, device=z.device)
    nu, sat, radii, amp = [], [], [], []
    for layer in model.layers:
        V = layer.V_weight if layer.version == "LFO" else layer.W_weight
        x_next = z - F.conv_transpose2d(u, V, padding=4)
        if layer.version == "LFO":
            tau = F.softplus(layer.tau)
        else:
            s, _ = sigma_max_power_iter(layer.W_weight, layer._sigma_u)
            tau = 1.99 / s ** 2
        step = tau * F.conv2d(x_next, layer.W_weight, bias=layer.W_bias, padding=4)
        u_in = u + step
        u = layer.prox(u_in, t)
        r = F.softplus(layer.prox.time_scaling(t)).view(B, -1).mean().item()
        nu.append(u.flatten(1).norm(dim=1).mean().item())
        sat.append((u_in.abs() >= r * 0.999).float().mean().item())
        radii.append(r)
        amp.append(x_next.abs().amax().item())
    return nu, sat, radii, amp


def main():
    p = argparse.ArgumentParser(description="Stabilite du deroule dual DFB vs ancrage ScCP.")
    p.add_argument("--dfb", type=str,
                   default="results/convdfb_zeros/ConvDFB_K20_ic64_L1_LNO/model.pt")
    p.add_argument("--sccp", type=str,
                   default="results/temp-4/ConvScCP_UNN_L1_LNO/model.pt",
                   help="ScCP en regard ('' pour s'en passer)")
    p.add_argument("--traj", type=str,
                   default="results/convdfb_zeros/ConvDFB_K20_ic64_L1_LNO/trajectory/trajectory.pt",
                   help="trajectoire deja calculee, pour prendre de VRAIS x_t ('' -> bruit)")
    p.add_argument("--t", type=float, default=0.5, help="temps auquel sonder le deroule")
    p.add_argument("--outdir", type=str, default="results/convdfb_zeros")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    # ------------------------------------------------------------------ DFB
    dfb, cfg = build_model(args.dfb, device)
    assert isinstance(dfb, ConvDFB_UNN), f"{args.dfb} n'est pas un ConvDFB_UNN"
    log(f"# ConvDFB  {args.dfb}")
    log(f"# K={cfg['K']} ic={cfg['internal_channel']} {cfg['version']} kernel={cfg['kernel']}")
    log("#")
    log("# couche | sigma_proxy | ||A|| (vraie) | ratio | tau=1.99/sigma^2 | tau*||A||^2 "
        "(seuil 2)")
    prox_norms, true_norms, taus, stab = [], [], [], []
    for k, layer in enumerate(dfb.layers):
        W = layer.W_weight.detach()
        s_proxy = float(torch.linalg.matrix_norm(W.view(W.shape[0], -1), ord=2))
        L = conv_operator_norm(W, pad=4, device=device)
        tau = 1.99 / s_proxy ** 2
        prox_norms.append(s_proxy); true_norms.append(L); taus.append(tau)
        stab.append(tau * L ** 2)
        log(f"  k={k:<3d}  {s_proxy:10.4f}  {L:12.4f}  {L/s_proxy:6.2f}  {tau:14.4f}  "
            f"{tau * L**2:10.2f}")
    n_bad = sum(1 for v in stab if v >= 2.0)
    log(f"#\n# couches au-dessus du seuil de stabilite tau*||A||^2 >= 2 : {n_bad}/{len(stab)}")
    log(f"# tau*||A||^2 : min={min(stab):.2f} median={sorted(stab)[len(stab)//2]:.2f} "
        f"max={max(stab):.2f}")

    # ---- deroule sur de vrais x_t : le dual sature-t-il le clamp ? ----
    if args.traj and os.path.exists(args.traj):
        tr = torch.load(args.traj, map_location="cpu")
        j = int(torch.argmin((tr["ts"] - args.t).abs()))
        x = tr["xt"][j].flatten(1).to(device)
        tval = float(tr["ts"][j])
        src = f"x_t reels de {args.traj} a t={tval:.2f}"
    else:
        x = torch.randn(6, 784, device=device)
        tval = args.t
        src = f"bruit gaussien a t={tval:.2f}"
    xt_t = torch.cat([x, torch.full((x.shape[0], 1), tval, device=device)], dim=-1)
    nu, sat, radii, amp = dfb_unroll_trace(dfb, xt_t)
    log(f"#\n# deroule sur {src}")
    log("# couche | ||u^(k)|| | saturation du clamp | r_k(t) | max|x^(k)|")
    for k in range(len(nu)):
        log(f"  k={k:<3d}  {nu[k]:10.2f}  {sat[k]*100:17.1f}%  {radii[k]:8.3f}  {amp[k]:10.1f}")

    # ------------------------------------------------------------------ ScCP
    sccp_contract, sccp_cp = None, None
    if args.sccp and os.path.exists(args.sccp):
        sccp, cfg2 = build_model(args.sccp, device)
        assert isinstance(sccp, ConvScCP_UNN)
        log(f"#\n# ConvScCP en regard : {args.sccp}  (K={cfg2['K']} ic={cfg2['internal_channel']} "
            f"{cfg2['version']})")
        log("# couche | tau_k | contraction primale 1/(1+tau_k) | sigma_k*tau_k*||A||^2 (CP: <=1)")
        taus_s = F.softplus(sccp.log_tau.detach()) if cfg2["version"] == "LNO" else None
        sccp_contract, sccp_cp = [], []
        for k, layer in enumerate(sccp.layers):
            W = layer.W_weight.detach()
            s_proxy = float(torch.linalg.matrix_norm(W.view(W.shape[0], -1), ord=2))
            L = conv_operator_norm(W, pad=layer.pad, device=device)
            tau_k = float(taus_s[k]) if taus_s is not None else float(F.softplus(sccp.log_tau0))
            sigma_k = 0.99 / (tau_k * s_proxy ** 2)
            sccp_contract.append(1.0 / (1.0 + tau_k))
            sccp_cp.append(sigma_k * tau_k * L ** 2)
            log(f"  k={k:<3d}  {tau_k:8.4f}  {1/(1+tau_k):24.4f}  {sigma_k*tau_k*L**2:14.2f}")
        cum = 1.0
        for c in sccp_contract:
            cum *= c
        log(f"#\n# contraction primale CUMULEE du ScCP sur les {len(sccp_contract)} couches : "
            f"{cum:.3e}")

    # ------------------------------------------------------------------ figure
    ncol = 3 if sccp_contract else 2
    fig, axs = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.2))
    ks = range(len(stab))
    axs[0].plot(ks, prox_norms, "o-", label=r"$\sigma_{proxy}$ (pas utilise par le code)")
    axs[0].plot(ks, true_norms, "s-", label=r"$\|A\|$ (vraie norme de la conv)")
    axs[0].set_xlabel("couche k"); axs[0].set_ylabel("norme"); axs[0].set_yscale("log")
    axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)
    axs[0].set_title("DFB : la norme qui fixe le pas\nsous-estime l'operateur", fontsize=10)

    axs[1].plot(ks, stab, "o-", color="crimson")
    axs[1].axhline(2.0, color="k", ls="--", lw=1, label="seuil de stabilite FB")
    axs[1].set_xlabel("couche k"); axs[1].set_ylabel(r"$\tau\,\|A\|^2$")
    axs[1].set_yscale("log"); axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)
    axs[1].set_title("DFB : pas dual vs seuil\n(au-dessus = iteration expansive)", fontsize=10)

    ax2 = axs[1].twinx()
    ax2.plot(ks, [s * 100 for s in sat], "^:", color="steelblue", alpha=0.7)
    ax2.set_ylabel("saturation du clamp (%)", color="steelblue", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="steelblue")

    if sccp_contract:
        axs[2].plot(range(len(sccp_contract)), sccp_contract, "o-", label=r"ScCP : $1/(1+\tau_k)$")
        axs[2].axhline(1.0, color="k", ls="--", lw=1, label="pas de contraction (DFB)")
        axs[2].set_xlabel("couche k"); axs[2].set_ylabel("facteur de contraction primal")
        axs[2].set_ylim(0, 1.1); axs[2].legend(fontsize=8); axs[2].grid(alpha=0.3)
        axs[2].set_title("ScCP : contraction structurelle\na chaque couche", fontsize=10)

    fig.suptitle("Pourquoi le deroule DFB ne debruite pas malgre l'ancrage a l'entree", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    png = os.path.join(args.outdir, "dfb_anchor_stability.png")
    plt.savefig(png, dpi=130)
    plt.close(fig)

    txt = os.path.join(args.outdir, "dfb_anchor_stability.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  -> {png}\n  -> {txt}", flush=True)


if __name__ == "__main__":
    main()
