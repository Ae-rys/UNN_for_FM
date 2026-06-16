# -*- coding: utf-8 -*-
"""
run_2moons_delta.py
Direction B: use the dual-trajectory length

    delta_t(x_t) = sum_{k=0}^{K-1} ||u^[k+1] - u^[k]||

of a DFB_UNN as an auxiliary "difficulty" indicator for (x_t, t), and check
whether it correlates with the per-example Flow Matching loss.

Experiment 1: scatter (delta_t, per-example FM loss), colored by t.
Experiment 4: profile of E[delta_t] and E[loss] vs t.

Usage
-----
    python run_2moons_delta.py [--epochs N] [--results-dir DIR]
"""

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from models import DFB_UNN, compute_delta
from run_2moons import DIM, BATCH_SIZE, sample_8gaussians, get_moons_loader, train_2moons, save_unn_paths, save_vector_field


# ---------------------------------------------------------------------------
# Delta_t evaluation
# ---------------------------------------------------------------------------

def evaluate_delta(model, train_loader, device, run_dir, n_samples=4000, n_t_bins=19):
    model.eval()
    FM = ConditionalFlowMatcher(sigma=0.01)

    x1_all = torch.cat([x for (x,) in train_loader])
    perm   = torch.randperm(x1_all.shape[0])[:n_samples]
    x1     = x1_all[perm].to(device)
    B      = x1.shape[0]
    x0     = sample_8gaussians(B, device=device)

    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
    t    = t.view(B, 1)
    xt_t = torch.cat([xt, t], dim=-1)

    with torch.no_grad():
        vt, x_traj, u_traj = model(xt_t, return_traj=True)
        delta_u          = compute_delta(u_traj)
        delta_x          = compute_delta(x_traj)
        per_example_loss = ((vt - ut) ** 2).mean(dim=-1)

    delta_u_np = delta_u.cpu().numpy()
    delta_x_np = delta_x.cpu().numpy()
    loss_np    = per_example_loss.cpu().numpy()
    t_np       = t.view(-1).cpu().numpy()

    deltas = {
        "dual":   (delta_u_np, r"$\delta_t$ (dual trajectory length)",   r"\delta_t"),
        "primal": (delta_x_np, r"$\delta^x_t$ (primal trajectory length)", r"\delta^x_t"),
    }
    corrs = {}

    for tag, (delta_np, xlabel, symbol) in deltas.items():
        # --- Experiment 1: scatter (delta, loss), colored by t ---
        fig, ax = plt.subplots(figsize=(5, 5))
        sc = ax.scatter(delta_np, loss_np, s=4, alpha=0.3, c=t_np, cmap="viridis")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"per-example FM loss $\varepsilon_t$")
        ax.set_title(f"{tag} delta_t vs reconstruction error")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("t")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"delta_{tag}_vs_loss.png"), dpi=100)
        plt.close(fig)

        corr = float(np.corrcoef(delta_np, loss_np)[0, 1])
        corrs[tag] = corr
        print(f"  Pearson correlation({tag} delta_t, loss) = {corr:.4f}")

        # --- Experiment 4: profile of E[delta] and E[loss] vs t ---
        bins        = np.linspace(0, 1, n_t_bins + 1)
        bin_idx     = np.clip(np.digitize(t_np, bins) - 1, 0, n_t_bins - 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        mean_delta = np.full(n_t_bins, np.nan)
        mean_loss  = np.full(n_t_bins, np.nan)
        for b in range(n_t_bins):
            mask = bin_idx == b
            if mask.any():
                mean_delta[b] = delta_np[mask].mean()
                mean_loss[b]  = loss_np[mask].mean()

        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(bin_centers, mean_delta, "o-", color="tab:blue", label=rf"$\mathbb{{E}}[{symbol}]$")
        ax1.set_xlabel("t")
        ax1.set_ylabel(rf"$\mathbb{{E}}[{symbol}]$", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(bin_centers, mean_loss, "s-", color="tab:red", label=r"$\mathbb{E}[\varepsilon_t]$")
        ax2.set_ylabel(r"$\mathbb{E}[\varepsilon_t]$ (FM loss)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        ax1.set_title(f"Profile of {tag} delta_t and FM loss vs t")
        fig.tight_layout()
        plt.savefig(os.path.join(run_dir, f"delta_{tag}_profile_vs_t.png"), dpi=100)
        plt.close(fig)

    np.savez(
        os.path.join(run_dir, "delta_data.npz"),
        delta_dual=delta_u_np, delta_primal=delta_x_np, loss=loss_np, t=t_np,
    )

    return corrs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="delta_t (dual trajectory length) as a difficulty indicator.")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--results-dir", type=str,   default="results_2moons_delta")
    parser.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--dual-dim",    type=int,   default=64)
    parser.add_argument("--K",           type=int,   default=10)
    parser.add_argument("--version",     type=str,   default="LFO", choices=["LFO", "LNO"])
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.results_dir, exist_ok=True)
    train_loader = get_moons_loader(batch_size=args.batch_size)

    model      = DFB_UNN(dim=DIM, K=args.K, dual_dim=args.dual_dim, version=args.version, learned_prox=False).to(device)
    model_name = f"DFB_UNN_{args.version}_delta"

    train_2moons(
        model        = model,
        train_loader = train_loader,
        device       = device,
        results_dir  = args.results_dir,
        model_name   = model_name,
        nb_epochs    = args.epochs,
    )

    run_dir = os.path.join(args.results_dir, model_name)
    print("\nEvaluating delta_t indicator...")
    evaluate_delta(model, train_loader, device, run_dir)

    print("Plotting unrolled primal paths...")
    save_unn_paths(
        model, train_loader, device,
        title=f"{model_name} — primal trajectory $x^{{[0..K]}}$",
        path=os.path.join(run_dir, "unn_paths.png"),
    )

    print("Plotting velocity field v_t(x)...")
    save_vector_field(
        model, train_loader, device,
        title=f"{model_name} — predicted velocity field $v_t(x)$",
        path=os.path.join(run_dir, "vector_field.png"),
    )


if __name__ == "__main__":
    main()
