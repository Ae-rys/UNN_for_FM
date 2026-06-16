# -*- coding: utf-8 -*-
"""
run_2moons.py
Flow Matching benchmark on 2-moons (dim=2), source = mixture of 8 Gaussians.

Usage
-----
    python run_2moons.py [--epochs N] [--results-dir DIR] [--only NAME] [--skip NAME]
"""

import argparse
import math
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, OTPlanSampler, ExactOptimalTransportConditionalFlowMatcher
from torchcfm.utils import torch_wrapper
from torchdyn.core import NeuralODE
from tqdm import tqdm
import torch.nn.functional as F

from models import DFB_UNN, DiFB_UNN, SharedDFB_UNN, SharedDiFB_UNN, ScCP_UNN, SharedScCP_UNN, small_MLP, SharedFLAT_DFB_UNN, FLAT_DFB_UNN

DIM        = 2
N_SAMPLES  = 10_000
BATCH_SIZE = 256


# ---------------------------------------------------------------------------
# Source distribution: 8 Gaussians on a circle
# ---------------------------------------------------------------------------

def sample_8gaussians(n, radius=2.0, std=0.3, device="cpu"):
    angles  = torch.linspace(0, 2 * math.pi, 9, device=device)[:-1]
    centers = radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    idx     = torch.randint(8, (n,), device=device)
    return centers[idx] + std * torch.randn(n, 2, device=device)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_moons_loader(n_samples=N_SAMPLES, batch_size=BATCH_SIZE, noise=0.05):
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    X     = torch.tensor(X, dtype=torch.float32)
    X     = (X - X.mean(0)) / X.std(0)
    return DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True, drop_last=True)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_scatter(samples, title, path, ref=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    if ref is not None:
        ax.scatter(ref[:, 0], ref[:, 1], s=4, alpha=0.25, c="gray",      label="target (2-moons)")
    ax.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.5, c="steelblue", label="generated")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.legend(markerscale=3, fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)


def save_overview(x0, x1_ref, generated, title, path):
    """Three-panel: source | target | generated."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, pts, lbl, col in zip(
        axes,
        [x0, x1_ref, generated],
        ["source (8 Gaussians)", "target (2-moons)", "generated"],
        ["darkorange", "gray", "steelblue"],
    ):
        ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.4, c=col)
        ax.set_title(lbl, fontsize=9)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)


def save_unn_paths(model, train_loader, device, title, path, n_samples=8, t_values=(0.1, 0.3, 0.5, 0.7, 0.9)):
    """Visualize the primal-trajectory x^[0..K] found by a DFB_UNN (model called
    with return_traj=True) for a handful of (x_t, t) samples.

    One panel per t value: the 2-moons target is shown in gray, each sample's
    path x^[0] -> ... -> x^[K] is drawn as a line with markers colored by
    iteration index (light = early, dark = late), the starting point x_t is a
    blue square, and the true target x1 is a green cross. This lets you check
    whether the unrolled iterations actually move x_t towards the target
    manifold (the "right path") rather than somewhere else.
    """
    model.eval()
    ref_data = torch.cat([x for (x,) in train_loader])[:2000].numpy()

    x1_all = torch.cat([x for (x,) in train_loader])
    perm   = torch.randperm(x1_all.shape[0])[:n_samples]
    x1     = x1_all[perm].to(device)
    B      = x1.shape[0]
    x0     = sample_8gaussians(B, device=device)

    n_t = len(t_values)
    fig, axes = plt.subplots(1, n_t, figsize=(4 * n_t, 4), constrained_layout=True)
    if n_t == 1:
        axes = [axes]

    with torch.no_grad():
        for ax, t_val in zip(axes, t_values):
            t    = torch.full((B, 1), float(t_val), device=device)
            xt   = (1 - t) * x0 + t * x1  # mean of the CFM linear path (sigma small)
            xt_t = torch.cat([xt, t], dim=-1)

            _, x_traj, _ = model(xt_t, return_traj=True)
            paths = torch.stack(x_traj, dim=1).cpu().numpy()  # (B, K+1, dim)

            ax.scatter(ref_data[:, 0], ref_data[:, 1], s=4, alpha=0.15, c="gray", label="target (2-moons)")
            for b in range(B):
                pts = paths[b]
                ax.plot(pts[:, 0], pts[:, 1], "-", color="black", alpha=0.3, linewidth=1, zorder=2)
                sc = ax.scatter(pts[:, 0], pts[:, 1], c=np.arange(pts.shape[0]), cmap="plasma", s=12, zorder=3)
            ax.scatter(xt.cpu().numpy()[:, 0], xt.cpu().numpy()[:, 1],
                       marker="s", facecolors="none", edgecolors="steelblue", s=50, label="$x_t$ (start)", zorder=4)
            ax.scatter(x1.cpu().numpy()[:, 0], x1.cpu().numpy()[:, 1],
                       marker="x", c="green", s=50, label="$x_1$ (target)", zorder=4)

            ax.set_title(f"t = {t_val:.2f}", fontsize=9)
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(-3.5, 3.5)

    axes[0].legend(markerscale=1.5, fontsize=7, loc="upper left")
    cbar = fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("unrolled iteration k")
    fig.suptitle(title, fontsize=10)
    plt.savefig(path, dpi=80)
    plt.close(fig)


def save_flat_unn_paths(model, x0_eval, ref_data, device, title, path, n_samples=8, mode="direct"):
    """Cascade trajectory x^[0..N] of FLAT_DFB_UNN.
    x^(0)=x0 (source, orange circle), x^(N)=x1_hat (green cross).
    Color encodes cascade step k (light=early, dark=late).
    """
    model.eval()
    perm = torch.randperm(x0_eval.shape[0])[:n_samples]
    x0   = x0_eval[perm].to(device)

    with torch.no_grad():
        _, x_traj = model(x0, return_traj=True, mode=mode)

    paths = torch.stack(x_traj, dim=1).cpu().numpy()  # (n_samples, N+1, 2)
    N     = paths.shape[1] - 1

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ref_data[:, 0], ref_data[:, 1], s=4, alpha=0.15, c="gray", label="target (2-moons)")
    sc = None
    for b in range(n_samples):
        pts = paths[b]
        ax.plot(pts[:, 0], pts[:, 1], "-", color="black", alpha=0.3, linewidth=1, zorder=2)
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=np.arange(N + 1), cmap="plasma",
                        vmin=0, vmax=N, s=20, zorder=3)
        ax.scatter(pts[0,  0], pts[0,  1], marker="o", facecolors="none",
                   edgecolors="darkorange", s=60, zorder=4)
        ax.scatter(pts[-1, 0], pts[-1, 1], marker="x", c="green", s=60, zorder=4)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("cascade step k")
    ax.legend(markerscale=2, fontsize=7, loc="upper left")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)


def save_flat_transport_field(model, ref_data, device, title, path,
                               grid_size=20, xlim=(-3.5, 3.5), mode="direct"):
    """Displacement field x0 -> model(x0) for FLAT_DFB_UNN (equivalent of vector_field).
    Arrows show where each source grid point is transported; color = ||x1_hat - x0||.
    """
    model.eval()
    xs, ys = torch.meshgrid(
        torch.linspace(*xlim, grid_size),
        torch.linspace(*xlim, grid_size),
        indexing="xy",
    )
    x0_grid = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).to(device)

    with torch.no_grad():
        x1_hat = model(x0_grid, mode="direct").cpu()

    x0_np   = x0_grid.cpu().numpy()
    disp_np = (x1_hat - x0_grid.cpu()).numpy()
    speed   = np.linalg.norm(disp_np, axis=-1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ref_data[:, 0], ref_data[:, 1], s=4, alpha=0.15, c="gray", label="target (2-moons)")
    q = ax.quiver(x0_np[:, 0], x0_np[:, 1], disp_np[:, 0], disp_np[:, 1],
                   speed, cmap="viridis", angles="xy", scale_units="xy", scale=1)
    cbar = fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\|x_1^{pred} - x_0\|$")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(*xlim); ax.set_ylim(*xlim)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def save_vector_field(model, train_loader, device, title, path,
                       t_values=(0.1, 0.3, 0.5, 0.7, 0.9), grid_size=20, xlim=(-3.5, 3.5), mode="direct"):
    """Plot the predicted velocity field v_t(x) = model(cat([x, t])) on a regular
    grid of x, one panel per t value, overlaid on the 2-moons target.

    Arrows are colored by their norm ||v_t(x)||. Useful to check whether the
    field points consistently from the source (8 Gaussians) towards the
    target (2-moons) as t goes from 0 to 1.
    """
    model.eval()
    ref_data = torch.cat([x for (x,) in train_loader])[:2000].numpy()

    xs, ys = torch.meshgrid(
        torch.linspace(*xlim, grid_size),
        torch.linspace(*xlim, grid_size),
        indexing="xy",
    )
    grid = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).to(device)

    n_t = len(t_values)
    fig, axes = plt.subplots(1, n_t, figsize=(5 * n_t, 5), constrained_layout=True)
    if n_t == 1:
        axes = [axes]

    with torch.no_grad():
        for ax, t_val in zip(axes, t_values):
            t    = torch.full((grid.shape[0], 1), float(t_val), device=device)
            xt_t = torch.cat([grid, t], dim=-1)
            vt   = model(xt_t, mode=mode)

            grid_np = grid.cpu().numpy()
            vt_np   = vt.cpu().numpy()
            speed   = np.linalg.norm(vt_np, axis=-1)

            ax.scatter(ref_data[:, 0], ref_data[:, 1], s=4, alpha=0.15, c="gray", label="target (2-moons)")
            q = ax.quiver(grid_np[:, 0], grid_np[:, 1], vt_np[:, 0], vt_np[:, 1],
                           speed, cmap="viridis", angles="xy")
            ax.set_title(f"t = {t_val:.2f}", fontsize=9)
            ax.set_xlim(*xlim)
            ax.set_ylim(*xlim)
            ax.set_aspect("equal")

    axes[0].legend(fontsize=7, loc="upper left")
    cbar = fig.colorbar(q, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label(r"$\|v_t(x)\|$")
    fig.suptitle(title, fontsize=10)
    plt.savefig(path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_2moons(
    model,
    train_loader,
    device,
    results_dir,
    model_name,
    nb_epochs=100,
    lr=1e-3,
    randomized_layer_nb=False,
    multi_iter=False,
):
    run_dir = os.path.join(results_dir, model_name)
    os.makedirs(run_dir, exist_ok=True)

    ref_data = torch.cat([x for (x,) in train_loader])[:2000].numpy()

    FM        = ExactOptimalTransportConditionalFlowMatcher(sigma=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}\nModel : {model_name}\nParams: {total_params:,}\n{'='*60}")

    with open(os.path.join(run_dir, "params.txt"), "w") as f:
        f.write(f"{total_params}\n")

    loss_history = []
    train_start  = time.perf_counter()
    eval_every   = max(1, nb_epochs // 5)

    for epoch in range(nb_epochs):
        model.train()
        total_loss = 0.0
        t0 = time.perf_counter()

        for (x1_batch,) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{nb_epochs}", leave=False):
            x1 = x1_batch.to(device)
            B  = x1.shape[0]
            x0 = torch.randn(B, DIM, device=device)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            xt_t = torch.cat([xt, t.view(B, 1)], dim=-1)

            if randomized_layer_nb:
                vt = model(xt_t, n_iter=random.randint(5, 15))
            else:
                vt = model(xt_t)

            loss = torch.mean((vt - ut) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        elapsed  = time.perf_counter() - t0
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch+1}/{nb_epochs} — loss: {avg_loss:.4f}  ({elapsed:.1f}s)")

        if (epoch + 1) % eval_every == 0 or epoch == nb_epochs - 1:
            model.eval()
            with torch.no_grad():
                x0_test = torch.randn(2000, DIM, device=device)
                t_span  = torch.linspace(0, 1, 2, device=device)

                if multi_iter:
                    for n_it in [5, 10, 20, 30]:
                        ode_fn = lambda t, x, ni=n_it, **kw: model(
                            torch.cat([x, t.expand(x.shape[0], 1)], dim=-1), n_iter=ni,
                        )
                        node = NeuralODE(
                            ode_fn, solver="dopri5", atol=1e-4, rtol=1e-4,
                            optimizable_params=model.parameters(),
                        )
                        traj = node.trajectory(x0_test, t_span=t_span)
                        save_scatter(
                            traj[-1].cpu().numpy(),
                            title=f"{model_name} — epoch {epoch+1}, {n_it} iters",
                            path=os.path.join(run_dir, f"epoch_{epoch+1}_niter_{n_it}.png"),
                            ref=ref_data,
                        )
                    # Overview panel at last eval iter (n_it=10)
                    ode_fn10 = lambda t, x, **kw: model(
                        torch.cat([x, t.expand(x.shape[0], 1)], dim=-1), n_iter=10,
                    )
                    node10 = NeuralODE(ode_fn10, solver="dopri5", atol=1e-4, rtol=1e-4,
                                       optimizable_params=model.parameters())
                    gen = node10.trajectory(x0_test, t_span=t_span)[-1].cpu().numpy()
                    save_overview(
                        x0_test.cpu().numpy(), ref_data, gen,
                        title=f"{model_name} — epoch {epoch+1}",
                        path=os.path.join(run_dir, f"overview_epoch_{epoch+1}.png"),
                    )
                else:
                    node = NeuralODE(torch_wrapper(model), solver="dopri5", atol=1e-4, rtol=1e-4)
                    traj = node.trajectory(x0_test, t_span=t_span)
                    gen  = traj[-1].cpu().numpy()
                    save_scatter(
                        gen,
                        title=f"{model_name} — epoch {epoch+1}",
                        path=os.path.join(run_dir, f"epoch_{epoch+1}.png"),
                        ref=ref_data,
                    )
                    save_overview(
                        x0_test.cpu().numpy(), ref_data, gen,
                        title=f"{model_name} — epoch {epoch+1}",
                        path=os.path.join(run_dir, f"overview_epoch_{epoch+1}.png"),
                    )

                save_vector_field(
                    model, train_loader, device,
                    title=f"{model_name} — epoch {epoch+1}",
                    path=os.path.join(run_dir, f"vector_field_epoch_{epoch+1}.png"),
                )
            model.train()

    # Loss curve
    plt.figure()
    plt.plot(range(1, nb_epochs + 1), loss_history, marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Average FM loss")
    plt.title(f"Training loss — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss.png"))
    plt.close()

    with open(os.path.join(run_dir, "loss.txt"), "w") as f:
        for ep, l in enumerate(loss_history, 1):
            f.write(f"{ep}\t{l:.6f}\n")

    total_time = time.perf_counter() - train_start
    print(f"  Results saved to: {run_dir}")
    return loss_history, total_params, total_time


def train_flat_2moons(
    model,
    train_loader,
    device,
    results_dir,
    model_name,
    nb_epochs=100,
    lr=1e-3,
    w_velocity=2.0,
    intermediate_supervision=True,
):
    """
    FLAT training (Qi et al. 2026) for FLAT_DFB_UNN on 2-moons.

    Loss: L = L_recon + w_velocity * (1/N) * sum_k L1(x^(k+1), x*_{t_{k+1}})
    where x*_{t_{k+1}} = (1-t_{k+1})*x0 + t_{k+1}*x1  (linearly interpolated target).

    At eval, generates with both modes so direct vs lookahead can be compared.
    """
    run_dir = os.path.join(results_dir, model_name)
    os.makedirs(run_dir, exist_ok=True)
    ref_data = torch.cat([x for (x,) in train_loader])[:2000].numpy()

    optimizer    = torch.optim.Adam(model.parameters(), lr=lr)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}\nModel : {model_name} (FLAT, alpha={model.alpha})\nParams: {total_params:,}\n{'='*60}")

    K            = model.N
    loss_history = []
    loss_recon_h = []
    loss_vel_h   = []
    train_start  = time.perf_counter()
    eval_every   = max(1, nb_epochs // 5)

    ot_sampler = OTPlanSampler(method="exact")

    for epoch in range(nb_epochs):
        model.train()
        total_loss = total_recon = total_vel = 0.0
        t0 = time.perf_counter()

        for (x1_batch,) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{nb_epochs}", leave=False):
            x1 = x1_batch.to(device)
            B  = x1.shape[0]
            x0 = torch.randn(B, DIM, device=device)

            x0_paired, x1_paired = ot_sampler.sample_plan(x0, x1)

            x_pred, x_states = model(x0_paired, return_x1_hats=True)
            # x_states = [x^(1), ..., x^(N)]
            x_traj  = [x0_paired] + list(x_states)
            t_sched = model.get_schedule(device=device)

            # Component I: reconstruction loss at final cascade
            loss_recon = F.mse_loss(x_pred, x1_paired)

            # Component II: velocity alignment at each intermediate cascade
            if intermediate_supervision:
                loss_vel = torch.zeros(1, device=device)
                for k in range(K):
                    t_kp1  = t_sched[k + 1]
                    x_star = (1.0 - t_kp1) * x0_paired + t_kp1 * x1_paired
                    loss_vel = loss_vel + F.l1_loss(x_traj[k + 1], x_star)
                loss_vel = loss_vel / K
                loss = loss_recon + w_velocity * loss_vel
            else:
                loss_vel = torch.zeros(1, device=device)
                loss     = loss_recon

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss  += loss.item()
            total_recon += loss_recon.item()
            total_vel   += loss_vel.item()

        n_batches = len(train_loader)
        avg_loss  = total_loss  / n_batches
        avg_recon = total_recon / n_batches
        avg_vel   = total_vel   / n_batches
        elapsed   = time.perf_counter() - t0
        loss_history.append(avg_loss)
        loss_recon_h.append(avg_recon)
        loss_vel_h.append(avg_vel)
        print(f"  Epoch {epoch+1}/{nb_epochs} — loss: {avg_loss:.4f} "
              f"(recon: {avg_recon:.4f}, vel: {avg_vel:.4f})  ({elapsed:.1f}s)")

        if (epoch + 1) % eval_every == 0 or epoch == nb_epochs - 1:
            model.eval()
            with torch.no_grad():
                x0_test      = torch.randn(2000, DIM, device=device)
                gen_direct   = model(x0_test, mode="direct").cpu().numpy()
                gen_lookahead = model(x0_test, mode="velocity_step").cpu().numpy()

                for tag, gen in [("lookahead", gen_lookahead)]: # [("direct", gen_direct), ("lookahead", gen_lookahead)]:
                    save_scatter(
                        gen,
                        title=f"{model_name} — epoch {epoch+1} ({tag})",
                        path=os.path.join(run_dir, f"epoch_{epoch+1}_{tag}.png"),
                        ref=ref_data,
                    )
                # save_overview(
                #     x0_test.cpu().numpy(), ref_data, gen_direct,
                #     title=f"{model_name} — epoch {epoch+1}",
                #     path=os.path.join(run_dir, f"overview_direct_epoch_{epoch+1}.png"),
                # )
                save_overview(
                    x0_test.cpu().numpy(), ref_data, gen_lookahead,
                    title=f"{model_name} — epoch {epoch+1}",
                    path=os.path.join(run_dir, f"overview_lookahead_epoch_{epoch+1}.png"),
                )
                # save_flat_unn_paths(
                #     model, x0_test, ref_data, device,
                #     title=f"{model_name} — epoch {epoch+1}",
                #     path=os.path.join(run_dir, f"unn_paths_epoch_{epoch+1}.png"),
                #     mode="direct",
                # )
                save_flat_unn_paths(
                    model, x0_test, ref_data, device,
                    title=f"{model_name} — epoch {epoch+1}",
                    path=os.path.join(run_dir, f"unn_paths_epoch_{epoch+1}.png"),
                    mode="velocity_step",
                )
                # save_flat_transport_field(
                #     model, ref_data, device,
                #     title=f"{model_name} — epoch {epoch+1}",
                #     path=os.path.join(run_dir, f"transport_field_epoch_{epoch+1}.png"),
                #     mode="direct",
                # )
                save_flat_transport_field(
                    model, ref_data, device,
                    title=f"{model_name} — epoch {epoch+1}",
                    path=os.path.join(run_dir, f"transport_field_epoch_{epoch+1}.png"),
                    mode="velocity_step",
                )
                
            model.train()

    plt.figure()
    plt.plot(range(1, nb_epochs + 1), loss_history,  marker="o", markersize=3, label="total")
    plt.plot(range(1, nb_epochs + 1), loss_recon_h,  marker="o", markersize=3, label="recon (MSE)")
    plt.plot(range(1, nb_epochs + 1), loss_vel_h,    marker="o", markersize=3, label="vel (L1, /N)")
    plt.xlabel("Epoch")
    plt.ylabel("Average FLAT loss")
    plt.title(f"Training loss — {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss.png"))
    plt.close()

    with open(os.path.join(run_dir, "loss.txt"), "w") as f:
        f.write("epoch\ttotal\trecon\tvel\n")
        for ep, (l, lr_, lv) in enumerate(zip(loss_history, loss_recon_h, loss_vel_h), 1):
            f.write(f"{ep}\t{l:.6f}\t{lr_:.6f}\t{lv:.6f}\n")

    total_time = time.perf_counter() - train_start
    print(f"  Results saved to: {run_dir}")
    return loss_history, total_params, total_time


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def build_experiments(device):
    return [
        # ---- Baselines ----
        dict(
            name  = "MLP_baseline",
            build = lambda: small_MLP(dim=DIM, w=64, time_varying=True).to(device),
            kwargs= {},
        ),
        # ---- Standard UNNs (flat, per-layer W, dual_dim=16) ----
        dict(
            name  = "DiFB_UNN_LFO",
            build = lambda: DiFB_UNN(dim=DIM, K=10, w=32, dual_dim=2, version="LFO").to(device),
            kwargs= {},
        ),
        dict(
            name  = "DiFB_UNN_LNO",
            build = lambda: DiFB_UNN(dim=DIM, K=10, w=64, dual_dim=32, version="LNO").to(device),
            kwargs= {},
        ),
        dict(
            name  = "ScCP_UNN_LFO",
            build = lambda: ScCP_UNN(dim=DIM, K=10, w=64, dual_dim=32, version="LFO").to(device),
            kwargs= {},
        ),
        dict(
            name  = "ScCP_UNN_LNO",
            build = lambda: ScCP_UNN(dim=DIM, K=10, w=64, dual_dim=32, version="LNO").to(device),
            kwargs= {},
        ),
        # ---- Standard UNNs + L1 prox ----
        dict(
            name  = "DiFB_UNN_L1_LFO",
            build = lambda: DiFB_UNN(dim=DIM, K=10, w=64, dual_dim=32, version="LFO", prox_type="l1").to(device),
            kwargs= {},
        ),
        dict(
            name  = "DiFB_UNN_L1_LNO",
            build = lambda: DiFB_UNN(dim=DIM, K=10, w=64, dual_dim=32, version="LNO", prox_type="l1").to(device),
            kwargs= {},
        ),
        dict(
            name  = "ScCP_UNN_L1_LFO",
            build = lambda: ScCP_UNN(dim=DIM, K=10, w=64, dual_dim=64, version="LFO", prox_type="l1").to(device),
            kwargs= {},
        ),
        dict(
            name  = "ScCP_UNN_L1_LNO",
            build = lambda: ScCP_UNN(dim=DIM, K=10, w=64, dual_dim=64, version="LNO", prox_type="l1").to(device),
            kwargs= {},
        ),
        # ---- DFB + L1 prox (dual L1 ball projection) ----
        dict(
            name  = "DFB_UNN_L1_LFO",
            build = lambda: DFB_UNN(dim=DIM, K=10, dual_dim=32, version="LFO", learned_prox=False).to(device),
            kwargs= {},
        ),
        dict(
            name  = "DFB_UNN_L1_LNO",
            build = lambda: DFB_UNN(dim=DIM, K=10, dual_dim=32, version="LNO", learned_prox=False).to(device),
            kwargs= {},
        ),
        # ---- Shared DFB (flat, shared W, dual_dim=32) ----
        dict(
            name  = "SharedDFB_UNN_LFO_rand",
            build = lambda: SharedDFB_UNN(dim=DIM, K=10, w=64, dual_dim=2, version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedDFB_UNN_LNO_rand",
            build = lambda: SharedDFB_UNN(dim=DIM, K=10, w=64, dual_dim=64, version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedDFB_UNN_LFO_fixed",
            build = lambda: SharedDFB_UNN(dim=DIM, K=10, w=64, dual_dim=64, version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        dict(
            name  = "SharedDFB_UNN_LNO_fixed",
            build = lambda: SharedDFB_UNN(dim=DIM, K=10, w=64, dual_dim=64, version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        # ---- Shared DFB + L1 prox ----
        dict(
            name  = "SharedDFB_UNN_L1_LFO_rand",
            build = lambda: SharedDFB_UNN(dim=DIM, K=10, dual_dim=64, version="LFO", prox_type="l1").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedDFB_UNN_L1_LNO_rand",
            build = lambda: SharedDFB_UNN(dim=DIM, K=10, dual_dim=64, version="LNO", prox_type="l1").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedDFB_UNN_L1_LFO_fixed",
            build = lambda: SharedDFB_UNN(dim=DIM, K=10, dual_dim=1024, version="LFO", prox_type="l1").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        dict(
            name  = "SharedDFB_UNN_L1_LNO_fixed",
            build = lambda: SharedDFB_UNN(dim=DIM, K=10, dual_dim=1024, version="LNO", prox_type="l1").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        # ---- Shared DiFB (flat, shared W, momentum on dual variable) ----
        dict(
            name  = "SharedDiFB_UNN_LFO_rand",
            build = lambda: SharedDiFB_UNN(dim=DIM, K=10, w=64, dual_dim=2, version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedDiFB_UNN_LNO_rand",
            build = lambda: SharedDiFB_UNN(dim=DIM, K=10, w=64, dual_dim=64, version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedDiFB_UNN_LFO_fixed",
            build = lambda: SharedDiFB_UNN(dim=DIM, K=10, w=64, dual_dim=64, version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        dict(
            name  = "SharedDiFB_UNN_LNO_fixed",
            build = lambda: SharedDiFB_UNN(dim=DIM, K=10, w=64, dual_dim=64, version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        # ---- Shared DiFB + L1 prox ----
        dict(
            name  = "SharedDiFB_UNN_L1_LFO_rand",
            build = lambda: SharedDiFB_UNN(dim=DIM, K=10, dual_dim=64, version="LFO", prox_type="l1").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedDiFB_UNN_L1_LNO_rand",
            build = lambda: SharedDiFB_UNN(dim=DIM, K=10, dual_dim=64, version="LNO", prox_type="l1").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedDiFB_UNN_L1_LFO_fixed",
            build = lambda: SharedDiFB_UNN(dim=DIM, K=10, dual_dim=1024, version="LFO", prox_type="l1").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        dict(
            name  = "SharedDiFB_UNN_L1_LNO_fixed",
            build = lambda: SharedDiFB_UNN(dim=DIM, K=10, dual_dim=1024, version="LNO", prox_type="l1").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        # ---- Shared ScCP (flat, shared W, adaptive tau/sigma/alpha schedule) ----
        dict(
            name  = "SharedScCP_UNN_LFO_rand",
            build = lambda: SharedScCP_UNN(dim=DIM, K=10, w=64, dual_dim=2, version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedScCP_UNN_LNO_rand",
            build = lambda: SharedScCP_UNN(dim=DIM, K=10, w=64, dual_dim=64, version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedScCP_UNN_LFO_fixed",
            build = lambda: SharedScCP_UNN(dim=DIM, K=10, w=64, dual_dim=64, version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        dict(
            name  = "SharedScCP_UNN_LNO_fixed",
            build = lambda: SharedScCP_UNN(dim=DIM, K=10, w=64, dual_dim=64, version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        # ---- Shared ScCP + L1 prox ----
        dict(
            name  = "SharedScCP_UNN_L1_LFO_rand",
            build = lambda: SharedScCP_UNN(dim=DIM, K=10, dual_dim=64, version="LFO", prox_type="l1").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedScCP_UNN_L1_LNO_rand",
            build = lambda: SharedScCP_UNN(dim=DIM, K=10, dual_dim=64, version="LNO", prox_type="l1").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedScCP_UNN_L1_LFO_fixed",
            build = lambda: SharedScCP_UNN(dim=DIM, K=10, dual_dim=1024, version="LFO", prox_type="l1").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        dict(
            name  = "SharedScCP_UNN_L1_LNO_fixed",
            build = lambda: SharedScCP_UNN(dim=DIM, K=10, dual_dim=1024, version="LNO", prox_type="l1").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        # ---- FLAT DFB  ----
        dict(
            name  = "FLAT_DFB_UNN_LFO",
            build = lambda: FLAT_DFB_UNN(dim=DIM, N=16, dual_dim=32, version="LFO", alpha=2.0).to(device),
            kwargs= dict(is_flat=True),
        ),
        # dict(
        #     name  = "SharedFLAT_DFB_UNN_LFO",
        #     build = lambda: SharedFLAT_DFB_UNN(dim=DIM, N=10, dual_dim=64, version="LFO").to(device),
        #     kwargs= dict(is_flat=True),
        # ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Flow Matching: 2-moons from 8 Gaussians.")
    parser.add_argument("--epochs",      type=int, default=100)
    parser.add_argument("--results-dir", type=str, default="results_2moons")
    parser.add_argument("--skip",        type=str, default="")
    parser.add_argument("--only",        type=str, default="")
    parser.add_argument("--batch-size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.results_dir, exist_ok=True)
    train_loader = get_moons_loader(batch_size=args.batch_size)

    experiments = build_experiments(device)
    if args.only:
        experiments = [e for e in experiments if args.only in e["name"]]
    if args.skip:
        experiments = [e for e in experiments if args.skip not in e["name"]]

    print(f"\n{len(experiments)} experiment(s) to run.\n")

    summary = []
    for i, exp in enumerate(experiments, 1):
        name = exp["name"]
        print(f"\n[{i}/{len(experiments)}] {name}")

        kwargs  = exp.get("kwargs", {})
        is_flat = kwargs.pop("is_flat", False)

        try:
            model = exp["build"]()
            if is_flat:
                losses, n_params, train_time = train_flat_2moons(
                    model        = model,
                    train_loader = train_loader,
                    device       = device,
                    results_dir  = args.results_dir,
                    model_name   = name,
                    nb_epochs    = args.epochs,
                    **kwargs
                )
            else:
                model = exp["build"]()
                losses, n_params, train_time = train_2moons(
                    model        = model,
                    train_loader = train_loader,
                    device       = device,
                    results_dir  = args.results_dir,
                    model_name   = name,
                    nb_epochs    = args.epochs,
                    **exp.get("kwargs", {}),
                )
            summary.append((name, losses[-1], n_params, train_time, "OK"))
        except Exception as exc:
            import traceback; traceback.print_exc()
            summary.append((name, float("nan"), 0, 0.0, str(exc)))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, final_loss, n_params, train_time, status in summary:
        print(f"  {name:<40} params={n_params:>8,}  final_loss={final_loss:.4f}  time={train_time:6.0f}s  [{status}]")

    summary_path = os.path.join(args.results_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("model_name\tn_params\tfinal_loss\ttrain_time_s\tstatus\n")
        for name, final_loss, n_params, train_time, status in summary:
            f.write(f"{name}\t{n_params}\t{final_loss:.6f}\t{train_time:.1f}\t{status}\n")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
