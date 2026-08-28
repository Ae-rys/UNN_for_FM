# -*- coding: utf-8 -*-
"""
run_2moons.py
Flow Matching benchmark on 2-moons (dim=2), source = 1 Gaussian.

Usage
-----
    python run_2moons.py [--epochs N] [--results-dir DIR] [--only NAME] [--skip NAME]
"""

import argparse
import math
import os
import random
import time

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, OTPlanSampler, ExactOptimalTransportConditionalFlowMatcher
from torchcfm.utils import torch_wrapper
from torchdyn.core import NeuralODE
from tqdm import tqdm
import torch.nn.functional as F

from models import DFB_UNN, DiFB_UNN, SharedDFB_UNN, SharedDiFB_UNN, ScCP_UNN, SharedScCP_UNN, small_MLP, FLAT_DFB_UNN, FLAT_DFB_UNN_v2
from flops_utils import (
    count_velocity_flops, flops_vector_model, write_train_time, write_velocity_flops,
)

DIM        = 2
N_SAMPLES  = 10_000
BATCH_SIZE = 256


# ---------------------------------------------------------------------------
# Source distribution: 1 Gaussian
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
        ["source (1 Gaussian)", "target (2-moons)", "generated"],
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
    x0     = torch.randn(B, DIM, device=device)

    n_t = len(t_values)
    fig, axes = plt.subplots(1, n_t, figsize=(4 * n_t, 4), constrained_layout=True)
    if n_t == 1:
        axes = [axes]

    with torch.no_grad():
        for ax, t_val in zip(axes, t_values):
            t    = torch.full((B, 1), float(t_val), device=device)
            xt   = (1 - t) * x0 + t * x1  # mean of the CFM linear path (sigma small)
            xt_t = torch.cat([xt, t], dim=-1)

            _, x_traj = model(xt_t, return_traj=True)
            paths = torch.stack(x_traj, dim=1).cpu().numpy()  # (B, K+1, dim)

            ax.scatter(ref_data[:, 0], ref_data[:, 1], s=4, alpha=0.15, c="gray", label="target (2-moons)")
            for b in range(B):
                pts = paths[b]
                ax.plot(pts[:, 0], pts[:, 1], "-", color="black", alpha=0.3, linewidth=1, zorder=2)
                sc = ax.scatter(pts[:, 0], pts[:, 1], c=np.arange(pts.shape[0]), cmap="plasma", s=12, zorder=3)
            ax.scatter(xt.cpu().numpy()[:, 0], xt.cpu().numpy()[:, 1],
                       marker="s", facecolors="none", edgecolors="steelblue", s=50, label="$x_t$", zorder=4)
            ax.scatter(x1.cpu().numpy()[:, 0], x1.cpu().numpy()[:, 1],
                       marker="x", c="green", s=50, label="$x_1$", zorder=4)

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


def save_per_layer_updates(model, ref_data, device, title, path,
                            n_samples=300, selected_layers=None, mode="direct"):
    """Per-layer displacement δ^(k) = x^(k+1) - x^(k) for FLAT_DFB_UNN.
    Each panel: quiver of what layer k contributes, at the positions x^(k).
    Reveals which layers do most of the work and where the cascade "hesitates".
    """
    model.eval()
    N = model.N
    if selected_layers is None:
        # ~5 evenly spaced layers, always including first and last
        step = max(1, (N - 1) // 4)
        selected_layers = sorted(set(range(0, N, step)) | {N - 1})

    x0 = torch.randn(n_samples, 2, device=device)
    with torch.no_grad():
        _, x_traj = model(x0, return_traj=True, mode=mode)

    paths = torch.stack(x_traj, dim=1).cpu().numpy()  # (n_samples, N+1, 2)
    t_sched = model.get_schedule().numpy()

    n_panels = len(selected_layers)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    q = None
    for ax, k in zip(axes, selected_layers):
        pos   = paths[:, k,     :]          # x^(k)
        delta = paths[:, k + 1, :] - pos    # δ^(k) = x^(k+1) - x^(k)
        speed = np.linalg.norm(delta, axis=-1)

        ax.scatter(ref_data[:, 0], ref_data[:, 1], s=3, alpha=0.12, c="gray")
        q = ax.quiver(pos[:, 0], pos[:, 1], delta[:, 0], delta[:, 1],
                       speed, cmap="plasma", angles="xy", scale_units="xy", scale=1,
                       width=0.003, headwidth=4)
        ax.set_title(f"k={k}  t: {t_sched[k]:.2f}→{t_sched[k+1]:.2f}", fontsize=9)
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")

    if q is not None:
        cbar = fig.colorbar(q, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label(r"$\|\delta^{(k)}\| = \|x^{(k+1)} - x^{(k)}\|$")
    fig.suptitle(title, fontsize=10)
    plt.savefig(path, dpi=100)
    plt.close(fig)


def save_density_diff(generated, ref_data, title, path, bins=60, xlim=(-3.5, 3.5)):
    """2D histogram difference: H(generated) - H(target).
    Red = over-generated regions, blue = under-generated / missed modes.
    Overlaid contours show the target density for reference.
    Directly diagnoses mode collapse and coverage failures.
    """
    edges = np.linspace(*xlim, bins + 1)
    H_gen, _, _ = np.histogram2d(generated[:, 0], generated[:, 1],
                                  bins=[edges, edges], density=True)
    H_ref, _, _ = np.histogram2d(ref_data[:, 0], ref_data[:, 1],
                                  bins=[edges, edges], density=True)
    H_diff = H_gen - H_ref
    centers = 0.5 * (edges[:-1] + edges[1:])
    XX, YY  = np.meshgrid(centers, centers)

    # Optional smoothing via scipy — degrades gracefully if not available
    try:
        from scipy.ndimage import gaussian_filter
        H_gen_s  = gaussian_filter(H_gen,  sigma=1.5)
        H_ref_s  = gaussian_filter(H_ref,  sigma=1.5)
        H_diff_s = gaussian_filter(H_diff, sigma=1.5)
    except ImportError:
        H_gen_s, H_ref_s, H_diff_s = H_gen, H_ref, H_diff

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # Panel 1: generated density
    im0 = axes[0].pcolormesh(XX, YY, H_gen_s.T, cmap="Blues", shading="auto")
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title("generated density", fontsize=9)

    # Panel 2: target density
    im1 = axes[1].pcolormesh(XX, YY, H_ref_s.T, cmap="Greys", shading="auto")
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_title("target density", fontsize=9)

    # Panel 3: difference (diverging)
    vmax = np.abs(H_diff_s).max() or 1.0
    im2  = axes[2].pcolormesh(XX, YY, H_diff_s.T, cmap="RdBu_r",
                               vmin=-vmax, vmax=vmax, shading="auto")
    axes[2].contour(XX, YY, H_ref_s.T, levels=5, colors="black", alpha=0.4, linewidths=0.7)
    fig.colorbar(im2, ax=axes[2])
    axes[2].set_title("generated − target  (red=over, blue=missed)", fontsize=9)

    for ax in axes:
        ax.set_xlim(*xlim); ax.set_ylim(*xlim)
        ax.set_aspect("equal")

    fig.suptitle(title, fontsize=10)
    plt.savefig(path, dpi=100)
    plt.close(fig)


def save_vector_field(model, train_loader, device, title, path,
                       t_values=(0.1, 0.3, 0.5, 0.7, 0.9), grid_size=20, xlim=(-3.5, 3.5), mode="direct"):
    """Plot the predicted velocity field v_t(x) = model(cat([x, t])) on a regular
    grid of x, one panel per t value, overlaid on the 2-moons target.

    Arrows are colored by their norm ||v_t(x)||. Useful to check whether the
    field points consistently from the source (Gaussian) towards the
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
            vt   = model(xt_t)

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
# Checkpoint inference: explicit Euler / Heun integration of the ODE
# ---------------------------------------------------------------------------

CHECKPOINT_MODEL_BUILDERS = {
    "DFB":  lambda K, dual_dim, version: DFB_UNN(
        dim=DIM, K=K, dual_dim=dual_dim, version=version, learned_prox=False,
    ),
    "ScCP": lambda K, dual_dim, version: ScCP_UNN(
        dim=DIM, K=K, dual_dim=dual_dim, version=version, prox_type="l1",
    ),
}


def integrate_ode(model, x0, n_steps, device, method="euler"):
    """Integrate dx/dt = v_t(x) from t=0 to t=1 with `n_steps` explicit steps
    of size dt = 1/n_steps (i.e. "diviser le temps en n_steps").

    method="euler" : x_{k+1} = x_k + dt * v(x_k, t_k)                     (1st order)
    method="heun"   : predictor-corrector / explicit RK2                  (2nd order)
                      x_pred = x_k + dt * v(x_k, t_k)
                      x_{k+1} = x_k + dt/2 * (v(x_k, t_k) + v(x_pred, t_{k+1}))

    Both methods are explicit single-step schemes, so they naturally expose
    every intermediate state x_k (unlike adaptive solvers such as dopri5,
    which only report the states at the requested t_span). Heun is a strict
    improvement over Euler at the same cost (one extra model call per step)
    and is the standard "a bit better than Euler" choice here; higher-order
    Runge-Kutta would also expose intermediate states but needs 3-4 calls/step.

    Returns a list of length n_steps+1: traj[k] = x at t = k/n_steps.
    """
    dt = 1.0 / n_steps
    x  = x0
    t  = torch.zeros(x.shape[0], 1, device=device)
    traj = [x.clone()]

    model.eval()
    with torch.no_grad():
        for _ in range(n_steps):
            xt_t = torch.cat([x, t], dim=-1)
            v1   = model(xt_t)

            if method == "euler":
                x_next = x + dt * v1
            elif method == "heun":
                x_pred      = x + dt * v1
                t_next      = t + dt
                xt_t_pred   = torch.cat([x_pred, t_next], dim=-1)
                v2          = model(xt_t_pred)
                x_next      = x + 0.5 * dt * (v1 + v2)
            else:
                raise ValueError(f"Unknown method: {method!r} (use 'euler' or 'heun')")

            x = x_next
            t = t + dt
            traj.append(x.clone())

    return traj


def plot_checkpoint_trajectory(
    model_pt_path,
    model_type,
    K,
    dual_dim,
    version,
    device,
    n_steps=1000,
    method="euler",
    n_snapshots=8,
    n_samples=2000,
    train_loader=None,
    out_path=None,
):
    """Load a trained DFB_UNN/ScCP_UNN from a `model.pt` checkpoint (as saved
    by train_2moons) and plot the generated distribution at several points
    along the t in [0, 1] trajectory, obtained by explicit Euler/Heun
    integration with `n_steps` steps (dt = 1/n_steps).

    model_type : "DFB" or "ScCP".
    K, dual_dim, version : architecture hyperparameters used when the
        checkpoint was trained (must match exactly — they are not stored in
        model.pt, only the weights are).
    n_snapshots : number of panels to draw (evenly spaced in step index,
        always including step 0 and step n_steps).
    out_path : if given, save the figure there instead of showing it.
    """
    if model_type not in CHECKPOINT_MODEL_BUILDERS:
        raise ValueError(f"Unknown model_type: {model_type!r} (use 'DFB' or 'ScCP')")

    model = CHECKPOINT_MODEL_BUILDERS[model_type](K, dual_dim, version).to(device)
    state = torch.load(model_pt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    if train_loader is None:
        train_loader = get_moons_loader()
    ref_data = torch.cat([x for (x,) in train_loader])[:2000].numpy()

    x0 = torch.randn(n_samples, DIM, device=device)

    traj = integrate_ode(model, x0, n_steps, device, method=method)

    snap_idx = sorted(set(np.linspace(0, n_steps, n_snapshots).round().astype(int)))
    fig, axes = plt.subplots(1, len(snap_idx), figsize=(3.5 * len(snap_idx), 3.5), constrained_layout=True)
    if len(snap_idx) == 1:
        axes = [axes]

    for ax, idx in zip(axes, snap_idx):
        pts   = traj[idx].cpu().numpy()
        t_val = idx / n_steps
        ax.scatter(ref_data[:, 0], ref_data[:, 1], s=4, alpha=0.15, c="gray", label="target (2-moons)")
        ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.5, c="steelblue", label="generated")
        ax.set_title(f"t={t_val:.3f}  (step {idx}/{n_steps})", fontsize=9)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")

    axes[0].legend(markerscale=3, fontsize=7, loc="upper left")
    fig.suptitle(
        f"{model_type} {version} K={K} dual_dim={dual_dim} — "
        f"{method} integration, n_steps={n_steps}",
        fontsize=10,
    )

    if out_path:
        plt.savefig(out_path, dpi=100)
        plt.close(fig)
    else:
        plt.show()

    return traj


# ---------------------------------------------------------------------------
# Quantitative error metric
# ---------------------------------------------------------------------------

def compute_w2(generated, target, max_n=500):
    """Exact 2-Wasserstein distance between `generated` and `target` point
    clouds (subsampled to `max_n` points each for tractable exact OT)."""
    n = min(len(generated), len(target), max_n)
    a = np.asarray(generated[:n], dtype=np.float64)
    b = np.asarray(target[:n], dtype=np.float64)
    M = ot.dist(a, b, metric="sqeuclidean")
    w = np.full(n, 1.0 / n)
    w2_sq = ot.emd2(w, w, M)
    return float(np.sqrt(max(w2_sq, 0.0)))


def write_param_file(model, run_dir):
    """Write "parametres.txt" in run_dir with the architecture hyperparameters
    of `model` (number of layers K, dual_dim, version, model class), read
    directly off the model's attributes. Used by make_grille.py to build the
    K x dual_dim grid without having to guess these values from the run name.
    """
    fields = {
        "model_class": type(model).__name__,
        "K":           getattr(model, "K", None),
        "N":           getattr(model, "N", None),  # FLAT_DFB_UNN uses N instead of K
        "dual_dim":    getattr(model, "dual_dim", None),
        "version":     getattr(model, "version", None),
        "begin_div":   getattr(model, "begin_div", None),
        "end_div":     getattr(model, "end_div", None),
        "pred":        getattr(model, "pred", None),
    }
    with open(os.path.join(run_dir, "parametres.txt"), "w") as f:
        for key, value in fields.items():
            if value is not None:
                f.write(f"{key}={value}\n")


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

    FM        = ExactOptimalTransportConditionalFlowMatcher(sigma=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}\nModel : {model_name}\nParams: {total_params:,}\n{'='*60}")

    with open(os.path.join(run_dir, "params.txt"), "w") as f:
        f.write(f"{total_params}\n")
    write_param_file(model, run_dir)
    write_velocity_flops(run_dir, flops_vector_model(model, DIM, device))

    loss_history  = []
    error_history = []  # (epoch, W2 distance to target)
    train_start   = time.perf_counter()
    eval_every    = max(1, nb_epochs // 5)

    for epoch in range(nb_epochs):
        model.train()
        total_loss = 0.0
        t0 = time.perf_counter()

        for (x1_batch,) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{nb_epochs}", leave=False):
            x1 = x1_batch.to(device)
            B  = x1.shape[0]
            x0 = torch.randn(B, DIM, device=device)

            t = torch.rand(B, 1, device=device) * 0.9 + 0.05  # avoid t=0 and t=1

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t=t)
            xt_t = torch.cat([xt, t.view(B, 1)], dim=-1)

            # on réordonne x0, x1
            x1 = xt + (ut * (1 - t.view(-1, 1)))
            x0 = xt - (ut * t.view(-1, 1))

            if randomized_layer_nb:
                out = model(xt_t, n_iter=random.randint(5, 15))
            else:
                out = model(xt_t)

            # Models flagged `predicts_x1` output D_theta(x_t, t) ≈ x1 directly
            # during training (the velocity division by (1-t) only happens at
            # eval/generation time), so the loss target is x1, not the FM
            # velocity ut.
            if getattr(model, "predicts_x1", False):
                loss = torch.mean(((out - x1)/(1 - t.view(-1, 1))) ** 2)
            else:
                loss = torch.mean((out - ut) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        elapsed  = time.perf_counter() - t0
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch+1}/{nb_epochs} — loss: {avg_loss:.4f}  ({elapsed:.1f}s)")
        write_train_time(run_dir, time.perf_counter() - train_start, epochs=epoch + 1)

        if (epoch + 1) % eval_every == 0 or epoch == nb_epochs - 1:
            model.eval()
            with torch.no_grad():
                x0_test = torch.randn(2000, DIM, device=device)
                t_span  = torch.linspace(0, 1, 101, device=device)

                if multi_iter:
                    for n_it in [5, 10, 20, 30]:
                        ode_fn = lambda t, x, ni=n_it, **kw: model(
                            torch.cat([x, t.expand(x.shape[0], 1)], dim=-1), n_iter=ni,
                        )
                        node = NeuralODE(
                            ode_fn, solver="euler",
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
                    node10 = NeuralODE(ode_fn10, solver="euler",
                                       optimizable_params=model.parameters())
                    gen = node10.trajectory(x0_test, t_span=t_span)[-1].cpu().numpy()
                    save_overview(
                        x0_test.cpu().numpy(), ref_data, gen,
                        title=f"{model_name} — epoch {epoch+1}",
                        path=os.path.join(run_dir, f"overview_epoch_{epoch+1}.png"),
                    )
                else:
                    node = NeuralODE(torch_wrapper(model), solver="euler")
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

                w2 = compute_w2(gen, ref_data)
                error_history.append((epoch + 1, w2))
                print(f"  Epoch {epoch+1}/{nb_epochs} — W2 error: {w2:.4f}")
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

    # Error (W2 to target) curve
    err_epochs = [ep for ep, _ in error_history]
    err_values = [w2 for _, w2 in error_history]
    plt.figure()
    plt.plot(err_epochs, err_values, marker="o", markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("W2 distance to target")
    plt.title(f"Training error — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "error.png"))
    plt.close()

    with open(os.path.join(run_dir, "error.txt"), "w") as f:
        f.write("epoch\tw2_error\n")
        for ep, w2 in error_history:
            f.write(f"{ep}\t{w2:.6f}\n")

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    total_time = time.perf_counter() - train_start
    write_train_time(run_dir, total_time, epochs=nb_epochs)
    print(f"  Results saved to: {run_dir}")
    return loss_history, total_params, total_time, error_history


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
    write_param_file(model, run_dir)
    # FLAT : "une estimation de vitesse" = une cascade complete x0 -> x1_hat
    write_velocity_flops(run_dir, count_velocity_flops(
        model, lambda m: m(torch.zeros(1, DIM, device=device), mode="direct")))

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

                for tag, gen in [("direct", gen_direct)]: # [("direct", gen_direct), ("lookahead", gen_lookahead)]:
                    save_scatter(
                        gen,
                        title=f"{model_name} — epoch {epoch+1} ({tag})",
                        path=os.path.join(run_dir, f"epoch_{epoch+1}_{tag}.png"),
                        ref=ref_data,
                    )
                save_overview(
                    x0_test.cpu().numpy(), ref_data, gen_direct,
                    title=f"{model_name} — epoch {epoch+1}",
                    path=os.path.join(run_dir, f"overview_direct_epoch_{epoch+1}.png"),
                )
                # save_overview(
                #     x0_test.cpu().numpy(), ref_data, gen_lookahead,
                #     title=f"{model_name} — epoch {epoch+1}",
                #     path=os.path.join(run_dir, f"overview_lookahead_epoch_{epoch+1}.png"),
                # )
                save_flat_unn_paths(
                    model, x0_test, ref_data, device,
                    title=f"{model_name} — epoch {epoch+1}",
                    path=os.path.join(run_dir, f"unn_paths_epoch_{epoch+1}.png"),
                    mode="direct",
                )
                # save_flat_unn_paths(
                #     model, x0_test, ref_data, device,
                #     title=f"{model_name} — epoch {epoch+1}",
                #     path=os.path.join(run_dir, f"unn_paths_epoch_{epoch+1}.png"),
                #     mode="velocity_step",
                # )
                save_flat_transport_field(
                    model, ref_data, device,
                    title=f"{model_name} — epoch {epoch+1}",
                    path=os.path.join(run_dir, f"transport_field_epoch_{epoch+1}.png"),
                    mode="direct",
                )
                # save_flat_transport_field(
                #     model, ref_data, device,
                #     title=f"{model_name} — epoch {epoch+1}",
                #     path=os.path.join(run_dir, f"transport_field_epoch_{epoch+1}.png"),
                #     mode="velocity_step",
                # )
                save_per_layer_updates(
                    model, ref_data, device,
                    title=f"{model_name} — epoch {epoch+1}",
                    path=os.path.join(run_dir, f"per_layer_updates_epoch_{epoch+1}.png"),
                    mode="direct"
                )
                # save_per_layer_updates(
                #     model, ref_data, device,
                #     title=f"{model_name} — epoch {epoch+1}",
                #     path=os.path.join(run_dir, f"per_layer_updates_epoch_{epoch+1}.png"),
                #     mode="velocity_step"
                # )
                # save_density_diff(
                #     gen_lookahead, ref_data,
                #     title=f"{model_name} — epoch {epoch+1} (lookahead)",
                #     path=os.path.join(run_dir, f"density_diff_epoch_{epoch+1}.png"),
                # )
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

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    total_time = time.perf_counter() - train_start
    write_train_time(run_dir, total_time, epochs=nb_epochs)
    print(f"  Results saved to: {run_dir}")
    return loss_history, total_params, total_time


def train_flat_2moons_v2(
    model,
    train_loader,
    device,
    results_dir,
    model_name,
    nb_epochs=100,
    lr=1e-3,
    randomized_layer_nb=False,
    multi_iter=False,
    w_velocity=2.0,
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
    write_param_file(model, run_dir)
    write_velocity_flops(run_dir, flops_vector_model(model, DIM, device))

    loss_history  = []
    loss_recon_h  = []
    loss_vel_h    = []
    error_history = []  # (epoch, W2 distance to target)
    train_start   = time.perf_counter()
    eval_every    = max(1, nb_epochs // 5)

    for epoch in range(nb_epochs):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_vel = 0.0
        t0 = time.perf_counter()

        for (x1_batch,) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{nb_epochs}", leave=False):
            x1 = x1_batch.to(device)
            B  = x1.shape[0]
            x0 = torch.randn(B, DIM, device=device)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            xt_t = torch.cat([xt, t.view(B, 1)], dim=-1)

            vt, x, x_traj = model(xt_t)

            loss_rec = torch.mean((vt - ut) ** 2)
            loss_vel = 0
            for k in range(len(x_traj) - 1):
                t_kp1  = model.get_schedule(device=device)[k + 1]
                x_star = (1.0 - t_kp1) * x0 + t_kp1 * x1
                loss_vel = loss_vel + F.l1_loss(x_traj[k + 1], x_star)
            loss_vel = loss_vel / (len(x_traj) - 1)
            
            loss = loss_rec + w_velocity * loss_vel

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_recon += loss_rec.item()
            total_vel += loss_vel.item()

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
                x0_test = torch.randn(2000, DIM, device=device)
                t_span  = torch.linspace(0, 1, 2, device=device)

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
                
                save_unn_paths(
                    model, train_loader, device,
                    title=f"{model_name} — epoch {epoch+1}",
                    path=os.path.join(run_dir, f"unn_paths_epoch_{epoch+1}.png")
                )

                save_vector_field(
                    model, train_loader, device,
                    title=f"{model_name} — epoch {epoch+1}",
                    path=os.path.join(run_dir, f"vector_field_epoch_{epoch+1}.png"),
                )

                w2 = compute_w2(gen, ref_data)
                error_history.append((epoch + 1, w2))
                print(f"  Epoch {epoch+1}/{nb_epochs} — W2 error: {w2:.4f}")
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

    # Error (W2 to target) curve
    err_epochs = [ep for ep, _ in error_history]
    err_values = [w2 for _, w2 in error_history]
    plt.figure()
    plt.plot(err_epochs, err_values, marker="o", markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("W2 distance to target")
    plt.title(f"Training error — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "error.png"))
    plt.close()

    with open(os.path.join(run_dir, "error.txt"), "w") as f:
        f.write("epoch\tw2_error\n")
        for ep, w2 in error_history:
            f.write(f"{ep}\t{w2:.6f}\n")

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    total_time = time.perf_counter() - train_start
    write_train_time(run_dir, total_time, epochs=nb_epochs)
    print(f"  Results saved to: {run_dir}")
    return loss_history, total_params, total_time, error_history

# ---------------------------------------------------------------------------
# FM-FLAT training  (entraînement "à la Flow Matching")
# ---------------------------------------------------------------------------

def train_flat_fm_2moons(
    model,
    train_loader,
    device,
    results_dir,
    model_name,
    nb_epochs=100,
    lr=1e-3,
    w_velocity=2.0,
):
    """
    Entraînement FM-FLAT :
      - pour chaque batch, t ~ U[0,1] et x_t = (1-t)*x0 + t*x1  (comme FM)
      - on passe x_t dans les N couches DFB avec sous-schedule adapté à [t, 1]
      - loss = MSE(x_hat_N, x1)  +  w_vel * (1/N) * Σ_k L1(x_hat_k, x_{t_{k+1}^n})

    L'inférence peut utiliser soit :
      - mode "direct" depuis x0 (identique à FLAT classique)
      - NeuralODE avec mode "fm_velocity" (champ de vitesse estimé par K pas DFB)
    """
    run_dir = os.path.join(results_dir, model_name)
    os.makedirs(run_dir, exist_ok=True)
    ref_data = torch.cat([x for (x,) in train_loader])[:2000].numpy()

    optimizer    = torch.optim.Adam(model.parameters(), lr=lr)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}\nModel : {model_name} (FM-FLAT, alpha={model.alpha})\nParams: {total_params:,}\n{'='*60}")
    write_param_file(model, run_dir)
    # FM-FLAT : la vitesse utilisee a l'inference est le mode "fm_velocity"
    write_velocity_flops(run_dir, flops_vector_model(model, DIM, device, mode="fm_velocity"))

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

            # ── Tirage FM : t ~ U[0,1] par exemple, x_t = interpolant ──────
            t   = torch.rand(B, device=device) * 1e-3 + 1e-3                                    # (B,)
            x_t = (1.0 - t[:, None]) * x0_paired + t[:, None] * x1_paired       # (B, dim)

            # ── Forward : toutes les couches depuis x_t, sous-schedule [t, 1] ──
            x_pred, x_states = model(x_t, t_start=t, return_x1_hats=True)
            x_traj = [x_t] + list(x_states)

            # sous-schedule per-exemple pour les cibles intermédiaires
            t_base = model.get_schedule(device=device)                            # (N+1,)
            t_s    = t[:, None] + (1.0 - t[:, None]) * t_base[None, :]           # (B, N+1)

            # ── Loss I : reconstruction end-to-end ──────────────────────────
            loss_recon = F.mse_loss(x_pred, x1_paired)

            # ── Loss II : supervision FLAT par couche (teacher forcing sur cibles vraies) ──
            loss_vel = torch.zeros(1, device=device)
            for k in range(K):
                t_kp1  = t_s[:, k + 1].unsqueeze(1)                              # (B, 1)
                x_star = (1.0 - t_kp1) * x0_paired + t_kp1 * x1_paired          # (B, dim)
                loss_vel = loss_vel + F.l1_loss(x_traj[k + 1], x_star)
            loss_vel = loss_vel / K

            loss = loss_recon + w_velocity * loss_vel

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
                x0_test = torch.randn(2000, DIM, device=device)
                t_span  = torch.linspace(0, 1, 2, device=device)

                # Option A : FLAT direct depuis x0 (inference sans ODE)
                gen_direct = model(x0_test).cpu().numpy()
                save_scatter(gen_direct,
                             title=f"{model_name} — epoch {epoch+1} (direct)",
                             path=os.path.join(run_dir, f"epoch_{epoch+1}_direct.png"),
                             ref=ref_data)

                # Option B : NeuralODE avec le champ de vitesse fm_velocity
                ode_fn = lambda t, x, **kw: model(
                    torch.cat([x, t * torch.ones(x.shape[0], 1, device=x.device)], dim=-1),
                    mode="fm_velocity",
                )
                node = NeuralODE(ode_fn, solver="dopri5", atol=1e-4, rtol=1e-4, optimizable_params=[])
                gen_ode = node.trajectory(x0_test, t_span=t_span)[-1].cpu().numpy()
                save_scatter(gen_ode,
                             title=f"{model_name} — epoch {epoch+1} (ODE)",
                             path=os.path.join(run_dir, f"epoch_{epoch+1}_ode.png"),
                             ref=ref_data)

                save_overview(x0_test.cpu().numpy(), ref_data, gen_ode,
                              title=f"{model_name} — epoch {epoch+1}",
                              path=os.path.join(run_dir, f"overview_epoch_{epoch+1}.png"))
            model.train()

    plt.figure()
    plt.plot(range(1, nb_epochs + 1), loss_history, marker="o", markersize=3, label="total")
    plt.plot(range(1, nb_epochs + 1), loss_recon_h,  marker="o", markersize=3, label="recon (MSE)")
    plt.plot(range(1, nb_epochs + 1), loss_vel_h,    marker="o", markersize=3, label="vel (L1, /N)")
    plt.xlabel("Epoch"); plt.ylabel("FM-FLAT loss")
    plt.title(f"Training loss — {model_name}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss.png")); plt.close()

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    total_time = time.perf_counter() - train_start
    write_train_time(run_dir, total_time, epochs=nb_epochs)
    print(f"  Results saved to: {run_dir}")
    return loss_history, total_params, total_time


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
#
# DFB and ScCP UNNs (LFO / LNO), prox L1 only, with two ablations:
#   - "K study"    : impact of the number of layers K, dual_dim fixed.
#   - "dual study" : impact of the number of features (dual_dim), K fixed.

K_STUDY_VALUES    = [3, 5, 10, 15, 20, 25]
K_STUDY_DUAL_DIM  = 16

DUAL_STUDY_VALUES = [4, 8, 16, 32, 64]
DUAL_STUDY_K      = 10


def build_experiments(device, pairs=None, div_pairs=None):
    """Build the list of experiments to run.

    If `pairs` is given (a list of (K, dual_dim) tuples), it takes over
    entirely: one DFB_UNN_L1 + one ScCP_UNN_L1 experiment per (K, dual_dim)
    pair, for both LFO and LNO, named "*_K{K}_dual{D}". This lets you fill
    in arbitrary cells of the make_grille.py K x dual_dim grid instead of
    being restricted to the K-study / dual-study crosses below.

    If `div_pairs` is given (a list of (K, dual_dim) tuples), it takes over
    instead: for each pair and each version (LFO/LNO), builds the 8 DFB_UNN_L1
    experiments covering every combination of `begin_div`, `end_div` and `pred`:
    - begin_div: whether z = x_t is divided by t (clamped away from 0)
      before the unrolled DFB iterations (a "warm start" rescaling).
    - end_div: whether the predicted velocity v_t = x - x_t is divided by
      (1-t) (clamped away from 0) before being returned. Only has an effect
      when pred="x" (it's skipped entirely when pred="v").
    - pred: "x" (the unrolled layers predict the state x, v_t = x - x_t) or
      "v" (the unrolled layers predict v_t directly, returned as-is).
    Named "DFB_UNN_L1_{version}_K{K}_dual{D}_begin{True|False}_end{True|False}_pred{x|v}".
    ScCP_UNN has no such flags, so this study only covers DFB.

    Otherwise, falls back to the two ablation studies (K_STUDY_VALUES /
    DUAL_STUDY_VALUES).
    """
    experiments = [
        dict(
            name  = "MLP_baseline",
            build = lambda: small_MLP(dim=DIM, w=64, time_varying=True).to(device),
            kwargs= {},
        ),
    ]
    
    experiments.append(dict(
        name = "FLAT_DFB_UNN_v2",
        build = lambda: FLAT_DFB_UNN_v2(
            dim=DIM, K=25, dual_dim=16, version="LNO", alpha=2.0
        ).to(device),
        kwargs= {"is_flat_v2": True},
    ))

    if div_pairs:
        for K, D in div_pairs:
            for version in ["LFO", "LNO"]:
                for begin_div in [False, True]:
                    for end_div in [False, True]:
                        for pred in ["x", "v"]:
                            experiments.append(dict(
                                name  = f"DFB_UNN_L1_{version}_K{K}_dual{D}_begin{begin_div}_end{end_div}_pred{pred}",
                                build = (lambda K=K, D=D, version=version, bd=begin_div, ed=end_div, pr=pred: DFB_UNN(
                                    dim=DIM, K=K, dual_dim=D,
                                    version=version, learned_prox=False, begin_div=bd, end_div=ed, pred=pr,
                                ).to(device)),
                                kwargs= {},
                            ))
        return experiments

    if pairs:
        for K, D in pairs:
            for version in ["LFO", "LNO"]:
                experiments.append(dict(
                    name  = f"DFB_UNN_L1_{version}_K{K}_dual{D}",
                    build = (lambda K=K, D=D, version=version: DFB_UNN(
                        dim=DIM, K=K, dual_dim=D,
                        version=version, learned_prox=False,
                    ).to(device)),
                    kwargs= {},
                ))
                experiments.append(dict(
                    name  = f"ScCP_UNN_L1_{version}_K{K}_dual{D}",
                    build = (lambda K=K, D=D, version=version: ScCP_UNN(
                        dim=DIM, K=K, dual_dim=D,
                        version=version, prox_type="l1",
                    ).to(device)),
                    kwargs= {},
                ))
        return experiments

    # ---- Study 1: impact of the number of layers K (dual_dim fixed) ----
    for K in K_STUDY_VALUES:
        for version in ["LFO", "LNO"]:
            experiments.append(dict(
                name  = f"DFB_UNN_L1_{version}_K{K}",
                build = (lambda K=K, version=version: DFB_UNN(
                    dim=DIM, K=K, dual_dim=K_STUDY_DUAL_DIM,
                    version=version, learned_prox=False,
                ).to(device)),
                kwargs= {},
            ))
            experiments.append(dict(
                name  = f"ScCP_UNN_L1_{version}_K{K}",
                build = (lambda K=K, version=version: ScCP_UNN(
                    dim=DIM, K=K, dual_dim=K_STUDY_DUAL_DIM,
                    version=version, prox_type="l1",
                ).to(device)),
                kwargs= {},
            ))

    # ---- Study 2: impact of the number of features / dual_dim (K fixed) ----
    for D in DUAL_STUDY_VALUES:
        for version in ["LFO", "LNO"]:
            experiments.append(dict(
                name  = f"DFB_UNN_L1_{version}_dual{D}",
                build = (lambda D=D, version=version: DFB_UNN(
                    dim=DIM, K=DUAL_STUDY_K, dual_dim=D,
                    version=version, learned_prox=False,
                ).to(device)),
                kwargs= {},
            ))
            experiments.append(dict(
                name  = f"ScCP_UNN_L1_{version}_dual{D}",
                build = (lambda D=D, version=version: ScCP_UNN(
                    dim=DIM, K=DUAL_STUDY_K, dual_dim=D,
                    version=version, prox_type="l1",
                ).to(device)),
                kwargs= {},
            ))

    return experiments


def merge_shard_summaries(results_dir, num_shards):
    """Read summary_shard{0..num_shards-1}.txt from results_dir, merge them
    into summary.txt, and regenerate the ablation-study plots."""
    summary = []
    for shard_id in range(num_shards):
        path = os.path.join(results_dir, f"summary_shard{shard_id}.txt")
        if not os.path.exists(path):
            print(f"  [warn] missing {path}, skipping")
            continue
        with open(path) as f:
            next(f)  # header
            for line in f:
                name, n_params, final_loss, final_error, train_time, status = line.rstrip("\n").split("\t")
                summary.append((name, float(final_loss), float(final_error),
                                 int(n_params), float(train_time), status))

    summary_path = os.path.join(results_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("model_name\tn_params\tfinal_loss\tfinal_w2_error\ttrain_time_s\tstatus\n")
        for name, final_loss, final_error, n_params, train_time, status in summary:
            f.write(f"{name}\t{n_params}\t{final_loss:.6f}\t{final_error:.6f}\t{train_time:.1f}\t{status}\n")
    print(f"Merged {len(summary)} run(s) from {num_shards} shard(s) into: {summary_path}")

    plot_ablation_studies(summary, results_dir)


def plot_ablation_studies(summary, results_dir):
    """From the run summary, build comparison plots of final loss / W2 error
    vs. the swept hyperparameter (K or dual_dim), one line per model+version,
    for the two ablations defined in build_experiments()."""
    by_metric = {"final_loss": {}, "final_w2": {}}
    k_pattern    = re.compile(r"^(DFB_UNN_L1|ScCP_UNN_L1)_(LFO|LNO)_K(\d+)$")
    dual_pattern = re.compile(r"^(DFB_UNN_L1|ScCP_UNN_L1)_(LFO|LNO)_dual(\d+)$")

    studies = {"K": {}, "dual": {}}
    for name, final_loss, final_error, n_params, train_time, status in summary:
        if status != "OK":
            continue
        m = k_pattern.match(name)
        if m:
            model, version, x = m.group(1), m.group(2), int(m.group(3))
            studies["K"].setdefault((model, version), []).append((x, final_loss, final_error))
            continue
        m = dual_pattern.match(name)
        if m:
            model, version, x = m.group(1), m.group(2), int(m.group(3))
            studies["dual"].setdefault((model, version), []).append((x, final_loss, final_error))

    labels = {"K": "Number of layers K", "dual": "Feature / dual dimension"}
    for study_name, data in studies.items():
        if not data:
            continue
        for metric_idx, metric_name in [(1, "loss"), (2, "w2_error")]:
            plt.figure()
            for (model, version), points in sorted(data.items()):
                points = sorted(points, key=lambda p: p[0])
                xs = [p[0] for p in points]
                ys = [p[metric_idx] for p in points]
                plt.plot(xs, ys, marker="o", label=f"{model}_{version}")
            plt.xlabel(labels[study_name])
            plt.ylabel("Final training loss" if metric_name == "loss" else "Final W2 error")
            plt.title(f"Impact of {labels[study_name].lower()} on final {metric_name}")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"study_{study_name}_{metric_name}.png"))
            plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Flow Matching: 2-moons from 1 Gaussian.")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--results-dir", type=str, default="results_2moons")
    parser.add_argument("--skip",        type=str, default="")
    parser.add_argument("--only",        type=str, default="")
    parser.add_argument("--batch-size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--device",      type=str, default="",
                         help="Torch device to use, e.g. 'cuda:0', 'cuda:1', 'cpu'. "
                              "Defaults to cuda if available, else cpu.")
    parser.add_argument("--num-shards",  type=int, default=1,
                         help="Split the experiment list into this many shards "
                              "(use with --shard-id to run shards in parallel, e.g. on separate GPUs).")
    parser.add_argument("--shard-id",    type=int, default=0,
                         help="Index (0-based) of the shard to run, in [0, num_shards).")
    parser.add_argument("--merge-shards", action="store_true",
                         help="Merge summary_shard*.txt files in --results-dir into summary.txt "
                              "and (re)generate the ablation-study plots, then exit.")
    parser.add_argument("--pairs", type=str, default="",
                         help="Comma-separated list of K:dual_dim pairs to test instead of the "
                              "default K-study/dual-study ablations, e.g. '5:16,10:32,20:64'. "
                              "Runs DFB_UNN_L1 + ScCP_UNN_L1, LFO + LNO, for each pair.")
    parser.add_argument("--div-pairs", type=str, default="",
                         help="Comma-separated list of K:dual_dim pairs to compare DFB_UNN's "
                              "4 begin_div/end_div combinations on, e.g. '10:32,20:32'. "
                              "Takes priority over --pairs if both are given.")
    args = parser.parse_args()

    if args.merge_shards:
        merge_shard_summaries(args.results_dir, args.num_shards)
        return

    def _parse_pairs(s):
        result = []
        for chunk in s.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            k_str, d_str = chunk.split(":")
            result.append((int(k_str), int(d_str)))
        return result

    pairs     = _parse_pairs(args.pairs) if args.pairs else None
    div_pairs = _parse_pairs(args.div_pairs) if args.div_pairs else None

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.results_dir, exist_ok=True)
    train_loader = get_moons_loader(batch_size=args.batch_size)

    experiments = build_experiments(device, pairs=pairs, div_pairs=div_pairs)
    if args.only:
        experiments = [e for e in experiments if args.only in e["name"]]
    if args.skip:
        experiments = [e for e in experiments if args.skip not in e["name"]]

    if args.num_shards > 1:
        assert 0 <= args.shard_id < args.num_shards, "--shard-id must be in [0, num_shards)"
        experiments = experiments[args.shard_id::args.num_shards]
        print(f"\nShard {args.shard_id}/{args.num_shards} on {device}: "
              f"{len(experiments)} experiment(s) assigned.")

    print(f"\n{len(experiments)} experiment(s) to run.\n")
    print("List of experiments:")
    for exp in experiments:
        print(f"  - {exp['name']}")

    summary = []
    for i, exp in enumerate(experiments, 1):
        name = exp["name"]
        print(f"\n[{i}/{len(experiments)}] {name}")

        kwargs      = exp.get("kwargs", {})
        is_flat     = kwargs.pop("is_flat",    False)
        is_flat_v2  = kwargs.pop("is_flat_v2", False)
        is_flat_fm  = kwargs.pop("is_flat_fm", False)

        try:
            model = exp["build"]()
            if is_flat_fm:
                losses, n_params, train_time = train_flat_fm_2moons(
                    model        = model,
                    train_loader = train_loader,
                    device       = device,
                    results_dir  = args.results_dir,
                    model_name   = name,
                    nb_epochs    = args.epochs,
                    **kwargs
                )
                final_error = float("nan")
            elif is_flat:
                losses, n_params, train_time = train_flat_2moons(
                    model        = model,
                    train_loader = train_loader,
                    device       = device,
                    results_dir  = args.results_dir,
                    model_name   = name,
                    nb_epochs    = args.epochs,
                    **kwargs
                )
                final_error = float("nan")
            elif is_flat_v2:
                losses, n_params, train_time, errors = train_flat_2moons_v2(
                    model        = model,
                    train_loader = train_loader,
                    device       = device,
                    results_dir  = args.results_dir,
                    model_name   = name,
                    nb_epochs    = args.epochs,
                    w_velocity    = 10.0,
                    **kwargs
                )
                final_error = errors[-1][1] if errors else float("nan")
            else:
                model = exp["build"]()
                losses, n_params, train_time, errors = train_2moons(
                    model        = model,
                    train_loader = train_loader,
                    device       = device,
                    results_dir  = args.results_dir,
                    model_name   = name,
                    nb_epochs    = args.epochs,
                    **exp.get("kwargs", {}),
                )
                final_error = errors[-1][1] if errors else float("nan")
            summary.append((name, losses[-1], final_error, n_params, train_time, "OK"))
        except Exception as exc:
            import traceback; traceback.print_exc()
            summary.append((name, float("nan"), float("nan"), 0, 0.0, str(exc)))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, final_loss, final_error, n_params, train_time, status in summary:
        print(f"  {name:<40} params={n_params:>8,}  final_loss={final_loss:.4f}  "
              f"final_w2={final_error:.4f}  time={train_time:6.0f}s  [{status}]")

    summary_name = "summary.txt" if args.num_shards <= 1 else f"summary_shard{args.shard_id}.txt"
    summary_path = os.path.join(args.results_dir, summary_name)
    with open(summary_path, "w") as f:
        f.write("model_name\tn_params\tfinal_loss\tfinal_w2_error\ttrain_time_s\tstatus\n")
        for name, final_loss, final_error, n_params, train_time, status in summary:
            f.write(f"{name}\t{n_params}\t{final_loss:.6f}\t{final_error:.6f}\t{train_time:.1f}\t{status}\n")
    print(f"\nSummary saved to: {summary_path}")

    if args.num_shards > 1:
        print(f"\nThis was shard {args.shard_id}/{args.num_shards} — once all shards have finished, run:\n"
              f"  python run_2moons.py --merge-shards --results-dir {args.results_dir} --num-shards {args.num_shards}\n"
              f"to combine the summaries and generate the ablation-study plots.")
    else:
        plot_ablation_studies(summary, args.results_dir)


if __name__ == "__main__":
    main()
