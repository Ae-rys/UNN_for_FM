# -*- coding: utf-8 -*-
"""
make_grille.py
Build a grid of generated-distribution images: rows = number of layers K,
columns = number of features / dual_dim, one collage per (model, version)
combo, stacked into a single PNG per epoch.

Usage
-----
    python make_grille.py
        Build the K x dual_dim grids (default behaviour).

    python make_grille.py --trajectory FOLDER [--n-steps 1000] [--method heun]
        Load FOLDER's "model.pt" checkpoint (FOLDER given relative to this
        script, e.g. "results_2moons_DFB_ScCP_L1_small_K20/ScCP_UNN_L1_LNO_K32")
        and plot its generated distribution at several points of the t in
        [0, 1] trajectory (explicit Euler/Heun integration). Saves
        "trajectory_{method}_{n_steps}steps.png" inside FOLDER.

Edit RESULT_DIRS / EPOCHS below to change which experiment folders and
which training epochs are used. K and dual_dim are read directly from each
experiment's "parametres.txt" (written by run_2moons.py / write_param_file),
so no per-directory hyperparameter bookkeeping is needed here anymore.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Config — edit this to add/remove the result folders to pull plots from.
# Just list the directory names (relative to this script); every
# subfolder containing a "parametres.txt" with K + dual_dim is picked up
# automatically.
# ---------------------------------------------------------------------------
RESULT_DIRS = [
    # "2moons_ot_fixed",
    # "results_2moons_DFB_ScCP_L1",
    # "results_2moons_DFB_ScCP_L1_small",
    # "results_2moons_DFB_ScCP_L1_K20",
    # "results_2moons_pairs",
    # "results_2moons_test_division",
    # "temp",
    "2moons_ot_fixed_x_pred_epochs_200"
]

# Epochs to render one collage for (must match existing "epoch_{N}.png" files).
EPOCHS = [200]

# Output file: "{OUT_PREFIX}.png" for the last epoch, "{OUT_PREFIX}_epoch{N}.png" otherwise.
OUT_PREFIX = "grille_K_vs_features"

# Rendering quality.
CELL_SIZE_IN = 3.2     # figure size (inches) per grid cell
FIG_DPI      = 220     # savefig dpi for each per-(model,version) block
INTERP       = "bilinear"  # imshow interpolation when upscaling the small source PNGs

# ---------------------------------------------------------------------------
# Discovery: read "parametres.txt" from every experiment subfolder of
# RESULT_DIRS and build, for each (model, version), a {(K, dual_dim): folder} map.
# ---------------------------------------------------------------------------

def parse_param_file(path):
    params = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            params[key] = value
    return params


def discover_points():
    """Returns {(model, version): {(K, dual_dim): abs_folder_path}}."""
    points = {}

    for dirname in RESULT_DIRS:
        dir_path = os.path.join(HERE, dirname)
        if not os.path.isdir(dir_path):
            print(f"  [warn] missing results dir, skipping: {dir_path}")
            continue

        for name in sorted(os.listdir(dir_path)):
            folder = os.path.join(dir_path, name)
            param_path = os.path.join(folder, "parametres.txt")
            if not os.path.isdir(folder) or not os.path.exists(param_path):
                continue

            params = parse_param_file(param_path)
            model   = params.get("model_class")
            version = params.get("version")
            K       = params.get("K") or params.get("N")
            D       = params.get("dual_dim")
            if model is None or version is None or K is None or D is None:
                continue

            points.setdefault((model, version), {})[(int(K), int(D))] = folder

    return points


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_block(model, version, cell_map, rows, cols, epoch, out_path):
    fig, axes = plt.subplots(
        len(rows), len(cols),
        figsize=(CELL_SIZE_IN * len(cols), CELL_SIZE_IN * len(rows)),
        squeeze=False,  # always get a 2D (nrows, ncols) array, even for 1x1/1xN/Nx1 grids
    )
    fig.suptitle(f"{model} — {version}", fontsize=16, y=0.995)

    for i, K in enumerate(rows):
        for j, D in enumerate(cols):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            folder = cell_map.get((K, D))
            img_path = os.path.join(folder, f"epoch_{epoch}.png") if folder else None
            if img_path is None or not os.path.exists(img_path):
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=14, color="gray",
                        transform=ax.transAxes)
            else:
                ax.imshow(mpimg.imread(img_path), interpolation=INTERP)
            if i == 0:
                ax.set_title(f"dual_dim={D}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"K={K}", fontsize=11)
            for spine in ax.spines.values():
                spine.set_visible(True)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def make_collage(epoch, points):
    combos = sorted(points.keys())
    if not combos:
        print("  [warn] no experiments discovered, nothing to plot.")
        return

    all_K = sorted({K for cell_map in points.values() for (K, D) in cell_map})
    all_D = sorted({D for cell_map in points.values() for (K, D) in cell_map})

    tmp_paths = []
    for model, version in combos:
        out_path = os.path.join(HERE, f"_tmp_grille_{model}_{version}.png")
        make_block(model, version, points[(model, version)], all_K, all_D, epoch, out_path)
        tmp_paths.append(out_path)

    # All blocks are rendered with the same figsize/dpi, so they already have
    # identical pixel dimensions — no resizing (and its blurring) needed.
    imgs = [Image.open(p) for p in tmp_paths]
    w, h = imgs[0].size

    n_cols = 2
    n_rows = (len(imgs) + n_cols - 1) // n_cols
    collage = Image.new("RGB", (n_cols * w, n_rows * h), "white")
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, n_cols)
        collage.paste(im, (c * w, r * h))

    suffix = "" if epoch == max(EPOCHS) else f"_epoch{epoch}"
    out = os.path.join(HERE, f"{OUT_PREFIX}{suffix}.png")
    collage.save(out)
    print(f"Saved: {out}")

    for p in tmp_paths:
        os.remove(p)


def make_grids():
    points = discover_points()
    print(f"Discovered {len(points)} (model, version) combo(s):")
    for (model, version), cell_map in sorted(points.items()):
        print(f"  - {model} {version}: {len(cell_map)} point(s) -> {sorted(cell_map.keys())}")

    for epoch in EPOCHS:
        make_collage(epoch, points)


# ---------------------------------------------------------------------------
# Division study: 2x4 grid (begin_div x [end_div, pred]) per (model, version,
# K, dual_dim) point that has begin_div/end_div/pred fields in
# parametres.txt (currently only DFB_UNN, see run_2moons.py --div-pairs).
# Note: end_div has no effect when pred="v" (skipped in DFB_UNN.forward),
# so those columns are expected to look identical across end_div for a
# given begin_div — that's a property of the model, not a plotting bug.
# ---------------------------------------------------------------------------

DIV_ROWS = [False, True]  # begin_div
DIV_COLS = [(False, "x"), (True, "x"), (False, "v"), (True, "v")]  # (end_div, pred)


def discover_division_points():
    """Returns {(model, version, K, dual_dim): {(begin_div, end_div, pred): folder}},
    restricted to points that have at least 2 begin_div/end_div/pred variants
    (plain K/dual_dim studies always carry begin_div=False/end_div=False/pred="x"
    too, but as a single point — not an actual division study)."""
    points = {}

    for dirname in RESULT_DIRS:
        dir_path = os.path.join(HERE, dirname)
        if not os.path.isdir(dir_path):
            continue

        for name in sorted(os.listdir(dir_path)):
            folder = os.path.join(dir_path, name)
            param_path = os.path.join(folder, "parametres.txt")
            if not os.path.isdir(folder) or not os.path.exists(param_path):
                continue

            params = parse_param_file(param_path)
            if "begin_div" not in params and "end_div" not in params and "pred" not in params:
                continue

            model   = params.get("model_class")
            version = params.get("version")
            K       = params.get("K") or params.get("N")
            D       = params.get("dual_dim")
            if model is None or version is None or K is None or D is None:
                continue

            begin_div = params.get("begin_div", "False") == "True"
            end_div   = params.get("end_div", "False") == "True"
            pred      = params.get("pred", "x")
            key = (model, version, int(K), int(D))
            points.setdefault(key, {})[(begin_div, end_div, pred)] = folder

    # Drop points that only have a single (begin_div, end_div, pred) variant —
    # those are regular K/dual_dim study runs, not an actual division study.
    return {key: variants for key, variants in points.items() if len(variants) > 1}


def make_division_block(model, version, K, D, variant_map, epoch, out_path):
    fig, axes = plt.subplots(
        len(DIV_ROWS), len(DIV_COLS),
        figsize=(CELL_SIZE_IN * len(DIV_COLS), CELL_SIZE_IN * len(DIV_ROWS)),
        squeeze=False,
    )
    fig.suptitle(f"{model} — {version} — K={K} dual_dim={D}", fontsize=14, y=0.995)

    for i, begin_div in enumerate(DIV_ROWS):
        for j, (end_div, pred) in enumerate(DIV_COLS):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            folder = variant_map.get((begin_div, end_div, pred))
            img_path = os.path.join(folder, f"epoch_{epoch}.png") if folder else None
            if img_path is None or not os.path.exists(img_path):
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=14, color="gray",
                        transform=ax.transAxes)
            else:
                ax.imshow(mpimg.imread(img_path), interpolation=INTERP)
            if i == 0:
                ax.set_title(f"end_div={end_div}\npred={pred}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"begin_div={begin_div}", fontsize=11)
            for spine in ax.spines.values():
                spine.set_visible(True)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def make_division_collage(epoch, division_points):
    keys = sorted(division_points.keys())
    if not keys:
        return

    tmp_paths = []
    for model, version, K, D in keys:
        out_path = os.path.join(HERE, f"_tmp_grille_div_{model}_{version}_K{K}_dual{D}.png")
        make_division_block(model, version, K, D, division_points[(model, version, K, D)], epoch, out_path)
        tmp_paths.append(out_path)

    imgs = [Image.open(p) for p in tmp_paths]
    w, h = imgs[0].size

    collage = Image.new("RGB", (w, h * len(imgs)), "white")
    for idx, im in enumerate(imgs):
        collage.paste(im, (0, idx * h))

    suffix = "" if epoch == max(EPOCHS) else f"_epoch{epoch}"
    out = os.path.join(HERE, f"grille_division{suffix}.png")
    collage.save(out)
    print(f"Saved: {out}")

    for p in tmp_paths:
        os.remove(p)


def make_division_grids():
    division_points = discover_division_points()
    if not division_points:
        print("No begin_div/end_div/pred division-study data found, skipping division grids.")
        return

    print(f"Discovered {len(division_points)} division-study point(s):")
    for key, variant_map in sorted(division_points.items()):
        print(f"  - {key}: {sorted(variant_map.keys())}")

    for epoch in EPOCHS:
        make_division_collage(epoch, division_points)


# ---------------------------------------------------------------------------
# --trajectory: plot a single checkpoint's generated distribution along the
# t in [0, 1] ODE trajectory (explicit Euler/Heun), using parametres.txt to
# get the architecture right automatically.
# ---------------------------------------------------------------------------

MODEL_CLASS_TO_TYPE = {"DFB_UNN": "DFB", "ScCP_UNN": "ScCP"}


def plot_trajectory(folder, n_steps=1000, method="heun", n_snapshots=8,
                     n_samples=2000, device=None, out_path=None):
    """Simple wrapper around run_2moons.plot_checkpoint_trajectory: give it
    just an experiment folder (relative to this script, or absolute), it
    reads parametres.txt + model.pt from there and infers everything else
    (model type, K, dual_dim, version). Source distribution is always a
    single standard Gaussian, matching run_2moons.py's training."""
    import torch
    from run_2moons import plot_checkpoint_trajectory

    folder = folder if os.path.isabs(folder) else os.path.join(HERE, folder)
    param_path = os.path.join(folder, "parametres.txt")
    model_pt_path = os.path.join(folder, "model.pt")
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"no parametres.txt in {folder}")
    if not os.path.exists(model_pt_path):
        raise FileNotFoundError(f"no model.pt in {folder} (checkpoint wasn't saved for this run)")

    params      = parse_param_file(param_path)
    model_class = params["model_class"]
    model_type  = MODEL_CLASS_TO_TYPE.get(model_class)
    if model_type is None:
        raise ValueError(f"don't know how to rebuild a {model_class!r} (only DFB_UNN / ScCP_UNN supported)")

    K        = int(params.get("K") or params["N"])
    dual_dim = int(params["dual_dim"])
    version  = params["version"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if out_path is None:
        out_path = os.path.join(folder, f"trajectory_{method}_{n_steps}steps.png")

    print(f"  {model_class} {version} K={K} dual_dim={dual_dim}, device={device}")
    plot_checkpoint_trajectory(
        model_pt_path=model_pt_path,
        model_type=model_type,
        K=K, dual_dim=dual_dim, version=version,
        device=device,
        n_steps=n_steps, method=method, n_snapshots=n_snapshots,
        n_samples=n_samples,
        out_path=out_path,
    )
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=str, default=None,
                         help="Experiment folder (relative to this script) to plot the "
                              "Euler/Heun ODE trajectory for, instead of building the grids.")
    parser.add_argument("--n-steps",     type=int, default=1000)
    parser.add_argument("--method",      type=str, default="heun", choices=["euler", "heun"])
    parser.add_argument("--n-snapshots", type=int, default=8)
    args = parser.parse_args()

    if args.trajectory:
        plot_trajectory(args.trajectory, n_steps=args.n_steps,
                         method=args.method, n_snapshots=args.n_snapshots)
    else:
        make_grids()
        make_division_grids()


if __name__ == "__main__":
    main()
