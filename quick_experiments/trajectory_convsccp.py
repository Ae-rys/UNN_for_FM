# -*- coding: utf-8 -*-
"""
trajectory_convsccp.py
Save the INTERMEDIATE images of the FM ODE: the successive x_t for t = 0, 0.1, ... 1.

Target question: does ConvScCP denoise progressively (x_t cleans up little by little)
or does it pass through states that make no sense? Two complementary views:

  "x_t" row      : the current ODE state (what is actually integrated).
  "x1_pred" row  : the digit the network THINKS it is generating at time t, i.e.
                   x_t + (1-t) * v(x_t, t). For an x-pred model (ConvScCP_UNN in
                   eval returns v = (x1_pred - z)/clamp(1-t, 0.05)), this is exactly
                   the network's internal output. This is the view that answers the
                   question: if x1_pred is noise/garbage at small t then suddenly
                   becomes a digit, the model "decides" late; if it looks like a
                   blurry digit from the start and refines, it denoises progressively.

The model config (K, internal_channel, version, w_bias, kernel...) is auto-detected
from the state_dict, via infer_config() of generate_digits.py.

Outputs (in --outdir, default <ckpt dir>/trajectory/):
  trajectory_xt.png      grid rows = samples, columns = t
  trajectory_x1pred.png  same for the predicted x1
  trajectory_both.png    both views stacked, sample by sample
  velocity_norm.png      ||v(x_t,t)|| and ||x_t|| as a function of t
  trajectory.pt          raw tensors {ts, xt, x1pred, v} (n_steps+1, n, 1, 28, 28)
  frames/                (option --save-frames) each image as a separate PNG

Usage
-----
    CUDA_VISIBLE_DEVICES=1 python trajectory_convsccp.py \
        --ckpt results/grid50/ConvScCP_k3_K6_ic128_L1_LNO/model.pt --n 6 --steps 10

    # fine trajectory + explicit solver (no adaptive)
    python trajectory_convsccp.py --ckpt <path> --steps 20 --solver euler

    # only show 4 unroll layers in the (t, k) grid -> smaller image
    python trajectory_convsccp.py --ckpt <path> --n-layers 4
"""
import argparse
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper

from models.architectures import ConvScCP_UNN, ConvDFB_UNN, SmallUNetX1
from generate_digits import infer_config


def build_model(ckpt, device):
    """Load the state_dict and rebuild the right model (config auto-detected)."""
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # ConvScCP et ConvDFB partagent "layers.k.W_weight" : ce qui les separe est le pas
    # primal, parametre GLOBAL chez ScCP (log_tau / log_tau0) et absent chez DFB
    # (tau par couche en LFO, 1.99/sigma^2 en LNO).
    is_sccp = "log_tau" in sd or "log_tau0" in sd
    if "layers.0.W_weight" in sd and is_sccp:                # ConvScCP_UNN
        cfg = infer_config(sd)
        model = ConvScCP_UNN(dim=cfg["dim"], K=cfg["K"], internal_channel=cfg["internal_channel"],
                             use_Unet=cfg["use_Unet"], version=cfg["version"],
                             w_bias=cfg["w_bias"], in_channels=cfg["in_channels"],
                             img_size=cfg["img_size"], kernel_size=cfg["kernel"],
                             prox_w=cfg["prox_w"]).to(device)
        cfg["algo"] = "ScCP"
    elif "layers.0.W_weight" in sd:                          # ConvDFB_UNN (28x28, 1 canal, 9x9)
        cfg = infer_config(sd)
        cfg["algo"] = "DFB"
        model = ConvDFB_UNN(dim=cfg["dim"], K=cfg["K"], internal_channel=cfg["internal_channel"],
                            use_Unet=cfg["use_Unet"], version=cfg["version"],
                            w_bias=cfg["w_bias"], prox_w=cfg["prox_w"]).to(device)
    elif "inc.conv1.weight" in sd:                           # baseline SmallUNet
        base_ch, in_ch = sd["inc.conv1.weight"].shape[0], sd["inc.conv1.weight"].shape[1]
        cfg = dict(dim=in_ch * 28 * 28, internal_channel=base_ch, img_size=28,
                   in_channels=in_ch, K="-", version="SmallUNet", algo="UNet")
        model = SmallUNetX1(in_channels=in_ch, out_channels=in_ch, base_ch=base_ch).to(device)
    else:
        raise ValueError(f"Unrecognized model type from keys: {list(sd)[:5]}")
    # strict=True: a missing/extra key means the auto-detected config does NOT match
    # the checkpoint (e.g. wrong LNO/LFO version). With strict=False the model would
    # load only halfway, with weights left random, and produce wrong figures without
    # any warning -> we prefer to crash.
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, cfg


@torch.no_grad()
def rollout(model, n, cfg, device, steps, solver, seed):
    """Integrate the ODE from t=0 to t=1, recording the state at the requested steps+1 times.

    Returns ts (S,), xt (S, n, C, H, W), v (S, n, C, H, W), x1pred (S, n, C, H, W).
    x1_pred = x_t + clamp(1-t, min=0.05) * v: the exact inverse of the x-pred -> velocity
    conversion done in ConvScCP_UNN.forward() in eval.
    """
    torch.manual_seed(seed)
    dim, C, S = cfg["dim"], cfg["in_channels"], cfg["img_size"]
    x0 = torch.randn(n, dim, device=device)
    ts = torch.linspace(0, 1, steps + 1, device=device)

    node = (NeuralODE(torch_wrapper(model), solver="dopri5", atol=1e-5, rtol=1e-5)
            if solver == "dopri5" else
            NeuralODE(torch_wrapper(model), solver=solver))
    traj = node.trajectory(x0, t_span=ts)                     # (steps+1, n, dim)
    print(f"  ODE integrated ({solver}, {steps + 1} recorded times)")

    # velocity and predicted x1 at each recorded time (one forward per time)
    vs, x1s = [], []
    for i, t in enumerate(ts):
        xt = traj[i]
        xt_t = torch.cat([xt, t.expand(n, 1)], dim=-1)
        v = model(xt_t)
        vs.append(v)
        x1s.append(xt + torch.clamp(1 - t, min=0.05) * v)
    shape = (-1, n, C, S, S)
    return (ts.cpu(), traj.reshape(shape).cpu(),
            torch.stack(vs).reshape(shape).cpu(), torch.stack(x1s).reshape(shape).cpu())


@torch.no_grad()
def iterates_at_times(model, xt, ts, device):
    """INTERNAL iterates of the ScCP unroll (primal variable x^(k)), at each time t.

    For each recorded t, we redo an instrumented forward on the REAL x_t of the ODE
    trajectory -> we see what the unrolled algorithm builds inside ONE step.
    Returns (S_t, K+1, n, C, H, W): x^(0)=x_t (input z), ..., x^(K)= network's x1_pred.
    """
    out = []
    for j, t in enumerate(ts):
        x = xt[j].flatten(1).to(device)                       # (n, dim)
        xt_t = torch.cat([x, t.to(device).expand(x.shape[0], 1)], dim=-1)
        _, it = model(xt_t, return_iterates=True)             # (K+1, n, C, H, W)
        out.append(it.cpu())
    return torch.stack(out)                                   # (S_t, K+1, n, C, H, W)


def select_layer_indices(n_available, n_layers):
    """Pick which unroll-layer columns k to display in the (t, k) grid.

    n_available is K+1 (columns k=0..K). If n_layers is None or >= n_available, keep
    every column. Otherwise return n_layers evenly-spaced indices that ALWAYS include
    the input k=0 and the final output k=K, so a smaller grid still shows both ends.
    """
    if n_layers is None or n_layers >= n_available:
        return list(range(n_available))
    n_layers = max(2, n_layers)                               # need at least the two ends
    idx = torch.linspace(0, n_available - 1, n_layers).round().long().tolist()
    return sorted(set(idx))


def _prep_panel(a, norm):
    """Prepare a thumbnail (H, W) for imshow according to the scaling mode. Returns
    (data, vmin, vmax, real_amplitude). The internal iterates of the unroll go very
    far outside [-1,1] (factor ~1000 mid-unroll), so the fixed scale saturates to
    white/black: hence the per-image and symlog modes.
    """
    amp = float(a.abs().max())
    if norm == "fixed":                       # absolute scale [-1,1] (comparable across thumbnails)
        return a.clamp(-1, 1), -1.0, 1.0, amp
    if norm == "symlog":                      # signed log: keeps the sign, compresses the extremes
        a = torch.sign(a) * torch.log1p(a.abs())
        v = float(a.abs().max()) or 1.0
        return a, -v, v, amp
    # "per-image": symmetric around 0, robust bound (99.5th percentile) -> a few
    # extreme pixels do not crush the contrast of the rest of the image.
    v = float(torch.quantile(a.abs().flatten().float(), 0.995)) or 1.0
    return a.clamp(-v, v), -v, v, amp


def grid_iterates(it, ts, i, path, title, norm="per-image", layer_idx=None):
    """it: (S_t, K+1, n, C, H, W) -> rows = t, columns = k, for sample i.

    layer_idx: optional list of k columns to display (see select_layer_indices).
               None means every column. Fewer columns = smaller image.
    norm: "per-image" (default, each thumbnail has its own symmetric scale),
          "symlog" (signed log, thumbnail-specific scale),
          "fixed" ([-1,1] absolute like the other grids).
    Each thumbnail is annotated with its REAL amplitude max|x|: the normalization
    makes the structure visible, the annotation prevents forgetting the scale.
    """
    S, Kp1 = it.shape[0], it.shape[1]
    cols = layer_idx if layer_idx is not None else list(range(Kp1))
    ncol = len(cols)
    fig, axes = plt.subplots(S, ncol, figsize=(0.95 * ncol, 1.15 * S), squeeze=False)
    for r in range(S):
        for c, k in enumerate(cols):
            ax = axes[r, c]
            data, vmin, vmax, amp = _prep_panel(it[r, k, i, 0], norm)
            ax.imshow(data, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if norm != "fixed":               # real amplitude, otherwise the scale is invisible
                ax.set_xlabel(f"{amp:.3g}", fontsize=5, labelpad=1)
            if r == 0:
                ax.set_title(f"k={k}", fontsize=7)
            if c == 0:
                ax.set_ylabel(f"t={ts[r]:.2f}", fontsize=7)
    sub = {"per-image": "per-thumbnail scale (99.5th percentile, symmetric)",
           "symlog": "signed-log per-thumbnail scale",
           "fixed": "absolute scale [-1,1]"}[norm]
    #fig.suptitle(f"{title}\n{sub} — the number under each thumbnail = real max|x|", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  -> {path}")


def iterates_amplitude_plot(it, ts, path, title, algo="ScCP"):
    """max|x^(k)| as a function of k, one curve per t: shows the unroll's excursion
    (the iterates leave [-1,1] by a factor ~1000 before contracting back)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in range(it.shape[0]):
        amp = it[r].flatten(2).abs().max(dim=-1).values.mean(1)      # (K+1,)
        ax.plot(range(it.shape[1]), amp, "o-", ms=3, label=f"t={ts[r]:.2f}")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="image scale [-1,1]")
    ax.set_xlabel(f"iteration k of the {algo} unroll"); ax.set_ylabel("max |x^(k)|")
    ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
   # ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  -> {path}")


def iterates_conv_plot(it, ts, path, title, algo="ScCP"):
    """||x^(k) - x^(K)|| as a function of k, one curve per t: does the unroll CONVERGE
    within K iterations, or is it still moving at the last one?"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in range(it.shape[0]):
        d = (it[r] - it[r, -1]).flatten(2).norm(dim=-1).mean(1)   # (K+1,)
        ax.plot(range(it.shape[1]), d, "o-", ms=3, label=f"t={ts[r]:.2f}")
    ax.set_xlabel(f"iteration k of the {algo} unroll"); ax.set_ylabel("||x^(k) - x^(K)||")
    ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    #ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  -> {path}")


def grid(imgs, ts, path, title, row_label="sample"):
    """imgs: (S, n, C, H, W) -> grid rows = samples, columns = t."""
    S, n = imgs.shape[0], imgs.shape[1]
    fig, axes = plt.subplots(n, S, figsize=(1.1 * S, 1.15 * n), squeeze=False)
    for i in range(n):
        for j in range(S):
            ax = axes[i, j]
            ax.imshow(imgs[j, i, 0].clamp(-1, 1), cmap="gray", vmin=-1, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"t={ts[j]:.2f}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{row_label} {i}", fontsize=8)
    #fig.suptitle(title, fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  -> {path}")


def grid_both(xt, x1p, ts, path, title):
    """Per sample: one x_t row then one x1_pred row, to compare the two."""
    S, n = xt.shape[0], xt.shape[1]
    fig, axes = plt.subplots(2 * n, S, figsize=(1.1 * S, 1.15 * 2 * n), squeeze=False)
    for i in range(n):
        for row, (data, name) in enumerate([(xt, "x_t"), (x1p, "x1_pred")]):
            for j in range(S):
                ax = axes[2 * i + row, j]
                ax.imshow(data[j, i, 0].clamp(-1, 1), cmap="gray", vmin=-1, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                if 2 * i + row == 0:
                    ax.set_title(f"t={ts[j]:.2f}", fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"#{i} {name}", fontsize=7)
    #fig.suptitle(title, fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  -> {path}")


def norms_plot(xt, v, ts, path, title):
    """||x_t|| and ||v(x_t,t)|| (batch means): a velocity that blows up near t=1 or an
    erratic ||x_t|| shows up here before showing up to the eye."""
    nx = xt.flatten(2).norm(dim=-1).mean(1)
    nv = v.flatten(2).norm(dim=-1).mean(1)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ts, nx, "o-", label="||x_t||")
    ax.plot(ts, nv, "s-", label="||v(x_t,t)||")
    ax.set_xlabel("t"); ax.set_ylabel("L2 norm (batch mean)")
    #ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  -> {path}")


def main():
    p = argparse.ArgumentParser(description="Save the intermediate x_t of a ConvScCP's FM ODE.")
    p.add_argument("--ckpt", type=str, default="results/temp-4/ConvScCP_UNN_L1_LNO/model.pt", help="checkpoint of the ConvScCP model to evaluate")
    p.add_argument("--n", type=int, default=6, help="number of trajectories (rows)")
    p.add_argument("--steps", type=int, default=10, help="number of intervals: t = 0, 1/steps, ..., 1")
    p.add_argument("--solver", type=str, default="dopri5", choices=["dopri5", "euler", "rk4"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="", help="default: <ckpt dir>/trajectory/")
    p.add_argument("--no-iterates", dest="iterates", action="store_false",
                   help="do not save the internal iterates x^(k) of the ScCP unroll")
    p.add_argument("--n-iter-samples", type=int, default=2,
                   help="number of samples for which to plot the (t, k) grid")
    p.add_argument("--n-layers", type=int, default=None,
                   help="number of unroll layers (k columns) to show in the (t, k) grid; "
                        "None = all. Endpoints k=0 and k=K are always kept -> smaller image")
    p.add_argument("--iter-norm", type=str, default="per-image",
                   choices=["per-image", "symlog", "fixed"],
                   help="scale of the (t,k) grid: per-image (default), symlog, or fixed [-1,1]")
    p.add_argument("--save-frames", action="store_true", help="also save each image as a separate PNG")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg = build_model(args.ckpt, device)
    print(f"Auto-detected config: {cfg}")
    print(f"Checkpoint loaded: {args.ckpt}  ({sum(q.numel() for q in model.parameters()):,} params)")

    ts, xt, v, x1p = rollout(model, args.n, cfg, device, args.steps, args.solver, args.seed)

    outdir = args.outdir or os.path.join(os.path.dirname(args.ckpt), "trajectory")
    os.makedirs(outdir, exist_ok=True)
    algo = cfg.get("algo", "ScCP")
    tag = (f"{os.path.basename(os.path.dirname(args.ckpt))} — {algo} K={cfg['K']} "
           f"ic={cfg['internal_channel']} {cfg['version']} — {args.solver}")

    grid(xt,  ts, os.path.join(outdir, "trajectory_xt.png"),
         f"x_t along the ODE — {tag}")
    grid(x1p, ts, os.path.join(outdir, "trajectory_x1pred.png"),
         f"x1_pred = x_t + (1-t)·v(x_t,t) — {tag}")
    grid_both(xt, x1p, ts, os.path.join(outdir, "trajectory_both.png"),
              f"x_t (top) vs x1_pred (bottom) per sample — {tag}")
    norms_plot(xt, v, ts, os.path.join(outdir, "velocity_norm.png"),
               f"norms along the trajectory — {tag}")

    saved = {"ts": ts, "xt": xt, "x1pred": x1p, "v": v, "cfg": cfg, "ckpt": args.ckpt}

    # ---- internal iterates of the ScCP unroll (k axis, in addition to the t axis) ----
    if args.iterates and isinstance(model, (ConvScCP_UNN, ConvDFB_UNN)):
        it = iterates_at_times(model, xt, ts, device)
        print(f"  internal iterates: {tuple(it.shape)}  (t, k, sample, C, H, W)")
        print(f"  iterates amplitude: max|x^(k)| = {it.abs().amax(dim=(0,2,3,4,5)).tolist()}")
        layer_idx = select_layer_indices(it.shape[1], args.n_layers)
        print(f"  layers shown in the (t,k) grid: k = {layer_idx}")
        for i in range(min(args.n_iter_samples, args.n)):
            grid_iterates(it, ts, i, os.path.join(outdir, f"iterates_sample{i}.png"),
                          f"internal x^(k) of the {algo} unroll — sample #{i} — {tag}",
                          norm=args.iter_norm, layer_idx=layer_idx)
        iterates_conv_plot(it, ts, os.path.join(outdir, "iterates_convergence.png"),
                           f"convergence of the internal unroll — {tag}", algo=algo)
        iterates_amplitude_plot(it, ts, os.path.join(outdir, "iterates_amplitude.png"),
                                f"amplitude of the internal iterates — {tag}", algo=algo)
        saved["iterates"] = it
    elif args.iterates:
        print("  [info] neither ConvScCP_UNN nor ConvDFB_UNN -> no internal iterates")

    torch.save(saved, os.path.join(outdir, "trajectory.pt"))
    print(f"  -> {os.path.join(outdir, 'trajectory.pt')}  (xt/x1pred/v: {tuple(xt.shape)})")

    if args.save_frames:
        d = os.path.join(outdir, "frames")
        os.makedirs(d, exist_ok=True)
        for j, t in enumerate(ts):
            for i in range(args.n):
                for data, name in [(xt, "xt"), (x1p, "x1pred")]:
                    plt.imsave(os.path.join(d, f"{name}_sample{i:02d}_t{t:.2f}.png"),
                               data[j, i, 0].clamp(-1, 1), cmap="gray", vmin=-1, vmax=1)
        print(f"  -> {d}/ ({(args.steps + 1) * args.n * 2} PNG)")

    print(f"\nDone. Everything is in {outdir}/")


if __name__ == "__main__":
    main()
