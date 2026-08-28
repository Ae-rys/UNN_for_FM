"""Overfitting / memorization test for a trained ConvScCP checkpoint.

Idea (à la Kamb): the Ideal Score machine ALWAYS outputs a memorized training
example. If the trained model's samples match the IS output on matched seeds,
the model memorizes. We build an IS machine in the SAME Flow-Matching framework
as the model (ideal FM velocity over the empirical training set), integrate it
with the SAME dopri5 solver and the SAME initial noise, and compare.

We also do the gold-standard memorization check: nearest training-set neighbor
of each generated sample (L2), vs the same statistic for held-out TEST images.

All in the model's data space: MNIST normalized to [-1, 1].

Outputs: overfit_grid.png  (model | IS-FM | nearest train, per seed)
         overfit_metrics.txt

Run:  python overfit_test_convsccp.py
"""
import torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torchvision, torchvision.transforms as T
from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper
from models.architectures import ConvScCP_UNN

import argparse
dev  = "cuda" if torch.cuda.is_available() else "cpu"
DIM, S = 784, 28
N_SEED   = 16       # seeds for metrics
N_SHOW   = 8        # rows in the figure


def r2(a, b):
    a = a - a.mean(); a = a / (a.norm() + 1e-12)
    b = b - b.mean(); b = b / (b.norm() + 1e-12)
    return float((a * b).sum())


def load_model(ckpt, K, ic, kernel):
    m = ConvScCP_UNN(dim=DIM, K=K, internal_channel=ic, use_Unet="l1",
                     version="LNO", use_checkpoint=False, w_bias=True,
                     in_channels=1, img_size=S, kernel_size=kernel).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    m.eval()
    return m


def train_set():
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])   # -> [-1,1]
    ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=tf)
    X = torch.stack([ds[i][0] for i in range(len(ds))]).view(len(ds), DIM)
    return X.to(dev)


def test_set(n):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    ds = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tf)
    return torch.stack([ds[i][0] for i in range(n)]).view(n, DIM).to(dev)


class IdealFMVelocity(torch.nn.Module):
    """Ideal Flow-Matching velocity for the empirical training set (memorizes).
    Same interface as the model: forward(xt_t) with xt_t = cat([x, t])."""
    def __init__(self, X1):
        super().__init__()
        self.register_buffer("X1", X1)      # (Ntr, DIM) in [-1,1]

    def forward(self, xt_t):
        x  = xt_t[:, :DIM]                   # (b, DIM)
        ts = xt_t[0, DIM].clamp(0.0, 1.0)    # scalar: t is uniform across the batch during ODE
        omt = (1 - ts).clamp(min=1e-2)       # (1-t)
        # posterior over training images:  w_j ∝ exp(-||x - t X1_j||^2 / (2 (1-t)^2))
        d2 = torch.cdist(x, ts * self.X1) ** 2           # (b, Ntr)
        logw = -d2 / (2 * omt ** 2)
        logw = logw - logw.max(dim=1, keepdim=True).values
        w = torch.softmax(logw, dim=1)                   # (b, Ntr)
        Ex1 = w @ self.X1                                # (b, DIM)
        return (Ex1 - x) / (1 - ts).clamp(min=0.05)      # E[ut|x_t], denom clamp like the model


@torch.no_grad()
def sample(vel_module, x0):
    node = NeuralODE(torch_wrapper(vel_module), solver="dopri5", atol=1e-5, rtol=1e-5)
    traj = node.trajectory(x0, t_span=torch.linspace(0, 1, 2, device=dev))
    return traj[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/temp-4/ConvScCP_UNN_L1_LNO/model.pt")
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--ic", type=int, default=64)
    p.add_argument("--kernel", type=int, default=9)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    global CKPT, K, IC
    CKPT, K, IC = args.ckpt, args.K, args.ic
    tag = ("_" + args.tag) if args.tag else ""

    torch.manual_seed(0)
    model = load_model(args.ckpt, args.K, args.ic, args.kernel)
    X1 = train_set()
    print(f"train set: {tuple(X1.shape)}  range [{X1.min():.2f},{X1.max():.2f}]")

    x0 = torch.randn(N_SEED, DIM, device=dev)            # SAME seeds for both
    with torch.no_grad():
        gen = sample(model, x0.clone()).clamp(-1, 1)     # model samples
    is_mod = IdealFMVelocity(X1).to(dev)
    ideal = sample(is_mod, x0.clone()).clamp(-1, 1)      # IS-FM samples (memorized)

    # nearest training neighbor of each model sample
    d = torch.cdist(gen, X1)                             # (N_SEED, Ntr)
    nn_dist, nn_idx = d.min(dim=1)
    nn_img = X1[nn_idx]

    # baseline: nearest-train distance for held-out TEST images
    Xte = test_set(N_SEED)
    nn_test = torch.cdist(Xte, X1).min(dim=1).values

    r_is = [r2(gen[i], ideal[i]) for i in range(N_SEED)]
    r_nn = [r2(gen[i], nn_img[i]) for i in range(N_SEED)]

    lines = []
    lines.append(f"checkpoint: {CKPT}  (K={K}, ic={IC})   N_seed={N_SEED}")
    lines.append(f"median r2(model, IS-FM)          = {np.median(r_is):.3f}   "
                 "(haut => memorise)")
    lines.append(f"median r2(model, nearest train)  = {np.median(r_nn):.3f}")
    lines.append(f"median L2(model sample -> train) = {nn_dist.median():.2f}")
    lines.append(f"median L2(TEST image  -> train)  = {nn_test.median():.2f}   "
                 "(reference d'un vrai voisin non-memorise)")
    lines.append(f"ratio sample/test nearest-dist   = {(nn_dist.median()/nn_test.median()):.2f}   "
                 "(<<1 => samples collent au train => overfit)")
    txt = "\n".join(lines)
    print(txt); open(f"overfit_metrics{tag}.txt", "w").write(txt + "\n")

    # figure
    def dn(v): return (v.view(S, S).cpu().numpy() + 1) / 2
    fig, ax = plt.subplots(N_SHOW, 3, figsize=(6, 2.0 * N_SHOW))
    for i in range(N_SHOW):
        for j, (img, ttl) in enumerate([
                (gen[i], "ConvScCP"),
                (ideal[i], f"IS-FM\nr2={r_is[i]:.2f}"),
                (nn_img[i], f"+proche train\nr2={r_nn[i]:.2f} L2={nn_dist[i]:.1f}")]):
            ax[i, j].imshow(dn(img), cmap="gray", vmin=0, vmax=1); ax[i, j].axis("off")
            if i == 0: ax[i, j].set_title(ttl, fontsize=9)
            elif j > 0: ax[i, j].set_title(ttl.split("\n")[-1], fontsize=8)
    plt.suptitle(f"Test overfitting ConvScCP {args.tag} — même bruit initial", fontsize=11)
    plt.tight_layout(); plt.savefig(f"overfit_grid{tag}.png", dpi=130, bbox_inches="tight")
    print(f"saved -> overfit_grid{tag}.png , overfit_metrics{tag}.txt")


if __name__ == "__main__":
    main()
