"""Replicate Kamb & Ganguli Fig. 4a on a TRAINED ConvScCP checkpoint:
effective receptive field of the center output pixel as a function of flow time t.

Measures  |d x1_pred(center) / d x_t|  averaged over noised real-MNIST inputs,
for a grid of t. In this repo's Flow-Matching convention t=0 is noise (high noise)
and t=1 is data (low noise): Kamb's coarse-to-fine shows a LARGE RF near t=0
shrinking toward t=1.

Real digits are normalized to [-1,1] to match the model's training space.

Usage:
  python erf_vs_time_checkpoint.py --ckpt <path> --K 6 --ic 128 --kernel 3 --tag k3
"""
import argparse, torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torchvision, torchvision.transforms as T
from models.architectures import ConvScCP_UNN

dev  = "cuda" if torch.cuda.is_available() else "cpu"
S, DIM   = 28, 784
TS       = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]
N_SAMP   = 32
CENTER   = (S // 2) * S + (S // 2)


def load_model(ckpt, K, ic, kernel):
    m = ConvScCP_UNN(dim=DIM, K=K, internal_channel=ic, use_Unet="l1",
                     version="LNO", use_checkpoint=False, w_bias=True,
                     in_channels=1, img_size=S, kernel_size=kernel).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    m.train()          # ConvScCP returns x1_pred in train() mode; no norm/dropout in L1 variant
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def real_digits(n):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])   # -> [-1,1]
    ds = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tf)
    return torch.stack([ds[i][0] for i in range(n)]).view(n, DIM).to(dev)


def radial_stats(m):
    c = S // 2
    ys, xs = np.mgrid[0:S, 0:S]
    d = np.sqrt((ys - c) ** 2 + (xs - c) ** 2)
    w = m / (m.sum() + 1e-12)
    return float((w * d).sum()), float(m[(np.abs(ys - c) > 8) | (np.abs(xs - c) > 8)].sum() / (m.sum() + 1e-12))


def erf_at_t(model, x1, t):
    acc = torch.zeros(S, S)
    for i in range(x1.shape[0]):
        x0 = torch.randn(1, DIM, device=dev)
        xt = ((1 - t) * x0 + t * x1[i:i+1]).detach().requires_grad_(True)
        inp = torch.cat([xt, torch.full((1, 1), t, device=dev)], dim=1)
        out = model(inp)
        model.zero_grad(set_to_none=True)
        out[0, CENTER].backward()
        acc += xt.grad.detach().abs().view(S, S).cpu()
    return (acc / x1.shape[0]).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/temp-4/ConvScCP_UNN_L1_LNO/model.pt")
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--ic", type=int, default=64)
    p.add_argument("--kernel", type=int, default=9)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    tag = ("_" + args.tag) if args.tag else ""

    model = load_model(args.ckpt, args.K, args.ic, args.kernel)
    x1 = real_digits(N_SAMP)
    print(f"ckpt={args.ckpt}  K={args.K} ic={args.ic} kernel={args.kernel}  (t=0 bruit / t=1 data)")
    maps, stats = {}, {}
    for t in TS:
        m = erf_at_t(model, x1, t)
        er, o17 = radial_stats(m); maps[t] = m; stats[t] = (er, o17)
        print(f"t={t:.2f}:  eff.radius = {er:5.2f} px   mass outside 17x17 = {100*o17:5.1f}%")

    fig, ax = plt.subplots(1, len(TS), figsize=(2.6 * len(TS), 3.0))
    for j, t in enumerate(TS):
        mn = maps[t] / maps[t].max()
        ax[j].imshow(np.log10(mn + 1e-4), cmap="magma", vmin=-4, vmax=0)
        er, o17 = stats[t]
        ax[j].set_title(f"t={t}\nR={er:.1f}px  out17={100*o17:.0f}%", fontsize=8)
        ax[j].set_xticks([]); ax[j].set_yticks([])
    plt.suptitle(f"Fig.4a — RF effectif vs t  (K={args.K}, ic={args.ic}, kernel {args.kernel}x{args.kernel})",
                 fontsize=10)
    plt.tight_layout(); plt.savefig(f"erf_vs_time{tag}.png", dpi=140, bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(5, 3.3))
    ax2.plot(TS, [stats[t][0] for t in TS], "o-")
    ax2.axhline(10.6, color="gray", ls="--", lw=1, label="RF uniforme (~global)")
    ax2.set_xlabel("t  (0 = bruit, 1 = data)"); ax2.set_ylabel("rayon effectif (px)")
    ax2.set_title(f"RF effectif vs t  (kernel {args.kernel}, K={args.K})"); ax2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"erf_radius_vs_time{tag}.png", dpi=140, bbox_inches="tight")
    print(f"saved -> erf_vs_time{tag}.png , erf_radius_vs_time{tag}.png")


if __name__ == "__main__":
    main()
