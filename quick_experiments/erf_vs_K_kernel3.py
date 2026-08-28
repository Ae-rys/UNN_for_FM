"""Effective receptive field (ERF) of the ConvScCP-LNO cascade at FIXED kernel
3x3 (padding=1), as a function of the number of unrolled iterations K.

Companion to erf_kernel_analysis.py (which fixes K and varies the kernel). Here
we test whether K acts as a genuine LOCALITY knob when the kernel is small: with
3x3 convs the per-iteration spread is +-1 px, so the receptive field should grow
roughly linearly with K and only become global for large K. This is the regime
in which K plays the role of the P scale of the LS/ELS machines (Kamb Fig. 4).

Reuses the parametrized cascade from erf_kernel_analysis.py.

Run:  python erf_vs_K_kernel3.py
"""
import torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from erf_kernel_analysis import ScCPCascade, radial_stats, S, IC, T_VAL, dev

KERNEL  = 3
KS      = [5, 10, 20, 40, 80]     # unrolled-iteration counts to sweep
N_INIT  = 6
N_INPUT = 8


def erf_map(kernel, K):
    acc = torch.zeros(S, S)
    c = S // 2
    for _ in range(N_INIT):
        model = ScCPCascade(IC, kernel, K).to(dev).eval()
        for _ in range(N_INPUT):
            z = torch.randn(1, 1, S, S, device=dev, requires_grad=True)
            t = torch.full((1, 1), T_VAL, device=dev)
            out = model(z, t)
            model.zero_grad(set_to_none=True)
            out[0, 0, c, c].backward()
            acc += z.grad.detach().abs()[0, 0].cpu()
    return (acc / (N_INIT * N_INPUT)).numpy()


def main():
    print(f"kernel {KERNEL}x{KERNEL} (pad {KERNEL//2})  S={S} IC={IC} t={T_VAL}  "
          f"(avg over {N_INIT} inits x {N_INPUT} inputs)")
    maps, stats = {}, {}
    for K in KS:
        m = erf_map(KERNEL, K)
        er, o17 = radial_stats(m)
        maps[K] = m; stats[K] = (er, o17)
        print(f"K={K:3d}:  eff.radius = {er:5.2f} px   mass outside 17x17 = {100*o17:5.1f}%")

    fig, ax = plt.subplots(1, len(KS), figsize=(3.2 * len(KS), 3.4))
    for j, K in enumerate(KS):
        mn = maps[K] / maps[K].max()
        ax[j].imshow(np.log10(mn + 1e-4), cmap="magma", vmin=-4, vmax=0)
        er, o17 = stats[K]
        ax[j].set_title(f"K={K}\neff.R={er:.1f}px  out17={100*o17:.0f}%", fontsize=9)
        ax[j].axhline(S//2, color="cyan", lw=0.4, alpha=0.5)
        ax[j].axvline(S//2, color="cyan", lw=0.4, alpha=0.5)
        ax[j].set_xticks([]); ax[j].set_yticks([])
    plt.suptitle(f"ERF of ConvScCP-LNO center pixel, kernel {KERNEL}x{KERNEL}, "
                 f"vs iterations K  (log10 |d out/d in|)", fontsize=11)
    plt.tight_layout()
    plt.savefig("erf_vs_K_kernel3.png", dpi=140, bbox_inches="tight")
    print("saved -> erf_vs_K_kernel3.png")

    # eff. radius vs K curve
    fig2, ax2 = plt.subplots(figsize=(5, 3.4))
    ers = [stats[K][0] for K in KS]
    ax2.plot(KS, ers, "o-")
    ax2.axhline(10.6, color="gray", ls="--", lw=1, label="RF uniforme (~global)")
    ax2.set_xlabel("K (itérations dépliées)"); ax2.set_ylabel("rayon effectif (px)")
    ax2.set_title(f"ConvScCP-LNO kernel {KERNEL}x{KERNEL}: RF vs K")
    ax2.legend(fontsize=8); plt.tight_layout()
    plt.savefig("erf_radius_vs_K_kernel3.png", dpi=140, bbox_inches="tight")
    print("saved -> erf_radius_vs_K_kernel3.png")


if __name__ == "__main__":
    main()
