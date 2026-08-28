"""Effective receptive field (ERF) of the ConvScCP-LNO cascade as a function of
convolution kernel size, à la Kamb & Ganguli Fig. 4a.

We reimplement the ScCP-LNO iteration with a PARAMETRIZED kernel size k and
padding p=k//2 (both hardcoded to 9 / 4 in architectures.py). The ERF is the
map  |d output(center pixel) / d input pixel(x')|  averaged over random inputs
and random weight inits. It is a purely architectural quantity (no training
needed): it says which input pixels can influence a given output pixel.

Run:  python erf_kernel_analysis.py
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

torch.backends.mkldnn.enabled = False
dev = "cuda" if torch.cuda.is_available() else "cpu"

S       = 28      # MNIST spatial size
IC      = 64      # internal_channel (as in the trained model)
K       = 20      # unrolled iterations (as in generated_32.png)
KERNELS = [3, 5, 7, 9]
T_VAL   = 0.5     # flow time at which the ERF is probed
N_INIT  = 6       # random weight inits to average over
N_INPUT = 8       # random inputs per init


def kaiming(*shape):
    w = torch.empty(*shape); nn.init.kaiming_normal_(w, nonlinearity="relu"); return w


def sigma_max_power_iter(W, u, n_iter=5):
    # step size is a constant w.r.t. autograd (matches architectures.py)
    with torch.no_grad():
        mat = W.view(W.shape[0], -1)
        for _ in range(n_iter):
            v = F.normalize(mat.t() @ u, dim=0)
            u = F.normalize(mat @ v, dim=0)
        return (u @ mat @ v).detach(), u


class ScCPLayer(nn.Module):
    """One accelerated Chambolle-Pock step, LNO, kernel k, padding k//2, L1 prox."""
    def __init__(self, ic, k):
        super().__init__()
        self.pad = k // 2                                   # <-- padding tracks kernel
        self.W = nn.Parameter(kaiming(ic, 1, k, k))
        self.register_buffer("_su", F.normalize(torch.randn(ic), dim=0))
        self.Wb = nn.Parameter(torch.zeros(ic))
        self.radius = nn.Sequential(nn.Linear(1, 8), nn.SiLU(), nn.Linear(8, 1))

    def spectral_norm(self):
        s, self._su = sigma_max_power_iter(self.W, self._su)
        return s

    def forward(self, x, u, z, t, tau, sigma, alpha):
        primal = x - tau * F.conv_transpose2d(u, self.W, padding=self.pad)
        x_next = (primal + tau * z) / (1 + tau)
        y      = x_next + alpha * (x_next - x)
        du     = sigma * F.conv2d(y, self.W, bias=self.Wb, padding=self.pad)
        r      = F.softplus(self.radius(t)).view(t.shape[0], 1, 1, 1)
        u_next = torch.clamp(u + du, -r, r)                 # L1 dual prox
        return x_next, u_next


class ScCPCascade(nn.Module):
    def __init__(self, ic, k, K):
        super().__init__()
        self.log_tau = nn.Parameter(torch.full((K,), -0.5))
        self.layers  = nn.ModuleList([ScCPLayer(ic, k) for _ in range(K)])

    def forward(self, z, t):
        taus = F.softplus(self.log_tau)
        x = z.clone(); u = torch.zeros(z.shape[0], self.layers[0].W.shape[0],
                                       z.shape[2], z.shape[3], device=z.device)
        for k, layer in enumerate(self.layers):
            tau = taus[k]
            alpha = (1 + 2 * tau).pow(-0.5)
            sigma = 0.99 / (tau * layer.spectral_norm() ** 2)
            x, u = layer(x, u, z, t, tau, sigma, alpha)
        return x                                            # predicted x1


def erf_map(kernel):
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


def radial_stats(m):
    c = S // 2
    ys, xs = np.mgrid[0:S, 0:S]
    d = np.sqrt((ys - c) ** 2 + (xs - c) ** 2)
    w = m / (m.sum() + 1e-12)
    eff_radius = float((w * d).sum())                       # mean influence distance
    # fraction of gradient mass OUTSIDE a 17x17 window (paper's ResNet RF cap)
    outside_17 = float(m[(np.abs(ys - c) > 8) | (np.abs(xs - c) > 8)].sum() / (m.sum() + 1e-12))
    return eff_radius, outside_17


def main():
    print(f"S={S} IC={IC} K={K} t={T_VAL}  (avg over {N_INIT} inits x {N_INPUT} inputs)")
    maps, stats = {}, {}
    for k in KERNELS:
        m = erf_map(k)
        er, o17 = radial_stats(m)
        maps[k] = m; stats[k] = (er, o17)
        print(f"kernel {k}x{k} (pad {k//2}):  eff.radius = {er:5.2f} px   "
              f"mass outside 17x17 = {100*o17:5.1f}%")

    fig, ax = plt.subplots(1, len(KERNELS), figsize=(3.2 * len(KERNELS), 3.4))
    for j, k in enumerate(KERNELS):
        m = maps[k]; mn = m / m.max()
        ax[j].imshow(np.log10(mn + 1e-4), cmap="magma", vmin=-4, vmax=0)
        er, o17 = stats[k]
        ax[j].set_title(f"{k}x{k}  (pad {k//2})\neff.R={er:.1f}px  out17={100*o17:.0f}%", fontsize=9)
        ax[j].axhline(S//2, color="cyan", lw=0.4, alpha=0.5)
        ax[j].axvline(S//2, color="cyan", lw=0.4, alpha=0.5)
        ax[j].set_xticks([]); ax[j].set_yticks([])
    plt.suptitle(f"ERF of ConvScCP-LNO center pixel, K={K}  (log10 |d out/d in|)", fontsize=11)
    plt.tight_layout()
    out = "erf_by_kernel.png"; plt.savefig(out, dpi=140, bbox_inches="tight")
    print("saved ->", out)


if __name__ == "__main__":
    main()
