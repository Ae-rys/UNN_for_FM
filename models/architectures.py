# -*- coding: utf-8 -*-
"""
architectures.py
All neural network building blocks and UNN model definitions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchcfm.models.unet import UNetModel
import numpy as np

torch.backends.mkldnn.enabled = False

def sigma_max(W: torch.Tensor) -> torch.Tensor:
    """Largest singular value of a weight tensor (works for 2-D and 4-D conv kernels)."""
    if W.dim() == 4:
        mat = W.view(W.shape[0], -1)
    else:
        mat = W
    return torch.linalg.matrix_norm(mat, ord=2)


def sigma_max_power_iter(W: torch.Tensor, u: torch.Tensor, n_iter: int = 3):
    """
    Approximate largest singular value via power iteration.
    O(n_iter * m * n) vs O(m * n * min(m,n)) for full SVD.
    u: persistent unit vector buffer of shape (m,), updated and returned each call.
    Returns (sigma_approx, new_u).
    """
    if W.dim() == 4:
        mat = W.view(W.shape[0], -1)
    else:
        mat = W
    with torch.no_grad():
        for _ in range(n_iter):
            v = F.normalize(mat.T @ u, dim=0)
            u = F.normalize(mat @ v, dim=0)
    return (u @ mat @ v).detach(), u

def kaiming_init(*shape):
    """Kaiming-normal initialized tensor of the given shape (fan_in, relu)."""
    w = torch.empty(*shape)
    nn.init.kaiming_normal_(w, a=0, mode='fan_in', nonlinearity='relu')
    return w


def compute_delta(u_traj):
    """Dual-trajectory length delta_t = sum_k ||u^[k+1] - u^[k]||, per example.

    u_traj: list of K+1 tensors of shape (B, dual_dim), as returned by
    a UNN's forward(..., return_dual_traj=True).
    Returns a tensor of shape (B,).
    """
    delta = torch.zeros(u_traj[0].shape[0], device=u_traj[0].device)
    for k in range(len(u_traj) - 1):
        delta = delta + torch.norm(u_traj[k + 1] - u_traj[k], dim=-1)
    return delta

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def soft_threshold(z, T):
    # T: (B,) — reshape to broadcast against z of any rank (B, *)
    T_bc = T.view(T.shape[0], *([1] * (z.dim() - 1)))
    return torch.sign(z) * torch.maximum(abs(z) - T_bc, torch.zeros_like(z))


def proj_l_inf(x, radius):
    """Projection onto the L∞ ball of radius `radius`."""
    return torch.clamp(x, -radius, radius)


def prox_l1_via_frame(x2d, W, Wt, lam):
    """Proximity operator of lam * ||W(x)||_1 at x2d, for a tight frame W/Wt."""
    Px = W(x2d)
    p = soft_threshold(Px, lam)
    return x2d + Wt(p - W(x2d))


# ---------------------------------------------------------------------------
# Small MLP
# ---------------------------------------------------------------------------

class small_MLP(nn.Module):
    def __init__(self, dim, w, time_varying=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + (1 if time_varying else 0), w),
            nn.ReLU(),
            nn.Linear(w, w),
            nn.ReLU(),
            nn.Linear(w, dim),
        )

    def forward(self, x):
        return self.net(x)


# Alias used by DFB_UNN / DiFB_UNN
MLP = small_MLP


class L1ProxFlat(nn.Module):
    """Dual L1 prox for flat models: L∞-ball projection with a learned radius r(t).

    prox_{(μ‖·‖₁)*}(u) = clip(u, -r(t), r(t))  where r(t) = softplus(MLP(t)).
    Interface matches small_MLP(time_varying=True): takes cat([u, t]) → u.

    warmstart_scale=True: multiplies radius by (1-t)/t.clamp(0.1), matching the
    correction magnitude required when using z = xt/t as DFB warm start.
    """
    def __init__(self, dim, w=32, warmstart_scale=False):
        super().__init__()
        self.dim = dim
        self.warmstart_scale = warmstart_scale
        self.time_scaling = nn.Sequential(
            nn.Linear(1, w),
            nn.SiLU(),
            nn.Linear(w, 1),
        )

    def forward(self, u_t):
        u      = u_t[:, :self.dim]
        t      = u_t[:, self.dim:]                           # (B, 1)
        radius = F.softplus(self.time_scaling(t))            # (B, 1), broadcasts over dim
        if self.warmstart_scale:
            # correction needed ∝ (1-t)/t when warm-starting DFB from z = xt/t
            radius = radius * (1.0 - t) / t.clamp(min=1e-1)
        return torch.clamp(u, -radius, radius)


class L1ProxConv(nn.Module):
    """Dual L1 prox for conv models: L∞-ball projection with a learned radius r(t).

    prox_{(μ‖·‖₁)*}(u) = clip(u, -r(t), r(t))  where r(t) = softplus(MLP(t)).
    Interface matches DoubleConvTime: forward(u, t) where u is (B, C, H, W).

    If `channels` is given, r(t) is predicted **per dual channel** instead of a
    single global scalar — this is still exactly the prox of a (per-channel
    weighted) L1 norm, ‖u‖_{1,w} = sum_c w_c ‖u_c‖_1, whose dual ball is a box
    with one radius w_c per channel. Strictly more expressive than the shared
    scalar (which is the special case w_c constant), without leaving the L1
    prox family.
    """
    def __init__(self, w=32, channels=None):
        super().__init__()
        self.channels = channels
        out_dim = channels if channels is not None else 1
        self.time_scaling = nn.Sequential(
            nn.Linear(1, w),
            nn.SiLU(),
            nn.Linear(w, out_dim),
        )

    def forward(self, u, t):
        out = self.time_scaling(t)                                         # (B, out_dim)
        if self.channels is not None:
            shape = (out.shape[0], -1, *([1] * (u.dim() - 2)))
        else:
            shape = (out.shape[0], *([1] * (u.dim() - 1)))                 # (B, 1, ..., 1)
        r_bc = F.softplus(out).view(*shape)
        return torch.clamp(u, -r_bc, r_bc)


class SiLUProxConv(nn.Module):
    """Prox APPRIS pointwise à base de SiLU, conditionné en temps (FiLM).

    Remplace le prox l1 (clamp dur de L1ProxConv) par une non-linéarité douce
    SiLU, tout en restant **strictement pointwise** (convs 1x1) : ne change NI le
    champ récepteur NI l'équivariance -> seule la non-linéarité change vs le l1.
    Sert à tester si c'est le prox l1 (non lisse) qui gêne la convergence (ex. OT).

    Forme résiduelle : prox ~ identité + correction, pour démarrer stable dans le
    déroulé. Le temps t module (scale, shift) par canal caché (FiLM).
    Interface : forward(u, t) avec u:(B,C,H,W), t:(B,1) — identique à L1ProxConv.
    """
    def __init__(self, channels, hidden=None, w=16):
        super().__init__()
        hidden = hidden if hidden is not None else channels
        self.time_mlp = nn.Sequential(               # t -> (scale, shift) par canal caché
            nn.Linear(1, w), nn.SiLU(), nn.Linear(w, 2 * hidden))
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)   # pointwise
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)   # pointwise
        self.act = nn.SiLU()
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)      # démarre = identité

    def forward(self, u, t):                          # u:(B,C,H,W), t:(B,1)
        gamma, beta = self.time_mlp(t).chunk(2, dim=1)                      # (B, hidden) chacun
        h = self.fc1(u)
        h = self.act(gamma[:, :, None, None] * h + beta[:, :, None, None])  # FiLM(t) + SiLU
        return u + self.fc2(h)                         # résiduel


class UNetProxConv(nn.Module):
    """Wraps torchcfm UNetModel as a convolutional prox for SharedConv* models.

    Called as prox(u, t) where u: (B, C, H, W) and t: (B, 1).
    Passes u directly to UNetModel with time conditioning.
    """
    def __init__(self, in_channels, img_size=28, num_channels=32, num_res_blocks=1):
        super().__init__()
        self.unet = UNetModel(
            dim=(in_channels, img_size, img_size),
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
        )

    def forward(self, u, t):
        return self.unet(t.squeeze(-1), u)


# ---------------------------------------------------------------------------
# UNet building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Standard two-convolution block."""
    def __init__(self, in_ch, out_ch, embed_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, embed_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(embed_dim, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(4, embed_dim)
        self.gn2 = nn.GroupNorm(4, out_ch)
        self.silu = nn.SiLU()

    def forward(self, x):
        h = self.silu(self.gn1(self.conv1(x)))
        h = self.silu(self.gn2(self.conv2(h)))
        return h


class SmallUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_ch=32):
        super().__init__()
        self.time_scaling = nn.Sequential(
            nn.Linear(in_channels, base_ch),
            nn.SiLU(),
            nn.Linear(base_ch, base_ch),
        )
        self.inc   = DoubleConv(in_channels, base_ch, base_ch)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch, base_ch * 2, base_ch * 2))
        self.bot   = DoubleConv(base_ch * 2, base_ch * 2, base_ch * 2)
        self.up1   = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1  = DoubleConv(base_ch * 2, base_ch, base_ch)
        self.outc  = nn.Conv2d(base_ch, out_channels, kernel_size=1)
        self.in_channels = in_channels

    def forward(self, xt_t):
        x = xt_t[..., :-1]
        t = xt_t[..., -1:]
        batch_size = x.shape[0]
        x_img = x.view(batch_size, self.in_channels, 28, 28)
        if t.dim() > 2:
            t = t.squeeze(-1)
        t_emb = self.time_scaling(t).view(batch_size, -1, 1, 1)
        x1    = self.inc(x_img) + t_emb
        x2    = self.down1(x1)
        x_bot = self.bot(x2)
        x_up  = self.up1(x_bot)
        x_dec = self.dec1(torch.cat([x_up, x1], dim=1))
        out   = self.outc(x_dec)
        return out.view(batch_size, -1)


class SmallUNetX1(SmallUNet):
    """SmallUNet strictement identique, mais paramétrisation x1-pred (comme
    MinimalResNetFM) au lieu de vitesse : training -> prédit x1 ; eval -> vitesse
    (x1_pred - z)/(1-t). Contrôle pour isoler l'effet de la paramétrisation
    (x1 vs vitesse) sur le collapse OT, à RECEPTIVE FIELD global identique.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predicts_x1 = True

    def forward(self, xt_t):
        z = xt_t[..., :-1]                                   # x_t courant (B,dim)
        t = xt_t[..., -1:]                                   # (B,1)
        out = super().forward(xt_t)                          # x1_pred (B,dim)
        if self.training:
            return out
        return (out - z) / torch.clamp(1 - t, min=0.00005)  # vitesse (B,dim)


class UNet(nn.Module):
    #Bigger UNet, to get better results
    def __init__(self, in_channels=1, out_channels=1, base_ch=64):
        super().__init__()
        self.time_scaling = nn.Sequential(
            nn.Linear(in_channels, base_ch),
            nn.SiLU(),
            nn.Linear(base_ch, base_ch),
        )
        self.inc   = DoubleConv(in_channels, base_ch, base_ch)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch, base_ch * 2, base_ch * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_ch * 2, base_ch * 4, base_ch * 4))
        self.bot   = DoubleConv(base_ch * 4, base_ch * 4, base_ch * 4)
        self.up1   = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec1  = DoubleConv(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up2   = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec2  = DoubleConv(base_ch * 2, base_ch, base_ch)
        self.outc  = nn.Conv2d(base_ch, out_channels, kernel_size=1)
        self.in_channels = in_channels

    def forward(self, xt_t):
        x = xt_t[..., :-1]
        t = xt_t[..., -1:]
        batch_size = x.shape[0]
        x_img = x.view(batch_size, self.in_channels, 28, 28)
        if t.dim() > 2:
            t = t.squeeze(-1)
        t_emb = self.time_scaling(t).view(batch_size, -1, 1, 1)
        x1    = self.inc(x_img) + t_emb
        x2    = self.down1(x1)
        x3    = self.down2(x2)
        x_bot = self.bot(x3)
        x_up  = self.up1(x_bot)
        x_dec1 = self.dec1(torch.cat([x_up, x2], dim=1))
        x_up2  = self.up2(x_dec1)
        x_dec2 = self.dec2(torch.cat([x_up2, x1], dim=1))
        out   = self.outc(x_dec2)
        return out.view(batch_size, -1)


class DoubleConvTime(nn.Module):
    """Two convolutions with time injection."""
    def __init__(self, in_ch, out_ch, embed_dim):
        super().__init__()
        self.time_scaling = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.conv1 = nn.Conv2d(in_ch, embed_dim, 3, padding=1)
        self.gn1   = nn.GroupNorm(4, embed_dim)
        self.conv2 = nn.Conv2d(embed_dim, out_ch, 3, padding=1)
        self.gn2   = nn.GroupNorm(4, out_ch)
        self.silu  = nn.SiLU()

    def forward(self, x, t):
        t_emb = self.time_scaling(t)
        h = self.silu(self.gn1(self.conv1(x)))
        h = h + t_emb[:, :, None, None]
        h = self.silu(self.gn2(self.conv2(h)))
        return h


# ---------------------------------------------------------------------------
# Forward-Backward (analytical, no learning)
# ---------------------------------------------------------------------------

def FB(x0, x_t, t, W, Wt, L, alpha, NbIt=200):
    """Forward-Backward algorithm (cf. section 6 of the notes)."""
    eps   = 1e-8
    t_col = t[:, None]
    sigma = ((1 - t) / (t + eps)) ** 2
    mu    = alpha * sigma
    z     = x_t / (t_col + eps)
    x     = x0.detach().clone()
    gamma = 1 / L
    for _ in range(1, NbIt):
        u = x - gamma * (x - z)
        x = prox_l1_via_frame(u, W, Wt, mu * gamma)
    return (x - x_t) / ((1 - t)[:, None])


class FB_model(nn.Module):
    def __init__(self, x0, W, Wt, Lips, alpha, NbIt=200):
        super().__init__()
        self.x0    = x0
        self.W     = W
        self.Wt    = Wt
        self.Lips  = Lips
        self.alpha = alpha
        self.NbIt  = NbIt

    def forward(self, x):
        x_t, t = x[:, :2], x[:, 2]
        x0 = self.x0.repeat(x_t.shape[0], 1)
        return FB(x0, x_t, t, self.W, self.Wt, self.Lips, self.alpha, NbIt=self.NbIt)


def DFB(x0, x_t, t, W, Wt, L, alpha, NbIt=200):
    """Dual Forward-Backward algorithm (cf. section 6 of the notes)."""
    eps   = 1e-8
    t_col = t[:, None]
    sigma = ((1 - t) / (t + eps)) ** 2
    mu    = alpha * sigma
    z     = x_t / (t_col + eps)
    x     = x0.detach().clone()
    u     = torch.zeros_like(x)
    tau   = 1 / L
    for _ in range(1, NbIt):
        x = z - Wt(u)
        u = proj_l_inf(u + tau * W(x), (mu * tau)[:, None])
    return (x - x_t) / ((1 - t)[:, None])


class DFB_model(nn.Module):
    def __init__(self, x0, W, Wt, Lips, alpha, NbIt=200):
        super().__init__()
        self.x0    = x0
        self.W     = W
        self.Wt    = Wt
        self.Lips  = Lips
        self.alpha = alpha
        self.NbIt  = NbIt

    def forward(self, x):
        x_t, t = x[:, :2], x[:, 2]
        x0 = self.x0.repeat(x_t.shape[0], 1)
        return DFB(x0, x_t, t, self.W, self.Wt, self.Lips, self.alpha, NbIt=self.NbIt)


# ---------------------------------------------------------------------------
# DFB-UNN
# ---------------------------------------------------------------------------

class DFB_Iteration(nn.Module):
    def __init__(self, dim, prox, dual_dim=None, version="LFO", w_bias=False):
        super().__init__()
        self.version  = version
        dual_dim      = dual_dim or dim
        if version == "LFO":
            self.W_weight = nn.Parameter(kaiming_init(dual_dim, dim))
            # V: (dim, dual_dim) so that F.linear(u, V) with u:(B,dual_dim) → (B,dim)
            self.V_weight = nn.Parameter(kaiming_init(dim, dual_dim))
            self.tau = nn.Parameter(torch.tensor(0.5))
        else:
            W_init = torch.randn(dual_dim, dim) * 0.01
            self.W_weight = nn.Parameter(W_init)
            self.register_buffer('_sigma_u', F.normalize(torch.randn(dual_dim), dim=0))
        # variante (i) : biais appris sur W (pénalise g(W·-b)) -> opérateur non-impair
        # en fonction de l'entrée. Zero-init => démarre exactement symétrique.
        self.W_bias = nn.Parameter(torch.zeros(dual_dim)) if w_bias else None
        self.prox = prox

    def forward(self, u, z, t):
        V      = self.V_weight if self.version == "LFO" else self.W_weight.T
        x_next = z - F.linear(u, V, None)
        if self.version == "LFO":
            tau = F.softplus(self.tau)
        else:
            _s, self._sigma_u = sigma_max_power_iter(self.W_weight, self._sigma_u)
            tau = 1.99 / _s ** 2
        step   = tau * F.linear(x_next, self.W_weight, self.W_bias)
        u_next = self.prox(torch.cat((u + step, t), dim=-1))
        return x_next, u_next


class DFB_UNN(nn.Module):
    def __init__(self, dim, K=10, learned_prox=False, w=32, dual_dim=None, version="LFO", end_div=True, begin_div=False, pred="x", w_bias=True):
        super().__init__()
        self.dim           = dim
        self.K             = K
        self.dual_dim      = dual_dim or dim
        self.version       = version
        self.end_div = end_div
        self.begin_div = begin_div
        self.pred = pred
        self.predicts_x1 = False # (pred != "v")
        if learned_prox:
            self.proxs = nn.ModuleList([MLP(dim=self.dual_dim, time_varying=True, w=w) for _ in range(K)])
        else:
            self.proxs = nn.ModuleList([L1ProxFlat(dim=self.dual_dim) for _ in range(K)])
        self.layers = nn.ModuleList([
            DFB_Iteration(dim, self.proxs[i], dual_dim=self.dual_dim, version=version, w_bias=w_bias) for i in range(K)
        ])

    def forward(self, xt_t, return_u=False):
        xt = xt_t[:, :self.dim]
        t  = xt_t[:, self.dim:]
        z  = xt
        if self.begin_div:
            z = xt/torch.clamp(t, min=0.05)
        x  = z
        u  = torch.zeros(xt.shape[0], self.dual_dim, device=xt.device)
        for layer in self.layers:
            x, u = layer(u, z, t)
        if self.pred == "v":
            vt = x
        # elif self.training:
        #     # During training D_theta(xt,t) predicts x1 directly; the loss
        #     # is computed against x1, not against a velocity target.
        #     return x
        else:
            vt = x - xt
            if self.end_div:
                vt = vt / torch.clamp(1 - t, min=0.05)
        if return_u:
            return vt, u
        return vt

class FLAT_DFB_UNN(nn.Module):
    """
    DFB Unrolled Network trained with FLAT (FLow-Aligned Training).

    The time schedule is fixed (not learned):
        t_k = 1 - (1 - k/N)^(1+alpha),  sum(delta_k) = 1, t_0=0, t_N=1.
    Larger alpha concentrates more timesteps near t=1.

    Two forward modes:
      "direct"        — standard cascade, one pass through all N layers in order.
                        Used for training and baseline inference.
      "velocity_step" — lookahead inference: at each step n, run all remaining
                        layers from the current state to estimate x1, then take
                        the Euler step v_n = (x1_hat - x) / (1 - t_n).

    Use train_flat_2moons() which implements:
        L = L_recon + w_velocity * (1/N) * sum_k L1(x^(k+1), x*_{t_{k+1}})
    """
    def __init__(self, dim, N=10, w=64, dual_dim=None, learned_prox=False,
                 version="LFO", alpha=4.0):
        super().__init__()
        self.dim      = dim
        self.N        = N
        self.dual_dim = dual_dim or dim
        self.alpha    = alpha

        if learned_prox:
            self.proxs = nn.ModuleList([
                small_MLP(dim=self.dual_dim, time_varying=True, w=w) for _ in range(N)
            ])
        else:
            self.proxs = nn.ModuleList([
                L1ProxFlat(dim=self.dual_dim) for _ in range(N)
            ])
        self.layers = nn.ModuleList([
            DFB_Iteration(dim, self.proxs[i], dual_dim=self.dual_dim, version=version)
            for i in range(N)
        ])
        print("time schedule:")
        with torch.no_grad():
            t_sched = self.get_schedule()
            for n in range(self.N + 1):
                print(f"  t_{n} = {t_sched[n].item():.4f}")

    def get_schedule(self, device=None):
        """FLAT schedule (Eq. 15): t_k = 1 - (1 - k/N)^(1+alpha). Dense near t=1 for alpha > 0."""
        k = torch.arange(self.N + 1, dtype=torch.float32)
        t = 1.0 - (1.0 - k / self.N).pow(1.0 + self.alpha)
        if device is not None:
            t = t.to(device)
        return t  # (N+1,), t[0]=0, t[N]=1

    def forward(self, x_input, mode="direct", t_start=None,
                return_traj=False, return_x1_hats=False):
        """
        x_input  : (B, dim)   pour "direct" / "velocity_step"
                   (B, dim+1) pour "fm_velocity" — cat([x_t, t]), interface torch_wrapper.
        t_start  : None (= 0, schedule complet [0..1]),
                   float, ou tensor (B,).
                   Sous-schedule adapté à [t_start, 1] pour chaque exemple.
        """
        t_base = self.get_schedule(device=x_input.device)  # (N+1,), scalaire

        # ── MODE fm_velocity ─────────────────────────────────────────────────
        # Interface compatible torch_wrapper / NeuralODE.
        # x_input = cat([x_t, t·1]) de shape (B, dim+1).
        # Retourne la vitesse (x1_hat - x_t) / (1-t), shape (B, dim).
        if mode == "fm_velocity":
            x_t   = x_input[:, :self.dim]          # (B, dim)
            t_vec = x_input[:, self.dim]            # (B,)
            B     = x_t.shape[0]
            # sous-schedule par exemple : t_k^n = t + (1-t) * t_base[k]
            t_s   = t_vec[:, None] + (1.0 - t_vec[:, None]) * t_base[None, :]  # (B, N+1)
            x = x_t
            u = torch.zeros(B, self.dual_dim, device=x.device)
            for n, layer in enumerate(self.layers):
                x, u = layer(u, x, t_s[:, n].unsqueeze(1))  # (B,1) par exemple
            denom = (1.0 - t_vec).clamp(min=0.05).unsqueeze(1)
            return (x - x_t) / denom                         # vitesse (B, dim)

        # ── MODES direct / velocity_step ─────────────────────────────────────
        B = x_input.shape[0]

        if t_start is None:
            # FLAT original : schedule scalaire [0, ..., 1]
            t_s        = t_base          # (N+1,)
            per_sample = False
        else:
            if not isinstance(t_start, torch.Tensor):
                t_start = torch.full((B,), float(t_start), device=x_input.device)
            # sous-schedule : t_k^n = t_start + (1-t_start) * t_base[k],  shape (B, N+1)
            t_s        = t_start[:, None] + (1.0 - t_start[:, None]) * t_base[None, :]
            per_sample = True

        def _t_tensor(n):
            if per_sample:
                return t_s[:, n].unsqueeze(1)                          # (B, 1)
            return torch.full((B, 1), t_s[n].item(), device=x_input.device)

        if mode == "direct":
            x        = x_input
            u        = torch.zeros(B, self.dual_dim, device=x.device)
            x_traj   = [x]
            x_states = []

            for n, layer in enumerate(self.layers):
                x_next, u = layer(u, x, _t_tensor(n))
                x_states.append(x_next)
                x = x_next
                x_traj.append(x)

            if return_traj and return_x1_hats: return x, x_traj, x_states
            if return_traj: return x, x_traj
            if return_x1_hats: return x, x_states
            return x

        elif mode == "velocity_step":
            x      = x_input
            u      = torch.zeros(B, self.dual_dim, device=x.device)
            x_traj = [x]

            for n in range(self.N):
                if per_sample:
                    t_n     = t_s[:, n].mean().item()
                    delta_n = (t_s[:, n + 1] - t_s[:, n]).mean().item()
                else:
                    t_n     = t_s[n].item()
                    delta_n = (t_s[n + 1] - t_s[n]).item()

                with torch.no_grad():
                    x_temp, u_temp = x, u
                    for m in range(n, self.N):
                        x_temp, u_temp = self.layers[m](u_temp, x_temp, _t_tensor(m))
                    x1_hat = x_temp

                denom    = max(1.0 - t_n, 0.05)
                lambda_n = delta_n / denom
                x_new    = (1.0 - lambda_n) * x + lambda_n * x1_hat

                _, u = self.layers[n](u, x, _t_tensor(n))
                x    = x_new
                x_traj.append(x)

            if return_traj: return x, x_traj
            return x

class FLAT_DFB_UNN_v2(nn.Module):
    def __init__(self, dim, K=10, learned_prox=False, w=32, dual_dim=None, version="LFO", alpha=2.0):
        super().__init__()
        self.dim           = dim
        self.K             = K
        self.dual_dim      = dual_dim or dim
        self.version       = version
        self.alpha         = alpha
        if learned_prox:
            self.proxs = nn.ModuleList([MLP(dim=self.dual_dim, time_varying=True, w=w) for _ in range(K)])
        else:
            self.proxs = nn.ModuleList([L1ProxFlat(dim=self.dual_dim) for _ in range(K)])
        self.layers = nn.ModuleList([
            DFB_Iteration(dim, self.proxs[i], dual_dim=self.dual_dim, version=version) for i in range(K)
        ])
        
    
    def get_schedule(self, device=None):
        """FLAT schedule (Eq. 15): t_k = 1 - (1 - k/K)^(1+alpha). Dense near t=1 for alpha > 0."""
        k = torch.arange(self.K + 1, dtype=torch.float32)
        t = 1.0 - (1.0 - k / self.K).pow(1.0 + self.alpha)
        if device is not None:
            t = t.to(device)
        return t  # (K+1,), t[0]=0, t[K]=1

    def forward(self, xt_t, return_traj=False):
        xt = xt_t[:, :self.dim]
        t  = xt_t[:, self.dim:]
        z  = xt
        x  = z
        u  = torch.zeros(xt.shape[0], self.dual_dim, device=xt.device)
        x_traj = [x]
        for layer in self.layers:
            x, u = layer(u, z, t)
            x_traj.append(x)

        vt = (xt - x) / torch.clamp(1 - t, min=0.05)

        if self.training:
            return vt, x, x_traj
        else:
            if return_traj:
                return vt, x_traj
            else:
                return vt

# ---------------------------------------------------------------------------
# DiFB-UNN (DFB with inertia/momentum)
# ---------------------------------------------------------------------------

class DiFB_UNN(nn.Module):
    def __init__(self, dim, K=10, w=32, dual_dim=None, version="LFO", prox_type="mlp", a=3.0):
        super().__init__()
        self.dim      = dim
        self.K        = K
        self.dual_dim = dual_dim or dim
        self.version  = version
        self.a        = a  # Paramètre a > 2 pour la suite LNO (Corollaire 1)
        self.predicts_x1 = True

        if version == "LFO":
            # DDiFB-LFO : Le paramètre d'inertie rho_k (noté alpha_k dans le Tableau 1)
            # est APPRIS pour chaque couche k.
            self.rho = nn.Parameter(torch.full((K,), 0.5))
            
        if prox_type == "l1":
            self.proxs = nn.ModuleList([L1ProxFlat(dim=self.dual_dim) for _ in range(K)])
        else:
            self.proxs = nn.ModuleList([MLP(dim=self.dual_dim, time_varying=True, w=w) for _ in range(K)])
            
        self.layers = nn.ModuleList([
            DFB_Iteration(dim, self.proxs[i], dual_dim=self.dual_dim, version=version) for i in range(K)
        ])

    def forward(self, xt_t):
        xt     = xt_t[:, :self.dim]
        t      = xt_t[:, self.dim:]
        z      = xt
        x      = z
        u      = torch.zeros(xt.shape[0], self.dual_dim, device=xt.device)
        u_prev = torch.zeros(xt.shape[0], self.dual_dim, device=xt.device)
        
        for k, layer in enumerate(self.layers):
            x, u_tilde = layer(u, z, t)
            
            if self.version == "LFO":
                # On utilise le paramètre appris
                rho_k = self.rho[k]
            else:
                # DDiFB-LNO : On applique rigoureusement la suite inertielle de FISTA (Corollaire 1)
                # t_k = (k + a - 1) / a  et  rho_k = t_{k-1} / t_{k+1}
                # (Note : l'indice k commence à 0 ici, on adapte pour coller à la théorie)
                t_k_minus_1 = max(1.0, (k - 1 + self.a) / self.a) 
                t_k_plus_1  = (k + 1 + self.a) / self.a
                rho_k = t_k_minus_1 / t_k_plus_1
                
            u_new  = (1 + rho_k) * u_tilde - rho_k * u_prev
            u_prev = u_tilde.clone()
            u      = u_new

        if self.training:
            return x
        return (x - xt) / torch.clamp(1 - t, 0.05)

# ---------------------------------------------------------------------------
# SharedDFB-UNN (shared W across iterations, flat/linear)
# ---------------------------------------------------------------------------

class SharedDFB_UNN(nn.Module):
    def __init__(self, dim, K=10, w=32, dual_dim=None, version="LFO", prox_type="mlp"):
        super().__init__()
        self.dim      = dim
        self.K        = K
        self.version  = version
        self.dual_dim = dual_dim or dim
        self.predicts_x1 = True
        self.shared_W = nn.Parameter(kaiming_init(self.dual_dim, dim))
        if version == "LFO":
            # shared_V: (dim, dual_dim) so F.linear(u, V) with u:(B,dual_dim) → (B,dim)
            self.shared_V = nn.Parameter(kaiming_init(dim, self.dual_dim))
            self.tau = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('_sigma_u', F.normalize(torch.randn(self.dual_dim), dim=0))
        if prox_type == "l1":
            self.prox = L1ProxFlat(dim=self.dual_dim)
        else:
            self.prox = small_MLP(dim=self.dual_dim, w=w, time_varying=True)

    def forward(self, xt_t, n_iter=None, return_traj=False):
        iters  = n_iter if n_iter is not None else self.K
        xt     = xt_t[:, :self.dim]
        t      = xt_t[:, self.dim:]
        z      = xt
        u      = torch.zeros(xt.shape[0], self.dual_dim, device=xt.device)
        x_next = z
        V = self.shared_V if self.version == "LFO" else self.shared_W.T
        if self.version == "LFO":
            tau = F.softplus(self.tau)
        else:
            _s, self._sigma_u = sigma_max_power_iter(self.shared_W, self._sigma_u)
            tau = 1.99 / _s ** 2
        x_traj = [x_next]
        for _ in range(iters):
            x_next = z - F.linear(u, V, None)
            step   = tau * F.linear(x_next, self.shared_W, None)
            u      = self.prox(torch.cat((u + step, t), dim=-1))
            x_traj.append(x_next)
        if self.training:
            out = x_next
        else:
            out = (x_next - xt) / torch.clamp(1 - t, 0.05)
        if return_traj:
            return out, x_traj
        return out


# ---------------------------------------------------------------------------
# SharedDiFB-UNN (SharedDFB with inertia/momentum on the dual variable)
# ---------------------------------------------------------------------------

class SharedDiFB_UNN(nn.Module):
    def __init__(self, dim, K=10, w=32, dual_dim=None, version="LFO", prox_type="mlp", a=3.0):
        super().__init__()
        self.dim      = dim
        self.K        = K
        self.version  = version
        self.a        = a
        self.dual_dim = dual_dim or dim
        self.predicts_x1 = True
        self.rho = nn.Parameter(torch.full((K,), 0.5))
        self.shared_W = nn.Parameter(kaiming_init(self.dual_dim, dim))
        if version == "LFO":
            # shared_V: (dim, dual_dim) so F.linear(u, V) with u:(B,dual_dim) → (B,dim)
            self.shared_V = nn.Parameter(kaiming_init(dim, self.dual_dim))
            self.tau = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('_sigma_u', F.normalize(torch.randn(self.dual_dim), dim=0))
        if prox_type == "l1":
            self.prox = L1ProxFlat(dim=self.dual_dim)
        else:
            self.prox = small_MLP(dim=self.dual_dim, w=w, time_varying=True)

    def forward(self, xt_t, n_iter=None):
        iters  = n_iter if n_iter is not None else self.K
        xt     = xt_t[:, :self.dim]
        t      = xt_t[:, self.dim:]
        z      = xt
        u      = torch.zeros(xt.shape[0], self.dual_dim, device=xt.device)
        u_prev = torch.zeros(xt.shape[0], self.dual_dim, device=xt.device)
        x_next = z
        V = self.shared_V if self.version == "LFO" else self.shared_W.T
        if self.version == "LFO":
            tau = F.softplus(self.tau)
        else:
            _s, self._sigma_u = sigma_max_power_iter(self.shared_W, self._sigma_u)
            tau = 1.99 / _s ** 2
        for k in range(iters):
            x_next  = z - F.linear(u, V, None)
            step    = tau * F.linear(x_next, self.shared_W, None)
            u_tilde = self.prox(torch.cat((u + step, t), dim=-1))
            
            if self.version == "LFO":
                rho_k = self.rho[k]
            else:
                # DDiFB-LNO : suite inertielle de FISTA (Corollaire 1)
                t_k_minus_1 = max(1.0, (k - 1 + self.a) / self.a)
                t_k_plus_1  = (k + 1 + self.a) / self.a
                rho_k = t_k_minus_1 / t_k_plus_1
            
            u_new   = (1 + rho_k) * u_tilde - rho_k * u_prev
            u_prev  = u_tilde.clone()
            u       = u_new
        if self.training:
            return x_next
        return (x_next - xt) / torch.clamp(1 - t, 0.05)


# ---------------------------------------------------------------------------
# ConvDFB-UNN (each layer has its own W)
# ---------------------------------------------------------------------------

class ConvDFB_Iteration(nn.Module):
    """w_bias : biais convolutif appris sur W (on pénalise g(W·−b)), même rôle que
    dans _OrigConvScCP_Iteration. Sans lui le déroulé est linéaire en (x, u, z) avec
    un prox impair => v(-x_t,t) = -v(x_t,t) => la moitié des chiffres sortent en
    couleurs inversées. Zero-init : le modèle démarre exactement symétrique.
    prox_w : largeur du MLP r(t) du prox l1 (32 = défaut historique de L1ProxConv)."""
    def __init__(self, internal_channel, use_Unet=False, version="LFO", w_bias=True,
                 prox_w=32):
        super().__init__()
        self.version  = version
        self.W_weight = nn.Parameter(kaiming_init(internal_channel, 1, 9, 9))
        if version == "LFO":
            self.V_weight = nn.Parameter(kaiming_init(internal_channel, 1, 9, 9))
            self.tau = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('_sigma_u', F.normalize(torch.randn(internal_channel), dim=0))
        self.W_bias = nn.Parameter(torch.zeros(internal_channel)) if w_bias else None
        if use_Unet == "l1":
            self.prox = L1ProxConv(w=prox_w)
        else:
            self.prox = DoubleConvTime(
                in_ch=internal_channel, out_ch=internal_channel,
                embed_dim=internal_channel // 2,
            )

    def forward(self, u, z, t):
        V      = self.V_weight if self.version == "LFO" else self.W_weight
        x_next = z - F.conv_transpose2d(u, V, padding=4)
        if self.version == "LFO":
            tau = F.softplus(self.tau)
        else:
            with torch.no_grad():
                _s, new_u = sigma_max_power_iter(self.W_weight, self._sigma_u)
                self._sigma_u.copy_(new_u)
            tau = 1.99 / _s ** 2
        step   = tau * F.conv2d(x_next, self.W_weight, bias=self.W_bias, padding=4)
        u_next = self.prox(u + step, t)
        return x_next, u_next


class ConvDFB_UNN(nn.Module):
    def __init__(self, dim, K=10, internal_channel=64, use_Unet=False, version="LFO",
                 use_checkpoint=False, w_bias=True, prox_w=32):
        super().__init__()
        self.dim              = dim
        self.K                = K
        self.internal_channel = internal_channel
        self.version          = version
        self.use_checkpoint   = use_checkpoint
        self.predicts_x1      = True
        self.img_size         = 28
        self.in_channels      = 1
        self.layers = nn.ModuleList([
            ConvDFB_Iteration(internal_channel=internal_channel,
                              use_Unet=use_Unet, version=version,
                              w_bias=w_bias, prox_w=prox_w)
            for _ in range(K)
        ])

    def forward(self, xt_t, return_iterates=False):
        """return_iterates : renvoie aussi les itérés primaux internes du déroulé DFB,
        [x^(0)=z, x^(1), ..., x^(K)] empilés en (K+1, B, 1, 28, 28) — outil d'analyse,
        n'affecte pas la sortie renvoyée (même convention que ConvScCP_UNN, voir
        trajectory_convsccp.py)."""
        batch_size = xt_t.shape[0]
        z = xt_t[:, :self.dim].contiguous().view(batch_size, 1, 28, 28)
        t = xt_t[:, self.dim:]
        u = torch.zeros(batch_size, self.internal_channel, 28, 28, device=z.device)
        x = z
        iterates = [x.clone()] if return_iterates else None
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x, u = checkpoint(layer, u, z, t, use_reentrant=False)
            else:
                x, u = layer(u, z, t)
            if return_iterates:
                iterates.append(x.clone())

        if self.training:
            out = x.view(batch_size, -1)
        else:
            out = (x - z).view(batch_size, -1) / torch.clamp(1 - t, 0.05)
        if return_iterates:
            return out, torch.stack(iterates)
        return out


# ---------------------------------------------------------------------------
# ConvDiFB-UNN (ConvDFB with inertia/momentum on the dual variable)
# ---------------------------------------------------------------------------

class ConvDiFB_UNN(nn.Module):
    def __init__(self, dim, K=10, internal_channel=64, use_Unet=False, version="LFO", a=3.0,
                 use_checkpoint=False, w_bias=True, prox_w=32):
        super().__init__()
        self.dim              = dim
        self.K                = K
        self.internal_channel = internal_channel
        self.version          = version
        self.a                = a  # Paramètre a > 2 pour la suite LNO (Corollaire 1)
        self.use_checkpoint   = use_checkpoint
        self.predicts_x1      = True

        if version == "LFO":
            # DDiFB-LFO : le paramètre d'inertie rho_k est appris pour chaque couche k.
            self.rho = nn.Parameter(torch.full((K,), 0.5))

        self.layers = nn.ModuleList([
            ConvDFB_Iteration(internal_channel=internal_channel,
                              use_Unet=use_Unet, version=version,
                              w_bias=w_bias, prox_w=prox_w)
            for _ in range(K)
        ])

    def forward(self, xt_t):
        batch_size = xt_t.shape[0]
        z = xt_t[:, :self.dim].view(batch_size, 1, 28, 28)
        t = xt_t[:, self.dim:]
        x = z
        u      = torch.zeros(batch_size, self.internal_channel, 28, 28, device=z.device)
        u_prev = torch.zeros(batch_size, self.internal_channel, 28, 28, device=z.device)

        for k, layer in enumerate(self.layers):
            if self.use_checkpoint and self.training:
                x, u_tilde = checkpoint(layer, u, z, t, use_reentrant=False)
            else:
                x, u_tilde = layer(u, z, t)

            if self.version == "LFO":
                rho_k = self.rho[k]
            else:
                # DDiFB-LNO : suite inertielle de FISTA (Corollaire 1)
                t_k_minus_1 = max(1.0, (k - 1 + self.a) / self.a)
                t_k_plus_1  = (k + 1 + self.a) / self.a
                rho_k = t_k_minus_1 / t_k_plus_1

            u_new  = (1 + rho_k) * u_tilde - rho_k * u_prev
            u_prev = u_tilde.clone()
            u      = u_new

        if self.training:
            return x.view(batch_size, -1)
        return (x - z).view(batch_size, -1) / torch.clamp(1 - t, 0.05)

# =========================================================================== #
#  1.  Autoencodeur MNIST  (E gelé après pré-entraînement, D appris)
# =========================================================================== #
class MnistAE(nn.Module):
    """28x28x1  <->  (C_lat, 7, 7).  k=4,s=2,p=1 -> 28<->14<->7 exact.

    VAE-léger : l'encodeur prédit (mu, logvar) ; le décodeur est entraîné sur
    un échantillon reparamétrisé z = mu + sigma*eps, pas sur mu directement.
    But : robustesse du décodeur autour du point encodé (cf. Dieleman) plutôt
    qu'une vraie structuration de l'espace latent par le KL (poids volontairement
    faible). encode() reste déterministe (renvoie mu) pour les usages en aval
    (cible x1 du Flow Matching).
    """
    def __init__(self, c_lat=4, base=32):
        super().__init__()
        self.c_lat = c_lat
        self.latent_spatial = 7
        self.enc = nn.Sequential(
            nn.Conv2d(1,    base,   4, 2, 1), nn.GroupNorm(8, base),   nn.SiLU(),
            nn.Conv2d(base, base*2, 4, 2, 1), nn.GroupNorm(8, base*2), nn.SiLU(),
            nn.Conv2d(base*2, 2 * c_lat, 3, 1, 1),     # mu, logvar concaténés
        )
        self.dec = nn.Sequential(
            nn.Conv2d(c_lat, base*2, 3, 1, 1), nn.GroupNorm(8, base*2), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base*2, base, 3, 1, 1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base, 1, 3, 1, 1),
        )

    def encode_dist(self, x_img):     # (B,1,28,28) -> mu, logvar : (B,C_lat,7,7)
        mu, logvar = self.enc(x_img).chunk(2, dim=1)
        return mu, logvar

    def encode(self, x_img):          # déterministe (mu), pour le FM en aval
        mu, _ = self.encode_dist(x_img)
        return mu

    def decode(self, z_lat):          # (B,C_lat,7,7) -> (B,1,28,28)
        return self.dec(z_lat)

    def forward(self, x_img):
        return self.decode(self.encode(x_img))


def pretrain_ae(ae, train_loader, device, epochs=10, lr=1e-3, kl_weight=1e-4):
    """Pré-entraînement reconstruction (L1) + KL faible (VAE-léger). À faire
    AVANT le FM latent. Le rôle du KL ici n'est pas de gaussianiser
    l'espace latent (poids trop faible pour ça, cf. Dieleman) mais le bruit de
    reparamétrisation force le décodeur à rester correct autour du point
    encodé, et pas seulement pile dessus. L1 plutôt que MSE pour la
    reconstruction : la MSE moyenne vers des bords flous (cf. Dieleman,
    "excessive high-frequency erasure"), L1 donne des bords plus nets pour
    un coût quasi nul (vérifié empiriquement : MSE de reconstruction mesurée
    en eval baisse de ~0.04 à ~0.032 sur les zéros MNIST)."""
    ae = ae.to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    ae.train()
    for ep in range(epochs):
        tot_rec, tot_kl = 0.0, 0.0
        for x_img, _ in train_loader:
            x_img = x_img.to(device).view(-1, 1, 28, 28)
            mu, logvar = ae.encode_dist(x_img)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            rec = ae.decode(z)

            recon_loss = F.l1_loss(rec, x_img)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_weight * kl

            opt.zero_grad(); loss.backward(); opt.step()
            tot_rec += recon_loss.item()
            tot_kl  += kl.item()
        n = len(train_loader)
        print(f"[AE] epoch {ep+1}/{epochs}  recon L1 = {tot_rec/n:.5f}  KL = {tot_kl/n:.5f}")
    ae.eval()
    for p in ae.parameters():          # GELER l'AE pour le FM
        p.requires_grad_(False)
    return ae


# =========================================================================== #
#  2.  Itération Chambolle-Pock dans le latent
# =========================================================================== #
class ConvScCP_Iteration(nn.Module):
    """Un pas CP accéléré, entièrement latent.
       W : C_lat -> C_dual  (channel mixing automatique car C_lat > 1)."""
    def __init__(self, c_lat, c_dual, use_Unet=False, version="LFO",
                 kernel_size=9, spatial=7, w_bias=False):
        super().__init__()
        self.version = version
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.pad = k // 2
        self.W_weight = nn.Parameter(kaiming_init(c_dual, c_lat, k, k))
        if version == "LFO":
            self.V_weight = nn.Parameter(kaiming_init(c_dual, c_lat, k, k))
        else:
            self.register_buffer("_v", F.normalize(
                torch.randn(1, c_lat, spatial, spatial), dim=None))
        # variante (i) : biais convolutif appris sur W -> non-impair en fct de l'entree
        self.W_bias = nn.Parameter(torch.zeros(c_dual)) if w_bias else None
        if use_Unet == "l1":
            self.prox = L1ProxConv(channels=c_dual)
        else:
            self.prox = DoubleConvTime(in_ch=c_dual, out_ch=c_dual,
                                       embed_dim=c_dual // 2)

    def spectral_norm(self):
        """||W|| comme opérateur C_lat->C_dual (power iteration, 1 pas/forward)."""
        with torch.no_grad():
            v = self._v
            Wv = F.conv2d(v, self.W_weight, padding=self.pad)          # C_dual
            WtWv = F.conv_transpose2d(Wv, self.W_weight, padding=self.pad)  # C_lat
            s2 = (v * WtWv).sum() / ((v * v).sum() + 1e-12)
            self._v.copy_(WtWv / (WtWv.norm() + 1e-12))
        return s2.clamp_min(1e-12).sqrt()

    def forward(self, x, z, u, t, tau, sigma, alpha_mom):
        # x, z : (B,C_lat,Hl,Wl) ; u : (B,C_dual,Hl,Wl) ; tout latent
        V = self.V_weight if self.version == "LFO" else self.W_weight
        # Primal : prox_{tau ½||·-z||²}( x - tau W^T u )
        primal_input = x - tau * F.conv_transpose2d(u, V, padding=self.pad)
        x_next = (primal_input + tau * z) / (1 + tau)
        # Extrapolation
        y = x_next + alpha_mom * (x_next - x)
        # Dual
        dual_step = sigma * F.conv2d(y, self.W_weight, bias=self.W_bias, padding=self.pad)
        u_next = self.prox(u + dual_step, t)
        return x_next, u_next


# =========================================================================== #
#  3.  Modèle complet : latent -> latent  (prédit x1 latent)
# =========================================================================== #
class LatentScCP_UNN(nn.Module):
    def __init__(self, c_lat=4, latent_spatial=7, K=10, internal_channel=64,
                 use_Unet=False, version="LFO", use_checkpoint=False,
                 kernel_size=3, predicts_x1=True, x1_skip=False,
                 vloss_weight=False, w_bias=False):
        super().__init__()
        self.c_lat = c_lat
        self.Hl = latent_spatial
        self.dim = c_lat * latent_spatial * latent_spatial   # dim latente aplatie
        self.K = K
        self.c_dual = internal_channel
        self.version = version
        self.use_checkpoint = use_checkpoint
        self.predicts_x1 = predicts_x1   # False : la sortie de l'itération ScCP EST la vitesse
        self.x1_skip = x1_skip           # True (avec predicts_x1) : x1_pred = z + (1-t)*out, dependance a z forcee
        # marqueur lu par le script d'entrainement (pas par forward). Avec predicts_x1 (reseau sort
        # x_pred), si True la loss x1 est calculee dans l'ESPACE V : ponderee par 1/(1-t)^2.
        # C'est la config recommandee par le papier "Back to Basics" (Algo 1, Tab 1(3)(a)) :
        # reseau=x_pred + loss=v-space. Sans skip. Quadrant jamais teste auparavant.
        self.vloss_weight = vloss_weight

        if version == "LNO":
            self.log_tau = nn.Parameter(torch.full((K,), -0.5))
        else:
            self.log_tau0 = nn.Parameter(torch.tensor(-0.5))

        self.layers = nn.ModuleList([
            ConvScCP_Iteration(c_lat, internal_channel, use_Unet=use_Unet,
                               version=version, kernel_size=kernel_size,
                               spatial=latent_spatial, w_bias=w_bias)
            for _ in range(K)
        ])

    def forward(self, xt_t):
        B = xt_t.shape[0]
        z = xt_t[:, :self.dim].view(B, self.c_lat, self.Hl, self.Hl)  # latent FM point
        t = xt_t[:, self.dim:]
        x = z.clone()
        u = torch.zeros(B, self.c_dual, self.Hl, self.Hl, device=z.device)

        if self.version == "LNO":
            taus = F.softplus(self.log_tau)
            for k, layer in enumerate(self.layers):
                tau_k = taus[k]
                alpha_k = (1.0 + 2.0 * tau_k).pow(-0.5)
                sn = layer.spectral_norm()
                sigma_k = 0.99 / (tau_k * sn ** 2)
                if self.use_checkpoint and self.training:
                    x, u = checkpoint(layer, x, z, u, t, tau_k, sigma_k, alpha_k,
                                      use_reentrant=False)
                else:
                    x, u = layer(x, z, u, t, tau_k, sigma_k, alpha_k)
        else:
            tau_k = F.softplus(self.log_tau0)
            sigma_k = 1.0
            for layer in self.layers:
                alpha_k = (1.0 + 2.0 * tau_k).pow(-0.5)
                if self.use_checkpoint and self.training:
                    x, u = checkpoint(layer, x, z, u, t, tau_k, sigma_k, alpha_k,
                                      use_reentrant=False)
                else:
                    x, u = layer(x, z, u, t, tau_k, sigma_k, alpha_k)
                tau_k = alpha_k * tau_k

        out = x.view(B, -1)

        if not self.predicts_x1:
            return out                                # v-prediction : la sortie EST la vitesse

        z_flat = z.view(B, -1)
        if self.x1_skip:
            x1_pred = z_flat + (1 - t) * out          # skip non-appris : dependance a z forcee
        else:
            x1_pred = out                             # x-prediction nue : loss = ||x1_pred - x1_lat||^2

        if self.training:
            return x1_pred
        # eval : vitesse FM dans le latent  v = (x1 - xt)/(1 - t)
        return (x1_pred - z_flat) / torch.clamp(1 - t, min=0.05)


class SmallUNetLatent(nn.Module):
    """Baseline générique (non-proximale) à la même interface que LatentScCP_UNN
    (forward(xt_t) -> x1_pred latent, predicts_x1=True), pour le test de
    contrôle "décodeur figé identique" : isoler ce qui vient du VAE partagé de
    ce qui vient de la structure proximale ScCP. Pas de pooling spatial (grille
    latente 7x7 trop petite pour un vrai down/up-sampling) — juste une pile de
    blocs convolutifs avec conditionnement temporel, sans aucune interprétation
    variationnelle/proximale."""
    def __init__(self, c_lat=4, latent_spatial=7, base_ch=64, depth=4):
        super().__init__()
        self.c_lat = c_lat
        self.Hl = latent_spatial
        self.dim = c_lat * latent_spatial * latent_spatial
        self.predicts_x1 = True
        self.inc = nn.Conv2d(c_lat, base_ch, 3, padding=1)
        self.blocks = nn.ModuleList([
            DoubleConvTime(base_ch, base_ch, embed_dim=base_ch) for _ in range(depth)
        ])
        self.outc = nn.Conv2d(base_ch, c_lat, 3, padding=1)

    def forward(self, xt_t):
        B = xt_t.shape[0]
        z = xt_t[:, :self.dim].view(B, self.c_lat, self.Hl, self.Hl)
        t = xt_t[:, self.dim:]
        h = self.inc(z)
        for block in self.blocks:
            h = block(h, t)
        x1_pred = self.outc(h).view(B, -1)

        if self.training:
            return x1_pred
        z_flat = z.view(B, -1)
        return (x1_pred - z_flat) / torch.clamp(1 - t, min=0.05)


class SmallUNetLatentV2(nn.Module):
    """Comme SmallUNetLatent, mais avec la VRAIE structure down/up + skip
    connection du SmallUNet image ([architectures.py:206]), adaptée à la
    grille 7x7 via une conv stride-2 exacte (7->4) / ConvTranspose2d exacte
    (4->7) à la place du MaxPool2d (qui tronquerait 7->3). Sert de baseline
    générique honnête pour le test "décodeur figé" : même inductive bias
    multi-échelle que le SmallUNet image, juste sur le latent.

    predicts_x1 : si True (x-prediction, comme LatentScCP_UNN), la loss est
    ||out - x1||² et la vitesse est calculée après-coup. Si False
    (v-prediction, comme le VRAI SmallUNet image qui n'a pas l'attribut
    predicts_x1), la loss est directement ||out - (x1-x0)||² — cible à
    variance bien plus grande, donc gradient bien plus fort pour forcer le
    réseau à dépendre de son entrée plutôt que de "tricher" en prédisant la
    moyenne. Hypothèse à tester pour expliquer le collapse en x-prediction.

    x1_skip (seulement avec predicts_x1=True) : au lieu de renvoyer out
    directement comme x1_pred, calcule x1_pred = z + (1-t)*out. Le terme z
    n'est pas appris : le réseau ne peut plus "shortcuter" en ignorant son
    entrée, même si la variance de x1 est très faible (cas digit=0 seul).
    Reste de la x-prediction au sens de l'interface (sortie/loss = x1)."""
    def __init__(self, c_lat=4, latent_spatial=7, base_ch=64, predicts_x1=True,
                 x1_skip=False, vloss_weight=False):
        super().__init__()
        self.c_lat = c_lat
        self.Hl = latent_spatial
        self.dim = c_lat * latent_spatial * latent_spatial
        self.predicts_x1 = predicts_x1
        self.x1_skip = x1_skip
        # marqueur lu par le script d'entrainement (pas par forward). Avec predicts_x1 (reseau sort
        # x_pred), si True la loss x1 est calculee dans l'ESPACE V : ponderee par 1/(1-t)^2.
        # Config recommandee par "Back to Basics" (Algo 1, Tab 1(3)(a)) : reseau=x_pred + loss=v-space.
        self.vloss_weight = vloss_weight
        self.time_scaling = nn.Sequential(
            nn.Linear(1, base_ch), nn.SiLU(), nn.Linear(base_ch, base_ch),
        )
        self.inc   = DoubleConv(c_lat, base_ch, base_ch)
        self.down1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),   # 7 -> 4 exact
            DoubleConv(base_ch * 2, base_ch * 2, base_ch * 2),
        )
        self.bot  = DoubleConv(base_ch * 2, base_ch * 2, base_ch * 2)
        self.up1  = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=3,
                                        stride=2, padding=1, output_padding=0)  # 4 -> 7 exact
        self.dec1 = DoubleConv(base_ch * 2, base_ch, base_ch)
        self.outc = nn.Conv2d(base_ch, c_lat, kernel_size=1)

    def forward(self, xt_t):
        B = xt_t.shape[0]
        z = xt_t[:, :self.dim].view(B, self.c_lat, self.Hl, self.Hl)
        t = xt_t[:, self.dim:]
        t_emb = self.time_scaling(t).view(B, -1, 1, 1)
        x1    = self.inc(z) + t_emb
        x2    = self.down1(x1)
        x_bot = self.bot(x2)
        x_up  = self.up1(x_bot)
        x_dec = self.dec1(torch.cat([x_up, x1], dim=1))
        out   = self.outc(x_dec).view(B, -1)

        if not self.predicts_x1:
            return out                      # v-prediction : out est deja la vitesse

        z_flat = z.view(B, -1)
        if self.x1_skip:
            x1_pred = z_flat + (1 - t) * out   # skip non-appris : dependance a z forcee
        else:
            x1_pred = out                      # x-prediction nue : loss = ||out - x1||^2

        if self.training:
            return x1_pred
        return (x1_pred - z_flat) / torch.clamp(1 - t, min=0.05)



# ---------------------------------------------------------------------------
# SharedConvDFB-UNN (shared weights across iterations)
# ---------------------------------------------------------------------------

class SharedConvDFB_UNN(nn.Module):
    def __init__(self, dim=784, K=10, internal_channel=64,
                 use_Unet=False, version="LFO"):
        super().__init__()
        self.dim = dim
        self.K   = K
        self.internal_channel = internal_channel
        self.version  = version
        self.predicts_x1 = True
        self.shared_W = nn.Parameter(kaiming_init(internal_channel, 1, 3, 3))
        if version == "LFO":
            self.shared_V = nn.Parameter(kaiming_init(internal_channel, 1, 3, 3))
            self.tau = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('_sigma_u', F.normalize(torch.randn(internal_channel), dim=0))
        if use_Unet is True or use_Unet == "small":
            self.prox = SmallUNet(
                in_channels=internal_channel, out_channels=internal_channel,
                base_ch=internal_channel // 2,
            )
            self.use_Unet = "small"
        elif use_Unet == "cfm":
            self.prox = UNetProxConv(in_channels=internal_channel)
            self.use_Unet = "cfm"
        elif use_Unet == "l1":
            self.prox = L1ProxConv()
            self.use_Unet = "l1"
        else:
            self.prox = DoubleConvTime(
                in_ch=internal_channel, out_ch=internal_channel,
                embed_dim=internal_channel // 2,
            )
            self.use_Unet = False

    def forward(self, xt_t, n_iter=None):
        iters      = n_iter if n_iter is not None else self.K
        batch_size = xt_t.shape[0]
        z = xt_t[:, :self.dim].view(batch_size, 1, 28, 28)
        t = xt_t[:, self.dim:]
        u = torch.zeros(batch_size, self.internal_channel, 28, 28, device=z.device)
        x_next = z
        V = self.shared_V if self.version == "LFO" else self.shared_W
        if self.version == "LFO":
            tau = F.softplus(self.tau)
        else:
            _s, self._sigma_u = sigma_max_power_iter(self.shared_W, self._sigma_u)
            tau = 1.99 / _s ** 2
        for _ in range(iters):
            x_next = (z - F.conv_transpose2d(u, V, padding=1)).contiguous()
            step   = (tau * F.conv2d(x_next, self.shared_W, padding=1)).contiguous()
            if self.use_Unet == "small":
                u_flat   = (u + step).view(batch_size, -1, 784)
                ut_t_flat = torch.cat(
                    [u_flat, t[:, 0].view(batch_size, 1, 1).expand(batch_size, self.internal_channel, 1)],
                    dim=-1,
                )
                u = self.prox(ut_t_flat).view(batch_size, self.internal_channel, 28, 28)
            else:
                u = self.prox(u + step, t)
        if self.training:
            return x_next.view(batch_size, -1)
        return (x_next - z).view(batch_size, -1) / torch.clamp(1 - t, 0.05)



# ---------------------------------------------------------------------------
# ScCP-UNN (Accelerated Chambolle-Pock, strongly convex variant)
# ---------------------------------------------------------------------------
# Schedule:  alpha_k = (1 + 2*rho*tau_k)^{-1/2}
#            tau_{k+1}   = alpha_k * tau_k          (decreasing)
#            sigma_{k+1} = sigma_k / alpha_k         (increasing)
#            sigma_k * tau_k = sigma_0 * tau_0 = const   (stability product preserved)
# Extrapolation: y = x^{k+1} + alpha_k*(x^{k+1} - x^k)   (alpha_k <= 1)
# ---------------------------------------------------------------------------

class ScCP_Iteration(nn.Module):
    """Single accelerated CP step.

    mu (pas primal), tau (pas dual), alpha are passed in by ScCP_UNN (adaptive
    schedule), so this class does not own them as parameters.
    Notation du papier : mu_k = pas primal, tau_k = pas dual.
    """
    def __init__(self, dim, prox_dual, dual_dim=None, version="LFO", w_bias=True):        
        super().__init__()
        self.version  = version
        dual_dim      = dual_dim or dim        
        if version == "LFO":
            self.W_weight = nn.Parameter(kaiming_init(dual_dim, dim))
            # V: (dim, dual_dim) so that F.linear(u, V) with u:(B,dual_dim) → (B,dim)
            self.V_weight = nn.Parameter(kaiming_init(dim, dual_dim))
        else:
            W_init = torch.randn(dual_dim, dim) * 0.01
            self.W_weight = nn.Parameter(W_init)
            
            self.register_buffer('_sigma_u', F.normalize(torch.randn(dual_dim), dim=0))
        # variante (i) : biais appris sur W (pénalise g(W·-b)) -> opérateur non-impair
        # en fonction de l'entrée. Zero-init => démarre exactement symétrique.
        self.W_bias = nn.Parameter(torch.zeros(dual_dim)) if w_bias else None
        self.prox_dual = prox_dual

    def spectral_norm(self):
        _s, self._sigma_u = sigma_max_power_iter(self.W_weight, self._sigma_u)
        return _s

    def forward(self, x, u, z, t, tau, sigma, alpha):
        V = self.V_weight if self.version == "LFO" else self.W_weight.T
        primal_input = x - tau * F.linear(u, V, None)
        x_next = (primal_input + tau * z) / (1 + tau)
        y = x_next + alpha * (x_next - x)
        dual_step = sigma * F.linear(y, self.W_weight, self.W_bias)
        u_next = self.prox_dual(torch.cat((u + dual_step, t), dim=-1))
        return x_next, u_next


class ScCP_UNN(nn.Module):
    def __init__(self, dim, K=10, w=32, dual_dim=None, version="LFO", prox_type="mlp", w_bias=True):
        super().__init__()
        self.dim      = dim
        self.K        = K
        self.version  = version
        self.dual_dim = dual_dim or dim
        self.predicts_x1 = False # True

        if version == "LNO":
            # DScCP-LNO : On apprend \mu_k (tau dans le code) à CHAQUE itération
            self.log_tau = nn.Parameter(torch.full((K,), -0.5))
        else:
            # DScCP-LFO : On apprend seulement \mu_0 et on applique la récurrence
            self.log_tau0 = nn.Parameter(torch.tensor(-0.5))
            # log_sigma0 est supprimé car le pas dual est absorbé dans W_k (Table 1)

        # log_rho est supprimé pour respecter alpha_k = (1 + 2*mu_k)^{-1/2}

        if prox_type == "l1":
            self.prox_list = nn.ModuleList([L1ProxFlat(dim=self.dual_dim) for _ in range(K)])
        else:
            self.prox_list = nn.ModuleList([small_MLP(dim=self.dual_dim, w=w, time_varying=True) for _ in range(K)])
        self.layers = nn.ModuleList([
            ScCP_Iteration(dim, self.prox_list[i], dual_dim=self.dual_dim, version=version, w_bias=w_bias) for i in range(K)
        ])

    def forward(self, xt_t):
        xt = xt_t[:, :self.dim]
        t  = xt_t[:, self.dim:]
        z  = xt
        x  = xt.clone()
        u  = torch.zeros(xt.shape[0], self.dual_dim, device=xt.device)
        
        if self.version == "LNO":
            taus = F.softplus(self.log_tau) # Shape: [K]
            for k, layer in enumerate(self.layers):
                tau_k = taus[k]
                alpha_k = (1.0 + 2.0 * tau_k).pow(-0.5) # rho=1.0 implicite
                
                # \tau_k dans le papier (qui correspond au pas dual sigma_k dans le code)
                sn = layer.spectral_norm()
                sigma_k = 0.99 / (tau_k * sn ** 2)
                
                x, u  = layer(x, u, z, t, tau_k, sigma_k, alpha_k)
        else: # LFO
            tau_k = F.softplus(self.log_tau0)
            sigma_k = 1.0 # Absorbé dans W_weight (Table 1)
            for layer in self.layers:
                alpha_k = (1.0 + 2.0 * tau_k).pow(-0.5)
                x, u  = layer(x, u, z, t, tau_k, sigma_k, alpha_k)
                
                # Mise à jour récursive de tau_k uniquement pour LFO
                tau_k = alpha_k * tau_k

        # if self.training:
        #     return x
        return x # (x - xt) / torch.clamp(1-t, 0.05)


class SharedScCP_UNN(nn.Module):
    """Accelerated CP UNN with shared flat weights across iterations.

    Same adaptive (tau, sigma, alpha) schedule as ScCP_UNN, but W/V (and the
    prox) are tied across all K steps, enabling variable iteration count at
    inference via `model(xt_t, n_iter=N)`.
    """
    def __init__(self, dim, K=10, w=32, dual_dim=None, version="LFO", prox_type="mlp"):
        super().__init__()
        self.dim      = dim
        self.K        = K
        self.version  = version
        self.dual_dim = dual_dim or dim
        self.predicts_x1 = True
        self.shared_W = nn.Parameter(kaiming_init(self.dual_dim, dim))
        if version == "LFO":
            # shared_V: (dim, dual_dim) so F.linear(u, V) with u:(B,dual_dim) → (B,dim)
            self.shared_V   = nn.Parameter(kaiming_init(dim, self.dual_dim))
            self.log_sigma0 = nn.Parameter(torch.tensor(-0.5))
        else:
            self.register_buffer('_sigma_u', F.normalize(torch.randn(self.dual_dim), dim=0))
        self.log_tau0 = nn.Parameter(torch.tensor(-0.5))
        self.log_rho  = nn.Parameter(torch.tensor(-1.0))
        if prox_type == "l1":
            self.prox = L1ProxFlat(dim=self.dual_dim)
        else:
            self.prox = small_MLP(dim=self.dual_dim, w=w, time_varying=True)

    def forward(self, xt_t, n_iter=None):
        iters = n_iter if n_iter is not None else self.K
        xt = xt_t[:, :self.dim]
        t  = xt_t[:, self.dim:]
        z  = xt
        x  = xt.clone()
        u  = torch.zeros(xt.shape[0], self.dual_dim, device=xt.device)

        V   = self.shared_V if self.version == "LFO" else self.shared_W.T
        tau = F.softplus(self.log_tau0)
        rho = F.softplus(self.log_rho)
        if self.version == "LFO":
            sigma = F.softplus(self.log_sigma0)
        else:
            _s, self._sigma_u = sigma_max_power_iter(self.shared_W, self._sigma_u)
            sigma = 0.99 / (tau * _s ** 2)

        for _ in range(iters):
            alpha         = (1.0 + 2.0 * rho * tau).pow(-0.5)
            primal_input  = x - tau * F.linear(u, V, None)
            x_next        = (primal_input + tau * z) / (1 + tau)
            y             = x_next + alpha * (x_next - x)
            dual_step     = sigma * F.linear(y, self.shared_W, None)
            u             = self.prox(torch.cat((u + dual_step, t), dim=-1))
            x             = x_next
            tau   = alpha * tau
            sigma = sigma / alpha
        if self.training:
            return x
        return (x - xt) / torch.clamp(1-t, 0.05)


class SharedConvScCP_UNN(nn.Module):
    """Accelerated CP UNN with shared convolutional weights across iterations.

    For LNO: sigma_0 = 0.99 / (tau_0 * ||W||^2), ensuring stability.
    Since sigma_k * tau_k = const, the constraint holds for all k.
    """
    def __init__(self, dim=784, K=10, internal_channel=64,
                 img_size=28, use_Unet=False, version="LFO"):
        super().__init__()
        self.dim              = dim
        self.K                = K
        self.internal_channel = internal_channel
        self.img_size         = img_size
        self.version          = version
        self.predicts_x1      = True
        self.shared_W = nn.Parameter(kaiming_init(internal_channel, 1, 3, 3))
        if version == "LFO":
            self.shared_V   = nn.Parameter(kaiming_init(internal_channel, 1, 3, 3))
            self.log_sigma0 = nn.Parameter(torch.tensor(-0.5))
        else:
            self.register_buffer('_sigma_u', F.normalize(torch.randn(internal_channel), dim=0))
        self.log_tau0   = nn.Parameter(torch.tensor(-0.5))
        self.log_rho    = nn.Parameter(torch.tensor(-1.0))
        if use_Unet is True or use_Unet == "small":
            self.prox = SmallUNet(
                in_channels=internal_channel, out_channels=internal_channel,
                base_ch=internal_channel // 2,
            )
            self.use_Unet = "small"
        elif use_Unet == "cfm":
            self.prox = UNetProxConv(in_channels=internal_channel, img_size=img_size)
            self.use_Unet = "cfm"
        elif use_Unet == "l1":
            self.prox = L1ProxConv()
            self.use_Unet = "l1"
        else:
            self.prox = DoubleConvTime(
                in_ch=internal_channel, out_ch=internal_channel,
                embed_dim=internal_channel // 2,
            )
            self.use_Unet = False

    def forward(self, xt_t, n_iter=None):
        iters      = n_iter if n_iter is not None else self.K
        batch_size = xt_t.shape[0]
        z = xt_t[:, :self.dim].view(batch_size, 1, self.img_size, self.img_size)
        t = xt_t[:, self.dim:]
        x = z.clone()
        u = torch.zeros(batch_size, self.internal_channel, self.img_size, self.img_size, device=z.device)
        tau = F.softplus(self.log_tau0)
        rho = F.softplus(self.log_rho)
        if self.version == "LFO":
            sigma = F.softplus(self.log_sigma0)
            V     = self.shared_V
        else:
            _s, self._sigma_u = sigma_max_power_iter(self.shared_W, self._sigma_u)
            sigma = 0.99 / (tau * _s ** 2)
            V     = self.shared_W
        for _ in range(iters):
            alpha = (1.0 + 2.0 * rho * tau).pow(-0.5)
            # Primal update
            grad_u       = F.conv_transpose2d(u, V, padding=1)
            primal_input = x - tau * grad_u
            x_next       = ((primal_input + tau * z) / (1 + tau)).contiguous()
            # Extrapolation with momentum alpha_k <= 1
            y = (x_next + alpha * (x_next - x)).contiguous()
            # Dual update on extrapolated point y
            dual_step = (sigma * F.conv2d(y, self.shared_W, padding=1)).contiguous()
            if self.use_Unet == "small":
                u_flat    = (u + dual_step).view(batch_size, -1, 784)
                ut_t_flat = torch.cat(
                    [u_flat, t[:, 0].view(batch_size, 1, 1).expand(batch_size, self.internal_channel, 1)],
                    dim=-1,
                )
                u = self.prox(ut_t_flat).view(batch_size, self.internal_channel, self.img_size, self.img_size)
            else:
                u = self.prox(u + dual_step, t)
            x     = x_next
            tau   = alpha * tau
            sigma = sigma / alpha
        if self.training:
            return x.view(batch_size, -1)
        return (x - z).view(batch_size, -1) / torch.clamp(1 - t, 0.05)


# =========================================================================== #
#  ConvScCP_UNN ORIGINAL (extrait exact de git HEAD) : LE modele qui produisait
#  des chiffres flous. PAS de lifting analyse/synthese. Primal = image 1 canal ;
#  W : 1 -> internal_channel (pour le dual) ; init *0.05 ; sortie residu (x - z)
#  en v-pred. Iteration renommee pour ne pas collisionner avec la ConvScCP_Iteration
#  active (version latente). Sert de reference "modele qui marche".
# =========================================================================== #
class _OrigConvScCP_Iteration(nn.Module):
    """Single accelerated CP step with convolutional W (version d'origine, HEAD).
    init : "small" (randn*0.05, comportement HEAD d'origine) ou "kaiming".
    in_channels : nombre de canaux de l'image primale (1 = MNIST, 3 = RGB/ImageNet).
    W : in_channels -> internal_channel (mixing des canaux couleur automatique)."""
    def __init__(self, internal_channel, use_Unet=False, version="LFO", init=None,
                 w_bias=True, in_channels=1, kernel_size=9, prox_w=32,
                 prox_channels=False):
        """prox_channels : rayon du prox l1 appris PAR CANAL dual au lieu d'un
        scalaire partage (variante "l1c"). Reste un prox l1 exact — celui de la
        norme ponderee ||u||_{1,w} = sum_c w_c ||u_c||_1 — donc la propriete
        "la sortie est un operateur proximal" est preservee."""
        super().__init__()
        self.version  = version
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1   # force impair
        self.pad = k // 2                                              # conv "same" -> RF ~ 2*(k//2)*K
        if init is None:
                init = "kaiming" if version == "LFO" else "kaiming"
        def _init_W():
            if init == "kaiming":
                return kaiming_init(internal_channel, in_channels, k, k)
            return torch.randn(internal_channel, in_channels, k, k) * 0.05
        self.W_weight = nn.Parameter(_init_W())
        if version == "LFO":
            self.V_weight = nn.Parameter(_init_W())
        else:
            self.register_buffer('_sigma_u', F.normalize(torch.randn(internal_channel), dim=0))
        # variante (i) : biais convolutif appris sur W (penalise g(W·-b)) -> non-impair
        # en fonction de l'entree. Zero-init => demarre exactement symetrique.
        self.W_bias = nn.Parameter(torch.zeros(internal_channel)) if w_bias else None
        if use_Unet == "l1":
            self.prox = L1ProxConv(
                w=prox_w, channels=(internal_channel if prox_channels else None))
        elif use_Unet == "silu":
            self.prox = SiLUProxConv(channels=internal_channel)   # prox SiLU pointwise, conditionné t
        else:
            self.prox = DoubleConvTime(
                in_ch=internal_channel, out_ch=internal_channel,
                embed_dim=internal_channel // 2,
            )

    def spectral_norm(self):
        _s, self._sigma_u = sigma_max_power_iter(self.W_weight, self._sigma_u)
        return _s
    
    def forward(self, x, u, z, t, tau, sigma, alpha_mom):
        V = self.V_weight if self.version == "LFO" else self.W_weight
        primal_input = x - tau * F.conv_transpose2d(u, V, padding=self.pad)
        x_next = (primal_input + tau * z) / (1 + tau)
        y         = x_next + alpha_mom * (x_next - x)
        dual_step = sigma * F.conv2d(y, self.W_weight, bias=self.W_bias, padding=self.pad)
        u_next    = self.prox(u + dual_step, t)
        return x_next, u_next

def weights_init_kaiming(m):
        """Init kaiming (style DnCNN), applique via model.apply().

        Le test se fait sur le TYPE et pas sur le nom de classe : des modules
        conteneurs comme ConvScCP_UNN, DoubleConvTime ou L1ProxConv contiennent
        "Conv" dans leur nom mais n'ont pas de .weight (ou un .weight qui n'est
        pas celui d'une convolution)."""
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                          nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
                          nn.Linear)):
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.weight is not None:            # affine=False -> pas de weight/bias
                m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
                nn.init.constant_(m.bias.data, 0.0)

def fm_velocity_denom(t, t_max=None):
    """Denominateur (1-t) de la conversion x-pred -> vitesse  v = (x1_pred - z)/(1-t).

    t_max=None : comportement HISTORIQUE, clamp min=0.05. Ce clamp MORD des t > 0.95 :
        la vitesse renvoyee y est trop PETITE d'un facteur (1-t)/0.05 (x0.2 a t=0.99),
        donc la fin de trajectoire n'arrive jamais a x1.
    t_max=T : aucun clamp, mais un ASSERT. Le modele n'a ete entraine que sur
        t in [0, T] (cf. run_cifar10_torchcfm_recipe --euler-steps) ; l'echantillonneur
        ne doit jamais evaluer au-dela, sinon 1/(1-t) extrapole hors distribution.
        Avec Euler-N sur [0,1] le plus grand t evalue est 1 - 1/N, d'ou T = 1 - 1/N :
        Euler-10 -> 0.90,  Euler-50 -> 0.98,  Euler-100 -> 0.99.
    """
    if t_max is None:
        return torch.clamp(1 - t, min=0.05)
    tm = float(t.max())
    assert tm <= t_max + 1e-6, (
        f"t={tm:.6f} hors du domaine d'entrainement [0, {t_max}] : l'echantillonneur "
        f"evalue un t jamais vu et 1/(1-t) y explose. Reduis le t_span du sampler "
        f"(Euler-N -> t_max = 1-1/N) ou reentraine avec un t_max plus grand.")
    return 1 - t

class ConvScCP_UNN(nn.Module):
    """ScCP convolutif (base git HEAD : primal 1 canal, dual internal_channel,
    W:1->C init *0.05, prox scalaire). predicts_x1=True (x-pred) : training renvoie
    x~x1 (loss ||x-x1||²), eval renvoie la vitesse (x-z)/(1-t). Mettre predicts_x1=False
    + retirer la branche training pour retrouver le v-pred d'origine (sortie (x-z))."""
    def __init__(self, dim=784, K=10, internal_channel=64,
                 use_Unet=False, version="LFO", use_checkpoint=False, init="kaiming",
                 w_bias=True, in_channels=1, img_size=28, kernel_size=9, prox_w=32,
                 prox_channels=False):
        super().__init__()
        self.dim              = dim
        self.K                = K
        self.internal_channel = internal_channel
        self.in_channels      = in_channels    # 1 = MNIST (gris) ; 3 = RGB / ImageNet-32
        self.img_size         = img_size       # 28 = MNIST ; 32 = ImageNet-32
        self.kernel_size      = kernel_size    # 9 = defaut HEAD ; 3 = regime local facon Kamb ResNet
        self.version          = version
        self.use_checkpoint   = use_checkpoint
        self.predicts_x1      = True
        # None = clamp historique min=0.05 a l'eval. Mis a T par le script
        # d'entrainement quand t est tire sur [0, T] : la conversion en
        # vitesse devient alors EXACTE (cf. fm_velocity_denom). Attribut
        # simple et non un buffer : les ckpt existants rechargent strict=True.
        self.t_max            = None
        assert dim == in_channels * img_size * img_size, (
            f"dim={dim} incohérent avec in_channels={in_channels}, img_size={img_size} "
            f"(attendu {in_channels * img_size * img_size})")
        if version == "LNO":
            self.log_tau = nn.Parameter(torch.full((K,), -0.5))
        else:
            self.log_tau0 = nn.Parameter(torch.tensor(-0.5))
        self.layers = nn.ModuleList([
            _OrigConvScCP_Iteration(internal_channel=internal_channel,
                                    use_Unet=use_Unet, version=version, init=init,
                                    w_bias=w_bias, in_channels=in_channels,
                                    kernel_size=kernel_size, prox_w=prox_w,
                                    prox_channels=prox_channels)
            for _ in range(K)
        ])

    def cold_dual(self, x):
        """Dual "froid" u^(0) = 0 de la bonne forme, (B, internal_channel, S, S).
        `x` sert seulement a lire le batch, le device et le dtype : accepte aussi bien
        (B, dim) / (B, dim+1) que (B, C, S, S). Evite aux appelants de construire le
        tenseur de zeros a la main (et donc de se tromper de forme)."""
        return x.new_zeros(x.shape[0], self.internal_channel, self.img_size, self.img_size)

    def forward(self, xt_t, return_iterates=False, u_init=None, return_u=False):
        """return_iterates : renvoie aussi les itérés primaux internes du déroulé CP,
        [x^(0)=z, x^(1), ..., x^(K)] empilés en (K+1, B, C, S, S) — outil d'analyse,
        n'affecte pas la sortie renvoyée (voir trajectory_convsccp.py).

        u_init : état dual initial du déroulé, (B, internal_channel, S, S). None
        (défaut) = warm-start standard u^(0)=0, comportement historique inchangé.
        Passer le u^(K) d'un appel précédent = TRANSFERT DE L'ESPACE LATENT d'un pas
        d'Euler au suivant (voir sample_sccp_utransfer.py).
        return_u : renvoie aussi le dual final u^(K), pour le rechaîner.

        Sortie : out, puis dans l'ordre les extras demandés (iterates, u)."""
        batch_size = xt_t.shape[0]
        C, S = self.in_channels, self.img_size
        z = xt_t[:, :self.dim].view(batch_size, C, S, S)
        t = xt_t[:, self.dim:]
        x = z.clone()
        if u_init is None:
            u = z.new_zeros(batch_size, self.internal_channel, S, S)
        else:
            assert u_init.shape == (batch_size, self.internal_channel, S, S), (
                f"u_init de forme {tuple(u_init.shape)}, attendu "
                f"{(batch_size, self.internal_channel, S, S)}")
            u = u_init.to(z.device)
        iterates = [x.clone()] if return_iterates else None
        if self.version == "LNO":
            taus = F.softplus(self.log_tau)
            for k, layer in enumerate(self.layers):
                tau_k = taus[k]
                alpha_k = (1.0 + 2.0 * tau_k).pow(-0.5)
                sn = layer.spectral_norm()
                sigma_k = 0.99 / (tau_k * sn ** 2)
                if self.use_checkpoint and self.training:
                    x, u = checkpoint(layer, x, u, z, t, tau_k, sigma_k, alpha_k, use_reentrant=False)
                else:
                    x, u = layer(x, u, z, t, tau_k, sigma_k, alpha_k)
                if return_iterates:
                    iterates.append(x.clone())
        else:
            tau_k = F.softplus(self.log_tau0)
            sigma_k = 1.0
            for layer in self.layers:
                alpha_k = (1.0 + 2.0 * tau_k).pow(-0.5)
                if self.use_checkpoint and self.training:
                    x, u = checkpoint(layer, x, u, z, t, tau_k, sigma_k, alpha_k, use_reentrant=False)
                else:
                    x, u = layer(x, u, z, t, tau_k, sigma_k, alpha_k)
                tau_k = alpha_k * tau_k
                if return_iterates:
                    iterates.append(x.clone())

        if self.training:
            out = x.view(batch_size, -1)                  # x-pred : x ~ x1 (loss vs x1)
        else:
            # eval : conversion en vitesse FM  v = (x1_pred - z)/(1 - t)
            out = (x - z).view(batch_size, -1) / fm_velocity_denom(t, self.t_max)
        extras = []
        if return_iterates:
            extras.append(torch.stack(iterates))
        if return_u:
            extras.append(u)
        if extras:
            return (out, *extras)
        return out



# =========================================================================== #
#  ConvScCP v2 : le terme d'attache du VRAI probleme inverse
# =========================================================================== #
class _ScCPv2_Iteration(_OrigConvScCP_Iteration):
    """Iteration ScCP avec le prox du bon terme d'attache.

    v1 (_OrigConvScCP_Iteration) suppose  f(x) = 1/2 ||x_t - x||^2, c'est-a-dire
    une observation a GAIN UNITE sur le signal. Or le modele direct du Flow
    Matching est x_t = t.x1 + (1-t).eps : operateur A = t.I, bruit sigma = 1-t.
    Le terme d'attache correct est donc

        f(x) = ||x_t - t.x||^2 / (2(1-t)^2)

    dont le prox a pour forme fermee (verifiee par la condition d'optimalite dans
    verif_fidelite_sccp.py, residu 2e-11 en float64) :

        x_next = [ (1-t)^2 . v  +  tau . t . x_t ] / [ (1-t)^2 + tau . t^2 ]

    Aucune division par t ni par (1-t) : a t->0 l'attache s'eteint d'elle-meme
    (x_t ne dit rien de x1), a t->1 elle domine (x_t = x1).

    SOUS-CLASSE et non copie : W, V, W_bias et le prox sont hérités tels quels,
    donc v1 et v2 ont exactement les memes parametres et la meme initialisation.
    Seule la ligne du primal change — c'est ce qui rend la comparaison propre.
    """

    def forward(self, x, u, z, t, tau, sigma, alpha_mom):
        V = self.V_weight if self.version == "LFO" else self.W_weight
        primal_input = x - tau * F.conv_transpose2d(u, V, padding=self.pad)
        tt = t.view(-1, 1, 1, 1)
        s2 = (1.0 - tt) ** 2
        x_next = (s2 * primal_input + tau * tt * z) / (s2 + tau * tt ** 2)
        y = x_next + alpha_mom * (x_next - x)
        dual_step = sigma * F.conv2d(y, self.W_weight, bias=self.W_bias, padding=self.pad)
        u_next = self.prox(u + dual_step, t)
        return x_next, u_next


class ConvScCP_UNN_v2(nn.Module):
    """ConvScCP_UNN avec le terme d'attache derive du vrai probleme inverse.

    Trois differences avec ConvScCP_UNN, toutes derivees, aucune reglee a la main :

    1. mise a jour primale : cf. _ScCPv2_Iteration.

    2. momentum. Le schema accelere de Chambolle-Pock est pilote par la constante
       de forte convexite de f. Pour v1, f_z est 1-fortement convexe, d'ou
       alpha_k = (1 + 2 tau_k)^(-1/2). Pour v2, la hessienne vaut t^2/(1-t)^2 . I :

           mu(t) = t^2 / (1-t)^2        alpha_k = (1 + 2 mu(t) tau_k)^(-1/2)

       mu = 1 SEULEMENT a t = 0.5 : v1 est donc la methode acceleree du bon
       probleme en un unique point de l'axe des temps. tau_k et alpha_k deviennent
       des tenseurs (B,1,1,1) puisqu'ils dependent de t.

       C'est ce point-la qui justifie v2. Une erreur d'echelle sur l'OBJECTIF se
       reporte sur g, qui est appris et conditionne en t (mesure : la sortie de v1
       est calibree a 1 % pres, cf. analyse_input_scaling.py). Le calendrier
       d'acceleration, lui, est dans la DYNAMIQUE : tau_k est appris mais
       independant de t, donc rien ne peut y suppleer.

    3. x0. Dans v1, x0 = z etait coherent avec l'accroche sur x_t. Ici l'attache
       s'eteint a t->0, donc partir de x_t y laisserait la sortie sur du bruit
       alors que la bonne reponse est la moyenne du prior. x0_mode="zero" (defaut)
       part de 0 ; "xt" reproduit le choix de v1, pour pouvoir le tester.

    Le reste — W, V, prox, dual, conversion en vitesse a l'eval — est identique.
    """

    def __init__(self, dim=784, K=10, internal_channel=64,
                 use_Unet=False, version="LFO", use_checkpoint=False, init="kaiming",
                 w_bias=True, in_channels=1, img_size=28, kernel_size=9, prox_w=32,
                 prox_channels=False, x0_mode="zero"):
        super().__init__()
        self.dim              = dim
        self.K                = K
        self.internal_channel = internal_channel
        self.in_channels      = in_channels
        self.img_size         = img_size
        self.kernel_size      = kernel_size
        self.version          = version
        self.use_checkpoint   = use_checkpoint
        self.predicts_x1      = True
        self.t_max            = None      # cf. fm_velocity_denom
        self.x0_mode          = x0_mode
        assert x0_mode in ("zero", "xt"), f"x0_mode={x0_mode} inconnu"
        assert dim == in_channels * img_size * img_size
        if version == "LNO":
            self.log_tau = nn.Parameter(torch.full((K,), -0.5))
        else:
            self.log_tau0 = nn.Parameter(torch.tensor(-0.5))
        self.layers = nn.ModuleList([
            _ScCPv2_Iteration(internal_channel=internal_channel,
                              use_Unet=use_Unet, version=version, init=init,
                              w_bias=w_bias, in_channels=in_channels,
                              kernel_size=kernel_size, prox_w=prox_w,
                              prox_channels=prox_channels)
            for _ in range(K)
        ])

    def cold_dual(self, x):
        return x.new_zeros(x.shape[0], self.internal_channel, self.img_size, self.img_size)

    def forward(self, xt_t, return_iterates=False, u_init=None, return_u=False):
        B = xt_t.shape[0]
        C, S = self.in_channels, self.img_size
        z = xt_t[:, :self.dim].view(B, C, S, S)
        t = xt_t[:, self.dim:]
        tt = t.view(-1, 1, 1, 1)
        # mu(t) = t^2/(1-t)^2. Le clamp_min protege le seul endroit du modele ou
        # (1-t) apparait au denominateur ; il est inerte pour t <= 1 - 1e-4.
        mu = tt ** 2 / (1.0 - tt).pow(2).clamp_min(1e-8)

        x = z.clone() if self.x0_mode == "xt" else torch.zeros_like(z)
        u = self.cold_dual(z) if u_init is None else u_init.to(z.device)
        iterates = [x.clone()] if return_iterates else None

        if self.version == "LNO":
            taus = F.softplus(self.log_tau)
            for k, layer in enumerate(self.layers):
                tau_k = taus[k].view(1, 1, 1, 1).expand_as(mu)
                alpha_k = (1.0 + 2.0 * mu * tau_k).pow(-0.5)
                sn = layer.spectral_norm()
                sigma_k = 0.99 / (tau_k * sn ** 2)
                if self.use_checkpoint and self.training:
                    x, u = checkpoint(layer, x, u, z, t, tau_k, sigma_k, alpha_k,
                                      use_reentrant=False)
                else:
                    x, u = layer(x, u, z, t, tau_k, sigma_k, alpha_k)
                if return_iterates:
                    iterates.append(x.clone())
        else:
            tau_k = F.softplus(self.log_tau0).view(1, 1, 1, 1).expand_as(mu)
            sigma_k = 1.0
            for layer in self.layers:
                alpha_k = (1.0 + 2.0 * mu * tau_k).pow(-0.5)
                if self.use_checkpoint and self.training:
                    x, u = checkpoint(layer, x, u, z, t, tau_k, sigma_k, alpha_k,
                                      use_reentrant=False)
                else:
                    x, u = layer(x, u, z, t, tau_k, sigma_k, alpha_k)
                tau_k = alpha_k * tau_k
                if return_iterates:
                    iterates.append(x.clone())

        if self.training:
            out = x.view(B, -1)
        else:
            out = (x - z).view(B, -1) / fm_velocity_denom(t, self.t_max)
        extras = []
        if return_iterates:
            extras.append(torch.stack(iterates))
        if return_u:
            extras.append(u)
        return (out, *extras) if extras else out



# =========================================================================== #
#  ConvScCP v3 : v2 + la condition de pas de Chambolle-Pock, cas LFO
#  Derivation complete : DERIVATION_LFO.md
# =========================================================================== #
class _ScCPv3_Iteration(_ScCPv2_Iteration):
    """Iteration de v2 (bon terme d'attache) + le calcul correct du gain de boucle.

    En LFO le pas primal utilise V et le pas dual utilise W, independants : ce
    n'est plus Chambolle-Pock, qui exige K et K*. La quantite qui gouverne la
    stabilite est le gain de la boucle primal->dual->primal,

        A x = conv2d(x, W, p)                  X -> U   (pas dual)
        B u = conv_transpose2d(u, V, p)        U -> X   (pas primal)
        L^2 := ||B o A||_op

    qui redonne ||A||^2 quand B = A* (donc LNO). On l'estime par iteration de la
    puissance sur M* M, avec M = B o A et

        M  (x) = conv_transpose2d( conv2d(x, W, p), V, p )
        M* (y) = conv_transpose2d( conv2d(y, V, p), W, p )

    en utilisant que conv_transpose2d(., W, p) est EXACTEMENT l'adjoint de
    conv2d(., W, p) a stride 1 et padding egal (verifie a 0 pres en float64).

    NB : sigma_max_power_iter, utilise par LNO, aplatit W en (C_out, C_in.k^2) et
    en prend la plus grande valeur singuliere. Ce n'est PAS la norme d'operateur
    de la convolution : mesure, le reshape sous-estime d'un facteur ~1.7, donc
    LNO autorise des pas trop grands. Ici on evite ce raccourci.
    """

    def loop_gain(self, img_size, n_iter=1):
        """L^2 = ||B o A|| par iteration de la puissance, buffer persistant warm-start.

        UNE iteration par forward suffit (meme discipline que torch.nn.utils
        spectral_norm) : entre deux steps d'optimisation W et V bougent peu, donc le
        vecteur propre est deja presque convergé. Le Mv calcule pour l'iteration sert
        aussi a lire la norme -> 4 convolutions par appel, pas 10.

        ||Mv||/||v|| avec v unitaire converge vers ||M|| PAR EN DESSOUS, donc
        l'estimation est optimiste tant qu'elle n'a pas converge. Au demarrage on
        fait donc plusieurs tours pour partir d'une valeur juste."""
        W = self.W_weight
        V = self.V_weight if self.version == "LFO" else self.W_weight
        c_in = W.shape[1]
        fresh = getattr(self, "_pi_v", None) is None or self._pi_v.shape[-1] != img_size
        if fresh:
            v = torch.randn(1, c_in, img_size, img_size, device=W.device, dtype=W.dtype)
            self._pi_v = F.normalize(v.flatten(), dim=0).view_as(v)
        v = self._pi_v.to(W.device, W.dtype)
        with torch.no_grad():
            for _ in range(30 if fresh else n_iter):     # amorcage soigne, puis 1/forward
                Mv = F.conv_transpose2d(F.conv2d(v, W, padding=self.pad), V,
                                        padding=self.pad)
                MtMv = F.conv_transpose2d(F.conv2d(Mv, V, padding=self.pad), W,
                                          padding=self.pad)
                v = F.normalize(MtMv.flatten(), dim=0).view_as(v)
            self._pi_v = v
            l2 = Mv.flatten().norm() / 1.0              # v est unitaire par construction
        if getattr(self, "diff_l2", False):
            # variante DIFFERENTIABLE : vecteur propre detache, quotient de Rayleigh
            # garde dans le graphe (cf. _ScCPv4_Iteration.loop_gain). Sert a isoler
            # la correction du gradient des autres changements de v4/v5.
            vd = self._pi_v.detach()
            Mvd = F.conv_transpose2d(F.conv2d(vd, W, padding=self.pad), V,
                                     padding=self.pad)
            return Mvd.flatten().norm().clamp_min(1e-8)
        return l2.clamp_min(1e-8)


class ConvScCP_UNN_v3(nn.Module):
    """v2 + la condition de pas retablie. Cf. DERIVATION_LFO.md.

    v2 avait corrige le terme d'attache et la constante de forte convexite mu(t),
    mais avait garde de v1 un defaut invisible : sigma_k = 1.0, jamais mis a jour.
    Or dans ALG2 de Chambolle-Pock c'est sigma_{k+1} = sigma_k/theta_k qui preserve
    l'invariant tau_k.sigma_k = cte, donc la condition tau.sigma.L^2 <= 1. Sans lui,
    tau s'effondre (theta = 0.114 a t=0.9, ou mu = 81) pendant que sigma reste a 1 :
    le deroule se reduit a un pas utile. C'est la bosse mesuree a t = 0.85-0.90.

    Ici :
        L^2     = loop_gain()                 gain de boucle, calcule correctement
        tau_0   = softplus(log_tau0)          LIBRE, comme v1 : la condition ne
                                              contraint que le produit tau.sigma
        sigma_k = 0.99 / (tau_k L^2)          -> tau_k sigma_k = 0.99/L^2 pour tout k
        theta_k = (1 + 2 mu(t) tau_k)^(-1/2)  mu(t) = t^2/(1-t)^2
        tau_{k+1} = theta_k tau_k             -> sigma croit comme 1/theta, gratuitement

    Le reste (W, V, prox, x0_mode, conversion en vitesse) est celui de v2.
    """

    def __init__(self, dim=784, K=10, internal_channel=64,
                 use_Unet=False, version="LFO", use_checkpoint=False, init="kaiming",
                 w_bias=True, in_channels=1, img_size=28, kernel_size=9, prox_w=32,
                 prox_channels=False, x0_mode="zero", cp_safety=0.99,
                 diff_loop_gain=False):
        super().__init__()
        self.dim              = dim
        self.K                = K
        self.internal_channel = internal_channel
        self.in_channels      = in_channels
        self.img_size         = img_size
        self.kernel_size      = kernel_size
        self.version          = version
        self.use_checkpoint   = use_checkpoint
        self.predicts_x1      = True
        self.t_max            = None
        self.x0_mode          = x0_mode
        self.cp_safety        = cp_safety
        assert x0_mode in ("zero", "xt")
        assert dim == in_channels * img_size * img_size
        # tau_0 appris LIBREMENT, meme initialisation que v1 (softplus(-0.5) ~ 0.474).
        # La condition de pas ne contraint que le PRODUIT tau.sigma ; tau lui-meme
        # est libre. Une version anterieure posait tau_0 = gamma/L "pour rendre le
        # parametre sans dimension" : erreur. Le poids du terme d'attache dans la
        # mise a jour primale vaut tau.t/((1-t)^2 + tau.t^2), qui n'est PAS invariant
        # d'echelle en tau. Diviser par L ~ 17.7 le faisait tomber de 0.144 a 0.0084
        # a t=0.2 : attache eteinte, deroule inerte, puis emballement (L^2 x89 en
        # 1500 steps). Historique dans DERIVATION_LFO.md §6.
        self.log_tau0 = nn.Parameter(torch.tensor(-0.5))
        self.layers = nn.ModuleList([
            _ScCPv3_Iteration(internal_channel=internal_channel,
                              use_Unet=use_Unet, version=version, init=init,
                              w_bias=w_bias, in_channels=in_channels,
                              kernel_size=kernel_size, prox_w=prox_w,
                              prox_channels=prox_channels)
            for _ in range(K)
        ])
        for lay in self.layers:
            lay.diff_l2 = diff_loop_gain


    def cold_dual(self, x):
        return x.new_zeros(x.shape[0], self.internal_channel, self.img_size, self.img_size)

    def forward(self, xt_t, return_iterates=False, u_init=None, return_u=False,
                return_steps=False):
        B = xt_t.shape[0]
        C, S = self.in_channels, self.img_size
        z = xt_t[:, :self.dim].view(B, C, S, S)
        t = xt_t[:, self.dim:]
        tt = t.view(-1, 1, 1, 1)
        mu = tt ** 2 / (1.0 - tt).pow(2).clamp_min(1e-8)

        x = z.clone() if self.x0_mode == "xt" else torch.zeros_like(z)
        u = self.cold_dual(z) if u_init is None else u_init.to(z.device)
        iterates = [x.clone()] if return_iterates else None
        steps = [] if return_steps else None

        tau_k = F.softplus(self.log_tau0).view(1, 1, 1, 1).expand_as(mu)
        for layer in self.layers:
            l2 = layer.loop_gain(S)                       # L^2 de CETTE iteration
            sigma_k = self.cp_safety / (tau_k * l2)       # tau_k.sigma_k.L^2_k = safety
            alpha_k = (1.0 + 2.0 * mu * tau_k).pow(-0.5)
            if return_steps:
                steps.append((float(l2), tau_k.flatten()[0].item(),
                              sigma_k.flatten()[0].item(), alpha_k.flatten()[0].item()))
            if self.use_checkpoint and self.training:
                x, u = checkpoint(layer, x, u, z, t, tau_k, sigma_k, alpha_k,
                                  use_reentrant=False)
            else:
                x, u = layer(x, u, z, t, tau_k, sigma_k, alpha_k)
            tau_k = alpha_k * tau_k                       # ALG2 : tau_{k+1} = theta_k tau_k
            if return_iterates:
                iterates.append(x.clone())

        if self.training:
            out = x.view(B, -1)
        else:
            out = (x - z).view(B, -1) / fm_velocity_denom(t, self.t_max)
        extras = []
        if return_iterates:
            extras.append(torch.stack(iterates))
        if return_u:
            extras.append(u)
        if return_steps:
            extras.append(steps)
        return (out, *extras) if extras else out



# =========================================================================== #
#  ConvScCP v4 : conforme au spec "ScCP with direct x_t input", variante A
#  §3 rho = t^2 (borne)  ·  §4 rayon du prox en (1-t)^gamma
#  + deux corrections etablies par la mesure : gain de boucle ||BoA|| au lieu de
#    ||W||_S^2, et l2 DIFFERENTIABLE (cf. DERIVATION_LFO.md)
# =========================================================================== #
class L1ProxRadiusPow(nn.Module):
    """prox_{sigma g*} pour g(y) = mu_t ||y||_1.

    Le conjugue d'une norme l1 ponderee est l'indicatrice de la boule l_inf de
    rayon mu_t, dont le prox est la projection : un clamp au rayon mu_t,
    INDEPENDANT de sigma. C'est ce que fait deja L1ProxConv ; ce qui change ici
    est la PARAMETRISATION du rayon.

    La theorie prescrit mu_t proportionnel a (1-t)^2 — une VARIANCE, pas un
    ecart-type. D'ou

        r(t) = softplus(a) * (1-t)^gamma,   gamma = softplus(gamma_hat), init 2.0

    au lieu du MLP libre de L1ProxConv. L'exposant est appris plutot que fixe : a
    profondeur tronquee (K ~ 5..15) le seuil optimal par couche n'est pas le seuil
    MAP exact, donc une correction apprise reste defendable. Un gamma qui reste
    proche de 2 corrobore la lecture MAP ; une derive systematique est elle-meme
    un resultat sur la troncature — d'ou `gamma_value()`, a journaliser.

    Comportements aux bords (testes dans test_sccp_v4.py) :
      t -> 1 : r -> 0, dual ecrase a 0, le primal se reduit a l'identite x -> x_t.
               Correct : il n'y a rien a debruiter.
      t -> 0 : r -> softplus(a), regularisation maximale, la sortie tend vers le
               mode du prior. Correct : x_t ne dit rien de x_1.
    """

    def __init__(self, channels=None):
        super().__init__()
        out = channels if channels is not None else 1
        self.channels = channels
        self.a = nn.Parameter(torch.full((out,), 0.541325))
        self.gamma_hat = nn.Parameter(torch.tensor(1.854587))

    def gamma_value(self):
        return F.softplus(self.gamma_hat)

    def radius(self, t):
        """r(t), forme (B, C, 1, 1) ou (B, 1, 1, 1)."""
        g = F.softplus(self.gamma_hat)
        base = (1.0 - t).clamp_min(0.0)                       # (B,1)
        return (F.softplus(self.a).view(1, -1) * base.pow(g)).view(
            t.shape[0], -1, 1, 1)

    def forward(self, u, t):
        r = self.radius(t)
        return torch.clamp(u, -r, r)


class _ScCPv4_Iteration(_OrigConvScCP_Iteration):
    """Iteration du spec, variante A.

        x_{k+1} = (x_k - tau_k V u_k + tau_k t x_t) / (1 + tau_k t^2)

    soit le prox de f(x) = 1/2 ||x_t - t x||^2, dont la hessienne vaut t^2.I :
    rho = t^2, BORNE dans [0,1]. C'est la difference decisive avec v2/v3, qui
    mettaient le 1/(1-t)^2 du cote de l'attache et obtenaient mu = t^2/(1-t)^2,
    divergent en t -> 1 — donc alpha -> 0, tau effondre, deroule inerte (bosse
    mesuree a t = 0.85-0.90).

    Les deux formulations sont le meme probleme a un facteur (1-t)^2 pres sur
    l'objectif ; elles ne coincident que si tau_spec = tau_v3/(1-t)^2. Comme les
    deux apprennent un tau_0 constant en t, elles different reellement.
    """

    def loop_gain(self, img_size, n_iter=1):
        """L^2 = ||B o A||, DIFFERENTIABLE en (W, V).

        Le vecteur propre est mis a jour sous no_grad puis DETACHE ; le quotient
        de Rayleigh ||M v|| reste dans le graphe. d||M||/dW est alors correct
        (theoreme de l'enveloppe), meme technique que torch spectral_norm.

        v3 renvoyait un l2 entierement detache : le gradient croyait alors que
        grossir W augmentait le signal dual, alors que sigma = 0.99/(tau L^2)
        compense exactement. Tapis roulant mesure : ||W|| x25 et L^2 x27000 en
        20k steps, sigma effondre a 0, modele debranche.
        """
        W = self.W_weight
        V = self.V_weight if self.version == "LFO" else self.W_weight
        fresh = getattr(self, "_pi_v", None) is None or self._pi_v.shape[-1] != img_size
        if fresh:
            v0 = torch.randn(1, W.shape[1], img_size, img_size,
                             device=W.device, dtype=W.dtype)
            self._pi_v = F.normalize(v0.flatten(), dim=0).view_as(v0)
        with torch.no_grad():
            v = self._pi_v.to(W.device, W.dtype)
            for _ in range(30 if fresh else n_iter):
                Mv = F.conv_transpose2d(F.conv2d(v, W, padding=self.pad), V,
                                        padding=self.pad)
                MtMv = F.conv_transpose2d(F.conv2d(Mv, V, padding=self.pad), W,
                                          padding=self.pad)
                v = F.normalize(MtMv.flatten(), dim=0).view_as(v)
            self._pi_v = v
        v = self._pi_v.detach()
        Mv = F.conv_transpose2d(F.conv2d(v, W, padding=self.pad), V, padding=self.pad)
        return Mv.flatten().norm().clamp_min(1e-8)

    def forward(self, x, u, z, t, tau, sigma, alpha_mom):
        V = self.V_weight if self.version == "LFO" else self.W_weight
        primal_input = x - tau * F.conv_transpose2d(u, V, padding=self.pad)
        tt = t.view(-1, 1, 1, 1)
        x_next = (primal_input + tau * tt * z) / (1.0 + tau * tt ** 2)
        y = x_next + alpha_mom * (x_next - x)
        dual_step = sigma * F.conv2d(y, self.W_weight, bias=self.W_bias,
                                     padding=self.pad)
        u_next = self.prox(u + dual_step, t)
        return x_next, u_next


class ConvScCP_UNN_v4(nn.Module):
    """ScCP conforme au spec "ScCP with direct x_t input", variante A.

    Par rapport a v3 :
      §3  rho = t^2 au lieu de mu = t^2/(1-t)^2. Borne, donc plus d'effondrement
          de tau en t -> 1.
      §4  rayon du prox r(t) = softplus(a).(1-t)^gamma, gamma appris (init 2),
          au lieu du MLP libre de L1ProxConv. Le rayon est une VARIANCE.
    Conserve de v3, parce que mesure :
      - L^2 = ||B o A|| (gain de boucle) et non ||W||_S^2 : le reshape de
        sigma_max_power_iter sous-estime la norme d'operateur d'un facteur 2 a 3.6.
      - l2 DIFFERENTIABLE : sans ca le gradient est faux et W s'emballe.

    sigma_k = cp_safety / (tau_k L^2) donne tau_k.sigma_k.L^2 = cp_safety pour
    tout k et tout echantillon, et sigma croit comme 1/alpha sans code dedie.
    """

    def __init__(self, dim=784, K=10, internal_channel=64,
                 use_Unet=False, version="LFO", use_checkpoint=False, init="kaiming",
                 w_bias=True, in_channels=1, img_size=28, kernel_size=9, prox_w=32,
                 prox_channels=False, x0_mode="xt", cp_safety=0.99):
        super().__init__()
        self.dim              = dim
        self.K                = K
        self.internal_channel = internal_channel
        self.in_channels      = in_channels
        self.img_size         = img_size
        self.kernel_size      = kernel_size
        self.version          = version
        self.use_checkpoint   = use_checkpoint
        self.predicts_x1      = True
        self.t_max            = None
        self.x0_mode          = x0_mode
        self.cp_safety        = cp_safety
        assert x0_mode in ("zero", "xt")
        assert dim == in_channels * img_size * img_size
        self.log_tau0 = nn.Parameter(torch.tensor(-0.5))
        self.layers = nn.ModuleList([
            _ScCPv4_Iteration(internal_channel=internal_channel,
                              use_Unet=use_Unet, version=version, init=init,
                              w_bias=w_bias, in_channels=in_channels,
                              kernel_size=kernel_size, prox_w=prox_w,
                              prox_channels=prox_channels)
            for _ in range(K)
        ])
        # §4 : le rayon en (1-t)^gamma remplace le MLP, pour le prox l1 UNIQUEMENT.
        # DoubleConvTime n'est pas concerne : son MLP n'implemente pas un rayon de
        # seuillage et la derivation ne s'y transporte pas.
        if use_Unet == "l1":
            for lay in self.layers:
                lay.prox = L1ProxRadiusPow(
                    channels=internal_channel if prox_channels else None)

    def cold_dual(self, x):
        return x.new_zeros(x.shape[0], self.internal_channel, self.img_size, self.img_size)

    def gammas(self):
        """Les exposants appris, un par couche — a journaliser (§8.4)."""
        return [float(l.prox.gamma_value().detach()) for l in self.layers
                if hasattr(l.prox, "gamma_value")]

    @torch.no_grad()
    def check_bound(self, t, img_size=None, n_iter=50, tol=1e-3):
        """§8.2 : le bound de pas tient-il PAR ECHANTILLON ? Recalcule L^2 avec
        beaucoup d'iterations de puissance (pas l'estimation warm-start) et verifie
        tau_k.sigma_k.L^2 <= cp_safety(1+tol) pour tout k et tout element du batch."""
        S = img_size or self.img_size
        tt = t.view(-1, 1, 1, 1)
        rho = tt ** 2
        tau_k = F.softplus(self.log_tau0).view(1, 1, 1, 1).expand_as(rho)
        worst = 0.0
        for layer in self.layers:
            layer._pi_v = None                       # force un recalcul propre
            l2 = layer.loop_gain(S, n_iter=n_iter)
            sigma_k = self.cp_safety / (tau_k * l2)
            worst = max(worst, float((tau_k * sigma_k * l2).max()))
            tau_k = (1.0 + 2.0 * rho * tau_k).pow(-0.5) * tau_k
        assert worst <= self.cp_safety * (1 + tol), (
            f"bound de pas viole : max tau.sigma.L^2 = {worst:.6f} > "
            f"{self.cp_safety}")
        return worst

    def forward(self, xt_t, return_iterates=False, u_init=None, return_u=False,
                return_steps=False):
        B = xt_t.shape[0]
        C, S = self.in_channels, self.img_size
        z = xt_t[:, :self.dim].view(B, C, S, S)
        t = xt_t[:, self.dim:]
        tt = t.view(-1, 1, 1, 1)
        rho = tt ** 2                                  # §3 : borne dans [0,1]

        x = z.clone() if self.x0_mode == "xt" else torch.zeros_like(z)
        u = self.cold_dual(z) if u_init is None else u_init.to(z.device)
        iterates = [x.clone()] if return_iterates else None
        steps = [] if return_steps else None

        tau_k = F.softplus(self.log_tau0).view(1, 1, 1, 1).expand_as(rho)
        for layer in self.layers:
            l2 = layer.loop_gain(S)
            sigma_k = self.cp_safety / (tau_k * l2)
            alpha_k = (1.0 + 2.0 * rho * tau_k).pow(-0.5)
            if return_steps:
                steps.append((float(l2), tau_k.flatten()[0].item(),
                              sigma_k.flatten()[0].item(), alpha_k.flatten()[0].item()))
            if self.use_checkpoint and self.training:
                x, u = checkpoint(layer, x, u, z, t, tau_k, sigma_k, alpha_k,
                                  use_reentrant=False)
            else:
                x, u = layer(x, u, z, t, tau_k, sigma_k, alpha_k)
            tau_k = alpha_k * tau_k
            if return_iterates:
                iterates.append(x.clone())

        if self.training:
            out = x.view(B, -1)
        else:
            out = (x - z).view(B, -1) / fm_velocity_denom(t, self.t_max)
        extras = []
        if return_iterates:
            extras.append(torch.stack(iterates))
        if return_u:
            extras.append(u)
        if return_steps:
            extras.append(steps)
        return (out, *extras) if extras else out



class ConvScCP_UNN_v5(ConvScCP_UNN_v4):
    """v4, mais avec le MLP de temps de L1ProxConv comme rayon du prox.

    C'est le CONTROLE qui isole le §4 du spec. v4 et v5 partagent tout le reste :
    rho = t^2 et le prox (v + tau.t.x_t)/(1 + tau.t^2) du §3, le gain de boucle
    L^2 = ||B o A||, le l2 differentiable. Seul change le rayon du clamp dual :

        v4  r(t) = softplus(a) * (1-t)^gamma      gamma appris, init 2   (§4)
        v5  r(t) = softplus(MLP(t))               MLP libre, comme v1/v2/v3

    L'ecart v5 - v4 mesure donc exactement ce qu'apporte — ou coute — la forme
    prescrite par la theorie, sans rien confondre d'autre. Un MLP libre est
    strictement plus expressif : s'il gagne, la contrainte (1-t)^gamma coute ;
    s'il perd, c'est que la forme theorique regularise utilement.
    """

    def __init__(self, dim=784, K=10, internal_channel=64,
                 use_Unet=False, version="LFO", use_checkpoint=False, init="kaiming",
                 w_bias=True, in_channels=1, img_size=28, kernel_size=9, prox_w=32,
                 prox_channels=False, x0_mode="xt", cp_safety=0.99):
        super().__init__(dim=dim, K=K, internal_channel=internal_channel,
                         use_Unet=use_Unet, version=version,
                         use_checkpoint=use_checkpoint, init=init, w_bias=w_bias,
                         in_channels=in_channels, img_size=img_size,
                         kernel_size=kernel_size, prox_w=prox_w,
                         prox_channels=prox_channels, x0_mode=x0_mode,
                         cp_safety=cp_safety)
        # v4 a installe L1ProxRadiusPow ; on rebascule sur le MLP libre.
        if use_Unet == "l1":
            for lay in self.layers:
                lay.prox = L1ProxConv(
                    w=prox_w, channels=internal_channel if prox_channels else None)


class MinimalResNetFM(nn.Module):
    """Kamb & Ganguli `MinimalResNet` (8 conv 3x3, résiduel, SANS norm spatiale,
    RF ~17x17) porté en modèle Flow-Matching sur MNIST. Même paramétrisation
    x1-pred que ConvScCP_UNN (training -> x1_pred ; eval -> vitesse (x1-z)/(1-t)),
    donc contrôle « toutes choses égales sauf l'architecture » : même cadre FM,
    architecture CONNUE pour être une machine ELS (score) chez Kamb.

    La seule norm (GroupNorm sur le MLP du plongement temps) agit sur les canaux
    du vecteur-temps par échantillon, PAS sur les cartes spatiales -> localité
    préservée. Padding zéro, kernel 3, ReLU.
    """
    def __init__(self, dim=784, in_channels=1, img_size=28, emb_dim=256,
                 num_layers=6, kernel_size=3):
        super().__init__()
        self.dim = dim; self.in_channels = in_channels; self.img_size = img_size
        self.emb_dim = emb_dim; self.num_layers = num_layers
        self.predicts_x1 = True
        self.t_max = None        # cf. fm_velocity_denom
        p = kernel_size // 2
        self.up = nn.Conv2d(in_channels, emb_dim, kernel_size, padding=p)      # zero-pad
        self.embs = nn.ModuleList([
            nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.GroupNorm(8, emb_dim), nn.ReLU())
            for _ in range(num_layers + 1)])
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(emb_dim, emb_dim, kernel_size, padding=p), nn.ReLU())
            for _ in range(num_layers)])
        self.down = nn.Conv2d(emb_dim, in_channels, kernel_size, padding=p)

    def _temb(self, t):                                # t:(B,1) -> (B,emb_dim) sinusoïdal
        d = self.emb_dim // 2
        freqs = 10000.0 ** (torch.arange(d, device=t.device) / (d - 1))
        targ = t / freqs[None, :]
        return torch.cat((torch.sin(targ), torch.cos(targ)), dim=1)

    def forward(self, xt_t):
        B = xt_t.shape[0]; C, S = self.in_channels, self.img_size
        z = xt_t[:, :self.dim].view(B, C, S, S)
        t = xt_t[:, self.dim:]                          # (B,1)
        emb = self._temb(t)
        state = self.up(z)
        for i in range(self.num_layers):
            state = state + self.convs[i](state + self.embs[i](emb)[:, :, None, None])
        state = state + self.embs[-1](emb)[:, :, None, None]
        out = self.down(state)                          # (B,C,S,S) = x1_pred
        if self.training:
            return out.view(B, -1)
        return (out - z).view(B, -1) / fm_velocity_denom(t, self.t_max)


class _UBlockFM(nn.Module):
    """UBlock de Kamb (MinimalUNet) : `depth` convs 'same' à padding circulaire + ReLU,
    avec injection additive du plongement temps (projeté sur infeatures)."""
    def __init__(self, infeatures, outfeatures, emb_dim, depth=2, kernel_size=3, mode="circular"):
        super().__init__()
        self.emb = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, infeatures))
        mods = []
        for i in range(depth):
            cin = infeatures if i == 0 else outfeatures
            mods += [nn.Conv2d(cin, outfeatures, kernel_size, padding="same", padding_mode=mode), nn.ReLU()]
        self.model = nn.Sequential(*mods)

    def forward(self, x, emb):
        return self.model(x + self.emb(emb)[:, :, None, None])


class MinimalUNetFM(nn.Module):
    """UNet multi-échelle de Kamb & Ganguli (`MinimalUNet`, fsizes [32,64,128,256] =
    3 poolings, padding CIRCULAIRE, plongement temps sinusoïdal emb_dim=256, skips)
    porté en Flow-Matching sur MNIST. MÊME convention que MinimalResNetFM :
    predicts_x1=True (training -> x1_pred ; eval -> vitesse (x1_pred - z)/(1-t)).
    28x28 est zero-paddé en 32x32 (les 3 poolings divisent proprement 32) puis recroppé
    — port fidèle de l'archi de Kamb (qui opère en puissances de 2)."""
    def __init__(self, dim=784, in_channels=1, img_size=28, emb_dim=256,
                 fsizes=(32, 64, 128, 256), kernel_size=3, mode="circular"):
        super().__init__()
        self.dim = dim; self.in_channels = in_channels; self.img_size = img_size
        self.emb_dim = emb_dim; self.predicts_x1 = True
        self.t_max = None        # cf. fm_velocity_denom
        fs = list(fsizes)
        cin = in_channels
        self.feature_blocks = nn.ModuleList()
        for f in fs[:-1]:
            self.feature_blocks.append(_UBlockFM(cin, f, emb_dim, kernel_size=kernel_size, mode=mode))
            cin = f
        self.bottleneck = _UBlockFM(fs[-2], fs[-1], emb_dim, kernel_size=kernel_size, mode=mode)
        self.upsamples = nn.ModuleList()
        self.output_blocks = nn.ModuleList()
        for i in range(len(fs) - 1, 0, -1):
            self.upsamples.append(nn.ConvTranspose2d(fs[i], fs[i - 1], kernel_size=2, stride=2))
            self.output_blocks.append(_UBlockFM(2 * fs[i - 1], fs[i - 1], emb_dim, kernel_size=kernel_size, mode=mode))
        self.last_emb = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, fs[0]))
        self.output_conv = nn.Conv2d(fs[0], in_channels, kernel_size=1, padding="same", padding_mode=mode)
        self.pool = nn.MaxPool2d(2, 2)

    def _temb(self, t):                                # t:(B,1) -> (B,emb_dim), = EmbeddingModule de Kamb
        d = self.emb_dim // 2
        freqs = 10000.0 ** (torch.arange(d, device=t.device) / (d - 1))
        targ = t / freqs[None, :]
        return torch.cat((torch.sin(targ), torch.cos(targ)), dim=1)

    def forward(self, xt_t):
        B = xt_t.shape[0]; C, S = self.in_channels, self.img_size
        z = xt_t[:, :self.dim].view(B, C, S, S)
        t = xt_t[:, self.dim:]                          # (B,1)
        emb = self._temb(t)
        pad = (32 - S) // 2                             # 28 -> 32 (poolings divisent proprement)
        x = F.pad(z, (pad, pad, pad, pad))
        skips = []
        for down in self.feature_blocks:
            x = down(x, emb); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x, emb)
        skips = skips[::-1]
        for i in range(len(self.upsamples)):
            up = self.upsamples[i](x)
            x = torch.cat((skips[i], up), dim=1)
            x = self.output_blocks[i](x, emb)
        x = self.output_conv(x + self.last_emb(emb)[:, :, None, None])
        out = x[..., pad:pad + S, pad:pad + S].contiguous()     # crop 32 -> 28 = x1_pred
        if self.training:
            return out.view(B, -1)
        return (out - z).view(B, -1) / fm_velocity_denom(t, self.t_max)
