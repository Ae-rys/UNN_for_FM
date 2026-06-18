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
    """
    def __init__(self, w=32):
        super().__init__()
        self.time_scaling = nn.Sequential(
            nn.Linear(1, w),
            nn.SiLU(),
            nn.Linear(w, 1),
        )

    def forward(self, u, t):
        radius = F.softplus(self.time_scaling(t))                          # (B, 1)
        r_bc   = radius.view(radius.shape[0], *([1] * (u.dim() - 1)))     # (B, 1, ..., 1)
        return torch.clamp(u, -r_bc, r_bc)


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
    def __init__(self, dim, prox, dual_dim=None, version="LFO"):
        super().__init__()
        self.version  = version
        dual_dim      = dual_dim or dim
        W_init = torch.randn(dual_dim, dim) * 1
        if dual_dim == dim:
            W_init = W_init + torch.eye(dim)
        self.W_weight = nn.Parameter(W_init)
        if version == "LFO":
            # V: (dim, dual_dim) so that F.linear(u, V) with u:(B,dual_dim) → (B,dim)
            V_init = torch.randn(dim, dual_dim) * 1
            if dual_dim == dim:
                V_init = V_init + torch.eye(dim)
            self.V_weight = nn.Parameter(V_init)
            self.tau = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('_sigma_u', F.normalize(torch.randn(dual_dim), dim=0))
        self.prox = prox

        self.time_scaling = nn.Sequential(
            nn.Linear(1, 8),
            nn.SiLU(),
            nn.Linear(8, 1),
            nn.Softplus(),
        )

    def forward(self, u, z, t):
        V      = self.V_weight if self.version == "LFO" else self.W_weight.T
        x_next = z - F.linear(u, V, None)
        if self.version == "LFO":
            tau = F.softplus(self.tau)
        else:
            _s, self._sigma_u = sigma_max_power_iter(self.W_weight, self._sigma_u)
            tau = 1.99 / _s ** 2
        step   = tau * F.linear(x_next, self.W_weight, None)
        u_next = self.prox(torch.cat((u + step, self.time_scaling(t)), dim=-1))
        return x_next, u_next


class DFB_UNN(nn.Module):
    def __init__(self, dim, K=10, learned_prox=False, w=32, dual_dim=None, version="LFO"):
        super().__init__()
        self.dim      = dim
        self.K        = K
        self.dual_dim = dual_dim or dim
        if learned_prox:
            self.proxs = nn.ModuleList([MLP(dim=self.dual_dim, time_varying=True, w=w) for _ in range(K)])
        else:
            self.proxs = nn.ModuleList([L1ProxFlat(dim=self.dual_dim) for _ in range(K)])
        self.layers = nn.ModuleList([
            DFB_Iteration(dim, self.proxs[i], dual_dim=self.dual_dim, version=version) for i in range(K)
        ])

    def forward(self, xt_t, return_u=False):
        xt = xt_t[:, :self.dim]
        t  = xt_t[:, self.dim:]
        z  = xt
        x  = z
        u  = torch.zeros(xt.shape[0], self.dual_dim, device=xt.device)
        for layer in self.layers:
            x, u = layer(u, z, t)
        vt = x - xt
        vt = vt
        if return_u:
            return vt, u
        return vt

    
# ---------------------------------------------------------------------------
# FLAT-DFB-UNN
# ---------------------------------------------------------------------------
    
# class FLAT_DFB_UNN(nn.Module):
#     def __init__(self, dim, N=10, w=64, dual_dim=None, learned_prox=False, version="LFO"):
#         super().__init__()
#         self.dim      = dim
#         self.N        = N
#         self.dual_dim = dual_dim or dim

#         # learned_prox=True est NON NÉGOCIABLE pour la génération.
#         # L1ProxFlat contracte structurellement tout vers 0 (voir analyse ci-dessus).
#         if learned_prox:
#             self.proxs = nn.ModuleList([
#                     small_MLP(dim=self.dual_dim, time_varying=True, w=w) for _ in range(N)
#                 ])
#         else:
#             self.proxs = nn.ModuleList([
#                 L1ProxFlat(dim=self.dual_dim) for _ in range(N)
#             ])
#         self.layers = nn.ModuleList([
#             DFB_Iteration(dim, self.proxs[i], dual_dim=self.dual_dim, version=version)
#             for i in range(N)
#         ])

#     def forward(self, x0, return_traj=False, return_x1_hats=False):
#         B = x0.shape[0]
#         x = x0

#         # u PERSISTANT sur toute la trajectoire.
#         # Si on reset u à chaque pas :
#         #   x_next = z - V @ 0 = z  →  le prox ne sert à rien,
#         #   son output u_next est immédiatement jeté.
#         u = torch.zeros(B, self.dual_dim, device=x.device)

#         x_traj   = [x]
#         x_states = []

#         for n, layer in enumerate(self.layers):
#             t_n      = (n + 1) / (self.N + 1)   # évite 0 et 1
#             t_tensor = torch.full((B, 1), t_n, device=x.device)

#             # z = x  (pas z = x/t)
#             # ──────────────────────────────────────────────────────────────
#             # La couche DFB calcule  x_next = x - V @ u  (pas de MAP estimate).
#             # C'est directement l'étape d'Euler de l'ODE FM, pas une prox approchée.
#             # Passer z = x/t avec u reset donne x_next = x/t (le prox est ignoré),
#             # puis la combinaison convexe fait DIVERGER x (facteur (1 + λ*(1/t-1)) > 1).
#             # ──────────────────────────────────────────────────────────────
#             x_next, u = layer(u, x, t_tensor)
#             # x_next = x + x_next
#             x_states.append(x_next)

#             # PAS de combinaison convexe supplémentaire.
#             # Le DFB encode déjà le step via (x_next = x - V@u).
#             # Rajouter (1-λ)*x + λ*x_next revient à multiplier le step par λ,
#             # ce qui fait (1-λ*tau)*x → contraction vers 0.
#             x = x_next
#             x_traj.append(x)

#         if return_traj and return_x1_hats:
#             return x, x_traj, x_states
#         if return_traj:
#             return x, x_traj
#         if return_x1_hats:
#             return x, x_states
#         return x

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
            denom = (1.0 - t_vec).clamp(min=1e-6).unsqueeze(1)
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

                denom    = max(1.0 - t_n, 1e-6)
                lambda_n = delta_n / denom
                x_new    = (1.0 - lambda_n) * x + lambda_n * x1_hat

                _, u = self.layers[n](u, x, _t_tensor(n))
                x    = x_new
                x_traj.append(x)

            if return_traj: return x, x_traj
            return x


class SharedFLAT_DFB_UNN(nn.Module):
    """
    Version à poids partagés de FLAT DFB (limite Deep Equilibrium).
    Idéal pour varier le nombre d'itérations N dynamiquement.
    """
    def __init__(self, dim, N=10, w=32, dual_dim=None, version="LFO", prox_type="mlp"):
        super().__init__()
        self.dim = dim
        self.N = N
        self.version = version
        self.dual_dim = dual_dim or dim
        
        W_init = torch.randn(self.dual_dim, dim) * 0.01
        if self.dual_dim == dim:
            W_init = W_init + torch.eye(dim)
        self.shared_W = nn.Parameter(W_init)
        
        if version == "LFO":
            V_init = torch.randn(dim, self.dual_dim) * 0.01
            if self.dual_dim == dim:
                V_init = V_init + torch.eye(dim)
            self.shared_V = nn.Parameter(V_init)
            self.tau = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('_sigma_u', F.normalize(torch.randn(self.dual_dim), dim=0))
            
        if prox_type == "l1":
            self.prox = L1ProxFlat(dim=self.dual_dim)
        else:
            self.prox = small_MLP(dim=self.dual_dim, w=w, time_varying=True)

    def forward(self, x0, return_traj=False, return_x1_hats=False):
        B = x0.shape[0]
        x = x0
        u = torch.zeros(B, self.dual_dim, device=x.device)
        
        x_traj = [x]
        x1_hats = []
        
        V = self.shared_V if self.version == "LFO" else self.shared_W.T
        if self.version == "LFO":
            tau = F.softplus(self.tau)
        else:
            _s, self._sigma_u = sigma_max_power_iter(self.shared_W, self._sigma_u)
            tau = 1.99 / _s ** 2
            
        for n in range(self.N):
            t_n = n / self.N
            lam_n = 1.0 / (self.N - n)
            t_tensor = torch.full((B, 1), t_n, device=x.device)
            
            # DFB Iteration Manuelle
            x1_hat = x - F.linear(u, V, None)
            step = tau * F.linear(x1_hat, self.shared_W, None)
            u = self.prox(torch.cat((u + step, t_tensor), dim=-1))
            
            x1_hats.append(x1_hat)
            
            # Euler step
            x = (1 - lam_n) * x + lam_n * x1_hat
            x_traj.append(x)
            
        if return_traj and return_x1_hats:
            return x, x_traj, x1_hats
        if return_traj:
            return x, x_traj
        if return_x1_hats:
            return x, x1_hats
        return x

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
            
        return x - xt

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
        W_init = torch.randn(self.dual_dim, dim) * 0.01
        if self.dual_dim == dim:
            W_init = W_init + torch.eye(dim)
        self.shared_W = nn.Parameter(W_init)
        if version == "LFO":
            # shared_V: (dim, dual_dim) so F.linear(u, V) with u:(B,dual_dim) → (B,dim)
            V_init = torch.randn(dim, self.dual_dim) * 0.01
            if self.dual_dim == dim:
                V_init = V_init + torch.eye(dim)
            self.shared_V = nn.Parameter(V_init)
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
        vt = x_next - xt
        if return_traj:
            return vt, x_traj
        return vt


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
        W_init = torch.randn(self.dual_dim, dim) * 0.01
        self.rho = nn.Parameter(torch.full((K,), 0.5))
        if self.dual_dim == dim:
            W_init = W_init + torch.eye(dim)
        self.shared_W = nn.Parameter(W_init)
        if version == "LFO":
            # shared_V: (dim, dual_dim) so F.linear(u, V) with u:(B,dual_dim) → (B,dim)
            V_init = torch.randn(dim, self.dual_dim) * 0.01
            if self.dual_dim == dim:
                V_init = V_init + torch.eye(dim)
            self.shared_V = nn.Parameter(V_init)
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
        return x_next - xt


# ---------------------------------------------------------------------------
# ConvDFB-UNN (each layer has its own W)
# ---------------------------------------------------------------------------

class ConvDFB_Iteration(nn.Module):
    def __init__(self, internal_channel, use_Unet=False, version="LFO"):
        super().__init__()
        self.version  = version
        self.W_weight = nn.Parameter(torch.randn(internal_channel, 1, 3, 3) * 0.05)
        if version == "LFO":
            self.V_weight = nn.Parameter(torch.randn(internal_channel, 1, 3, 3) * 0.05)
            self.tau = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('_sigma_u', F.normalize(torch.randn(internal_channel), dim=0))
        if use_Unet == "l1":
            self.prox = L1ProxConv()
        else:
            self.prox = DoubleConvTime(
                in_ch=internal_channel, out_ch=internal_channel,
                embed_dim=internal_channel // 2,
            )

    def forward(self, u, z, t):
        V      = self.V_weight if self.version == "LFO" else self.W_weight
        x_next = z - F.conv_transpose2d(u, V, padding=1)
        if self.version == "LFO":
            tau = F.softplus(self.tau)
        else:
            _s, self._sigma_u = sigma_max_power_iter(self.W_weight, self._sigma_u)
            tau = 1.99 / _s ** 2
        step   = tau * F.conv2d(x_next, self.W_weight, padding=1)
        u_next = self.prox(u + step, t)
        return x_next, u_next


class ConvDFB_UNN(nn.Module):
    def __init__(self, dim, K=10, internal_channel=64, use_Unet=False, version="LFO",
                 use_checkpoint=False):
        super().__init__()
        self.dim              = dim
        self.K                = K
        self.internal_channel = internal_channel
        self.use_checkpoint   = use_checkpoint
        self.layers = nn.ModuleList([
            ConvDFB_Iteration(internal_channel=internal_channel,
                              use_Unet=use_Unet, version=version)
            for _ in range(K)
        ])

    def forward(self, xt_t):
        batch_size = xt_t.shape[0]
        z = xt_t[:, :self.dim].view(batch_size, 1, 28, 28)
        t = xt_t[:, self.dim:]
        u = torch.zeros(batch_size, self.internal_channel, 28, 28, device=z.device)
        x = z
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x, u = checkpoint(layer, u, z, t, use_reentrant=False)
            else:
                x, u = layer(u, z, t)
        return (x - z).view(batch_size, -1)


# ---------------------------------------------------------------------------
# ConvDiFB-UNN (ConvDFB with inertia/momentum on the dual variable)
# ---------------------------------------------------------------------------

class ConvDiFB_UNN(nn.Module):
    def __init__(self, dim, K=10, internal_channel=64, use_Unet=False, version="LFO", a=3.0,
                 use_checkpoint=False):
        super().__init__()
        self.dim              = dim
        self.K                = K
        self.internal_channel = internal_channel
        self.version          = version
        self.a                = a  # Paramètre a > 2 pour la suite LNO (Corollaire 1)
        self.use_checkpoint   = use_checkpoint

        if version == "LFO":
            # DDiFB-LFO : le paramètre d'inertie rho_k est appris pour chaque couche k.
            self.rho = nn.Parameter(torch.full((K,), 0.5))

        self.layers = nn.ModuleList([
            ConvDFB_Iteration(internal_channel=internal_channel,
                              use_Unet=use_Unet, version=version)
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

        return (x - z).view(batch_size, -1)


# ---------------------------------------------------------------------------
# ConvScCP-UNN (per-layer W, accelerated Chambolle-Pock, convolutional)
# ---------------------------------------------------------------------------

class ConvScCP_Iteration(nn.Module):
    """Single accelerated CP step with convolutional W.

    tau, sigma, alpha_mom are passed in by ConvScCP_UNN (adaptive schedule).
    """
    def __init__(self, internal_channel, use_Unet=False, version="LFO"):
        super().__init__()
        self.version  = version
        self.W_weight = nn.Parameter(torch.randn(internal_channel, 1, 3, 3) * 0.05)
        if version == "LFO":
            self.V_weight = nn.Parameter(torch.randn(internal_channel, 1, 3, 3) * 0.05)
        else:
            self.register_buffer('_sigma_u', F.normalize(torch.randn(internal_channel), dim=0))
        if use_Unet == "l1":
            self.prox = L1ProxConv()
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
        # Primal: prox_{tau * 0.5||·-z||^2}(x - tau * W^T u)
        primal_input = x - tau * F.conv_transpose2d(u, V, padding=1)
        x_next       = (primal_input + tau * z) / (1 + tau)
        # Extrapolation
        y         = x_next + alpha_mom * (x_next - x)
        # Dual
        dual_step = sigma * F.conv2d(y, self.W_weight, padding=1)
        u_next    = self.prox(u + dual_step, t)
        return x_next, u_next


class ConvScCP_UNN(nn.Module):
    def __init__(self, dim=784, K=10, internal_channel=64,
                 use_Unet=False, version="LFO", use_checkpoint=False):
        super().__init__()
        self.dim              = dim
        self.K                = K
        self.internal_channel = internal_channel
        self.version          = version
        self.use_checkpoint   = use_checkpoint

        if version == "LNO":
            # DScCP-LNO : Step size par couche
            self.log_tau = nn.Parameter(torch.full((K,), -0.5))
        else:
            # DScCP-LFO
            self.log_tau0 = nn.Parameter(torch.tensor(-0.5))

        self.layers = nn.ModuleList([
            ConvScCP_Iteration(internal_channel=internal_channel,
                               use_Unet=use_Unet, version=version)
            for _ in range(K)
        ])

    def forward(self, xt_t):
        batch_size = xt_t.shape[0]
        z = xt_t[:, :self.dim].view(batch_size, 1, 28, 28)
        t = xt_t[:, self.dim:]
        x = z.clone()
        u = torch.zeros(batch_size, self.internal_channel, 28, 28, device=z.device)
        
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
        else:
            tau_k = F.softplus(self.log_tau0)
            sigma_k = 1.0 # Absorbé
            for layer in self.layers:
                alpha_k = (1.0 + 2.0 * tau_k).pow(-0.5)
                if self.use_checkpoint and self.training:
                    x, u = checkpoint(layer, x, u, z, t, tau_k, sigma_k, alpha_k, use_reentrant=False)
                else:
                    x, u = layer(x, u, z, t, tau_k, sigma_k, alpha_k)
                tau_k = alpha_k * tau_k
                
        return (x - z).view(batch_size, -1)


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
        self.shared_W = nn.Parameter(torch.randn(internal_channel, 1, 3, 3) * 0.05)
        if version == "LFO":
            self.shared_V = nn.Parameter(torch.randn(internal_channel, 1, 3, 3) * 0.05)
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
        return (x_next - z).view(batch_size, -1)



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

    tau, sigma, alpha are passed in by ScCP_UNN (adaptive schedule),
    so this class does not own them as parameters.
    """
    def __init__(self, dim, prox_dual, dual_dim=None, version="LFO"):
        super().__init__()
        self.version  = version
        dual_dim      = dual_dim or dim
        W_init = torch.randn(dual_dim, dim) * 0.01
        if dual_dim == dim:
            W_init = W_init + torch.eye(dim)
        self.W_weight = nn.Parameter(W_init)
        if version == "LFO":
            # V: (dim, dual_dim) so that F.linear(u, V) with u:(B,dual_dim) → (B,dim)
            V_init = torch.randn(dim, dual_dim) * 0.01
            if dual_dim == dim:
                V_init = V_init + torch.eye(dim)
            self.V_weight = nn.Parameter(V_init)
        else:
            self.register_buffer('_sigma_u', F.normalize(torch.randn(dual_dim), dim=0))
        self.prox_dual = prox_dual

    def spectral_norm(self):
        _s, self._sigma_u = sigma_max_power_iter(self.W_weight, self._sigma_u)
        return _s

    def forward(self, x, u, z, t, tau, sigma, alpha):
        V = self.V_weight if self.version == "LFO" else self.W_weight.T
        primal_input = x - tau * F.linear(u, V, None)
        x_next = (primal_input + tau * z) / (1 + tau)
        y = x_next + alpha * (x_next - x)
        dual_step = sigma * F.linear(y, self.W_weight, None)
        u_next = self.prox_dual(torch.cat((u + dual_step, t), dim=-1))
        return x_next, u_next


class ScCP_UNN(nn.Module):
    def __init__(self, dim, K=10, w=32, dual_dim=None, version="LFO", prox_type="mlp"):
        super().__init__()
        self.dim      = dim
        self.K        = K
        self.version  = version
        self.dual_dim = dual_dim or dim
        
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
            ScCP_Iteration(dim, self.prox_list[i], dual_dim=self.dual_dim, version=version) for i in range(K)
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
                
        return x - xt


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
        W_init = torch.randn(self.dual_dim, dim) * 0.01
        if self.dual_dim == dim:
            W_init = W_init + torch.eye(dim)
        self.shared_W = nn.Parameter(W_init)
        if version == "LFO":
            # shared_V: (dim, dual_dim) so F.linear(u, V) with u:(B,dual_dim) → (B,dim)
            V_init = torch.randn(dim, self.dual_dim) * 0.01
            if self.dual_dim == dim:
                V_init = V_init + torch.eye(dim)
            self.shared_V   = nn.Parameter(V_init)
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
        return x - xt


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
        self.shared_W = nn.Parameter(torch.randn(internal_channel, 1, 3, 3) * 0.05)
        if version == "LFO":
            self.shared_V   = nn.Parameter(torch.randn(internal_channel, 1, 3, 3) * 0.05)
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
        return (x - z).view(batch_size, -1)