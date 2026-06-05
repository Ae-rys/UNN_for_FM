# -*- coding: utf-8 -*-
"""
architectures.py
All neural network building blocks and UNN model definitions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def soft_threshold(z, T):
    return torch.sign(z) * torch.maximum(abs(z) - T[:, None], torch.zeros_like(z))


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


# Alias used by DFB_UNN / DiFB_UNN / CP_UNN
MLP = small_MLP


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
    def __init__(self, in_channels=1, out_channels=1, base_ch=32, alpha=1.0):
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
        self.alpha = alpha

    def forward(self, xt_t):
        x = xt_t[..., :-1]
        t = xt_t[..., -1:]
        batch_size = x.shape[0]
        x_img = x.view(batch_size, self.in_channels, 28, 28)
        if t.dim() > 2:
            t = t.squeeze(-1)
        alpha = F.softplus(self.alpha) if torch.is_tensor(self.alpha) else self.alpha
        t_emb = alpha * self.time_scaling(t).view(batch_size, -1, 1, 1)
        x1    = self.inc(x_img) + t_emb
        x2    = self.down1(x1)
        x_bot = self.bot(x2)
        x_up  = self.up1(x_bot)
        x_dec = self.dec1(torch.cat([x_up, x1], dim=1))
        out   = self.outc(x_dec)
        return out.view(batch_size, -1)


class DoubleConvTime(nn.Module):
    """Two convolutions with time injection."""
    def __init__(self, in_ch, out_ch, embed_dim, alpha):
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
        self.alpha = alpha

    def forward(self, x, t):
        alpha = F.softplus(self.alpha) if torch.is_tensor(self.alpha) else self.alpha
        t_emb = alpha * self.time_scaling(t)
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
    def __init__(self, dim, prox, alpha=None, W=None, V=None, learned_prox=True, version="LFO"):
        super().__init__()
        self.version = version
        self.W_weight = nn.Parameter(torch.eye(dim) + torch.randn(dim, dim) * 0.01) if W is None else W
        if version == "LFO":
            self.V_weight = nn.Parameter(torch.eye(dim) + torch.randn(dim, dim) * 0.01) if V is None else V
            self.tau = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('_sigma_u', F.normalize(torch.randn(dim), dim=0))
        if not learned_prox:
            self.time_scaling = nn.Sequential(
                nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 1),
            )
            self.alpha = alpha
        self.prox = prox
        self.learned_prox = learned_prox

    def forward(self, u, z, t):
        V      = self.V_weight if self.version == "LFO" else self.W_weight.T
        x_next = z - F.linear(u, V, None)
        if self.version == "LFO":
            tau = F.softplus(self.tau)
        else:
            _s, self._sigma_u = sigma_max_power_iter(self.W_weight, self._sigma_u)
            tau = 1.99 / _s ** 2
        step   = tau * F.linear(x_next, self.W_weight, None)
        if self.learned_prox:
            u_next = self.prox(torch.cat((u + step, t), dim=-1))
        else:
            radius = F.softplus(self.alpha) * self.time_scaling(t)
            u_next = self.prox(u + step, radius)
        return x_next, u_next


class DFB_UNN(nn.Module):
    def __init__(self, dim, K=10, alpha_init=1.0, learned_prox=False, w=32, version="LFO"):
        super().__init__()
        self.dim   = dim
        self.K     = K
        self.learned_prox = learned_prox
        if learned_prox:
            self.proxs  = nn.ModuleList([MLP(dim=dim, time_varying=True, w=w) for _ in range(K)])
            alpha_param = None
        else:
            self.proxs  = [proj_l_inf for _ in range(K)]
            self.alpha  = nn.Parameter(torch.tensor(alpha_init))
            alpha_param = self.alpha
        self.layers = nn.ModuleList([
            DFB_Iteration(dim, self.proxs[i], alpha=alpha_param, version=version, learned_prox=learned_prox) for i in range(K)
        ])

    def forward(self, xt_t):
        xt = xt_t[:, :self.dim]
        t  = xt_t[:, self.dim:]
        z  = xt
        x  = z.clone()
        u  = torch.zeros_like(xt)
        for layer in self.layers:
            x, u = layer(u, z, t)
        return x - xt

# ---------------------------------------------------------------------------
# DiFB-UNN (DFB with inertia/momentum)
# ---------------------------------------------------------------------------

class DiFB_UNN(nn.Module):
    def __init__(self, dim, K=10, w=32, version="LFO"):
        super().__init__()
        self.dim   = dim
        self.K     = K
        self.proxs = nn.ModuleList([MLP(dim=dim, time_varying=True, w=w) for _ in range(K)])
        self.layers = nn.ModuleList([
            DFB_Iteration(dim, self.proxs[i], version=version) for i in range(K)
        ])

    def forward(self, xt_t):
        xt     = xt_t[:, :self.dim]
        t      = xt_t[:, self.dim:]
        z      = xt
        x      = z.clone()
        u      = torch.zeros_like(xt)
        u_prev = torch.zeros_like(xt)
        for layer in self.layers:
            v      = u + 0.5 * (u - u_prev)
            u_prev = u.clone()
            x, u   = layer(v, z, t)
        return x - xt


# ---------------------------------------------------------------------------
# ConvDFB-UNN (each layer has its own W)
# ---------------------------------------------------------------------------

class ConvDFB_Iteration(nn.Module):
    def __init__(self, internal_channel, alpha, version="LFO"):
        super().__init__()
        self.version  = version
        self.W_weight = nn.Parameter(torch.randn(internal_channel, 1, 3, 3) * 0.05)
        if version == "LFO":
            self.V_weight = nn.Parameter(torch.randn(internal_channel, 1, 3, 3) * 0.05)
            self.tau = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('_sigma_u', F.normalize(torch.randn(internal_channel), dim=0))
        self.prox = DoubleConvTime(
            in_ch=internal_channel, out_ch=internal_channel,
            embed_dim=internal_channel // 2, alpha=alpha,
        )

    def forward(self, u, z, t):
        V      = self.V_weight if self.version == "LFO" else self.W_weight
        x_next = z - F.conv_transpose2d(u, self.W_weight, padding=1)
        if self.version == "LFO":
            tau = F.softplus(self.tau)
        else:
            _s, self._sigma_u = sigma_max_power_iter(self.W_weight, self._sigma_u)
            tau = 1.99 / _s ** 2
        step   = tau * F.conv2d(x_next, V, padding=1)
        u_next = self.prox(u + step, t)
        return x_next, u_next


class ConvDFB_UNN(nn.Module):
    def __init__(self, dim, K=10, alpha_init=1.0, internal_channel=64, version="LFO"):
        super().__init__()
        self.dim = dim
        self.K   = K
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.layers = nn.ModuleList([
            ConvDFB_Iteration(internal_channel=internal_channel, alpha=self.alpha, version=version)
            for _ in range(K)
        ])
        self.internal_channel = internal_channel

    def forward(self, xt_t):
        batch_size = xt_t.shape[0]
        z = xt_t[:, :self.dim].view(batch_size, 1, 28, 28)
        t = xt_t[:, self.dim:]
        u = torch.zeros(batch_size, self.internal_channel, 28, 28, device=z.device)
        x = z
        for layer in self.layers:
            x, u = layer(u, z, t)
        return (x - z).view(batch_size, -1)


# ---------------------------------------------------------------------------
# SharedConvDFB-UNN (shared weights across iterations)
# ---------------------------------------------------------------------------

class SharedConvDFB_UNN(nn.Module):
    def __init__(self, dim=784, K=10, alpha_init=1.0, internal_channel=64,
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
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        if use_Unet:
            self.prox = SmallUNet(
                in_channels=internal_channel, out_channels=internal_channel,
                base_ch=internal_channel // 2, alpha=self.alpha,
            )
        else:
            self.prox = DoubleConvTime(
                in_ch=internal_channel, out_ch=internal_channel,
                embed_dim=internal_channel // 2, alpha=self.alpha,
            )
        self.use_Unet = use_Unet

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
            x_next = z - F.conv_transpose2d(u, V, padding=1)
            step   = tau * F.conv2d(x_next, self.shared_W, padding=1)
            if self.use_Unet:
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
# CP-UNN (Chambolle-Pock)
# ---------------------------------------------------------------------------

class CP_Iteration(nn.Module):
    def __init__(self, dim, prox_dual, version="LFO"):
        super().__init__()
        self.version  = version
        self.W_weight = nn.Parameter(torch.eye(dim) + torch.randn(dim, dim) * 0.01)
        self.sigma    = nn.Parameter(torch.tensor(0.5))
        if version == "LFO":
            self.V_weight = nn.Parameter(torch.eye(dim) + torch.randn(dim, dim) * 0.01)
            self.tau      = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('_sigma_u', F.normalize(torch.randn(dim), dim=0))
        self.prox_dual = prox_dual

    def forward(self, x, x_bar, u, z, t):
        sigma = F.softplus(self.sigma)
        if self.version == "LFO":
            tau = F.softplus(self.tau)
        else:
            _s, self._sigma_u = sigma_max_power_iter(self.W_weight, self._sigma_u)
            tau = 0.99 / (sigma * _s ** 2)
        V = self.V_weight if self.version == "LFO" else self.W_weight.T
        primal_input = x - tau * F.linear(u, V, None)
        x_next  = (primal_input + tau * z) / (1 + tau)
        dual_step = sigma * F.linear(x_bar, self.W_weight, None)
        u_next  = self.prox_dual(torch.cat((u + dual_step, t), dim=-1))
        x_bar_next = 2 * x_next - x
        return x_next, x_bar_next, u_next


class CP_UNN(nn.Module):
    def __init__(self, dim, K=10, w=32, version="LFO"):
        super().__init__()
        self.dim   = dim
        self.K     = K
        self.prox_list = nn.ModuleList([small_MLP(dim=dim, w=w, time_varying=True) for _ in range(K)])
        self.layers    = nn.ModuleList([
            CP_Iteration(dim, self.prox_list[i], version=version) for i in range(K)
        ])

    def forward(self, xt_t):
        xt    = xt_t[:, :self.dim]
        t     = xt_t[:, self.dim:]
        z     = xt
        x     = xt.clone()
        x_bar = xt.clone()
        u     = torch.zeros_like(xt)
        for layer in self.layers:
            x, x_bar, u = layer(x, x_bar, u, z, t)
        return x - xt


# ---------------------------------------------------------------------------
# ConvCP-UNN (each layer has its own W)
# ---------------------------------------------------------------------------

class ConvCP_Iteration(nn.Module):
    def __init__(self, internal_channels=64, alpha=1.0, version="LFO"):
        super().__init__()
        self.version  = version
        self.W_weight = nn.Parameter(torch.randn(internal_channels, 1, 3, 3) * 0.05)
        self.sigma    = nn.Parameter(torch.tensor(0.5))
        if version == "LFO":
            self.V_weight = nn.Parameter(torch.randn(internal_channels, 1, 3, 3) * 0.05)
            self.tau      = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('_sigma_u', F.normalize(torch.randn(internal_channels), dim=0))
        self.prox_dual = DoubleConvTime(
            in_ch=internal_channels, out_ch=internal_channels,
            embed_dim=internal_channels // 2, alpha=alpha,
        )

    def forward(self, x, x_bar, u, z, t_prox):
        sigma = F.softplus(self.sigma)
        if self.version == "LFO":
            tau = F.softplus(self.tau)
        else:
            _s, self._sigma_u = sigma_max_power_iter(self.W_weight, self._sigma_u)
            tau = 0.99 / (sigma * _s ** 2)
        V       = self.V_weight if self.version == "LFO" else self.W_weight
        grad_u  = F.conv_transpose2d(u, V, padding=1)
        primal_input = x - tau * grad_u
        x_next  = (primal_input + tau * z) / (1 + tau)
        dual_step = F.conv2d(x_bar, self.W_weight, padding=1)
        u_next  = self.prox_dual(u + sigma * dual_step, t_prox)
        x_bar_next = 2 * x_next - x
        return x_next, x_bar_next, u_next


class ConvCP_UNN(nn.Module):
    def __init__(self, dim, K=10, alpha_init=1.0, internal_channels=64, img_size=28, version="LFO"):
        super().__init__()
        self.dim = dim
        self.K   = K
        self.internal_channels = internal_channels
        self.img_size = img_size
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.layers = nn.ModuleList([
            ConvCP_Iteration(internal_channels=internal_channels, alpha=self.alpha, version=version)
            for _ in range(K)
        ])

    def forward(self, xt_t):
        batch_size = xt_t.shape[0]
        z = xt_t[:, :self.dim].view(batch_size, 1, self.img_size, self.img_size)
        t = xt_t[:, self.dim:]
        x     = z.clone()
        x_bar = z.clone()
        u     = torch.zeros(batch_size, self.internal_channels, self.img_size, self.img_size, device=z.device)
        for layer in self.layers:
            x, x_bar, u = layer(x, x_bar, u, z, t)
        return (x - z).view(batch_size, -1)


# ---------------------------------------------------------------------------
# SharedConvCP-UNN (shared weights across iterations)
# ---------------------------------------------------------------------------

class SharedConvCP_UNN(nn.Module):
    def __init__(self, dim=784, K=10, alpha_init=1.0, internal_channel=64,
                 img_size=28, use_Unet=False, version="LFO"):
        super().__init__()
        self.dim = dim
        self.K   = K
        self.internal_channel = internal_channel
        self.img_size = img_size
        self.version  = version
        self.shared_W = nn.Parameter(torch.randn(internal_channel, 1, 3, 3) * 0.05)
        self.sigma    = nn.Parameter(torch.tensor(0.5))
        if version == "LFO":
            self.shared_V = nn.Parameter(torch.randn(internal_channel, 1, 3, 3) * 0.05)
            self.tau = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer('_sigma_u', F.normalize(torch.randn(internal_channel), dim=0))
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        if use_Unet:
            self.prox = SmallUNet(
                in_channels=internal_channel, out_channels=internal_channel,
                base_ch=internal_channel // 2, alpha=self.alpha,
            )
        else:
            self.prox = DoubleConvTime(
                in_ch=internal_channel, out_ch=internal_channel,
                embed_dim=internal_channel // 2, alpha=self.alpha,
            )
        self.use_Unet = use_Unet

    def forward(self, xt_t, n_iter=None):
        iters      = n_iter if n_iter is not None else self.K
        batch_size = xt_t.shape[0]
        z = xt_t[:, :self.dim].view(batch_size, 1, self.img_size, self.img_size)
        t = xt_t[:, self.dim:]
        x     = z.clone()
        x_bar = z.clone()
        u     = torch.zeros(batch_size, self.internal_channel, self.img_size, self.img_size, device=z.device)
        sigma = F.softplus(self.sigma)
        if self.version == "LFO":
            tau = F.softplus(self.tau)
        else:
            _s, self._sigma_u = sigma_max_power_iter(self.shared_W, self._sigma_u)
            tau = 0.99 / (sigma * _s ** 2)
        V = self.shared_V if self.version == "LFO" else self.shared_W
        for _ in range(iters):
            grad_u       = F.conv_transpose2d(u, V, padding=1)
            primal_input = x - tau * grad_u
            x_next       = (primal_input + tau * z) / (1 + tau)
            dual_step    = F.conv2d(x_bar, self.shared_W, padding=1)
            if self.use_Unet:
                u_flat    = (u + sigma * dual_step).view(batch_size, -1, 784)
                ut_t_flat = torch.cat(
                    [u_flat, t[:, 0].view(batch_size, 1, 1).expand(batch_size, self.internal_channel, 1)],
                    dim=-1,
                )
                u = self.prox(ut_t_flat).view(batch_size, self.internal_channel, self.img_size, self.img_size)
            else:
                u = self.prox(u + sigma * dual_step, t)
            x_bar = 2 * x_next - x
            x     = x_next
        return (x - z).view(batch_size, -1)
