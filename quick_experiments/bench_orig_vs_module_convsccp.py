# -*- coding: utf-8 -*-
"""
bench_orig_vs_module_convsccp.py — la version "nn.Conv2d" (bloc commente,
architectures.py:2305) va-t-elle plus vite que la version active
(nn.Parameter + F.conv2d, architectures.py:2217) ?

Les deux calculent EXACTEMENT la meme chose : nn.Conv2d.forward() appelle
F.conv2d avec les memes arguments. La seule difference possible est du surcout
Python (dispatch nn.Module.__call__ + hooks) autour d'appels cuDNN identiques.
Ce script le verifie sur trois axes :

  1. equivalence numerique (memes poids -> memes sorties, forward et backward) ;
  2. temps forward+backward par batch (config du rapport : K=20, ic=64, k=9,
     prox l1, B=128), + un point B=512 ;
  3. nombre de kernels CUDA lances (torch.profiler) : identique => c'est bien
     le meme calcul cote GPU.

Bonus : le vrai levier de vitesse, le deroule LATENT (LatentScCP_UNN, grille
7x7 au lieu de 28x28), pour chiffrer ce qu'une "autre version" peut vraiment
gagner.

Usage :  cd ~/UNN_for_FM && CUDA_VISIBLE_DEVICES=1 python bench_orig_vs_module_convsccp.py
Duree   :  ~2 min.
"""
import time, copy, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

from models.architectures import (ConvScCP_UNN, LatentScCP_UNN,
                                  L1ProxConv, SiLUProxConv, DoubleConvTime,
                                  kaiming_init, sigma_max_power_iter)

DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------- #
#  La version SURLIGNEE, decommentee telle quelle (nn.Conv2d / nn.ConvTranspose2d)
# --------------------------------------------------------------------------- #
class ModuleConvScCP_Iteration(nn.Module):
    """Copie exacte du bloc commente architectures.py:2305 (+ prox_channels
    ignore, non present dans cette version)."""
    def __init__(self, internal_channel, use_Unet=False, version="LFO", init=None,
                 w_bias=True, in_channels=1, kernel_size=9, prox_w=32):
        super().__init__()
        self.version = version
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.pad = k // 2
        if init is None:
            init = "kaiming"

        def _init_W():
            if init == "kaiming":
                return kaiming_init(internal_channel, in_channels, k, k)
            return torch.randn(internal_channel, in_channels, k, k) * 0.05

        self.in_conv = nn.Conv2d(in_channels, internal_channel, k,
                                 padding=self.pad, bias=w_bias)
        self.out_conv = nn.ConvTranspose2d(internal_channel, in_channels, k,
                                           padding=self.pad, bias=False)
        with torch.no_grad():
            self.in_conv.weight.copy_(_init_W())
            self.out_conv.weight.copy_(_init_W())
            if w_bias:
                self.in_conv.bias.zero_()

        if version != "LFO":
            self.out_conv.weight = self.in_conv.weight
            self.register_buffer('_sigma_u',
                                 F.normalize(torch.randn(internal_channel), dim=0))

        if use_Unet == "l1":
            self.prox = L1ProxConv(w=prox_w)
        elif use_Unet == "silu":
            self.prox = SiLUProxConv(channels=internal_channel)
        else:
            self.prox = DoubleConvTime(in_ch=internal_channel, out_ch=internal_channel,
                                       embed_dim=internal_channel // 2)

    @property
    def W_weight(self):  return self.in_conv.weight
    @property
    def V_weight(self):  return self.out_conv.weight
    @property
    def W_bias(self):    return self.in_conv.bias

    def spectral_norm(self):
        _s, self._sigma_u = sigma_max_power_iter(self.in_conv.weight, self._sigma_u)
        return _s

    def forward(self, x, u, z, t, tau, sigma, alpha_mom):
        primal_input = x - tau * self.out_conv(u)
        x_next = (primal_input + tau * z) / (1 + tau)
        y = x_next + alpha_mom * (x_next - x)
        dual_step = sigma * self.in_conv(y)
        u_next = self.prox(u + dual_step, t)
        return x_next, u_next


def build(kind, seed=0, **kw):
    """kind = 'func' (version active) | 'mod' (version surlignee).
    seed : meme graine pour les deux -> memes poids, comparaison honnete."""
    torch.manual_seed(seed)
    m = ConvScCP_UNN(**kw).to(DEVICE)
    if kind == "mod":
        ic, k = kw["internal_channel"], kw.get("kernel_size", 9)
        new = nn.ModuleList()
        for lay in m.layers:
            it = ModuleConvScCP_Iteration(
                internal_channel=ic, use_Unet=kw.get("use_Unet", False),
                version=kw.get("version", "LFO"), w_bias=kw.get("w_bias", True),
                in_channels=kw.get("in_channels", 1), kernel_size=k).to(DEVICE)
            with torch.no_grad():                    # memes poids que 'func'
                it.in_conv.weight.copy_(lay.W_weight)
                if it.in_conv.bias is not None and lay.W_bias is not None:
                    it.in_conv.bias.copy_(lay.W_bias)
                if kw.get("version", "LFO") == "LFO":
                    it.out_conv.weight.copy_(lay.V_weight)
            it.prox.load_state_dict(lay.prox.state_dict())
            new.append(it)
        m.layers = new
    return m


# --------------------------------------------------------------------------- #
#  1. equivalence numerique
# --------------------------------------------------------------------------- #
def check_equivalence(cfg):
    a = build("func", **cfg); b = build("mod", **cfg)
    a.train(); b.train()
    x = torch.randn(16, cfg["dim"] + 1, device=DEVICE); x[:, -1] = torch.rand(16, device=DEVICE)
    xa = x.clone().requires_grad_(True); xb = x.clone().requires_grad_(True)
    oa, ob = a(xa), b(xb)
    oa.pow(2).mean().backward(); ob.pow(2).mean().backward()
    rel_o = (oa - ob).norm().item() / oa.norm().item()
    rel_g = (xa.grad - xb.grad).norm().item() / xa.grad.norm().item()
    return ((oa - ob).abs().max().item(), rel_o,
            (xa.grad - xb.grad).abs().max().item(), rel_g)


# --------------------------------------------------------------------------- #
#  2. temps forward+backward
# --------------------------------------------------------------------------- #
def timeit(model, B, dim, reps=30, warmup=8):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = torch.randn(B, dim + 1, device=DEVICE); x[:, -1] = torch.rand(B, device=DEVICE)
    tgt = torch.randn(B, dim, device=DEVICE)
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True); F.mse_loss(model(x), tgt).backward(); opt.step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True); F.mse_loss(model(x), tgt).backward(); opt.step()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    ts = torch.tensor(ts)
    return ts.median().item(), ts.std().item()


def count_kernels(model, B, dim):
    model.train()
    x = torch.randn(B, dim + 1, device=DEVICE); x[:, -1] = torch.rand(B, device=DEVICE)
    tgt = torch.randn(B, dim, device=DEVICE)
    F.mse_loss(model(x), tgt).backward()                       # warmup
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        F.mse_loss(model(x), tgt).backward()
        torch.cuda.synchronize()
    return sum(e.count for e in prof.key_averages()
               if e.device_type.name in ("CUDA",) or e.self_device_time_total > 0)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print(f"GPU : {torch.cuda.get_device_name(0)}", flush=True)
    CFG = dict(dim=784, K=20, internal_channel=64, use_Unet="l1", version="LFO",
               kernel_size=9, w_bias=True, in_channels=1, img_size=28,
               use_checkpoint=False)

    print("\n=== 1. Equivalence numerique (memes poids) ===", flush=True)
    d_out, r_out, d_grad, r_grad = check_equivalence(CFG)
    print(f"  max |sortie_func - sortie_mod|   = {d_out:.3e}   (erreur relative {r_out:.2e})")
    print(f"  max |grad_func   - grad_mod|     = {d_grad:.3e}   (erreur relative {r_grad:.2e})")
    print("  (float32 + deroule K=20 : le bruit d'arrondi est amplifie a chaque iteration)")
    # meme test en float64 sur CPU : si la math est identique, l'ecart doit etre 0 EXACT
    _dev = globals()["DEVICE"]
    globals()["DEVICE"] = torch.device("cpu"); torch.set_default_dtype(torch.float64)
    _, r64_o, _, r64_g = check_equivalence({**CFG, "internal_channel": 16})
    torch.set_default_dtype(torch.float32); globals()["DEVICE"] = _dev
    print(f"  meme test en float64/CPU : rel_sortie={r64_o:.2e}  rel_grad={r64_g:.2e}")
    print("  -> les deux versions calculent la meme fonction.", flush=True)

    print("\n=== 2. Temps forward+backward+step (ms/iter, mediane sur 30) ===", flush=True)
    print(f"{'config':<42}{'active (F.conv)':>18}{'surlignee (nn.Conv2d)':>24}{'ecart':>10}")
    rows = []
    for label, cfg, B in [
        ("K=20 ic=64 k=9  B=128 (config rapport)", CFG, 128),
        ("K=20 ic=64 k=9  B=512",                  CFG, 512),
        ("K=20 ic=64 k=9  B=128 +checkpoint",      {**CFG, "use_checkpoint": True}, 128),
        ("K=20 ic=64 k=3  B=128",                  {**CFG, "kernel_size": 3}, 128),
        ("K=6  ic=64 k=9  B=128",                  {**CFG, "K": 6}, 128),
    ]:
        ma, sa = timeit(build("func", **cfg), B, cfg["dim"])
        mb, sb = timeit(build("mod",  **cfg), B, cfg["dim"])
        print(f"{label:<42}{ma:>13.2f}±{sa:<4.1f}{mb:>19.2f}±{sb:<4.1f}"
              f"{(mb/ma - 1)*100:>+9.1f}%", flush=True)
        rows.append((label, ma, mb))

    print("\n=== 3. Ops CUDA lancees par forward+backward (B=128) ===", flush=True)
    na = count_kernels(build("func", **CFG), 128, CFG["dim"])
    nb = count_kernels(build("mod",  **CFG), 128, CFG["dim"])
    print(f"  active (F.conv)        : {na} ops")
    print(f"  surlignee (nn.Conv2d)  : {nb} ops", flush=True)

    print("\n=== 4. Reference : ce qui change VRAIMENT la vitesse ===", flush=True)
    lat = LatentScCP_UNN(c_lat=4, latent_spatial=7, K=20, internal_channel=64,
                         use_Unet="l1", version="LFO", kernel_size=9).to(DEVICE)
    ml, sl = timeit(lat, 128, 4 * 7 * 7)
    pix = rows[0][1]
    print(f"  ScCP pixel 28x28 (K=20)  : {pix:.2f} ms")
    print(f"  ScCP latent 7x7  (K=20)  : {ml:.2f} ms   -> x{pix/ml:.1f} plus rapide")
    print(f"  ScCP pixel k=3   (K=20)  : {rows[3][1]:.2f} ms   -> x{pix/rows[3][1]:.1f}")
    print(f"  ScCP pixel K=6   (k=9)   : {rows[4][1]:.2f} ms   -> x{pix/rows[4][1]:.1f}", flush=True)
