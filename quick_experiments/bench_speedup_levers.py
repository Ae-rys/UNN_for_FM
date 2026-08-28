# -*- coding: utf-8 -*-
"""
bench_speedup_levers.py — quels leviers font vraiment gagner du temps sur un
deroule ScCP ? Mesure sur la config EXACTEMENT en cours sur GPU 1
(denoise_probe.py : arch=sccp_v5, k=9, K=10, ic=128, x0=xt, B=128, MNIST 28x28,
ckpt=1 par defaut) et sur les variantes qui isolent chaque levier :

  ckpt=0            : couper le gradient checkpointing (la VRAM est libre a 88 %)
  cudnn.benchmark   : laisser cuDNN choisir l'algo de conv
  channels_last     : layout NHWC (aligne les convs sur les tensor cores)
  AMP fp16          : moitie du trafic memoire — le modele est memory-bound
  AMP + NHWC        : combinaison
  torch.compile     : fusion des ops elementwise de l'algebre CP
  compile+cudagraphs: mode="reduce-overhead", supprime le cout de lancement des
                      ~2000 kernels minuscules (l'hypothese "latency-bound")
  B=256 / B=512     : temps PAR ECHANTILLON, pour voir le rendement du batch

Sortie : ms/iter (median), acceleration vs baseline, VRAM pic.

ATTENTION : le GPU est partage avec un entrainement en cours -> les valeurs
absolues sont gonflees, mais toutes les variantes subissent la meme contention,
donc les RAPPORTS restent valides.

Usage : cd ~/UNN_for_FM && CUDA_VISIBLE_DEVICES=1 python bench_speedup_levers.py
Duree : ~8 min (dont ~3-4 min de compilation pour les deux variantes compile).
"""
import time, sys, traceback
import torch
import torch.nn.functional as F
import torch._inductor.config as _ind

# Sans ca, torch.compile PLANTE au backward sur ce modele (torch 2.12) :
#   assert_size_stride(buf5, (128,128,28,28), (100352,1,3584,128), convolution_backward)
# inductor decide un layout channels_last pour les convs du deroule mais recoit
# du contigu. Desactiver l'optimisation de layout rend la compilation fonctionnelle.
_ind.layout_optimization = False

from denoise_probe import build_model, DEFAULTS

DEV = torch.device("cuda")
IMG, CH = 28, 1
DIM = CH * IMG * IMG
CFG = dict(DEFAULTS, arch="sccp_v5", k=9, K=10, ic=128, x0="xt", ckpt=1)


def make_batch(B, nhwc=False):
    x1 = torch.randn(B, DIM, device=DEV)
    t = torch.rand(B, 1, device=DEV) * 0.95
    x0 = torch.randn(B, DIM, device=DEV)
    xt = t * x1 + (1 - t) * x0
    if nhwc:
        xt = xt.view(B, CH, IMG, IMG).to(memory_format=torch.channels_last).reshape(B, DIM)
    return torch.cat([xt, t], dim=1), x1, t


def run(label, B=128, ckpt=1, benchmark=False, nhwc=False, amp=False,
        compile_mode=None, reps=15, warmup=6):
    """Renvoie (ms/iter median, ms par 128 echantillons, VRAM pic Mo) ou None."""
    torch.backends.cudnn.benchmark = benchmark
    torch.manual_seed(0)
    model = build_model(dict(CFG, ckpt=ckpt), DEV, CH, IMG)
    if nhwc:
        model = model.to(memory_format=torch.channels_last)
    if compile_mode is not None:
        t0 = time.perf_counter()
        model = torch.compile(model, mode=compile_mode)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    xt_t, x1, t = make_batch(B, nhwc)
    model.train()
    torch.cuda.reset_peak_memory_stats()

    def step(read=False):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            pred = model(xt_t)
            loss = torch.mean((pred.float() - x1) ** 2 / torch.clamp((1 - t) ** 2, min=0.05))
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        # mode="reduce-overhead" : les sorties vivent dans le pool CUDA Graphs et
        # sont ecrasees au pas suivant -> ne JAMAIS garder le tenseur, cloner.
        return loss.detach().clone() if read else None

    try:
        for i in range(warmup):
            l = step(read=(i == warmup - 1))
        torch.cuda.synchronize()
        if compile_mode is not None:
            print(f"    (compilation + warmup : {time.perf_counter() - t0:.0f} s)", flush=True)
        if not torch.isfinite(l):
            print(f"  {label:<34} loss non finie -> variante inutilisable", flush=True)
            return None
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize(); s = time.perf_counter()
            step()
            torch.cuda.synchronize(); ts.append((time.perf_counter() - s) * 1e3)
    except Exception as e:
        print(f"  {label:<34} ECHEC : {type(e).__name__}: {str(e)[:90]}", flush=True)
        return None
    finally:
        vram = torch.cuda.max_memory_allocated() / 2**20
        del model, opt
        torch.cuda.empty_cache()
    ms = torch.tensor(ts).median().item()
    return ms, ms * 128 / B, vram


if __name__ == "__main__":
    print(f"GPU : {torch.cuda.get_device_name(0)}  | torch {torch.__version__}")
    print("config = sccp_v5 k=9 K=10 ic=128 x0=xt, MNIST 28x28, B=128 "
          "(identique au run en cours sur GPU 1)")
    print("NB : GPU partage avec un entrainement -> lire les RAPPORTS, "
          "pas les ms absolues.\n", flush=True)

    variants = [
        ("baseline (ckpt=1, fp32)",        dict()),
        ("ckpt=0",                         dict(ckpt=0)),
        ("ckpt=0 + cudnn.benchmark",       dict(ckpt=0, benchmark=True)),
        ("ckpt=0 + channels_last",         dict(ckpt=0, benchmark=True, nhwc=True)),
        ("ckpt=0 + AMP fp16",              dict(ckpt=0, benchmark=True, amp=True)),
        ("ckpt=0 + AMP fp16 + NHWC",       dict(ckpt=0, benchmark=True, amp=True, nhwc=True)),
        ("ckpt=0 + compile",               dict(ckpt=0, benchmark=True, compile_mode="default")),
        ("ckpt=0 + compile+cudagraphs",    dict(ckpt=0, benchmark=True,
                                                compile_mode="reduce-overhead")),
        ("ckpt=0 + compile + AMP fp16",    dict(ckpt=0, benchmark=True, amp=True,
                                                compile_mode="default")),
        ("ckpt=0, B=256",                  dict(ckpt=0, benchmark=True, B=256)),
        ("ckpt=0, B=512",                  dict(ckpt=0, benchmark=True, B=512)),
        ("ckpt=0 + compile, B=512",        dict(ckpt=0, benchmark=True, B=512,
                                                compile_mode="default")),
    ]

    print(f"{'variante':<34}{'ms/iter':>10}{'ms/128 ech.':>14}{'gain':>9}{'VRAM Mo':>10}")
    base = None
    for label, kw in variants:
        r = run(label, **kw)
        if r is None:
            continue
        ms, ms128, vram = r
        if base is None:
            base = ms128
        print(f"  {label:<32}{ms:>10.1f}{ms128:>14.1f}{base/ms128:>8.2f}x{vram:>10.0f}",
              flush=True)
