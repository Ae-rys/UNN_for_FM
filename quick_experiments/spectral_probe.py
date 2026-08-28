# -*- coding: utf-8 -*-
"""
spectral_probe.py
Mesure de "pate" : densite spectrale de puissance radiale des echantillons
GENERES, comparee a celle des vraies images.

Pourquoi la MSE ne peut pas voir les pates
------------------------------------------
Les images naturelles ont un spectre en ~1/f^2 : l'essentiel de l'energie, donc
de la MSE, est dans les basses frequences. Un modele qui reproduit parfaitement
la structure grossiere et n'emet AUCUNE haute frequence paie une penalite MSE
minuscule. Pire, la MSE de debruitage est minimisee par E[x1|x_t], la moyenne
conditionnelle, qui est floue par construction : la metrique RECOMPENSE le flou.

D'ou cet instrument. On compare, frequence par frequence :

    ratio(f) = PSD_generee(f) / PSD_reelle(f)

  ratio ~ 1 partout        spectre correct
  ratio < 1 en haut        PATE : il manque du detail (le diagnostic recherche)
  ratio > 1 en haut        bruit / artefacts haute frequence

Le chiffre resume `HF ratio` est la moyenne geometrique du ratio sur la moitie
haute des frequences (f > f_Nyquist/2).

Deux modes
----------
  --ckpt   checkpoint GENERATIF (results_afhq32/...) : on echantillonne par
           l'EDO complete. C'est la mesure directe du pate.
  --probe  modele du banc de debruitage (results_denoise_probe*/...) : on mesure
           le spectre de x1_pred a un t donne. Repond a "le probe pouvait-il
           voir venir le pate ?".

Usage
-----
    source ~/.venvs/unn/bin/activate

    python spectral_probe.py --ckpt \\
        results_afhq32/ConvScCP_UNN_rgb_k9_K20_ic256_L1_LFO/latest.pt \\
        results_afhq32/ConvScCP_UNN_rgb_k15_K20_ic256_L1_LFO/latest.pt --n 128

    python spectral_probe.py --probe \\
        results_denoise_probe_20k/ScCP_k9_K20_ic128_l1_LFO \\
        results_denoise_probe_20k/unet_ref_ch32_b1_m1-2-2 --probe-t 0.5

Sorties -> spectral_probe.png + spectral_probe.txt
"""

import argparse
import gc
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def radial_psd(imgs):
    """PSD radialement moyennee. imgs : (B, C, S, S) dans [-1, 1].

    On retire la moyenne par image (le terme DC porterait toute l'energie et
    ecraserait le reste), on prend |FFT2|^2, on moyenne sur le batch et les
    canaux, puis on moyenne sur des anneaux de rayon entier."""
    x = imgs - imgs.mean(dim=(-2, -1), keepdim=True)
    F = torch.fft.fftshift(torch.fft.fft2(x.double()), dim=(-2, -1))
    p = (F.real ** 2 + F.imag ** 2).mean(dim=(0, 1))            # (S, S)
    S = p.shape[-1]
    c = S // 2
    yy, xx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
    r = torch.sqrt(((yy - c).double()) ** 2 + ((xx - c).double()) ** 2).round().long()
    nb = c + 1
    out = torch.zeros(nb, dtype=torch.float64)
    cnt = torch.zeros(nb, dtype=torch.float64)
    rf, pf = r.flatten(), p.flatten().cpu()
    keep = rf < nb
    out.index_add_(0, rf[keep], pf[keep])
    cnt.index_add_(0, rf[keep], torch.ones(keep.sum(), dtype=torch.float64))
    return (out / cnt.clamp_min(1)).numpy()


def hf_ratio(ratio):
    """Moyenne geometrique du ratio sur la moitie haute du spectre."""
    half = ratio[len(ratio) // 2:]
    half = half[np.isfinite(half) & (half > 0)]
    return float(np.exp(np.log(half).mean())) if half.size else float("nan")


def real_psd(cache, n, device, seed=0):
    d = torch.load(cache)
    x = d["data"].float().div_(127.5).sub_(1.0)
    g = torch.Generator().manual_seed(seed)
    x = x[torch.randperm(x.shape[0], generator=g)][:n]
    return radial_psd(x.to(device)), x.shape[-1]


def gen_samples(ckpt_path, n, steps, device, weights="ema", batch=64):
    from sample_checkpoint import resolve_checkpoint
    from compute_fid_cifar10 import _VelocityWrapper, sample_batch
    model, is_unet, name, keys = resolve_checkpoint(ckpt_path, device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    key = keys.get(weights) or keys.get("raw")
    model.load_state_dict(ck[key])
    model.eval()
    vf = _VelocityWrapper(model, is_unet)
    outs = []
    with torch.no_grad():
        for i in range(0, n, batch):
            outs.append(sample_batch(vf, min(batch, n - i), device,
                                     solver="euler", steps=steps).cpu())
            print(f"    {min(i+batch, n)}/{n} echantillons", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    return torch.cat(outs), name


def probe_pred(run_dir, t, n, device, cache, n_val=512, seed=0):
    from denoise_probe import (build_model, forward_x1, load_data, make_val_set,
                               name_to_config)
    name = os.path.basename(run_dir.rstrip("/"))
    _, x_val = load_data(cache, n_val, device, seed=seed)
    C, S = x_val.shape[1], x_val.shape[2]
    val = make_val_set(x_val, [t], seed=seed + 1234)
    xt, x1 = val[t]
    cfg = name_to_config(name)
    model = build_model(cfg, device, C, S)
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location=device,
                    weights_only=False)
    model.load_state_dict(ck.get("ema_model", ck["state_dict"]))
    model.eval()
    with torch.no_grad():
        tb = torch.full((min(n, xt.shape[0]), 1), float(t), device=device)
        pred = forward_x1(model, xt[:tb.shape[0]], tb, C, S, cfg["arch"] == "unet_ref")
    del model; gc.collect(); torch.cuda.empty_cache()
    return pred.view(-1, C, S, S).cpu(), x1[:tb.shape[0]].view(-1, C, S, S).cpu(), name


def main():
    p = argparse.ArgumentParser(description="Spectre des echantillons : detecteur de pate.")
    p.add_argument("--ckpt", nargs="*", default=[], help="Checkpoints generatifs.")
    p.add_argument("--probe", nargs="*", default=[], help="Dossiers de run du banc.")
    p.add_argument("--probe-t", type=float, default=0.5)
    p.add_argument("--cache", type=str, default="./data/afhq_cat32_train.pt")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--steps", type=int, default=100, help="Pas d'Euler pour l'EDO.")
    p.add_argument("--weights", type=str, default="ema", choices=["ema", "raw"])
    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--out", type=str, default="spectral_probe.png")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and (device.index or 0) >= torch.cuda.device_count():
        device = torch.device("cuda:0")

    ref, S = real_psd(args.cache, max(args.n, 512), device)
    freqs = np.arange(len(ref))
    curves, lines = [], []

    for c in args.ckpt:
        print(f"  EDO ({args.steps} pas d'Euler) : {c}", flush=True)
        imgs, name = gen_samples(c, args.n, args.steps, device, args.weights)
        psd = radial_psd(imgs.to(device))
        curves.append((f"{name} [genere]", psd / ref))

    for run in args.probe:
        print(f"  probe t={args.probe_t} : {run}", flush=True)
        pred, truth, name = probe_pred(run, args.probe_t, args.n, device, args.cache)
        # normalise par le spectre des MEMES images de reference (x1 du batch),
        # pas par celui du dataset : on isole ce que le modele perd, pas
        # l'ecart d'echantillonnage.
        curves.append((f"{name} [x1_pred, t={args.probe_t:g}]",
                       radial_psd(pred.to(device)) / radial_psd(truth.to(device))))

    if not curves:
        print("Rien a faire : donner --ckpt et/ou --probe.", flush=True)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, ratio in curves:
        h = hf_ratio(ratio)
        ax.plot(freqs, ratio, "o-", ms=3, lw=1.4, label=f"{label}   HF={h:.2f}")
        lines.append(f"{label:<52} HF_ratio={h:.3f}")
    ax.axhline(1.0, color="k", lw=1.2, ls="--")
    ax.axvline(len(ref) / 2, color="gray", lw=0.8, ls=":")
    ax.text(len(ref) / 2 + 0.2, ax.get_ylim()[1] * 0.9, "moitie haute", fontsize=7,
            color="gray")
    ax.set_yscale("log")
    ax.set_xlabel("frequence radiale (pixels^-1, 0 = basses freq.)")
    ax.set_ylabel("PSD generee / PSD reelle")
    ax.set_title("Detecteur de pate : ratio spectral\n"
                 "<1 en haut = detail manquant  |  =1 = spectre correct", fontsize=10)
    ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(args.out, dpi=110); plt.close(fig)

    txt = os.path.splitext(args.out)[0] + ".txt"
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)
    print(f"\n-> {args.out}\n-> {txt}", flush=True)


if __name__ == "__main__":
    main()
