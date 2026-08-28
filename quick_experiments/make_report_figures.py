# -*- coding: utf-8 -*-
"""
make_report_figures.py
Figures du document de synthese ScCP vs UNet : les memes images, les memes
niveaux de bruit, les memes poids que ceux dont on cite les chiffres.

Trois planches
--------------
  fig_denoise_compare.png   LA comparaison a l'oeil. Six images de validation,
                            quatre niveaux de bruit, et pour chacun la prediction
                            du ScCP, du UNet torchcfm et du UNet de Kamb —
                            tous entraines 20 000 steps sous le MEME protocole.
  fig_samples_fm.png        Les echantillons GENERES par les runs Flow Matching
                            de results_afhq32, tries par budget de steps. C'est
                            la seule planche qui montre de la generation.
  fig_t0_ladder.png         Depart retarde : on part d'un vrai x_t0 et on integre
                            [t0, 1]. La ligne ou ca devient net borne l'intervalle
                            de temps encore mal appris.

Toutes les predictions utilisent le bruit FIXE de denoise_probe.py (memes seeds),
donc ce qui est montre est exactement ce qui est chiffre dans le document.

Usage
-----
    source ~/.venvs/unn/bin/activate
    python make_report_figures.py

Sorties -> report_figs/*.png
"""

import gc
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from denoise_probe import (build_model, forward_x1, load_data, make_val_set,
                           name_to_config)

OUT = "report_figs"
CACHE = "./data/afhq_cat32_train.pt"
T_LIST = [0.2, 0.4, 0.6, 0.8]
N_SHOW = 6
DEV = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

PROBE = [
    ("results_denoise_probe_20k/ScCP_k9_K20_ic128_l1_LFO", "ScCP  k9/K20/ic128", "1.25M"),
    ("results_denoise_probe_20k/unet_ref_ch32_b1_m1-2-2", "UNet torchcfm ch32", "1.11M"),
    ("results_denoise_probe_20k/unet_kamb", "UNet Kamb", "2.11M"),
]

FM_RUNS = [
    ("ConvScCP_UNN_rgb_k15_K20_ic256_L1_LFO", 165000, "k15/K20/ic256", "6.92M", 0.1976),
    ("ConvScCP_UNN_rgb_k9_K10_ic128_L1_LFO", 200000, "k9/K10/ic128", "0.62M", 0.2056),
    ("ConvScCP_UNN_rgb_k9_K10_ic256_L1_LFO", 55000, "k9/K10/ic256", "1.25M", 0.2142),
    ("ConvScCP_UNN_rgb_k9_K20_ic256_L1_LFO", 45000, "k9/K20/ic256", "2.50M", 0.2157),
    ("ConvScCP_UNN_rgb_k15_K15_ic512_L1_LFO", 50000, "k15/K15/ic512", "10.38M", 0.2239),
]


def to_img(x, C=3, S=32):
    return (x.reshape(-1, C, S, S).cpu() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()


def load_probe_model(run_dir, C, S):
    name = os.path.basename(run_dir)
    cfg = name_to_config(name)
    model = build_model(cfg, DEV, C, S)
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location=DEV,
                    weights_only=False)
    model.load_state_dict(ck.get("ema_model", ck["state_dict"]))
    model.eval()
    return model, cfg["arch"] == "unet_ref"


def fig_denoise_compare(val_sets, C, S):
    """Lignes : x_t puis une prediction par modele, bloc par niveau de bruit."""
    preds = {}
    for run_dir, label, _ in PROBE:
        model, is_ref = load_probe_model(run_dir, C, S)
        with torch.no_grad():
            for t in T_LIST:
                xt, _ = val_sets[t]
                tb = torch.full((N_SHOW, 1), float(t), device=DEV)
                preds[(label, t)] = forward_x1(model, xt[:N_SHOW], tb, C, S, is_ref).cpu()
        del model; gc.collect(); torch.cuda.empty_cache()

    labels = [l for _, l, _ in PROBE]
    rows, names, kinds = [], [], []
    for t in T_LIST:
        xt, x1 = val_sets[t]
        rows.append(xt[:N_SHOW].cpu()); names.append(f"$x_t$   t={t:g}"); kinds.append("in")
        for l in labels:
            rows.append(preds[(l, t)]); names.append(l); kinds.append("pred")
    rows.append(val_sets[T_LIST[0]][1][:N_SHOW].cpu())
    names.append("verite  $x_1$"); kinds.append("truth")

    nr = len(rows)
    fig, axes = plt.subplots(nr, N_SHOW, figsize=(1.5 * N_SHOW, 1.42 * nr), squeeze=False)
    for r, (row, nm, kd) in enumerate(zip(rows, names, kinds)):
        imgs = to_img(row, C, S)
        for c in range(N_SHOW):
            ax = axes[r][c]
            ax.imshow(imgs[c]); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color({"in": "#9AA6A4", "truth": "#1A1F22"}.get(kd, "#C9D2D0"))
                sp.set_linewidth(1.6 if kd != "pred" else 0.8)
            if c == 0:
                ax.set_ylabel(nm, fontsize=8.5, rotation=0, ha="right", va="center",
                              fontweight="bold" if kd != "pred" else "normal")
    fig.suptitle("Debruitage a bruit fixe — memes images, memes bruits, 20 000 steps chacun",
                 fontsize=11, y=0.997)
    plt.tight_layout(rect=(0.10, 0, 1, 0.99))
    plt.savefig(f"{OUT}/fig_denoise_compare.png", dpi=105, bbox_inches="tight")
    plt.close(fig)
    print("  fig_denoise_compare.png", flush=True)


def fig_samples_fm():
    """Echantillons generes, une ligne par run FM, tries par nmse."""
    from PIL import Image
    rows, labels = [], []
    for d, step, short, params, nmse in FM_RUNS:
        p = f"results_afhq32/{d}/step_{step}.png"
        if not os.path.exists(p):
            cands = sorted(f for f in os.listdir(f"results_afhq32/{d}")
                           if f.startswith("step_") and f.endswith(".png"))
            if not cands:
                continue
            p = f"results_afhq32/{d}/{cands[-1]}"
        rows.append(np.asarray(Image.open(p).convert("RGB")))
        labels.append(f"{short}\n{params} · {step//1000}k steps\nnmse {nmse:.4f}")
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 1.85 * len(rows)), squeeze=False)
    for i, (im, lab) in enumerate(zip(rows, labels)):
        ax = axes[i][0]
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_ylabel(lab, fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle("Echantillons generes (EDO complete) — runs Flow Matching, tries par nmse",
                 fontsize=11)
    plt.tight_layout(rect=(0.13, 0, 1, 0.97))
    plt.savefig(f"{OUT}/fig_samples_fm.png", dpi=105, bbox_inches="tight")
    plt.close(fig)
    print("  fig_samples_fm.png", flush=True)


def fig_t0_ladder():
    """Recadre les deux planches de depart retarde deja produites, cote a cote."""
    from PIL import Image
    pairs = [
        ("results_afhq32/ConvScCP_UNN_rgb_k9_K20_ic256_L1_LFO/"
         "sampler_diag_ConvScCP_UNN_rgb_k9_K20_ic256_L1_LFO_step45000_start.png",
         "45 000 steps — pates jusqu'a t0 = 0.5"),
        ("results_afhq32/ConvScCP_UNN_rgb_k15_K20_ic256_L1_LFO/"
         "sampler_diag_ConvScCP_UNN_rgb_k15_K20_ic256_L1_LFO_step165000_start.png",
         "165 000 steps — net des t0 = 0"),
    ]
    have = [(p, l) for p, l in pairs if os.path.exists(p)]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(6.2 * len(have), 8), squeeze=False)
    for i, (p, lab) in enumerate(have):
        ax = axes[0][i]
        ax.imshow(np.asarray(Image.open(p).convert("RGB")))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(lab, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_t0_ladder.png", dpi=95, bbox_inches="tight")
    plt.close(fig)
    print("  fig_t0_ladder.png", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    _, x_val = load_data(CACHE, 512, DEV, seed=0)
    C, S = x_val.shape[1], x_val.shape[2]
    val_sets = make_val_set(x_val, T_LIST, seed=1234)
    print(f"Figures -> {OUT}/", flush=True)
    fig_denoise_compare(val_sets, C, S)
    fig_samples_fm()
    fig_t0_ladder()


if __name__ == "__main__":
    main()
