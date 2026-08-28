# -*- coding: utf-8 -*-
"""
fid_ckpt_mnist.py
mini-FID d'un checkpoint MNIST (run_mnist.py) a N grand, avec les deux
references etudiees dans fid_floor_mnist.py :
    - APPARIEE : N vraies images (protocole actuel des sweeps)
    - COMPLETE : les 60000 du train (gratuit, moitie du bruit en moins)

Une seule passe d'echantillonnage a N_max : les valeurs a N plus petit sont
obtenues en sous-echantillonnant les features generees, ce qui donne la courbe
de convergence sans regenerer quoi que ce soit.

L'archi est reconstruite depuis <run_dir>/parametres.txt (model_class, K,
dual_dim, version), les poids depuis <run_dir>/model.pt.

Usage
-----
    # smoke-test chronometre (donne l'ETA pour le vrai run)
    CUDA_VISIBLE_DEVICES=0 python fid_ckpt_mnist.py \
        --run-dir results/temp-4/ConvScCP_UNN_L1_LNO --n 512

    # le vrai run
    CUDA_VISIBLE_DEVICES=0 python fid_ckpt_mnist.py \
        --run-dir results/temp-4/ConvScCP_UNN_L1_LNO --n 10000

Sortie -> <run_dir>/fid_eval/{fid_ckpt.md, fid_vs_n.png, samples.png}
"""
import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper

from models.architectures import (ConvScCP_UNN, ConvDFB_UNN, ConvDiFB_UNN,
                                  SmallUNet, SmallUNetX1)
from mnist_metrics import (train_or_load_classifier, _embed_all, class_entropy,
                           frechet_distance)
from fid_floor_mnist import get_datasets, as_tensor, fid_from_feats

BUILDERS = {"ConvScCP_UNN": ConvScCP_UNN, "ConvDFB_UNN": ConvDFB_UNN,
            "ConvDiFB_UNN": ConvDiFB_UNN}
UNET_BUILDERS = {"SmallUNet": SmallUNet, "SmallUNetX1": SmallUNetX1}
FEAT_CACHE = "results/fid_floor_mnist/feats_train.npy"


def read_params(run_dir):
    """<run_dir>/parametres.txt, format cle=valeur."""
    cfg = {}
    with open(os.path.join(run_dir, "parametres.txt")) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                cfg[k] = v
    return cfg


def build_model(cfg, state, device):
    """Reconstruit l'archi. Les defauts du constructeur ont bouge depuis les vieux
    runs (prox_w=8 a l'epoque, 32 aujourd'hui), donc les hyperparametres de FORME
    sont deduits du state_dict lui-meme ; parametres.txt ne sert que de repli."""
    name = cfg["model_class"]
    if name in UNET_BUILDERS:                           # baselines UNet : pas de K ni de prox
        base_ch, in_ch = state["time_scaling.0.weight"].shape
        print(f"[archi] deduite du checkpoint : {name} base_ch={base_ch} "
              f"in_ch={in_ch}", flush=True)
        return UNET_BUILDERS[name](in_channels=in_ch, out_channels=in_ch,
                                   base_ch=base_ch).to(device)

    cls = BUILDERS[name]
    w = state["layers.0.W_weight"]                      # (ic, in_ch, k, k)
    ic, in_ch, k = w.shape[0], w.shape[1], w.shape[-1]
    prox_w = state["layers.0.prox.time_scaling.0.weight"].shape[0]
    K = state["log_tau"].shape[0] if "log_tau" in state else int(cfg["K"])
    img = int(round((784 // in_ch) ** 0.5)) if in_ch == 1 else 32
    print(f"[archi] deduite du checkpoint : K={K} ic={ic} in_ch={in_ch} "
          f"kernel={k} prox_w={prox_w} w_bias={'layers.0.W_bias' in state}", flush=True)
    return cls(dim=in_ch * img * img, K=K, internal_channel=ic, use_Unet="l1",
               version=cfg["version"], use_checkpoint=False,
               w_bias="layers.0.W_bias" in state, in_channels=in_ch,
               img_size=img, kernel_size=k, prox_w=prox_w).to(device)


@torch.no_grad()
def generate(model, n, device, batch_size, solver, steps, seed=0):
    """Echantillonne n images, par lots, avec progression + ETA."""
    torch.manual_seed(seed)
    node = NeuralODE(torch_wrapper(model), solver=solver, atol=1e-5, rtol=1e-5)
    t_span = torch.linspace(0, 1, 2 if solver == "dopri5" else steps + 1, device=device)
    out, t0 = [], time.perf_counter()
    done = 0
    while done < n:
        b = min(batch_size, n - done)
        x0 = torch.randn(b, 784, device=device)
        out.append(node.trajectory(x0, t_span=t_span)[-1].cpu())
        done += b
        el = time.perf_counter() - t0
        print(f"  [gen] {done}/{n}  {el:.0f}s ecoulees  "
              f"ETA {el / done * (n - done):.0f}s", flush=True)
    return torch.cat(out).view(-1, 1, 28, 28).clamp(-1, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--n", type=int, default=10000, help="Nombre d'images generees.")
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--solver", default="dopri5", choices=["dopri5", "euler"],
                   help="dopri5 = le solveur utilise pendant l'entrainement.")
    p.add_argument("--steps", type=int, default=100, help="Pas fixes si --solver euler.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=5,
                   help="Tirages pour les N inferieurs a --n.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.run_dir, "fid_eval")
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.perf_counter()

    cfg = read_params(args.run_dir)
    print(f"[cfg] {cfg}", flush=True)
    state = torch.load(os.path.join(args.run_dir, "model.pt"), map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    model = build_model(cfg, state, device)
    model.load_state_dict(state)          # strict : une archi mal deduite doit crasher
    model.eval()
    n_par = sum(q.numel() for q in model.parameters())
    desc = " ".join(f"{k}={cfg[k]}" for k in ("K", "dual_dim", "version") if k in cfg)
    print(f"[model] {cfg['model_class']} {desc} | {n_par} params | device={device}",
          flush=True)

    # ------------------------------------------------------------ generation
    print(f"[gen] {args.n} images, solveur {args.solver}"
          f"{'' if args.solver == 'dopri5' else f'-{args.steps}'}, batch {args.batch}",
          flush=True)
    t_gen = time.perf_counter()
    gen = generate(model, args.n, device, args.batch, args.solver, args.steps, args.seed)
    t_gen = time.perf_counter() - t_gen
    print(f"[gen] fini en {t_gen:.0f}s ({t_gen / args.n * 1000:.1f} ms/image)", flush=True)

    # -------------------------------------------------- reference + features
    train_ds, test_ds = get_datasets()
    x_train, _ = as_tensor(train_ds)
    clf = train_or_load_classifier(
        DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2), device)

    if os.path.exists(FEAT_CACHE):
        feat_train = np.load(FEAT_CACHE)
        print(f"[feat] reference chargee depuis {FEAT_CACHE} {feat_train.shape}", flush=True)
    else:
        feat_train = _embed_all(clf, x_train, device)
        np.save(FEAT_CACHE, feat_train)
        print(f"[feat] reference calculee et mise en cache -> {FEAT_CACHE}", flush=True)
    feat_gen = _embed_all(clf, gen, device)

    ent, counts = class_entropy(clf, gen, device)
    std_gen = float(gen.view(len(gen), -1).std(dim=0).mean().item())
    print(f"[div] entropie de classe = {ent:.4f} | std_gen = {std_gen:.4f}", flush=True)
    print(f"[div] repartition des chiffres predits : {counts}", flush=True)

    # ------------------------------------------------------------ evaluation
    rng = np.random.default_rng(args.seed)
    sizes = [n for n in (2000, 5000, 10000) if n <= args.n] or [args.n]
    rows = []
    for n in sizes:
        n_rep = 1 if n == args.n else args.n_seeds
        matched, full = [], []
        for _ in range(n_rep):
            i_g = (np.arange(args.n) if n == args.n
                   else rng.choice(args.n, n, replace=False))
            i_r = rng.choice(len(feat_train), n, replace=False)
            matched.append(fid_from_feats(feat_train[i_r], feat_gen[i_g]))
            full.append(fid_from_feats(feat_train, feat_gen[i_g]))
        rows.append(dict(n=n, m=float(np.mean(matched)), m_std=float(np.std(matched)),
                         f=float(np.mean(full)), f_std=float(np.std(full))))
        r = rows[-1]
        print(f"[FID] N={n:6d}  ref appariee = {r['m']:7.3f} +/- {r['m_std']:.3f}"
              f"   ref complete = {r['f']:7.3f} +/- {r['f_std']:.3f}", flush=True)

    # ---------------------------------------------------------------- sorties
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = gen[:64].view(8, 8, 28, 28).permute(0, 2, 1, 3).reshape(8 * 28, 8 * 28)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.numpy(), cmap="gray", vmin=-1, vmax=1)
    plt.axis("off")
    plt.title(f"{os.path.basename(args.run_dir)} — 64 / {args.n} echantillons")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "samples.png"), dpi=140)
    plt.close()

    # planchers mesures dans fid_floor_mnist.py, pour situer la valeur du modele
    FLOOR = {2000: (3.78, 1.99), 5000: (1.87, 1.42), 10000: (1.33, 1.03)}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ns = [r["n"] for r in rows]
    ax.errorbar(ns, [r["m"] for r in rows], yerr=[r["m_std"] for r in rows],
                marker="o", capsize=3, label="modele, ref appariee (N reels)")
    ax.errorbar(ns, [r["f"] for r in rows], yerr=[r["f_std"] for r in rows],
                marker="^", capsize=3, ls="-.", label="modele, ref complete (60k)")
    fl = [FLOOR[n] for n in ns if n in FLOOR]
    if len(fl) == len(ns):
        ax.plot(ns, [f[0] for f in fl], ls=":", color="gray", marker="o",
                label="plancher, ref appariee")
        ax.plot(ns, [f[1] for f in fl], ls=":", color="k", marker="^",
                label="plancher, ref complete")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N echantillons generes"); ax.set_ylabel("mini-FID")
    ax.set_title(f"{os.path.basename(args.run_dir)} — mini-FID vs plancher")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fid_vs_n.png"), dpi=140)

    md = [f"# mini-FID — {os.path.basename(args.run_dir)}\n",
          f"`{cfg['model_class']}` {desc}, "
          f"{n_par} parametres. Echantillonnage : {args.n} images, solveur "
          f"{args.solver}, {t_gen:.0f}s ({t_gen / args.n * 1000:.1f} ms/image).\n",
          f"Entropie de classe = {ent:.4f} (1.0 = les 10 chiffres equilibres), "
          f"std_gen = {std_gen:.4f}.\n",
          f"Chiffres predits : {counts}\n",
          "Chaque reference donne une paire (FID du modele, plancher qu'un modele "
          "PARFAIT atteindrait avec ce meme protocole).\n",
          "| N generes | FID, ref appariee (N reels) | plancher apparie "
          "| FID, ref complete (60k reels) | plancher complet |",
          "|---|---|---|---|---|"]
    for r in rows:
        f0, f1 = FLOOR.get(r["n"], (float("nan"), float("nan")))
        md.append(f"| {r['n']} | {r['m']:.2f} +/- {r['m_std']:.2f} | {f0:.2f} "
                  f"| {r['f']:.2f} +/- {r['f_std']:.2f} | {f1:.2f} |")
    md.append("\nPlanchers : fid_floor_mnist.py (train vs test, memes N). Un modele "
              "parfait atteindrait la colonne plancher, pas 0.\n")
    with open(os.path.join(out_dir, "fid_ckpt.md"), "w") as f:
        f.write("\n".join(md))
    print(f"\n[out] {out_dir}/{{fid_ckpt.md, fid_vs_n.png, samples.png}}", flush=True)
    print(f"[done] total {time.perf_counter() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
