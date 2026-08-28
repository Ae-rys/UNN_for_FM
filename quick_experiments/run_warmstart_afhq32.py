# -*- coding: utf-8 -*-
"""
run_warmstart_afhq32.py
Warm-start du dual ConvScCP sur AFHQ-chats 32x32 (RGB) : meme experience que
run_warmstart_mnist.py, mais sur la recette torchcfm de run_afhq32.py (EMA, warmup,
checkpoints + reprise automatique).

Rappel du mecanisme (Jabri, Fleet & Chen, ICML 2023, arXiv:2212.11972 §2.3) : le dual u
du deroule Chambolle-Pock est remis a zero a chaque evaluation du champ. On le REPORTE
d'un pas d'Euler au suivant (report = IDENTITE, aucune porte apprise), et a
l'entrainement on fournit au modele, avec probabilite `self_cond_rate`, le u^(K) d'une
passe FROIDE sans gradient sur x_{t-dt} de la MEME paire (x0, x1).

Pourquoi les chats plutot que MNIST : le canal dual y est LARGEMENT ouvert. Mesure sur
le checkpoint existant results_afhq32/…k9_K10_ic256_L1_LFO (diag_u0_forgetting.py) :
apres les 10 iterations, l'ecart entre deroule chaud et froid vaut encore ~35 % (primal
et dual). A titre de comparaison, MNIST K=20 tombe a 0.8 % (canal ferme, l'experience
n'y mesure rien) et MNIST K=6 garde 50 % (c'est la que le warm-start a paye : mini-FID
86 -> 64). K=10 sur AFHQ est donc dans le bon regime.

Deux configurations comparees :
    baseline   self_cond_rate=0.0, echantillonnage a dual froid  (= modele actuel)
    warmstart  self_cond_rate=0.9, echantillonnage a dual chaine

Par defaut SEUL `warmstart` est entraine : le baseline est le run AFHQ deja fait
(--baseline-ckpt, 55k steps, meme recette torchcfm, meme K/ic/kernel/version/coupling),
ce qui economise ~4 h de GPU. --steps vaut donc 55000, le budget de ce checkpoint, pour
que la comparaison reste a budget egal. Deux ecarts assumes, sans effet sur la fonction
calculee : le run existant a ete construit avec use_checkpoint=True (recompute des
activations : memoire/vitesse, pas les gradients), et les deux runs ne partagent ni la
graine d'init ni l'ordre des batches — la recette RGB de ce repo n'a jamais fixe de
graine. Pour une paire strictement appariee : --runs both --baseline-ckpt ''.

Usage
-----
    source ~/.venvs/unn/bin/activate

    # smoke-test / ETA d'abord (~2 min)
    python run_warmstart_afhq32.py --steps 200 --outdir /tmp/smoke_ws_afhq

    # le vrai run (warmstart seul, baseline reutilise), suivable dans claude.log ;
    # reprise auto apres crash : relancer la MEME commande
    CUDA_VISIBLE_DEVICES=1 nohup python run_warmstart_afhq32.py >> claude.log 2>&1 &

    # entrainer aussi le baseline ici (paire strictement appariee, ~+4 h)
    python run_warmstart_afhq32.py --runs both --baseline-ckpt ''

    # comparaison seule, sans rien entrainer
    python run_warmstart_afhq32.py --eval-only

Sorties (--outdir, defaut results_warmstart_afhq32/) :
    <name>_<tag>/           checkpoints (latest.pt + archives), loss.png, step_N.png
    comparison_samples.png  grilles cold/warm x baseline/warmstart, meme bruit
    du_curve.png            ||u_K^(n) - u_K^(n-1)|| / ||u_K^(n-1)|| le long de l'ODE
    summary.txt             GE / std / temps, et rappel du protocole

Pas de FID ici : clean-fid n'a pas de statistiques de reference AFHQ dans ce repo
(compute_fid_cifar10.py est specifique CIFAR-10). La comparaison est visuelle + GE/std
contre les vraies images, comme sur les autres runs AFHQ.
"""
import argparse
import gc
import os
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.architectures import ConvScCP_UNN
from run_afhq32 import get_afhq_loader
from run_cifar10_torchcfm_recipe import RECIPE, train_one

IMG_SIZE, CHANNELS = 32, 3
DIM = CHANNELS * IMG_SIZE * IMG_SIZE
# Nombre de pas d'Euler, PARTAGE entre entrainement (dt du self-conditioning) et
# echantillonnage : l'entrainement doit voir le dt que l'inference utilisera.
N_STEPS = 100


# ------------------------------------------------------------------ echantillonnage
@torch.no_grad()
def sample(model, n, device, warm, n_steps=N_STEPS, x0=None, record_du=False):
    """Euler a pas fixe, t: 0 -> 1. warm=True : u^(K) est reporte d'un pas au suivant
    (u n'est JAMAIS remis a zero dans la boucle). warm=False : dual froid a chaque pas
    = echantillonneur actuel. record_du : suit ||u_K^(n)-u_K^(n-1)||/||u_K^(n-1)||."""
    was_training = model.training
    model.eval()
    x = torch.randn(n, DIM, device=device) if x0 is None else x0.clone()
    u = model.cold_dual(x)
    u_last, du = None, []
    for k in range(n_steps):
        t = torch.full((x.shape[0], 1), k / n_steps, device=device)
        v, u_K = model(torch.cat([x, t], dim=-1), u_init=u, return_u=True)
        if record_du:
            if u_last is not None:
                num = (u_K - u_last).flatten(1).norm(dim=1)
                den = u_last.flatten(1).norm(dim=1).clamp(min=1e-12)
                du.append((num / den).mean().item())
            u_last = u_K
        x = x + v / n_steps
        u = u_K if warm else model.cold_dual(x)
    if was_training:
        model.train()
    return (x, du) if record_du else (x, None)


def make_sampler(device, warm, n=8, seed=0):
    """Callable(model) -> images (n, DIM), pour les PNG de progression de train_one.
    Meme bruit a chaque appel (seed fixe) -> les PNG successifs sont comparables."""
    def _s(model):
        g = torch.Generator().manual_seed(seed)
        x0 = torch.randn(n, DIM, generator=g).to(device)
        return sample(model, n, device, warm=warm, x0=x0)[0]
    return _s


def to_img(x):
    """(B, DIM) dans [-1,1] -> (B, 32, 32, 3) dans [0,1]."""
    return ((x.detach().cpu().view(-1, CHANNELS, IMG_SIZE, IMG_SIZE) * 0.5 + 0.5)
            .clamp(0, 1).permute(0, 2, 3, 1).numpy())


def grad_energy(x):
    """Nettete : variation totale moyenne. x : (B, DIM) ou (B, C, H, W)."""
    im = x.view(-1, CHANNELS, IMG_SIZE, IMG_SIZE)
    gx = (im[..., 1:, :] - im[..., :-1, :]).abs().mean()
    gy = (im[..., :, 1:] - im[..., :, :-1]).abs().mean()
    return float((gx + gy).item())


# ------------------------------------------------------------------------- modele
def build(K, ic, kernel, version, device):
    return ConvScCP_UNN(dim=DIM, K=K, internal_channel=ic, kernel_size=kernel,
                        in_channels=CHANNELS, img_size=IMG_SIZE, use_Unet="l1",
                        version=version, use_checkpoint=False, w_bias=True).to(device)


def run_name(K, ic, kernel, version, tag):
    # prefixe compatible build_from_name() de compute_fid_cifar10.py (re.match, le
    # suffixe _<tag> est ignore) -> les checkpoints restent lisibles par l'outillage RGB.
    return f"ConvScCP_UNN_rgb_k{kernel}_K{K}_ic{ic}_L1_{version}_{tag}"


def load_ema(path, model, device):
    """Recharge les poids EMA (ce sont eux qu'on echantillonne). `path` = dossier de run
    (on y cherche latest.pt) ou fichier .pt directement."""
    if os.path.isdir(path):
        path = os.path.join(path, "latest.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("ema_model", ck["state_dict"]), strict=True)
    model.eval()
    return ck.get("step", "?")


def main():
    p = argparse.ArgumentParser(description="Warm-start du dual ConvScCP sur AFHQ-chats 32x32.")
    p.add_argument("--runs", type=str, default="warmstart", choices=["both", "baseline", "warmstart"])
    p.add_argument("--baseline-ckpt", type=str, default="auto",
                   help="Run baseline DEJA ENTRAINE a reutiliser pour la comparaison "
                        "(dossier ou .pt) : evite de le reentrainer. 'auto' (defaut) = "
                        "results_afhq32/<nom deduit de --K/--ic/--kernel/--version>, donc "
                        "le defaut SUIT les flags. 'none' pour l'entrainer ici "
                        "(avec --runs both).")
    p.add_argument("--self-cond-rate", type=float, default=0.9,
                   help="Taux du run `warmstart` (le run `baseline` est toujours a 0.0).")
    p.add_argument("--steps", type=int, default=None,
                   help="Budget de steps. Par defaut : celui du baseline reutilise, pour "
                        "que la comparaison soit a budget egal (55000 pour le run k9, "
                        "50000 pour le k25). Sans baseline reutilise : 50000.")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--ic", type=int, default=256)
    p.add_argument("--kernel", type=int, default=9)
    p.add_argument("--version", type=str, default="LFO", choices=["LFO", "LNO"])
    p.add_argument("--coupling", type=str, default="ot", choices=["ot", "indep"])
    p.add_argument("--batch-size", type=int, default=RECIPE["batch_size"])
    p.add_argument("--cache", type=str, default="./data/afhq_cat32_train.pt")
    p.add_argument("--no-flip", action="store_true")
    p.add_argument("--sample-every", type=int, default=2500)
    p.add_argument("--save-every", type=int, default=2500)
    p.add_argument("--keep-every", type=int, default=10000)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--eval-only", action="store_true", help="Ne reentraine pas, compare.")
    p.add_argument("--n-grid", type=int, default=8, help="Images par ligne de comparaison.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="results_warmstart_afhq32")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    # ---- baseline reutilise : resolution 'auto' + verification de coherence ----
    expected = (f"ConvScCP_UNN_rgb_k{args.kernel}_K{args.K}_ic{args.ic}"
                f"_L1_{args.version}")                      # nom du run baseline attendu
    if args.baseline_ckpt == "auto":
        args.baseline_ckpt = os.path.join("results_afhq32", expected)
        if not os.path.exists(args.baseline_ckpt):
            raise SystemExit(
                f"[erreur] pas de baseline entraine pour cette config : {args.baseline_ckpt} "
                f"absent.\n  -> soit corriger --K/--ic/--kernel/--version pour viser un run "
                f"existant,\n  -> soit --baseline-ckpt <chemin>,\n"
                f"  -> soit --baseline-ckpt none --runs both pour l'entrainer ici.")
    if args.baseline_ckpt == "none":
        args.baseline_ckpt = ""
    if args.baseline_ckpt:
        _p = args.baseline_ckpt
        _p = os.path.join(_p, "latest.pt") if os.path.isdir(_p) else _p
        _ck = torch.load(_p, map_location="cpu", weights_only=False)
        _name = _ck.get("name", "")
        if _name != expected:
            # sinon load_state_dict crache un mur de size mismatch illisible
            raise SystemExit(
                f"[erreur] le baseline reutilise ne correspond pas a la config demandee.\n"
                f"  checkpoint : {_name}\n  demande    : {expected}  "
                f"(--K {args.K} --ic {args.ic} --kernel {args.kernel} --version {args.version})\n"
                f"  -> aligner les flags sur le checkpoint, ou pointer --baseline-ckpt "
                f"vers le bon run.")
        if args.steps is None:                     # budget egal a celui du baseline
            args.steps = _ck.get("step", 50000)
            print(f"[ws-afhq] --steps non precise -> {args.steps:,} (budget du baseline "
                  f"reutilise, comparaison a budget egal)", flush=True)
    if args.steps is None:
        args.steps = 50000

    to_train = ["baseline", "warmstart"] if args.runs == "both" else [args.runs]
    # Le baseline externe n'est pas reentraine, mais il entre dans la comparaison.
    tags = to_train if not args.baseline_ckpt else sorted(set(to_train) | {"baseline"},
                                                          key=lambda s: s != "baseline")
    rate_of = {"baseline": 0.0, "warmstart": args.self_cond_rate}
    warm_of = {"baseline": False, "warmstart": True}

    print(f"[ws-afhq] device={device} runs={tags} K={args.K} ic={args.ic} "
          f"kernel={args.kernel} {args.version} | steps={args.steps:,} "
          f"coupling={args.coupling} N_STEPS={N_STEPS}", flush=True)

    need_train = (not args.eval_only) and any(
        t in to_train and not (t == "baseline" and args.baseline_ckpt) for t in tags)
    train_loader = get_afhq_loader(args.cache, args.batch_size) if need_train else None

    infos = {}
    for tag in tags:
        name = run_name(args.K, args.ic, args.kernel, args.version, tag)
        run_dir = os.path.join(args.outdir, name)
        model = build(args.K, args.ic, args.kernel, args.version, device)
        external = args.baseline_ckpt if (tag == "baseline" and args.baseline_ckpt) else None
        if args.eval_only or external:
            src = external or run_dir
            step = load_ema(src, model, device)
            if external and step != args.steps:
                print(f"  [!] baseline externe a {step} steps, budget demande "
                      f"{args.steps} : comparaison a budget INEGAL.", flush=True)
            infos[tag] = dict(model=model, name=name, run_dir=src, step=step,
                              loss=float("nan"), secs=0.0)
            print(f"  [{tag}] recharge (EMA, step {step}) depuis {src}", flush=True)
            continue
        print(f"\n{'='*66}\n[{tag}] self_cond_rate={rate_of[tag]}  -> {run_dir}\n{'='*66}",
              flush=True)
        loss, n_params, secs = train_one(
            model, name, train_loader, device, run_dir=run_dir, total_steps=args.steps,
            coupling=args.coupling, flip=not args.no_flip,
            sample_every=args.sample_every, save_every=args.save_every,
            keep_every=args.keep_every, resume=not args.no_resume,
            self_cond_rate=rate_of[tag], n_steps=N_STEPS, seed=args.seed,
            # PNG de progression echantillonnes dans le regime du run (Euler, pas dopri5)
            sampler=make_sampler(device, warm=warm_of[tag], seed=args.seed),
        )
        load_ema(run_dir, model, device)          # comparer les poids EMA, pas les bruts
        infos[tag] = dict(model=model, name=name, run_dir=run_dir, step=args.steps,
                          loss=loss, secs=secs)
        gc.collect(); torch.cuda.empty_cache()

    # ------------------------------------------------------------------ comparaison
    # 4 combinaisons modele x echantillonneur, MEME bruit initial : les deux nominales
    # (baseline/froid, warmstart/chaud) et les deux croisees, qui separent "le warm-start
    # a l'echantillonnage nuit" de "l'entrainement self-conditionne nuit".
    g = torch.Generator().manual_seed(args.seed)
    x0 = torch.randn(args.n_grid, DIM, generator=g).to(device)
    rows, du_curves = [], {}
    for tag in tags:
        for warm in (False, True):
            label = f"{tag} / {'chaud' if warm else 'froid'}"
            t0 = time.perf_counter()
            xf, du = sample(infos[tag]["model"], args.n_grid, device, warm=warm,
                            x0=x0, record_du=True)
            rows.append((label, xf.cpu(), grad_energy(xf),
                         float(xf.std(dim=0).mean().item())))
            du_curves[label] = du
            print(f"  [eval] {label}: GE={rows[-1][2]:.4f} std={rows[-1][3]:.4f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    # reference reelle
    real = torch.load(args.cache)["data"][:args.n_grid].float().div_(127.5).sub_(1.0)
    ge_real = grad_energy(real.flatten(1))
    std_real = float(real.flatten(1).std(dim=0).mean().item())

    fig, axes = plt.subplots(len(rows) + 1, args.n_grid,
                             figsize=(1.15 * args.n_grid, 1.3 * (len(rows) + 1)),
                             squeeze=False)
    for r, (label, imgs, ge, std) in enumerate(rows):
        arr = to_img(imgs)
        for c in range(args.n_grid):
            axes[r, c].imshow(arr[c]); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(f"{label}\nGE {ge:.3f}", fontsize=7)
    arr = to_img(real.flatten(1))
    for c in range(args.n_grid):
        axes[-1, c].imshow(arr[c]); axes[-1, c].set_xticks([]); axes[-1, c].set_yticks([])
    axes[-1, 0].set_ylabel(f"AFHQ reel\nGE {ge_real:.3f}", fontsize=7)
    fig.suptitle(f"AFHQ-chats — warm-start du dual (K={args.K} ic={args.ic}, "
                 f"{N_STEPS} pas d'Euler, meme bruit)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(args.outdir, "comparison_samples.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, du in du_curves.items():
        ax.plot(range(1, len(du) + 1), du, label=label)
    ax.set_xlabel("pas d'ODE n"); ax.set_ylabel(r"$\|u_K^{(n)}-u_K^{(n-1)}\|/\|u_K^{(n-1)}\|$")
    ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Stabilite du dual le long de la trajectoire\n"
                 "(decroissance = le warm-start transporte de l'information)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "du_curve.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(f"# AFHQ-chats warm-start — K={args.K} ic={args.ic} kernel={args.kernel} "
                f"{args.version} coupling={args.coupling} steps={args.steps} "
                f"self_cond_rate={args.self_cond_rate} N_STEPS={N_STEPS}\n")
        f.write(f"# reference reelle : GE={ge_real:.4f} std={std_real:.4f}\n")
        f.write("configuration\tGE\tstd_gen\tdu_final\n")
        for label, _, ge, std in rows:
            f.write(f"{label}\t{ge:.4f}\t{std:.4f}\t{du_curves[label][-1]:.4f}\n")
        f.write("\n# entrainement\n")
        for tag in tags:
            f.write(f"{tag}\tloss={infos[tag]['loss']:.6f}\t"
                    f"temps={infos[tag]['secs']/3600:.2f}h\trun_dir={infos[tag]['run_dir']}\n")
    print(f"\n[ws-afhq] sorties dans {args.outdir}/ (comparison_samples.png, "
          f"du_curve.png, summary.txt)", flush=True)


if __name__ == "__main__":
    main()
