# -*- coding: utf-8 -*-
"""
run_convdfb_zeros.py
ConvDFB_UNN sur MNIST *zeros uniquement*, puis TRAJECTOIRES de l'ODE de Flow
Matching — en regard d'un ConvScCP_UNN entraine avec EXACTEMENT la meme recette.

Question posee : le deroule DFB (primal x = z - W^T u, dual forward-backward)
produit-il les memes trajectoires que le deroule ScCP (Chambolle-Pock accelere) ?

Ce qui est compare, aux memes x0 (meme seed) et aux memes temps t :
  - x_t          : l'etat reellement integre par l'ODE ;
  - x1_pred      : le chiffre que le reseau CROIT generer a l'instant t,
                   x_t + (1-t) v(x_t,t) — vue qui dit si le modele denoise
                   progressivement ou "decide" tard ;
  - x^(k)        : les iteres INTERNES du deroule (axe k), a chaque t ;
  - ||x_t||, ||v|| le long de la trajectoire ;
  - accord des CHAMPS : cos(v_DFB, v_ScCP) evalue aux MEMES points (x_t de la
    trajectoire DFB). Deux modeles peuvent avoir des trajectoires proches par
    hasard ; c'est cette metrique qui dit si le champ appris est le meme.

Les deux modeles utilisent w_bias=True (biais convolutif appris sur W). Sans lui
le deroule est impair en x_t -> une moitie des chiffres sort en couleurs
inversees, ce qui pollue completement la lecture des trajectoires.

Sorties (dans --results-dir, defaut results/convdfb_zeros/) :
  <ConvDFB_...>/ , <ConvScCP_...>/      run train_mnist : model.pt, loss.png/txt, epoch_*.png
  <...>/trajectory/                      figures par modele (memes fichiers que
                                         trajectory_convsccp.py)
  compare_xt.png, compare_x1pred.png     DFB vs ScCP cote a cote, meme x0
  compare_norms.png                      ||x_t|| et ||v|| des deux modeles
  compare_metrics.txt                    distances trajectoires + cos des champs

Usage
-----
    source ~/.venvs/unn/bin/activate
    CUDA_VISIBLE_DEVICES=1 python run_convdfb_zeros.py --epochs 100 \
        > claude.log 2>&1 &        # puis  tail -f claude.log

    # ne re-tracer que les figures a partir des model.pt deja la
    python run_convdfb_zeros.py --skip-train

    # ConvDFB seul (pas de ScCP de reference entraine ici)
    python run_convdfb_zeros.py --models dfb

    # reutiliser un checkpoint ScCP existant au lieu d'en entrainer un
    python run_convdfb_zeros.py --models dfb \
        --sccp-ckpt results/temp-4/ConvScCP_UNN_L1_LNO/model.pt
"""
import argparse
import os
import time

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.architectures import ConvDFB_UNN, ConvScCP_UNN
from train import train_mnist
import trajectory_convsccp as T


# --------------------------------------------------------------------------- #
#  Donnees : MNIST restreint a un chiffre (0 par defaut)
# --------------------------------------------------------------------------- #
def get_loader(batch_size, digit):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    ds = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                    transform=transform)
    idx = torch.where(ds.targets == digit)[0]
    print(f"MNIST digit={digit} : {len(idx)} images", flush=True)
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=True,
                      num_workers=2, pin_memory=True)


# --------------------------------------------------------------------------- #
#  Figures de comparaison (les figures par modele viennent de trajectory_convsccp)
# --------------------------------------------------------------------------- #
def compare_grid(runs, ts, path, key, title):
    """Une ligne par (echantillon, modele) : meme x0, meme t -> lecture directe.

    runs : liste de (nom, dict avec 'xt'/'x1pred' de forme (S, n, C, H, W)).
    """
    S = len(ts)
    n = runs[0][1][key].shape[1]
    nm = len(runs)
    fig, axes = plt.subplots(n * nm, S, figsize=(1.1 * S, 1.15 * n * nm), squeeze=False)
    for i in range(n):
        for m, (name, r) in enumerate(runs):
            for j in range(S):
                ax = axes[i * nm + m, j]
                ax.imshow(r[key][j, i, 0].clamp(-1, 1), cmap="gray", vmin=-1, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                if i * nm + m == 0:
                    ax.set_title(f"t={ts[j]:.2f}", fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"#{i} {name}", fontsize=7)
    fig.suptitle(title, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  -> {path}", flush=True)


def compare_norms(runs, ts, path, title):
    """||x_t|| et ||v(x_t,t)|| des deux modeles sur la meme figure."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for name, r in runs:
        axs[0].plot(ts, r["xt"].flatten(2).norm(dim=-1).mean(1), "o-", label=name)
        axs[1].plot(ts, r["v"].flatten(2).norm(dim=-1).mean(1), "s-", label=name)
    for ax, lab in [(axs[0], "||x_t||"), (axs[1], "||v(x_t,t)||")]:
        ax.set_xlabel("t"); ax.set_ylabel(f"{lab} (moyenne batch)")
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle(title, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  -> {path}", flush=True)


@torch.no_grad()
def field_agreement(models, xt_ref, ts, device):
    """Accord des champs de vitesse aux MEMES points (x_t de la trajectoire de
    reference, ici celle du DFB). Renvoie, par t, le cosinus moyen entre
    v_A(x_t,t) et v_B(x_t,t) et l'erreur relative ||v_A - v_B|| / ||v_B||.

    Sans ca on ne peut pas conclure : deux champs differents peuvent donner des
    trajectoires visuellement proches, et inversement.
    """
    (nameA, mA), (nameB, mB) = models
    cos, rel = [], []
    for j, t in enumerate(ts):
        x = xt_ref[j].flatten(1).to(device)
        xt_t = torch.cat([x, t.to(device).expand(x.shape[0], 1)], dim=-1)
        vA, vB = mA(xt_t), mB(xt_t)
        cos.append(torch.nn.functional.cosine_similarity(vA, vB, dim=-1).mean().item())
        rel.append(((vA - vB).norm(dim=-1) / vB.norm(dim=-1).clamp(min=1e-8)).mean().item())
    return cos, rel


# --------------------------------------------------------------------------- #
#  Entrainement + analyse d'un modele
# --------------------------------------------------------------------------- #
def build(kind, args, device):
    common = dict(dim=784, K=args.K, internal_channel=args.ic, use_Unet="l1",
                  version=args.version, use_checkpoint=True, w_bias=True)
    if kind == "dfb":
        return ConvDFB_UNN(**common).to(device)
    return ConvScCP_UNN(**common).to(device)


def main():
    p = argparse.ArgumentParser(description="ConvDFB sur les zeros de MNIST + trajectoires (vs ScCP).")
    p.add_argument("--models", type=str, default="dfb,sccp",
                   help="modeles a traiter, parmi dfb,sccp (defaut : les deux, meme recette)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-2, help="1e-2 = recette des runs ScCP de reference")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--ic", type=int, default=64, help="canaux du dual (internal_channel)")
    p.add_argument("--version", type=str, default="LNO", choices=["LNO", "LFO"])
    p.add_argument("--digit", type=int, default=0)
    p.add_argument("--coupling", type=str, default="indep", choices=["indep", "ot"])
    p.add_argument("--x1_weight", type=str, default="invsq", choices=["invsq", "uniform", "minsnr"])
    p.add_argument("--results-dir", type=str, default="results/convdfb_zeros")
    p.add_argument("--skip-train", action="store_true",
                   help="ne pas (re)entrainer : repartir des model.pt deja presents")
    p.add_argument("--sccp-ckpt", type=str, default="",
                   help="checkpoint ScCP existant a utiliser comme reference (au lieu d'en entrainer un)")
    # ---- trajectoires ----
    p.add_argument("--n", type=int, default=6, help="nombre de trajectoires (echantillons)")
    p.add_argument("--steps", type=int, default=10, help="t = 0, 1/steps, ..., 1")
    p.add_argument("--solver", type=str, default="dopri5", choices=["dopri5", "euler", "rk4"])
    p.add_argument("--seed", type=int, default=0, help="meme seed pour les deux modeles -> memes x0")
    p.add_argument("--n-iter-samples", type=int, default=2,
                   help="nb d'echantillons pour lesquels tracer la grille (t, k) des iteres internes")
    p.add_argument("--n-layers", type=int, default=None,
                   help="nb de colonnes k affichees dans la grille (t,k) ; None = toutes")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kinds = [k.strip() for k in args.models.split(",") if k.strip()]
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Device: {device} | modeles={kinds} K={args.K} ic={args.ic} {args.version} "
          f"digit={args.digit} epochs={args.epochs} lr={args.lr} coupling={args.coupling}",
          flush=True)

    names = {k: f"Conv{'DFB' if k == 'dfb' else 'ScCP'}_K{args.K}_ic{args.ic}_L1_{args.version}"
             for k in kinds}
    ckpts = {k: os.path.join(args.results_dir, names[k], "model.pt") for k in kinds}

    # ---------------- entrainement ----------------
    if not args.skip_train:
        loader = get_loader(args.batch_size, args.digit)
        # ~47 batches/epoch sur les 5923 zeros ; ~0.13 s/batch pour K=20 ic=64 sur A100
        # -> ~6 s/epoch + une generation dopri5 toutes les 2 epoques apres l'epoque 40.
        print(f"Estimation : ~10-20 min par modele pour {args.epochs} epoques "
              f"(soit ~{20 * len(kinds)} min au total, majore).", flush=True)
        for k in kinds:
            t0 = time.perf_counter()
            print(f"\n=== entrainement {names[k]} ===", flush=True)
            model = build(k, args, device)
            train_mnist(model=model, train_loader=loader, device=device,
                        results_dir=args.results_dir, model_name=names[k],
                        nb_epochs=args.epochs, lr=args.lr, coupling=args.coupling,
                        x1_weight=args.x1_weight, save_model=True)
            print(f"=== {names[k]} entraine en {(time.perf_counter()-t0)/60:.1f} min "
                  f"-> {ckpts[k]}", flush=True)
            del model
            torch.cuda.empty_cache()

    if args.sccp_ckpt:
        kinds = [k for k in kinds if k != "sccp"] + ["sccp"]
        names["sccp"] = os.path.basename(os.path.dirname(args.sccp_ckpt))
        ckpts["sccp"] = args.sccp_ckpt

    # ---------------- trajectoires, modele par modele ----------------
    runs, loaded = [], {}
    for k in kinds:
        print(f"\n=== trajectoires {names[k]} ===", flush=True)
        model, cfg = T.build_model(ckpts[k], device)
        algo = cfg.get("algo", "?")
        ts, xt, v, x1p = T.rollout(model, args.n, cfg, device, args.steps,
                                   args.solver, args.seed)
        outdir = os.path.join(os.path.dirname(ckpts[k]), "trajectory")
        os.makedirs(outdir, exist_ok=True)
        tag = f"{names[k]} — {algo} K={cfg['K']} ic={cfg['internal_channel']} {cfg['version']}"

        T.grid(xt, ts, os.path.join(outdir, "trajectory_xt.png"), f"x_t le long de l'ODE — {tag}")
        T.grid(x1p, ts, os.path.join(outdir, "trajectory_x1pred.png"),
               f"x1_pred = x_t + (1-t)·v — {tag}")
        T.grid_both(xt, x1p, ts, os.path.join(outdir, "trajectory_both.png"),
                    f"x_t (haut) vs x1_pred (bas) — {tag}")
        T.norms_plot(xt, v, ts, os.path.join(outdir, "velocity_norm.png"), f"normes — {tag}")

        # iteres internes du deroule (axe k) : c'est la que DFB et ScCP peuvent
        # differer le plus, meme si la trajectoire externe se ressemble.
        it = T.iterates_at_times(model, xt, ts, device)
        layer_idx = T.select_layer_indices(it.shape[1], args.n_layers)
        for i in range(min(args.n_iter_samples, args.n)):
            T.grid_iterates(it, ts, i, os.path.join(outdir, f"iterates_sample{i}.png"),
                            f"x^(k) internes du deroule {algo} — echantillon #{i} — {tag}",
                            layer_idx=layer_idx)
        T.iterates_conv_plot(it, ts, os.path.join(outdir, "iterates_convergence.png"),
                             f"convergence du deroule — {tag}", algo=algo)
        T.iterates_amplitude_plot(it, ts, os.path.join(outdir, "iterates_amplitude.png"),
                                  f"amplitude des iteres — {tag}", algo=algo)
        torch.save(dict(ts=ts, xt=xt, x1pred=x1p, v=v, iterates=it, cfg=cfg, ckpt=ckpts[k]),
                   os.path.join(outdir, "trajectory.pt"))
        print(f"  -> {outdir}/trajectory.pt", flush=True)

        runs.append((algo, dict(xt=xt, x1pred=x1p, v=v)))
        loaded[algo] = model

    # ---------------- comparaison ----------------
    if len(runs) < 2:
        print("\nUn seul modele -> pas de comparaison croisee.", flush=True)
        return

    print("\n=== comparaison DFB vs ScCP ===", flush=True)
    sub = (f"memes x0 (seed={args.seed}), K={args.K} ic={args.ic} {args.version}, "
           f"digit={args.digit}, {args.epochs} ep, {args.solver}")
    compare_grid(runs, ts, os.path.join(args.results_dir, "compare_xt.png"), "xt",
                 f"x_t le long de l'ODE\n{sub}")
    compare_grid(runs, ts, os.path.join(args.results_dir, "compare_x1pred.png"), "x1pred",
                 f"x1_pred = x_t + (1-t)·v\n{sub}")
    compare_norms(runs, ts, os.path.join(args.results_dir, "compare_norms.png"),
                  f"normes le long de la trajectoire\n{sub}")

    # distance entre trajectoires (memes x0) + accord des champs aux memes points
    (nA, rA), (nB, rB) = runs[0], runs[1]
    dtraj = ((rA["xt"] - rB["xt"]).flatten(2).norm(dim=-1)
             / rB["xt"].flatten(2).norm(dim=-1).clamp(min=1e-8)).mean(1)
    models = [(nA, loaded[nA]), (nB, loaded[nB])]
    cos, rel = field_agreement(models, rA["xt"], ts, device)

    lines = [f"# {nA} vs {nB} — {sub}",
             "# d_traj = ||x_t^A - x_t^B|| / ||x_t^B||  (memes x0)",
             f"# cos, rel_err = accord des champs v_A, v_B evalues aux MEMES points "
             f"(x_t de {nA})",
             "t\td_traj\tcos(vA,vB)\trel_err(vA,vB)"]
    for j, t in enumerate(ts):
        lines.append(f"{t:.2f}\t{dtraj[j]:.4f}\t{cos[j]:.4f}\t{rel[j]:.4f}")
    txt = os.path.join(args.results_dir, "compare_metrics.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(ts, dtraj, "o-"); axs[0].set_ylabel("||x_t^A - x_t^B|| / ||x_t^B||")
    axs[1].plot(ts, cos, "o-", label="cos(v_A, v_B)")
    axs[1].plot(ts, rel, "s-", label="||v_A - v_B|| / ||v_B||")
    axs[1].axhline(1.0, color="k", ls="--", lw=1); axs[1].legend()
    axs[1].set_ylabel("accord des champs (memes points)")
    for ax in axs:
        ax.set_xlabel("t"); ax.grid(alpha=0.3)
    fig.suptitle(f"{nA} vs {nB}\n{sub}", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(args.results_dir, "compare_metrics.png"), dpi=130)
    plt.close(fig)
    print(f"  -> {txt}\n  -> {os.path.join(args.results_dir, 'compare_metrics.png')}", flush=True)
    print(f"\nTermine. Tout est dans {args.results_dir}/", flush=True)


if __name__ == "__main__":
    main()
