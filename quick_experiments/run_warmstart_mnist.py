# -*- coding: utf-8 -*-
"""
run_warmstart_mnist.py
Warm-start du dual du ConvScCP le long de l'ODE de generation, avec self-conditioning
a l'entrainement (mecanisme de Jabri, Fleet & Chen, "Scalable Adaptive Computation for
Iterative Generation", ICML 2023, arXiv:2212.11972, §2.3).

Principe
--------
Le dual u du deroule Chambolle-Pock est actuellement remis a ZERO a chaque evaluation
du champ de vitesse. Deux pas d'ODE consecutifs resolvent pourtant des problemes
lentement variables : on transporte donc u_K du pas n-1 vers l'initialisation du pas n.
C'est le warm-start standard d'un solveur iteratif -- CP converge depuis n'importe
quelle initialisation, AUCUNE garantie de convergence n'est affectee.

Le report est l'IDENTITE : u_init = u_prev. Pas de porte apprise, pas de 1x1, pas de
LayerNorm, pas de changement de W_k/V_k/prox/pas/loss/cible/injection de z, pas de
solveur adaptatif, pas de BPTT.

Entrainement (self-conditioning) : avec probabilite SELF_COND_RATE, une passe FROIDE
sans gradient sur x_{t-dt} de LA MEME paire (x0, x1) fournit le u_prev de la passe
d'entrainement. dt = 1/N_STEPS, le meme qu'a l'inference.

Deux runs, identiques en tout sauf le flag :
    baseline   self_cond_rate=0.0, echantillonnage a dual froid  (= modele actuel)
    warmstart  self_cond_rate=0.9, echantillonnage a dual chaine

Usage
-----
    source ~/.venvs/unn/bin/activate
    CUDA_VISIBLE_DEVICES=1 python run_warmstart_mnist.py --epochs 200 2>&1 | tee -a claude.log

    # un seul run / reprise d'evaluation sans reentrainer
    python run_warmstart_mnist.py --runs baseline
    python run_warmstart_mnist.py --eval-only

Sorties (--outdir, defaut results/warmstart_mnist/) :
    <tag>/model.pt        state_dict + config + historique de loss
    <tag>/samples.png     grille d'echantillons
    loss_curves.png       les deux courbes d'entrainement
    du_curve.png          ||u_K^(n) - u_K^(n-1)|| / ||u_K^(n-1)|| le long de la trajectoire
    results_table.md      tableau des metriques
    summary.txt           idem en texte brut + configuration complete
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher)

from models.architectures import ConvScCP_UNN

# Nombre de pas d'Euler, PARTAGE entre entrainement et echantillonnage : l'entrainement
# doit voir exactement le dt = 1/N_STEPS que l'inference utilisera.
N_STEPS = 100


# --------------------------------------------------------------------------- donnees
def get_loader(digit, batch_size, seed):
    """MNIST filtre sur un chiffre, meme normalisation que le reste du pipeline
    ([-1,1], pas d'augmentation). Le generateur fixe l'ordre des batches -> deux runs
    de meme graine voient exactement la meme sequence de donnees."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                         transform=transform)
    if digit >= 0:
        dataset_f = Subset(dataset, torch.where(dataset.targets == digit)[0])
    else:
        dataset_f = dataset
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset_f, batch_size=batch_size, shuffle=True, num_workers=2,
                        pin_memory=True, generator=g, drop_last=False)
    return loader, dataset, dataset_f


# ------------------------------------------------------------------------ entrainement
def train_run(tag, self_cond_rate, args, device, log_every=10):
    """Entraine un ConvScCP_UNN. self_cond_rate=0.0 -> le dual froid est toujours passe
    et le forward est celui du modele actuel (run `baseline`)."""
    torch.manual_seed(args.seed)                       # init des poids
    model = ConvScCP_UNN(dim=784, K=args.K, internal_channel=args.ic, use_Unet="l1",
                         version=args.version, kernel_size=args.kernel,
                         use_checkpoint=False, w_bias=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    loader, _, dataset_f = get_loader(args.digit, args.batch, args.seed)
    FM = (ExactOptimalTransportConditionalFlowMatcher(sigma=0.0) if args.coupling == "ot"
          else ConditionalFlowMatcher(sigma=0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Tirage de Bernoulli du self-conditioning sur un generateur SEPARE : le flux
    # aleatoire principal (bruit x0, temps t, ordre des batches) reste bit-identique
    # entre baseline et warmstart, seule la piece qui differe est isolee.
    coin = torch.Generator().manual_seed(args.seed + 1234)
    dt = 1.0 / N_STEPS

    print(f"\n{'='*66}\n[{tag}] self_cond_rate={self_cond_rate}  K={args.K} ic={args.ic} "
          f"kernel={args.kernel} {args.version}\n  {n_params:,} params | {len(dataset_f)} images | "
          f"{len(loader)} steps/epoch x {args.epochs} epochs | coupling={args.coupling} "
          f"| lr={args.lr}\n{'='*66}", flush=True)

    loss_history, radii_history, n_warm = [], [], 0
    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total, t0 = 0.0, time.perf_counter()
        for x1_img, _ in loader:
            x1 = x1_img.to(device, non_blocking=True).view(x1_img.shape[0], -1)
            B = x1.shape[0]
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            # Reapparie (x0, x1) APRES le plan OT : sample_location_and_conditional_flow
            # permute les paires en interne et ne renvoie pas la permutation (c'est le
            # bug de desappariement historique). Avec sigma=0 l'inversion est exacte.
            t_col = t.view(-1, 1)
            x1 = xt + ut * (1 - t_col)
            x0 = xt - ut * t_col

            # ---- passe d'estimation (self-conditioning) : MEME paire (x0, x1) ----
            u_prev = model.cold_dual(xt)
            if self_cond_rate > 0 and torch.rand((), generator=coin).item() < self_cond_rate:
                t_prev = (t - dt).clamp_min(0.0).view(-1, 1)
                xt_prev = (1 - t_prev) * x0 + t_prev * x1         # etat ODE precedent, exact
                with torch.no_grad():
                    _, u_prev = model(torch.cat([xt_prev, t_prev], dim=-1),
                                      u_init=model.cold_dual(xt_prev), return_u=True)
                u_prev = u_prev.detach()
                n_warm += 1

            # ---- passe d'entrainement (inchangee : x-pred + ponderation espace-v) ----
            out, _ = model(torch.cat([xt, t_col], dim=-1), u_init=u_prev, return_u=True)
            w = 1.0 / torch.clamp((1 - t_col) ** 2, min=0.05 ** 2)
            loss = torch.mean(w * (out - x1) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()

        loss_history.append(total / len(loader))
        if (epoch + 1) % log_every == 0 or epoch == 0 or epoch == args.epochs - 1:
            done = epoch + 1
            elapsed = time.perf_counter() - t_start
            eta = elapsed / done * (args.epochs - done)
            rs = prox_radii(model)
            radii_history.append((done, rs))
            n_dead = sum(1 for r in rs if r < 1e-6)
            print(f"  [{tag}] epoch {done}/{args.epochs}  loss={loss_history[-1]:.4f}  "
                  f"r_min={min(rs):.2e} couches_mortes={n_dead}/{len(rs)}  "
                  f"({time.perf_counter()-t0:.1f}s/ep, ecoule {elapsed/60:.1f}min, "
                  f"ETA {eta/60:.1f}min)", flush=True)

    run_dir = os.path.join(args.outdir, tag)
    os.makedirs(run_dir, exist_ok=True)
    cfg = dict(K=args.K, internal_channel=args.ic, kernel_size=args.kernel,
               version=args.version, use_Unet="l1", w_bias=True, dim=784,
               self_cond_rate=self_cond_rate, epochs=args.epochs, lr=args.lr,
               batch=args.batch, digit=args.digit, coupling=args.coupling,
               seed=args.seed, n_steps=N_STEPS, n_params=n_params)
    torch.save(dict(state_dict=model.state_dict(), cfg=cfg, loss_history=loss_history,
                    radii_history=radii_history),
               os.path.join(run_dir, "model.pt"))
    print(f"  [{tag}] fini en {(time.perf_counter()-t_start)/60:.1f}min "
          f"(passes chaudes : {n_warm}/{args.epochs*len(loader)}) -> {run_dir}/model.pt",
          flush=True)
    return model, loss_history, cfg


def load_run(tag, args, device):
    """Recharge un run deja entraine (--eval-only)."""
    path = os.path.join(args.outdir, tag, "model.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = ConvScCP_UNN(dim=cfg["dim"], K=cfg["K"], internal_channel=cfg["internal_channel"],
                         use_Unet=cfg["use_Unet"], version=cfg["version"],
                         kernel_size=cfg["kernel_size"], w_bias=cfg["w_bias"],
                         use_checkpoint=False).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    print(f"  [{tag}] recharge depuis {path}", flush=True)
    return model, ck["loss_history"], cfg


# ----------------------------------------------------------------- echantillonnage
@torch.no_grad()
def sample(model, shape, device, warm, n_steps=N_STEPS, x0=None, record_du=False):
    """Euler a pas fixe, t: 0 -> 1. `warm=True` : u_K est reporte d'un pas au suivant
    (u n'est JAMAIS remis a zero dans la boucle). `warm=False` : dual froid a chaque pas
    = echantillonneur actuel.

    record_du : enregistre ||u_K^(n) - u_K^(n-1)|| / ||u_K^(n-1)|| (moyenne sur le batch).
    La suite u_K existe dans les deux modes, la courbe est donc comparable warm/froid.
    """
    x = torch.randn(shape, device=device) if x0 is None else x0.clone()
    u = model.cold_dual(x)
    u_last, du = None, []
    for n in range(n_steps):
        t = torch.full((shape[0], 1), n / n_steps, device=device)
        v, u_K = model(torch.cat([x, t], dim=-1), u_init=u, return_u=True)
        if record_du:
            if u_last is not None:
                num = (u_K - u_last).flatten(1).norm(dim=1)
                den = u_last.flatten(1).norm(dim=1).clamp(min=1e-12)
                du.append((num / den).mean().item())
            u_last = u_K
        x = x + v / n_steps
        u = u_K if warm else model.cold_dual(x)
    return (x, du) if record_du else (x, None)


@torch.no_grad()
def generate_many(model, n, device, warm, batch, seed, n_steps=N_STEPS):
    """n echantillons par lots ; meme graine -> memes x0 pour toutes les configs."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    outs, done = [], 0
    while done < n:
        b = min(batch, n - done)
        x0 = torch.randn(b, 784, generator=g).to(device)
        xf, _ = sample(model, (b, 784), device, warm=warm, n_steps=n_steps, x0=x0)
        outs.append(xf.cpu())
        done += b
        print(f"      {done}/{n}", flush=True)
    return torch.cat(outs, 0)[:n]


# ------------------------------------------------------------------------- metriques
@torch.no_grad()
def prox_radii(model, t_val=0.5):
    """Rayon r(t) du prox l1 de chaque couche. Sur le ckpt MNIST du rapport, plusieurs
    couches ont appris r ~ 1e-23 : le clamp ANNIHILE le dual (u^(3)| ~ 1e-20), ce qui
    tue tout warm-start possible. On surveille donc l'apparition de cette pathologie
    pendant l'entrainement -- si elle reapparait, l'experience ne peut rien mesurer.
    Voir diag_u0_forgetting.py."""
    t = torch.full((1, 1), t_val, device=next(model.parameters()).device)
    return [float(F.softplus(l.prox.time_scaling(t)).item()) for l in model.layers]


def grad_energy(imgs):
    """Nettete : variation totale moyenne (meme definition que grille_convsccp.py)."""
    gx = (imgs[..., 1:, :] - imgs[..., :-1, :]).abs().mean()
    gy = (imgs[..., :, 1:] - imgs[..., :, :-1]).abs().mean()
    return float((gx + gy).item())


def save_grid(imgs, path, title, n=8):
    fig, axes = plt.subplots(1, n, figsize=(1.2 * n, 1.7), squeeze=False)
    for i in range(n):
        axes[0, i].imshow(imgs[i, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
    fig.suptitle(title, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Warm-start du dual ConvScCP : baseline vs warmstart.")
    p.add_argument("--runs", type=str, default="both", choices=["both", "baseline", "warmstart"])
    p.add_argument("--self-cond-rate", type=float, default=0.9,
                   help="Taux de self-conditioning du run `warmstart` (le run `baseline` "
                        "est toujours a 0.0). Defaut 0.9 (RIN).")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--ic", type=int, default=64, help="internal_channel (dimension duale).")
    p.add_argument("--kernel", type=int, default=9)
    p.add_argument("--version", type=str, default="LNO", choices=["LNO", "LFO"])
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--digit", type=int, default=0, help="-1 = tous les chiffres.")
    p.add_argument("--coupling", type=str, default="ot", choices=["ot", "indep"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-eval", type=int, default=2000, help="Echantillons par configuration.")
    p.add_argument("--eval-batch", type=int, default=500)
    p.add_argument("--eval-only", action="store_true", help="Recharge les modeles, n'entraine pas.")
    p.add_argument("--outdir", type=str, default="results/warmstart_mnist")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)
    tags = (["baseline", "warmstart"] if args.runs == "both" else [args.runs])
    rate_of = {"baseline": 0.0, "warmstart": args.self_cond_rate}
    print(f"[warmstart] device={device} runs={tags} N_STEPS={N_STEPS} "
          f"outdir={args.outdir}", flush=True)

    models, losses, cfgs = {}, {}, {}
    for tag in tags:
        if args.eval_only:
            models[tag], losses[tag], cfgs[tag] = load_run(tag, args, device)
        else:
            models[tag], losses[tag], cfgs[tag] = train_run(tag, rate_of[tag], args, device)

    # ------------------------------------------------------------ reference reelle
    from mnist_metrics import train_or_load_classifier, mini_fid

    _, full_dataset, dataset_f = get_loader(args.digit, args.batch, args.seed)
    clf = train_or_load_classifier(
        DataLoader(full_dataset, batch_size=256, shuffle=True, num_workers=2), device)
    n_ref = min(args.n_eval, len(dataset_f))
    # lot de reference SEEDE : sans generateur, chaque depouillement tire un lot reel
    # different et le mini-FID bouge de ~0.5 point d'une execution a l'autre.
    g_ref = torch.Generator().manual_seed(args.seed)
    real = next(iter(DataLoader(dataset_f, batch_size=n_ref, shuffle=True,
                                generator=g_ref)))[0][:n_ref]
    ge_real = grad_energy(real)
    std_real = float(real.view(n_ref, -1).std(dim=0).mean().item())
    print(f"\n[eval] reference : {n_ref} images reelles (digit="
          f"{'tous' if args.digit < 0 else args.digit})  GE_reel={ge_real:.4f}  "
          f"std_reel={std_real:.4f}", flush=True)

    # ------------------------------------------------------------------ evaluation
    # Les 2 configurations du protocole (baseline/froid, warmstart/chaud) + les 2
    # CROISEES : elles separent "le warm-start a l'echantillonnage nuit" de
    # "l'entrainement self-conditionne nuit" (cf. le mode d'echec attendu, §9).
    combos = [(tag, warm) for tag in tags for warm in (False, True)]
    rows, du_curves = [], {}
    for tag, warm in combos:
        label = f"{tag} / {'chaud' if warm else 'froid'}"
        print(f"  [eval] {label} ...", flush=True)
        t0 = time.perf_counter()
        samples = generate_many(models[tag], args.n_eval, device, warm,
                                args.eval_batch, args.seed)
        imgs = samples.view(-1, 1, 28, 28).clamp(-1, 1)
        row = dict(tag=tag, warm=warm, label=label,
                   fid=mini_fid(clf, imgs, real, device),
                   ge=grad_energy(imgs),
                   std=float(samples.std(dim=0).mean().item()),
                   loss=losses[tag][-1] if losses[tag] else float("nan"))
        rows.append(row)
        print(f"    mini-FID={row['fid']:.2f}  GE={row['ge']:.4f} (reel {ge_real:.4f})  "
              f"std={row['std']:.4f}  ({time.perf_counter()-t0:.0f}s)", flush=True)

        # diagnostic ||delta u|| sur un petit lot, meme bruit pour toutes les configs
        torch.manual_seed(args.seed)
        x0_diag = torch.randn(64, 784, device=device)
        _, du = sample(models[tag], (64, 784), device, warm=warm, x0=x0_diag, record_du=True)
        du_curves[label] = du

        if warm == (tag == "warmstart"):        # config nominale du run -> grille sauvee
            save_grid(imgs[:8], os.path.join(args.outdir, tag, "samples.png"),
                      f"{label} — mini-FID={row['fid']:.1f} GE={row['ge']:.3f}")

    # --------------------------------------------------------------------- figures
    if any(losses.values()):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for tag in tags:
            if losses[tag]:
                ax.plot(range(1, len(losses[tag]) + 1), losses[tag],
                        label=f"{tag} (rate={rate_of[tag]})")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss x1-pred ponderee"); ax.set_yscale("log")
        ax.grid(alpha=0.3); ax.legend()
        ax.set_title(f"Entrainement — ConvScCP K={args.K} ic={args.ic}, digit={args.digit}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "loss_curves.png"), dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, du in du_curves.items():
        ax.plot(range(1, len(du) + 1), du, label=label)
    ax.set_xlabel("pas d'ODE n"); ax.set_ylabel(r"$\|u_K^{(n)}-u_K^{(n-1)}\| / \|u_K^{(n-1)}\|$")
    ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Stabilite du dual le long de la trajectoire\n"
                 "(decroissance = le warm-start transporte de l'information)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "du_curve.png"), dpi=130)
    plt.close(fig)

    # --------------------------------------------------------------------- tableau
    hdr = (f"| configuration | mini-FID | GE (reel {ge_real:.3f}) | std_gen "
           f"(reel {std_real:.3f}) | loss finale |\n|---|---|---|---|---|\n")
    body = "".join(f"| {r['label']} | {r['fid']:.2f} | {r['ge']:.4f} | {r['std']:.4f} "
                   f"| {r['loss']:.4f} |\n" for r in rows)
    table = hdr + body
    with open(os.path.join(args.outdir, "results_table.md"), "w") as f:
        f.write(f"# Warm-start du dual ConvScCP — digit {args.digit}, "
                f"K={args.K} ic={args.ic}, {args.epochs} epochs, N_STEPS={N_STEPS}\n\n")
        f.write(table)
        f.write("\n`froid` = u remis a zero a chaque pas (echantillonneur actuel) ; "
                "`chaud` = u_K reporte d'un pas au suivant.\n")
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(json.dumps(dict(args=vars(args), n_steps=N_STEPS, ge_real=ge_real,
                                std_real=std_real, cfgs=cfgs,
                                rows=[{k: v for k, v in r.items()} for r in rows],
                                du_curves=du_curves), indent=2, default=str))
    print("\n" + table, flush=True)
    print(f"[warmstart] sorties dans {args.outdir}/ (results_table.md, du_curve.png, "
          f"loss_curves.png, <tag>/samples.png, <tag>/model.pt)", flush=True)


if __name__ == "__main__":
    main()
