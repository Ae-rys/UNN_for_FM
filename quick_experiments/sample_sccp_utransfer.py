# -*- coding: utf-8 -*-
"""
sample_sccp_utransfer.py
Generation MNIST avec un ConvScCP_UNN DEJA ENTRAINE, mais en TRANSFERANT L'ESPACE
LATENT (la variable duale u du deroule Chambolle-Pock) d'un pas d'Euler au suivant.

Idee
----
A chaque evaluation du champ de vitesse, ConvScCP_UNN deroule K iterations CP en
repartant TOUJOURS de u^(0) = 0 (warm-start froid) : l'etat latent construit au pas
de temps precedent est jete. Ici on le garde :

    pas i :   v_i, u_i = model(x_t_i, t_i, u_init = gamma * u_{i-1})
              x_{i+1}  = x_t_i + dt * v_i

gamma = 0.0 -> u_init = 0 : c'est EXACTEMENT l'Euler standard (baseline de controle,
                            meme code, meme bruit, meme grille de temps).
gamma = 1.0 -> transfert complet du dual d'un pas au suivant.
gamma in (0,1) -> transfert amorti (le modele a ete entraine avec u^(0)=0, un
                  transfert partiel reste plus proche de sa distribution d'entrainement).

C'est pour cela que la boucle d'Euler est reecrite a la main ici : NeuralODE /
torchdyn appelle le modele comme une fonction sans etat, il n'y a aucun moyen d'y
faire circuler u. Le modele, lui, expose maintenant `u_init` / `return_u`
(cf. ConvScCP_UNN.forward dans models/architectures.py).

ATTENTION methodo : le modele n'a JAMAIS vu u^(0) != 0 pendant l'entrainement.
Une degradation a gamma=1 n'est donc pas une refutation de l'idee "transferer le
latent aide" -- c'est d'abord un ecart a la distribution d'entrainement. Le
diagnostic delta_v (ci-dessous) mesure exactement l'ampleur de cet ecart.

Sorties (dans --outdir, defaut results/sccp_utransfer/)
-------------------------------------------------------
  samples.png       grille : une LIGNE par gamma, meme bruit initial -> l'effet du
                    transfert se lit colonne par colonne.
  diagnostics.png   par pas de temps : ||u||, cos(u_i, u_{i-1}) (stabilite du latent
                    d'un pas a l'autre), et delta_v = ||v_warm - v_froid|| / ||v_froid||
                    (a quel point le transfert change reellement le champ).
  metrics.png       mini-FID / nettete GE / diversite std, en fonction de gamma.
  summary.txt       toutes les valeurs chiffrees.

Usage
-----
    source ~/.venvs/unn/bin/activate
    CUDA_VISIBLE_DEVICES=1 python sample_sccp_utransfer.py \
        --ckpt results/temp-13-200-epochs/ConvScCP_UNN_L1_LFO/model.pt \
        --digit 0 --gammas 0,0.25,0.5,1.0 --steps 100 2>&1 | tee -a claude.log

    # rapide, sans metriques quantitatives (juste la grille + diagnostics)
    python sample_sccp_utransfer.py --ckpt <path> --no-metrics

Checkpoints ConvScCP MNIST disponibles dans le repo (--digit doit correspondre au
chiffre sur lequel le modele a ete entraine, il ne sert qu'au jeu d'images REELLES
de reference du mini-FID) :
    results/temp-13-200-epochs/ConvScCP_UNN_L1_LFO/model.pt   (K=20 ic=64, digit 0, 200 ep)
    results/temp-12/ConvScCP_UNN_L1_LNO/model.pt              (digit 0)
    results/grid50/ConvScCP_k3_K6_ic128_L1_LNO/model.pt       (K=6 k=3, TOUS chiffres -> --digit -1)
"""
import argparse
import os
import time

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.architectures import ConvScCP_UNN
from generate_digits import infer_config


# ----------------------------------------------------------------------------- modele
def build_model(ckpt_path, device, weights="ema"):
    """Recharge un ConvScCP_UNN entraine (config auto-detectee depuis le state_dict).

    Accepte les deux formats du repo : state_dict nu (runs MNIST, train.py --save-model)
    et checkpoint dict des runs RGB (run_cifar10_torchcfm_recipe / run_imagenet32), qui
    contient a la fois 'state_dict' (poids bruts) et 'ema_model' (poids EMA).
    weights : 'ema' (defaut pour les ckpt RGB, ceux qui comptent en fin d'entrainement)
    ou 'raw'. Sans effet sur un state_dict nu."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and ("state_dict" in sd or "ema_model" in sd):
        key = "ema_model" if weights == "ema" else "state_dict"
        if key not in sd:
            key = "state_dict" if "state_dict" in sd else "ema_model"
            print(f"  [!] poids '{weights}' absents -> repli sur '{key}'", flush=True)
        print(f"  checkpoint dict (step={sd.get('step', '?')}, name={sd.get('name', '?')}) "
              f"-> poids '{key}'", flush=True)
        sd = sd[key]
    if "layers.0.W_weight" not in sd:
        raise ValueError("Ce script est specifique a ConvScCP_UNN (pas de 'layers.0.W_weight' "
                         f"dans le checkpoint). Cles : {list(sd)[:5]}")
    cfg = infer_config(sd)
    model = ConvScCP_UNN(dim=cfg["dim"], K=cfg["K"], internal_channel=cfg["internal_channel"],
                         use_Unet=cfg["use_Unet"], version=cfg["version"],
                         w_bias=cfg["w_bias"], in_channels=cfg["in_channels"],
                         img_size=cfg["img_size"], kernel_size=cfg["kernel"],
                         prox_w=cfg["prox_w"]).to(device)
    # strict=True : une cle manquante = config mal detectee -> on prefere planter
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, cfg


# ----------------------------------------------------- integrateur d'Euler avec dual
@torch.no_grad()
def euler_sample_utransfer(model, x0, steps, gamma, diagnostics=False):
    """Euler explicite t: 0 -> 1 en `steps` pas, en rechainant le dual du deroule CP.

    x0 : (B, dim). Retourne (x_final (B,dim), diag dict | None).

    gamma = 0 -> u_init = None a chaque pas = comportement standard du modele.
    Sinon u_init = gamma * u_{pas precedent} (u_{-1} = 0, donc le PREMIER pas est
    toujours identique a la baseline, quel que soit gamma).

    diag (si diagnostics=True), une entree par pas :
      u_norm    ||u^(K)||_2 moyen par echantillon (etat latent sortant)
      cos_prev  cos(u_i, u_{i-1}) moyen : stabilite du latent d'un pas au suivant
      v_norm    ||v||_2 moyen
      delta_v   ||v_warm - v_froid|| / ||v_froid|| : forward SUPPLEMENTAIRE a u_init=0
                sur le MEME x_t -> effet net du transfert sur le champ (0 si gamma=0)
    """
    B = x0.shape[0]
    dt = 1.0 / steps
    x = x0.clone()
    u_prev = None
    diag = {k: [] for k in ("t", "u_norm", "cos_prev", "v_norm", "delta_v")} if diagnostics else None

    for i in range(steps):
        t_val = i * dt
        t_col = torch.full((B, 1), t_val, device=x.device)
        xt_t = torch.cat([x, t_col], dim=-1)

        u_init = None if (gamma == 0.0 or u_prev is None) else gamma * u_prev
        v, u_out = model(xt_t, u_init=u_init, return_u=True)

        if diagnostics:
            un = u_out.flatten(1).norm(dim=1)
            cos = (float("nan") if u_prev is None else
                   torch.nn.functional.cosine_similarity(
                       u_out.flatten(1), u_prev.flatten(1), dim=1).mean().item())
            dv = 0.0
            if u_init is not None:
                v_cold = model(xt_t, u_init=None)
                dv = ((v - v_cold).norm(dim=1) / v_cold.norm(dim=1).clamp(min=1e-8)).mean().item()
            diag["t"].append(t_val)
            diag["u_norm"].append(un.mean().item())
            diag["cos_prev"].append(cos)
            diag["v_norm"].append(v.norm(dim=1).mean().item())
            diag["delta_v"].append(dv)

        x = x + dt * v
        u_prev = u_out

    return x, diag


@torch.no_grad()
def generate(model, n, dim, steps, gamma, device, batch, seed):
    """n echantillons par lots de `batch`, meme graine -> memes x0 pour tous les gamma."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    outs = []
    done = 0
    while done < n:
        b = min(batch, n - done)
        x0 = torch.randn(b, dim, generator=g).to(device)
        xf, _ = euler_sample_utransfer(model, x0, steps, gamma)
        outs.append(xf.cpu())
        done += b
        print(f"    {done}/{n}", flush=True)
    return torch.cat(outs, 0)[:n]


# ------------------------------------------------------------------------- metriques
def grad_energy(imgs):
    """Nettete : variation totale moyenne (meme definition que grille_convsccp.py)."""
    gx = (imgs[..., 1:, :] - imgs[..., :-1, :]).abs().mean()
    gy = (imgs[..., :, 1:] - imgs[..., :, :-1]).abs().mean()
    return float((gx + gy).item())


def main():
    p = argparse.ArgumentParser(
        description="Generation ConvScCP avec transfert du dual u entre pas d'Euler.")
    p.add_argument("--ckpt", required=True, help="state_dict d'un ConvScCP_UNN entraine.")
    p.add_argument("--weights", type=str, default="ema", choices=["ema", "raw"],
                   help="Jeu de poids pour les checkpoints RGB (CIFAR/AFHQ/ImageNet).")
    p.add_argument("--gammas", type=str, default="0,0.25,0.5,1.0",
                   help="Facteurs de transfert du dual (0 = baseline Euler standard).")
    p.add_argument("--steps", type=int, default=100, help="Pas d'Euler (defaut 100).")
    p.add_argument("--n-grid", type=int, default=8, help="Echantillons de la grille visuelle.")
    p.add_argument("--n-metrics", type=int, default=1000,
                   help="Echantillons par gamma pour mini-FID / entropie / std.")
    p.add_argument("--no-metrics", action="store_true",
                   help="Saute les metriques quantitatives (grille + diagnostics seulement).")
    p.add_argument("--digit", type=int, default=0,
                   help="Chiffre d'entrainement du checkpoint (-1 = tous) : sert au jeu "
                        "d'images REELLES de reference du mini-FID.")
    p.add_argument("--batch", type=int, default=500, help="Taille de lot en generation.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="results/sccp_utransfer")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    gammas = [float(g) for g in args.gammas.split(",")]
    device = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    model, cfg = build_model(args.ckpt, device, weights=args.weights)
    dim, C, S = cfg["dim"], cfg["in_channels"], cfg["img_size"]
    if C != 1 and not args.no_metrics:
        # le mini-FID et l'entropie de classe reposent sur un classifieur MNIST : ils
        # n'ont aucun sens sur des images RGB. On garde GE / std, qui sont generiques.
        print("  [!] checkpoint RGB -> metriques MNIST desactivees (mini-FID/entropie). "
              "Un vrai FID CIFAR se calcule avec compute_fid_cifar10.py.", flush=True)
        args.no_metrics = True
    n_params = sum(q.numel() for q in model.parameters())
    print(f"[utransfer] {args.ckpt}\n  K={cfg['K']} ic={cfg['internal_channel']} "
          f"kernel={cfg['kernel']} version={cfg['version']} w_bias={cfg['w_bias']} "
          f"prox={cfg['use_Unet']} | {n_params:,} params | device={device}", flush=True)
    print(f"  gammas={gammas} steps={args.steps} n_grid={args.n_grid} "
          f"n_metrics={0 if args.no_metrics else args.n_metrics}", flush=True)

    # ---- grille visuelle + diagnostics (meme bruit pour tous les gamma) ----
    torch.manual_seed(args.seed)
    x0_grid = torch.randn(args.n_grid, dim, device=device)
    grids, diags = {}, {}
    for gamma in gammas:
        t0 = time.time()
        xf, diag = euler_sample_utransfer(model, x0_grid, args.steps, gamma, diagnostics=True)
        grids[gamma] = xf.view(args.n_grid, C, S, S).cpu()
        diags[gamma] = diag
        print(f"  [grille] gamma={gamma}: {time.time() - t0:.1f}s  "
              f"delta_v_final={diag['delta_v'][-1]:.3f}  "
              f"cos_prev_moyen={np.nanmean(diag['cos_prev']):.3f}", flush=True)

    # ---- metriques quantitatives ----
    metrics = {}
    if not args.no_metrics:
        from mnist_metrics import train_or_load_classifier, mini_fid, class_entropy

        if args.n_metrics < 1000:
            # covariance 128x128 estimee sur peu de points -> mal conditionnee, le
            # mini-FID devient bruite. (Le LinAlgWarning de scipy peut apparaitre meme
            # avec plus d'echantillons : les features ReLU du classifieur sont creuses,
            # donc la covariance est de rang deficient. Il n'invalide pas le CLASSEMENT
            # relatif entre gammas, calcule au meme protocole.)
            print(f"  [!] n_metrics={args.n_metrics} < 1000 : mini-FID bruite "
                  f"(covariance 128-d mal conditionnee).", flush=True)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                             transform=transform)
        clf_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2)
        clf = train_or_load_classifier(clf_loader, device)

        if args.digit >= 0:
            idx = torch.where(dataset.targets == args.digit)[0]
            ref_set = Subset(dataset, idx)
        else:
            ref_set = dataset
        n_ref = min(args.n_metrics, len(ref_set))
        ref_loader = DataLoader(ref_set, batch_size=n_ref, shuffle=True)
        real = next(iter(ref_loader))[0][:n_ref]                       # (n_ref,1,28,28)
        ge_real = grad_energy(real)
        print(f"  reference reelle : {n_ref} images (digit="
              f"{'tous' if args.digit < 0 else args.digit}), GE_reel={ge_real:.4f}", flush=True)

        for gamma in gammas:
            t0 = time.time()
            print(f"  [metriques] gamma={gamma} ...", flush=True)
            samples = generate(model, args.n_metrics, dim, args.steps, gamma,
                               device, args.batch, args.seed)
            imgs = samples.view(-1, C, S, S).clamp(-1, 1)
            fid = mini_fid(clf, imgs, real, device)
            ent, counts = class_entropy(clf, imgs, device)
            metrics[gamma] = dict(fid=fid, entropy=ent, counts=counts,
                                  ge=grad_energy(imgs),
                                  std=float(samples.std(dim=0).mean().item()),
                                  secs=time.time() - t0)
            m = metrics[gamma]
            print(f"    mini-FID={m['fid']:.2f}  GE={m['ge']:.4f}  std={m['std']:.4f}  "
                  f"entropie_classe={m['entropy']:.3f}  ({m['secs']:.0f}s)", flush=True)
    else:
        ge_real = float("nan")

    # ---- figure 1 : grille d'echantillons, une ligne par gamma ----
    fig, axes = plt.subplots(len(gammas), args.n_grid,
                             figsize=(max(8.0, 1.25 * args.n_grid), 1.45 * len(gammas)),
                             squeeze=False)
    for r, gamma in enumerate(gammas):
        for c in range(args.n_grid):
            ax = axes[r, c]
            img = grids[gamma][c]
            if C == 1:
                ax.imshow(img[0].numpy(), cmap="gray", vmin=-1, vmax=1)
            else:                                   # RGB : [-1,1] -> [0,1], (C,S,S)->(S,S,C)
                ax.imshow((img * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
        lbl = "gamma=0\n(baseline)" if gamma == 0.0 else f"gamma={gamma}"
        axes[r, 0].set_ylabel(lbl, fontsize=9)
    fig.suptitle(f"ConvScCP K={cfg['K']} ic={cfg['internal_channel']} — transfert du dual u\n"
                 f"entre pas d'Euler ({args.steps} pas, meme bruit initial)", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(args.outdir, "samples.png"), dpi=130)
    plt.close(fig)

    # ---- figure 2 : diagnostics le long de la trajectoire ----
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for gamma in gammas:
        d = diags[gamma]
        lbl = f"gamma={gamma}"
        axs[0].plot(d["t"], d["u_norm"], label=lbl)
        axs[1].plot(d["t"], d["cos_prev"], label=lbl)
        axs[2].plot(d["t"], d["delta_v"], label=lbl)
    axs[0].set_title("||u^(K)|| (etat latent sortant)")
    axs[1].set_title("cos(u_i, u_{i-1}) : stabilite du latent")
    axs[2].set_title("delta_v = ||v_warm - v_froid|| / ||v_froid||")
    for ax in axs:
        ax.set_xlabel("t"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Diagnostics du transfert de dual le long de l'ODE", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(args.outdir, "diagnostics.png"), dpi=130)
    plt.close(fig)

    # ---- figure 3 : metriques vs gamma ----
    if metrics:
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        xs = list(metrics.keys())
        for ax, key, name in [(axs[0], "fid", "mini-FID (bas = mieux)"),
                              (axs[1], "ge", f"nettete GE (reel={ge_real:.3f})"),
                              (axs[2], "std", "diversite std_gen")]:
            ax.plot(xs, [metrics[g][key] for g in xs], "o-")
            ax.set_xlabel("gamma (transfert du dual)"); ax.set_title(name); ax.grid(alpha=0.3)
        axs[1].axhline(ge_real, color="k", ls="--", lw=1, label="reel")
        axs[1].legend(fontsize=8)
        fig.suptitle(f"Effet du transfert de l'espace latent — {args.n_metrics} echantillons/gamma",
                     fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(os.path.join(args.outdir, "metrics.png"), dpi=130)
        plt.close(fig)

    # ---- summary ----
    with open(os.path.join(args.outdir, "summary.txt"), "w") as f:
        f.write(f"# transfert du dual u entre pas d'Euler — ckpt={args.ckpt}\n")
        f.write(f"# K={cfg['K']} ic={cfg['internal_channel']} kernel={cfg['kernel']} "
                f"version={cfg['version']} w_bias={cfg['w_bias']} params={n_params}\n")
        f.write(f"# steps={args.steps} seed={args.seed} digit={args.digit} "
                f"n_metrics={0 if args.no_metrics else args.n_metrics} GE_reel={ge_real:.4f}\n")
        f.write("gamma\tdelta_v_moy\tdelta_v_final\tcos_prev_moy\tu_norm_final"
                "\tGE_grille\tdiff_L2_vs_gamma0\tmini_FID\tGE\tstd\tentropie_classe\n")
        base = grids[gammas[0]]
        for gamma in gammas:
            d = diags[gamma]
            m = metrics.get(gamma, {})
            # ecart pixel a pixel des images generees vs le premier gamma, meme bruit :
            # la quantite qui dit si le transfert change VISIBLEMENT quelque chose.
            diff = ((grids[gamma] - base).flatten(1).norm(dim=1)
                    / base.flatten(1).norm(dim=1).clamp(min=1e-12)).mean().item()
            f.write(f"{gamma}\t{np.mean(d['delta_v']):.4f}\t{d['delta_v'][-1]:.4f}\t"
                    f"{np.nanmean(d['cos_prev']):.4f}\t{d['u_norm'][-1]:.4f}\t"
                    f"{grad_energy(grids[gamma]):.4f}\t{diff:.6f}\t"
                    f"{m.get('fid', float('nan')):.3f}\t{m.get('ge', float('nan')):.4f}\t"
                    f"{m.get('std', float('nan')):.4f}\t{m.get('entropy', float('nan')):.4f}\n")
        if metrics:
            f.write("\n# repartition des classes predites (0..9)\n")
            for gamma in gammas:
                f.write(f"gamma={gamma}\t{metrics[gamma]['counts']}\n")

    print(f"\n[utransfer] sorties dans {args.outdir}/ "
          f"(samples.png, diagnostics.png"
          f"{', metrics.png' if metrics else ''}, summary.txt)", flush=True)


if __name__ == "__main__":
    main()
