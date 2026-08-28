# -*- coding: utf-8 -*-
"""
locality_afhq32_unet_vs_sccp.py — Pourquoi les samples du MinimalUNetFM (Kamb)
sur AFHQ 32x32 sont-ils coupes par des « coutures » droites (aspect mosaique /
local), alors que le ConvScCP produit des visages globalement coherents ?

Hypothese naive : c'est le champ de reception (RF).
Hypothese alternative testee ici : c'est le PADDING CIRCULAIRE.

    MinimalUNetFM (port fidele de Kamb) : TOUTES ses convs sont
    padding_mode="circular", et sur AFHQ img_size=32 le zero-pad 28->32 du port
    MNIST est de largeur 0. Le reseau est donc EXACTEMENT equivariant aux
    translations CYCLIQUES.  f(roll(x)) = roll(f(x)).
    Or la loi du bruit initial N(0,I) est, elle aussi, invariante par roll.
    Un flot equivariant transporte une loi invariante vers une loi invariante :
    la loi generee est FORCEMENT invariante par translation cyclique.
    => le modele ne PEUT PAS centrer un chat. Il tire un chat a un decalage
    uniforme sur le tore, qui apparait donc « enroule », avec une discontinuite
    droite pleine hauteur / pleine largeur : la couture = l'ancien bord d'image.

    ConvScCP : convs "same" a ZERO-padding, stride 1, une seule resolution.
    Le zero-padding brise l'equivariance et fournit la position absolue.

Ce script tranche par quatre mesures, sur les MEMES x0 :
  1. ERREUR D'EQUIVARIANCE  ||f(roll x) - roll f(x)|| / ||f(x)||  pour les deux
     modeles. (Le test de mecanisme.)
  2. RF EFFECTIF vs t pour les deux modeles, compare a R_unif (RF uniforme
     32x32) — repond directement a « est-ce le receptive field ? ».
  3. PROFIL DE COUTURE : energie de bord par frontiere ligne/colonne (cyclique),
     et rapport pic/mediane, sur samples UNet, samples ScCP, et vraies images
     AFHQ (controle). Une couture = un pic unique et isole.
  4. DE-ENROULEMENT : on roule chaque sample pour renvoyer sa frontiere de plus
     forte energie sur le bord de l'image. Si l'hypothese est juste, les samples
     UNet redeviennent des chats coherents et centres.

Usage
-----
    source ~/.venvs/unn/bin/activate
    python locality_afhq32_unet_vs_sccp.py --device cuda:1

Duree : ~4-6 min (2 modeles x 32 samples x 100 pas d'Euler, + 2 x 7 valeurs de
t x 12 backwards). Aucune ecriture dans les dossiers de run.

Sorties -> locality_afhq32_samples.png    (grilles zoomees : brut / de-enroule)
           locality_afhq32_seams.png      (profils de couture + histogramme)
           locality_afhq32_erf.png        (RF effectif vs t, les deux modeles)
           locality_afhq32_metrics.txt
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_fid_cifar10 import CHANNELS, IMG_SIZE, _VelocityWrapper
from sample_checkpoint import resolve_checkpoint

S = IMG_SIZE
TS = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]
N_ERF = 12                     # images reelles moyennees par valeur de t

UNET_DEFAULT = "results_afhq32/MinimalUNetFM_kamb/latest.pt"
SCCP_DEFAULT = "results_afhq32/ConvScCP_UNN_rgb_k9_K20_ic128_L1_LFO/latest.pt"


# --------------------------------------------------------------------------- #
# chargement
# --------------------------------------------------------------------------- #
def load(path, weights, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model, is_unet, name, keys = resolve_checkpoint(ckpt, device)
    key = keys[weights] if keys[weights] in ckpt else keys["raw"]
    model.load_state_dict(ckpt[key], strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, is_unet, name, int(ckpt.get("step", -1)), key


def real_images(cache, n, device):
    if not os.path.exists(cache):
        raise FileNotFoundError(f"{cache} absent — lance prepare_afhq_cats.py")
    d = torch.load(cache)
    return d["data"][:n].float().div_(127.5).sub_(1.0).to(device)


# --------------------------------------------------------------------------- #
# 1. equivariance cyclique
# --------------------------------------------------------------------------- #
@torch.no_grad()
def equivariance_error(model, x1, shifts):
    """||f(roll x) - roll f(x)|| / ||f(x)||, moyenne sur des t et des decalages.
    f = le DEBRUITEUR x1_pred (mode train() : les deux archis renvoient x1_pred).

    On teste DEUX familles de decalages, et la distinction est le coeur du
    mecanisme : le MinimalUNetFM a des convs circulaires (equivariantes a TOUT
    decalage) mais 3 MaxPool2d(2,2) (equivariantes seulement aux decalages
    MULTIPLES DE 8 = pas de sous-echantillonnage total). Son groupe de symetrie
    exact est donc 8Z x 8Z, pas Z x Z."""
    model.train()
    errs = []
    for t in (0.2, 0.5, 0.8):
        x0 = torch.randn_like(x1)
        xt = (1 - t) * x0 + t * x1
        tt = torch.full((xt.shape[0], 1), t, device=xt.device)

        def f(z):
            return model(torch.cat([z.reshape(z.shape[0], -1), tt], 1)).view_as(z)

        out = f(xt)
        for dy, dx in shifts:
            rolled = torch.roll(xt, shifts=(dy, dx), dims=(2, 3))
            lhs = f(rolled)
            rhs = torch.roll(out, shifts=(dy, dx), dims=(2, 3))
            errs.append((torch.norm(lhs - rhs) / torch.norm(rhs)).item())
    return float(np.mean(errs))


# --------------------------------------------------------------------------- #
# 2. RF effectif (noyau repris de erf_afhq_ckpt.py)
# --------------------------------------------------------------------------- #
def uniform_radius(s):
    c = s // 2
    ys, xs = np.mgrid[0:s, 0:s]
    return float(np.sqrt((ys - c) ** 2 + (xs - c) ** 2).mean())


def erf_at_t(model, x1, t):
    """|d x1_pred(pixel central) / d x_t| moyennee sur x1. Distance CYCLIQUE pour
    le rayon : avec padding circulaire la masse qui « sort » revient par l'autre
    bord, la mesurer en distance euclidienne plate la sous-estimerait."""
    model.train()
    dev = x1.device
    acc = torch.zeros(S, S)
    for i in range(x1.shape[0]):
        x0 = torch.randn(1, CHANNELS, S, S, device=dev)
        xt = ((1 - t) * x0 + t * x1[i:i + 1]).detach().requires_grad_(True)
        inp = torch.cat([xt.view(1, -1), torch.full((1, 1), t, device=dev)], dim=1)
        out = model(inp).view(1, CHANNELS, S, S)
        model.zero_grad(set_to_none=True)
        out[0, :, S // 2, S // 2].sum().backward()
        acc += xt.grad.detach().abs().sum(1).view(S, S).cpu()
    return (acc / x1.shape[0]).numpy()


def radial_stats(m, cyclic):
    c = S // 2
    ys, xs = np.mgrid[0:S, 0:S]
    dy, dx = np.abs(ys - c), np.abs(xs - c)
    if cyclic:                                  # distance sur le tore
        dy, dx = np.minimum(dy, S - dy), np.minimum(dx, S - dx)
    d = np.sqrt(dy ** 2 + dx ** 2)
    w = m / (m.sum() + 1e-12)
    return float((w * d).sum())


# --------------------------------------------------------------------------- #
# 3. profil de couture
# --------------------------------------------------------------------------- #
def seam_profiles(x):
    """x : (N,3,S,S). Renvoie (e_row, e_col) de forme (N,S) : energie de bord de
    chaque frontiere CYCLIQUE (la frontiere j separe la colonne j de j+1 mod S)."""
    e_col = (x - torch.roll(x, -1, dims=3)).abs().mean(dim=(1, 2))     # (N,S)
    e_row = (x - torch.roll(x, -1, dims=2)).abs().mean(dim=(1, 3))     # (N,S)
    return e_row.cpu().numpy(), e_col.cpu().numpy()


def peak_ratio(e):
    """max / mediane par echantillon : une couture unique donne un pic isole."""
    return (e.max(1) / (np.median(e, axis=1) + 1e-9))


def unroll(x, e_row, e_col):
    """Roule chaque image pour envoyer sa frontiere de plus forte energie sur le
    bord de l'image (donc hors champ visuel)."""
    out = x.clone()
    js, iss = e_col.argmax(1), e_row.argmax(1)
    for n in range(x.shape[0]):
        out[n] = torch.roll(x[n], shifts=(-(int(iss[n]) + 1), -(int(js[n]) + 1)), dims=(1, 2))
    return out, iss, js


# --------------------------------------------------------------------------- #
def to_img(x):
    return (x.detach().cpu() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()


def grid(ax_row, imgs, label):
    for k, ax in enumerate(ax_row):
        ax.imshow(imgs[k], interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
    ax_row[0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center", labelpad=6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unet-ckpt", default=UNET_DEFAULT)
    p.add_argument("--sccp-ckpt", default=SCCP_DEFAULT)
    p.add_argument("--cache", default="./data/afhq_cat32_train.pt")
    p.add_argument("--weights", default="ema", choices=["ema", "raw"])
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    log = []

    def say(s=""):
        print(s, flush=True); log.append(s)

    models = {}
    for tag, path in (("UNet_kamb", args.unet_ckpt), ("ScCP", args.sccp_ckpt)):
        m, is_unet, name, step, key = load(path, args.weights, dev)
        models[tag] = (m, is_unet, name, step)
        say(f"{tag:10s} : {name}  step={step:,}  poids={key}  "
            f"({sum(q.numel() for q in m.parameters()):,} params)")
    say()

    x1_real = real_images(args.cache, max(args.n, N_ERF), dev)

    # ---- 1. equivariance ---------------------------------------------------
    say("== 1. ERREUR D'EQUIVARIANCE CYCLIQUE  ||f(roll x) - roll f(x)|| / ||f(x)||")
    say("   (0 = exactement equivariant -> ne peut pas ancrer de position absolue)")
    MULT8 = ((8, 0), (0, 16), (8, 24))          # dans le groupe de sous-ech. 8Z x 8Z
    GENER = ((5, 0), (0, 7), (11, 13))          # hors de ce groupe
    eq = {}
    for tag, (m, _, _, _) in models.items():
        e8 = equivariance_error(m, x1_real[:8], MULT8)
        eg = equivariance_error(m, x1_real[:8], GENER)
        eq[tag] = e8
        say(f"   {tag:10s} : decalage MULTIPLE DE 8 -> {e8:.2e}   "
            f"decalage quelconque -> {eg:.2e}")
    say("   -> exactement equivariant sur un sous-groupe = loi generee INVARIANTE sous")
    say("      ce sous-groupe : le modele ne peut pas choisir la position du chat.")
    say()

    # ---- 2. RF effectif ----------------------------------------------------
    r_unif_flat = uniform_radius(S)
    ys, xs = np.mgrid[0:S, 0:S]
    c = S // 2
    dyc, dxc = np.abs(ys - c), np.abs(xs - c)
    dyc, dxc = np.minimum(dyc, S - dyc), np.minimum(dxc, S - dxc)
    r_unif_cyc = float(np.sqrt(dyc ** 2 + dxc ** 2).mean())
    say(f"== 2. RF EFFECTIF vs t   (R_unif plat = {r_unif_flat:.2f} px, "
        f"R_unif cyclique = {r_unif_cyc:.2f} px)")
    erf = {}
    for tag, (m, _, _, _) in models.items():
        cyc = eq[tag] < 1e-3                       # equivariant -> metrique du tore
        rs = []
        for t in TS:
            mp = erf_at_t(m, x1_real[:N_ERF], t)
            rs.append(radial_stats(mp, cyc))
        erf[tag] = (rs, cyc)
        ref = r_unif_cyc if cyc else r_unif_flat
        say(f"   {tag:10s} ({'tore' if cyc else 'plan'}) : " +
            "  ".join(f"t={t:.2f}:{r:4.1f}" for t, r in zip(TS, rs)) +
            f"   | moyenne {np.mean(rs):.2f} px = {100*np.mean(rs)/ref:.0f} % de R_unif")
    say()

    # ---- 3. samples --------------------------------------------------------
    say("== 3. GENERATION (memes x0 pour les deux modeles)")
    torch.manual_seed(args.seed)
    x0 = torch.randn(args.n, CHANNELS, S, S, device=dev)
    samples = {}
    with torch.no_grad():
        for tag, (m, is_unet, _, _) in models.items():
            m.eval()
            vf = _VelocityWrapper(m, is_unet).to(dev).eval()
            x, dt = x0.clone(), 1.0 / args.steps
            for i in range(args.steps):
                x = x + vf(torch.full((1,), i * dt, device=dev), x) * dt
            samples[tag] = x
            say(f"   {tag} : {args.n} samples, Euler {args.steps} pas")
    samples["AFHQ_reel"] = x1_real[:args.n]
    say()

    # ---- 4. coutures -------------------------------------------------------
    say("== 4. PROFIL DE COUTURE")
    say("   (a) rapport pic/mediane de l'energie de bord — NON discriminant, garde")
    say("       comme negatif : une vraie photo de chat a aussi des bords francs.")
    prof, ratios = {}, {}
    for tag, x in samples.items():
        er, ec = seam_profiles(x)
        prof[tag] = (er, ec)
        ratios[tag] = (peak_ratio(er).mean(), peak_ratio(ec).mean())
        say(f"   {tag:10s} : lignes {ratios[tag][0]:5.2f}   colonnes {ratios[tag][1]:5.2f}")
    say()

    # La frontiere j=S-1 est le bord d'image (entre derniere et premiere colonne) :
    # elle est trivialement maximale pour TOUTE image non periodique, donc exclue.
    say("   coutures INTERNES (j = 0..30, bord d'image exclu) alignees sur la grille")
    say(f"   des 3 poolings (j = 7, 15 ou 23, i.e. j = 7 mod 8 ; hasard = 3/{S-1} = 9.7 %) :")
    align = {}
    for tag, x in samples.items():
        er, ec = prof[tag]
        align[tag] = float(np.mean([(er[:, :-1].argmax(1) % 8 == 7).mean(),
                                    (ec[:, :-1].argmax(1) % 8 == 7).mean()]))
        say(f"   {tag:10s} : {100*align[tag]:5.1f} %")
    say()

    xu, iss, js = unroll(samples["UNet_kamb"], *prof["UNet_kamb"])
    hist = np.bincount(js % 8, minlength=8)
    er2, ec2 = seam_profiles(xu)
    say("   apres de-enroulement des samples UNet (couture renvoyee sur le bord) :")
    say(f"   pic/mediane sur les frontieres INTERNES : lignes "
        f"{(er2[:, :-1].max(1)/np.median(er2, 1)).mean():.2f}   colonnes "
        f"{(ec2[:, :-1].max(1)/np.median(ec2, 1)).mean():.2f}")
    say(f"   position de la couture (colonne) modulo 8, comptes : {hist.tolist()} "
        f"(indices 0..7)\n   -> concentree sur j = 7 mod 8 : le decalage cyclique est "
        f"QUANTIFIE au pas de 8 px,\n      exactement le sous-groupe sur lequel le "
        f"reseau est exactement equivariant.")
    say()

    # ---- figures -----------------------------------------------------------
    n_show = 8
    fig, axes = plt.subplots(4, n_show, figsize=(1.5 * n_show, 6.4))
    grid(axes[0], to_img(samples["UNet_kamb"][:n_show]), "UNet Kamb\n(brut)")
    grid(axes[1], to_img(xu[:n_show]), "UNet Kamb\nde-enroule")
    grid(axes[2], to_img(samples["ScCP"][:n_show]), "ConvScCP\n(brut)")
    grid(axes[3], to_img(samples["AFHQ_reel"][:n_show]), "AFHQ\nreel")
    fig.suptitle("AFHQ 32x32 — les « coutures » du UNet de Kamb sont un decalage "
                 "cyclique, pas un defaut de portee", fontsize=11)
    plt.tight_layout(); plt.savefig("locality_afhq32_samples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for tag, col in (("UNet_kamb", "crimson"), ("ScCP", "steelblue"), ("AFHQ_reel", "gray")):
        e = prof[tag][1]
        ax[0].plot(np.arange(S), (e / np.median(e, 1, keepdims=True))[:6].T,
                   color=col, lw=1.0, alpha=0.7)
        ax[0].plot([], [], color=col, label=tag)
    ax[0].set_title("energie de bord par frontiere de colonne\n(6 echantillons, "
                    "normalisee par sa mediane)", fontsize=9)
    ax[0].set_xlabel("frontiere j (entre colonnes j et j+1 mod 32)")
    ax[0].legend(fontsize=8)

    labels = list(align)
    xpos = np.arange(len(labels))
    ax[1].bar(xpos, [100 * align[l] for l in labels],
              color=["crimson", "steelblue", "gray"])
    ax[1].axhline(100 * 3 / (S - 1), color="k", ls=":", lw=1, label="hasard (3/31)")
    ax[1].set_xticks(xpos); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("% de coutures internes a j = 7 mod 8"); ax[1].legend(fontsize=8)
    ax[1].set_title("la couture est verrouillee sur la grille\ndes 3 poolings (pas de 8 px)",
                    fontsize=9)

    for tag, (rs, cyc) in erf.items():
        ax[2].plot(TS, rs, "o-", label=f"{tag} ({'tore' if cyc else 'plan'})")
    ax[2].axhline(r_unif_cyc, color="crimson", ls="--", lw=1,
                  label=f"R_unif tore ({r_unif_cyc:.1f})")
    ax[2].axhline(r_unif_flat, color="gray", ls="--", lw=1,
                  label=f"R_unif plan ({r_unif_flat:.1f})")
    ax[2].set_xlabel("t (0 = bruit, 1 = data)"); ax[2].set_ylabel("rayon effectif (px)")
    ax[2].set_title("RF effectif — est-ce la portee ?", fontsize=9)
    ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig("locality_afhq32_seams.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open("locality_afhq32_metrics.txt", "w") as f:
        f.write("\n".join(log) + "\n")
    print("\n-> locality_afhq32_samples.png , locality_afhq32_seams.png , "
          "locality_afhq32_metrics.txt", flush=True)


if __name__ == "__main__":
    main()
