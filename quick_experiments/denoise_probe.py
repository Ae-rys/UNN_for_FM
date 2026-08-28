# -*- coding: utf-8 -*-
"""
denoise_probe.py
Banc de prototypage RAPIDE d'architectures : au lieu d'entrainer le Flow Matching
complet (tous les t, 50k steps, ~3 h par config), on entraine un DEBRUITEUR a
quelques niveaux de bruit FIXES et on regarde la MSE de validation par niveau.

EST-CE QUE C'EST DU FLOW MATCHING ? Oui — c'est l'objectif FM du vrai run
(`run_afhq32.py` -> `run_cifar10_torchcfm_recipe.train_one`), a une seule
difference pres : t n'est plus tire dans U[0,1] mais dans une liste FINIE. Tout
le reste est identique par construction :

    construction de x_t      x_t = (1-t) x0 + t x1, x0 ~ N(0,I)   (sigma=0)   idem
    couplage                 OT exact par minibatch (--coupling)              idem
    parametrisation          x-pred : le reseau rend x1_pred                  idem
    loss                     ||x1_pred - x1||^2 / clamp((1-t)^2, 0.05)        idem
                             (--loss v ; c'est la MSE en espace VITESSE)
    augmentation             flip horizontal aleatoire                        idem
    clip du gradient         1.0                                             idem

Formellement : la loss FM est E_{t~U[0,1]} L(t) ; ici on optimise la moyenne de
L(t) sur 4 points. Meme integrande, support discret. Ce qui disparait, c'est
l'echantillonnage (pas d'EDO, pas de FID, pas d'accumulation d'erreur) — donc on
teste la CAPACITE de l'architecture, pas la qualite generative finale. Une config
qui ne debruite pas ne generera jamais ; l'inverse n'est pas garanti, d'ou
l'usage : PRE-SELECTION, pas verdict final.

Attention a la ponderation : `--loss v` (defaut, = recette du vrai run) pondere
par 1/clamp((1-t)^2, 0.05), ce qui donne ~13x plus de poids a t=0.8 qu'a t=0.2.
`--loss x1` met tous les t sur un pied d'egalite — utile pour juger la capacite
brute a bruit fort, mais ce n'est plus l'objectif du vrai run.

Ce qu'on mesure
---------------
    mse(t)   MSE de validation (images JAMAIS vues), bruit FIXE (seed dediee)
             -> comparable d'une config a l'autre au bit pres.
    nmse(t)  mse(t) / mse d'un predicteur trivial (la moyenne du dataset).
             nmse = 1 -> le modele n'apprend rien ; nmse -> 0 -> reconstruction
             parfaite. C'est le chiffre a lire.
    psnr(t)  10 log10(4 / mse), images dans [-1, 1].
    ref_copy(t)  MSE de "je rends x_t tel quel" = (1-t)^2 E||x0-x1||^2. Si le
             modele ne fait pas mieux, il n'a rien appris du tout.

La COURBE nmse(t) est evaluee sur une grille DENSE (--eval-t, defaut 0.05..0.95
par pas de 0.05), alors que l'entrainement ne voit que les quelques t de --t.
Elle dit donc deux choses a la fois : (1) la qualite de reconstruction a chaque
niveau de bruit, (2) si le modele INTERPOLE entre les t vus (les t d'entrainement
sont marques sur la figure — si la courbe ne fait pas de bosse entre eux, 4
niveaux suffisent a couvrir le continuum, ce qui valide le proxy lui-meme).

Les t bas (tres bruites) sont ceux qui demandent de la structure globale : c'est
typiquement la que le ScCP, purement local et equivariant, decroche. Regarder la
courbe et pas seulement la moyenne.

Configs
-------
Chaque config est une liste `cle=valeur` ; les configs sont separees par ';'.
Cles ScCP : k (kernel), K (profondeur = nb d'iterations depliees), ic (taille de
l'espace dual), prox (l1 | l1c | silu | double), ver (LFO | LNO), pw (largeur du
MLP de rayon du prox l1), wbias (0|1), ckpt (0|1 = gradient checkpointing).
`arch=unet_kamb` (MinimalUNetFM) et `arch=unet_ref` (UNet torchcfm 128ch) servent
de PLAFOND : ils disent quelle MSE est atteignable a ce budget de steps, donc si
l'ecart du ScCP vient de l'archi ou de la tache.

    prox=l1c : rayon du prox l1 appris PAR CANAL dual (L1ProxConv(channels=ic))
    au lieu d'un scalaire partage. Toujours un vrai prox l1 (norme ponderee),
    strictement plus expressif — knob non cable dans architectures.py, monte ici.

Usage
-----
    source ~/.venvs/unn/bin/activate

    # 1) chiffrage AVANT de lancer : params + it/s mesures sur 20 steps -> ETA
    python denoise_probe.py --grid "K=5,10,20 ic=128,256" --dry-run

    # 2) le balayage (suivable en tail -f claude.log)
    nohup python denoise_probe.py --grid "K=5,10,20 ic=128,256" >> claude.log 2>&1 &

    # 3) quelques configs precises, plus long, pour departager les finalistes
    python denoise_probe.py --steps 8000 \
        --configs "k=9,K=10,ic=256; k=5,K=20,ic=256; k=9,K=10,ic=512,prox=l1c"

    # plafond de reference (a lancer une fois, resultats reutilisables)
    python denoise_probe.py --configs "arch=unet_kamb; arch=unet_ref"

Sorties -> results_denoise_probe/
    summary.txt / summary.csv       classement par nmse moyenne
    mse_vs_t.png                    LA figure : une courbe par config
    <config>/grid.png               x_t | prediction | verite, une ligne par t
    <config>/loss.png, metrics.txt, model.pt
"""

import argparse
import gc
import itertools
import os
import time

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.architectures import (ConvScCP_UNN, ConvScCP_UNN_v2, ConvScCP_UNN_v3,
                                  ConvScCP_UNN_v4, ConvScCP_UNN_v5, MinimalUNetFM,
                                  L1ProxConv)
from run_cifar10_torchcfm_recipe import EMA, warmup_lambda


# ---------------------------------------------------------------------------
# Donnees : cache uint8 (N, C, S, S) — meme format que CIFAR / AFHQ / ImageNet-32
# ---------------------------------------------------------------------------

def load_data(cache, n_val, device, seed=0):
    """Charge le cache en memoire GPU (5 653 images 32x32 = 17 Mo en float16
    equivalent, ~70 Mo en float32 : ca tient largement, et ca supprime le
    dataloader — pour un probe de 3 000 steps, l'IO ne doit pas dominer).

    Retourne (x_train, x_val) dans [-1, 1], shape (N, C, S, S)."""
    if not os.path.exists(cache):
        raise FileNotFoundError(
            f"{cache} absent — lance d'abord : python prepare_afhq_cats.py")
    d = torch.load(cache)
    x = d["data"].float().div_(127.5).sub_(1.0)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)
    x = x[perm]
    x_val, x_train = x[:n_val], x[n_val:]
    print(f"Donnees {cache} : {x_train.shape[0]} train / {x_val.shape[0]} val, "
          f"{tuple(x.shape[1:])}", flush=True)
    return x_train.to(device), x_val.to(device)


def make_val_set(x_val, t_list, seed=1234):
    """Bruit FIXE : le meme x0 pour toutes les configs et toutes les evaluations.

    Couplage INDEPENDANT ici, meme quand l'entrainement utilise l'OT : le plan OT
    est une propriete du minibatch, il n'est pas defini pour une image de test
    isolee. La validation mesure donc "debruiter du bruit gaussien", ce qui est
    exactement ce que fait le sampler a la generation.

    Retourne un dict t -> (xt, x1) aplatis (B, dim). Un tirage separe (generateur
    dedie) garantit que le flux aleatoire de l'entrainement ne le decale pas."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = x_val.shape[0]
    dim = x_val[0].numel()
    x1 = x_val.reshape(B, -1)
    out = {}
    for t in t_list:
        x0 = torch.randn(B, dim, generator=g).to(x1.device)
        out[t] = ((1 - t) * x0 + t * x1, x1)
    return out


# ---------------------------------------------------------------------------
# Modeles
# ---------------------------------------------------------------------------

DEFAULTS = dict(arch="sccp", k=9, K=10, ic=256, prox="l1", ver="LFO",
                pw=32, wbias=1, ckpt=1,
                # arch=sccp_v2 : meme archi, mais le terme d'attache du VRAI
                # probleme inverse (x_t = t.x1 + (1-t).eps) au lieu de
                # 1/2||x_t - x||^2, et le momentum pilote par mu(t) = t^2/(1-t)^2
                # au lieu de mu = 1. Cf. ConvScCP_UNN_v2 et verif_fidelite_sccp.py.
                # x0 : "zero" (defaut v2/v3, = moyenne du prior) ou "xt" (choix de v1).
                # arch=sccp_v3 : v2 + la condition de pas de Chambolle-Pock retablie
                # (sigma_k = 0.99/(tau_k L^2), L^2 = vrai gain de boucle ||BoA||).
                # Cf. DERIVATION_LFO.md et verif_lfo_condition.py.
                x0="zero",
                # dlg=1 : l2 du gain de boucle DIFFERENTIABLE (sccp_v3 seulement ;
                # v4/v5 l'ont toujours). Sert a isoler la correction du gradient.
                dlg=0,
                # knobs du UNet torchcfm (arch=unet_ref). Defauts = REF_UNET_CFG,
                # la config officielle a 35.7M params : trop grosse pour 5k images,
                # d'ou l'interet de pouvoir la reduire pour un plafond a capacite
                # COMPARABLE aux ScCP (0.3 - 7M).
                uch=128, ublocks=2, umult="1-2-2-2", uattn="16")
_INT_KEYS = {"k", "K", "ic", "pw", "wbias", "ckpt", "uch", "ublocks", "dlg"}


def parse_config(spec):
    cfg = dict(DEFAULTS)
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"config invalide : '{item}' (attendu cle=valeur)")
        key, val = (s.strip() for s in item.split("=", 1))
        if key not in DEFAULTS:
            raise ValueError(f"cle inconnue '{key}' — connues : {sorted(DEFAULTS)}")
        cfg[key] = int(val) if key in _INT_KEYS else val
    return cfg


def config_name(cfg):
    if cfg["arch"] == "unet_ref":
        n = f"unet_ref_ch{cfg['uch']}_b{cfg['ublocks']}"
        if cfg["umult"] != DEFAULTS["umult"]:
            n += f"_m{cfg['umult']}"
        if cfg["uattn"] != DEFAULTS["uattn"]:
            n += f"_a{cfg['uattn']}"
        return n
    if cfg["arch"] not in ("sccp", "sccp_v2", "sccp_v3", "sccp_v4", "sccp_v5"):
        return cfg["arch"]
    tag = {"sccp": "ScCP", "sccp_v2": "ScCPv2", "sccp_v3": "ScCPv3",
           "sccp_v4": "ScCPv4", "sccp_v5": "ScCPv5"}[cfg["arch"]]
    n = f"{tag}_k{cfg['k']}_K{cfg['K']}_ic{cfg['ic']}_{cfg['prox']}_{cfg['ver']}"
    if cfg["arch"] in ("sccp_v2", "sccp_v3", "sccp_v4", "sccp_v5") and cfg["x0"] != DEFAULTS["x0"]:
        n += f"_x0{cfg['x0']}"
    if cfg["arch"] == "sccp_v3" and cfg["dlg"]:
        n += "_dlg"
    if cfg["pw"] != DEFAULTS["pw"]:
        n += f"_pw{cfg['pw']}"
    if not cfg["wbias"]:
        n += "_nobias"
    return n


def remap_state_dict(sd):
    """Convertit un state_dict de la parametrisation in_conv/out_conv vers
    W_weight/V_weight/W_bias.

    `_OrigConvScCP_Iteration` a existe sous deux formes mathematiquement
    IDENTIQUES : des nn.Parameter bruts passes a F.conv2d / F.conv_transpose2d
    (forme actuelle), et des modules nn.Conv2d / nn.ConvTranspose2d les portant
    (forme active le 17/08 vers 15 h). Le shim de compatibilite present dans
    architectures.py ne convertit que dans le sens ancien -> nouveau ; il manque
    le retour, sans lequel les checkpoints entraines sous la forme module sont
    illisibles par les outils d'analyse.

    Sans effet sur un state_dict deja au bon format.
    """
    out = {}
    for k, v in sd.items():
        if k.endswith("in_conv.weight"):
            out[k.replace("in_conv.weight", "W_weight")] = v
        elif k.endswith("in_conv.bias"):
            out[k.replace("in_conv.bias", "W_bias")] = v
        elif k.endswith("out_conv.weight"):
            out[k.replace("out_conv.weight", "V_weight")] = v
        else:
            out[k] = v
    return out


def name_to_config(name):
    """Inverse de config_name : "ScCP_k9_K10_ic256_l1_LFO" -> dict de config.
    Sert a rebatir un modele depuis un run deja sur disque (denoise_curve.py)."""
    import re
    if name in ("unet_kamb", "unet_ref"):
        return dict(DEFAULTS, arch=name)
    m = re.fullmatch(r"unet_ref_ch(\d+)_b(\d+)(?:_m([\d-]+))?(?:_a([\d,]+))?", name)
    if m:
        cfg = dict(DEFAULTS, arch="unet_ref", uch=int(m.group(1)),
                   ublocks=int(m.group(2)))
        if m.group(3):
            cfg["umult"] = m.group(3)
        if m.group(4):
            cfg["uattn"] = m.group(4)
        return cfg
    m = re.fullmatch(r"ScCP_k(\d+)_K(\d+)_ic(\d+)_(l1c|l1|silu|double)_(LFO|LNO)"
                     r"(?:_pw(\d+))?(_nobias)?", name)
    if not m:
        raise ValueError(f"nom de run non reconnu : {name}")
    cfg = dict(DEFAULTS, arch="sccp", k=int(m.group(1)), K=int(m.group(2)),
               ic=int(m.group(3)), prox=m.group(4), ver=m.group(5))
    if m.group(6):
        cfg["pw"] = int(m.group(6))
    if m.group(7):
        cfg["wbias"] = 0
    return cfg


def build_model(cfg, device, channels, img_size):
    dim = channels * img_size * img_size
    if cfg["arch"] == "unet_kamb":
        return MinimalUNetFM(dim=dim, in_channels=channels, img_size=img_size).to(device)
    if cfg["arch"] == "unet_ref":
        from torchcfm.models.unet import UNetModel
        if cfg["uch"] % 32:
            raise ValueError(
                f"uch={cfg['uch']} invalide : le UNet torchcfm normalise par "
                f"GroupNorm(32, .), donc uch doit etre un multiple de 32 "
                f"(32, 64, 96, 128, ...).")
        # uattn="1" desactive de fait l'attention : la resolution 1 donne un
        # facteur de sous-echantillonnage (img_size // 1) jamais atteint par les
        # etages, alors que "16" place l'attention sur les cartes 16x16.
        return UNetModel(dim=(channels, img_size, img_size),
                         num_channels=cfg["uch"], num_res_blocks=cfg["ublocks"],
                         channel_mult=[int(v) for v in cfg["umult"].split("-")],
                         num_heads=4, num_head_channels=64,
                         attention_resolutions=cfg["uattn"], dropout=0.1).to(device)
    if cfg["arch"] not in ("sccp", "sccp_v2", "sccp_v3", "sccp_v4", "sccp_v5"):
        raise ValueError(f"arch inconnue : {cfg['arch']}")

    prox = cfg["prox"]
    Cls = {"sccp": ConvScCP_UNN, "sccp_v2": ConvScCP_UNN_v2,
           "sccp_v3": ConvScCP_UNN_v3, "sccp_v4": ConvScCP_UNN_v4,
           "sccp_v5": ConvScCP_UNN_v5}[cfg["arch"]]
    extra = {} if cfg["arch"] == "sccp" else dict(x0_mode=cfg["x0"])
    if cfg["arch"] == "sccp_v3":
        extra["diff_loop_gain"] = bool(cfg["dlg"])
    model = Cls(
        dim=dim, K=cfg["K"], internal_channel=cfg["ic"], kernel_size=cfg["k"],
        in_channels=channels, img_size=img_size,
        use_Unet=("l1" if prox in ("l1", "l1c") else
                  "silu" if prox == "silu" else False),
        version=cfg["ver"], use_checkpoint=bool(cfg["ckpt"]),
        w_bias=bool(cfg["wbias"]), prox_w=cfg["pw"], **extra,
    ).to(device)
    if prox == "l1c":
        # rayon du prox l1 appris par canal dual (cf. L1ProxConv.__doc__).
        for layer in model.layers:
            layer.prox = L1ProxConv(w=cfg["pw"], channels=cfg["ic"]).to(device)
    return model


# ---------------------------------------------------------------------------
# Prediction unifiee : tout modele -> x1_pred, en train comme en eval
# ---------------------------------------------------------------------------

def forward_x1(model, xt, t, channels, img_size, is_unet_ref):
    """xt : (B, dim) aplati, t : (B, 1). Retourne x1_pred (B, dim).

    En eval les modeles x-pred renvoient deja la vitesse v = (x1-x_t)/clamp(1-t) :
    on applique EXACTEMENT la transformation inverse (meme clamp), la conversion
    est donc sans perte."""
    B = xt.shape[0]
    if is_unet_ref:
        v = model(t.view(-1), xt.view(B, channels, img_size, img_size)).reshape(B, -1)
        return xt + v * (1 - t)
    out = model(torch.cat([xt, t], dim=-1))
    if model.training and getattr(model, "predicts_x1", False):
        return out                                    # deja x1_pred
    scale = torch.clamp(1 - t, min=0.05) if not model.training else (1 - t)
    return xt + out * scale


@torch.no_grad()
def evaluate(model, val_sets, channels, img_size, is_unet_ref, batch=256):
    """MSE par t sur la validation, en eval() (donc EMA-compatible et sans
    gradient checkpointing). Reduction : moyenne par pixel."""
    was = model.training
    model.eval()
    res = {}
    for t, (xt, x1) in val_sets.items():
        se, n = 0.0, 0
        for i in range(0, xt.shape[0], batch):
            xb, yb = xt[i:i + batch], x1[i:i + batch]
            tb = torch.full((xb.shape[0], 1), float(t), device=xb.device)
            pred = forward_x1(model, xb, tb, channels, img_size, is_unet_ref)
            se += F.mse_loss(pred, yb, reduction="sum").item()
            n += yb.numel()
        res[t] = se / n
    model.train(was)
    return res


def reference_mse(val_sets, x_val):
    """Deux garde-fous, calcules une fois pour toutes :
      mean : MSE du predicteur constant (moyenne du train ~ moyenne du dataset)
             = variance des donnees. C'est le denominateur de la nmse.
      copy : MSE de "je rends x_t" = (1-t)^2 E||x0-x1||^2 (aucun debruitage)."""
    x1 = x_val.reshape(x_val.shape[0], -1)
    mu = x1.mean(dim=0, keepdim=True)
    mse_mean = F.mse_loss(mu.expand_as(x1), x1).item()
    copy = {t: F.mse_loss(xt, y).item() for t, (xt, y) in val_sets.items()}
    return mse_mean, copy


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_grid(model, val_sets, path, title, channels, img_size, is_unet_ref, n=8,
              t_show=None):
    """Une ligne x_t + une ligne prediction par niveau de bruit, puis la verite.
    t_show : niveaux montres (defaut = tous ceux de val_sets ; on passe en general
    les seuls t d'entrainement, la grille dense donnerait 40 lignes)."""
    was = model.training
    model.eval()
    t_list = sorted(t_show) if t_show is not None else sorted(val_sets)
    rows, labels = [], []
    for t in t_list:
        xt, x1 = val_sets[t]
        xb = xt[:n]
        tb = torch.full((n, 1), float(t), device=xb.device)
        pred = forward_x1(model, xb, tb, channels, img_size, is_unet_ref)
        rows += [xb.cpu(), pred.cpu()]
        labels += [f"$x_t$  t={t:g}", f"pred  t={t:g}"]
    rows.append(val_sets[t_list[0]][1][:n].cpu())
    labels.append("$x_1$ (verite)")
    model.train(was)

    nr = len(rows)
    fig, axes = plt.subplots(nr, n, figsize=(1.35 * n, 1.45 * nr))
    fig.suptitle(title, fontsize=10)
    for r, (row, lab) in enumerate(zip(rows, labels)):
        imgs = (row.view(-1, channels, img_size, img_size) * 0.5 + 0.5).clamp(0, 1)
        for c in range(n):
            ax = axes[r, c]
            im = imgs[c].permute(1, 2, 0).numpy()
            ax.imshow(im.squeeze() if channels == 1 else im,
                      cmap="gray" if channels == 1 else None)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(lab, fontsize=7, rotation=0, ha="right", va="center")
    plt.tight_layout(rect=(0.02, 0, 1, 0.97))
    plt.savefig(path, dpi=90)
    plt.close(fig)


def plot_summary(results, mse_mean, copy_ref, path, t_train):
    """LA figure : nmse(t) sur la grille dense, une courbe par config.

    Les t d'entrainement sont marques (points pleins + traits verticaux) ; entre
    eux la courbe est de l'EXTRAPOLATION en t. Une bosse marquee entre deux t vus
    signifierait qu'il faut densifier --t ; une courbe lisse valide le proxy."""
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for tt in t_train:
        ax.axvline(tt, color="gray", lw=0.6, ls=":", zorder=0)
    for r in sorted(results, key=lambda r: r["nmse_mean"]):
        ts = sorted(r["mse"])
        ys = [r["mse"][t] / mse_mean for t in ts]
        line, = ax.plot(ts, ys, "-", lw=1.6,
                        label=f"{r['name']}  ({r['n_params']/1e6:.2f}M)")
        seen = [(t, r["mse"][t] / mse_mean) for t in ts if t in set(t_train)]
        if seen:
            ax.plot(*zip(*seen), "o", ms=5, color=line.get_color(), zorder=3)
    ts = sorted(copy_ref)
    ax.plot(ts, [copy_ref[t] / mse_mean for t in ts], "k--", lw=1,
            label="copie de $x_t$ (aucun debruitage)")
    ax.axhline(1.0, color="gray", lw=1.2, ls="-.")
    ax.text(ts[0], 1.05, "predicteur constant (moyenne du dataset)",
            fontsize=7, color="gray")
    ax.set_xlabel("t   (0 = bruit pur, 1 = image propre)")
    ax.set_ylabel("nmse = mse / var(donnees)   —   plus bas = mieux")
    ax.set_yscale("log")
    ax.set_ylim(top=max(3.0, ax.get_ylim()[1]))
    ax.set_title("Qualite de reconstruction par niveau de bruit\n"
                 "(points = t vus a l'entrainement, entre eux = extrapolation)",
                 fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------
# Entrainement d'une config
# ---------------------------------------------------------------------------

def train_probe(model, name, x_train, val_sets, run_dir, args, channels, img_size,
                is_unet_ref, t_list):
    os.makedirs(run_dir, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    dim = channels * img_size * img_size
    N = x_train.shape[0]
    ot = None
    if args.coupling == "ot":
        from torchcfm.optimal_transport import OTPlanSampler
        ot = OTPlanSampler(method="exact")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_lambda(max(args.warmup, 1)))
    ema = EMA(model, decay=args.ema)

    # --compile : ~1.9x sur le deroule ScCP (fusion des ops elementwise de
    # l'algebre CP, cf. bench_speedup_levers.py). Seule la BOUCLE D'ENTRAINEMENT
    # utilise `net` ; `model` reste le module nu, donc EMA, state_dict, reprise
    # de checkpoint et evaluation sont inchanges (pas de prefixe _orig_mod).
    net = model
    if getattr(args, "compile", False):
        import torch._inductor.config as _ind
        # sans ca, le backward plante : inductor choisit un layout channels_last
        # pour les convs du deroule et recoit du contigu (assert_size_stride).
        _ind.layout_optimization = False
        net = torch.compile(model)
        print("  torch.compile actif (premier pas ~60-90 s de compilation)", flush=True)

    print(f"\n{'='*70}\n{name}\nParams : {n_params:,} ({n_params/1e6:.2f}M)  |  "
          f"steps {args.steps:,}  lr {args.lr}  t {t_list}\n{'='*70}", flush=True)

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    t_tensor = torch.tensor(t_list, device=x_train.device)
    loss_log, running, n_run = [], 0.0, 0

    # ---- reprise ----
    # Les fichiers de resultat (metrics.txt, model.pt, grid.png) ne sont ecrits
    # qu'a la FIN. Sans ce checkpoint intermediaire, un run tue a 80 % ne laisse
    # rien du tout — c'est arrive, 3 h 40 de GPU perdues. On sauvegarde donc
    # l'etat COMPLET (poids, EMA, optimiseur, scheduler, RNG) tous les
    # --ckpt-every steps, et on repart automatiquement de la si le fichier existe.
    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    start_step, elapsed_before = 1, 0.0
    if args.resume and os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location=x_train.device, weights_only=False)
        model.load_state_dict(st["model"])
        ema.shadow.load_state_dict(st["ema"])
        opt.load_state_dict(st["optim"])
        sched.load_state_dict(st["sched"])
        # map_location a pu envoyer l'etat du generateur sur le GPU ; set_state
        # exige un ByteTensor CPU.
        g.set_state(st["gen"].cpu().to(torch.uint8))
        loss_log = st.get("loss_log", [])
        elapsed_before = float(st.get("elapsed", 0.0))
        start_step = int(st["step"]) + 1
        print(f"    reprise depuis {ckpt_path} au step {st['step']:,} "
              f"({elapsed_before/60:.1f} min deja cumulees)", flush=True)
        if start_step > args.steps:
            print(f"    deja au budget ({st['step']:,} >= {args.steps:,}).", flush=True)

    def save_ckpt(step, elapsed):
        """Ecriture atomique : un kill pendant torch.save laisserait sinon un
        fichier tronque, et la reprise echouerait au lieu de sauver le run."""
        payload = {"step": step, "model": model.state_dict(),
                   "ema": ema.model().state_dict(), "optim": opt.state_dict(),
                   "sched": sched.state_dict(), "gen": g.get_state(),
                   "loss_log": loss_log, "elapsed": elapsed}
        torch.save(payload, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    model.train()
    t0 = time.perf_counter()
    t_win = t0

    for step in range(start_step, args.steps + 1):
        idx = torch.randint(0, N, (args.batch_size,), generator=g).to(x_train.device)
        x1_img = x_train[idx]
        if not args.no_flip:
            m = torch.rand(x1_img.shape[0], device=x1_img.device) < 0.5
            x1_img = torch.where(m[:, None, None, None], torch.flip(x1_img, dims=[3]), x1_img)
        x1 = x1_img.reshape(args.batch_size, -1)
        x0 = torch.randn(args.batch_size, dim, device=x1.device)
        if ot is not None:
            # les DEUX sorties de sample_plan doivent etre reprises ensemble :
            # n'en garder qu'une desapparie les paires (cf. track_ot_coupling_bug).
            x0, x1 = ot.sample_plan(x0, x1)
        # un niveau de bruit tire par echantillon dans la liste fixe (variance
        # plus faible qu'un t par batch, et chaque batch informe tous les t)
        t = t_tensor[torch.randint(0, len(t_list), (args.batch_size,),
                                   generator=g).to(x_train.device)].view(-1, 1)
        xt = (1 - t) * x0 + t * x1

        pred = forward_x1(net, xt, t, channels, img_size, is_unet_ref)
        if args.loss == "v":
            # MSE en espace vitesse = MSE sur x1 ponderee par 1/(1-t)^2, clampee
            # a 0.05 comme dans train_one. C'est LA loss du vrai run.
            loss = torch.mean((pred - x1) ** 2 / torch.clamp((1 - t) ** 2, min=0.05))
        else:
            loss = F.mse_loss(pred, x1)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()
        ema.update(model)

        running += loss.item(); n_run += 1
        if step == 50 or step % args.log_every == 0 or step == args.steps:
            sps = n_run / (time.perf_counter() - t_win)
            print(f"  step {step:>6,}/{args.steps:,}  train_mse {running/n_run:.4f}  "
                  f"{sps:.2f} it/s  ETA {(args.steps-step)/sps/60:.1f} min", flush=True)
            loss_log.append((step, running / n_run))
            running, n_run = 0.0, 0
            t_win = time.perf_counter()

        if args.ckpt_every and (step % args.ckpt_every == 0 or step == args.steps):
            save_ckpt(step, elapsed_before + (time.perf_counter() - t0))

    dt = time.perf_counter() - t0
    mse_raw = evaluate(model, val_sets, channels, img_size, is_unet_ref)
    mse_ema = evaluate(ema.model(), val_sets, channels, img_size, is_unet_ref)
    # on garde le meilleur des deux : a 3k steps l'EMA n'est pas toujours mure
    use_ema = sum(mse_ema.values()) < sum(mse_raw.values())
    mse = mse_ema if use_ema else mse_raw
    best = ema.model() if use_ema else model

    save_grid(best, val_sets, os.path.join(run_dir, "grid.png"),
              f"{name} — {args.steps} steps ({'EMA' if use_ema else 'raw'})",
              channels, img_size, is_unet_ref, t_show=t_list)
    if loss_log:
        xs, ys = zip(*loss_log)
        plt.figure(figsize=(5, 3.2)); plt.plot(xs, ys)
        plt.xlabel("step"); plt.ylabel("train MSE"); plt.yscale("log")
        plt.title(name, fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "loss.png"), dpi=100); plt.close()
    if args.save_ckpt:
        torch.save({"name": name, "state_dict": model.state_dict(),
                    "ema_model": ema.model().state_dict(), "steps": args.steps,
                    "mse": mse, "t_list": t_list},
                   os.path.join(run_dir, "model.pt"))

    return dict(name=name, n_params=n_params, mse=mse, mse_raw=mse_raw,
                mse_ema=mse_ema, used=("ema" if use_ema else "raw"),
                time_s=dt, it_s=args.steps / dt,
                final_train=(loss_log[-1][1] if loss_log else float("nan")))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def expand_grid(grid):
    """"K=5,10,20 ic=128,256" -> liste de specs (produit cartesien)."""
    keys, values = [], []
    for token in grid.split():
        key, vals = token.split("=", 1)
        keys.append(key.strip()); values.append([v.strip() for v in vals.split(",")])
    return [",".join(f"{k}={v}" for k, v in zip(keys, combo))
            for combo in itertools.product(*values)]


def main():
    p = argparse.ArgumentParser(
        description="Prototypage d'archis par debruitage a bruit fixe.")
    p.add_argument("--cache", type=str, default="./data/afhq_cat32_train.pt")
    p.add_argument("--configs", type=str, default="",
                   help="Configs 'cle=valeur,...' separees par ';'. Defaut (si ni "
                        "--configs ni --grid) : k=9,K=10,ic=256.")
    p.add_argument("--grid", type=str, default="",
                   help="Produit cartesien, ex: \"K=5,10,20 ic=128,256\". Se combine "
                        "avec --configs si les deux sont donnes.")
    p.add_argument("--t", type=str, default="0.2,0.4,0.6,0.8",
                   help="Niveaux de bruit vus a l'ENTRAINEMENT (t du FM ; 0 = bruit pur).")
    p.add_argument("--eval-t", type=str, default="",
                   help="Grille d'EVALUATION de la courbe nmse(t). Defaut : dense, "
                        "0.05 a 0.95 par pas de 0.05 (les t de --t y sont ajoutes).")
    p.add_argument("--loss", type=str, default="v", choices=["v", "x1"],
                   help="v (defaut) = MSE en espace vitesse, ponderee 1/clamp((1-t)^2, "
                        "0.05) : la loss EXACTE du vrai run. x1 = MSE brute sur x1, "
                        "tous les t a poids egal.")
    p.add_argument("--coupling", type=str, default="ot", choices=["ot", "indep"],
                   help="ot (defaut) = plan OT exact par minibatch, comme run_afhq32.")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4,
                   help="5e-4 (defaut) converge vite sur 3k steps ; 2e-4 = recette "
                        "torchcfm du vrai run.")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--n-val", type=int, default=512)
    p.add_argument("--no-flip", action="store_true")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--ckpt-every", type=int, default=2000,
                   help="Sauvegarde de reprise tous les N steps (0 = desactive). "
                        "Les resultats finaux ne sont ecrits qu'a la fin : sans ce "
                        "filet, un run interrompu ne laisse RIEN.")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Repart de zero meme si un ckpt.pt existe.")
    p.set_defaults(resume=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default="results_denoise_probe")
    p.add_argument("--save-ckpt", action="store_true", default=True)
    p.add_argument("--no-save-ckpt", dest="save_ckpt", action="store_false")
    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile la boucle d'entrainement : ~1.9x sur les "
                        "ScCP (mesure : bench_speedup_levers.py). Coute 60-90 s de "
                        "compilation au demarrage de chaque config, et la loss n'est "
                        "pas bit-a-bit identique a l'eager (~1e-3 relatif, "
                        "reassociation flottante).")
    p.add_argument("--dry-run", action="store_true",
                   help="Ne fait que compter les params et chronometrer 20 steps "
                        "par config -> ETA total, puis s'arrete.")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and (device.index or 0) >= torch.cuda.device_count():
        print(f"[warn] {device} invalide ({torch.cuda.device_count()} GPU visible(s)) "
              f"-> repli sur cuda:0", flush=True)
        device = torch.device("cuda:0")
    t_list = [float(s) for s in args.t.split(",") if s.strip()]
    if args.eval_t:
        eval_t = [float(s) for s in args.eval_t.split(",") if s.strip()]
    else:
        eval_t = [round(0.05 * i, 2) for i in range(1, 20)]      # 0.05 .. 0.95
    eval_t = sorted(set(eval_t) | set(t_list))
    torch.manual_seed(args.seed)

    x_train, x_val = load_data(args.cache, args.n_val, device, seed=args.seed)
    channels, img_size = x_train.shape[1], x_train.shape[2]
    val_sets = make_val_set(x_val, eval_t, seed=args.seed + 1234)
    mse_mean, copy_ref = reference_mse(val_sets, x_val)
    print(f"Device {device} | loss={args.loss} coupling={args.coupling} | "
          f"train t={t_list} | eval sur {len(eval_t)} niveaux "
          f"[{eval_t[0]:g} .. {eval_t[-1]:g}] | var(donnees)={mse_mean:.4f}", flush=True)

    specs = [s for s in args.configs.split(";") if s.strip()]
    if args.grid:
        specs += expand_grid(args.grid)
    if not specs:
        specs = ["k=9,K=10,ic=256"]
    cfgs = [parse_config(s) for s in specs]
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"\n{len(cfgs)} config(s) : " + ", ".join(config_name(c) for c in cfgs) + "\n",
          flush=True)

    # ---- chiffrage : params + it/s mesures, avant de s'engager ----
    if args.dry_run:
        total = 0.0
        ot = None
        if args.coupling == "ot":
            from torchcfm.optimal_transport import OTPlanSampler
            ot = OTPlanSampler(method="exact")
        print(f"{'config':<44}{'params':>12}   it/s   train    eval   total")
        for cfg in cfgs:
            model = build_model(cfg, device, channels, img_size)
            n_params = sum(p.numel() for p in model.parameters())
            is_ref = cfg["arch"] == "unet_ref"
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            model.train()
            for i in range(25):                      # 5 de chauffe + 20 chronometres
                if i == 5:
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                x1 = x_train[:args.batch_size].reshape(args.batch_size, -1)
                x0 = torch.randn_like(x1)
                if ot is not None:
                    x0, x1 = ot.sample_plan(x0, x1)
                t = torch.full((args.batch_size, 1), t_list[0], device=device)
                xt = (1 - t) * x0 + t * x1
                loss = F.mse_loss(forward_x1(model, xt, t, channels, img_size, is_ref), x1)
                opt.zero_grad(); loss.backward(); opt.step()
            torch.cuda.synchronize()
            sps = 20 / (time.perf_counter() - t0)
            # cout fixe de l'evaluation : la grille dense est parcourue 2x (raw + EMA)
            t0 = time.perf_counter()
            evaluate(model, val_sets, channels, img_size, is_ref)
            ev = 2 * (time.perf_counter() - t0)
            est = args.steps / sps + ev
            total += est
            print(f"{config_name(cfg):<44}{n_params:>12,}  {sps:5.2f}  "
                  f"{args.steps/sps/60:5.1f}m  {ev/60:5.1f}m  {est/60:5.1f}m", flush=True)
            del model, opt; gc.collect(); torch.cuda.empty_cache()
        print(f"\nTOTAL estime : {total/60:.1f} min ({total/3600:.2f} h) pour "
              f"{args.steps:,} steps x {len(cfgs)} config(s), loss={args.loss} "
              f"coupling={args.coupling}, eval sur {len(eval_t)} niveaux de bruit.",
              flush=True)
        return

    results = []
    t_all = time.perf_counter()
    for i, cfg in enumerate(cfgs, 1):
        name = config_name(cfg)
        print(f"\n[{i}/{len(cfgs)}] {name}", flush=True)
        model = None
        try:
            model = build_model(cfg, device, channels, img_size)
            r = train_probe(model, name, x_train, val_sets,
                            os.path.join(args.results_dir, name), args,
                            channels, img_size, cfg["arch"] == "unet_ref", t_list)
            r["nmse"] = {t: v / mse_mean for t, v in r["mse"].items()}
            # moyenne sur les SEULS t d'entrainement : c'est la quantite optimisee,
            # la grille dense sert a tracer la courbe, pas a classer.
            r["nmse_mean"] = sum(r["nmse"][t] for t in t_list) / len(t_list)
            results.append(r)
            with open(os.path.join(args.results_dir, name, "metrics.txt"), "w") as f:
                f.write(f"name={name}\nn_params={r['n_params']}\nsteps={args.steps}\n"
                        f"loss={args.loss}\ncoupling={args.coupling}\n"
                        f"train_t={t_list}\nselected={r['used']}\nit_s={r['it_s']:.3f}\n"
                        f"train_time_s={r['time_s']:.1f}\nnmse_mean={r['nmse_mean']:.5f}\n\n"
                        f"t\tmse\tnmse\tpsnr\tvu_a_l_entrainement\n")
                for t in sorted(r["mse"]):
                    f.write(f"{t:g}\t{r['mse'][t]:.6f}\t{r['nmse'][t]:.5f}\t"
                            f"{10*torch.log10(torch.tensor(4.0/r['mse'][t])).item():.2f}\t"
                            f"{int(t in set(t_list))}\n")
            done = time.perf_counter() - t_all
            print(f"  -> nmse moyenne {r['nmse_mean']:.4f} | " +
                  " ".join(f"t={t:g}:{r['nmse'][t]:.3f}" for t in t_list) +
                  f" | {r['time_s']/60:.1f} min | reste ~{done/i*(len(cfgs)-i)/60:.0f} min",
                  flush=True)
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"  [ECHEC] {name} : {exc}", flush=True)
        finally:
            model = None; gc.collect(); torch.cuda.empty_cache()

    if not results:
        print("Aucune config n'a abouti.", flush=True)
        return

    results.sort(key=lambda r: r["nmse_mean"])
    ts = sorted(t_list)                     # table = t d'entrainement (lisible)
    header = (f"{'config':<44}{'params':>12}{'nmse':>8}  " +
              "".join(f"{'t='+format(t,'g'):>9}" for t in ts) + f"{'min':>8}")
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(f"{r['name']:<44}{r['n_params']:>12,}{r['nmse_mean']:>8.4f}  " +
                     "".join(f"{r['nmse'][t]:>9.3f}" for t in ts) +
                     f"{r['time_s']/60:>8.1f}")
    lines.append("-" * len(header))
    lines.append(f"{'[ref] copie de x_t':<44}{'':>12}{'':>8}  " +
                 "".join(f"{copy_ref[t]/mse_mean:>9.3f}" for t in ts))
    lines.append(f"{'[ref] predicteur constant (moyenne)':<44}{'':>12}{'':>8}  " +
                 "".join(f"{1.0:>9.3f}" for t in ts))
    table = "\n".join(lines)
    print("\n" + "=" * len(header) + f"\nClassement (nmse = mse / var(donnees), plus bas = mieux)"
          f"\n{'='*len(header)}\n{table}", flush=True)

    with open(os.path.join(args.results_dir, "summary.txt"), "w") as f:
        f.write(f"steps={args.steps} batch={args.batch_size} lr={args.lr} "
                f"loss={args.loss} coupling={args.coupling} train_t={t_list} "
                f"n_val={args.n_val} var_donnees={mse_mean:.6f}\n\n{table}\n")
    # le CSV porte la COURBE complete (grille dense) : de quoi retracer/comparer
    with open(os.path.join(args.results_dir, "summary.csv"), "w") as f:
        f.write("name,n_params,it_s,time_s,selected,nmse_mean," +
                ",".join(f"nmse_t{t:g}" for t in eval_t) + "\n")
        for r in results:
            f.write(f"{r['name']},{r['n_params']},{r['it_s']:.3f},{r['time_s']:.1f},"
                    f"{r['used']},{r['nmse_mean']:.6f}," +
                    ",".join(f"{r['nmse'][t]:.6f}" for t in eval_t) + "\n")
        f.write("[ref]copie_de_xt,,,,," +
                ",".join(f"{copy_ref[t]/mse_mean:.6f}" for t in eval_t) + "\n")
    plot_summary(results, mse_mean, copy_ref,
                 os.path.join(args.results_dir, "mse_vs_t.png"), t_list)
    print(f"\nSorties -> {args.results_dir}/ (summary.txt, summary.csv, mse_vs_t.png, "
          f"<config>/grid.png)", flush=True)


if __name__ == "__main__":
    main()
