# -*- coding: utf-8 -*-
"""
run_cifar10_torchcfm_recipe.py
Variante de run_imagenet32.py alignee sur la recette officielle torchcfm
(examples/images/cifar10/train_cifar10.py), pour que nos chiffres soient
comparables au FID-50k 4.80 publie par Tong et al. 2023.

Ce que run_imagenet32.py fait differemment (et qui explique l'essentiel de
l'ecart de qualite) :

    | reglage        | run_imagenet32 | ICI (recette torchcfm) |
    |----------------|----------------|------------------------|
    | lr             | 1e-3           | 2e-4                   |
    | warmup         | aucun          | 5000 steps             |
    | EMA            | aucune         | 0.9999                 |
    | steps          | ~19.5k (50 ep) | 400k                   |
    | augmentation   | aucune         | random horizontal flip |
    | UNet           | 64ch / 3 blocs | 128ch / 2 blocs, drop 0.1 |
    | grad clip      | 1.0            | 1.0  (deja identique)  |
    | batch size     | 128            | 128  (deja identique)  |

Entrainement pilote en STEPS (pas en epochs) : c'est l'unite du papier, et ca
rend la comparaison budget-a-budget lisible.

Le ConvScCP peut tourner sous la MEME recette (--only ConvScCP) : c'est la
comparaison honnete "meme objectif, meme couplage, meme budget, backbone
different".

ATTENTION AU COUT : 400k steps sur le UNet de reference = plusieurs jours de
GPU. Voir l'ETA affiche apres les 100 premiers steps. Pour la baseline UNet
convergee, prefere les poids officiels (sample_otcfm_pretrained.py) plutot que
de refaire 400k steps ; ce script sert surtout a entrainer NOS modeles sous
leur recette.

Checkpoints / reprise
---------------------
    latest.pt        etat COMPLET (net, EMA, optimizer, scheduler, step, loss,
                     RNG), ecrase tous les --save-every. Ecriture atomique.
    ckpt_step_N.pt   archives numerotees, conservees, tous les --keep-every.

La reprise est AUTOMATIQUE : si latest.pt existe dans le run_dir, on repart de
son step. Relancer exactement la meme commande apres un crash suffit. Pour
repartir de zero, --no-resume (ou supprimer latest.pt).

Reprendre un run en augmentant le budget marche aussi : `--steps 200000` sur un
run arrete a 100k continue jusqu'a 200k (le warmup et l'EMA gardent leur etat).

Usage
-----
    # ETA + smoke-test rapide
    python run_cifar10_torchcfm_recipe.py --only UNet_ref --steps 200

    # notre ConvScCP sous la recette torchcfm, budget raisonnable
    python run_cifar10_torchcfm_recipe.py --only ConvScCP --steps 100000 --coupling ot

    # ... le GPU a ete preempte -> MEME commande, ca reprend tout seul
    python run_cifar10_torchcfm_recipe.py --only ConvScCP --steps 100000 --coupling ot

    # prolonger un run termine a 100k jusqu'a 200k
    python run_cifar10_torchcfm_recipe.py --only ConvScCP --steps 200000 --coupling ot

    # reproduction complete du papier (tres long)
    python run_cifar10_torchcfm_recipe.py --only UNet_ref --steps 400000

    # comparaison a budget egal entre les deux backbones
    python run_cifar10_torchcfm_recipe.py --steps 50000

Sorties -> results_cifar10_torchcfm_recipe/<nom>/
"""

import argparse
import copy
import gc
import os
import time

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
from torchcfm.models.unet import UNetModel

from models.architectures import ConvScCP_UNN, MinimalUNetFM, weights_init_kaiming
from run_imagenet32 import get_train_loader, plot_images, generate, IMG_SIZE, CHANNELS, DIM
from flops_utils import (
    flops_unet_model, flops_vector_model, update_param_file, write_train_time,
    write_velocity_flops,
)

# Config UNet exacte du repo torchcfm (identique a REF_CFG de
# sample_otcfm_pretrained.py -> les poids officiels y chargent en strict=True).
REF_UNET_CFG = dict(
    dim=(CHANNELS, IMG_SIZE, IMG_SIZE),
    num_channels=128,
    num_res_blocks=2,
    channel_mult=[1, 2, 2, 2],
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="16",
    dropout=0.1,
)

# Hyperparametres de train_cifar10.py
RECIPE = dict(lr=2e-4, warmup=5000, ema_decay=0.9999, grad_clip=1.0, batch_size=128)


# ---------------------------------------------------------------------------
# EMA — sans elle on perd typiquement 5-10 points de FID sur CIFAR-10.
# ---------------------------------------------------------------------------

class EMA:
    """Moyenne mobile exponentielle des poids (decay 0.9999, comme torchcfm)."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                s.copy_(m)          # buffers entiers (compteurs, etc.)

    def model(self):
        return self.shadow


def warmup_lambda(warmup):
    """LR lineaire de 0 a 1 sur `warmup` steps, puis constant (= torchcfm)."""
    return lambda step: min((step + 1) / warmup, 1.0)


# ---------------------------------------------------------------------------
# Checkpoints
#
# Deux familles de fichiers dans run_dir :
#   latest.pt         — etat COMPLET, ecrase a chaque --save-every. Sert a
#                       reprendre apres un crash / une preemption du GPU.
#   ckpt_step_N.pt    — archives numerotees, conservees, tous les --keep-every.
#                       Sert a comparer la qualite a differents budgets et a
#                       revenir en arriere si un run diverge.
#
# Ecriture atomique (tmp + os.replace) : un kill pendant torch.save laisserait
# sinon un fichier tronque, et on perdrait le run entier.
# ---------------------------------------------------------------------------

def save_checkpoint(path, name, step, model, ema, optimizer, sched, loss_log,
                    train_time_s=0.0, t_max=None):
    """Etat complet -> reprise bit-a-bit (aux batchs du loader pres, cf. resume).

    `train_time_s` est le temps d'entrainement CUMULE depuis le tout premier
    lancement du run (toutes sessions confondues) : le stocker ici est le seul
    moyen de le retrouver apres un crash / une preemption.
    """
    payload = {
        "name": name,
        "step": step,
        # borne du tirage de t : definit l'objectif (loss sans clamp) ET le domaine
        # ou le modele est valide. Absente = checkpoint d'avant ce changement.
        "t_max": t_max,
        "train_time_s": train_time_s,
        "state_dict": model.state_dict(),
        "ema_model": ema.model().state_dict(),
        "optim": optimizer.state_dict(),
        "sched": sched.state_dict(),
        "loss_log": loss_log,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)          # atomique sur le meme filesystem


def load_checkpoint(path, model, ema, optimizer, sched, device):
    """Restaure l'etat. Retourne (step, loss_log, train_time_s cumule, t_max)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    ema.shadow.load_state_dict(ckpt["ema_model"])
    optimizer.load_state_dict(ckpt["optim"])
    sched.load_state_dict(ckpt["sched"])
    if ckpt.get("torch_rng") is not None:
        torch.set_rng_state(ckpt["torch_rng"].cpu().to(torch.uint8))
    if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in ckpt["cuda_rng"]])
        except Exception as exc:
            # p.ex. nb de GPU visibles different d'un run a l'autre : pas bloquant
            print(f"    [warn] RNG CUDA non restaure ({exc})", flush=True)
    step = ckpt["step"]
    loss_log = ckpt.get("loss_log", [])
    # .get : les checkpoints ecrits avant l'ajout du suivi de temps n'ont pas la cle
    train_time_s = float(ckpt.get("train_time_s", 0.0) or 0.0)
    t_max = ckpt.get("t_max", None)      # None = ckpt d'avant le passage sans clamp
    print(f"    reprise depuis {path} au step {step:,} "
          f"({train_time_s/3600:.2f}h d'entrainement deja cumulees, "
          f"t_max={t_max})", flush=True)
    return step, loss_log, train_time_s, t_max


# ---------------------------------------------------------------------------
# Modeles
# ---------------------------------------------------------------------------

def build_experiments(device, K=10, ic=256, kernel=9, prox="l1"):
    """ATTENTION AUX NOMS. Trois branchements en dependent :
      - train_one       : `is_unet = name.startswith("UNet")` -> entree image (B,C,S,S)
                          et appel model(t, x) ; sinon entree aplatie (B, dim+1).
      - generate        : meme test, pour choisir le wrapper de l'EDO.
      - build_from_name : reconstruit l'archi depuis le champ 'name' du checkpoint,
                          donc TOUT nouveau nom doit y etre ajoute, sinon les
                          outils d'analyse (sample_checkpoint, denoise_grid_ckpt,
                          sampler_diagnosis, budget_matched) ne sauront pas le relire.
    D'ou `UNet_torchcfm_ch32` (UNetModel, entree image) et `MinimalUNetFM_kamb`
    (entree aplatie, x-pred comme le ScCP) : le prefixe n'est pas cosmetique.
    """
    return [
        dict(
            name="UNet_ref",
            build=lambda: UNetModel(**REF_UNET_CFG).to(device),
        ),
        # UNet torchcfm reduit a la capacite des ScCP (1.11M) : c'est LA baseline
        # a capacite comparable, celle qui gagne de 9.6 % sur le banc de debruitage.
        dict(
            name="UNet_torchcfm_ch32",
            build=lambda: UNetModel(
                dim=(CHANNELS, IMG_SIZE, IMG_SIZE), num_channels=32,
                num_res_blocks=1, channel_mult=[1, 2, 2], num_heads=4,
                num_head_channels=64, attention_resolutions="16", dropout=0.1,
            ).to(device),
        ),
        # Meme recette que ch32, capacite doublee (num_channels 32 -> 64) : sert de
        # baseline "UNet plus gros" a comparer au ScCP et au ch32 a donnees egales.
        # Tout le reste (res_blocks, channel_mult, attention, dropout) est inchange,
        # pour que l'ecart mesure soit bien celui de la largeur.
        dict(
            name="UNet_torchcfm_ch64",
            build=lambda: UNetModel(
                dim=(CHANNELS, IMG_SIZE, IMG_SIZE), num_channels=64,
                num_res_blocks=1, channel_mult=[1, 2, 2], num_heads=4,
                num_head_channels=64, attention_resolutions="16", dropout=0.1,
            ).to(device),
        ),
        # UNet minimal de Kamb & Ganguli : x-pred, donc MEME objectif que le ScCP
        # (loss ponderee en espace vitesse), et pas d'attention.
        dict(
            name="MinimalUNetFM_kamb",
            build=lambda: MinimalUNetFM(
                dim=DIM, in_channels=CHANNELS, img_size=IMG_SIZE,
            ).to(device),
        ),
        dict(
            # "L1c" dans le nom = rayon du prox appris par canal dual. Le suffixe
            # n'est pas decoratif : build_from_name s'en sert pour reconstruire
            # l'archi depuis un checkpoint.
            name=f"ConvScCP_UNN_rgb_k{kernel}_K{K}_ic{ic}_"
                 f"{'L1c' if prox == 'l1c' else 'L1'}_LFO",
            build=lambda: ConvScCP_UNN(
                dim=DIM, K=K, internal_channel=ic, kernel_size=kernel,
                in_channels=CHANNELS, img_size=IMG_SIZE,
                use_Unet="l1", version="LFO", use_checkpoint=True,
                w_bias=True, prox_channels=(prox == "l1c"),
            ).to(device),
        ),
    ]


# ---------------------------------------------------------------------------
# Echantillonneur Euler tronque a [0, t_max]
# ---------------------------------------------------------------------------

def t_max_for(n_steps):
    """Le plus grand t qu'un Euler-N sur [0,1] evalue jamais : 1 - 1/N.
    Euler-10 -> 0.90 | Euler-50 -> 0.98 | Euler-100 -> 0.99."""
    return 1.0 - 1.0 / n_steps


@torch.no_grad()
def euler_sample(model, is_unet, device, n=8, n_steps=50, t_max=None, seed=0):
    """Euler-N sur la grille 0, 1/N, ..., 1-1/N, tronquee a t <= t_max.

    Le DERNIER pas va jusqu'a t=1 d'un coup : pour un modele x-pred cela revient
    exactement a emettre x1_pred(x_t, t), la sortie native du reseau. Aucun t > t_max
    n'est donc jamais evalue -> l'assert de fm_velocity_denom ne peut pas sauter, et
    le modele n'est jamais interroge hors de son domaine d'entrainement.
    Avec t_max = 1 - 1/N la grille n'est pas tronquee du tout : c'est l'Euler-N usuel.
    """
    if t_max is None:
        t_max = t_max_for(n_steps)
    was_training = model.training
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    if is_unet:
        x = torch.randn(n, CHANNELS, IMG_SIZE, IMG_SIZE, generator=g).to(device)
    else:
        x = torch.randn(n, DIM, generator=g).to(device)

    grid = [i / n_steps for i in range(n_steps) if i / n_steps <= t_max + 1e-9]
    for i, t in enumerate(grid):
        t_next = grid[i + 1] if i + 1 < len(grid) else 1.0      # dernier pas -> x1
        tt = torch.full((n, 1), t, device=device)
        v = (model(tt.view(-1), x) if is_unet
             else model(torch.cat([x, tt], dim=-1)))
        x = x + v * (t_next - t)
    if was_training:
        model.train()
    return x.view(n, CHANNELS, IMG_SIZE, IMG_SIZE)


# ---------------------------------------------------------------------------
# Boucle d'entrainement (step-based)
# ---------------------------------------------------------------------------

def train_one(model, name, train_loader, device, run_dir, total_steps,
              coupling="indep", flip=True, sample_every=10000, save_every=10000,
              keep_every=50000, resume=True, self_cond_rate=0.0, n_steps=20,
              sampler=None, seed=0, t_max=None, allow_tmax_change=False,
              compile_model=False):
    """self_cond_rate : probabilite (par batch) de fournir au modele le dual u^(K) d'une
    passe FROIDE sans gradient sur x_{t-dt} de LA MEME paire (x0, x1) — le
    self-conditioning latent de RIN (arXiv:2212.11972 §2.3), cf. run_warmstart_mnist.py.
    0.0 (defaut) = comportement historique BIT POUR BIT (le dual froid est toujours
    passe, aucun tirage aleatoire supplementaire n'est consomme).
    n_steps : nb de pas d'Euler de l'inference — fixe dt = 1/n_steps, que l'entrainement
    doit voir. sampler : callable(model) -> images, pour les PNG de progression (defaut
    None = `euler_sample`, Euler-n_steps tronque a t_max).

    t_max : borne SUPERIEURE du tirage de t a l'entrainement (defaut 1 - 1/n_steps).
    C'est ce qui permet de SUPPRIMER le clamp de la loss x-pred. Rappel du probleme :
    pour un modele x-pred, ||x1p - x1||^2/(1-t)^2 == ||v_pred - ut||^2 (MSE vitesse),
    mais l'ancien `clamp((1-t)^2, min=0.05)` cassait cette identite des t > 0.7764 et
    rendait la loss journalisee ~15x trop basse (donc incomparable a celle d'un modele
    v-pred). Ici le clamp est remplace par un ASSERT : c'est le tirage t ~ U(0, t_max)
    qui borne le poids, a 1/(1-t_max)^2. Comme un Euler-N n'evalue jamais au-dela de
    1 - 1/N, prendre t_max = 1 - 1/N ne retire AUCUN t que l'echantillonneur verra.
        Euler-10  -> t_max 0.90, poids <=  100
        Euler-20  -> t_max 0.95, poids <=  400   (defaut : bonus, l'ancien clamp
                     d'eval min=0.05 sur (1-t) y est exactement inerte)
        Euler-50  -> t_max 0.98, poids <= 2500
        Euler-100 -> t_max 0.99, poids <= 10000
    Plus t_max est grand, plus la trajectoire est complete mais plus le gradient est
    domine par une poignee d'echantillons a t proche de 1 (le clamp etait justement
    la pour ca — le supprimer deplace le compromis, il ne le fait pas disparaitre).

    compile_model : torch.compile le forward d'entrainement UNIQUEMENT. Le handle
    compile est une VARIABLE SEPAREE (`net`) qui partage les parametres de `model` ;
    `model` reste le module eager et c'est LUI qui alimente l'EMA, le clip de gradient
    et `state_dict()`. Sans cette separation, torch.compile prefixe les cles du
    state_dict par `_orig_mod.` et les checkpoints deviennent illisibles par
    build_from_name()/load_our_ckpt() (et la reprise d'un run non compile casse).
    La branche self-conditioning reste eager : c'est celle qui fait planter inductor
    au backward sur le deroule ScCP (cf. bench_speedup_levers.py).

    Retourne (derniere loss, n_params, temps d'entrainement CUMULE du run). Le temps
    est cumule sur toutes les sessions (il transite par latest.pt), et il est reecrit
    dans run_dir/parametres.txt tous les 1000 steps et a chaque checkpoint — avec les
    FLOPs d'une evaluation de vitesse, mesures au demarrage."""
    os.makedirs(run_dir, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    is_unet = name.startswith("UNet")
    predicts_x1 = getattr(model, "predicts_x1", False)

    # t ~ U(0, t_max). Pose AVANT la construction de l'EMA : le shadow est un
    # deepcopy, il herite donc de t_max et son echantillonnage est coherent.
    if t_max is None:
        t_max = t_max_for(n_steps)
    if not 0.0 < t_max < 1.0:
        raise ValueError(f"t_max={t_max} doit etre dans ]0,1[ : a t=1 le poids "
                         f"1/(1-t)^2 de la loss x-pred est infini.")
    w_min = (1.0 - t_max) ** 2
    model.t_max = t_max          # supprime le clamp d'eval (cf. fm_velocity_denom)

    print(f"\n{'='*66}\nModel : {name}\nParams: {n_params:,} ({n_params/1e6:.2f}M)\n"
          f"Recette torchcfm : lr={RECIPE['lr']} warmup={RECIPE['warmup']} "
          f"ema={RECIPE['ema_decay']} flip={flip}\n"
          f"Steps : {total_steps:,}  |  coupling: {coupling}\n"
          f"t ~ U(0, {t_max:g})  (Euler-{n_steps} : le sampler n'evalue jamais "
          f"au-dela de {t_max:g})\n"
          + (f"loss v-pred = MSE vitesse ||v_pred - ut||^2\n"
             if not predicts_x1 else
             f"loss x-pred SANS clamp -> = MSE vitesse exacte ; poids max "
             f"1/(1-t_max)^2 = {1.0/w_min:,.0f}\n")
          + f"{'='*66}", flush=True)
    with open(os.path.join(run_dir, "params.txt"), "w") as f:
        f.write(f"{n_params}\n")

    # ---- parametres.txt : archi + FLOPs d'une evaluation de vitesse (batch 1) ----
    # update_param_file (et non une reecriture) : sur une reprise, le temps cumule
    # deja inscrit dans le fichier ne doit pas etre efface.
    update_param_file(
        run_dir,
        model_class=type(model).__name__,
        K=getattr(model, "K", None),
        dual_dim=getattr(model, "internal_channel", None),
        version=getattr(model, "version", None),
        n_params=n_params,
    )
    # t_max / euler_steps ne sont PAS ecrits ici : update_param_file fusionne les
    # cles sans les effacer, donc une tentative de reprise ensuite REFUSEE laisserait
    # dans parametres.txt un t_max que le run n'a jamais eu. On les ecrit une fois la
    # garde franchie (plus bas), quand ils decrivent vraiment ce qui va s'entrainer.
    write_velocity_flops(run_dir, (flops_unet_model(model, (CHANNELS, IMG_SIZE, IMG_SIZE), device)
                                   if is_unet else flops_vector_model(model, DIM, device)))

    FM = (ExactOptimalTransportConditionalFlowMatcher(sigma=0.0) if coupling == "ot"
          else ConditionalFlowMatcher(sigma=0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=RECIPE["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda(RECIPE["warmup"]))
    ema = EMA(model, decay=RECIPE["ema_decay"])

    # net : handle d'entrainement. Partage les parametres de `model` (torch.compile
    # ne copie rien), donc optimizer/EMA/checkpoints continuent de voir `model`.
    if compile_model:
        net = torch.compile(model)
        print("  torch.compile actif (premiers pas ~1-2 min de compilation)", flush=True)
    else:
        net = model

    loss_log = []          # (step, loss moyenne glissante)
    running, n_run = 0.0, 0
    step = 0
    # tirage du self-conditioning sur un generateur SEPARE : le flux aleatoire principal
    # (donnees, bruit x0, temps t) reste identique entre un run rate=0 et un run rate>0.
    coin = torch.Generator().manual_seed(seed + 1234)
    dt_sc = 1.0 / n_steps
    n_warm = 0

    # temps deja accumule par les sessions precedentes du meme run (0 si run neuf)
    elapsed_before = 0.0

    latest_path = os.path.join(run_dir, "latest.pt")
    if resume and os.path.exists(latest_path):
        step, loss_log, elapsed_before, ckpt_t_max = load_checkpoint(
            latest_path, model, ema, optimizer, sched, device)
        # Un run repris avec un autre t_max change d'objectif en cours de route : la
        # loss journalisee devient un melange de deux quantites. On refuse par defaut.
        if ckpt_t_max != t_max and not allow_tmax_change:
            raise ValueError(
                f"{latest_path} a ete entraine avec t_max={ckpt_t_max} "
                f"(None = ancienne loss avec clamp), or on demande t_max={t_max}. "
                f"Relance avec le meme --euler-steps, dans un run_dir neuf, ou passe "
                f"allow_tmax_change=True si tu assumes le melange.")
        model.t_max = t_max
        if step >= total_steps:
            print(f"    deja au budget ({step:,} >= {total_steps:,}) — rien a faire. "
                  f"Utilise --no-resume ou augmente --steps.", flush=True)
            write_train_time(run_dir, elapsed_before, steps=step)
            return (loss_log[-1][1] if loss_log else float("nan")), n_params, elapsed_before
    else:
        model.apply(weights_init_kaiming)

    t_start = time.perf_counter()
    t_win = time.perf_counter()

    def total_elapsed():
        """Temps d'entrainement cumule du run : sessions precedentes + celle-ci."""
        return elapsed_before + (time.perf_counter() - t_start)

    write_train_time(run_dir, total_elapsed(), steps=step)

    # garde de reprise franchie : t_max decrit maintenant l'entrainement qui demarre.
    update_param_file(run_dir, t_max=t_max, euler_steps=n_steps)

    model.train()
    while step < total_steps:
        for x1_img, _ in train_loader:
            if step >= total_steps:
                break
            x1_img = x1_img.to(device, non_blocking=True)

            if flip:
                # flip horizontal aleatoire par image (l'augmentation de torchcfm).
                # Fait sur GPU : marche pour le cache TensorDataset comme pour ImageNet-32,
                # qui n'ont pas de pipeline `transform`.
                mask = torch.rand(x1_img.shape[0], device=device) < 0.5
                x1_img[mask] = torch.flip(x1_img[mask], dims=[3])

            bs = x1_img.shape[0]
            x1 = x1_img if is_unet else x1_img.view(bs, -1)
            x0 = torch.randn_like(x1)
            # t ~ U(0, t_max) au lieu de U(0,1) : c'est CE tirage qui borne le poids
            # 1/(1-t)^2 de la loss x-pred et permet de se passer du clamp. Les t
            # retires (]t_max, 1[) ne sont jamais evalues par un Euler-n_steps.
            t_draw = torch.rand(bs, device=x1.device) * t_max
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t=t_draw)

            if is_unet:
                loss = torch.mean((net(t, xt) - ut) ** 2)
            else:
                xt_t = torch.cat([xt, t.view(bs, 1)], dim=-1)
                if self_cond_rate > 0.0:
                    # paire (x0, x1) REAPPARIEE apres le plan OT (exact a sigma=0) :
                    # indispensable pour que x_{t-dt} soit sur le MEME chemin conditionnel.
                    t_col = t.view(-1, 1)
                    x1_r = xt + ut * (1 - t_col)
                    x0_r = xt - ut * t_col
                    u_prev = model.cold_dual(xt)
                    if torch.rand((), generator=coin).item() < self_cond_rate:
                        t_prev = (t_col - dt_sc).clamp_min(0.0)
                        xt_prev = (1 - t_prev) * x0_r + t_prev * x1_r
                        with torch.no_grad():
                            _, u_prev = model(torch.cat([xt_prev, t_prev], dim=-1),
                                              u_init=model.cold_dual(xt_prev), return_u=True)
                        u_prev = u_prev.detach()
                        n_warm += 1
                    out, _ = model(xt_t, u_init=u_prev, return_u=True)
                else:
                    out = net(xt_t)
                if predicts_x1:
                    # cible x1 reconstruite depuis (xt, ut) : paire coherente meme sous OT
                    x1_tgt = xt + ut * (1 - t.view(-1, 1))
                    w = (1 - t.view(-1, 1)) ** 2
                    # PAS de clamp. Sans lui, ||x1p - x1||^2/(1-t)^2 EST la MSE
                    # vitesse ||v_pred - ut||^2 : la loss journalisee ici devient
                    # exactement celle de la branche v-pred ci-dessus, donc les deux
                    # familles de modeles sont enfin comparables. Ce qui borne le
                    # poids n'est plus un clamp mais le tirage t ~ U(0, t_max) ;
                    # l'assert est la pour que ca reste vrai si quelqu'un touche au
                    # tirage (il coute un sync GPU, negligeable a >100 ms/step).
                    assert w.min().item() >= w_min * (1 - 1e-6), (
                        f"poids 1/(1-t)^2 = {1.0/w.min().item():.1f} au-dela de la "
                        f"borne {1.0/w_min:.1f} imposee par t_max={t_max} : le tirage "
                        f"de t est sorti de [0, t_max].")
                    loss = torch.mean((out - x1_tgt) ** 2 / w)
                else:
                    loss = torch.mean((out - ut) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=RECIPE["grad_clip"])
            optimizer.step()
            sched.step()
            ema.update(model)

            running += loss.item()
            n_run += 1
            step += 1

            # ETA des les 100 premiers steps : on sait tout de suite si c'est jouable.
            if step == 100 or step % 1000 == 0:
                sps = n_run / (time.perf_counter() - t_win)
                eta_h = (total_steps - step) / sps / 3600
                print(f"  step {step:>7,}/{total_steps:,}  loss {running/n_run:.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {sps:.2f} it/s  "
                      f"ETA {eta_h:.1f}h", flush=True)
                loss_log.append((step, running / n_run))
                running, n_run = 0.0, 0
                # temps cumule rafraichi tous les 1000 steps : suivable pendant le run
                write_train_time(run_dir, total_elapsed(), steps=step)
                t_win = time.perf_counter()

            if step % sample_every == 0 or step == total_steps:
                # on echantillonne les poids EMA : ce sont eux qui portent les FID publies
                # Euler-n_steps tronque, PAS dopri5 : un solveur adaptatif evalue
                # des t arbitrairement proches de 1, donc hors du domaine [0, t_max].
                imgs = (sampler(ema.model()) if sampler is not None else
                        euler_sample(ema.model(), is_unet, device, n=8,
                                     n_steps=n_steps, t_max=t_max, seed=seed))
                plot_images(imgs, f"{name} — step {step} (EMA)",
                            os.path.join(run_dir, f"step_{step}.png"))
                model.train()

            if step % save_every == 0 or step == total_steps:
                save_checkpoint(latest_path, name, step, model, ema, optimizer,
                                sched, loss_log, train_time_s=total_elapsed(), t_max=t_max)
                write_train_time(run_dir, total_elapsed(), steps=step)
                print(f"    latest -> {latest_path} (step {step:,})", flush=True)

            if step % keep_every == 0 or step == total_steps:
                # archive numerotee : conservee, jamais ecrasee
                arch = os.path.join(run_dir, f"ckpt_step_{step}.pt")
                save_checkpoint(arch, name, step, model, ema, optimizer, sched, loss_log,
                                train_time_s=total_elapsed(), t_max=t_max)
                print(f"    archive -> {arch}", flush=True)

    dt = total_elapsed()
    write_train_time(run_dir, dt, steps=step)
    if self_cond_rate > 0.0:
        print(f"    self-conditioning : {n_warm:,} passes chaudes sur {step:,} steps "
              f"(taux vise {self_cond_rate})", flush=True)
    if loss_log:
        xs, ys = zip(*loss_log)
        plt.figure(); plt.plot(xs, ys)
        plt.xlabel("Step"); plt.ylabel("FM loss"); plt.title(f"Training loss — {name}")
        plt.tight_layout(); plt.savefig(os.path.join(run_dir, "loss.png")); plt.close()
        with open(os.path.join(run_dir, "loss.txt"), "w") as f:
            for s, l in loss_log:
                f.write(f"{s}\t{l:.6f}\n")

    return (loss_log[-1][1] if loss_log else float("nan")), n_params, dt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="CIFAR-10 Flow Matching sous la recette officielle torchcfm.")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenet32"])
    p.add_argument("--data-dir", type=str, default="./data/imagenet32")
    p.add_argument("--steps", type=int, default=400000, help="Steps totaux (papier : 400k).")
    p.add_argument("--results-dir", type=str, default="results_cifar10_torchcfm_recipe")
    p.add_argument("--batch-size", type=int, default=RECIPE["batch_size"])
    p.add_argument("--coupling", type=str, default="ot", choices=["indep", "ot"],
                   help="ot = la variante OT-CFM du papier (defaut ici).")
    p.add_argument("--no-flip", action="store_true", help="Desactive l'augmentation flip.")
    p.add_argument("--only", type=str, default="")
    p.add_argument("--skip", type=str, default="")
    p.add_argument("--kernel", type=int, default=9)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--ic", type=int, default=256)
    p.add_argument("--sample-every", type=int, default=10000)
    p.add_argument("--save-every", type=int, default=10000,
                   help="Frequence d'ecriture de latest.pt (reprise apres crash).")
    p.add_argument("--keep-every", type=int, default=50000,
                   help="Frequence des archives ckpt_step_N.pt (conservees).")
    p.add_argument("--no-resume", action="store_true",
                   help="Repart de zero meme si latest.pt existe (l'ecrase).")
    p.add_argument("--euler-steps", type=int, default=20,
                   help="Pas d'Euler de l'inference. Fixe aussi t_max = 1 - 1/N, la "
                        "borne du tirage de t a l'entrainement (Euler-20 -> 0.95).")
    p.add_argument("--t-max", type=float, default=None,
                   help="Force t_max au lieu de 1 - 1/euler-steps. Doit rester >= "
                        "1 - 1/euler-steps, sinon le sampler sort du domaine appris.")
    p.add_argument("--allow-tmax-change", action="store_true",
                   help="Autorise la reprise d'un run entraine avec un autre t_max "
                        "(melange deux objectifs dans la meme courbe de loss).")
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device)
    tm = args.t_max if args.t_max is not None else t_max_for(args.euler_steps)
    print(f"Device: {device}  |  dataset: {args.dataset}  |  coupling: {args.coupling}  "
          f"|  steps: {args.steps:,}  |  Euler-{args.euler_steps} -> t ~ U(0, {tm:g})",
          flush=True)

    os.makedirs(args.results_dir, exist_ok=True)
    train_loader = get_train_loader(args.dataset, args.data_dir, batch_size=args.batch_size)

    experiments = build_experiments(device, K=args.K, ic=args.ic, kernel=args.kernel)
    if args.only:
        experiments = [e for e in experiments if args.only in e["name"]]
    if args.skip:
        experiments = [e for e in experiments if args.skip not in e["name"]]
    print(f"\n{len(experiments)} experiment(s) a lancer.\n", flush=True)

    summary = []
    for i, exp in enumerate(experiments, 1):
        name = exp["name"]
        print(f"\n[{i}/{len(experiments)}] {name}", flush=True)
        model = None
        try:
            model = exp["build"]()
            final_loss, n_params, dt = train_one(
                model, name, train_loader, device,
                run_dir=os.path.join(args.results_dir, name),
                total_steps=args.steps, coupling=args.coupling,
                flip=not args.no_flip,
                sample_every=args.sample_every, save_every=args.save_every,
                keep_every=args.keep_every, resume=not args.no_resume,
                n_steps=args.euler_steps, t_max=args.t_max,
                allow_tmax_change=args.allow_tmax_change,
            )
            summary.append((name, final_loss, n_params, dt, "OK"))
        except Exception as exc:
            import traceback; traceback.print_exc()
            summary.append((name, float("nan"), 0, 0.0, str(exc)))
        finally:
            model = None
            gc.collect()
            torch.cuda.empty_cache()

    print("\n" + "=" * 66 + "\nSummary\n" + "=" * 66, flush=True)
    for name, fl, n, dt, st in summary:
        print(f"  {name:<40} params={n:>11,}  final_loss={fl:.4f}  "
              f"time={dt/3600:5.1f}h  [{st}]", flush=True)
    with open(os.path.join(args.results_dir, "summary.txt"), "w") as f:
        f.write("model_name\tn_params\tfinal_loss\ttrain_time_s\tstatus\n")
        for name, fl, n, dt, st in summary:
            f.write(f"{name}\t{n}\t{fl:.6f}\t{dt:.1f}\t{st}\n")


if __name__ == "__main__":
    main()
