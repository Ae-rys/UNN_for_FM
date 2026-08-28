# -*- coding: utf-8 -*-
"""
sample_otcfm_pretrained.py
Charge les poids OT-CFM officiels (Tong et al. 2023, 400k steps) et genere des
images, pour disposer d'une baseline forte sans depenser 60h de GPU.

Poids : release assets du repo atong01/conditional-flow-matching, tag 1.0.4.
    otcfm_* : couplage minibatch-OT   (FID-50k 4.80 Euler-100 / 3.82 dopri5)
    cfm_*   : couplage independant    (= notre --coupling indep)
    fm_*    : Lipman et al. original

L'archi est le UNet de guided-diffusion (OpenAI) tel quel : OT-CFM n'apporte
aucune archi propre, seulement le couplage. Il faut donc l'instancier avec LEUR
config exacte (voir REF_CFG) sinon load_state_dict casse sur les shapes.

Le checkpoint contient net_model ET ema_model : ce sont les poids EMA qui
donnent les FID publies, on charge ema_model par defaut.

Usage
-----
    python sample_otcfm_pretrained.py                      # otcfm, EMA, Euler-100
    python sample_otcfm_pretrained.py --variant cfm        # baseline couplage indep
    python sample_otcfm_pretrained.py --solver dopri5      # ODE adaptatif
    python sample_otcfm_pretrained.py --compare-mine       # vs notre UNet 11.8M
    python sample_otcfm_pretrained.py --weights net_model  # sans EMA (pour voir)

Sorties -> results_imagenet32/OTCFM_pretrained_400k/
"""

import argparse
import os
import time
import urllib.request

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchdyn.core import NeuralODE
from torchcfm.models.unet import UNetModel

IMG_SIZE = 32
CHANNELS = 3

BASE_URL = ("https://github.com/atong01/conditional-flow-matching/"
            "releases/download/1.0.4/{variant}_cifar10_weights_step_400000.pt")

# Config exacte de examples/images/cifar10/train_cifar10.py (repo torchcfm).
# Ne PAS toucher : elle doit matcher le state_dict au parametre pres.
REF_CFG = dict(
    dim=(CHANNELS, IMG_SIZE, IMG_SIZE),
    num_channels=128,
    num_res_blocks=2,
    channel_mult=[1, 2, 2, 2],
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="16",
    dropout=0.1,
)

# Notre baseline maison (run_imagenet32.build_experiments), pour --compare-mine.
MINE_CFG = dict(
    dim=(CHANNELS, IMG_SIZE, IMG_SIZE),
    num_channels=64,
    num_res_blocks=3,
)


# ---------------------------------------------------------------------------
# Poids
# ---------------------------------------------------------------------------

def download_weights(variant, out_dir):
    """Telecharge le checkpoint s'il n'est pas deja la. ~546 Mo (poids+optim+sched)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{variant}_cifar10_weights_step_400000.pt")
    if os.path.exists(path):
        print(f"Checkpoint deja present : {path} "
              f"({os.path.getsize(path)/1e6:.0f} Mo)", flush=True)
        return path

    url = BASE_URL.format(variant=variant)
    print(f"Telechargement {variant} -> {path}\n  {url}", flush=True)

    def _hook(blocks, bsize, total):
        if total > 0 and blocks % 200 == 0:
            pct = 100.0 * blocks * bsize / total
            print(f"  {min(pct,100):5.1f}%  ({blocks*bsize/1e6:.0f}/{total/1e6:.0f} Mo)",
                  flush=True)

    t0 = time.perf_counter()
    urllib.request.urlretrieve(url, path, reporthook=_hook)
    print(f"  OK en {time.perf_counter()-t0:.0f}s "
          f"({os.path.getsize(path)/1e6:.0f} Mo)", flush=True)
    return path


def load_pretrained(variant, out_dir, which="ema_model", device="cuda:0"):
    """Instancie le UNet de reference et y charge le state_dict demande."""
    path = download_weights(variant, out_dir)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if which not in ckpt:
        raise KeyError(f"'{which}' absent du checkpoint. Cles : {list(ckpt.keys())}")
    print(f"Checkpoint : step={ckpt.get('step','?')}  |  poids charges : {which}", flush=True)

    model = UNetModel(**REF_CFG).to(device)
    # strict=True volontaire : une cle manquante voudrait dire que REF_CFG a derive
    # du checkpoint, et on generait alors du bruit sans s'en rendre compte.
    model.load_state_dict(ckpt[which], strict=True)
    model.eval()

    n = sum(p.numel() for p in model.parameters())
    print(f"UNet de reference : {n:,} params ({n/1e6:.2f}M)", flush=True)
    return model, n


# ---------------------------------------------------------------------------
# Echantillonnage
# ---------------------------------------------------------------------------

class _VelocityWrapper(torch.nn.Module):
    """Adapte model(t, x) -> signature (t, x) attendue par NeuralODE."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(t.expand(x.shape[0]), x)


@torch.no_grad()
def sample_euler(model, x0, n_steps=100):
    """Euler explicite de t=0 a t=1. n_steps=100 -> le FID 4.80 du papier."""
    x = x0.clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        x = x + model(t, x) * dt
    return x


@torch.no_grad()
def sample_dopri5(model, x0, atol=1e-5, rtol=1e-5):
    """ODE adaptatif -> le FID 3.82 du papier (plus lent)."""
    node = NeuralODE(_VelocityWrapper(model), solver="dopri5", atol=atol, rtol=rtol)
    traj = node.trajectory(x0, t_span=torch.linspace(0, 1, 2, device=x0.device))
    return traj[-1]


@torch.no_grad()
def generate(model, n, device, solver="euler", steps=100, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x0 = torch.randn(n, CHANNELS, IMG_SIZE, IMG_SIZE, generator=g).to(device)
    return generate_from(model, x0, solver=solver, steps=steps)


@torch.no_grad()
def generate_from(model, x0, solver="euler", steps=100):
    t0 = time.perf_counter()
    x1 = sample_euler(model, x0, steps) if solver == "euler" else sample_dopri5(model, x0)
    print(f"  {solver} : {x0.shape[0]} images en {time.perf_counter()-t0:.1f}s", flush=True)
    return x1


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _to_img(x):
    """(B,3,32,32) dans [-1,1] -> (B,32,32,3) dans [0,1]."""
    return (x.detach().cpu() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()


def plot_grid(images, title, save_path, n=8):
    imgs = _to_img(images[:n])
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.4))
    fig.suptitle(title, fontsize=9)
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i])
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=90)
    plt.close(fig)
    print(f"  -> {save_path}", flush=True)


def plot_comparison(rows, save_path, n=8):
    """rows : liste de (label, images). Meme x0 par colonne -> comparaison honnete."""
    fig, axes = plt.subplots(len(rows), n, figsize=(1.7 * n, 1.9 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for r, (label, imgs) in enumerate(rows):
        arr = _to_img(imgs[:n])
        for c in range(n):
            axes[r, c].imshow(arr[c])
            axes[r, c].axis("off")
        axes[r, 0].set_ylabel(label)
        # set_ylabel est invisible avec axis("off") -> titre a gauche via text
        axes[r, 0].text(-0.08, 0.5, label, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {save_path}", flush=True)


# ---------------------------------------------------------------------------
# Notre baseline (pour --compare-mine)
# ---------------------------------------------------------------------------

def load_mine(results_dir, device):
    """Charge results_imagenet32/UNet_torchCFM_baseline/model.pt (notre run 50 epochs)."""
    path = os.path.join(results_dir, "UNet_torchCFM_baseline", "model.pt")
    if not os.path.exists(path):
        print(f"[skip] notre baseline introuvable : {path}", flush=True)
        return None, 0
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    model = UNetModel(**MINE_CFG).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Notre baseline : {n:,} params ({n/1e6:.2f}M), epoch {ckpt.get('epoch','?')}",
          flush=True)
    return model, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Echantillonne les poids OT-CFM officiels (400k steps).")
    p.add_argument("--variant", type=str, default="otcfm", choices=["otcfm", "cfm", "fm"],
                   help="otcfm=couplage OT, cfm=independant, fm=Lipman.")
    p.add_argument("--weights", type=str, default="ema_model",
                   choices=["ema_model", "net_model"],
                   help="ema_model = les poids qui donnent les FID publies.")
    p.add_argument("--results-dir", type=str, default="results_imagenet32")
    p.add_argument("--out-name", type=str, default="OTCFM_pretrained_400k")
    p.add_argument("--n", type=int, default=8, help="Nb d'images generees.")
    p.add_argument("--solver", type=str, default="euler", choices=["euler", "dopri5", "both"])
    p.add_argument("--steps", type=int, default=100, help="Steps Euler (100 = recette FID 4.80).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compare-mine", action="store_true",
                   help="Grille comparative avec notre UNet 11.8M, meme bruit initial.")
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = os.path.join(args.results_dir, args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Device: {device}  |  variant: {args.variant}  |  poids: {args.weights}", flush=True)

    model, n_ref = load_pretrained(args.variant, out_dir, which=args.weights, device=device)

    # Meme x0 pour toutes les variantes -> les differences viennent du modele, pas du bruit.
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    x0 = torch.randn(args.n, CHANNELS, IMG_SIZE, IMG_SIZE, generator=g).to(device)

    solvers = ["euler", "dopri5"] if args.solver == "both" else [args.solver]
    outputs = {}
    for s in solvers:
        print(f"\nEchantillonnage {s}...", flush=True)
        imgs = generate_from(model, x0, solver=s, steps=args.steps)
        outputs[s] = imgs
        tag = f"{args.steps}steps" if s == "euler" else "adaptive"
        plot_grid(imgs, f"{args.variant} pretrained 400k ({args.weights}) — {s} {tag}",
                  os.path.join(out_dir, f"samples_{args.variant}_{args.weights}_{s}.png"),
                  n=args.n)

    with open(os.path.join(out_dir, "params.txt"), "w") as f:
        f.write(f"{n_ref}\n")

    if args.compare_mine:
        print("\nComparaison avec notre baseline...", flush=True)
        mine, n_mine = load_mine(args.results_dir, device)
        rows = [(f"OT-CFM 400k\n{n_ref/1e6:.1f}M", outputs[solvers[0]])]
        if mine is not None:
            mine_imgs = generate_from(mine, x0, solver=solvers[0], steps=args.steps)
            rows.append((f"nous 50ep\n{n_mine/1e6:.1f}M", mine_imgs))
            del mine
            torch.cuda.empty_cache()
        plot_comparison(rows, os.path.join(out_dir, "comparison_pretrained_vs_mine.png"),
                        n=args.n)

    print(f"\nTermine. Sorties dans {out_dir}", flush=True)


if __name__ == "__main__":
    main()
