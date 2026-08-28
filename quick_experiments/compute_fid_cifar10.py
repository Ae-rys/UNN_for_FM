# -*- coding: utf-8 -*-
"""
compute_fid_cifar10.py
FID-50k sur CIFAR-10, calcule EXACTEMENT comme le papier torchcfm
(examples/images/cifar10/compute_fid.py) pour que nos chiffres soient
comparables au FID-50k 4.80 publie par Tong et al. 2023.

Protocole (a ne pas bricoler, sinon les chiffres ne veulent plus rien dire)
--------------------------------------------------------------------------
    clean-fid, mode="legacy_tensorflow", dataset_name="cifar10",
    dataset_res=32, dataset_split="train", num_gen=50000
    conversion image : (x * 127.5 + 128).clip(0, 255).uint8
    echantillonnage  : Euler 100 steps  -> le 4.80 du papier
                       (Euler 1000 -> 3.92 ; dopri5 -> 3.82)

torchmetrics NE convient PAS ici : poids Inception et resize differents
=> chiffre non comparable au 4.80. D'ou clean-fid en legacy_tensorflow.

Le premier appel telecharge les poids Inception + les stats de reference
CIFAR-10 (~100 Mo, une seule fois, necessite le reseau).

Modeles supportes
-----------------
    --pretrained otcfm|cfm|fm   les poids officiels 400k (baseline forte)
    --ckpt PATH                 un de nos checkpoints (latest.pt / ckpt_step_N.pt)
                                l'archi est deduite du champ "name" du checkpoint
    --sweep DIR                 tous les ckpt_step_*.pt d'un run -> courbe
                                FID vs budget (fid_vs_steps.png)

Par defaut on evalue les poids EMA : ce sont eux qui portent les FID publies.

Usage
-----
    # baseline officielle (doit retomber sur ~4.80)
    python compute_fid_cifar10.py --pretrained otcfm

    # un de nos runs
    python compute_fid_cifar10.py --ckpt results_cifar10_torchcfm_recipe/ConvScCP_.../latest.pt

    # courbe qualite vs budget sur les archives d'un run
    python compute_fid_cifar10.py --sweep results_cifar10_torchcfm_recipe/ConvScCP_...

    # test rapide du pipeline (NON comparable au 4.80 : FID biaise a la hausse)
    python compute_fid_cifar10.py --pretrained otcfm --num-gen 2000
"""

import argparse
import glob
import os
import re
import time

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cleanfid import fid
from torchdyn.core import NeuralODE
from torchcfm.models.unet import UNetModel

from models.architectures import ConvScCP_UNN
from sample_otcfm_pretrained import REF_CFG, MINE_CFG, download_weights

IMG_SIZE = 32
CHANNELS = 3
DIM = CHANNELS * IMG_SIZE * IMG_SIZE


# ---------------------------------------------------------------------------
# Reconstruction de l'archi depuis le nom du checkpoint
# ---------------------------------------------------------------------------

def build_from_name(name, device):
    """Deduit l'archi du champ 'name' du checkpoint. Retourne (model, is_unet)."""
    if name == "UNet_ref":
        return UNetModel(**REF_CFG).to(device), True
    if name == "UNet_torchCFM_baseline":
        return UNetModel(**MINE_CFG).to(device), True
    if name == "UNet_torchcfm_ch32":
        return UNetModel(dim=(CHANNELS, IMG_SIZE, IMG_SIZE), num_channels=32,
                         num_res_blocks=1, channel_mult=[1, 2, 2], num_heads=4,
                         num_head_channels=64, attention_resolutions="16",
                         dropout=0.1).to(device), True
    if name == "UNet_torchcfm_ch64":
        return UNetModel(dim=(CHANNELS, IMG_SIZE, IMG_SIZE), num_channels=64,
                         num_res_blocks=1, channel_mult=[1, 2, 2], num_heads=4,
                         num_head_channels=64, attention_resolutions="16",
                         dropout=0.1).to(device), True
    if name == "MinimalUNetFM_kamb":
        from models.architectures import MinimalUNetFM
        return MinimalUNetFM(dim=DIM, in_channels=CHANNELS,
                             img_size=IMG_SIZE).to(device), False

    m = re.match(r"ConvScCP_UNN_rgb_k(\d+)_K(\d+)_ic(\d+)_L1(c?)_(LFO|LNO)", name)
    if m:
        kernel, K, ic = int(m.group(1)), int(m.group(2)), int(m.group(3))
        prox_ch, version = (m.group(4) == "c"), m.group(5)
        model = ConvScCP_UNN(
            dim=DIM, K=K, internal_channel=ic, kernel_size=kernel,
            in_channels=CHANNELS, img_size=IMG_SIZE,
            use_Unet="l1", version=version, use_checkpoint=False, w_bias=True,
            prox_channels=prox_ch,
        ).to(device)
        return model, False

    raise ValueError(f"Archi non reconnue depuis name='{name}'. "
                     f"Ajoute une regle dans build_from_name().")


def load_our_ckpt(path, which, device):
    """Charge un de nos checkpoints (run_cifar10_torchcfm_recipe / run_imagenet32)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    name = ckpt.get("name")
    if name is None:
        raise KeyError(f"Pas de champ 'name' dans {path} — archi indeduisible.")
    model, is_unet = build_from_name(name, device)

    key = which if which in ckpt else "state_dict"
    if which not in ckpt:
        print(f"  [warn] '{which}' absent du checkpoint, repli sur 'state_dict' "
              f"(FID moins bon : les poids EMA sont ceux qui comptent)", flush=True)
    model.load_state_dict(ckpt[key], strict=True)
    # t_max du run : None pour les ckpt d'avant le passage a la loss sans clamp
    # (le modele garde alors le clamp d'eval min=0.05 -- comportement historique).
    model.t_max = ckpt.get("t_max", None)
    model.eval()

    step = ckpt.get("step", ckpt.get("epoch", "?"))
    n = sum(p.numel() for p in model.parameters())
    print(f"  {name} | step {step} | {n/1e6:.2f}M params | poids: {key} | "
          f"t_max: {model.t_max}", flush=True)
    return model, is_unet, name, step


def load_pretrained_ref(variant, out_dir, which, device):
    path = download_weights(variant, out_dir)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = UNetModel(**REF_CFG).to(device)
    model.load_state_dict(ckpt[which], strict=True)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"  {variant} officiel | step {ckpt.get('step','?')} | {n/1e6:.2f}M params | "
          f"poids: {which}", flush=True)
    return model, True, f"{variant}_pretrained", ckpt.get("step", 400000)


# ---------------------------------------------------------------------------
# Echantillonnage
# ---------------------------------------------------------------------------

class _VelocityWrapper(torch.nn.Module):
    def __init__(self, model, is_unet):
        super().__init__()
        self.model = model
        self.is_unet = is_unet

    def forward(self, t, x, *a, **kw):
        if self.is_unet:
            return self.model(t.expand(x.shape[0]), x)
        # ConvScCP : entree aplatie [x_t, t] ; en eval le modele renvoie la VITESSE
        # (cf. architectures.py : "training renvoie x~x1, eval renvoie (x-z)/(1-t)").
        b = x.shape[0]
        flat = x.view(b, -1)
        xt_t = torch.cat([flat, t.expand(b).view(b, 1)], dim=-1)
        return self.model(xt_t).view_as(x)


@torch.no_grad()
def sample_batch(vf, n, device, solver="euler", steps=100, t_max=None):
    """t_max : borne du domaine d'entrainement du modele (1 - 1/N du run). La grille
    d'Euler y est tronquee et le dernier pas va jusqu'a t=1 d'un coup — pour un modele
    x-pred c'est exactement emettre x1_pred, sa sortie native. Sans ca, echantillonner
    un run t_max=0.95 en Euler-100 evaluerait t=0.99, hors du domaine appris (et
    l'assert de fm_velocity_denom saute). t_max=None : ancien comportement."""
    x = torch.randn(n, CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    if solver == "euler":
        grid = [i / steps for i in range(steps)]
        if t_max is not None:
            grid = [t for t in grid if t <= t_max + 1e-9]
        for i, t in enumerate(grid):
            t_next = grid[i + 1] if i + 1 < len(grid) else 1.0
            x = x + vf(torch.full((1,), t, device=device), x) * (t_next - t)
        return x
    if t_max is not None:
        raise ValueError(
            f"dopri5 evalue des t arbitrairement proches de 1, hors du domaine "
            f"[0, {t_max}] de ce modele. Utilise --solver euler.")
    node = NeuralODE(vf, solver="dopri5", atol=1e-5, rtol=1e-5)
    return node.trajectory(x, t_span=torch.linspace(0, 1, 2, device=device))[-1]


def make_gen(vf, device, solver, steps, batch_size, total, t_max=None):
    """Callback pour cleanfid : ignore le latent z, renvoie un batch uint8."""
    state = {"done": 0, "t0": time.perf_counter()}

    @torch.no_grad()
    def gen(_z):
        x = sample_batch(vf, batch_size, device, solver=solver, steps=steps,
                         t_max=t_max)
        # conversion exacte du repo torchcfm (127.5 / +128, pas +127.5)
        img = (x * 127.5 + 128).clip(0, 255).to(torch.uint8)
        state["done"] += batch_size
        el = time.perf_counter() - state["t0"]
        rate = state["done"] / el
        if state["done"] % (batch_size * 10) == 0 or state["done"] >= total:
            print(f"    {state['done']:>6,}/{total:,} images  {rate:.0f} img/s  "
                  f"ETA {(total-state['done'])/rate/60:.1f} min", flush=True)
        return img

    return gen


def compute(model, is_unet, device, num_gen, batch_size, solver, steps):
    vf = _VelocityWrapper(model, is_unet).to(device).eval()
    gen = make_gen(vf, device, solver, steps, batch_size, num_gen,
                   t_max=getattr(model, "t_max", None))
    t0 = time.perf_counter()
    score = fid.compute_fid(
        gen=gen,
        dataset_name="cifar10",
        dataset_res=32,
        dataset_split="train",
        mode="legacy_tensorflow",     # le mode du papier — ne pas changer
        num_gen=num_gen,
        batch_size=batch_size,
        device=device,
    )
    print(f"  FID = {score:.3f}   ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="FID-50k CIFAR-10, protocole torchcfm.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pretrained", type=str, choices=["otcfm", "cfm", "fm"],
                     help="Poids officiels 400k (baseline forte).")
    src.add_argument("--ckpt", type=str, help="Un de nos checkpoints (.pt).")
    src.add_argument("--sweep", type=str,
                     help="Dossier de run : FID sur tous les ckpt_step_*.pt -> courbe.")
    p.add_argument("--weights", type=str, default="ema_model",
                   choices=["ema_model", "state_dict", "net_model"])
    p.add_argument("--num-gen", type=int, default=50000,
                   help="50000 = le protocole du papier. Moins => FID biaise a la hausse, "
                        "NON comparable au 4.80.")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--solver", type=str, default="euler", choices=["euler", "dopri5"])
    p.add_argument("--steps", type=int, default=100, help="Steps Euler (100 = recette 4.80).")
    p.add_argument("--out", type=str, default="", help="Fichier resultat (defaut: a cote du ckpt).")
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device)
    if args.num_gen < 50000:
        print(f"[!] num_gen={args.num_gen} < 50000 : FID biaise a la hausse, "
              f"NON comparable au 4.80 publie. Test de pipeline uniquement.\n", flush=True)

    print(f"Device: {device} | solver: {args.solver}-{args.steps} | num_gen: {args.num_gen:,}",
          flush=True)

    results = []      # (label, step, fid)

    if args.pretrained:
        out_dir = os.path.join("results_imagenet32", "OTCFM_pretrained_400k")
        which = "ema_model" if args.weights == "state_dict" else args.weights
        model, is_unet, label, step = load_pretrained_ref(args.pretrained, out_dir, which, device)
        score = compute(model, is_unet, device, args.num_gen, args.batch_size,
                        args.solver, args.steps)
        results.append((label, step, score))
        out_path = args.out or os.path.join(out_dir, f"fid_{args.pretrained}.txt")

    elif args.ckpt:
        model, is_unet, label, step = load_our_ckpt(args.ckpt, args.weights, device)
        score = compute(model, is_unet, device, args.num_gen, args.batch_size,
                        args.solver, args.steps)
        results.append((label, step, score))
        out_path = args.out or os.path.join(os.path.dirname(args.ckpt), "fid.txt")

    else:
        ckpts = sorted(glob.glob(os.path.join(args.sweep, "ckpt_step_*.pt")),
                       key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
        if not ckpts:
            raise FileNotFoundError(f"Aucun ckpt_step_*.pt dans {args.sweep}")
        print(f"Sweep : {len(ckpts)} checkpoints\n", flush=True)
        for c in ckpts:
            print(f"[{os.path.basename(c)}]", flush=True)
            model, is_unet, label, step = load_our_ckpt(c, args.weights, device)
            score = compute(model, is_unet, device, args.num_gen, args.batch_size,
                            args.solver, args.steps)
            results.append((label, step, score))
            del model
            torch.cuda.empty_cache()

        steps_x = [r[1] for r in results]
        fids_y = [r[2] for r in results]
        plt.figure(figsize=(6, 4))
        plt.plot(steps_x, fids_y, marker="o", label=results[0][0])
        plt.axhline(4.80, ls="--", c="crimson", lw=1,
                    label="OT-CFM 400k (papier) = 4.80")
        plt.xlabel("Training steps"); plt.ylabel(f"FID-{args.num_gen//1000}k")
        plt.title("Qualite vs budget d'entrainement")
        plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
        fig_path = os.path.join(args.sweep, "fid_vs_steps.png")
        plt.savefig(fig_path, dpi=110); plt.close()
        print(f"\n  -> {fig_path}", flush=True)
        out_path = args.out or os.path.join(args.sweep, "fid.txt")

    with open(out_path, "w") as f:
        f.write(f"# clean-fid legacy_tensorflow | cifar10 train | num_gen={args.num_gen} "
                f"| {args.solver}-{args.steps} | weights={args.weights}\n")
        f.write("label\tstep\tfid\n")
        for label, step, score in results:
            f.write(f"{label}\t{step}\t{score:.4f}\n")

    print(f"\nResultats -> {out_path}", flush=True)
    for label, step, score in results:
        print(f"  {label:<32} step={step:<8} FID={score:.3f}", flush=True)


if __name__ == "__main__":
    main()
