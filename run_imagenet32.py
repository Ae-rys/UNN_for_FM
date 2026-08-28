# -*- coding: utf-8 -*-
"""
run_imagenet32.py
Flow Matching sur ImageNet-32 (3 canaux, 32x32) avec ConvScCP_UNN.

Variante RGB du run MNIST pixel qui marche (cf. confirm_convsccp_pixel.py) :
    ConvScCP_UNN, espace PIXEL, primal 3 canaux, dual `ic` canaux,
    conv 3 -> ic (noyau 9x9, pad 4), prox l1 (use_Unet="l1"),
    x-pred + loss ponderee espace-v 1/(1-t)^2, LNO, couplage independant.
    w_bias : casse la parite impaire de x_t
    (sinon gen(-x0) = -gen(x0) -> inversion couleur, cf. note oddsym MNIST).

Script auto-suffisant (ne touche pas train.py / run_mnist.py, cales sur MNIST
28x28 mono). Boucle d'entrainement + generation calquees sur le run MNIST.

Donnees
-------
    --dataset imagenet32 --data-dir DIR
        DIR contient les batchs ImageNet-32 downsampled (Chrabaszcz et al.) :
        fichiers pickle `train_data_batch_1` .. `train_data_batch_10`
        (chacun un dict {'data': uint8 (N,3072) plans R|G|B, 'labels': ...}).
    --dataset cifar10   (defaut, telechargeable)
        Fallback 3x32x32 pour tester le pipeline immediatement sans ImageNet.

Usage
-----
    python run_imagenet32.py --dataset cifar10 --epochs 5
    python run_imagenet32.py --dataset imagenet32 --data-dir /path/to/imagenet32 --epochs 20
    python run_imagenet32.py --only ConvScCP     # filtre par sous-chaine du nom
"""

import argparse
import gc
import glob
import os
import pickle
import time

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchdyn.core import NeuralODE
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.utils import torch_wrapper
from torchcfm.models.unet import UNetModel

from models.architectures import ConvScCP_UNN

IMG_SIZE = 32
CHANNELS = 3
DIM = CHANNELS * IMG_SIZE * IMG_SIZE          # 3072


# ---------------------------------------------------------------------------
# Experiment registry (meme structure que run_mnist.build_experiments)
# ---------------------------------------------------------------------------

def build_experiments(device):
    return [
        # ---- Baseline generatif RGB (v-pred), comme UNet_torchCFM_baseline MNIST ----
        dict(
            name  = "UNet_torchCFM_baseline",
            build = lambda: UNetModel(dim=(CHANNELS, IMG_SIZE, IMG_SIZE),
                                      num_channels=64, num_res_blocks=2).to(device),
        ),
        # ---- La variante : ConvScCP pixel RGB (recette MNIST gagnante) ----
        dict(
            name  = "ConvScCP_UNN_L1_LNO_rgb",
            build = lambda: ConvScCP_UNN(
                dim=DIM, K=10, internal_channel=256,
                in_channels=CHANNELS, img_size=IMG_SIZE,
                use_Unet="l1", version="LNO", use_checkpoint=True,
                w_bias=True,
            ).to(device),
        ),
    ]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_imagenet32(data_dir):
    """Charge les batchs ImageNet-32 downsampled -> tenseur (N,3,32,32) dans [-1,1]."""
    files = sorted(glob.glob(os.path.join(data_dir, "train_data_batch_*")))
    if not files:
        raise FileNotFoundError(
            f"Aucun 'train_data_batch_*' dans {data_dir}. "
            f"Telecharge ImageNet-32 downsampled (Chrabaszcz et al.) ou utilise "
            f"--dataset cifar10 pour tester le pipeline.")
    xs = []
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        data = np.asarray(d["data"], dtype=np.uint8)          # (N, 3072), plans R|G|B
        xs.append(data.reshape(-1, CHANNELS, IMG_SIZE, IMG_SIZE))
    x = np.concatenate(xs, axis=0)
    x = torch.from_numpy(x).float().div_(255.0).sub_(0.5).div_(0.5)   # -> [-1, 1]
    print(f"ImageNet-32 : {x.shape[0]} images depuis {len(files)} batch(s).")
    return TensorDataset(x, torch.zeros(x.shape[0], dtype=torch.long))


def get_train_loader(dataset, data_dir, batch_size=128):
    if dataset == "imagenet32":
        ds = _load_imagenet32(data_dir)
    elif dataset == "cifar10":
        # Cache local (uint8 (N,3,32,32)) construit depuis le parquet HuggingFace :
        # le miroir toronto de torchvision rampe (~9 Ko/s). Repli torchvision si absent.
        cache = "./data/cifar10_train_rgb.pt"
        if os.path.exists(cache):
            d = torch.load(cache)
            x = d["data"].float().div_(127.5).sub_(1.0)       # uint8 [0,255] -> [-1,1]
            ds = TensorDataset(x, d["labels"])
            print(f"CIFAR-10 (cache {cache}) : {len(ds)} images.")
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),                        # [0,1], (3,32,32)
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            ds = torchvision.datasets.CIFAR10(root="./data", train=True,
                                              download=True, transform=transform)
            print(f"CIFAR-10 (torchvision) : {len(ds)} images.")
    else:
        raise ValueError(dataset)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=4, pin_memory=True, drop_last=True)


# ---------------------------------------------------------------------------
# Visualisation RGB
# ---------------------------------------------------------------------------

def plot_images(images, title, save_path, n=8):
    """Grille de n images RGB. images : (>=n, 3*32*32) ou (>=n,3,32,32) dans [-1,1]."""
    imgs = images.detach().cpu().view(-1, CHANNELS, IMG_SIZE, IMG_SIZE)[:n]
    imgs = (imgs * 0.5 + 0.5).clamp(0, 1)                     # [-1,1] -> [0,1]
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.4))
    fig.suptitle(title, fontsize=9)
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i].permute(1, 2, 0).numpy())
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generation (ODE dopri5 de t=0 a t=1)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, name, device, n=8, seed=0):
    model.eval()
    torch.manual_seed(seed)
    if name == "UNet_torchCFM_baseline":
        x0 = torch.randn(n, CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
        class _W(torch.nn.Module):
            def __init__(s, m): super().__init__(); s.m = m
            def forward(s, t, x, **kw): return s.m(t.expand(x.shape[0]), x)
        node = NeuralODE(_W(model), solver="dopri5", atol=1e-5, rtol=1e-5)
    else:
        x0 = torch.randn(n, DIM, device=device)
        node = NeuralODE(torch_wrapper(model), solver="dopri5", atol=1e-5, rtol=1e-5)
    traj = node.trajectory(x0, t_span=torch.linspace(0, 1, 2, device=device))
    model.train()
    return traj[-1]


# ---------------------------------------------------------------------------
# Train (recette MNIST gagnante : x-pred + loss v-space, couplage independant)
# ---------------------------------------------------------------------------

def train_one(model, name, train_loader, device, run_dir, nb_epochs, lr):
    os.makedirs(run_dir, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}\nModel : {name}\nParams: {n_params:,}\n{'='*60}")
    with open(os.path.join(run_dir, "params.txt"), "w") as f:
        f.write(f"{n_params}\n")

    FM        = ConditionalFlowMatcher(sigma=0.1)             # couplage INDEP (pas OT : casse ScCP)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    is_unet   = (name == "UNet_torchCFM_baseline")
    predicts_x1 = getattr(model, "predicts_x1", False)

    loss_history = []
    train_start  = time.perf_counter()
    for epoch in range(nb_epochs):
        model.train()
        total, t0 = 0.0, time.perf_counter()
        for x1_img, _ in train_loader:
            x1_img = x1_img.to(device)
            bs = x1_img.shape[0]
            x1 = x1_img if is_unet else x1_img.view(bs, -1)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

            if is_unet:
                out  = model(t, xt)
                loss = torch.mean((out - ut) ** 2)
            else:
                xt_t = torch.cat([xt, t.view(bs, 1)], dim=-1)
                out  = model(xt_t)
                if predicts_x1:
                    # loss CFM exacte = MSE(out, x1) ponderee 1/(1-t)^2 (espace vitesse)
                    loss = torch.mean((out - x1) ** 2 / torch.clamp((1 - t.view(-1, 1)) ** 2, min=0.05))
                else:
                    loss = torch.mean((out - ut) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()

        avg = total / len(train_loader)
        loss_history.append(avg)
        print(f"  Epoch {epoch+1}/{nb_epochs} — loss: {avg:.4f}  ({time.perf_counter()-t0:.1f}s)")

        if epoch % 5 == 0 or epoch == nb_epochs - 1:
            imgs = generate(model, name, device)
            plot_images(imgs, f"{name} — epoch {epoch+1}",
                        os.path.join(run_dir, f"epoch_{epoch+1}.png"))

    plt.figure(); plt.plot(range(1, nb_epochs + 1), loss_history, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("FM loss"); plt.title(f"Training loss — {name}")
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "loss.png")); plt.close()
    with open(os.path.join(run_dir, "loss.txt"), "w") as f:
        for ep, l in enumerate(loss_history, 1):
            f.write(f"{ep}\t{l:.6f}\n")

    return loss_history, n_params, time.perf_counter() - train_start


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Flow Matching UNN sur ImageNet-32 (RGB 32x32).")
    p.add_argument("--dataset",     type=str, default="cifar10", choices=["imagenet32", "cifar10"])
    p.add_argument("--data-dir",    type=str, default="./data/imagenet32")
    p.add_argument("--epochs",      type=int, default=5)
    p.add_argument("--results-dir", type=str, default="results_imagenet32")
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch-size",  type=int, default=128)
    p.add_argument("--only",        type=str, default="")
    p.add_argument("--skip",        type=str, default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  dataset: {args.dataset}")

    os.makedirs(args.results_dir, exist_ok=True)
    train_loader = get_train_loader(args.dataset, args.data_dir, batch_size=args.batch_size)

    experiments = build_experiments(device)
    if args.only:
        experiments = [e for e in experiments if args.only in e["name"]]
    if args.skip:
        experiments = [e for e in experiments if args.skip not in e["name"]]
    print(f"\n{len(experiments)} experiment(s) a lancer.\n")

    summary = []
    for i, exp in enumerate(experiments, 1):
        name = exp["name"]
        print(f"\n[{i}/{len(experiments)}] {name}")
        model = None
        try:
            model = exp["build"]()
            losses, n_params, dt = train_one(
                model, name, train_loader, device,
                run_dir=os.path.join(args.results_dir, name),
                nb_epochs=args.epochs, lr=args.lr,
            )
            summary.append((name, losses[-1], n_params, dt, "OK"))
        except Exception as exc:
            import traceback; traceback.print_exc()
            summary.append((name, float("nan"), 0, 0.0, str(exc)))
        finally:
            model = None
            gc.collect()
            torch.cuda.empty_cache()

    print("\n" + "=" * 60 + "\nSummary\n" + "=" * 60)
    for name, final_loss, n_params, dt, status in summary:
        print(f"  {name:<40} params={n_params:>11,}  final_loss={final_loss:.4f}  time={dt:6.0f}s  [{status}]")
    with open(os.path.join(args.results_dir, "summary.txt"), "w") as f:
        f.write("model_name\tn_params\tfinal_loss\ttrain_time_s\tstatus\n")
        for name, final_loss, n_params, dt, status in summary:
            f.write(f"{name}\t{n_params}\t{final_loss:.6f}\t{dt:.1f}\t{status}\n")
    print(f"\nSummary -> {os.path.join(args.results_dir, 'summary.txt')}")


if __name__ == "__main__":
    main()
