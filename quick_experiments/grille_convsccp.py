"""Grille K x internal_channel (features) pour le ConvScCP_UNN IMAGE (ScCP "normal",
sans espace latent), avec une ligne BASELINE SmallUNet en bas. Chaque cellule est
entrainee ici (digit=0) et affiche ses zeros generes. Lignes = K, colonnes =
internal_channel (features) ; derniere ligne = SmallUNet(base_ch = ic de la colonne).

Respecte le mode du modele : si predicts_x1 (x-pred), loss x1 ponderee en espace v ;
sinon v-pred. Couplage independant par defaut (OT detruit la qualite en 784-dim).

Sorties dans results/grille_convsccp/ : grille_samples.png, heatmap.png, summary.txt

Exemple :
  python grille_convsccp.py --K 5,10,20 --ic 32,64,128 --epochs 30
"""
import argparse
import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchdyn.core import NeuralODE
from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher, ConditionalFlowMatcher)
from torchcfm.utils import torch_wrapper

from models.architectures import ConvScCP_UNN, SmallUNet

parser = argparse.ArgumentParser(description="Grille K x internal_channel du ConvScCP_UNN image + baseline SmallUNet.")
parser.add_argument("--K", type=str, default="5,10,20", help="valeurs de K (lignes)")
parser.add_argument("--ic", type=str, default="32,64,128", help="valeurs de internal_channel / features (colonnes)")
parser.add_argument("--version", type=str, default="LNO", choices=["LNO", "LFO"])
parser.add_argument("--coupling", type=str, default="indep", choices=["indep", "ot"])
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--digit", type=int, default=0)
parser.add_argument("--n-cell", type=int, default=4, help="nb d'echantillons generes par cellule")
parser.add_argument("--lr", type=float, default=1e-2, help="taux d'apprentissage")
parser.add_argument("--w-bias", action="store_true", help="(i) biais convolutif appris sur W -> non-impair en fct de l'entree")
args = parser.parse_args()

K_values  = [int(x) for x in args.K.split(",")]
ic_values = [int(x) for x in args.ic.split(",")]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | version={args.version} coupling={args.coupling} "
      f"K={K_values} ic={ic_values} epochs={args.epochs}")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
idx = torch.where(dataset.targets == args.digit)[0]
train_loader = DataLoader(Subset(dataset, idx), batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
print(f"Nb images (digit={args.digit}): {len(idx)}")

run_dir = "results/grille_convsccp"
os.makedirs(run_dir, exist_ok=True)

FM = (ConditionalFlowMatcher(sigma=0.1) if args.coupling == "indep"
      else ExactOptimalTransportConditionalFlowMatcher(sigma=0.1))


def grad_energy(imgs):
    gx = (imgs[..., 1:, :] - imgs[..., :-1, :]).abs().mean()
    gy = (imgs[..., :, 1:] - imgs[..., :, :-1]).abs().mean()
    return (gx + gy).item()


def train_and_gen(model):
    """Entraine un modele FM image quelconque (ScCP ou SmallUNet) et genere n_cell
    echantillons au meme bruit. Respecte model.predicts_x1 pour la loss."""
    predicts_x1 = getattr(model, "predicts_x1", False)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_hist = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x_img, _ in train_loader:
            x1 = x_img.to(device).view(-1, 784)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            xt_t = torch.cat([xt, t.view(x1.size(0), 1)], dim=-1)
            out = model(xt_t)
            if predicts_x1:                                   # x-pred : loss x1 en espace v
                w = 1.0 / torch.clamp((1 - t.view(-1, 1)) ** 2, min=0.05 ** 2)
                loss = torch.mean(w * (out - x1) ** 2)
            else:                                             # v-pred
                loss = torch.mean((out - ut) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
        loss_hist.append(total / len(train_loader))
    last_loss = loss_hist[-1] if loss_hist else float("nan")
    model.eval()
    torch.manual_seed(0)
    x0_test = torch.randn(args.n_cell, 784, device=device)
    with torch.no_grad():
        node = NeuralODE(torch_wrapper(model), solver="dopri5", atol=1e-5, rtol=1e-5)
        z_final = node.trajectory(x0_test, t_span=torch.linspace(0, 1, 2, device=device))[-1]
        std_gen = z_final.std(dim=0).mean().item()
        imgs = z_final.view(args.n_cell, 1, 28, 28).cpu()
    ge = grad_energy(imgs)
    montage = torch.cat([imgs[i, 0] for i in range(args.n_cell)], dim=1).numpy()   # (28, n_cell*28)
    del model
    torch.cuda.empty_cache()
    return dict(params=n_params, loss=last_loss, loss_hist=loss_hist,
                std_gen=std_gen, ge=ge, montage=montage, xpred=predicts_x1)


# ---- grille ScCP ----
cells = {}
for K in K_values:
    for ic in ic_values:
        print(f"\n=== ScCP K={K}, ic={ic} ===")
        model = ConvScCP_UNN(dim=784, K=K, internal_channel=ic, use_Unet="l1",
                             version=args.version, use_checkpoint=True,
                             w_bias=args.w_bias).to(device)
        cells[(K, ic)] = train_and_gen(model)
        c = cells[(K, ic)]
        print(f"  params={c['params']:,}  loss={c['loss']:.4f}  std_gen={c['std_gen']:.4f}  GE={c['ge']:.4f}")

# ---- ligne baseline SmallUNet (base_ch = ic de la colonne) ----
base_cells = {}
for ic in ic_values:
    print(f"\n=== SmallUNet baseline base_ch={ic} ===")
    model = SmallUNet(in_channels=1, out_channels=1, base_ch=ic).to(device)
    base_cells[ic] = train_and_gen(model)
    c = base_cells[ic]
    print(f"  params={c['params']:,}  loss={c['loss']:.4f}  std_gen={c['std_gen']:.4f}  GE={c['ge']:.4f}")

real = next(iter(train_loader))[0][:args.n_cell].view(-1, 1, 28, 28)
ge_real = grad_energy(real)
mode = "x-pred" if cells[(K_values[0], ic_values[0])]["xpred"] else "v-pred"
title = f"ConvScCP_UNN image ({mode}, {args.version}) + baseline SmallUNet — couplage={args.coupling} / {args.epochs} ep — GE_reel={ge_real:.3f}"

# ---- grille d'echantillons : lignes K + 1 ligne baseline ----
nrows = len(K_values) + 1
fig, axes = plt.subplots(nrows, len(ic_values),
                         figsize=(3.0 * len(ic_values), 1.6 * nrows), squeeze=False)
fig.suptitle(title, fontsize=13, y=0.99)
for i, K in enumerate(K_values):
    for j, ic in enumerate(ic_values):
        ax = axes[i, j]; c = cells[(K, ic)]
        ax.imshow(c["montage"], cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"GE={c['ge']:.2f} std={c['std_gen']:.2f}", fontsize=8)
        if i == 0:
            ax.set_title(f"ic={ic}", fontsize=11)
        if j == 0:
            ax.set_ylabel(f"K={K}", fontsize=11)
for j, ic in enumerate(ic_values):                          # ligne baseline
    ax = axes[-1, j]; c = base_cells[ic]
    ax.imshow(c["montage"], cmap="gray", vmin=-1, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"GE={c['ge']:.2f} std={c['std_gen']:.2f}", fontsize=8)
    if j == 0:
        ax.set_ylabel("SmallUNet\n(base_ch=ic)", fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(run_dir, "grille_samples.png"), dpi=130)
plt.close(fig)

# ---- heatmaps (ligne baseline "UNet" ajoutee en bas) ----
ge_grid  = np.array([[cells[(K, ic)]["ge"]      for ic in ic_values] for K in K_values]
                    + [[base_cells[ic]["ge"]      for ic in ic_values]])
std_grid = np.array([[cells[(K, ic)]["std_gen"] for ic in ic_values] for K in K_values]
                    + [[base_cells[ic]["std_gen"] for ic in ic_values]])
ylabels = [f"K={K}" for K in K_values] + ["UNet"]
fig, axs = plt.subplots(1, 2, figsize=(5.5 * 2, 4.8))
for ax, grid, name in [(axs[0], ge_grid, f"nettete GE (reel={ge_real:.3f})"),
                       (axs[1], std_grid, "diversite std_gen")]:
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(ic_values))); ax.set_xticklabels(ic_values)
    ax.set_yticks(range(len(ylabels)));   ax.set_yticklabels(ylabels)
    ax.set_xlabel("internal_channel / base_ch"); ax.set_title(name, fontsize=11)
    for i in range(len(ylabels)):
        for j in range(len(ic_values)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle(title, fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(run_dir, "heatmap.png"), dpi=130)
plt.close(fig)

# ---- courbes de loss (une ligne par cellule ScCP + baselines) ----
fig, ax = plt.subplots(figsize=(8, 5))
eps = range(1, args.epochs + 1)
for K in K_values:
    for ic in ic_values:
        ax.plot(eps, cells[(K, ic)]["loss_hist"], label=f"ScCP K={K} ic={ic}", linewidth=1.2)
for ic in ic_values:
    ax.plot(eps, base_cells[ic]["loss_hist"], "--", label=f"SmallUNet bc={ic}", linewidth=1.2)
ax.set_xlabel("epoch"); ax.set_ylabel("loss FM (moyenne)"); ax.set_yscale("log")
ax.set_title(title, fontsize=10)
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "loss_curves.png"), dpi=130)
plt.close(fig)

# ---- donnees brutes des courbes (format long : re-traçable a volonte) ----
with open(os.path.join(run_dir, "loss_data.tsv"), "w") as f:
    f.write("model\tK\tic\tparams\tepoch\tloss\n")
    for K in K_values:
        for ic in ic_values:
            c = cells[(K, ic)]
            for e, l in enumerate(c["loss_hist"], 1):
                f.write(f"ScCP\t{K}\t{ic}\t{c['params']}\t{e}\t{l:.6f}\n")
    for ic in ic_values:
        c = base_cells[ic]
        for e, l in enumerate(c["loss_hist"], 1):
            f.write(f"SmallUNet\t\t{ic}\t{c['params']}\t{e}\t{l:.6f}\n")

# ---- summary ----
with open(os.path.join(run_dir, "summary.txt"), "w") as f:
    f.write(f"# ConvScCP_UNN image {mode} + baseline SmallUNet, version={args.version} "
            f"coupling={args.coupling} epochs={args.epochs} batch={args.batch} digit={args.digit} "
            f"GE_reel={ge_real:.4f}\n")
    for K in K_values:
        for ic in ic_values:
            c = cells[(K, ic)]
            f.write(f"ScCP\tK={K}\tic={ic}\tparams={c['params']}\tfinal_loss={c['loss']:.6f}\t"
                    f"std_gen={c['std_gen']:.4f}\tgrad_energy={c['ge']:.4f}\n")
    for ic in ic_values:
        c = base_cells[ic]
        f.write(f"SmallUNet\tbase_ch={ic}\tparams={c['params']}\tfinal_loss={c['loss']:.6f}\t"
                f"std_gen={c['std_gen']:.4f}\tgrad_energy={c['ge']:.4f}\n")

print(f"\nResultats dans {run_dir}/ (grille_samples.png, heatmap.png, loss_curves.png, loss_data.tsv, summary.txt)")
