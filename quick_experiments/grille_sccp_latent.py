"""Grille K x internal_channel pour le ScCP latent, dans l'esprit de
results/make_grille.py (lignes = K, colonnes = internal_channel), mais
auto-contenue : chaque cellule est ENTRAINEE ici puis ses echantillons generes
sont affiches dans la grille.

Contrairement a make_grille (qui relit des dossiers deja produits), ce script
entraine tout lui-meme. Seul LatentScCP_UNN a des K/internal_channel (SmallUNet
a base_ch fixe), donc le balayage porte sur le ScCP.

Sorties dans results/grille_sccp_latent/ :
  - grille_samples.png : grille K x ic, chaque cellule = n_cell echantillons generes
  - heatmap.png        : deux heatmaps (nettete = energie de gradient, diversite = std_gen) sur K x ic
  - summary.txt        : params / loss finale / std_gen / grad_energy par (K, ic)

Exemples :
  python grille_sccp_latent.py --K 10,20,30 --ic 64,128,256 --variant vpred --epochs 30
  python grille_sccp_latent.py --variant xpred_xloss --coupling ot     # voir ou le collapse apparait
"""
import argparse
import os
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

from models.architectures import LatentScCP_UNN, SmallUNetLatentV2, SmallUNet
from ae_diag import load_or_train_ae

parser = argparse.ArgumentParser(description="Grille K x internal_channel du ScCP latent (style make_grille).")
parser.add_argument("--K", type=str, default="10,20,30", help="valeurs de K (lignes), separees par des virgules")
parser.add_argument("--ic", type=str, default="64,128,256", help="valeurs de internal_channel (colonnes)")
parser.add_argument("--variant", type=str, default="vpred",
                    choices=["vpred", "xpred_xloss", "xpred_vloss"], help="parametrisation / espace de loss")
parser.add_argument("--coupling", type=str, default="ot", choices=["ot", "indep"],
                    help="couplage FM : ot (Exact OT) ou indep (aleatoire)")
parser.add_argument("--space", type=str, default="latent", choices=["latent", "pixel"],
                    help="latent (c_lat=16, sp=7, + AE) ou pixel (c_lat=1, sp=28, sans AE)")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--digit", type=int, default=0)
parser.add_argument("--n-cell", type=int, default=4, help="nb d'echantillons generes affiches par cellule")
parser.add_argument("--w-bias", action="store_true", help="(i) biais convolutif appris sur W -> non-impair en fct de l'entree")
args = parser.parse_args()

K_values  = [int(x) for x in args.K.split(",")]
ic_values = [int(x) for x in args.ic.split(",")]
# variant -> (predicts_x1, vloss_weight)
PARAM = {"vpred": (False, False), "xpred_xloss": (True, False), "xpred_vloss": (True, True)}
predicts_x1, vloss_weight = PARAM[args.variant]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | variant={args.variant} coupling={args.coupling} "
      f"K={K_values} ic={ic_values} epochs={args.epochs}")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
idx = torch.where(dataset.targets == args.digit)[0]
train_loader = DataLoader(Subset(dataset, idx), batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
print(f"Nb images (digit={args.digit}): {len(idx)}")

run_dir = f"results/grille_sccp_{args.space}_{args.variant}"   # namespace par variante -> pas d'ecrasement
os.makedirs(run_dir, exist_ok=True)

# ---- espace de travail : latent (AE, c_lat=16, sp=7) ou pixel (c_lat=1, sp=28) ----
is_latent = args.space == "latent"
if is_latent:
    ae = load_or_train_ae(train_loader, device, c_lat=16, epochs=500)
    c_lat, Hl = ae.c_lat, ae.latent_spatial
else:
    ae = None
    c_lat, Hl = 1, 28                                   # ScCP direct en espace pixel 28x28
lat_dim = c_lat * Hl * Hl

FM = (ExactOptimalTransportConditionalFlowMatcher(sigma=0.1) if args.coupling == "ot"
      else ConditionalFlowMatcher(sigma=0.1))


def encode_x1(x_img):
    """x1 cible : latent encode (gelé) ou image brute aplatie (pixel)."""
    if is_latent:
        with torch.no_grad():
            return ae.encode(x_img).flatten(1)
    return x_img.flatten(1)


def decode(z):
    """z (B, lat_dim) -> image (B,1,28,28) : decodage AE (latent) ou reshape (pixel)."""
    if is_latent:
        return ae.decode(z.view(-1, c_lat, Hl, Hl))
    return z.view(-1, 1, 28, 28)


def grad_energy(imgs):
    """Energie de gradient moyenne : proxy de nettete (bas = flou, ~reel = net, haut = bruite)."""
    gx = (imgs[..., 1:, :] - imgs[..., :-1, :]).abs().mean()
    gy = (imgs[..., :, 1:] - imgs[..., :, :-1]).abs().mean()
    return (gx + gy).item()


def train_and_gen(model):
    """Entraine un modele FM latent/pixel quelconque (ScCP ou baseline SmallUNet) et
    genere n_cell echantillons au meme bruit. Lit predicts_x1/vloss_weight du modele."""
    m_pred = getattr(model, "predicts_x1", False)
    m_vw = getattr(model, "vloss_weight", False)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_hist = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x_img, _ in train_loader:
            x1_img = x_img.to(device).view(-1, 1, 28, 28)
            x1 = encode_x1(x1_img)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            xt_t = torch.cat([xt, t.view(x1.size(0), 1)], dim=-1)
            out = model(xt_t)
            if m_pred:
                err2 = (out - x1) ** 2
                if m_vw:
                    w = 1.0 / torch.clamp((1 - t.view(x1.size(0), 1)) ** 2, min=0.05 ** 2)
                    loss = torch.mean(w * err2)
                else:
                    loss = torch.mean(err2)
            else:
                loss = torch.mean((out - ut) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
        loss_hist.append(total / len(train_loader))
    last_loss = loss_hist[-1] if loss_hist else float("nan")
    # ---- generation : MEME bruit pour toutes les cellules (comparabilite) ----
    model.eval()
    torch.manual_seed(0)
    x0_test = torch.randn(args.n_cell, lat_dim, device=device)
    with torch.no_grad():
        node = NeuralODE(torch_wrapper(model), solver="dopri5", atol=1e-5, rtol=1e-5)
        z_final = node.trajectory(x0_test, t_span=torch.linspace(0, 1, 2, device=device))[-1]
        std_gen = z_final.std(dim=0).mean().item()
        imgs = decode(z_final).cpu()   # (n_cell,1,28,28)
    ge = grad_energy(imgs)
    # montage horizontal des n_cell echantillons -> une seule image par cellule
    montage = torch.cat([imgs[i, 0] for i in range(args.n_cell)], dim=1).numpy()   # (28, n_cell*28)
    del model
    torch.cuda.empty_cache()
    return dict(params=n_params, loss=last_loss, loss_hist=loss_hist,
                std_gen=std_gen, ge=ge, montage=montage)


def build_baseline(ic):
    """Baseline SmallUNet, meme mode (predicts_x1/vloss) que la variante ScCP."""
    if is_latent:
        return SmallUNetLatentV2(c_lat=c_lat, latent_spatial=Hl, base_ch=ic,
                                 predicts_x1=predicts_x1, vloss_weight=vloss_weight).to(device)
    return SmallUNet(in_channels=1, out_channels=1, base_ch=ic).to(device)


baseline_name = "SmallUNetV2" if is_latent else "SmallUNet"

# ---- balayage ScCP ----
cells = {}
for K in K_values:
    for ic in ic_values:
        print(f"\n=== ScCP K={K}, ic={ic} ===")
        model = LatentScCP_UNN(c_lat=c_lat, latent_spatial=Hl, K=K, internal_channel=ic,
                               use_Unet="l1", version="LNO", use_checkpoint=True,
                               predicts_x1=predicts_x1, vloss_weight=vloss_weight,
                               w_bias=args.w_bias).to(device)
        cells[(K, ic)] = train_and_gen(model)
        c = cells[(K, ic)]
        print(f"  params={c['params']:,}  loss={c['loss']:.4f}  std_gen={c['std_gen']:.4f}  GE={c['ge']:.4f}")

# ---- ligne baseline SmallUNet (base_ch = ic de la colonne) ----
base_cells = {}
for ic in ic_values:
    print(f"\n=== {baseline_name} baseline base_ch={ic} ===")
    base_cells[ic] = train_and_gen(build_baseline(ic))
    c = base_cells[ic]
    print(f"  params={c['params']:,}  loss={c['loss']:.4f}  std_gen={c['std_gen']:.4f}  GE={c['ge']:.4f}")

# ---- reference de nettete : vraies donnees ----
real = next(iter(train_loader))[0][:args.n_cell].view(-1, 1, 28, 28)
ge_real = grad_energy(real)

title = (f"ScCP {args.space} — {args.variant} / couplage={args.coupling} / {args.epochs} ep "
         f"+ baseline {baseline_name} — GE_reel={ge_real:.3f}")

# =========== grille d'echantillons (lignes = K + 1 ligne baseline, colonnes = ic) ===========
nrows = len(K_values) + 1
fig, axes = plt.subplots(nrows, len(ic_values),
                         figsize=(3.0 * len(ic_values), 1.6 * nrows), squeeze=False)
fig.suptitle(title, fontsize=13, y=0.99)
for i, K in enumerate(K_values):
    for j, ic in enumerate(ic_values):
        ax = axes[i, j]
        c = cells[(K, ic)]
        ax.imshow(c["montage"], cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"GE={c['ge']:.2f} std={c['std_gen']:.2f}", fontsize=8)
        if i == 0:
            ax.set_title(f"ic={ic}", fontsize=11)
        if j == 0:
            ax.set_ylabel(f"K={K}", fontsize=11)
for j, ic in enumerate(ic_values):                          # ligne baseline
    ax = axes[-1, j]
    c = base_cells[ic]
    ax.imshow(c["montage"], cmap="gray", vmin=-1, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"GE={c['ge']:.2f} std={c['std_gen']:.2f}", fontsize=8)
    if j == 0:
        ax.set_ylabel(f"{baseline_name}\n(base_ch=ic)", fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(run_dir, "grille_samples.png"), dpi=130)
plt.close(fig)

# =========== heatmaps : nettete (GE) et diversite (std_gen) ===========
import numpy as np
ge_grid  = np.array([[cells[(K, ic)]["ge"]      for ic in ic_values] for K in K_values]
                    + [[base_cells[ic]["ge"]      for ic in ic_values]])
std_grid = np.array([[cells[(K, ic)]["std_gen"] for ic in ic_values] for K in K_values]
                    + [[base_cells[ic]["std_gen"] for ic in ic_values]])
ylabels = [f"K={K}" for K in K_values] + [baseline_name]
fig, axs = plt.subplots(1, 2, figsize=(5.5 * 2, 4.8))
for ax, grid, name in [(axs[0], ge_grid, f"nettete GE (reel={ge_real:.3f})"),
                       (axs[1], std_grid, "diversite std_gen")]:
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(ic_values))); ax.set_xticklabels(ic_values)
    ax.set_yticks(range(len(ylabels)));   ax.set_yticklabels(ylabels)
    ax.set_xlabel("internal_channel / base_ch"); ax.set_title(name, fontsize=11)
    for i in range(len(ylabels)):
        for j in range(len(ic_values)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle(title, fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(run_dir, "heatmap.png"), dpi=130)
plt.close(fig)

# =========== courbes de loss (une ligne par cellule ScCP + baselines) ===========
fig, ax = plt.subplots(figsize=(8, 5))
eps = range(1, args.epochs + 1)
for K in K_values:
    for ic in ic_values:
        ax.plot(eps, cells[(K, ic)]["loss_hist"], label=f"ScCP K={K} ic={ic}", linewidth=1.2)
for ic in ic_values:
    ax.plot(eps, base_cells[ic]["loss_hist"], "--", label=f"{baseline_name} bc={ic}", linewidth=1.2)
ax.set_xlabel("epoch"); ax.set_ylabel("loss FM (moyenne)"); ax.set_yscale("log")
ax.set_title(title, fontsize=10)
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "loss_curves.png"), dpi=130)
plt.close(fig)

# =========== donnees brutes des courbes (format long : re-traçable a volonte) ===========
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
            f.write(f"{baseline_name}\t\t{ic}\t{c['params']}\t{e}\t{l:.6f}\n")

# =========== summary ===========
with open(os.path.join(run_dir, "summary.txt"), "w") as f:
    f.write(f"# variant={args.variant} coupling={args.coupling} epochs={args.epochs} "
            f"batch={args.batch} digit={args.digit} GE_reel={ge_real:.4f}\n")
    for K in K_values:
        for ic in ic_values:
            c = cells[(K, ic)]
            f.write(f"ScCP\tK={K}\tic={ic}\tparams={c['params']}\tfinal_loss={c['loss']:.6f}\t"
                    f"std_gen={c['std_gen']:.4f}\tgrad_energy={c['ge']:.4f}\n")
    for ic in ic_values:
        c = base_cells[ic]
        f.write(f"{baseline_name}\tbase_ch={ic}\tparams={c['params']}\tfinal_loss={c['loss']:.6f}\t"
                f"std_gen={c['std_gen']:.4f}\tgrad_energy={c['ge']:.4f}\n")

print(f"\nResultats dans {run_dir}/ (grille_samples.png, heatmap.png, loss_curves.png, loss_data.tsv, summary.txt)")
