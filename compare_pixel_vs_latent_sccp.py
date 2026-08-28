"""Test : le flou des zeros ScCP (results/temp/ConvScCP_UNN_L1_LNO) venait-il de
la CAPACITE / l'espace PIXEL, ou de "1 seul canal" ? Contrôle propre en réutilisant
la MEME itération ScCP (LatentScCP_UNN) en trois configs, toutes en v-pred (comme
le run flou d'origine), digit=0 :

  pixel_repro    : c_lat=1, sp=28  (espace pixel), ~70k params  -> reproduit la TAILLE du run flou (64k)
  pixel_big      : c_lat=1, sp=28  (espace pixel), ~277k params -> 4x la capacite, MEME espace
  latent_matched : c_lat=16, sp=7  (espace latent+decodeur AE), ~258k params -> params ~egaux a pixel_big

Deux comparaisons :
  (A) pixel_repro vs pixel_big      : la capacite defloue-t-elle en espace pixel ?
  (B) pixel_big  vs latent_matched  : a params ~egaux, le latent/decodeur apporte-t-il de la nettete ?

Metrique de nettete objective : energie de gradient moyenne |grad image| (les images floues
ont une faible energie de gradient). Comparee a l'energie des vraies donnees.
"""
import argparse
import os
import torch
import torch.nn.functional as F
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

from models.architectures import LatentScCP_UNN
from ae_diag import load_or_train_ae

parser = argparse.ArgumentParser(description="ScCP pixel (repro/gros) vs latent, a params ~egaux, v-pred, digit=0.")
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--batch", type=int, default=256, help="batch train (pixel ic=512 est lourd en memoire)")
parser.add_argument("--coupling", type=str, default="indep", choices=["indep", "ot"],
                    help="indep (ConditionalFlowMatcher, comme le run flou d'origine) ou ot "
                         "(Exact OT — deconseille en 784-dim, detruit le champ)")
parser.add_argument("--only", type=str, default="")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
idx = torch.where(dataset.targets == 0)[0]
train_loader = DataLoader(Subset(dataset, idx), batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
print(f"Nb images (digit=0): {len(idx)}")

run_dir = "results/compare_pixel_vs_latent_sccp"
os.makedirs(run_dir, exist_ok=True)

# AE partage (le meme que les autres scripts latents) pour la config latente
ae = load_or_train_ae(train_loader, device, c_lat=16, epochs=500)
c_lat_ae, Hl_ae = ae.c_lat, ae.latent_spatial

# --- 3 modeles, tous v-pred (predicts_x1=False) ---
def build(c_lat, sp, ic):
    return LatentScCP_UNN(c_lat=c_lat, latent_spatial=sp, K=30, internal_channel=ic,
                          use_Unet="l1", version="LNO", use_checkpoint=True,
                          predicts_x1=False).to(device)

specs = {   # name -> (c_lat, spatial, internal_channel, is_latent)
    "pixel_repro":    (1,  28, 128, False),
    "pixel_big":      (1,  28, 512, False),
    "latent_matched": (16, 7,  56,  True),
}
if args.only:
    specs = {k: v for k, v in specs.items() if args.only in k}

models, is_latent, dims = {}, {}, {}
for name, (cl, sp, ic, lat) in specs.items():
    models[name] = build(cl, sp, ic)
    is_latent[name] = lat
    dims[name] = cl * sp * sp
    print(f"{name}: {sum(p.numel() for p in models[name].parameters()):,} params  (dim={dims[name]}, {'latent' if lat else 'pixel'})")

FM = (ConditionalFlowMatcher(sigma=0.1) if args.coupling == "indep"
      else ExactOptimalTransportConditionalFlowMatcher(sigma=0.1))
print(f"Couplage FM : {args.coupling}")


def encode_x1(name, x_img):
    """x1 cible : image brute aplatie (pixel) ou latent encode (latent)."""
    if is_latent[name]:
        with torch.no_grad():
            return ae.encode(x_img).flatten(1)
    return x_img.flatten(1)


def decode(name, z):
    """z (B, dim) -> image (B,1,28,28)."""
    if is_latent[name]:
        return ae.decode(z.view(-1, c_lat_ae, Hl_ae, Hl_ae))
    return z.view(-1, 1, 28, 28)


def grad_energy(imgs):
    """Energie de gradient moyenne |di/dx|+|di/dy| : proxy de nettete (bas = flou)."""
    gx = (imgs[..., 1:, :] - imgs[..., :-1, :]).abs().mean()
    gy = (imgs[..., :, 1:] - imgs[..., :, :-1]).abs().mean()
    return (gx + gy).item()


loss_histories = {name: [] for name in models}
for name, model in models.items():
    print(f"\n{'='*60}\nTraining {name}\n{'='*60}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x_img, _ in train_loader:
            x_img = x_img.to(device).view(-1, 1, 28, 28)
            x1 = encode_x1(name, x_img)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            xt_t = torch.cat([xt, t.view(x1.size(0), 1)], dim=-1)
            out = model(xt_t)
            loss = torch.mean((out - ut) ** 2)      # v-pred
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
        loss_histories[name].append(total / len(train_loader))
        print(f"  [{name}] epoch {epoch+1}/{args.epochs}  loss={loss_histories[name][-1]:.4f}")
    torch.cuda.empty_cache()

# ---- generation : meme bruit par modele (dim propre a chaque espace) ----
n_samples = 8
fig, axes = plt.subplots(len(models) + 1, n_samples, figsize=(2 * n_samples, 2 * (len(models) + 1)))

# ligne 0 : vraies donnees (reference de nettete)
real = next(iter(train_loader))[0][:n_samples].to(device).view(-1, 1, 28, 28)
ge_real = grad_energy(real)
for col in range(n_samples):
    axes[0, col].imshow(real[col, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
    axes[0, col].axis("off")
axes[0, 0].set_ylabel(f"REEL\nGE={ge_real:.3f}", fontsize=9, rotation=0, labelpad=40)

sharpness = {"REEL": ge_real}
for row, (name, model) in enumerate(models.items(), start=1):
    model.eval()
    torch.manual_seed(0)
    x0_test = torch.randn(n_samples, dims[name], device=device)
    with torch.no_grad():
        node = NeuralODE(torch_wrapper(model), solver="dopri5", atol=1e-5, rtol=1e-5)
        traj = node.trajectory(x0_test, t_span=torch.linspace(0, 1, 2, device=device))
        imgs = decode(name, traj[-1]).cpu()
    ge = grad_energy(imgs)
    sharpness[name] = ge
    for col in range(n_samples):
        axes[row, col].imshow(imgs[col, 0], cmap="gray", vmin=-1, vmax=1)
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(f"{name}\nGE={ge:.3f}", fontsize=9, rotation=0, labelpad=45)
    print(f"{name}: energie de gradient (nettete) = {ge:.4f}  (reel={ge_real:.4f})")

plt.suptitle("ScCP pixel (repro/gros) vs latent, params ~egaux, v-pred, digit=0 — GE = energie de gradient (nettete)")
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "pixel_vs_latent_sharpness.png"), dpi=100)
plt.close()

with open(os.path.join(run_dir, "summary.txt"), "w") as f:
    f.write(f"# epochs={args.epochs} batch={args.batch}  (GE=energie de gradient, proxy nettete ; reel={ge_real:.4f})\n")
    for name in models:
        n = sum(p.numel() for p in models[name].parameters())
        f.write(f"{name}\tparams={n}\tdim={dims[name]}\tfinal_loss={loss_histories[name][-1]:.6f}\tgrad_energy={sharpness[name]:.4f}\n")

print(f"\nResultats dans {run_dir}/")
