"""Confirmation : l'ANCIEN ConvScCP_UNN (avec lifting analyse/synthese) reproduit-il
les zeros la ou LatentScCP_UNN echouait ? Espace PIXEL (28x28, 1 canal), digit=0,
v-pred, couplage independant (comme le run flou d'origine). Si les zeros reviennent
-> le probleme etait bien la nouvelle archi (primal etrangle), pas ScCP ni le latent.
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
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.utils import torch_wrapper

from models.architectures import ConvScCP_UNN

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--K", type=int, default=10)
parser.add_argument("--ic", type=int, default=256)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
idx = torch.where(dataset.targets == 0)[0]
train_loader = DataLoader(Subset(dataset, idx), batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
print(f"Nb images (digit=0): {len(idx)}")

run_dir = "results/confirm_convsccp_pixel"
os.makedirs(run_dir, exist_ok=True)

# Modele d'ORIGINE (git HEAD), espace image 28x28
model = ConvScCP_UNN(dim=784, K=args.K, internal_channel=args.ic, use_Unet="l1",
                     version="LNO", use_checkpoint=True).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"ConvScCP_UNN (origine HEAD) K={args.K} ic={args.ic} : {n_params:,} params")

FM = ConditionalFlowMatcher(sigma=0.1)          # couplage independant


def grad_energy(imgs):
    gx = (imgs[..., 1:, :] - imgs[..., :-1, :]).abs().mean()
    gy = (imgs[..., :, 1:] - imgs[..., :, :-1]).abs().mean()
    return (gx + gy).item()


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(args.epochs):
    model.train()
    total = 0.0
    for x_img, _ in train_loader:
        x1 = x_img.to(device).view(-1, 784)
        x0 = torch.randn_like(x1)
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        xt_t = torch.cat([xt, t.view(x1.size(0), 1)], dim=-1)
        out = model(xt_t)                          # v-pred : sortie = vitesse
        loss = torch.mean((out - ut) ** 2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
    print(f"  epoch {epoch+1}/{args.epochs}  loss={total/len(train_loader):.4f}")

# ---- generation ----
n = 8
model.eval()
torch.manual_seed(0)
x0_test = torch.randn(n, 784, device=device)
with torch.no_grad():
    node = NeuralODE(torch_wrapper(model), solver="dopri5", atol=1e-5, rtol=1e-5)
    imgs = node.trajectory(x0_test, t_span=torch.linspace(0, 1, 2, device=device))[-1].view(n, 1, 28, 28).cpu()
real = next(iter(train_loader))[0][:n].view(n, 1, 28, 28)
ge, ge_real = grad_energy(imgs), grad_energy(real)
print(f"grad_energy genere={ge:.4f}  reel={ge_real:.4f}")

fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
for i in range(n):
    axes[0, i].imshow(real[i, 0], cmap="gray", vmin=-1, vmax=1); axes[0, i].axis("off")
    axes[1, i].imshow(imgs[i, 0], cmap="gray", vmin=-1, vmax=1); axes[1, i].axis("off")
axes[0, 0].set_ylabel("REEL", fontsize=10); axes[1, 0].set_ylabel("genere", fontsize=10)
fig.suptitle(f"ConvScCP_UNN (origine HEAD) pixel v-pred indep — K={args.K} ic={args.ic} "
             f"{n_params:,}p — GE gen={ge:.3f} reel={ge_real:.3f}")
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "samples.png"), dpi=110)
plt.close()
print(f"\nResultat dans {run_dir}/samples.png")
