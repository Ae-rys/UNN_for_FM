"""ConvScCP_UNN IMAGE (pas latent) sur plusieurs chiffres (defaut {0,1}).
But : le ScCP image sait-il modeliser une cible MULTIMODALE (plusieurs classes) ?
Genere des echantillons, compte combien de chaque classe (via un petit classifieur
0/1 entraine sur les vraies donnees), et signale les inversions couleur.

Correctif de l'impairite : --w-bias (biais convolutif appris sur W).
Couplage : --coupling indep (defaut) ou ot (rappel : OT collapse en haute dim).

Sortie : results/run_convsccp_multidigit/samples.png + resume console.
Exemple :
  python run_convsccp_multidigit.py --digits 0,1 --K 20 --ic 64 --epochs 30
  python run_convsccp_multidigit.py --digits 0,1 --w-bias --epochs 30
"""
import argparse
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchdyn.core import NeuralODE
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher)
from torchcfm.utils import torch_wrapper

from models.architectures import ConvScCP_UNN

parser = argparse.ArgumentParser()
parser.add_argument("--digits", type=str, default="0,1")
parser.add_argument("--K", type=int, default=20)
parser.add_argument("--ic", type=int, default=64)
parser.add_argument("--version", type=str, default="LNO", choices=["LNO", "LFO"])
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--coupling", type=str, default="indep", choices=["indep", "ot"])
parser.add_argument("--w-bias", action="store_true", help="(i) biais convolutif sur W")
parser.add_argument("--n", type=int, default=32, help="nb d'echantillons generes")
args = parser.parse_args()

digits = [int(d) for d in args.digits.split(",")]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | digits={digits} K={args.K} ic={args.ic} coupling={args.coupling} "
      f"w_bias={args.w_bias}")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
mask = torch.zeros(len(dataset.targets), dtype=torch.bool)
for d in digits:
    mask |= (dataset.targets == d)
idx = torch.where(mask)[0]
train_loader = DataLoader(Subset(dataset, idx), batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
print(f"Nb images (digits={digits}): {len(idx)}")

run_dir = "results/run_convsccp_multidigit"
os.makedirs(run_dir, exist_ok=True)

FM = (ConditionalFlowMatcher(sigma=0.1) if args.coupling == "indep"
      else ExactOptimalTransportConditionalFlowMatcher(sigma=0.1))


# ---- petit classifieur (compte les classes generees) ----
class TinyClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, n_classes),
        )

    def forward(self, x):
        return self.net(x)


label_lookup = torch.full((10,), -1, dtype=torch.long)
for i, d in enumerate(digits):
    label_lookup[d] = i
clf = TinyClassifier(len(digits)).to(device)
clf_opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
for _ in range(3):
    for x_img, y in train_loader:
        logits = clf(x_img.to(device))
        loss = F.cross_entropy(logits, label_lookup[y].to(device))
        clf_opt.zero_grad(); loss.backward(); clf_opt.step()
clf.eval()
print(f"[classifieur] {len(digits)} classes entraine")

# ---- ScCP image ----
sccp = ConvScCP_UNN(dim=784, K=args.K, internal_channel=args.ic, use_Unet="l1",
                    version=args.version, use_checkpoint=True,
                    w_bias=args.w_bias).to(device)
predicts_x1 = getattr(sccp, "predicts_x1", False)
print(f"ConvScCP_UNN image : {sum(p.numel() for p in sccp.parameters()):,} params  (predicts_x1={predicts_x1})")

opt = torch.optim.Adam(sccp.parameters(), lr=1e-3)
for ep in range(args.epochs):
    sccp.train(); tot = 0.0
    for x_img, _ in train_loader:
        x1 = x_img.to(device).view(-1, 784); x0 = torch.randn_like(x1)
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        xt_t = torch.cat([xt, t.view(-1, 1)], dim=-1)
        out = sccp(xt_t)
        if predicts_x1:
            w = 1.0 / torch.clamp((1 - t.view(-1, 1)) ** 2, min=0.05 ** 2)
            loss = torch.mean(w * (out - x1) ** 2)
        else:
            loss = torch.mean((out - ut) ** 2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(sccp.parameters(), 1.0); opt.step(); tot += loss.item()
    print(f"    epoch {ep+1}/{args.epochs} loss={tot/len(train_loader):.4f}")

# ---- generation ----
sccp.eval()
torch.manual_seed(0)
x0 = torch.randn(args.n, 784, device=device)
with torch.no_grad():
    node = NeuralODE(torch_wrapper(sccp), solver="dopri5", atol=1e-5, rtol=1e-5)
    gen = node.trajectory(x0, t_span=torch.linspace(0, 1, 2, device=device))[-1].view(args.n, 1, 28, 28)
    means = gen.view(args.n, -1).mean(dim=1)
    # canonicalise la polarite (fond noir) avant classification, pour juger la FORME
    canon = torch.where((means > 0).view(-1, 1, 1, 1), -gen, gen)
    preds = clf(canon).argmax(dim=1).cpu()
gen = gen.cpu(); means = means.cpu()

# ---- resume ----
n_inv = int((means > 0).sum())
print(f"\n[Classes generees] (forme, polarite canonisee) :")
for i, d in enumerate(digits):
    print(f"   chiffre {d} : {int((preds == i).sum())}/{args.n}")
print(f"[Inversion couleur] {n_inv}/{args.n} echantillons a fond blanc (moy pixel > 0)")
std_gen = gen.view(args.n, -1).std(dim=0).mean().item()
print(f"[Diversite] std_gen (pixel) = {std_gen:.4f}")

# ---- figure ----
ncols = 8
nrows = math.ceil(args.n / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(1.6 * ncols, 1.9 * nrows), squeeze=False)
for k in range(nrows * ncols):
    ax = axes[k // ncols, k % ncols]; ax.axis("off")
    if k < args.n:
        ax.imshow(gen[k, 0], cmap="gray", vmin=-1, vmax=1)
        m = means[k].item(); inv = m > 0
        ax.set_title(f"{digits[preds[k]]}{' INV' if inv else ''}",
                     fontsize=8, color=("red" if inv else "black"))
tag = "wbias" if args.w_bias else "symetrique"
fig.suptitle(f"ConvScCP image {digits} — {tag} / couplage={args.coupling} / {args.epochs}ep — "
             f"inversions={n_inv}/{args.n}  std={std_gen:.3f}", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "samples.png"), dpi=120)
plt.close()
print(f"\nResultat dans {run_dir}/samples.png")
