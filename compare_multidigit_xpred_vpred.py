"""Test multi-classes : x-pred vs v-pred, avec un AE FRAIS dedie a ce sous-
ensemble de chiffres (le checkpoint digit=0-only ne doit pas etre reutilise
ici, il n'a jamais vu les autres formes).

Mesure la diversite GLOBALE (std_gen sur tous les echantillons, peut etre
artificiellement "sauvee" par la seule variance inter-classe) ET la
diversite INTRA-CLASSE (std_gen calculee a l'interieur de chaque groupe de
classe predite par un petit classifieur), pour trancher l'hypothese : avec
plusieurs chiffres, x-pred resout-il le collapse globalement, ou seulement
entre classes (collapse residuel a l'interieur de chaque chiffre) ?
"""
import argparse
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
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.utils import torch_wrapper

from models.architectures import LatentScCP_UNN, SmallUNetLatentV2
from ae_diag import load_or_train_ae, save_ae_reconstruction_check

parser = argparse.ArgumentParser()
parser.add_argument("--digits", type=int, nargs="+", default=[0, 1])
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--ae-epochs", type=int, default=30)
parser.add_argument("--only", type=str, default="")
parser.add_argument("--skip", type=str, default="")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, digits={args.digits}")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
mask = torch.zeros(len(dataset.targets), dtype=torch.bool)
for d in args.digits:
    mask |= (dataset.targets == d)
idx = torch.where(mask)[0]
batch_size_train = 512
train_loader = DataLoader(Subset(dataset, idx), batch_size=batch_size_train, shuffle=True, num_workers=2, pin_memory=True)
print(f"Nb images (digits={args.digits}): {len(idx)}")

digits_tag = "".join(map(str, args.digits))
run_dir = f"results/compare_multidigit_{digits_tag}"
os.makedirs(run_dir, exist_ok=True)

# AE dedie a ce sous-ensemble de classes : checkpoint distinct (suffixe digits_tag)
# pour ne jamais collisionner avec l'AE digit=0-only.
ae_ckpt = f"results/ae_check/mnist_ae_multidigit_{digits_tag}_clat16_base32.pt"
ae = load_or_train_ae(train_loader, device, c_lat=16, base=32, epochs=args.ae_epochs, ckpt_path=ae_ckpt)
save_ae_reconstruction_check(ae, train_loader, device, run_dir)
c_lat, Hl = ae.c_lat, ae.latent_spatial
lat_dim = c_lat * Hl * Hl

# Petit classifieur MNIST sur les memes classes, pour mesurer la diversite
# intra-classe des echantillons generes (pas seulement globale).
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
for i, d in enumerate(args.digits):
    label_lookup[d] = i

clf = TinyClassifier(len(args.digits)).to(device)
clf_opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
for epoch in range(3):
    for x_img, y in train_loader:
        x_img = x_img.to(device)
        y = label_lookup[y].to(device)
        logits = clf(x_img)
        loss = F.cross_entropy(logits, y)
        clf_opt.zero_grad()
        loss.backward()
        clf_opt.step()
clf.eval()
print(f"[classifieur] entraine ({len(args.digits)} classes)")

models = {
    "ScCP_L1_perchannel_xpred": LatentScCP_UNN(c_lat=c_lat, latent_spatial=Hl, K=38, internal_channel=512,
                                                use_Unet="l1", version="LNO", use_checkpoint=True,
                                                predicts_x1=True).to(device),
    "ScCP_L1_perchannel_vpred": LatentScCP_UNN(c_lat=c_lat, latent_spatial=Hl, K=38, internal_channel=512,
                                                use_Unet="l1", version="LNO", use_checkpoint=True,
                                                predicts_x1=False).to(device),
    "SmallUNetLatentV2_xpred": SmallUNetLatentV2(c_lat=c_lat, latent_spatial=Hl, base_ch=64,
                                                  predicts_x1=True).to(device),
    "SmallUNetLatentV2_vpred": SmallUNetLatentV2(c_lat=c_lat, latent_spatial=Hl, base_ch=64,
                                                  predicts_x1=False).to(device),
}
if args.only:
    models = {n: m for n, m in models.items() if args.only in n}
if args.skip:
    models = {n: m for n, m in models.items() if args.skip not in n}
print(f"Modeles selectionnes : {list(models.keys())}")
for name, m in models.items():
    n = sum(p.numel() for p in m.parameters())
    print(f"{name}: {n:,} params")

FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)
nb_epochs = args.epochs
loss_histories = {name: [] for name in models}

for name, model in models.items():
    print(f"\n{'='*60}\nTraining {name}\n{'='*60}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(nb_epochs):
        model.train()
        total_loss = 0.0
        for x1_batch, _ in train_loader:
            batch_size = x1_batch.size(0)
            x1_img = x1_batch.to(device).view(batch_size, 1, 28, 28)
            with torch.no_grad():
                x1 = ae.encode(x1_img).flatten(1)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            xt_t = torch.cat([xt, t.view(batch_size, 1)], dim=-1)
            out = model(xt_t)
            target = x1 if getattr(model, "predicts_x1", False) else ut
            loss = torch.mean((out - target) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        loss_histories[name].append(avg_loss)
        print(f"  [{name}] epoch {epoch+1}/{nb_epochs}  loss={avg_loss:.4f}")

# ---- generation : un seul x0 partage, beaucoup d'echantillons pour avoir des
# groupes par classe non-vides -> diversite globale ET intra-classe ----
n_samples = 64
torch.manual_seed(0)
x0_test = torch.randn(n_samples, lat_dim, device=device)

diversity_global = {}
diversity_intraclass = {}
class_counts = {}
n_show = 8
fig, axes = plt.subplots(len(models), n_show, figsize=(2 * n_show, 2 * len(models)))
for row, (name, model) in enumerate(models.items()):
    model.eval()
    with torch.no_grad():
        node = NeuralODE(torch_wrapper(model), solver="dopri5", atol=1e-5, rtol=1e-5)
        traj = node.trajectory(x0_test, t_span=torch.linspace(0, 1, 2, device=device))
        z_final = traj[-1]
        std_global = z_final.std(dim=0).mean().item()
        diversity_global[name] = std_global

        imgs = ae.decode(z_final.view(n_samples, c_lat, Hl, Hl))
        preds = clf(imgs).argmax(dim=1)
        intra_stds = []
        counts = {}
        for c in range(len(args.digits)):
            sel = (preds == c)
            counts[args.digits[c]] = int(sel.sum().item())
            if sel.sum() >= 2:
                intra_stds.append(z_final[sel].std(dim=0).mean().item())
        diversity_intraclass[name] = sum(intra_stds) / len(intra_stds) if intra_stds else float("nan")
        class_counts[name] = counts
        imgs_cpu = imgs.cpu()
    for col in range(n_show):
        ax = axes[row, col] if len(models) > 1 else axes[col]
        ax.imshow(imgs_cpu[col, 0], cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(name, fontsize=9)
    print(f"{name}: std_global={std_global:.4f}  std_intraclass(moy)={diversity_intraclass[name]:.4f}  counts={counts}")

plt.suptitle(f"Multi-digit {args.digits} — diversite globale vs intra-classe")
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "comparison.png"), dpi=100)
plt.close()

with open(os.path.join(run_dir, "summary.txt"), "w") as f:
    f.write(f"# digits={args.digits} nb_epochs={nb_epochs} batch_size={batch_size_train} c_lat={c_lat} n_samples={n_samples}\n")
    for name in models:
        n = sum(p.numel() for p in models[name].parameters())
        f.write(f"{name}\tparams={n}\tfinal_loss={loss_histories[name][-1]:.6f}\t"
                f"std_global={diversity_global[name]:.4f}\tstd_intraclass={diversity_intraclass[name]:.4f}\t"
                f"counts={class_counts[name]}\n")

print(f"\nResultats dans {run_dir}/")
