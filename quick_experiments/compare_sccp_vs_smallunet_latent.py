"""Test de contrôle "décodeur figé identique" (cf. retour de Repetti) :
compare ScCP-prox-l1-par-canal-latent et SmallUNetLatent (baseline générique,
non-proximale) à paramètres egaux, dans le MEME espace latent VAE et avec le
MEME décodeur gelé. But : isoler ce qui vient du VAE/décodeur de ce qui vient
de la structure proximale ScCP elle-même.

digit=0 uniquement (cas unimodal favorable, cf. réserve théorique sur la
multimodalité — à reproduire sur plusieurs classes si ce test est concluant).
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
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher, ConditionalFlowMatcher
from torchcfm.utils import torch_wrapper

from models.architectures import LatentScCP_UNN, SmallUNetLatentV2
from ae_diag import load_or_train_ae, save_ae_reconstruction_check

parser = argparse.ArgumentParser(description="Compare ScCP / SmallUNetLatentV2, x-pred / v-pred / x-pred+skip.")
parser.add_argument("--only", type=str, default="", help="Run only models whose name contains this substring.")
parser.add_argument("--skip", type=str, default="", help="Skip models whose name contains this substring.")
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs per model.")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
idx = torch.where(dataset.targets == 0)[0]
batch_size_train = 1024   # plus gros batch -> couplage OT plus informatif en haute dimension (hypothese a tester)
train_loader = DataLoader(Subset(dataset, idx), batch_size=batch_size_train, shuffle=True, num_workers=2, pin_memory=True)
print(f"Nb images (digit=0): {len(idx)}")

# ---- AE/decodeur PARTAGE, gelé, identique pour les deux modeles ----
# charge le checkpoint partagé (results/ae_check/mnist_ae.pt) s'il existe déjà,
# sinon l'entraîne une fois et le sauvegarde pour les prochains scripts.
ae = load_or_train_ae(train_loader, device, c_lat=16, epochs=500)
save_ae_reconstruction_check(ae, train_loader, device, "results/compare_sccp_vs_smallunet_latent")
c_lat, Hl = ae.c_lat, ae.latent_spatial
lat_dim = c_lat * Hl * Hl

models = {
    # --- ScCP : 3 quadrants (reseau x_pred + x-loss / reseau x_pred + v-loss=papier / reseau v_pred + v-loss) ---
    "ScCP_xpred_xloss":  LatentScCP_UNN(c_lat=c_lat, latent_spatial=Hl, K=30, internal_channel=128,
                                         use_Unet="l1", version="LNO", use_checkpoint=True,
                                         predicts_x1=True).to(device),
    "ScCP_xpred_vloss":  LatentScCP_UNN(c_lat=c_lat, latent_spatial=Hl, K=30, internal_channel=256,
                                         use_Unet="l1", version="LNO", use_checkpoint=False,
                                         predicts_x1=True, vloss_weight=True).to(device),
    "ScCP_vpred":        LatentScCP_UNN(c_lat=c_lat, latent_spatial=Hl, K=10, internal_channel=128,
                                         use_Unet="l1", version="LNO", use_checkpoint=True,
                                         predicts_x1=False).to(device),
    # --- SmallUNetLatentV2 : memes 3 quadrants ---
    "SmallUNet_xpred_xloss": SmallUNetLatentV2(c_lat=c_lat, latent_spatial=Hl, base_ch=64,
                                                predicts_x1=True).to(device),
    "SmallUNet_xpred_vloss": SmallUNetLatentV2(c_lat=c_lat, latent_spatial=Hl, base_ch=64,
                                                predicts_x1=True, vloss_weight=True).to(device),
    "SmallUNet_vpred":       SmallUNetLatentV2(c_lat=c_lat, latent_spatial=Hl, base_ch=64,
                                                predicts_x1=False).to(device),
}
all_names = list(models.keys())
if args.only:
    models = {name: m for name, m in models.items() if args.only in name}
if args.skip:
    models = {name: m for name, m in models.items() if args.skip not in name}
if not models:
    raise SystemExit(f"Aucun modele selectionne (--only={args.only!r} --skip={args.skip!r}). "
                     f"Modeles disponibles : {all_names}")
print(f"Modeles selectionnes : {list(models.keys())}")

for name, m in models.items():
    n = sum(p.numel() for p in m.parameters())
    print(f"{name}: {n:,} params")

FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)
nb_epochs = args.epochs
run_dir = "results/compare_sccp_vs_smallunet_latent"
os.makedirs(run_dir, exist_ok=True)

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
            if getattr(model, "predicts_x1", False):
                # out = x1_pred. Erreur dans l'espace x1.
                err2 = (out - x1) ** 2
                if getattr(model, "vloss_weight", False):
                    # loss calculee dans l'espace V (papier Back to Basics, Algo 1) :
                    # ||v_pred - v||^2 = (1/(1-t)^2) ||x_pred - x1||^2. Poids -> +inf en t=1,
                    # clampe le denominateur a 0.05 comme dans le papier.
                    w = 1.0 / torch.clamp((1 - t.view(batch_size, 1)) ** 2, min=0.05 ** 2)
                    loss = torch.mean(w * err2)
                else:
                    loss = torch.mean(err2)              # x-loss : poids uniforme en t
            else:
                loss = torch.mean((out - ut) ** 2)       # v-pred + v-loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        loss_histories[name].append(avg_loss)
        print(f"  [{name}] epoch {epoch+1}/{nb_epochs}  loss={avg_loss:.4f}")

# ---- generation : meme x0, meme decodeur, pour les deux modeles ----
n_samples = 8
torch.manual_seed(0)
x0_test = torch.randn(n_samples, lat_dim, device=device)

fig, axes = plt.subplots(len(models), n_samples, figsize=(2 * n_samples, 2 * len(models)))
diversity = {}
for row, (name, model) in enumerate(models.items()):
    model.eval()
    with torch.no_grad():
        node = NeuralODE(torch_wrapper(model), solver="dopri5", atol=1e-5, rtol=1e-5)
        traj = node.trajectory(x0_test, t_span=torch.linspace(0, 1, 2, device=device))
        z_final = traj[-1]
        std_gen = z_final.std(dim=0).mean().item()
        diversity[name] = std_gen
        imgs = ae.decode(z_final.view(n_samples, c_lat, Hl, Hl)).cpu()
    for col in range(n_samples):
        ax = axes[row, col] if len(models) > 1 else axes[col]
        ax.imshow(imgs[col, 0], cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(name, fontsize=9)
    print(f"{name}: std_gen (latent, intra-batch) = {std_gen:.4f}")

plt.suptitle("Meme x0, meme decodeur gele — ScCP vs SmallUNetLatentV2 x-pred vs v-pred")
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "comparison.png"), dpi=100)
plt.close()

with open(os.path.join(run_dir, "summary.txt"), "w") as f:
    f.write(f"# nb_epochs={nb_epochs} batch_size={batch_size_train} c_lat={c_lat}\n")
    for name in models:
        n = sum(p.numel() for p in models[name].parameters())
        f.write(f"{name}\tparams={n}\tfinal_loss={loss_histories[name][-1]:.6f}\tstd_gen={diversity[name]:.4f}\n")

print(f"\nResultats dans {run_dir}/")
