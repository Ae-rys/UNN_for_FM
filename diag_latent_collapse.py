"""Diagnostic : le flow latent (LatentScCP_L1_LNO) s'effondre-t-il vers un point fixe ?

Reproduit l'entrainement de run_mnist_latent.py a l'identique (digit=0, K=38, LNO,
internal_channel=512, use_Unet="l1") mais mesure, a chaque epoch, la diversite des
latents generes a partir de plusieurs bruits x0 differents.
"""
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
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.utils import torch_wrapper

from models.architectures import LatentScCP_UNN
from ae_diag import load_or_train_ae, save_ae_reconstruction_check

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---- donnees : digit=0 uniquement, comme le run par defaut ----
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
idx = torch.where(dataset.targets == 0)[0]
train_loader = DataLoader(Subset(dataset, idx), batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
print(f"Nb images (digit=0): {len(idx)}")

# ---- AE (charge le checkpoint partagé s'il existe, sinon l'entraîne) ----
ae = load_or_train_ae(train_loader, device, c_lat=4)
save_ae_reconstruction_check(ae, train_loader, device, "results/diag_latent_collapse")
c_lat, Hl = ae.c_lat, ae.latent_spatial
lat_dim = c_lat * Hl * Hl

# diversite des latents REELS (reference)
with torch.no_grad():
    x_real, _ = next(iter(train_loader))
    x_real = x_real.to(device).view(-1, 1, 28, 28)
    z_real = ae.encode(x_real).flatten(1)
print(f"[ref]  latents reels   : std intra-batch = {z_real.std(dim=0).mean().item():.4f}  "
      f"(norme moy = {z_real.norm(dim=1).mean().item():.3f})")

# ---- modele latent (config identique a run_mnist_latent.py) ----
model = LatentScCP_UNN(c_lat=4, latent_spatial=7, K=38, internal_channel=512,
                        use_Unet="l1", version="LNO", use_checkpoint=True).to(device)

FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

run_dir = "results/diag_latent_collapse"
os.makedirs(run_dir, exist_ok=True)

diversity_log = []   # (epoch, loss, std_gen, std_ratio)

def measure_diversity(n_batches=3, n_samples=10):
    """Integre l'ODE a partir de n_batches x0 differents, renvoie le std moyen
    intra-batch des latents finaux (mesure de diversite / collapse)."""
    model.eval()
    stds = []
    last_imgs = None
    with torch.no_grad():
        for b in range(n_batches):
            x0 = torch.randn(n_samples, lat_dim, device=device)
            node = NeuralODE(torch_wrapper(model), solver="dopri5", atol=1e-5, rtol=1e-5)
            traj = node.trajectory(x0, t_span=torch.linspace(0, 1, 2, device=device))
            z_final = traj[-1]
            stds.append(z_final.std(dim=0).mean().item())
            if b == 0:
                last_imgs = ae.decode(z_final.view(n_samples, c_lat, Hl, Hl)).cpu()
    model.train()
    return sum(stds) / len(stds), last_imgs

nb_epochs = 10
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
        loss = torch.mean((out - x1) ** 2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    std_gen, imgs = measure_diversity()
    std_ratio = std_gen / z_real.std(dim=0).mean().item()
    diversity_log.append((epoch + 1, avg_loss, std_gen, std_ratio))
    print(f"Epoch {epoch+1:2d}/{nb_epochs}  loss={avg_loss:.4f}  "
          f"std_gen={std_gen:.4f}  std_gen/std_real={std_ratio:.3f}")

    n = imgs.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
    fig.suptitle(f"epoch {epoch+1} — std_gen/std_real={std_ratio:.3f}", fontsize=9)
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i, 0], cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"epoch_{epoch+1}.png"), dpi=80)
    plt.close(fig)

# ---- courbe loss vs diversite ----
epochs_, losses_, stds_, ratios_ = zip(*diversity_log)
fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(epochs_, losses_, "o-", color="tab:blue", label="loss")
ax1.set_xlabel("epoch"); ax1.set_ylabel("FM loss", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(epochs_, ratios_, "s-", color="tab:red", label="std_gen/std_real")
ax2.set_ylabel("std(latents generes) / std(latents reels)", color="tab:red")
ax2.axhline(1.0, color="tab:red", linestyle="--", linewidth=0.8)
fig.suptitle("Loss vs diversite des latents generes (collapse si ratio -> 0)")
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "loss_vs_diversity.png"), dpi=120)
plt.close(fig)

with open(os.path.join(run_dir, "diversity_log.txt"), "w") as f:
    f.write("epoch\tloss\tstd_gen\tstd_gen/std_real\n")
    for e, l, s, r in diversity_log:
        f.write(f"{e}\t{l:.6f}\t{s:.6f}\t{r:.4f}\n")

print(f"\nResultats dans {run_dir}/")

# ---- analyse fine de la trajectoire : norme du latent en fonction de t ----
model.eval()
n_traj = 8
x0 = torch.randn(n_traj, lat_dim, device=device)
node = NeuralODE(torch_wrapper(model), solver="dopri5", atol=1e-5, rtol=1e-5)
t_span = torch.linspace(0, 1, 50, device=device)
with torch.no_grad():
    traj = node.trajectory(x0, t_span=t_span)   # (T, n_traj, lat_dim)
norms = traj.norm(dim=-1).cpu()                 # (T, n_traj)
real_norm = z_real.norm(dim=1).mean().item()

plt.figure(figsize=(6, 4))
for i in range(n_traj):
    plt.plot(t_span.cpu(), norms[:, i], alpha=0.8)
plt.axhline(real_norm, color="black", linestyle="--", label="norme moy. latents reels")
plt.xlabel("t"); plt.ylabel("||z(t)||")
plt.title("Norme du latent le long de la trajectoire ODE (8 x0 differents)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "trajectory_norms.png"), dpi=120)
plt.close()

# decoder les 8 latents finaux pour voir lesquels divergent
with torch.no_grad():
    imgs_final = ae.decode(traj[-1].view(n_traj, c_lat, Hl, Hl)).cpu()
fig, axes = plt.subplots(1, n_traj, figsize=(2 * n_traj, 2))
for i, ax in enumerate(axes):
    ax.imshow(imgs_final[i, 0], cmap="gray", vmin=-1, vmax=1)
    ax.set_title(f"||z||={norms[-1, i]:.2f}", fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "trajectory_final_imgs.png"), dpi=100)
plt.close()
print("Analyse de trajectoire sauvegardee.")
