from models.architectures import LatentScCP_UNN
from ae_diag import load_or_train_ae, save_ae_reconstruction_check
import argparse
import gc
import os
import sys
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchcfm.models.unet import UNetModel
import time
import random

import matplotlib
matplotlib.use("Agg")  # headless — save figures instead of displaying them
import matplotlib.pyplot as plt
import torch
from torchdyn.core import NeuralODE
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher
from torchcfm.utils import torch_wrapper
from tqdm import tqdm
import torch._inductor.config as ind

def plot_images(images, title, save_path):
    """Save a strip of 5 generated images to disk."""
    n = 5
    images = images.cpu().view(-1, 28, 28)
    fig, axes = plt.subplots(1, n, figsize=(10, 2))
    fig.suptitle(title, fontsize=9)
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=80)
    plt.close(fig)


def write_param_file(model, run_dir):
    """Write "parametres.txt" in run_dir with the architecture hyperparameters
    of `model`, read directly off its attributes. Mirrors run_2moons.py's
    write_param_file so make_grille.py's K x dual_dim grid also works on
    MNIST results — `internal_channel` (the MNIST equivalent of dual_dim,
    the conv UNNs' internal channel count) is stored under the "dual_dim"
    key for that reason.
    """
    fields = {
        "model_class": type(model).__name__,
        "K":           getattr(model, "K", None),
        "dual_dim":    getattr(model, "internal_channel", None),
        "version":     getattr(model, "version", None),
    }
    with open(os.path.join(run_dir, "parametres.txt"), "w") as f:
        for key, value in fields.items():
            if value is not None:
                f.write(f"{key}={value}\n")



# =========================================================================== #
#  (B) train_mnist modifié
# =========================================================================== #
def train_mnist(
    model,
    train_loader,
    device,
    results_dir,
    model_name,
    nb_epochs=5,
    lr=1e-3,
    randomized_layer_nb=False,
    multi_iter=False,
    ae=None,                       # <-- NOUVEAU : autoencodeur gelé (None = espace image)
):
    """
    Si `ae` est fourni : Flow Matching DANS LE LATENT.
      - x1 est encodé une fois (ae.encode, gelé), le bruit x0 est latent,
        la loss est latente.
      - la génération intègre l'ODE dans le latent puis décode (ae.decode)
        seulement pour sauvegarder les images.
    Si `ae is None` : comportement image d'origine (inchangé).
    """
    run_dir = os.path.join(results_dir, model_name)
    os.makedirs(run_dir, exist_ok=True)
    write_param_file(model, run_dir)                       # défini ailleurs

    latent = ae is not None
    if latent:
        ae = ae.to(device).eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        c_lat = ae.c_lat
        Hl    = ae.latent_spatial
        lat_dim = c_lat * Hl * Hl                          # = model.dim (196 par défaut)

    FM        = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}\nModel : {model_name}\nParams: {total_params:,}\n{'='*60}")
    with open(os.path.join(run_dir, "params.txt"), "w") as f:
        f.write(f"{total_params}\n")

    loss_history = []
    train_start  = time.perf_counter()

    for epoch in range(nb_epochs):
        model.train()
        total_loss = 0.0
        t0 = time.perf_counter()

        for x1_batch, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{nb_epochs}", leave=False):
            batch_size = x1_batch.size(0)

            # -------- cible x1 : latente (encodée, gelée) ou image --------
            if latent:
                x1_img = x1_batch.to(device).view(batch_size, 1, 28, 28)
                with torch.no_grad():
                    x1 = ae.encode(x1_img).flatten(1)      # (B, lat_dim)
            elif model_name == "UNet_torchCFM_baseline":
                x1 = x1_batch.to(device)
            else:
                x1 = x1_batch.to(device).view(batch_size, -1)

            x0 = torch.randn_like(x1)                      # bruit dans le bon espace
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

            # -------- forward --------
            if randomized_layer_nb and not latent:
                xt_t = torch.cat([xt, t.view(batch_size, 1)], dim=-1)
                out  = model(xt_t, n_iter=random.randint(5, 15))
            elif model_name == "UNet_torchCFM_baseline" and not latent:
                out = model(t, xt)
            else:
                xt_t = torch.cat([xt, t.view(batch_size, 1)], dim=-1)
                out  = model(xt_t)

            # -------- loss --------
            if getattr(model, "predicts_x1", False):
                loss = torch.mean((out - x1) ** 2)         # x1 latent si latent=True
            else:
                loss = torch.mean((out - ut) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch+1}/{nb_epochs} — loss: {avg_loss:.4f}  ({time.perf_counter()-t0:.1f}s)")

        # ---- génération périodique ----
        if epoch % 2 == 0 and epoch > 0:
            model.eval()
            with torch.no_grad():
                if latent:
                    # bruit latent -> intégration ODE dans le latent -> décodage
                    x0_test = torch.randn(10, lat_dim, device=device)
                    node = NeuralODE(torch_wrapper(model), solver="dopri5",
                                     atol=1e-5, rtol=1e-5)
                    traj = node.trajectory(x0_test, t_span=torch.linspace(0, 1, 2, device=device))
                    lat_final = traj[-1].view(10, c_lat, Hl, Hl)
                    imgs = ae.decode(lat_final)            # (10,1,28,28)
                    plot_images(imgs, title=f"{model_name} — epoch {epoch+1}",
                                save_path=os.path.join(run_dir, f"epoch_{epoch+1}.png"))
                elif model_name == "UNet_torchCFM_baseline":
                    x0_test = torch.randn(10, 1, 28, 28, device=device)
                    class _W(torch.nn.Module):
                        def __init__(s, m): super().__init__(); s.m = m
                        def forward(s, t, x, **kw): return s.m(t.expand(x.shape[0]), x)
                    node = NeuralODE(_W(model), solver="dopri5", atol=1e-5, rtol=1e-5)
                    traj = node.trajectory(x0_test, t_span=torch.linspace(0, 1, 2, device=device))
                    plot_images(traj[-1], title=f"{model_name} — epoch {epoch+1}",
                                save_path=os.path.join(run_dir, f"epoch_{epoch+1}.png"))
                else:
                    x0_test = torch.randn(10, 784, device=device)
                    node = NeuralODE(torch_wrapper(model), solver="dopri5", atol=1e-5, rtol=1e-5)
                    traj = node.trajectory(x0_test, t_span=torch.linspace(0, 1, 2, device=device))
                    plot_images(traj[-1], title=f"{model_name} — epoch {epoch+1}",
                                save_path=os.path.join(run_dir, f"epoch_{epoch+1}.png"))
            model.train()

    # ---- courbes / log ----
    plt.figure(); plt.plot(range(1, nb_epochs + 1), loss_history, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Average FM loss"); plt.title(f"Training loss — {model_name}")
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "loss.png")); plt.close()
    with open(os.path.join(run_dir, "loss.txt"), "w") as f:
        for ep, l in enumerate(loss_history, 1):
            f.write(f"{ep}\t{l:.6f}\n")

    print(f"  Results saved to: {run_dir}")
    return loss_history, total_params, time.perf_counter() - train_start

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_train_loader(batch_size=128, digit=0):
    """MNIST train loader restricted to a single digit class (default: 0).
    Pass digit=None to train on the full MNIST training set instead."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform,
    )

    if digit is None:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    idx = torch.where(dataset.targets == digit)[0]
    digit_dataset = Subset(dataset, idx)
    return DataLoader(digit_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)



parser = argparse.ArgumentParser(description="Run all UNN-FM experiments on MNIST.")
parser.add_argument("--epochs",      type=int,   default=5,         help="Number of training epochs per model.")
parser.add_argument("--results-dir", type=str,   default="results", help="Root directory for outputs.")
parser.add_argument("--skip",        type=str,   default="",        help="Skip experiments whose name contains this substring.")
parser.add_argument("--only",        type=str,   default="",        help="Run only experiments whose name contains this substring.")
parser.add_argument("--batch-size",  type=int,   default=128)
parser.add_argument("--digit",       type=int,   default=0,
                        help="Train only on this MNIST digit class (default: 0, i.e. zeros only). "
                            "Use --digit -1 to train on the full MNIST training set instead.")
parser.add_argument("--device",      type=str, default="",
                         help="Torch device to use, e.g. 'cuda:0', 'cuda:1', 'cpu'. "
                              "Defaults to cuda if available, else cpu.")
args = parser.parse_args()

if args.device:
    device = torch.device(args.device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

digit = None if args.digit < 0 else args.digit
print(f"Training digit filter: {'all digits' if digit is None else digit}")


train_loader = get_train_loader(batch_size=args.batch_size, digit=digit)


# 1) AE : charge le checkpoint partagé s'il existe déjà, sinon l'entraîne une
#    fois et le sauvegarde (réutilisable par tous les scripts latents)
ae = load_or_train_ae(train_loader, device, c_lat=4)
save_ae_reconstruction_check(ae, train_loader, device, args.results_dir)

# 2) modèle latent (note : internal_channel = C_dual ; c_lat fixé par l'AE)
model = LatentScCP_UNN(c_lat=4, latent_spatial=7, K=38,
                       internal_channel=1024, use_Unet="l1",
                       version="LNO", use_checkpoint=True).to(device)

# 3) entraînement latent
train_mnist(model, train_loader, device, results_dir="results",
            model_name="LatentScCP_L1_LNO", nb_epochs=100, ae=ae)