"""Utilitaire partagé : entraîner (une fois) ou recharger le MnistAE/VAE-léger
commun à tous les scripts latents, et vérifier visuellement son expressivité
(original vs reconstruit + MSE)."""
import os
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.architectures import MnistAE, pretrain_ae

AE_CKPT_DIR = "results/ae_check"


def load_or_train_ae(train_loader, device, c_lat=4, base=32, epochs=10,
                      ckpt_path=None, force_retrain=False):
    """Charge l'AE depuis ckpt_path s'il existe (poids gelés, prêts à l'emploi),
    sinon l'entraîne (pretrain_ae) et sauvegarde ses poids pour les prochains
    appels. Tous les scripts qui partagent le même (c_lat, base) réutilisent
    ainsi le même AE déjà entraîné, au lieu de le ré-entraîner à chaque run.

    ckpt_path par défaut inclut c_lat/base dans le nom (mnist_ae_clatX_baseY.pt)
    pour qu'une config différente n'écrase/ne charge jamais par erreur les
    poids d'une autre — sinon le load_state_dict planterait (mismatch de
    shape) ou, pire, chargerait silencieusement la mauvaise config."""
    if ckpt_path is None:
        ckpt_path = os.path.join(AE_CKPT_DIR, f"mnist_ae_clat{c_lat}_base{base}.pt")
    ae = MnistAE(c_lat=c_lat, base=base)
    if os.path.exists(ckpt_path) and not force_retrain:
        ae.load_state_dict(torch.load(ckpt_path, map_location=device))
        ae = ae.to(device).eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        print(f"[AE] poids chargés depuis {ckpt_path}")
        return ae

    ae = pretrain_ae(ae, train_loader, device, epochs=epochs)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(ae.state_dict(), ckpt_path)
    print(f"[AE] entraîné et sauvegardé dans {ckpt_path}")
    return ae


def save_ae_reconstruction_check(ae, train_loader, device, run_dir, n=10):
    """Sauvegarde run_dir/ae_reconstruction_check.png (original vs reconstruit
    sur n images du loader) et renvoie la MSE de reconstruction mesurée."""
    os.makedirs(run_dir, exist_ok=True)
    ae.eval()
    x_img, _ = next(iter(train_loader))
    x_img = x_img[:n].to(device).view(-1, 1, 28, 28)
    with torch.no_grad():
        rec = ae(x_img)
        mse = F.mse_loss(rec, x_img).item()

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(x_img[i, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(rec[i, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Reconstruit", fontsize=10)
    fig.suptitle(f"AE recon check — recon MSE = {mse:.5f}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(run_dir, "ae_reconstruction_check.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[AE check] recon MSE = {mse:.5f}  ->  {path}")
    return mse
