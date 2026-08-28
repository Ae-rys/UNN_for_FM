"""Point d'entree explicite pour (re)entrainer l'AE/VAE-leger partage par tous
les scripts latents (run_mnist_latent.py, diag_latent_collapse.py,
compare_sccp_vs_smallunet_latent.py) et verifier sa qualite de reconstruction.

Ecrase results/ae_check/mnist_ae.pt a chaque execution : les autres scripts
rechargeront ensuite ces poids au lieu de re-entrainer. digit=0 (meme scope
que les autres scripts, pour que ce soit bien le meme AE "partout")."""
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from ae_diag import load_or_train_ae, save_ae_reconstruction_check

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
idx = torch.where(dataset.targets == 0)[0]
train_loader = DataLoader(Subset(dataset, idx), batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

ae = load_or_train_ae(train_loader, device, c_lat=4, epochs=10, force_retrain=True)
save_ae_reconstruction_check(ae, train_loader, device, "results/ae_check")
