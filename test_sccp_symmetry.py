"""Test de la symetrie impaire du ScCP (prox l1 impair -> champ de vitesse impair).

Deux niveaux :
  (1) champ de vitesse : v(-x_t, t) == -v(x_t, t) ?  -> erreur relative ||v(-x)+v(x)||/||v(x)||
      (propriete ARCHITECTURALE, vraie meme sans entrainement)
  (2) generation : gen(-x0) == -gen(x0) ?  -> inversion couleur pour ~la moitie du bruit

Espace image, digit=0. Entraine vite un ConvScCP (ou --ckpt pour charger).
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
    ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher)
from torchcfm.utils import torch_wrapper

from models.architectures import ConvScCP_UNN

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--K", type=int, default=20)
parser.add_argument("--ic", type=int, default=64)
parser.add_argument("--version", type=str, default="LNO")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--coupling", type=str, default="indep", choices=["indep", "ot"])
parser.add_argument("--digit", type=int, default=0)
parser.add_argument("--n", type=int, default=8, help="nb d'echantillons generes")
parser.add_argument("--ckpt", type=str, default="", help="charger un ConvScCP depuis ce model.pt (skip entrainement ScCP)")
parser.add_argument("--asymmetric", action="store_true", help="(ii) prox l1 asymetrique deux-seuils clamp(u,-lam-,lam+) -> brise l'impairite")
parser.add_argument("--w-bias", action="store_true", help="(i) biais convolutif appris sur W -> non-impair en fct de l'entree")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
idx = torch.where(dataset.targets == args.digit)[0]
train_loader = DataLoader(Subset(dataset, idx), batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)

run_dir = "results/test_sccp_symmetry"
os.makedirs(run_dir, exist_ok=True)

FM = (ConditionalFlowMatcher(sigma=0.1) if args.coupling == "indep"
      else ExactOptimalTransportConditionalFlowMatcher(sigma=0.1))


def train(model):
    predicts_x1 = getattr(model, "predicts_x1", False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for x_img, _ in train_loader:
            x1 = x_img.to(device).view(-1, 784); x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            xt_t = torch.cat([xt, t.view(-1, 1)], dim=-1)
            out = model(xt_t)
            if predicts_x1:
                w = 1.0 / torch.clamp((1 - t.view(-1, 1)) ** 2, min=0.05 ** 2)
                loss = torch.mean(w * (out - x1) ** 2)
            else:
                loss = torch.mean((out - ut) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += loss.item()
        print(f"    epoch {ep+1}/{args.epochs} loss={tot/len(train_loader):.4f}")
    return model


def _snapshot_bufs(model):
    return {k: v.clone() for k, v in model.named_buffers()}


def _restore_bufs(model, snap):
    for k, v in model.named_buffers():
        v.copy_(snap[k])


@torch.no_grad()
def velocity_oddness(model):
    """erreur relative ||v(-x_t,t)+v(x_t,t)|| / ||v(x_t,t)|| moyennee sur un batch.
    NB : on FIGE le buffer de power-iteration (spectral norm) entre les deux appels,
    sinon il mute et rend le champ non-deterministe -> fausse la mesure d'impairite."""
    model.eval()
    x_img, _ = next(iter(train_loader))
    x1 = x_img.to(device).view(-1, 784)
    x0 = torch.randn_like(x1)
    t, xt, _ = FM.sample_location_and_conditional_flow(x0, x1)
    tv = t.view(-1, 1)
    snap = _snapshot_bufs(model)
    v_pos = model(torch.cat([xt, tv], dim=-1))
    _restore_bufs(model, snap)
    v_neg = model(torch.cat([-xt, tv], dim=-1))
    return ((v_pos + v_neg).norm() / (v_pos.norm() + 1e-9)).item()


@torch.no_grad()
def generate(model, x0):
    model.eval()
    node = NeuralODE(torch_wrapper(model), solver="dopri5", atol=1e-5, rtol=1e-5)
    return node.trajectory(x0, t_span=torch.linspace(0, 1, 2, device=device))[-1]


# ---- modeles ----
sccp = ConvScCP_UNN(dim=784, K=args.K, internal_channel=args.ic, use_Unet="l1",
                    version=args.version, use_checkpoint=True,
                    asymmetric_prox=args.asymmetric, w_bias=args.w_bias).to(device)
if args.ckpt:
    sccp.load_state_dict(torch.load(args.ckpt, map_location=device)); print(f"[ScCP] charge {args.ckpt}")
else:
    print("== entrainement ScCP (asymmetric prox: {}, bias: {}) ==".format(args.asymmetric, args.w_bias)); train(sccp)

# ---- test 1 : impairite du champ de vitesse ----
odd_sccp = velocity_oddness(sccp)
print(f"\n[Champ de vitesse] erreur relative d'impairite  ||v(-x)+v(x)||/||v(x)|| :")
print(f"   ScCP = {odd_sccp:.2e}   (proche de 0 => v impair symetrique EXACT ; grand => impairite brisee)")

# ---- test 2 : gen(-x0) vs -gen(x0) ----
torch.manual_seed(0)
x0 = torch.randn(args.n, 784, device=device)
snap = _snapshot_bufs(sccp)               # meme etat de power-iter pour les deux solves
g_pos = generate(sccp, x0)
_restore_bufs(sccp, snap)
g_neg = generate(sccp, -x0)
sym_err = ((g_neg - (-g_pos)).norm() / (g_pos.norm() + 1e-9)).item()       # 0 si gen(-x0)=-gen(x0)
g_pos = g_pos.view(args.n, 1, 28, 28).cpu()
g_neg = g_neg.view(args.n, 1, 28, 28).cpu()
print(f"[Generation] ScCP : ||gen(-x0) - (-gen(x0))|| / ||gen(x0)|| = {sym_err:.2e}")

# ---- signe des pixels : un vrai zero MNIST a un fond noir (~-1) => moyenne NEGATIVE ;
#      sa version inverse-couleur a un fond blanc (~+1) => moyenne POSITIVE. Comme
#      gen(-x0) = -gen(x0), l'une des deux images est forcement "inversee". ----
print(f"\n[Signe des pixels] moyenne par image  (<0 = fond noir = OK ; >0 = fond blanc = INVERSE) :")
rows = [("gen(x0)", g_pos), ("gen(-x0)", g_neg)]
for label, imgs in rows:
    means = imgs.view(args.n, -1).mean(dim=1)
    n_inv = int((means > 0).sum())
    print(f"   ScCP {label:9s} : moyenne globale={means.mean():+.3f}  |  inversees (moy>0) = {n_inv}/{args.n}")

# ---- figure : gen(x0) et gen(-x0) (titre = moyenne pixel, rouge si inverse) ----
fig, axes = plt.subplots(2, args.n, figsize=(1.6 * args.n, 4.0))
for r, (label, imgs) in enumerate(rows):
    for c in range(args.n):
        m = imgs[c].mean().item()
        axes[r, c].imshow(imgs[c, 0], cmap="gray", vmin=-1, vmax=1); axes[r, c].axis("off")
        axes[r, c].set_title(f"{m:+.2f}" + (" INV" if m > 0 else ""),
                             fontsize=7, color=("red" if m > 0 else "black"))
    axes[r, 0].set_ylabel(f"ScCP {label}", fontsize=9, rotation=0, labelpad=42)
fig.suptitle(f"Symetrie impaire ScCP — gen(-x0) vs gen(x0)  |  "
             f"impairite champ={odd_sccp:.1e}  gen sym-err={sym_err:.1e}", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "symmetry.png"), dpi=120)
plt.close()
print(f"\nResultat dans {run_dir}/symmetry.png")
print("Symetrique : gen(-x0) = inverse-couleur exact de gen(x0). Avec --asymmetric / --w-bias : brise.")
