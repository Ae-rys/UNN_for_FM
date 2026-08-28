# -*- coding: utf-8 -*-
"""
hparam_sweep_mnist.py
Sweep "un facteur a la fois" (OFAT) des hyperparametres du ConvScCP_UNN pixel
sur MNIST COMPLET (les 10 chiffres, pas le digit=0 jouet de grille_convsccp.py),
pour repondre a la question ouverte du papier ("Unrolled Neural Network is all
you need") : quels choix de design (section V) pesent reellement sur la
qualite, et lesquels sont accessoires ?

Design : on fige une config BASELINE (celle qui se rapproche le plus de la
recette du rapport : kernel=9, K=20, ic=128, LNO, couplage indep, prox l1,
x-pred + loss v-space invsq clampee), puis pour chaque axe du papier on fait
varier UN SEUL hyperparametre a la fois :

    axe                     |  question du papier
    ------------------------|----------------------------------------------
    kernel_size             |  IV. "no multiscale" -> la localite compte-t-elle ?
    K (nb iterations)        |  III/VIII. profondeur du deroule vs qualite
    internal_channel (ic)    |  VIII. limitation d'expressivite = capacite ?
    version LNO vs LFO       |  V-D. contrainte de normalisation par couche
    coupling indep vs OT     |  V-C. impact qualite (pas juste vitesse d'entrainement)
    prox l1 vs silu          |  V-E. le choix du prox importe-t-il ?
    x1_weight                |  V-B. ponderation de la loss x-pred (invsq/uniform/minsnr)

Un factoriel complet (3*3*3*2*2*2*3 = 972 runs) est hors de portee ; l'OFAT
(1 baseline + 2 niveaux par axe) donne ~11 runs, largement suffisant pour un
premier classement d'importance et bien moins cher que la grille complete.

Metriques (mnist_metrics.py) : mini-FID (embedding d'un petit classifieur,
PAS comparable a un FID publie, mais classement relatif valide ENTRE nos
propres configs), entropie de classe (couverture des 10 chiffres, detecte
le collapse), std_gen (diversite brute), ratio distance-au-train (memorisation).

Usage
-----
    # Un premier passage rapide pour calibrer le temps par run (voir README
    # affiche a la fin de chaque run : "ETA restante")
    python hparam_sweep_mnist.py --epochs 15

    # Filtrer les axes (utile pour relancer juste ceux qui manquent)
    python hparam_sweep_mnist.py --epochs 15 --only kernel_size,coupling

    # Budget plus long une fois qu'on sait quels axes comptent
    python hparam_sweep_mnist.py --epochs 40 --only kernel_size

Sorties dans <results-dir>/ : sweep_results.tsv (une ligne par run, append
au fur et a mesure -> lisible meme si interrompu), samples_<name>.png
(grille d'echantillons par config), importance.png (effet de chaque axe sur
le mini-FID vs baseline), summary.txt (classement final).
"""
import argparse
import copy
import os
import time

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.architectures import ConvScCP_UNN
from train import train_mnist
from mnist_metrics import (
    train_or_load_classifier, mini_fid, class_entropy, distance_to_train_ratio,
)

# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--epochs", type=int, default=15, help="epoques par config (budget MNIST complet)")
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--n-gen", type=int, default=1000, help="echantillons generes pour les metriques")
parser.add_argument("--euler-steps", type=int, default=100, help="steps Euler a l'echantillonnage (meme protocole que CIFAR)")
parser.add_argument("--results-dir", type=str, default="results/hparam_sweep_mnist")
parser.add_argument("--only", type=str, default="", help="sous-ensemble d'axes a lancer, separes par des virgules "
                                                          "(ex: kernel_size,coupling). Vide = tout, dont baseline.")
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
os.makedirs(args.results_dir, exist_ok=True)
print(f"Device: {device} | epochs/config={args.epochs} | batch={args.batch} | n_gen={args.n_gen}", flush=True)

# --------------------------------------------------------------------------- #
# Donnees : MNIST COMPLET (10 chiffres), normalise [-1,1] comme le reste du pipeline.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
print(f"MNIST complet : {len(train_set)} train / {len(test_set)} test", flush=True)

# Classifieur pour les metriques (entraine une fois, mis en cache sur disque).
clf = train_or_load_classifier(
    DataLoader(train_set, batch_size=512, shuffle=True, num_workers=2),
    device, ckpt_path="results/mnist_classifier/clf.pt", epochs=3,
)

# Pools pour le ratio distance-au-train (kNN pixels, cf. mnist_metrics).
train_pool = torch.stack([train_set[i][0] for i in range(3000)])
test_pool = torch.stack([test_set[i][0] for i in range(500)])

# --------------------------------------------------------------------------- #
# Config baseline : la plus proche de la recette du rapport (section VI-B / footnote).
BASELINE = dict(
    kernel_size=9, K=20, internal_channel=128, version="LNO",
    coupling="indep", use_Unet="l1", x1_weight="invsq",
)

# Axes OFAT : {nom_axe: [valeurs a tester, EXCLUANT la valeur baseline]}
AXES = {
    "kernel_size":      [3, 15],                 # 3 = local façon Kamb-ResNet ; 15 = plus global que 9
    "K":                [10, 40],                 # profondeur du deroule
    "internal_channel": [32, 256],                 # capacite du dual (question d'expressivite VIII)
    "version":          ["LFO"],
    "coupling":         ["ot"],
    "use_Unet":         ["silu"],
    "x1_weight":        ["uniform"],
}

if args.only:
    wanted = set(args.only.split(","))
    AXES = {k: v for k, v in AXES.items() if k in wanted}

# --------------------------------------------------------------------------- #
# Construction de la liste des runs : baseline + un run par (axe, valeur).
def make_run(name, overrides):
    cfg = copy.deepcopy(BASELINE)
    cfg.update(overrides)
    return dict(name=name, cfg=cfg)

runs = [make_run("baseline", {})]
for axis, values in AXES.items():
    for v in values:
        runs.append(make_run(f"{axis}={v}", {axis: v}))

print(f"\n{len(runs)} runs planifies :", flush=True)
for r in runs:
    print(f"  - {r['name']}", flush=True)

# --------------------------------------------------------------------------- #
def build_model(cfg):
    return ConvScCP_UNN(
        dim=784, K=cfg["K"], internal_channel=cfg["internal_channel"],
        use_Unet=cfg["use_Unet"], version=cfg["version"], use_checkpoint=True,
        w_bias=True, in_channels=1, img_size=28, kernel_size=cfg["kernel_size"],
    ).to(device)


@torch.no_grad()
def generate(model, n, device, steps):
    """Euler a pas fixe (meme protocole que compute_fid_cifar10.py) : cout
    previsible = n/batch * steps forward passes, pas de solveur adaptatif."""
    model.eval()
    batch = 250
    imgs = []
    for i in range(0, n, batch):
        b = min(batch, n - i)
        x = torch.randn(b, 784, device=device)
        dt = 1.0 / steps
        for s in range(steps):
            t = torch.full((b, 1), s * dt, device=device)
            v = model(torch.cat([x, t], dim=-1))
            x = x + v * dt
        imgs.append(x.view(b, 1, 28, 28).cpu())
    return torch.cat(imgs, dim=0)


def save_grid(imgs, path, title, n_show=10):
    fig, axes = plt.subplots(1, n_show, figsize=(1.6 * n_show, 1.8))
    fig.suptitle(title, fontsize=9)
    for i in range(n_show):
        axes[i].imshow(imgs[i, 0], cmap="gray", vmin=-1, vmax=1)
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
tsv_path = os.path.join(args.results_dir, "sweep_results.tsv")
write_header = not os.path.exists(tsv_path)
tsv_f = open(tsv_path, "a")
if write_header:
    tsv_f.write("name\tparams\tfinal_loss\ttrain_time_s\tmini_fid\tclass_entropy\t"
                "std_gen\tdist_ratio\tclass_counts\n")
    tsv_f.flush()

results = []
sweep_t0 = time.perf_counter()
n_done = 0

for r in runs:
    name, cfg = r["name"], r["cfg"]
    print(f"\n{'='*70}\n[{n_done+1}/{len(runs)}] {name}  cfg={cfg}\n{'='*70}", flush=True)
    t0 = time.perf_counter()

    model = build_model(cfg)
    loss_hist, n_params, train_time = train_mnist(
        model, train_loader, device, args.results_dir, name,
        nb_epochs=args.epochs, lr=args.lr, coupling=cfg["coupling"],
        x1_weight=cfg["x1_weight"], save_model=True,
    )
    final_loss = loss_hist[-1] if loss_hist else float("nan")

    gen_imgs = generate(model, args.n_gen, device, args.euler_steps)
    std_gen = gen_imgs.std(dim=0).mean().item()
    mfid = mini_fid(clf, gen_imgs, train_pool[:args.n_gen] if args.n_gen <= 3000
                     else torch.stack([train_set[i][0] for i in range(args.n_gen)]), device)
    entropy, counts = class_entropy(clf, gen_imgs, device)
    dist_ratio, dist_gen, dist_test = distance_to_train_ratio(gen_imgs, train_pool, test_pool)

    save_grid(gen_imgs, os.path.join(args.results_dir, f"samples_{name.replace('=', '_')}.png"),
              title=f"{name} | loss={final_loss:.4f} mini-FID={mfid:.2f} H={entropy:.2f} std={std_gen:.3f}")

    row = dict(name=name, params=n_params, final_loss=final_loss, train_time_s=train_time,
               mini_fid=mfid, class_entropy=entropy, std_gen=std_gen, dist_ratio=dist_ratio,
               class_counts=counts)
    results.append(row)
    tsv_f.write(f"{name}\t{n_params}\t{final_loss:.6f}\t{train_time:.1f}\t{mfid:.4f}\t"
                f"{entropy:.4f}\t{std_gen:.4f}\t{dist_ratio:.4f}\t{counts}\n")
    tsv_f.flush()

    del model
    torch.cuda.empty_cache()

    n_done += 1
    elapsed = time.perf_counter() - sweep_t0
    eta = elapsed / n_done * (len(runs) - n_done)
    print(f"  -> params={n_params:,} loss={final_loss:.4f} train_time={train_time:.0f}s "
          f"mini_fid={mfid:.2f} class_entropy={entropy:.2f} std_gen={std_gen:.3f} "
          f"dist_ratio={dist_ratio:.2f}", flush=True)
    print(f"  [{n_done}/{len(runs)} runs] ecoule={elapsed/60:.1f}min  ETA restante={eta/60:.1f}min", flush=True)

tsv_f.close()

# --------------------------------------------------------------------------- #
# Classement final + graphe d'importance (effet de chaque axe vs baseline, en mini-FID).
results.sort(key=lambda r: r["mini_fid"])
with open(os.path.join(args.results_dir, "summary.txt"), "w") as f:
    f.write(f"# BASELINE={BASELINE}\n")
    f.write(f"# {len(runs)} runs, {args.epochs} epochs chacun, MNIST complet, n_gen={args.n_gen}\n")
    f.write("# classement par mini-FID croissant (plus bas = plus proche des vraies images)\n")
    for row in results:
        f.write(f"{row['name']:<22}\tparams={row['params']:>9,}\tloss={row['final_loss']:.4f}\t"
                f"mini_fid={row['mini_fid']:7.2f}\tclass_entropy={row['class_entropy']:.2f}\t"
                f"std_gen={row['std_gen']:.3f}\tdist_ratio={row['dist_ratio']:.2f}\t"
                f"train_time={row['train_time_s']:.0f}s\n")

baseline_row = next(r for r in results if r["name"] == "baseline")
by_name = {r["name"]: r for r in results}
axis_labels, axis_deltas = [], []
for axis, values in AXES.items():
    for v in values:
        rname = f"{axis}={v}"
        if rname in by_name:
            axis_labels.append(rname)
            axis_deltas.append(by_name[rname]["mini_fid"] - baseline_row["mini_fid"])

if axis_labels:
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(axis_labels) + 1.5))
    colors = ["crimson" if d > 0 else "seagreen" for d in axis_deltas]
    ax.barh(axis_labels, axis_deltas, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"Δ mini-FID vs baseline ({baseline_row['mini_fid']:.2f})  "
                  f"[rouge = pire, vert = mieux]")
    ax.set_title(f"Importance des hyperparametres (OFAT, {args.epochs} epochs, MNIST complet)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "importance.png"), dpi=130)
    plt.close(fig)

print(f"\nTermine. Resultats dans {args.results_dir}/ "
      f"(sweep_results.tsv, summary.txt, importance.png, samples_*.png)", flush=True)
