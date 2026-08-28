# -*- coding: utf-8 -*-
"""
make_mnist_report_figs.py
Figures et chiffres MNIST du rapport, depuis results/mnist_report_100ep/.

Produit :
  internship_report/images/LNO_generated_4.png          (fig:LNO_mnist)
  internship_report/images/LNO_ot_trajectory_100_epochs.png  (fig:LNO_mnist_traj)
  internship_report/images/LFO_ot_trajectory_100_epochs.png  (fig:LFO_mnist_traj)
  stdout : la colonne FID de tab:quantitative (mini-FID, cf. subsec:mnist_fid)

Echantillonnage : Euler-100, PAS dopri5. Ces modeles ont t_max=None, donc la
conversion x-pred -> vitesse passe par clamp(1-t, 0.05) ; un solveur adaptatif
evalue des t arbitrairement proches de 1, ou le clamp ecrase la vitesse.

Le mini-FID est celui de mnist_metrics (espace de features d'un petit
classifieur MNIST, pas InceptionV3) : comparable entre CES runs uniquement.
Son plancher a n=2000 est ~4, cf. subsec:mnist_fid — un ecart de moins de
quelques unites n'est pas interpretable.

Usage :  python make_mnist_report_figs.py [--n-eval 2000]
"""
import argparse, os, shutil
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
from models.architectures import ConvScCP_UNN, ConvDFB_UNN, SmallUNet
from generate_digits import infer_config
from run_mnist import get_train_loader
from mnist_metrics import train_or_load_classifier, mini_fid

RUN = "results/mnist_report_100ep"   # surchargeable par --run
OUT = "internship_report/images"
DIM, IMG = 784, 28


def load(name):
    """Reconstruit le modele depuis son state_dict. ConvScCP et ConvDFB ont la MEME
    structure de couche (W_weight/V_weight/prox), donc infer_config() marche pour les
    deux ; seule la classe a instancier change, et elle est lue dans le nom."""
    sd = torch.load(os.path.join(RUN, name, "model.pt"), map_location="cpu", weights_only=False)
    if name.startswith("SmallUNet"):
        m = SmallUNet(base_ch=32)
    else:
        c = infer_config(sd)
        common = dict(dim=c["dim"], K=c["K"], internal_channel=c["internal_channel"],
                      use_Unet=c["use_Unet"], version=c["version"],
                      use_checkpoint=False, w_bias=c["w_bias"], prox_w=c["prox_w"])
        if name.startswith("ConvDFB"):
            # ConvDFB_UNN n'expose ni kernel_size ni in_channels/img_size : son noyau
            # est fixe a la construction. Passer ces kwargs leve un TypeError.
            m = ConvDFB_UNN(**common)
        else:
            m = ConvScCP_UNN(kernel_size=c["kernel"], in_channels=c["in_channels"],
                             img_size=c["img_size"], **common)
    m.load_state_dict(sd)
    return m.eval()


def discover(run_dir):
    """Tous les modeles entraines presents dans le run, ordre lisible."""
    names = sorted(d for d in os.listdir(run_dir)
                   if os.path.exists(os.path.join(run_dir, d, "model.pt")))
    pretty = {"ConvScCP_UNN_L1_LFO": "ScCP LFO", "ConvScCP_UNN_L1_LNO": "ScCP LNO",
              "ConvDFB_UNN_L1_LFO": "DFB LFO",  "ConvDFB_UNN_L1_LNO": "DFB LNO",
              "SmallUNet_baseline": "SmallUNet"}
    order = list(pretty)
    names.sort(key=lambda n: order.index(n) if n in order else 99)
    return [(n, pretty.get(n, n)) for n in names]


@torch.no_grad()
def euler_sample(model, n, device, n_steps=100, seed=0, batch=500):
    """Euler explicite sur [0,1]. Le modele renvoie deja une VITESSE en eval."""
    g = torch.Generator().manual_seed(seed)
    outs, done = [], 0
    while done < n:
        b = min(batch, n - done)
        x = torch.randn(b, DIM, generator=g).to(device)
        for i in range(n_steps):
            t = torch.full((b, 1), i / n_steps, device=device)
            x = x + model(torch.cat([x, t], dim=-1)) / n_steps
        outs.append(x.cpu()); done += b
    return torch.cat(outs, 0)[:n]


PAIR_H = 4.8   # hauteur (pouces) PARTAGEE par la courbe de loss et la grille de
               # chiffres : \includegraphics[height=...] identique des deux cotes
               # => cadres alignes quand les deux figures sont cote a cote.


def save_grid(imgs, path, title="", nrow=6, ncol=6):
    """Grille CARREE (6x6) : meme hauteur que la courbe de loss, donc appariable.
    Pas de titre interne — c'est la legende LaTeX qui porte l'info."""
    fig, axes = plt.subplots(nrow, ncol, figsize=(PAIR_H, PAIR_H))
    for a, im in zip(axes.flat, imgs):
        a.imshow(im.view(IMG, IMG).clamp(-1, 1), cmap="gray", vmin=-1, vmax=1)
        a.axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    print("ecrit :", path)


def loss_curve(out_path):
    """Courbes de loss MNIST par famille, echelle log, legende dans le cadre,
    sans titre interne. Meme hauteur que la grille."""
    import plot_losses as pl
    orig_plot, orig_save = pl._plot_by_algo, pl._save_or_show

    def inside(ax, data, title, legend_outside=False, legend_fontsize=6.5, **kw):
        orig_plot(ax, data, "", legend_outside=False, legend_fontsize=9, **kw)
        ax.get_legend().get_frame().set_alpha(0.9)

    def log_save(fig, path):
        for ax in fig.get_axes():
            if ax.lines:
                ax.set_yscale("log")
                ax.grid(True, which="both", alpha=0.3)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print("ecrit :", path)

    pl._plot_by_algo, pl._save_or_show = inside, log_save
    try:
        pl.plot_losses_by_algo(os.path.abspath(RUN), output_path=out_path,
                               figsize=(6.4, PAIR_H))
    finally:
        pl._plot_by_algo, pl._save_or_show = orig_plot, orig_save


def main():
    global RUN
    p = argparse.ArgumentParser()
    p.add_argument("--n-eval", type=int, default=2000)
    p.add_argument("--run", type=str, default=RUN, help="dossier du run a depouiller")
    p.add_argument("--suffix", type=str, default="", help="suffixe des PNG produits, ex '_K10'")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    RUN = args.run
    sfx = args.suffix
    os.makedirs(OUT, exist_ok=True)

    # lot reel : MNIST complet, meme normalisation [-1,1] que l'entrainement
    loader = get_train_loader(batch_size=256, digit=None)
    clf = train_or_load_classifier(loader, dev)
    # Lot reel FIXE (pas de shuffle) : le loader d'entrainement melange, et prendre
    # ses premiers lots donnait un lot reel different a chaque appel — d'ou un
    # mini-FID qui bougeait de plusieurs unites entre deux executions identiques.
    ds = loader.dataset
    real = torch.stack([ds[i][0] for i in range(args.n_eval)]).view(-1, 1, IMG, IMG).clamp(-1, 1)

    print(f"\n--- MNIST, 100 epochs, 46 900 iterations (tab:quantitative) ---")
    print(f"{'Model':<26}{'#Params':>10}{'Loss':>9}{'mini-FID':>11}")
    rows = discover(RUN)
    print(f"modeles trouves : {', '.join(l for _, l in rows)}")
    for name, label in rows:
        if not os.path.exists(os.path.join(RUN, name, "model.pt")):
            print(f"{label:<26}{'absent de ce run':>30}")
            continue
        model = load(name).to(dev)
        gen = euler_sample(model, args.n_eval, dev)
        fid = mini_fid(clf, gen.view(-1, 1, IMG, IMG).clamp(-1, 1), real, dev)
        n_par = sum(p_.numel() for p_ in model.parameters())
        loss = float(open(os.path.join(RUN, name, "loss.txt")).read().strip().split("\n")[-1].split("\t")[1])
        print(f"{label:<26}{n_par:>10,}{loss:>9.4f}{fid:>11.2f}")
        tag = {"ConvScCP_UNN_L1_LNO": "LNO", "ConvScCP_UNN_L1_LFO": "LFO",
               "ConvDFB_UNN_L1_LNO": "DFB_LNO", "ConvDFB_UNN_L1_LFO": "DFB_LFO",
               "SmallUNet_baseline": "SmallUNet"}.get(name, name)
        save_grid(gen[:36], os.path.join(OUT, f"{tag}_generated{sfx}.png"))
        if name == "ConvScCP_UNN_L1_LNO":       # nom historique attendu par le rapport
            save_grid(gen[:36], os.path.join(OUT, f"LNO_generated_4{sfx}.png"))
        del model
        torch.cuda.empty_cache()

    # trajectoires : deja produites par trajectory_convsccp.py, on les publie
    for v, sub in [("LNO", "ConvScCP_UNN_L1_LNO"), ("LFO", "ConvScCP_UNN_L1_LFO"),
                   ("DFB_LNO", "ConvDFB_UNN_L1_LNO"), ("DFB_LFO", "ConvDFB_UNN_L1_LFO")]:
        src = os.path.join(RUN, sub, "trajectory", "iterates_sample0.png")
        if os.path.exists(src):
            dst = os.path.join(OUT, f"{v}_ot_trajectory_100_epochs{sfx}.png")
            shutil.copyfile(src, dst); print("ecrit :", dst)
        else:
            print(f"(pas de trajectoire pour {v})")

    loss_curve(os.path.join(OUT, f"mnist_algo{sfx}.png"))

    print(f"\nRappel : plancher du mini-FID a n={args.n_eval} ~ 4 (subsec:mnist_fid).")


if __name__ == "__main__":
    main()
