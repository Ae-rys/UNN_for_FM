"""Grille [ELS-FM | modèle-FM | IS-FM] par graine, générique pour ScCP/ResNet/UNet,
couplage indep ou OT. r²(modèle,ELS) et r²(modèle,IS) = DÉBRUITEUR le long de la
trajectoire (métrique FM correcte). best-P choisi par modèle (médiane sur graines).

Sauve comparison_mnist_<tag>.png. Le modèle est reconstruit via build_experiments
(même archi que l'entraînement) ; patch L1ProxConv w=8 (checkpoints ScCP sont w=8 ;
inoffensif pour ResNet/UNet).

Usage :
  python make_comparison_grid.py --name SmallUNet_baseline \
     --ckpt results/temp-5/SmallUNet_baseline/model.pt --tag unet_indep --title "UNet, indep"
"""
import argparse, torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import nifty_els_fm as N
from run_mnist import build_experiments

NSEED, NSTEPS = 8, 50
PS = [3, 5, 7, 9, 11]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)         # entrée build_experiments
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)          # -> comparison_mnist_<tag>.png
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    entry = [e for e in build_experiments(N.dev) if e["name"] == args.name][0]
    model = entry["build"]()
    model.load_state_dict(torch.load(args.ckpt, map_location=N.dev, weights_only=True))
    model.eval()
    print(f"[load] {args.name}  <- {args.ckpt}", flush=True)

    Xsub = N.mnist_train(2000, seed=0); X1 = Xsub.view(2000, N.DIM)
    dicts = {P: (*N.build_dict(Xsub, P), N.gauss_patch_weight(P)) for P in PS}

    torch.manual_seed(0)
    x0 = torch.randn(NSEED, N.DIM, device=N.dev)
    gen, traj = N.euler_model(model, x0.clone(), NSTEPS); gen = gen.clamp(-1, 1)

    tskip = [(x, t) for (x, t) in traj if 0.02 <= t <= 0.95]
    acc_els = {P: torch.zeros(NSEED, device=N.dev) for P in PS}
    acc_is = torch.zeros(NSEED, device=N.dev)
    for (x, t) in tskip:
        omt = max(1.0 - t, 0.05)
        with torch.no_grad():
            ex1_m = x + omt * N.model_velocity(model, x, t)
        acc_is += N.cos_field(ex1_m, N.ex1_is(x, t, X1))
        for P in PS:
            pat, pn, gw = dicts[P]
            acc_els[P] += N.cos_field(ex1_m, N.ex1_els_nifty(x, t, pat, pn, P, gw))
    n = len(tskip)
    med = {P: float((acc_els[P] / n).median()) for P in PS}
    bestP = max(med, key=med.get)
    rels = (acc_els[bestP] / n).cpu().numpy(); ris = (acc_is / n).cpu().numpy()
    print(f"  best P={bestP}  r²(ELS)={med[bestP]:.3f}  r²(IS)={float((acc_is/n).median()):.3f} "
          f"| tous P: {({P: round(med[P],3) for P in PS})}", flush=True)

    pat, pn, gw = dicts[bestP]
    els_img = N.euler_denoiser(lambda x, t: N.ex1_els_nifty(x, t, pat, pn, bestP, gw), x0.clone(), NSTEPS).clamp(-1, 1)
    is_img = N.euler_denoiser(lambda x, t: N.ex1_is(x, t, X1), x0.clone(), NSTEPS).clamp(-1, 1)

    dn = lambda v: (v.view(N.S, N.S).cpu().numpy() + 1) / 2
    fig, ax = plt.subplots(NSEED, 3, figsize=(6, 2.0 * NSEED))
    for i in range(NSEED):
        cols = [(els_img[i], f"ELS-FM\nr²={rels[i]:.2f}"),
                (gen[i], "modèle-FM"),
                (is_img[i], f"IS-FM\nr²={ris[i]:.2f}")]
        for j, (img, ttl) in enumerate(cols):
            ax[i, j].imshow(dn(img), cmap="gray", vmin=0, vmax=1); ax[i, j].axis("off")
            if j == 1:
                if i == 0: ax[i, j].set_title(ttl, fontsize=10)
            else:
                ax[i, j].set_title(ttl if i == 0 else ttl.split("\n")[-1], fontsize=8)
    ttl = args.title or args.tag
    plt.suptitle(f"{ttl} — ELS≈modèle vs IS (mémo)\n"
                 f"médian r²(ELS)={np.median(rels):.2f}  r²(IS)={np.median(ris):.2f}  "
                 f"(P={bestP}, ELS>IS {int((rels>ris).sum())}/{NSEED})", fontsize=10)
    plt.tight_layout(); plt.savefig(f"comparison_mnist_{args.tag}.png", dpi=130, bbox_inches="tight")
    print(f"saved -> comparison_mnist_{args.tag}.png", flush=True)


if __name__ == "__main__":
    main()
