"""Vue DÉBRUITEUR (fidèle au r² 0.85, sans l'artefact d'intégration indépendante) :
à des points xt pris SUR la trajectoire du modèle, à un t donné, on affiche
E[x1|xt] du MODÈLE vs ELS (NIFTY patch+fold, best-P) vs IS (mémo), avec cos(model,·).
C'est exactement l'objet que mesure le r² du discriminant.

Usage : python make_denoiser_view.py --name MinimalResNetFM_L6_ic256 \
    --ckpt results/temp-5/MinimalResNetFM_L6_ic256/model.pt --tag resnet_indep --ts 0.5 0.85
"""
import argparse, torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import nifty_els_fm as N
from run_mnist import build_experiments

PS = [3, 5, 7, 9, 11]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ts", type=float, nargs="*", default=[0.5, 0.85])
    ap.add_argument("--nseed", type=int, default=8)
    ap.add_argument("--nsteps", type=int, default=50)
    ap.add_argument("--nsub", type=int, default=2000)
    args = ap.parse_args()

    entry = [e for e in build_experiments(N.dev) if e["name"] == args.name][0]
    model = entry["build"](); model.load_state_dict(torch.load(args.ckpt, map_location=N.dev, weights_only=True)); model.eval()
    Xsub = N.mnist_train(args.nsub, seed=0); X1 = Xsub.view(args.nsub, N.DIM)
    dicts = {P: (*N.build_dict(Xsub, P), N.gauss_patch_weight(P)) for P in PS}

    torch.manual_seed(0)
    x0 = torch.randn(args.nseed, N.DIM, device=N.dev)
    _, traj = N.euler_model(model, x0.clone(), args.nsteps)
    tvals = np.array([t for (_, t) in traj])

    dn = lambda v: (v.view(N.S, N.S).cpu().numpy() + 1) / 2
    for tt in args.ts:
        k = int(np.argmin(np.abs(tvals - tt)))
        xt, t = traj[k]; omt = max(1.0 - t, 0.05)
        with torch.no_grad():
            ex1_m = xt + omt * N.model_velocity(model, xt, t)                     # débruiteur modèle
            is_d = N.ex1_is(xt, t, X1)                                            # débruiteur IS
            # best P par cos médian modèle<->ELS_P à ce t
            elsP = {P: N.ex1_els_nifty(xt, t, dicts[P][0], dicts[P][1], P, dicts[P][2]) for P in PS}
            medc = {P: float(N.cos_field(ex1_m, elsP[P]).median()) for P in PS}
            bestP = max(medc, key=medc.get)
            els_d = elsP[bestP]
        cE = N.cos_field(ex1_m, els_d).cpu().numpy()
        cI = N.cos_field(ex1_m, is_d).cpu().numpy()
        print(f"[{args.tag} t={t:.2f}] bestP={bestP}  cos(model,ELS)={np.median(cE):.3f}  "
              f"cos(model,IS)={np.median(cI):.3f}", flush=True)

        cols = [("x_t (entrée)", xt, None), ("modèle E[x1|xt]", ex1_m, None),
                (f"ELS E[x1|xt] (P={bestP})", els_d, cE), ("IS E[x1|xt] (mémo)", is_d, cI)]
        fig, ax = plt.subplots(args.nseed, 4, figsize=(7.5, 1.7 * args.nseed))
        for i in range(args.nseed):
            for j, (ttl, img, c) in enumerate(cols):
                ax[i, j].imshow(dn(img[i]), cmap="gray", vmin=0, vmax=1); ax[i, j].axis("off")
                if i == 0: ax[i, j].set_title(ttl, fontsize=9)
                if c is not None: ax[i, j].set_title((ttl if i == 0 else "") + f"\ncos={c[i]:.2f}", fontsize=7)
        plt.suptitle(f"{args.tag} — débruiteur à t={t:.2f}  |  "
                     f"médian cos(model,ELS)={np.median(cE):.2f}  cos(model,IS)={np.median(cI):.2f}",
                     fontsize=10)
        plt.tight_layout()
        fn = f"denoiser_view_{args.tag}_t{int(round(t*100)):02d}.png"
        plt.savefig(fn, dpi=120, bbox_inches="tight"); print(f"saved -> {fn}", flush=True)


if __name__ == "__main__":
    main()
