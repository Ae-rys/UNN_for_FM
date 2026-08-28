"""Grille [ELS-tf | modèle | IS-tf] TEACHER-FORCÉE : au lieu d'intégrer ELS/IS depuis le
bruit (endpoints qui divergent), on intègre leur champ v=(E[x1|xt]-xt)/(1-t) ÉVALUÉ AUX
POINTS DE LA TRAJECTOIRE DU MODÈLE. Si le débruiteur ELS ≈ celui du modèle, l'image ELS-tf
colle au modèle (comme le score de Kamb sur images finales). Isole la prédiction du
débruiteur, sans l'artefact de divergence EDO.

  x_ELS(1) = x0 + Σ_k dt · v_ELS(x_{t_k}^modele, t_k)   (idem IS)
  r² annoté = cos(image finale modèle, image finale ELS-tf / IS-tf)

Usage : python make_teacher_forced_grid.py --name MinimalResNetFM_L6_ic256 \
    --ckpt results/temp-5/MinimalResNetFM_L6_ic256/model.pt --tag resnet_indep
"""
import argparse, torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import nifty_els_fm as N
from run_mnist import build_experiments

NSEED, NSTEPS = 8, 50
PS = [3, 5, 7, 9, 11]


def teacher_forced(x0, traj, denoiser_fn, dt):
    """x0 + Σ dt·(D(xm,t)-xm)/(1-t), D évalué aux points xm de la trajectoire modèle."""
    x = x0.clone()
    for (xm, t) in traj:
        omt = max(1.0 - t, 0.05)
        x = x + dt * (denoiser_fn(xm, t) - xm) / omt
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    entry = [e for e in build_experiments(N.dev) if e["name"] == args.name][0]
    model = entry["build"](); model.load_state_dict(torch.load(args.ckpt, map_location=N.dev, weights_only=True)); model.eval()
    Xsub = N.mnist_train(2000, seed=0); X1 = Xsub.view(2000, N.DIM)
    dicts = {P: (*N.build_dict(Xsub, P), N.gauss_patch_weight(P)) for P in PS}

    torch.manual_seed(0)
    x0 = torch.randn(NSEED, N.DIM, device=N.dev)
    gen, traj = N.euler_model(model, x0.clone(), NSTEPS); gen = gen.clamp(-1, 1)
    dt = 1.0 / NSTEPS

    # best P : cos débruiteur médian sur la trajectoire
    tskip = [(x, t) for (x, t) in traj if 0.02 <= t <= 0.95]
    accP = {P: torch.zeros(NSEED, device=N.dev) for P in PS}
    with torch.no_grad():
        for (x, t) in tskip:
            ex1_m = x + max(1 - t, 0.05) * N.model_velocity(model, x, t)
            for P in PS:
                accP[P] += N.cos_field(ex1_m, N.ex1_els_nifty(x, t, dicts[P][0], dicts[P][1], P, dicts[P][2]))
    bestP = max(PS, key=lambda P: float((accP[P] / len(tskip)).median()))
    pat, pn, gw = dicts[bestP]

    with torch.no_grad():
        els_tf = teacher_forced(x0, traj, lambda x, t: N.ex1_els_nifty(x, t, pat, pn, bestP, gw), dt).clamp(-1, 1)
        is_tf = teacher_forced(x0, traj, lambda x, t: N.ex1_is(x, t, X1), dt).clamp(-1, 1)
    rE = N.cos_field(gen, els_tf).cpu().numpy()      # r² IMAGE finale (teacher-forcé)
    rI = N.cos_field(gen, is_tf).cpu().numpy()
    print(f"[{args.tag}] bestP={bestP}  image-r²(model,ELS-tf)={np.median(rE):.3f}  "
          f"image-r²(model,IS-tf)={np.median(rI):.3f}", flush=True)

    dn = lambda v: (v.view(N.S, N.S).cpu().numpy() + 1) / 2
    fig, ax = plt.subplots(NSEED, 3, figsize=(6, 2.0 * NSEED))
    for i in range(NSEED):
        cols = [(els_tf[i], f"ELS-tf\nr²={rE[i]:.2f}"), (gen[i], "modèle-FM"),
                (is_tf[i], f"IS-tf\nr²={rI[i]:.2f}")]
        for j, (img, ttl) in enumerate(cols):
            ax[i, j].imshow(dn(img), cmap="gray", vmin=0, vmax=1); ax[i, j].axis("off")
            if j == 1:
                if i == 0: ax[i, j].set_title(ttl, fontsize=10)
            else:
                ax[i, j].set_title(ttl if i == 0 else ttl.split("\n")[-1], fontsize=8)
    ttl = args.title or args.tag
    plt.suptitle(f"{ttl} — TEACHER-FORCÉ (ELS/IS intégrés le long de la traj. modèle)\n"
                 f"image-r²(model,ELS)={np.median(rE):.2f}  image-r²(model,IS)={np.median(rI):.2f}  (P={bestP})",
                 fontsize=10)
    plt.tight_layout(); plt.savefig(f"teacher_forced_{args.tag}.png", dpi=130, bbox_inches="tight")
    print(f"saved -> teacher_forced_{args.tag}.png", flush=True)


if __name__ == "__main__":
    main()
