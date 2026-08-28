"""AXE 'localité -> expansivité' (celui qui survit, voir memory track_nonexpansivity_convex_prox).

Thèse : la LOCALITÉ (taille de patch P / champ récepteur) est le bouton continu qui règle
l'EXPANSIVITÉ (Lipschitz du débruiteur), donc la position sur l'axe créativité<->mémorisation.
  P petit  -> débruiteur mosaïque ELS, peu expansif (L~3), généralise/crée.
  P grand  -> débruiteur global, très expansif (L~20-50), mémorise.
Prédiction : L(t) croît MONOTONE avec P, interpolant ELS-local -> mémorisation globale.

Deux volets, mêmes points xt (données held-out, couplage indep) :
  A. CIBLE analytique : sweep P du débruiteur ELS + asymptote globale (ex1_is).
  B. MODÈLES entraînés : ScCP k=3/K=6 (RF local) vs ScCP k=9/K=20 (RF global) — la
     localité côté archi (champ récepteur) doit reproduire la même hiérarchie de L.

Sortie : locality_expansivity.png , locality_expansivity_metrics.txt (logge claude.log)
Usage : ~/.venvs/unn/bin/python locality_expansivity.py
"""
import argparse, time, torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from nifty_els_fm import (mnist_train, build_dict, gauss_patch_weight,
                          load_model, model_velocity, DIM, dev)
from target_lipschitz import lipschitz_els, lipschitz_global
from model_lipschitz import lipschitz_denoiser

LOG = open("claude.log", "a")
def log(*a): print(*a, flush=True); print(*a, file=LOG, flush=True)
def q(x, p): return torch.quantile(x, p).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsub", type=int, default=400)
    ap.add_argument("--nq", type=int, default=16)
    ap.add_argument("--nt", type=int, default=9)
    ap.add_argument("--Ps", type=int, nargs="*", default=[3, 5, 7, 11, 15, 21, 27])
    args = ap.parse_args()
    log(f"\n===== AXE localité->expansivité {time.strftime('%F %T')} =====")
    log(f"nsub={args.nsub} nq={args.nq} nt={args.nt} Ps={args.Ps}")
    t0 = time.time()

    Xsub = mnist_train(args.nsub, seed=0); X1 = Xsub.view(args.nsub, DIM)
    Xq = mnist_train(args.nq, seed=1).view(args.nq, DIM)
    ts = np.linspace(0.05, 0.95, args.nt)
    # xt fixés une fois par t (mêmes pour tous P et modèles) pour comparabilité
    noise = {float(t): torch.randn_like(Xq) for t in ts}
    xts = {float(t): t * Xq + (1 - t) * noise[float(t)] for t in ts}

    # --- A. sweep P du débruiteur ELS ---
    dicts = {P: (*build_dict(Xsub, P), gauss_patch_weight(P)) for P in args.Ps}
    els_L = {P: [] for P in args.Ps}
    glob_L = []
    for t in ts:
        xt = xts[float(t)]
        glob_L.append(q(lipschitz_global(xt, float(t), X1), .5))
        for P in args.Ps:
            pat, pn, gw = dicts[P]
            els_L[P].append(q(lipschitz_els(xt, float(t), pat, pn, P, gw, niter=8), .5))
        log(f"[A] t={t:.3f} | " + " ".join(f"P{P}={els_L[P][-1]:.2f}" for P in args.Ps)
            + f" | GLOBAL={glob_L[-1]:.2f}")

    # --- B. ScCP entraîné local (k3) vs global (k9) ---
    sccp_k3 = load_model("results/temp-5/ConvScCP_k3_K6_ic128_L1_LNO/model.pt", K=6, ic=128, kernel=3)
    sccp_k9 = load_model("results/temp-5/ConvScCP_UNN_L1_LNO/model_20_128_0_1747.pt", K=20, ic=128, kernel=9)
    mvel = {"ScCP k=3 (RF local)": lambda x, t: model_velocity(sccp_k3, x, t),
            "ScCP k=9 (RF global)": lambda x, t: model_velocity(sccp_k9, x, t)}
    mod_L = {k: [] for k in mvel}
    for t in ts:
        xt = xts[float(t)]
        for k, fn in mvel.items():
            mod_L[k].append(q(lipschitz_denoiser(fn, xt, float(t)), .5))
        log(f"[B] t={t:.3f} | " + " ".join(f"{k}={mod_L[k][-1]:.2f}" for k in mvel))

    # ---------- figures ----------
    T_ = list(ts)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    # A1 : L(t) par P (dégradé) + global
    cmap = plt.cm.viridis(np.linspace(0, .9, len(args.Ps)))
    for c, P in zip(cmap, args.Ps):
        ax[0].plot(T_, els_L[P], "-o", color=c, ms=4, label=f"ELS P={P}")
    ax[0].plot(T_, glob_L, "--", color="C3", lw=2, label="GLOBAL (mémo)")
    ax[0].axhline(1, color="k", ls=":", lw=1.2)
    ax[0].set_yscale("log"); ax[0].set_xlabel("t"); ax[0].set_ylabel("Lipschitz local")
    ax[0].set_title("A. Cible ELS : L(t) croît avec la taille de patch P"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

    # A2 : L_peak (max sur t) vs P  -> le bouton
    Lpeak = [max(els_L[P]) for P in args.Ps]
    ax[1].plot(args.Ps, Lpeak, "-o", color="C0", label="ELS (patch P)")
    ax[1].axhline(max(glob_L), color="C3", ls="--", label="GLOBAL (mémo)")
    ax[1].axhline(1, color="k", ls=":", lw=1.2, label="seuil non-expansif")
    ax[1].set_xlabel("taille de patch P (localité)"); ax[1].set_ylabel(r"$L_{peak}=\max_t L$")
    ax[1].set_title("A. Bouton localité : expansivité vs P"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    # B : modèles entraînés local vs global
    ax[2].plot(T_, mod_L["ScCP k=3 (RF local)"], "-o", color="C0", label="ScCP k=3 (RF local)")
    ax[2].plot(T_, mod_L["ScCP k=9 (RF global)"], "-s", color="C3", label="ScCP k=9 (RF global)")
    ax[2].axhline(1, color="k", ls=":", lw=1.2)
    ax[2].set_xlabel("t"); ax[2].set_ylabel("Lipschitz local")
    ax[2].set_title("B. ScCP entraîné : RF local vs global"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

    plt.tight_layout(); plt.savefig("locality_expansivity.png", dpi=115)
    log("[fig] locality_expansivity.png")

    with open("locality_expansivity_metrics.txt", "w") as f:
        f.write(f"# AXE localité->expansivité nsub={args.nsub} nq={args.nq}\n")
        f.write("# A. ELS L_peak(P) et GLOBAL :\n")
        for P in args.Ps: f.write(f"P={P}\tL_peak={max(els_L[P]):.3f}\n")
        f.write(f"GLOBAL\tL_peak={max(glob_L):.3f}\n\n")
        f.write("# A. L(t) par P :\nt\t" + "\t".join(f"P{P}" for P in args.Ps) + "\tGLOBAL\n")
        for i, t in enumerate(ts):
            f.write(f"{t:.3f}\t" + "\t".join(f"{els_L[P][i]:.3f}" for P in args.Ps) + f"\t{glob_L[i]:.3f}\n")
        f.write("\n# B. ScCP entraîné k3(local) vs k9(global) :\nt\tk3\tk9\n")
        for i, t in enumerate(ts):
            f.write(f"{t:.3f}\t{mod_L['ScCP k=3 (RF local)'][i]:.3f}\t{mod_L['ScCP k=9 (RF global)'][i]:.3f}\n")
    log(f"[metrics] locality_expansivity_metrics.txt | fini en {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
