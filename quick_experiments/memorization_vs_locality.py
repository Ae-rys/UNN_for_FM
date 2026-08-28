"""BOUCLAGE : localité -> expansivité -> mémorisation.
Relie directement la constante de Lipschitz L du débruiteur à une métrique de
MÉMORISATION (distance-au-train des échantillons générés), le long du sweep de localité P.

Chaîne à démontrer (tout analytique, aucun entraînement) :
  P petit (local)  -> L bas  -> samples LOIN du train (mosaïques créatives)
  P grand / global -> L haut -> samples COLLENT au train (mémorisation, dist->0)

Pour chaque P : (1) L = Lipschitz du débruiteur ELS à t=0.275 ;
                (2) génère nseed samples en intégrant le champ FM v=(E[x1|xt]-x)/(1-t) ;
                (3) distance L2 au plus proche voisin du TRAIN (=le dictionnaire).
Baseline = distance-au-train d'images TEST held-out (plancher d'un vrai voisin non mémorisé).
Endpoint global (ex1_is) = mémorisation pure (dist->0).

Sortie : memorization_vs_locality.png , _metrics.txt (logge claude.log)
Usage : ~/.venvs/unn/bin/python memorization_vs_locality.py
"""
import argparse, time, torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from nifty_els_fm import (mnist_train, build_dict, gauss_patch_weight,
                          ex1_els_nifty, ex1_is, euler_denoiser, DIM, dev)
from target_lipschitz import lipschitz_els, lipschitz_global

LOG = open("claude.log", "a")
def log(*a): print(*a, flush=True); print(*a, file=LOG, flush=True)
def med(x): return float(torch.as_tensor(x).median())


def test_set(n):
    import torchvision, torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    ds = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tf)
    return torch.stack([ds[i][0] for i in range(n)]).view(n, DIM).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsub", type=int, default=300)       # dictionnaire = train de référence
    ap.add_argument("--nseed", type=int, default=24)       # samples générés par P
    ap.add_argument("--nsteps", type=int, default=30)
    ap.add_argument("--nq", type=int, default=16)          # requêtes pour L
    ap.add_argument("--Ps", type=int, nargs="*", default=[3, 5, 7, 11, 19, 27])
    ap.add_argument("--tL", type=float, default=0.275)     # t où on lit l'expansivité
    args = ap.parse_args()
    log(f"\n===== BOUCLAGE localité->expansivité->mémorisation {time.strftime('%F %T')} =====")
    log(f"nsub={args.nsub} nseed={args.nseed} nsteps={args.nsteps} Ps={args.Ps}")
    t0 = time.time()

    Xsub = mnist_train(args.nsub, seed=0); X1 = Xsub.view(args.nsub, DIM)   # TRAIN
    Xq = mnist_train(args.nq, seed=1).view(args.nq, DIM)                    # requêtes L (held-out)
    x0 = torch.randn(args.nseed, DIM, device=dev)                          # bruit commun (seeds matchés)

    # baseline mémorisation : image TEST -> train
    Xte = test_set(args.nseed)
    base = med(torch.cdist(Xte, X1).min(1).values)
    log(f"[baseline] dist(TEST->train) médiane = {base:.2f}")

    # point requête pour L (fixe)
    xtL = args.tL * Xq + (1 - args.tL) * torch.randn_like(Xq)

    rows = []
    for P in args.Ps:
        pat, pn = build_dict(Xsub, P); gw = gauss_patch_weight(P)
        L = med(lipschitz_els(xtL, args.tL, pat, pn, P, gw, niter=8))
        gen = euler_denoiser(lambda x, t: ex1_els_nifty(x, t, pat, pn, P, gw),
                             x0.clone(), args.nsteps).clamp(-1, 1)
        dist = med(torch.cdist(gen, X1).min(1).values)
        rows.append(dict(P=P, L=L, dist=dist, ratio=dist / base))
        log(f"[P={P:2d}] L(t={args.tL})={L:6.2f} | dist(sample->train)={dist:6.2f} "
            f"| ratio/base={dist/base:.2f}  ({time.time()-t0:.0f}s)")

    # endpoint global (mémorisation)
    Lg = med(lipschitz_global(xtL, args.tL, X1))
    geng = euler_denoiser(lambda x, t: ex1_is(x, t, X1), x0.clone(), args.nsteps).clamp(-1, 1)
    distg = med(torch.cdist(geng, X1).min(1).values)
    rows.append(dict(P=99, L=Lg, dist=distg, ratio=distg / base))   # P=99 marque 'global'
    log(f"[GLOBAL] L={Lg:.2f} | dist={distg:.2f} | ratio/base={distg/base:.2f}")

    # ---------- figures ----------
    Pv = [r["P"] for r in rows[:-1]]; Lv = [r["L"] for r in rows]; Dv = [r["dist"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

    a = ax[0]; xlab = [str(p) for p in Pv] + ["global"]
    xpos = list(range(len(rows)))
    a.plot(xpos, Dv, "-o", color="C0", label="dist(sample→train)")
    a.axhline(base, color="C0", ls=":", label=f"baseline test→train ({base:.1f})")
    a.set_xticks(xpos); a.set_xticklabels(xlab)
    a.set_xlabel("taille de patch P  (← local | global →)"); a.set_ylabel("dist. au train (L2)", color="C0")
    a.tick_params(axis="y", labelcolor="C0")
    a2 = a.twinx(); a2.plot(xpos, Lv, "-s", color="C3", label="Lipschitz L(t=0.275)")
    a2.set_ylabel("Lipschitz L", color="C3"); a2.tick_params(axis="y", labelcolor="C3")
    a.set_title("Localité ↑ ⇒ expansivité L ↑ ET distance-au-train ↓\n(mémorisation)")
    a.grid(alpha=.3); a.legend(loc="upper right", fontsize=8); a2.legend(loc="center right", fontsize=8)

    a = ax[1]
    a.scatter(Lv[:-1], Dv[:-1], c=range(len(Pv)), cmap="viridis", s=90, zorder=3)
    for r, x in zip(rows[:-1], Lv[:-1]):
        a.annotate(f"P={r['P']}", (r["L"], r["dist"]), fontsize=8,
                   xytext=(4, 4), textcoords="offset points")
    a.scatter([Lv[-1]], [Dv[-1]], marker="*", s=260, color="C3", zorder=3, label="global (mémo)")
    a.axhline(base, color="C0", ls=":", label=f"baseline ({base:.1f})")
    a.set_xlabel("Lipschitz L (expansivité)"); a.set_ylabel("dist(sample→train)")
    a.set_title("Corrélation directe : plus expansif ⇒ plus proche du train"); a.legend(fontsize=8); a.grid(alpha=.3)

    plt.tight_layout(); plt.savefig("memorization_vs_locality.png", dpi=120)
    log("[fig] memorization_vs_locality.png")
    with open("memorization_vs_locality_metrics.txt", "w") as f:
        f.write(f"# localité->expansivité->mémorisation nsub={args.nsub} nseed={args.nseed} "
                f"nsteps={args.nsteps} | baseline test->train = {base:.3f}\n")
        f.write("P(99=global)\tL(t=0.275)\tdist_sample_train\tratio_over_baseline\n")
        for r in rows:
            f.write(f"{r['P']}\t{r['L']:.3f}\t{r['dist']:.3f}\t{r['ratio']:.3f}\n")
    log(f"[metrics] memorization_vs_locality_metrics.txt | fini en {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
