"""CONTRÔLE : déconfondre la falaise de mémorisation (patch généralise / global mémorise).
Trois ingrédients confondus entre 'GLOBAL' et 'patch P=27' : support (patch vs image entière),
augmentation du dictionnaire, et fold. On les sépare avec 5 débruiteurs analytiques,
mêmes seeds, distance-au-train comparée à la baseline test->train :

  1. ELS P=7  (patch + fold)            -> généralise (référence créative)
  2. ELS P=27 (patch + fold)            -> patch quasi-global AVEC fold
  3. ELS P=27 pixel-central (SANS fold) -> ablation du fold (patch sans agrégation)
  4. GLOBAL image-entière (N base)      -> mémorise (référence)
  5. GLOBAL + AUG. TRANSLATION          -> image-entière MAIS équivariante par translation
     (dict = N images × toutes translations ±T). Support global, pas de recombinaison locale.

Lecture :
  - si (5) mémorise encore (samples = chiffres entiers translatés, dist->0 vs dict augmenté)
    => l'équivariance SEULE ne suffit pas ; c'est le SUPPORT LOCAL (patch, recombinaison) qui crée.
  - si (5) généralise => l'équivariance par translation est le vrai levier, indépendamment du patch.
  - (3) vs (2) isole le rôle du fold.

Sortie : equivariance_control.png (barres + grille visuelle) , _metrics.txt (logge claude.log)
Usage : ~/.venvs/unn/bin/python equivariance_control.py
"""
import argparse, time, torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch.nn.functional as F
from nifty_els_fm import (mnist_train, build_dict, gauss_patch_weight, ex1_els_nifty,
                          ex1_els_center, ex1_is, euler_denoiser, DIM, S, dev)

LOG = open("claude.log", "a")
def log(*a): print(*a, flush=True); print(*a, file=LOG, flush=True)
def med(x): return float(torch.as_tensor(x).median())


def test_set(n):
    import torchvision, torchvision.transforms as T
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    ds = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tf)
    return torch.stack([ds[i][0] for i in range(n)]).view(n, DIM).to(dev)


def translate_aug(Ximg, T):
    """Ximg:(N,1,S,S) -> (N*(2T+1)^2, DIM) toutes translations entières ±T (remplissage 0)."""
    pads = F.pad(Ximg, (T, T, T, T), value=-1.0)                # -1 = fond (données dans [-1,1])
    outs = []
    for dy in range(-T, T + 1):
        for dx in range(-T, T + 1):
            outs.append(pads[..., T - dy:T - dy + S, T - dx:T - dx + S].reshape(Ximg.shape[0], DIM))
    return torch.cat(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsub", type=int, default=300)
    ap.add_argument("--nseed", type=int, default=24)
    ap.add_argument("--nsteps", type=int, default=30)
    ap.add_argument("--T", type=int, default=3)          # amplitude translation
    args = ap.parse_args()
    log(f"\n===== CONTRÔLE équivariance vs support local {time.strftime('%F %T')} =====")
    log(f"nsub={args.nsub} nseed={args.nseed} nsteps={args.nsteps} T={args.T}")
    t0 = time.time()

    Xsub = mnist_train(args.nsub, seed=0)                 # (N,1,S,S)
    X1 = Xsub.view(args.nsub, DIM)                        # TRAIN de base
    x0 = torch.randn(args.nseed, DIM, device=dev)         # bruit commun
    base = med(torch.cdist(test_set(args.nseed), X1).min(1).values)
    log(f"[baseline] dist(TEST->train) médiane = {base:.2f}")

    Xaug = translate_aug(Xsub, args.T)                    # dict image-entière équivariant
    log(f"[aug] dict translation : {Xaug.shape[0]:,} images entières (±{args.T}px)")

    pat7, pn7 = build_dict(Xsub, 7); gw7 = gauss_patch_weight(7)
    pat27, pn27 = build_dict(Xsub, 27); gw27 = gauss_patch_weight(27)
    cen = X1.reshape(-1).contiguous()                    # pixel central par patch (ordre = build_dict)

    arms = {
        "1. ELS P7 (patch+fold)":   lambda x, t: ex1_els_nifty(x, t, pat7, pn7, 7, gw7),
        "2. ELS P27 (patch+fold)":  lambda x, t: ex1_els_nifty(x, t, pat27, pn27, 27, gw27),
        "3. ELS P27 (sans fold)":   lambda x, t: ex1_els_center(x, t, pat27, pn27, cen, 27),
        "4. GLOBAL (image)":        lambda x, t: ex1_is(x, t, X1),
        "5. GLOBAL + transl.aug":   lambda x, t: ex1_is(x, t, Xaug),
    }
    res = {}
    samples = {}
    for name, fn in arms.items():
        gen = euler_denoiser(fn, x0.clone(), args.nsteps).clamp(-1, 1)
        samples[name] = gen
        d_base = med(torch.cdist(gen, X1).min(1).values)
        d_aug = med(torch.cdist(gen, Xaug).min(1).values)
        res[name] = dict(d_base=d_base, ratio=d_base / base, d_aug=d_aug)
        log(f"[{name}] dist->train={d_base:6.2f} (ratio {d_base/base:.2f}) | dist->dict_aug={d_aug:6.2f}"
            f"  ({time.time()-t0:.0f}s)")

    # ---------- figure ----------
    names = list(arms)
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.4], hspace=0.35)

    axb = fig.add_subplot(gs[0])
    ratios = [res[n]["ratio"] for n in names]
    cols = ["C2" if r > 0.9 else "C3" for r in ratios]
    axb.bar(range(len(names)), ratios, color=cols)
    axb.axhline(1.0, color="C0", ls=":", label="baseline test→train (=1, généralise)")
    axb.set_xticks(range(len(names))); axb.set_xticklabels(names, fontsize=8)
    axb.set_ylabel("dist→train / baseline")
    axb.set_title("Ratio distance-au-train (vert>0.9 généralise · rouge mémorise)")
    for i, r in enumerate(ratios):
        axb.annotate(f"{r:.2f}", (i, r), ha="center", va="bottom", fontsize=8)
    axb.legend(fontsize=8); axb.grid(alpha=.3, axis="y")

    nshow = 6
    axg = fig.add_subplot(gs[1]); axg.axis("off")
    grid = np.ones(((S + 2) * len(names), (S + 2) * nshow))
    for r, n in enumerate(names):
        for c in range(nshow):
            img = (samples[n][c].view(S, S).cpu().numpy() + 1) / 2
            grid[r*(S+2):r*(S+2)+S, c*(S+2):c*(S+2)+S] = img
    axg.imshow(grid, cmap="gray", vmin=0, vmax=1)
    for r, n in enumerate(names):
        axg.text(-4, r*(S+2)+S/2, n, ha="right", va="center", fontsize=8)
    axg.set_title("Échantillons (mêmes seeds) — mosaïque créative vs chiffre mémorisé", fontsize=9)

    plt.savefig("equivariance_control.png", dpi=120, bbox_inches="tight")
    log("[fig] equivariance_control.png")
    with open("equivariance_control_metrics.txt", "w") as f:
        f.write(f"# contrôle équivariance nsub={args.nsub} nseed={args.nseed} T={args.T} "
                f"| baseline test->train={base:.3f} | dict_aug={Xaug.shape[0]}\n")
        f.write("arm\tdist_train\tratio_base\tdist_dictaug\n")
        for n in names:
            f.write(f"{n}\t{res[n]['d_base']:.3f}\t{res[n]['ratio']:.3f}\t{res[n]['d_aug']:.3f}\n")
    log(f"[metrics] equivariance_control_metrics.txt | fini en {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
