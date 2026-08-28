"""MESURE 1 (piste 'prix de la convexité', voir memory track_nonexpansivity_convex_prox).

Question : la CIBLE du Flow Matching (débruiteur optimal E[x1|xt]) est-elle hors de la
classe fermement non-expansive (= la classe de ScCP-l1 à convergence, prox convexe :
Jacobien symétrique PSD, valeurs propres dans [0,1]) ?

Objet mesuré, par bin de t, en COUPLAGE INDÉPENDANT (terrain sain, immunisé au bug OT) :
  Cov(x1 | xt)                          estimée par kNN dans l'espace xt
  J(xt) = t/(1-t)^2 * Cov(x1|xt)        Jacobien du débruiteur (Tweedie)
  lambda_max(J)                          > 1  <=>  cible localement EXPANSIVE (hors classe prox)
  anisotropie (participation ratio, frac top-1)  = ce que voit le r² (cos centré),
                                          invariant au gain uniforme.

Point de méthode : l'expansivité est LOCALISÉE (frontières entre modes de patches).
Donc on rapporte la DISTRIBUTION de lambda_max sur les requêtes (médiane + hauts
percentiles), pas seulement la moyenne : le signal est dans la queue.

Biais connu du kNN : la cov conditionnelle estimée sur une boule de rayon fini inclut
la variation de la moyenne conditionnelle sur la boule -> SURESTIME un peu l'expansivité.
=> c'est un majorant : si lambda_max(J) ne dépasse PAS 1 même ici, la cible est
genuinement non-expansive à ce régime.

Sortie : target_jacobian_spectrum.png , target_jacobian_metrics.txt (logge dans claude.log)
Usage : ~/.venvs/unn/bin/python target_jacobian_spectrum.py --npool 12000 --k 64 --nq 800
"""
import argparse, time, torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torchvision, torchvision.transforms as T

dev = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 784
LOG = open("claude.log", "a")
def log(*a): print(*a, flush=True); print(*a, file=LOG, flush=True)


def mnist(n):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=tf)
    idx = torch.randperm(len(ds))[:n]
    return torch.stack([ds[i][0] for i in idx]).view(n, DIM).to(dev)


@torch.no_grad()
def cond_cov_spectrum(xt, x1, nq, k, topm=8):
    """Pour nq requêtes : kNN en xt, spectre de Cov(x1 | voisins).
    Renvoie par requête : lambda_max(Cov), trace(Cov), participation ratio (Σλ)²/Σλ²,
    et frac_top1 = lambda_max/trace. SVD batchée sur les blocs de voisins centrés."""
    N = xt.shape[0]
    qi = torch.randperm(N, device=dev)[:nq]
    lam_max, tr, pr, ftop1 = [], [], [], []
    for s in range(0, nq, 64):                             # chunk requêtes (mémoire)
        q = qi[s:s+64]
        D = torch.cdist(xt[q], xt)                         # (c, N)
        nn = D.topk(k, largest=False).indices              # (c, k) plus proches voisins en xt
        Y = x1[nn]                                          # (c, k, DIM) leurs x1
        Y = Y - Y.mean(dim=1, keepdim=True)                # centrer sur les k voisins
        sv = torch.linalg.svdvals(Y)                       # (c, min(k,DIM)) valeurs singulières
        eig = sv**2 / (k - 1)                               # valeurs propres de Cov
        lam_max.append(eig[:, 0])
        tr.append(eig.sum(dim=1))
        pr.append(eig.sum(dim=1)**2 / (eig**2).sum(dim=1)) # participation ratio (rang effectif)
        ftop1.append(eig[:, 0] / eig.sum(dim=1))
    return (torch.cat(lam_max), torch.cat(tr), torch.cat(pr), torch.cat(ftop1))


def pct(x, p):  # percentile robuste
    return torch.quantile(x, p).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npool", type=int, default=12000)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--nq", type=int, default=800)
    ap.add_argument("--nt", type=int, default=19)
    args = ap.parse_args()

    log(f"\n===== MESURE 1 : spectre Jacobien cible kNN (indep) {time.strftime('%F %T')} =====")
    log(f"npool={args.npool} k={args.k} nq={args.nq} nt={args.nt} dev={dev}")
    t0 = time.time()
    x1 = mnist(args.npool)

    # cov marginale (contrôle : à petit t, Cov(x1|xt) -> Cov(x1))
    xm = x1 - x1.mean(0, keepdim=True)
    marg_lam = (torch.linalg.svdvals(xm)**2 / (args.npool - 1))[0].item()
    log(f"[controle] lambda_max(Cov marginale x1) = {marg_lam:.3f}")

    ts = np.linspace(0.05, 0.95, args.nt)
    rows = []
    for t in ts:
        x0 = torch.randn_like(x1)                          # COUPLAGE INDÉPENDANT
        xt = t * x1 + (1 - t) * x0
        lam, tr, pr, ftop1 = cond_cov_spectrum(xt, x1, args.nq, args.k)
        fac = t / (1 - t) ** 2                              # Tweedie : J = fac * Cov
        Jmax = lam * fac
        row = dict(t=float(t), fac=float(fac),
                   covlam_med=pct(lam, .5), covlam_p90=pct(lam, .9),
                   Jmax_med=pct(Jmax, .5), Jmax_p90=pct(Jmax, .9),
                   Jmax_p99=pct(Jmax, .99), Jmax_max=Jmax.max().item(),
                   frac_expansive=(Jmax > 1).float().mean().item(),  # part des pts où lambda_max(J)>1
                   pr_med=pct(pr, .5), ftop1_med=pct(ftop1, .5))
        rows.append(row)
        log(f"t={t:.3f} fac={fac:7.2f} | Jmax med={row['Jmax_med']:.3f} "
            f"p90={row['Jmax_p90']:.3f} p99={row['Jmax_p99']:.3f} max={row['Jmax_max']:.2f} "
            f"| frac(J>1)={row['frac_expansive']:.3f} | PR={row['pr_med']:.1f} "
            f"ftop1={row['ftop1_med']:.3f}")

    # ---- figure ----
    T_ = [r["t"] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    a = ax[0, 0]
    a.plot(T_, [r["Jmax_med"] for r in rows], "-o", label="médiane")
    a.plot(T_, [r["Jmax_p90"] for r in rows], "-s", label="p90")
    a.plot(T_, [r["Jmax_p99"] for r in rows], "-^", label="p99")
    a.axhline(1.0, color="r", ls="--", lw=1.5, label="seuil non-expansif = 1")
    a.set_yscale("log"); a.set_xlabel("t"); a.set_ylabel(r"$\lambda_{max}(J)$")
    a.set_title(r"Jacobien débruiteur $J=\frac{t}{(1-t)^2}\mathrm{Cov}(x_1|x_t)$"
                "\n(au-dessus de 1 = hors classe prox convexe)")
    a.legend(); a.grid(alpha=.3)

    a = ax[0, 1]
    a.plot(T_, [r["frac_expansive"] for r in rows], "-o", color="C3")
    a.set_xlabel("t"); a.set_ylabel("fraction des points requête avec $\\lambda_{max}(J)>1$")
    a.set_title("Où (en t) la cible est-elle localement expansive ?")
    a.grid(alpha=.3)

    a = ax[1, 0]
    a.plot(T_, [r["covlam_med"] for r in rows], "-o", label="médiane")
    a.plot(T_, [r["covlam_p90"] for r in rows], "-s", label="p90")
    a.axhline(marg_lam, color="gray", ls=":", label=f"marginale ({marg_lam:.1f})")
    a.set_xlabel("t"); a.set_ylabel(r"$\lambda_{max}(\mathrm{Cov}(x_1|x_t))$")
    a.set_title("Top valeur propre de la covariance conditionnelle"); a.legend(); a.grid(alpha=.3)

    a = ax[1, 1]
    a.plot(T_, [r["pr_med"] for r in rows], "-o", color="C2", label="participation ratio (rang eff.)")
    a.set_xlabel("t"); a.set_ylabel("PR = $(\\Sigma\\lambda)^2/\\Sigma\\lambda^2$", color="C2")
    a2 = a.twinx()
    a2.plot(T_, [r["ftop1_med"] for r in rows], "-s", color="C4", label="frac top-1")
    a2.set_ylabel("frac trace dans top-1", color="C4")
    a.set_title("Anisotropie (bas PR / haute frac top-1 = engagement de mode)\n= ce que voit le r²")
    a.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig("target_jacobian_spectrum.png", dpi=110)
    log(f"[fig] target_jacobian_spectrum.png")

    with open("target_jacobian_metrics.txt", "w") as f:
        f.write(f"# MESURE 1 spectre Jacobien cible kNN (indep) npool={args.npool} k={args.k} nq={args.nq}\n")
        f.write(f"# lambda_max(Cov marginale) = {marg_lam:.4f}\n")
        f.write("t\tfac\tJmax_med\tJmax_p90\tJmax_p99\tJmax_max\tfrac_J>1\tPR\tftop1\n")
        for r in rows:
            f.write(f"{r['t']:.3f}\t{r['fac']:.3f}\t{r['Jmax_med']:.4f}\t{r['Jmax_p90']:.4f}\t"
                    f"{r['Jmax_p99']:.4f}\t{r['Jmax_max']:.4f}\t{r['frac_expansive']:.4f}\t"
                    f"{r['pr_med']:.2f}\t{r['ftop1_med']:.4f}\n")
    log(f"[metrics] target_jacobian_metrics.txt  |  fini en {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
