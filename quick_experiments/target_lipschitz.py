"""MESURE 1 (v2, EXACTE — remplace target_jacobian_spectrum.py qui échouait par
malédiction de la dimension : le kNN en pixel ne conditionnait rien).

On calcule la NORME D'OPÉRATEUR du Jacobien du débruiteur  L(xt) = ‖∂E[x1|xt]/∂xt‖
(= constante de Lipschitz locale) le long de xt réalistes, par bin de t.
Rappel : un opérateur fermement non-expansif (classe de ScCP-l1 à convergence, prox
convexe) a L ≤ 1 PARTOUT. Donc L > 1 <=> cible hors de la classe convexe.

Deux débruiteurs analytiques (forme close, mêmes que nifty_els_fm.py) :
  - GLOBAL / IS (mémorisation)  : E[x1|xt] = Σ_i x1_i softmax(...). J = coef·Cov_pondérée
    -> J symétrique PSD, lambda_max EXACT par power-iteration matrix-free.
  - ELS patch-local (NIFTY)     : le débruiteur mosaïque de Kamb. J via power-iteration
    à DIFFÉRENCES FINIES (Jv ≈ (f(x+εv)-f(x-εv))/2ε), quasi-exact (champ ~conservatif).

Enjeu : c'est le débruiteur ELS que ScCP doit reproduire (r²≈0.83). Prédiction du fil
'prix de la convexité' :
  * GLOBAL fortement expansif (L≫1) dans une bande de t -> une archi convexe NE PEUT PAS
    mémoriser (et ne doit pas, pour la créativité) ;
  * ELS patch-local BEAUCOUP moins expansif (la localité 'unimodalise' les patches) ->
    si L_ELS reste ~≤1, la contrainte convexe coûte peu contre ELS => explique ScCP≈ResNet
    sur le r² ELS. Si L_ELS>1 dans une bande, c'est là que ScCP doit décrocher.

Sortie : target_lipschitz.png , target_lipschitz_metrics.txt (logge dans claude.log)
Usage : ~/.venvs/unn/bin/python target_lipschitz.py --nsub 2000 --nq 32 --nt 15
"""
import argparse, time, torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from nifty_els_fm import (mnist_train, build_dict, gauss_patch_weight,
                          ex1_els_nifty, ex1_is, DIM, S, dev)

LOG = open("claude.log", "a")
def log(*a): print(*a, flush=True); print(*a, file=LOG, flush=True)


@torch.no_grad()
def lipschitz_global(x, t, X1, niter=25):
    """lambda_max EXACT de J = coef·Cov_pondérée, par power-iteration matrix-free.
    x:(b,DIM) points requête ; X1:(Ntr,DIM) dictionnaire. J symétrique PSD."""
    b = x.shape[0]; omt = max(1.0 - t, 1e-2)
    coef = t / omt ** 2; quad = (t ** 2) / (2 * omt ** 2)
    nj = (X1 ** 2).sum(1)
    logit = coef * (x @ X1.t()) - quad * nj[None, :]        # (b, Ntr)
    w = torch.softmax(logit, dim=1)                         # poids softmax du débruiteur
    d = w @ X1                                              # (b, DIM) = E[x1|xt]
    v = torch.randn(b, DIM, device=dev); v /= v.norm(dim=1, keepdim=True)
    lam = torch.zeros(b, device=dev)
    for _ in range(niter):
        s = v @ X1.t()                                      # (b, Ntr) : X1_j · v_i
        Cv = (w * s) @ X1 - d * (d * v).sum(1, keepdim=True)  # Cov_i · v_i
        Jv = coef * Cv
        lam = (v * Jv).sum(1)                               # quotient de Rayleigh
        nrm = Jv.norm(dim=1, keepdim=True) + 1e-30
        v = Jv / nrm
    return lam                                              # (b,) lambda_max(J) par requête


@torch.no_grad()
def lipschitz_els(x, t, pat, pn, P, gw, niter=10, eps=1e-3):
    """lambda_max de J du débruiteur ELS patch-local, power-iteration à diff. finies.
    (J ~symétrique car champ ~conservatif -> power-iter de J donne |lambda_max|≈‖J‖.)"""
    b = x.shape[0]
    f = lambda z: ex1_els_nifty(z, t, pat, pn, P, gw)
    v = torch.randn(b, DIM, device=dev); v /= v.norm(dim=1, keepdim=True)
    lam = torch.zeros(b, device=dev)
    for _ in range(niter):
        Jv = (f(x + eps * v) - f(x - eps * v)) / (2 * eps)  # dérivée directionnelle
        lam = (v * Jv).sum(1)
        nrm = Jv.norm(dim=1, keepdim=True) + 1e-30
        v = Jv / nrm
    return lam


def q(x, p): return torch.quantile(x, p).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsub", type=int, default=2000)      # taille dictionnaire
    ap.add_argument("--nq", type=int, default=32)          # points requête (held-out)
    ap.add_argument("--nt", type=int, default=15)
    ap.add_argument("--P", type=int, default=7)            # taille patch ELS (mid, ~calibré)
    args = ap.parse_args()

    log(f"\n===== MESURE 1 v2 : Lipschitz local du débruiteur (indep) {time.strftime('%F %T')} =====")
    log(f"nsub={args.nsub} nq={args.nq} nt={args.nt} P_ELS={args.P} dev={dev}")
    t0 = time.time()
    Xsub = mnist_train(args.nsub, seed=0)                   # dictionnaire
    X1 = Xsub.view(args.nsub, DIM)
    Xq = mnist_train(args.nq, seed=1).view(args.nq, DIM)    # requêtes HELD-OUT (seed≠dict)
    pat, pn = build_dict(Xsub, args.P); gw = gauss_patch_weight(args.P)
    log(f"[dict] P={args.P}: {pat.shape[0]:,} patches")

    ts = np.linspace(0.05, 0.95, args.nt)
    rows = []
    for t in ts:
        x0 = torch.randn_like(Xq)                           # COUPLAGE INDÉPENDANT
        xt = t * Xq + (1 - t) * x0
        Lg = lipschitz_global(xt, float(t), X1)
        Le = lipschitz_els(xt, float(t), pat, pn, args.P, gw)
        row = dict(t=float(t),
                   Lg_med=q(Lg, .5), Lg_p90=q(Lg, .9), Lg_max=Lg.max().item(),
                   Le_med=q(Le, .5), Le_p90=q(Le, .9), Le_max=Le.max().item(),
                   fe_g=(Lg > 1).float().mean().item(), fe_e=(Le > 1).float().mean().item())
        rows.append(row)
        log(f"t={t:.3f} | GLOBAL L med={row['Lg_med']:8.2f} p90={row['Lg_p90']:8.2f} "
            f"frac>1={row['fe_g']:.2f}  ||  ELS-P{args.P} L med={row['Le_med']:.3f} "
            f"p90={row['Le_p90']:.3f} max={row['Le_max']:.3f} frac>1={row['fe_e']:.2f}")

    T_ = [r["t"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]
    a.plot(T_, [r["Lg_med"] for r in rows], "-o", color="C3", label="GLOBAL (mémo) médiane")
    a.plot(T_, [r["Lg_p90"] for r in rows], "--", color="C3", alpha=.6, label="GLOBAL p90")
    a.plot(T_, [r["Le_med"] for r in rows], "-o", color="C0", label=f"ELS P={args.P} médiane")
    a.plot(T_, [r["Le_p90"] for r in rows], "--", color="C0", alpha=.6, label="ELS p90")
    a.axhline(1.0, color="k", ls=":", lw=1.5, label="seuil non-expansif = 1")
    a.set_yscale("log"); a.set_xlabel("t"); a.set_ylabel(r"$\|\partial E[x_1|x_t]/\partial x_t\|$")
    a.set_title("Lipschitz local du débruiteur (log)\nau-dessus de 1 = hors classe prox convexe")
    a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[1]
    a.plot(T_, [r["Le_med"] for r in rows], "-o", color="C0", label="médiane")
    a.plot(T_, [r["Le_p90"] for r in rows], "-s", color="C0", alpha=.6, label="p90")
    a.plot(T_, [r["Le_max"] for r in rows], "-^", color="C0", alpha=.4, label="max")
    a.axhline(1.0, color="k", ls=":", lw=1.5, label="seuil = 1")
    a.set_xlabel("t"); a.set_ylabel("Lipschitz ELS patch-local")
    a.set_title(f"Zoom débruiteur ELS (P={args.P}) — c'est la cible que ScCP reproduit")
    a.legend(fontsize=8); a.grid(alpha=.3)

    plt.tight_layout(); plt.savefig("target_lipschitz.png", dpi=115)
    log("[fig] target_lipschitz.png")
    with open("target_lipschitz_metrics.txt", "w") as f:
        f.write(f"# MESURE 1 v2 Lipschitz débruiteur (indep) nsub={args.nsub} nq={args.nq} P_ELS={args.P}\n")
        f.write("t\tLg_med\tLg_p90\tLg_max\tfe_g\tLe_med\tLe_p90\tLe_max\tfe_e\n")
        for r in rows:
            f.write(f"{r['t']:.3f}\t{r['Lg_med']:.3f}\t{r['Lg_p90']:.3f}\t{r['Lg_max']:.3f}\t{r['fe_g']:.3f}\t"
                    f"{r['Le_med']:.4f}\t{r['Le_p90']:.4f}\t{r['Le_max']:.4f}\t{r['fe_e']:.3f}\n")
    log(f"[metrics] target_lipschitz_metrics.txt | fini en {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
