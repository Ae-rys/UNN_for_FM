"""MESURE 2 (décisive) : constante de Lipschitz locale du DÉBRUITEUR des modèles
ENTRAÎNÉS (couplage indep, checkpoints temp-5), à superposer sur la cible ELS/global
de la mesure 1. Débruiteur = x + (1-t)·v_model(x,t) (même convention que le r² NIFTY).

Question qui tranche le 'prix de la convexité' :
  - ScCP (prox convexe + w-bias) est-il PLAFONNÉ (L≲1, sous la cible ELS) ?  -> le w-bias
    ne décapsule pas, la contrainte convexe MORD, le prix est réel et localisé.
  - ou atteint-il L~2-3 comme ELS ?  -> w-bias/K fini le sortent de la classe, PAS de prix.
Et : ScCP a-t-il un L systématiquement < ResNet/UNet libres (signature de la contrainte) ?

Mêmes points xt que la mesure 1 (données held-out, indep) -> courbes directement comparables.
Sortie : model_lipschitz.png , model_lipschitz_metrics.txt (logge dans claude.log)
Usage : ~/.venvs/unn/bin/python model_lipschitz.py --nq 16 --nt 11
"""
import argparse, time, torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from nifty_els_fm import (mnist_train, build_dict, gauss_patch_weight,
                          ex1_els_nifty, load_model, model_velocity, DIM, S, dev)
from target_lipschitz import lipschitz_els
from models.architectures import SmallUNet

LOG = open("claude.log", "a")
def log(*a): print(*a, flush=True); print(*a, file=LOG, flush=True)


@torch.no_grad()
def lipschitz_denoiser(vel_fn, x, t, niter=15, eps=1e-3):
    """L = ‖J‖ du débruiteur f(x)=x+(1-t)·v(x,t), power-iteration à diff. finies."""
    omt = max(1.0 - t, 1e-2)
    f = lambda z: z + omt * vel_fn(z, t)
    b = x.shape[0]
    v = torch.randn(b, DIM, device=dev); v /= v.norm(dim=1, keepdim=True)
    lam = torch.zeros(b, device=dev)
    for _ in range(niter):
        Jv = (f(x + eps * v) - f(x - eps * v)) / (2 * eps)
        lam = (v * Jv).sum(1)
        v = Jv / (Jv.norm(dim=1, keepdim=True) + 1e-30)
    return lam


def q(x, p): return torch.quantile(x, p).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq", type=int, default=16)
    ap.add_argument("--nt", type=int, default=11)
    ap.add_argument("--nsub", type=int, default=600)
    ap.add_argument("--P", type=int, default=7)
    args = ap.parse_args()
    log(f"\n===== MESURE 2 : Lipschitz débruiteur des modèles entraînés {time.strftime('%F %T')} =====")
    t0 = time.time()

    sccp = load_model("results/temp-5/ConvScCP_k3_K6_ic128_L1_LNO/model.pt", K=6, ic=128, kernel=3)
    resnet = load_model("results/temp-5/MinimalResNetFM_L6_ic256/model.pt", K=6, ic=128, kernel=3)
    unet = SmallUNet(in_channels=1).to(dev)
    unet.load_state_dict(torch.load("results/temp-5/SmallUNet_baseline/model.pt",
                                    map_location=dev, weights_only=True)); unet.eval()
    log("[load] SmallUNet_baseline")
    vel = {"ScCP (convexe)": lambda x, t: model_velocity(sccp, x, t),
           "ResNet (libre)": lambda x, t: model_velocity(resnet, x, t),
           "UNet (libre)":   lambda x, t: unet(torch.cat([x, torch.full((x.shape[0],1), t, device=dev)],1))}

    Xsub = mnist_train(args.nsub, seed=0); X1 = Xsub.view(args.nsub, DIM)
    Xq = mnist_train(args.nq, seed=1).view(args.nq, DIM)
    pat, pn = build_dict(Xsub, args.P); gw = gauss_patch_weight(args.P)

    ts = np.linspace(0.05, 0.95, args.nt)
    curves = {k: {"med": [], "p90": []} for k in list(vel) + ["ELS cible"]}
    for t in ts:
        x0 = torch.randn_like(Xq); xt = t * Xq + (1 - t) * x0
        line = f"t={t:.3f} |"
        for k, fn in vel.items():
            L = lipschitz_denoiser(fn, xt, float(t))
            curves[k]["med"].append(q(L, .5)); curves[k]["p90"].append(q(L, .9))
            line += f" {k}: med={q(L,.5):.2f} p90={q(L,.9):.2f} |"
        Le = lipschitz_els(xt, float(t), pat, pn, args.P, gw)
        curves["ELS cible"]["med"].append(q(Le, .5)); curves["ELS cible"]["p90"].append(q(Le, .9))
        line += f"  ELS: med={q(Le,.5):.2f}"
        log(line)

    T_ = list(ts)
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6))
    styl = {"ScCP (convexe)": ("C0", "-o"), "ResNet (libre)": ("C1", "-s"),
            "UNet (libre)": ("C2", "-^"), "ELS cible": ("k", "--D")}
    for k in curves:
        c, m = styl[k]
        ax.plot(T_, curves[k]["med"], m, color=c, label=k, lw=1.8, ms=5)
        ax.fill_between(T_, curves[k]["med"], curves[k]["p90"], color=c, alpha=.12)
    ax.axhline(1.0, color="r", ls=":", lw=1.5, label="seuil non-expansif = 1")
    ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_ylabel(r"Lipschitz local $\|\partial E[x_1|x_t]/\partial x_t\|$")
    ax.set_title("Lipschitz du débruiteur : modèles entraînés vs cible ELS\n"
                 "(ombré = médiane→p90)")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("model_lipschitz.png", dpi=120)
    log("[fig] model_lipschitz.png")

    with open("model_lipschitz_metrics.txt", "w") as f:
        f.write(f"# MESURE 2 Lipschitz débruiteur modèles entraînés (indep) nq={args.nq} P_ELS={args.P}\n")
        f.write("t\t" + "\t".join(f"{k}_med\t{k}_p90" for k in curves) + "\n")
        for i, t in enumerate(ts):
            f.write(f"{t:.3f}\t" + "\t".join(f"{curves[k]['med'][i]:.3f}\t{curves[k]['p90'][i]:.3f}"
                                             for k in curves) + "\n")
    log(f"[metrics] model_lipschitz_metrics.txt | fini en {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
