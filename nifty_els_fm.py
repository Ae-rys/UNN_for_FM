"""ELS-FM machine FIDÈLE À NIFTY (Chatillon-Rabin-Tschumperlé 2025, éq. 6 + Alg.1)
pour trancher : un conv-net FM entraîné (ScCP-FM / ResNet-FM) est-il une machine ELS ?

Correction vs l'ancien els_fm_test.py (confondu) :
  - NIFTY calcule le flot sur le PATCH ENTIER puis réagrège par FOLD GAUSSIEN
    (method.py Patch_Average), là où l'ancienne machine ne prenait que le PIXEL
    CENTRAL de chaque patch (convention score/Kamb). C'est LA différence suspectée.
  - noyau FM g_{tφ,(1-t)²} : w_j ∝ exp(-‖ψ-tφ_j‖²/2(1-t)²)  (déjà correct avant).

On compare au niveau du DÉBRUITEUR  E[x1|x_t]  (= ce que compare eval_script de Kamb),
le long de la trajectoire d'échantillonnage DU MODÈLE. Le r² est un cosinus centré
=> invariant au facteur 1/(1-t) de la vitesse, donc robuste.

  Ex1_model = x_t + (1-t) * v_model(x_t,t)
  Ex1_ELS   = fold_gaussien_j [ Σ_j φ_j w_j ]     (débruiteur local-patch, NIFTY)
  Ex1_IS    = Σ_i x1_i softmax(-‖x_t - t x1_i‖²/2(1-t)²)   (global = mémorisation)

Un conv-net FM EST une machine ELS ssi  r²(model,ELS) >> r²(model,IS).

Sorties (dans le repo) :
  nifty_els_metrics_<tag>.txt  : table r² (IS, ELS pour P fixes + P(t) calibré, + ancien pixel-central)
  nifty_els_grid_<tag>.png     : samples ConvScCP vs ELS-FM(fold) vs IS-FM

Usage :
  python nifty_els_fm.py --ckpt results/temp-5/ConvScCP_k3_K6_ic128_L1_LNO/model.pt \
      --K 6 --ic 128 --kernel 3 --tag sccp_k3
  python nifty_els_fm.py --ckpt results/temp-5/MinimalResNetFM_L6_ic256/model.pt --tag resnetfm
"""
import argparse, time, re, torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision, torchvision.transforms as T
from models.architectures import ConvScCP_UNN, MinimalResNetFM

dev = "cuda" if torch.cuda.is_available() else "cpu"
DIM, S = 784, 28
PAD_VAL = 0.0        # padding neutre (cohérent dict/query)


# ---------- utilitaires ----------
def cos_field(a, b):
    """cosinus centré par échantillon. a,b : (N, DIM) -> (N,)"""
    a = a - a.mean(1, keepdim=True); a = a / (a.norm(dim=1, keepdim=True) + 1e-12)
    b = b - b.mean(1, keepdim=True); b = b / (b.norm(dim=1, keepdim=True) + 1e-12)
    return (a * b).sum(1)


def mnist_train(nsub, seed=0):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])   # -> [-1,1]
    ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=tf)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:nsub]
    X = torch.stack([ds[i][0] for i in idx]).view(nsub, 1, S, S)
    return X.to(dev)


def load_model(ckpt, K, ic, kernel):
    sd = torch.load(ckpt, map_location=dev, weights_only=True)
    if "layers.0.W_weight" in sd:                            # ConvScCP_UNN
        m = ConvScCP_UNN(dim=DIM, K=K, internal_channel=ic, use_Unet="l1", version="LNO",
                         use_checkpoint=False, w_bias=True, in_channels=1, img_size=S,
                         kernel_size=kernel).to(dev)
        print(f"[load] ConvScCP_UNN K={K} ic={ic} kernel={kernel}")
    elif "up.weight" in sd:                                  # MinimalResNetFM
        nl = 1 + max(int(re.match(r"convs\.(\d+)\.", k).group(1)) for k in sd
                     if re.match(r"convs\.(\d+)\.", k))
        emb = sd["up.weight"].shape[0]; ksz = sd["up.weight"].shape[2]
        m = MinimalResNetFM(dim=DIM, num_layers=nl, emb_dim=emb, kernel_size=ksz,
                            in_channels=1, img_size=S).to(dev)
        print(f"[load] MinimalResNetFM num_layers={nl} emb_dim={emb} kernel={ksz}")
    else:
        raise ValueError(f"type modèle inconnu: {list(sd)[:4]}")
    m.load_state_dict(sd); m.eval()
    return m


def model_velocity(model, x, t):
    inp = torch.cat([x, torch.full((x.shape[0], 1), t, device=dev)], dim=1)
    return model(inp)


# ---------- dictionnaires de patches ----------
def build_dict(Xsub, P):
    """patches φ (Npatch, PP) + normes ; dense stride-1, zero-pad -> chaque pixel = centre."""
    pat = F.unfold(F.pad(Xsub, (P // 2,) * 4, value=PAD_VAL), P)     # (Nsub, PP, HW)
    pat = pat.permute(0, 2, 1).reshape(-1, P * P).contiguous()       # (Npatch, PP)
    return pat, (pat ** 2).sum(1)


def gauss_patch_weight(P, spot=0.5):
    """poids gaussien P×P pour l'agrégation (NIFTY Patch_Average)."""
    r = torch.linspace(-(P // 2), P // 2, P, device=dev)
    w = torch.exp(-(r ** 2) / (2 * (P * spot) ** 2))
    w = (w[:, None] * w[None, :]).reshape(-1)                        # (PP,)
    return w


# ---------- débruiteurs E[x1|x_t] ----------
@torch.no_grad()
def ex1_els_nifty(x, t, pat, pnorm, P, gw, chunk=20000):
    """Débruiteur LOCAL-PATCH, patch entier + fold gaussien (NIFTY). x:(b,DIM)->(b,DIM)."""
    b = x.shape[0]; omt = max(1.0 - t, 1e-2)
    xi = x.view(b, 1, S, S)
    Q = F.unfold(F.pad(xi, (P // 2,) * 4, value=PAD_VAL), P)         # (b, PP, HW)
    HW = Q.shape[2]; PP = P * P
    Qf = Q.permute(0, 2, 1).reshape(b * HW, PP)                      # (M, PP)
    M = Qf.shape[0]
    coef = t / (omt ** 2); quad = (t ** 2) / (2 * omt ** 2)
    run_max = torch.full((M,), -float("inf"), device=dev)
    num = torch.zeros(M, PP, device=dev); den = torch.zeros(M, device=dev)
    for s in range(0, pat.shape[0], chunk):
        Pj = pat[s:s + chunk]; nj = pnorm[s:s + chunk]
        logit = coef * (Qf @ Pj.t()) - quad * nj[None, :]           # (M, chunk)
        cmax = logit.max(dim=1).values
        new_max = torch.maximum(run_max, cmax)
        scale = torch.exp(run_max - new_max)
        e = torch.exp(logit - new_max[:, None])                     # (M, chunk)
        num = num * scale[:, None] + e @ Pj                         # (M, PP)  Σ φ_j w_j
        den = den * scale + e.sum(1)
        run_max = new_max
    Ex1_patch = (num / den[:, None]).reshape(b, HW, PP).permute(0, 2, 1)   # (b, PP, HW)
    # fold gaussien -> champ (b,1,28,28)
    fnum = F.fold(Ex1_patch * gw[None, :, None], (S, S), P, padding=P // 2)
    fden = F.fold(gw[None, :, None].expand(b, PP, HW), (S, S), P, padding=P // 2)
    return (fnum / fden).view(b, DIM)


@torch.no_grad()
def ex1_els_center(x, t, pat, pnorm, cen, P, chunk=20000):
    """ANCIENNE machine : pixel central seulement (pour comparaison)."""
    b = x.shape[0]; omt = max(1.0 - t, 1e-2)
    xi = x.view(b, 1, S, S)
    Q = F.unfold(F.pad(xi, (P // 2,) * 4, value=PAD_VAL), P).permute(0, 2, 1).reshape(-1, P * P)
    M = Q.shape[0]
    coef = t / (omt ** 2); quad = (t ** 2) / (2 * omt ** 2)
    run_max = torch.full((M,), -float("inf"), device=dev)
    num = torch.zeros(M, device=dev); den = torch.zeros(M, device=dev)
    for s in range(0, pat.shape[0], chunk):
        Pj = pat[s:s + chunk]; nj = pnorm[s:s + chunk]; cj = cen[s:s + chunk]
        logit = coef * (Q @ Pj.t()) - quad * nj[None, :]
        cmax = logit.max(dim=1).values
        new_max = torch.maximum(run_max, cmax)
        scale = torch.exp(run_max - new_max)
        e = torch.exp(logit - new_max[:, None])
        num = num * scale + (e * cj[None, :]).sum(1)
        den = den * scale + e.sum(1)
        run_max = new_max
    return (num / den).view(b, DIM)


@torch.no_grad()
def ex1_is(x, t, X1, chunk=20000):
    """Débruiteur GLOBAL (mémorisation). X1:(Ntr,DIM)."""
    omt = max(1.0 - t, 1e-2)
    coef = t / (omt ** 2); quad = (t ** 2) / (2 * omt ** 2)
    b = x.shape[0]
    run_max = torch.full((b,), -float("inf"), device=dev)
    num = torch.zeros(b, DIM, device=dev); den = torch.zeros(b, device=dev)
    for s in range(0, X1.shape[0], chunk):
        Xj = X1[s:s + chunk]; nj = (Xj ** 2).sum(1)
        logit = coef * (x @ Xj.t()) - quad * nj[None, :]
        cmax = logit.max(dim=1).values
        new_max = torch.maximum(run_max, cmax)
        scale = torch.exp(run_max - new_max)
        e = torch.exp(logit - new_max[:, None])
        num = num * scale[:, None] + e @ Xj
        den = den * scale + e.sum(1)
        run_max = new_max
    return num / den[:, None]


# ---------- intégration ----------
@torch.no_grad()
def euler_model(model, x0, nsteps):
    x = x0.clone(); dt = 1.0 / nsteps; traj = []
    for k in range(nsteps):
        t = k * dt
        v = model_velocity(model, x, t)
        traj.append((x.clone(), t))
        x = x + dt * v
    return x, traj


@torch.no_grad()
def euler_denoiser(ex1_fn, x0, nsteps):
    """intègre le champ FM v=(Ex1-x)/(1-t) d'un débruiteur donné."""
    x = x0.clone(); dt = 1.0 / nsteps
    for k in range(nsteps):
        t = k * dt; omt = max(1.0 - t, 0.05)
        x = x + dt * (ex1_fn(x, t) - x) / omt
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--ic", type=int, default=128)
    p.add_argument("--kernel", type=int, default=3)
    p.add_argument("--nsub", type=int, default=2000)
    p.add_argument("--nseed", type=int, default=8)
    p.add_argument("--nsteps", type=int, default=50)
    p.add_argument("--Ps", type=int, nargs="*", default=[3, 5, 7, 9, 11])
    p.add_argument("--tag", default="")
    args = p.parse_args()
    tag = ("_" + args.tag) if args.tag else ""
    torch.manual_seed(0); t0 = time.time()

    model = load_model(args.ckpt, args.K, args.ic, args.kernel)
    Xsub = mnist_train(args.nsub)
    X1 = Xsub.view(args.nsub, DIM)
    dicts = {}
    for P in args.Ps:
        pat, pn = build_dict(Xsub, P)
        cen = Xsub.view(args.nsub, DIM).reshape(-1).contiguous()      # pixel central (dense)
        dicts[P] = (pat, pn, cen, gauss_patch_weight(P))
        print(f"[dict] P={P}: {pat.shape[0]:,} patches", flush=True)

    # trajectoire DU MODÈLE (Euler nsteps) depuis un bruit commun
    x0 = torch.randn(args.nseed, DIM, device=dev)
    gen, traj = euler_model(model, x0.clone(), args.nsteps)
    gen = gen.clamp(-1, 1)
    print(f"[traj] {len(traj)} pas, modèle intégré  ({time.time()-t0:.0f}s)", flush=True)

    # --- r² DÉBRUITEUR le long de la trajectoire ---
    # accumulateurs (sum sur pas de temps, gardés par seed)
    tskip = [(x, t) for (x, t) in traj if 0.02 <= t <= 0.95]
    acc_is = torch.zeros(args.nseed, device=dev)
    acc_els = {P: torch.zeros(args.nseed, device=dev) for P in args.Ps}
    acc_cen = {P: torch.zeros(args.nseed, device=dev) for P in args.Ps}
    acc_calib = torch.zeros(args.nseed, device=dev)

    # P(t) calibré : à chaque t, P max cosinus(model, ELS-fold)
    calibP = {}
    for (x, t) in tskip:
        omt = max(1.0 - t, 0.05)
        with torch.no_grad():
            ex1_m = x + omt * model_velocity(model, x, t)              # débruiteur du modèle
        acc_is += cos_field(ex1_m, ex1_is(x, t, X1))
        best, bestc = args.Ps[0], -2.0
        for P in args.Ps:
            pat, pn, cen, gw = dicts[P]
            e_fold = ex1_els_nifty(x, t, pat, pn, P, gw)
            c = cos_field(ex1_m, e_fold)
            acc_els[P] += c
            acc_cen[P] += cos_field(ex1_m, ex1_els_center(x, t, pat, pn, cen, P))
            cm = float(c.median())
            if cm > bestc: bestc, best = cm, P
        calibP[round(t, 3)] = best
        pat, pn, cen, gw = dicts[best]
        acc_calib += cos_field(ex1_m, ex1_els_nifty(x, t, pat, pn, best, gw))
    n = len(tskip)
    print(f"[r2] {n} pas évalués  ({time.time()-t0:.0f}s)", flush=True)

    med = lambda a: float((a / n).median())
    r_is = med(acc_is)
    r_els = {P: med(acc_els[P]) for P in args.Ps}
    r_cen = {P: med(acc_cen[P]) for P in args.Ps}
    r_calib = med(acc_calib)
    bestP = max(r_els, key=r_els.get)

    lines = [
        f"ckpt={args.ckpt}  nsub={args.nsub} nseed={args.nseed} nsteps={args.nsteps}",
        f"r² DÉBRUITEUR (cos centré) médian le long de la trajectoire, {n} pas :",
        f"  IS-FM (global/mémo)            = {r_is:.3f}",
        "  ELS-FM NIFTY (patch+fold gaussien) :",
    ] + [f"      P={P:2d}  = {r_els[P]:.3f}" for P in args.Ps] + [
        f"  ELS-FM P(t) calibré            = {r_calib:.3f}",
        f"  >>> meilleur P fixe : P={bestP}  r²={r_els[bestP]:.3f}",
        "  (rappel ancienne machine PIXEL-CENTRAL seul :)",
    ] + [f"      P={P:2d}  = {r_cen[P]:.3f}" for P in args.Ps] + [
        "",
        f"VERDICT ELS vs IS : ELS(best={bestP})={r_els[bestP]:.3f}  vs  IS={r_is:.3f}  "
        f"-> {'ELS ≫ IS : machine ELS ✓' if r_els[bestP] > r_is + 0.05 else 'pas d écart net'}",
        f"P(t) calibré : {calibP}",
    ]
    txt = "\n".join(lines); print(txt)
    open(f"nifty_els_metrics{tag}.txt", "w").write(txt + "\n")

    # --- figure : samples (intègre le débruiteur ELS calibré et IS en FM) ---
    def ex1_calib_fn(x, t):
        P = calibP.get(round(t, 3), bestP)
        pat, pn, cen, gw = dicts[P]
        return ex1_els_nifty(x, t, pat, pn, P, gw)
    els_img = euler_denoiser(ex1_calib_fn, x0.clone(), args.nsteps).clamp(-1, 1)
    is_img = euler_denoiser(lambda x, t: ex1_is(x, t, X1), x0.clone(), args.nsteps).clamp(-1, 1)

    dn = lambda v: (v.view(S, S).cpu().numpy() + 1) / 2
    ns = min(args.nseed, 8)
    fig, ax = plt.subplots(ns, 3, figsize=(6, 2.0 * ns))
    for i in range(ns):
        for j, (img, ttl) in enumerate([(gen[i], "modèle FM"),
                                        (els_img[i], "ELS-FM\n(NIFTY fold)"),
                                        (is_img[i], "IS-FM\n(mémo)")]):
            ax[i, j].imshow(dn(img), cmap="gray", vmin=0, vmax=1); ax[i, j].axis("off")
            if i == 0: ax[i, j].set_title(ttl, fontsize=9)
    plt.suptitle(f"ELS-FM fidèle NIFTY vs modèle {args.tag}\nr²(ELS)={r_els[bestP]:.2f} r²(IS)={r_is:.2f}",
                 fontsize=10)
    plt.tight_layout(); plt.savefig(f"nifty_els_grid{tag}.png", dpi=130, bbox_inches="tight")
    print(f"saved -> nifty_els_grid{tag}.png , nifty_els_metrics{tag}.txt  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
