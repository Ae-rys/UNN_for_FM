"""Discriminant ELS vs IS PAR BIN DE t, pour les 6 modèles (ScCP/ResNet/UNet × indep/OT).
Au lieu de la médiane sur toute la trajectoire (émoussée), on résout r²(modèle,ELS) et
r²(modèle,IS) EN FONCTION DE t. La discrimination réelle est à t intermédiaire (aux deux
bouts ELS≈IS≈trivial). Attendu : gap ELS−IS pique au milieu, et l'ordre ScCP>ResNet>UNet
y ressort. ELS = NIFTY (patch+fold), calibré best-P par bin.

Sortie : discriminant_vs_t.png , discriminant_vs_t_metrics.txt
"""
import torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import nifty_els_fm as N
from run_mnist import build_experiments

NSEED, NSTEPS, NSUB = 8, 50, 1500
PS = [3, 5, 7, 9, 11]
CFG = [   # (arch, coupling, build_name, ckpt)
    ("ScCP",   "indep", "ConvScCP_k3_K6_ic128_L1_LNO", "results/grid50/ConvScCP_k3_K6_ic128_L1_LNO/model.pt"),
    ("ScCP",   "OT",    "ConvScCP_k3_K6_ic128_L1_LNO", "results/grid50_ot/ConvScCP_k3_K6_ic128_L1_LNO/model.pt"),
    ("ResNet", "indep", "MinimalResNetFM_L6_ic256",    "results/temp-5/MinimalResNetFM_L6_ic256/model.pt"),
    ("ResNet", "OT",    "MinimalResNetFM_L6_ic256",    "results/grid50_ot/MinimalResNetFM_L6_ic256/model.pt"),
    ("UNet",   "indep", "MinimalUNetFM_kamb",          "results/grid50/MinimalUNetFM_kamb/model.pt"),
    ("UNet",   "OT",    "MinimalUNetFM_kamb",          "results/grid50_ot/MinimalUNetFM_kamb/model.pt"),
]


def run_one(name, ckpt, Xsub, X1, dicts):
    entry = [e for e in build_experiments(N.dev) if e["name"] == name][0]
    m = entry["build"](); m.load_state_dict(torch.load(ckpt, map_location=N.dev, weights_only=True)); m.eval()
    torch.manual_seed(0)
    x0 = torch.randn(NSEED, N.DIM, device=N.dev)
    _, traj = N.euler_model(m, x0.clone(), NSTEPS)
    ts, els, iss = [], [], []
    for (x, t) in traj:
        if not (0.02 <= t <= 0.95):
            continue
        omt = max(1.0 - t, 0.05)
        with torch.no_grad():
            ex1_m = x + omt * N.model_velocity(m, x, t)
        c_is = N.cos_field(ex1_m, N.ex1_is(x, t, X1))
        c_els = torch.stack([                             # calibré : best P par point (max sur P)
            N.cos_field(ex1_m, N.ex1_els_nifty(x, t, dicts[P][0], dicts[P][1], P, dicts[P][2]))
            for P in PS]).max(0).values
        ts.append(float(t)); els.append(float(c_els.median())); iss.append(float(c_is.median()))
    return np.array(ts), np.array(els), np.array(iss)


def main():
    Xsub = N.mnist_train(NSUB, seed=0); X1 = Xsub.view(NSUB, N.DIM)
    dicts = {P: (*N.build_dict(Xsub, P), N.gauss_patch_weight(P)) for P in PS}  # (pat, pn, gw)
    res = {}
    for arch, coup, name, ckpt in CFG:
        ts, els, iss = run_one(name, ckpt, Xsub, X1, dicts)
        res[(arch, coup)] = (ts, els, iss)
        print(f"[{arch} {coup}] ELS_max={els.max():.3f}@t={ts[els.argmax()]:.2f} "
              f"gap_max={ (els-iss).max():.3f}@t={ts[(els-iss).argmax()]:.2f}", flush=True)

    archs = ["ScCP", "ResNet", "UNet"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for j, arch in enumerate(archs):
        a = ax[0, j]
        for coup, ls in [("indep", "-"), ("OT", "--")]:
            ts, els, iss = res[(arch, coup)]
            a.plot(ts, els, ls, color="C0", label=f"ELS {coup}")
            a.plot(ts, iss, ls, color="C3", label=f"IS {coup}")
        a.set_title(f"{arch} — r²(ELS) vs r²(IS)"); a.set_ylim(-0.1, 1.0); a.grid(alpha=.3)
        if j == 0: a.set_ylabel("r² débruiteur (cos)")
        a.legend(fontsize=7)
        b = ax[1, j]
        for coup, ls in [("indep", "-"), ("OT", "--")]:
            ts, els, iss = res[(arch, coup)]
            b.plot(ts, els - iss, ls, color="C2", label=f"gap {coup}")
        b.axhline(0, color="k", lw=.8); b.set_ylim(-0.3, 0.9); b.grid(alpha=.3)
        b.set_xlabel("t"); b.set_title(f"{arch} — gap ELS−IS")
        if j == 0: b.set_ylabel("r²(ELS) − r²(IS)")
        b.legend(fontsize=7)
    plt.suptitle("Discriminant ELS vs IS par bin de t (ELS = NIFTY patch+fold, best-P/bin)\n"
                 "plus local (ScCP) → gap grand ; plus global (UNet) → gap ~0", fontsize=11)
    plt.tight_layout(); plt.savefig("discriminant_vs_t.png", dpi=120)
    print("saved -> discriminant_vs_t.png", flush=True)

    with open("discriminant_vs_t_metrics.txt", "w") as f:
        for arch in archs:
            for coup in ["indep", "OT"]:
                ts, els, iss = res[(arch, coup)]
                g = els - iss
                f.write(f"# {arch} {coup} : gap_max={g.max():.3f}@t={ts[g.argmax()]:.2f} "
                        f"gap_mean={g.mean():.3f}\n")
                f.write("t\tELS\tIS\tgap\n")
                for i in range(len(ts)):
                    f.write(f"{ts[i]:.3f}\t{els[i]:.3f}\t{iss[i]:.3f}\t{g[i]:.3f}\n")
                f.write("\n")
    print("saved -> discriminant_vs_t_metrics.txt", flush=True)


if __name__ == "__main__":
    main()
