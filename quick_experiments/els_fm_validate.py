"""Ground-truth validation of the ELS-FM machine.

We build a target that IS, by construction, the output of an ELS machine at a
known scale P_gt (integrate the ELS-FM ODE from seeds). Then we check whether
the SAME pipeline recovers it:
  - ELS-FM at P=P_gt should predict it (self) with r2 ~ 1.0  (sanity)
  - ELS-FM at nearby scales should still predict it well      (scale sensitivity)
  - IS-FM (memorization) should predict it MUCH worse         (discrimination)

Decisive number = r2(ELS-target, IS-FM):
  << 1  -> ELS and IS are distinguishable -> my pipeline works
           -> the ConvScCP result (ELS ~ IS) is a REAL negative.
  ~ 1   -> ELS and IS give near-identical images on MNIST
           -> the test has no discriminative power -> ConvScCP result inconclusive.

Run:  python els_fm_validate.py
"""
import time, torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch.nn.functional as F
from els_fm_test import dev, DIM, S, r2, mnist_train, is_fm_velocity, els_fm_velocity, euler

NSUB, NSEED, NSTEPS, P_GT = 4000, 8, 50, 7
PS = [3, 5, 7, 9, 11]


def build_dict(Xsub, P):
    pat = F.unfold(F.pad(Xsub, (P // 2,) * 4), P).permute(0, 2, 1).reshape(-1, P * P).contiguous()
    cen = Xsub.view(Xsub.shape[0], DIM).reshape(-1).contiguous()
    return pat, cen, (pat ** 2).sum(1)


def med_r2(a, b):
    return float(np.median([r2(a[i], b[i]) for i in range(a.shape[0])]))


def main():
    torch.manual_seed(0); t0 = time.time()
    Xsub = mnist_train(NSUB); X1_flat = Xsub.view(NSUB, DIM)
    dicts = {P: build_dict(Xsub, P) for P in PS}
    print(f"dicts built ({time.time()-t0:.0f}s)", flush=True)

    x0 = torch.randn(NSEED, DIM, device=dev)
    # ground-truth target: output of an ELS machine at scale P_GT
    target = euler(lambda x, t: els_fm_velocity(x, t, *dicts[P_GT], P_GT), x0.clone(), NSTEPS).clamp(-1, 1)
    print(f"target (ELS P={P_GT}) generated ({time.time()-t0:.0f}s)", flush=True)

    res = {}
    for P in PS:
        s = euler(lambda x, t: els_fm_velocity(x, t, *dicts[P], P), x0.clone(), NSTEPS).clamp(-1, 1)
        res[f"ELS P={P}"] = med_r2(target, s)
        print(f"  r2(target, ELS P={P}) = {res[f'ELS P={P}']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    sIS = euler(lambda x, t: is_fm_velocity(x, t, X1_flat), x0.clone(), NSTEPS).clamp(-1, 1)
    res["IS"] = med_r2(target, sIS)
    print(f"  r2(target, IS)      = {res['IS']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    lines = [f"GROUND-TRUTH = ELS-FM machine at P={P_GT}  (nsub={NSUB}, nseed={NSEED})",
             *[f"  r2(target, {k:9s}) = {v:.3f}" for k, v in res.items()],
             "",
             f"DECISIVE r2(target, IS) = {res['IS']:.3f}",
             "  << 1  => ELS/IS distinguishables => pipeline OK => ConvScCP null RÉEL",
             "  ~ 1   => ELS~IS sur MNIST => test non-discriminant => ConvScCP inconcluant"]
    txt = "\n".join(lines); print(txt)
    open("els_fm_validate.txt", "w").write(txt + "\n")

    # bar chart
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ks = list(res.keys()); vs = [res[k] for k in ks]
    cols = ["#4C78A8" if k.startswith("ELS") else "#E45756" for k in ks]
    ax.bar(ks, vs, color=cols)
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_ylabel("r² médian vs cible (= ELS P=7)"); ax.set_ylim(0, 1.05)
    ax.set_title("Validation ELS-FM : prédire une cible qui EST une machine ELS")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig("els_fm_validate.png", dpi=140, bbox_inches="tight")
    print("saved -> els_fm_validate.png , els_fm_validate.txt")


if __name__ == "__main__":
    main()
