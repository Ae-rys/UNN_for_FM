"""comparison_mnist_sccp_ot.png : analogue OT/Flow-Matching de comparison_mnist_sccp.png
(repro_kamb, cas score). Grille [ELS-FM | ScCP-FM (OT) | IS-FM] par graine, avec
r²(modèle, ELS) et r²(modèle, IS) du DÉBRUITEUR le long de la trajectoire (métrique FM
correcte, = celle qui donne 0.90 ; l'endpoint serait pollué par 1/(1-t)).

Modèle = checkpoint OT ScCP k3 (w=8) reconstruit comme à l'entraînement.
"""
import torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import models.architectures as A
A.L1ProxConv.__init__.__defaults__ = (8, None)             # checkpoint OT = w8 (défaut changé depuis)
import nifty_els_fm as N
from run_mnist import build_experiments

CKPT = "results/ot_els_test/ConvScCP_k3_K6_ic128_L1_LNO/model.pt"
NAME = "ConvScCP_k3_K6_ic128_L1_LNO"
NSEED, NSTEPS, P = 8, 50, 9                                 # P=9 = meilleur P fixe (r²=0.901)

entry = [e for e in build_experiments(N.dev) if e["name"] == NAME][0]
model = entry["build"]()
model.load_state_dict(torch.load(CKPT, map_location=N.dev, weights_only=True)); model.eval()
print(f"[load] {NAME} w=8, checkpoint OT")

Xsub = N.mnist_train(2000, seed=0); X1 = Xsub.view(2000, N.DIM)
pat, pn = N.build_dict(Xsub, P); gw = N.gauss_patch_weight(P)

torch.manual_seed(0)
x0 = torch.randn(NSEED, N.DIM, device=N.dev)
gen, traj = N.euler_model(model, x0.clone(), NSTEPS); gen = gen.clamp(-1, 1)

# r² débruiteur le long de la trajectoire, par graine
tskip = [(x, t) for (x, t) in traj if 0.02 <= t <= 0.95]
acc_els = torch.zeros(NSEED, device=N.dev); acc_is = torch.zeros(NSEED, device=N.dev)
for (x, t) in tskip:
    omt = max(1.0 - t, 0.05)
    with torch.no_grad():
        ex1_m = x + omt * N.model_velocity(model, x, t)
    acc_els += N.cos_field(ex1_m, N.ex1_els_nifty(x, t, pat, pn, P, gw))
    acc_is += N.cos_field(ex1_m, N.ex1_is(x, t, X1))
n = len(tskip)
rels = (acc_els / n).cpu().numpy(); ris = (acc_is / n).cpu().numpy()

# images générées, graines appariées
els_img = N.euler_denoiser(lambda x, t: N.ex1_els_nifty(x, t, pat, pn, P, gw), x0.clone(), NSTEPS).clamp(-1, 1)
is_img = N.euler_denoiser(lambda x, t: N.ex1_is(x, t, X1), x0.clone(), NSTEPS).clamp(-1, 1)
print(f"médian r²(ELS)={np.median(rels):.3f}  r²(IS)={np.median(ris):.3f}  "
      f"ELS>IS {int((rels>ris).sum())}/{NSEED}")

dn = lambda v: (v.view(N.S, N.S).cpu().numpy() + 1) / 2
fig, ax = plt.subplots(NSEED, 3, figsize=(6, 2.0 * NSEED))
for i in range(NSEED):
    cols = [(els_img[i], f"ELS-FM\nr²={rels[i]:.2f}"),
            (gen[i], "ScCP-FM (OT)"),
            (is_img[i], f"IS-FM\nr²={ris[i]:.2f}")]
    for j, (img, ttl) in enumerate(cols):
        ax[i, j].imshow(dn(img), cmap="gray", vmin=0, vmax=1); ax[i, j].axis("off")
        if j == 1:
            if i == 0: ax[i, j].set_title(ttl, fontsize=10)
        else:
            ax[i, j].set_title(ttl if i == 0 else ttl.split("\n")[-1], fontsize=8)
plt.suptitle(f"ScCP Flow-Matching, couplage OT — ELS≈modèle vs IS (mémo)\n"
             f"médian r²(ELS)={np.median(rels):.2f}  r²(IS)={np.median(ris):.2f}  (P={P})",
             fontsize=10)
plt.tight_layout(); plt.savefig("comparison_mnist_sccp_ot.png", dpi=130, bbox_inches="tight")
print("saved -> comparison_mnist_sccp_ot.png")
