"""FINAL TEST: is the trained ConvScCP a (local, equivariant) ELS machine?

We build two analytic machines IN THE MODEL'S OWN Flow-Matching framework and
integrate them from the SAME initial noise as the model, then compare (r2):

  IS-FM  : ideal FM velocity, GLOBAL posterior over whole training images
           (memorization baseline).
  ELS-FM : ideal FM velocity, LOCAL + EQUIVARIANT posterior over all training
           PATCHES (locality scale P), à la Kamb Eq.9-10 but FM-parametrized.
           v(x)(p) = (Ex1(p) - x(p)) / (1-t),
           Ex1(p) = sum_{patch j} w_j * center(j),
           w_j ∝ exp( (t<Q_p,P_j> - t^2/2 ||P_j||^2) / (1-t)^2 ).

If r2(model, ELS) >> r2(model, IS), the ConvScCP implements the patch-mosaic
mechanism → it is (approximately) an ELS machine.

Data in [-1,1] (model's training space). Dictionary = Nsub training images.

Usage:
  python els_fm_test.py --ckpt <path> --K 6 --ic 128 --kernel 3 --P 3 --tag k3
"""
import argparse, torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision, torchvision.transforms as T
from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper
import re
from models.architectures import ConvScCP_UNN, MinimalResNetFM

dev  = "cuda" if torch.cuda.is_available() else "cpu"
DIM, S = 784, 28


def r2(a, b):
    a = a - a.mean(); a = a / (a.norm() + 1e-12)
    b = b - b.mean(); b = b / (b.norm() + 1e-12)
    return float((a * b).sum())


def mnist_train(nsub):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])   # -> [-1,1]
    ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=tf)
    idx = torch.randperm(len(ds))[:nsub]
    X = torch.stack([ds[i][0] for i in idx]).view(nsub, 1, S, S)
    return X.to(dev)


def load_model(ckpt, K, ic, kernel):
    sd = torch.load(ckpt, map_location=dev, weights_only=True)
    if "layers.0.W_weight" in sd:                            # ConvScCP_UNN
        m = ConvScCP_UNN(dim=DIM, K=K, internal_channel=ic, use_Unet="l1", version="LNO",
                         use_checkpoint=False, w_bias=True, in_channels=1, img_size=S,
                         kernel_size=kernel).to(dev)
    elif "up.weight" in sd:                                  # MinimalResNetFM (auto-detecté)
        nl  = 1 + max(int(re.match(r"convs\.(\d+)\.", k).group(1)) for k in sd
                      if re.match(r"convs\.(\d+)\.", k))
        emb = sd["up.weight"].shape[0]; ksz = sd["up.weight"].shape[2]
        m = MinimalResNetFM(dim=DIM, num_layers=nl, emb_dim=emb, kernel_size=ksz,
                            in_channels=1, img_size=S).to(dev)
        print(f"[load] MinimalResNetFM num_layers={nl} emb_dim={emb} kernel={ksz}")
    else:
        raise ValueError(f"Type de modele non reconnu: {list(sd)[:4]}")
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def model_sample(model, x0):
    node = NeuralODE(torch_wrapper(model), solver="dopri5", atol=1e-5, rtol=1e-5)
    return node.trajectory(x0, t_span=torch.linspace(0, 1, 2, device=dev))[-1]


@torch.no_grad()
def is_fm_velocity(x, ts, X1_flat):
    """Global memorization velocity. x:(b,DIM), X1_flat:(Ntr,DIM)."""
    omt = max(1.0 - ts, 1e-2)
    logw = -(torch.cdist(x, ts * X1_flat) ** 2) / (2 * omt ** 2)      # (b,Ntr)
    w = torch.softmax(logw, dim=1)
    Ex1 = w @ X1_flat
    return (Ex1 - x) / max(1.0 - ts, 0.05)


@torch.no_grad()
def els_ex1(x, ts, patches, centers, pnorm, P, chunk=40000):
    """Local-equivariant denoiser estimate E[x1|x_t]. Returns (b,DIM)."""
    b = x.shape[0]; omt = max(1.0 - ts, 1e-2)
    xi = x.view(b, 1, S, S)
    Q = F.unfold(F.pad(xi, (P // 2,) * 4), P).permute(0, 2, 1).contiguous()  # (b, HW, PP)
    HW = Q.shape[1]
    run_max = torch.full((b, HW), -float("inf"), device=dev)
    num = torch.zeros(b, HW, device=dev); den = torch.zeros(b, HW, device=dev)
    coef = ts / (omt ** 2); quad = (ts ** 2) / (2 * omt ** 2)
    for s in range(0, patches.shape[0], chunk):
        Pj = patches[s:s+chunk]; cj = centers[s:s+chunk]; nj = pnorm[s:s+chunk]
        logit = coef * (Q @ Pj.t()) - quad * nj[None, None, :]         # (b, HW, C)
        new_max = torch.maximum(run_max, logit.max(dim=-1).values)
        scale = torch.exp(run_max - new_max)
        e = torch.exp(logit - new_max[..., None])
        num = num * scale + (e * cj[None, None, :]).sum(-1)
        den = den * scale + e.sum(-1)
        run_max = new_max
    return (num / den).view(b, DIM)


def els_fm_velocity(x, ts, patches, centers, pnorm, P, chunk=40000):
    return (els_ex1(x, ts, patches, centers, pnorm, P, chunk) - x) / max(1.0 - ts, 0.05)


def model_velocity(model, x, ts):
    inp = torch.cat([x, torch.full((x.shape[0], 1), ts, device=dev)], dim=1)
    return model(inp)                                                 # eval -> velocity


@torch.no_grad()
def euler(vel_fn, x0, nsteps=50):
    x = x0.clone(); dt = 1.0 / nsteps
    for k in range(nsteps):
        x = x + dt * vel_fn(x, k * dt)
    return x


def cos_centered(a, b):
    """Row-wise centered cosine similarity. a,b:(N,DIM) -> (N,)."""
    a = a - a.mean(1, keepdim=True); a = a / (a.norm(dim=1, keepdim=True) + 1e-12)
    b = b - b.mean(1, keepdim=True); b = b / (b.norm(dim=1, keepdim=True) + 1e-12)
    return (a * b).sum(1)


@torch.no_grad()
def calibrate_Pt(model, dicts, Ps, Xval, tgrid):
    """Kamb C.2 calibration: at each t, pick P maximizing cosine(model x1_pred, ELS Ex1).
    Returns {t: bestP}."""
    sched = {}
    for t in tgrid:
        x0 = torch.randn_like(Xval)
        xt = (1 - t) * x0 + t * Xval
        x1_m = xt + max(1 - t, 0.05) * model_velocity(model, xt, t)     # model denoiser
        S = torch.stack([cos_centered(x1_m, els_ex1(xt, t, *dicts[P], P, chunk=8000))
                         for P in Ps], dim=1)
        sched[t] = Ps[int(S.argmax(dim=1).median())]                    # median-optimal scale
        print(f"  [calib] t={t:.2f} -> P={sched[t]}", flush=True)
    return sched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--ic", type=int, default=128)
    p.add_argument("--kernel", type=int, default=3)
    p.add_argument("--P", type=int, default=3, help="locality scale (patch size)")
    p.add_argument("--nsub", type=int, default=4000, help="training images in the dictionary")
    p.add_argument("--nseed", type=int, default=8)
    p.add_argument("--nsteps", type=int, default=50)
    p.add_argument("--schedule", action="store_true",
                   help="coarse-to-fine P(t) calé sur le RF mesuré (au lieu de P fixe)")
    p.add_argument("--calibrate", action="store_true",
                   help="P(t) calibré par cosine-sim modèle<->ELS (méthode Kamb C.2)")
    p.add_argument("--model_euler", action="store_true",
                   help="échantillonne le modèle en Euler nsteps (apparié à l'ELS) au lieu de dopri5")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    tag = ("_" + args.tag) if args.tag else ""
    torch.manual_seed(0)

    model = load_model(args.ckpt, args.K, args.ic, args.kernel)
    Xsub = mnist_train(args.nsub)                       # (Nsub,1,28,28)
    X1_flat = Xsub.view(args.nsub, DIM)

    def build_dict(Pd):
        pat = F.unfold(F.pad(Xsub, (Pd // 2,) * 4), Pd)                   # (Nsub, PP, HW)
        pat = pat.permute(0, 2, 1).reshape(-1, Pd * Pd).contiguous()      # (Npatch, PP)
        cen = Xsub.view(args.nsub, DIM).reshape(-1).contiguous()          # center pixel
        return pat, cen, (pat ** 2).sum(1)

    # --- P(t) schedule from the measured effective RF (erf_vs_time_k3) ---
    # eff. radius r(t): 3.77 @0.05 ... 1.74 @0.95  ->  patch P ~ 2r+1, snapped to odd, clamp[3,11]
    RF_T = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]
    RF_R = [3.77, 3.10, 2.65, 2.45, 2.13, 1.83, 1.74]
    def P_of_t(t):
        r = float(np.interp(t, RF_T, RF_R)); P = int(round(2 * r + 1))
        P = P if P % 2 == 1 else P + 1
        return min(11, max(3, P))

    x0 = torch.randn(args.nseed, DIM, device=dev)       # SAME seeds for all
    if args.model_euler:
        gen = euler(lambda x, t: model_velocity(model, x, t), x0.clone(), args.nsteps).clamp(-1, 1)
    else:
        gen = model_sample(model, x0.clone()).clamp(-1, 1)
    ideal = euler(lambda x, t: is_fm_velocity(x, t, X1_flat), x0.clone(), args.nsteps).clamp(-1, 1)

    if args.calibrate:
        Ps = [3, 5, 7, 9, 11]
        dicts = {P: build_dict(P) for P in Ps}
        Xval = mnist_train(16).view(16, DIM)                          # validation digits (flat)
        tgrid = list(np.linspace(0.05, 0.95, 13))
        sched = calibrate_Pt(model, dicts, Ps, Xval, tgrid)
        tg = np.array(tgrid)
        def P_cal(t): return sched[tgrid[int(np.abs(tg - t).argmin())]]
        print("P(t) calibré:", {round(t, 2): sched[t] for t in tgrid})
        els = euler(lambda x, t: els_fm_velocity(x, t, *dicts[P_cal(t)], P_cal(t)),
                    x0.clone(), args.nsteps).clamp(-1, 1)
        Pd = "calib"
    elif args.schedule:
        Psched = sorted({P_of_t(k / args.nsteps) for k in range(args.nsteps)})
        dicts = {P: build_dict(P) for P in Psched}
        print(f"schedule P(t): {[P_of_t(k/args.nsteps) for k in range(0, args.nsteps, max(1,args.nsteps//8))]} "
              f"(échelles utilisées: {Psched})")
        def els_vel(x, t):
            P = P_of_t(t); pat, cen, pn = dicts[P]
            return els_fm_velocity(x, t, pat, cen, pn, P)
        els = euler(els_vel, x0.clone(), args.nsteps).clamp(-1, 1)
        Pd = "sched"
    else:
        Pd = args.P
        pat, cen, pn = build_dict(Pd)
        print(f"dict: {pat.shape[0]:,} patches (P={Pd}), Nsub={args.nsub}")
        els = euler(lambda x, t: els_fm_velocity(x, t, pat, cen, pn, Pd),
                    x0.clone(), args.nsteps).clamp(-1, 1)

    r_is  = [r2(gen[i], ideal[i]) for i in range(args.nseed)]
    r_els = [r2(gen[i], els[i])   for i in range(args.nseed)]
    lines = [f"ckpt={args.ckpt}  P={Pd} nsub={args.nsub} nseed={args.nseed}",
             f"median r2(model, IS-FM)  = {np.median(r_is):.3f}",
             f"median r2(model, ELS-FM) = {np.median(r_els):.3f}",
             f"ELS>IS on {int(np.sum(np.array(r_els)>np.array(r_is)))}/{args.nseed} seeds"]
    txt = "\n".join(lines); print(txt); open(f"els_fm_metrics{tag}.txt", "w").write(txt + "\n")

    def dn(v): return (v.view(S, S).cpu().numpy() + 1) / 2
    ns = min(args.nseed, 8)
    fig, ax = plt.subplots(ns, 3, figsize=(6, 2.0 * ns))
    for i in range(ns):
        for j, (img, ttl) in enumerate([
                (gen[i], "ConvScCP"),
                (els[i], f"ELS-FM\nr2={r_els[i]:.2f}"),
                (ideal[i], f"IS-FM\nr2={r_is[i]:.2f}")]):
            ax[i, j].imshow(dn(img), cmap="gray", vmin=0, vmax=1); ax[i, j].axis("off")
            if i == 0: ax[i, j].set_title(ttl, fontsize=9)
            elif j > 0: ax[i, j].set_title(ttl.split("\n")[-1], fontsize=8)
    plt.suptitle(f"Test final ELS-FM vs IS-FM — ConvScCP {args.tag}", fontsize=11)
    plt.tight_layout(); plt.savefig(f"els_fm_grid{tag}.png", dpi=130, bbox_inches="tight")
    print(f"saved -> els_fm_grid{tag}.png , els_fm_metrics{tag}.txt")


if __name__ == "__main__":
    main()
