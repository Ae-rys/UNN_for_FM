"""Où vit l'erreur du x1-model vs v-model (OT) : à petit t ou à grand t ?
Teste la prédiction 'l'écart se concentre à petit t' (autre Claude).

Entraîne SmallUNetX1 (x1, uniform) et SmallUNet (vitesse) en OT, puis par bin de t
(sur des paires OT-appariées held-out) mesure :
  - erreur DÉBRUITEUR   E‖x1_pred - x1‖²   (reconstruction de l'image propre)
  - erreur VITESSE      E‖v_pred - u‖²
  - distance à μ         E‖x1_pred - μ‖²    (0 => le modèle prédit la moyenne)
x1_pred : g pour x1-model ; xt+(1-t)v pour v-model. v_pred : (g-xt)/(1-t) ; v.

Sortie : error_vs_t.png , error_vs_t_metrics.txt  (logge dans claude.log)
Usage : python error_vs_t.py --epochs 25
"""
import argparse, time, torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from models.architectures import SmallUNet, SmallUNetX1

dev = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 784
LOG = open("claude.log", "a")
def log(*a): print(*a, flush=True); print(*a, file=LOG, flush=True)


def mnist_loader(batch, n=None):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True), ds


def raw_g(model, xt, t):
    return SmallUNet.forward(model, torch.cat([xt, t.view(-1, 1)], -1))


def train(model, kind, FM, ld, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr); t0 = time.time()
    for ep in range(epochs):
        model.train(); tot = 0.0; n = 0
        for xb, _ in ld:
            x1 = xb.to(dev).view(xb.size(0), -1); x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            out = model(torch.cat([xt, t.view(-1, 1)], -1))
            loss = torch.mean((out - x1) ** 2) if kind == "x1" else torch.mean((out - ut) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); n += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            log(f"  [{kind}] ep {ep+1}/{epochs} loss={tot/n:.4g} ({time.time()-t0:.0f}s)")
    model.eval(); return model


@torch.no_grad()
def errors_at_t(m_x1, m_v, FM, x1_val, mu, tv, B=512):
    x1 = x1_val[torch.randperm(x1_val.size(0))[:B]]
    x0 = torch.randn_like(x1)
    x0p, x1p = FM.ot_sampler.sample_plan(x0, x1)            # appariement OT (comme l'entraînement)
    t = torch.full((B,), float(tv), device=dev)
    xt = (1 - tv) * x0p + tv * x1p
    u = x1p - x0p
    omt = max(1 - tv, 0.05)
    # x1-model
    g = raw_g(m_x1, xt, t); v_x1 = (g - xt) / omt
    # v-model
    v = m_v(torch.cat([xt, t.view(-1, 1)], -1)); g_v = xt + (1 - tv) * v
    e = lambda a, b: float(((a - b) ** 2).mean())
    return dict(
        den_x1=e(g, x1p),   den_v=e(g_v, x1p),      # erreur débruiteur
        vel_x1=e(v_x1, u),  vel_v=e(v, u),          # erreur vitesse
        mu_x1=e(g, mu.expand_as(g)), mu_v=e(g_v, mu.expand_as(g_v)),  # distance à μ
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--nt", type=int, default=20)
    args = p.parse_args()
    t0 = time.time(); FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)
    ld, ds = mnist_loader(args.batch)
    log(f"=== error_vs_t epochs={args.epochs} (OT) ===")

    torch.manual_seed(0); m_x1 = SmallUNetX1(base_ch=32).to(dev)
    torch.manual_seed(0); m_v = SmallUNet(base_ch=32).to(dev)
    log("[train] x1/uniform"); train(m_x1, "x1", FM, ld, args.epochs, args.lr)
    log("[train] vitesse");    train(m_v, "vel", FM, ld, args.epochs, args.lr)

    x1_val = torch.stack([ds[i][0] for i in range(4000)]).view(4000, DIM).to(dev)
    mu = x1_val.mean(0, keepdim=True)
    var_x1 = float(x1_val.var(0, unbiased=False).mean())
    tgrid = np.round(np.linspace(0.05, 0.95, args.nt), 3)
    keys = ["den_x1", "den_v", "vel_x1", "vel_v", "mu_x1", "mu_v"]
    R = {k: np.zeros(args.nt) for k in keys}
    for j, tv in enumerate(tgrid):
        d = errors_at_t(m_x1, m_v, FM, x1_val, mu, tv)
        for k in keys: R[k][j] = d[k]
        log(f"  t={tv:.2f} den(x1={d['den_x1']:.3f} v={d['den_v']:.3f}) "
            f"vel(x1={d['vel_x1']:.2f} v={d['vel_v']:.2f}) distμ(x1={d['mu_x1']:.4f} v={d['mu_v']:.3f})")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(tgrid, R["den_x1"], "-o", ms=3, color="tab:red", label="x1-model")
    ax[0].plot(tgrid, R["den_v"], "-o", ms=3, color="k", label="v-model")
    ax[0].axhline(var_x1, ls="--", color="gray", lw=1, label="‖μ-x1‖²=Var(x1)")
    ax[0].set_title("erreur DÉBRUITEUR  E‖x1_pred - x1‖²"); ax[0].set_xlabel("t"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(tgrid, R["vel_x1"], "-o", ms=3, color="tab:red", label="x1-model")
    ax[1].plot(tgrid, R["vel_v"], "-o", ms=3, color="k", label="v-model")
    ax[1].set_title("erreur VITESSE  E‖v_pred - u‖²"); ax[1].set_xlabel("t"); ax[1].set_yscale("log")
    ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot(tgrid, R["mu_x1"], "-o", ms=3, color="tab:red", label="x1-model")
    ax[2].plot(tgrid, R["mu_v"], "-o", ms=3, color="k", label="v-model")
    ax[2].set_title("distance à μ  E‖x1_pred - μ‖²  (0 => prédit la moyenne)")
    ax[2].set_xlabel("t"); ax[2].legend(); ax[2].grid(alpha=.3)
    plt.suptitle("Où vit l'erreur : x1-model vs v-model (OT)", fontsize=13)
    plt.tight_layout(); plt.savefig("error_vs_t.png", dpi=130, bbox_inches="tight")

    lines = [f"Var(x1)={var_x1:.4f}", "t\t" + "\t".join(keys)]
    for j, tv in enumerate(tgrid):
        lines.append(f"{tv:.3f}\t" + "\t".join(f"{R[k][j]:.4f}" for k in keys))
    gap = R["den_x1"] - R["den_v"]
    lines += ["", f"écart débruiteur (x1 - v) : petit t (t<0.3)={gap[tgrid<0.3].mean():.3f}  "
                  f"grand t (t>0.7)={gap[tgrid>0.7].mean():.3f}"]
    open("error_vs_t_metrics.txt", "w").write("\n".join(lines) + "\n")
    log("\n".join(lines[-2:])); log(f"saved -> error_vs_t.png , error_vs_t_metrics.txt ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
