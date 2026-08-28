"""Entraîne SmallUNet en mode VITESSE et en mode x1, en couplage OT, puis compare
leurs CHAMPS DE GRADIENT en fonction de t — pour voir À QUEL MOMENT (t) ils diffèrent.

Trois vues (toutes en fonction de t) :
  1. ‖∂L/∂θ‖(t)     : norme du gradient PARAMÈTRES par t = le vrai signal d'entraînement.
                      Sondé à l'INIT (poids identiques → isole l'effet paramétrisation/poids)
                      ET après entraînement.
  2. ‖∂L/∂out‖(t)   : gradient en espace-sortie, ramené en espace-VITESSE pour comparaison
                      équitable (x1 : ∂L/∂v = (1-t)·∂L/∂g).
  3. divergence des champs de vitesse APPRIS : ‖v_x1(xt,t) - v_vel(xt,t)‖ et cos, par t.

Modes x1 comparés : 'invsq' (legacy 1/(1-t)², explose) et 'uniform' (MSE borné).
=> on voit l'explosion du signal x1-invsq à t→1, et où les modèles appris divergent.

Sorties (repo) : grad_fields_<tag>.png , grad_fields_metrics_<tag>.txt
Usage :
  python compare_grad_fields.py --coupling ot --epochs 40 --tag ot
  python compare_grad_fields.py --coupling indep --epochs 40 --tag indep   # pour contraster
"""
import argparse, time, torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher)
from models.architectures import SmallUNet, SmallUNetX1

dev = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 784
LOG = open("claude.log", "a")
def log(*a):
    print(*a, flush=True); print(*a, file=LOG, flush=True)


def loader(batch):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])    # -> [-1,1]
    ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)


def x1_loss(out, x1, t, weight, gamma=5.0):
    omt2 = (1 - t.view(-1, 1)) ** 2
    if weight == "uniform":
        w = torch.ones_like(omt2)
    elif weight == "minsnr":
        w = torch.clamp(1.0 / omt2, max=gamma)
    else:  # invsq
        w = 1.0 / torch.clamp(omt2, min=0.000005 ** 2)
    return torch.mean(w * (out - x1) ** 2)


def make_model(kind, seed=0):
    """SmallUNet (vitesse) ou SmallUNetX1 (x1), MÊME init pour un seed donné."""
    torch.manual_seed(seed)
    m = (SmallUNetX1 if kind == "x1" else SmallUNet)(base_ch=32).to(dev)
    return m


def forward(model, xt, t):
    return model(torch.cat([xt, t.view(-1, 1)], dim=-1))


def compute_loss(model, kind, xt, t, x1, ut, x1_weight):
    out = forward(model, xt, t)
    if kind == "x1":
        return x1_loss(out, x1, t, x1_weight)
    return torch.mean((out - ut) ** 2)


def train(model, kind, FM, ld, epochs, lr, x1_weight):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for ep in range(epochs):
        model.train(); tot = 0.0; n = 0
        for xb, _ in ld:
            x1 = xb.to(dev).view(xb.size(0), -1)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            loss = compute_loss(model, kind, xt, t, x1, ut, x1_weight)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item(); n += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            log(f"    [{kind}/{x1_weight}] epoch {ep+1}/{epochs} loss={tot/n:.4g} ({time.time()-t0:.0f}s)")
    return model


@torch.no_grad()
def _batch_at_t(FM, ld_iter, ld, t_val, B):
    try:
        xb, _ = next(ld_iter)
    except StopIteration:
        ld_iter = iter(ld); xb, _ = next(ld_iter)
    x1 = xb.to(dev).view(xb.size(0), -1)[:B]
    x0 = torch.randn_like(x1)
    tvec = torch.full((x1.size(0),), float(t_val), device=dev)
    t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, t=tvec)   # OT interne si OT
    return t, xt, ut, x1, ld_iter


def grad_field(model, kind, FM, ld, tgrid, x1_weight, nprobe=4, B=128):
    """Par t : ‖∂L/∂θ‖ et ‖∂L/∂v‖ (espace vitesse), moyennés sur nprobe batches."""
    gtheta = np.zeros(len(tgrid)); gout = np.zeros(len(tgrid))
    ld_iter = iter(ld)
    for j, tv in enumerate(tgrid):
        gt = 0.0; go = 0.0
        for _ in range(nprobe):
            t, xt, ut, x1, ld_iter = _batch_at_t(FM, ld_iter, ld, tv, B)
            out = forward(model, xt, t); out.retain_grad()
            loss = (x1_loss(out, x1, t, x1_weight) if kind == "x1"
                    else torch.mean((out - ut) ** 2))
            model.zero_grad(); loss.backward()
            gt += float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()
                                       if p.grad is not None)))
            # ∂L/∂out en espace vitesse : x1 -> ∂L/∂v = (1-t)·∂L/∂g
            gvec = out.grad * ((1 - t.view(-1, 1)) if kind == "x1" else 1.0)
            go += float(gvec.norm())
        gtheta[j] = gt / nprobe; gout[j] = go / nprobe
    return gtheta, gout


@torch.no_grad()
def velocity(model, kind, xt, t):
    out = forward(model, xt, t)
    if kind == "x1":
        return (out - xt) / torch.clamp(1 - t.view(-1, 1), min=0.05)
    return out


@torch.no_grad()
def field_divergence(m_x1, m_v, FM, ld, tgrid, B=128):
    """‖v_x1 - v_vel‖ / ‖v_vel‖ et cos, aux MÊMES (xt,t), par t."""
    rel = np.zeros(len(tgrid)); cos = np.zeros(len(tgrid))
    ld_iter = iter(ld)
    for j, tv in enumerate(tgrid):
        t, xt, ut, x1, ld_iter = _batch_at_t(FM, ld_iter, ld, tv, B)
        v1 = velocity(m_x1, "x1", xt, t); vv = velocity(m_v, "vel", xt, t)
        rel[j] = float((v1 - vv).norm() / (vv.norm() + 1e-9))
        a = v1 - v1.mean(); b = vv - vv.mean()
        cos[j] = float((a * b).sum() / (a.norm() * b.norm() + 1e-9))
    return rel, cos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coupling", choices=["ot", "indep"], default="ot")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--x1_weights", nargs="*", default=["invsq", "uniform"])
    p.add_argument("--nprobe", type=int, default=4)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    tag = ("_" + args.tag) if args.tag else ""
    FM = (ExactOptimalTransportConditionalFlowMatcher(sigma=0.1) if args.coupling == "ot"
          else ConditionalFlowMatcher(sigma=0.1))
    ld = loader(args.batch)
    tgrid = np.round(np.linspace(0.05, 0.97, 24), 3)
    log(f"=== compare_grad_fields coupling={args.coupling} epochs={args.epochs} ===")

    # 1) champ de gradient À L'INIT (poids identiques -> isole l'effet paramétrisation)
    log("[init] sonde du champ de gradient à l'initialisation (poids identiques)")
    m_v0 = make_model("vel"); gt_v0, go_v0 = grad_field(m_v0, "vel", FM, ld, tgrid, None, args.nprobe)
    init_x1 = {}
    for w in args.x1_weights:
        m0 = make_model("x1")                 # même seed -> mêmes poids que m_v0
        init_x1[w] = grad_field(m0, "x1", FM, ld, tgrid, w, args.nprobe)

    # 2) entraînement des deux (des trois) modèles en OT
    log("[train] vitesse")
    m_v = train(make_model("vel"), "vel", FM, ld, args.epochs, args.lr, None)
    m_x1 = {}
    for w in args.x1_weights:
        log(f"[train] x1 / {w}")
        m_x1[w] = train(make_model("x1"), "x1", FM, ld, args.epochs, args.lr, w)

    # 3) champ de gradient APRÈS entraînement + divergence des champs de vitesse appris
    gt_v, go_v = grad_field(m_v, "vel", FM, ld, tgrid, None, args.nprobe)
    fin_x1 = {w: grad_field(m_x1[w], "x1", FM, ld, tgrid, w, args.nprobe) for w in args.x1_weights}
    divs = {w: field_divergence(m_x1[w], m_v, FM, ld, tgrid) for w in args.x1_weights}

    # ---- figures ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    C = {"invsq": "tab:red", "uniform": "tab:green", "minsnr": "tab:orange"}
    # (a) grad params à l'init
    ax[0, 0].semilogy(tgrid, gt_v0, "k-o", ms=3, label="vitesse")
    for w in args.x1_weights:
        ax[0, 0].semilogy(tgrid, init_x1[w][0], "-o", ms=3, color=C.get(w), label=f"x1/{w}")
    ax[0, 0].set_title("(a) ‖∂L/∂θ‖(t) À L'INIT (poids identiques)"); ax[0, 0].set_xlabel("t")
    ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)
    # (b) grad params après entraînement
    ax[0, 1].semilogy(tgrid, gt_v, "k-o", ms=3, label="vitesse")
    for w in args.x1_weights:
        ax[0, 1].semilogy(tgrid, fin_x1[w][0], "-o", ms=3, color=C.get(w), label=f"x1/{w}")
    ax[0, 1].set_title("(b) ‖∂L/∂θ‖(t) APRÈS entraînement"); ax[0, 1].set_xlabel("t")
    ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)
    # (c) grad sortie en espace vitesse (init)
    ax[1, 0].semilogy(tgrid, go_v0, "k-o", ms=3, label="vitesse")
    for w in args.x1_weights:
        ax[1, 0].semilogy(tgrid, init_x1[w][1], "-o", ms=3, color=C.get(w), label=f"x1/{w}")
    ax[1, 0].set_title("(c) ‖∂L/∂v‖(t) espace-vitesse, à l'init"); ax[1, 0].set_xlabel("t")
    ax[1, 0].legend(); ax[1, 0].grid(alpha=.3)
    # (d) divergence des champs de vitesse appris
    for w in args.x1_weights:
        ax[1, 1].plot(tgrid, divs[w][0], "-o", ms=3, color=C.get(w), label=f"x1/{w} vs vitesse (rel.)")
    ax[1, 1].set_title("(d) divergence champs appris  ‖v_x1-v_vel‖/‖v_vel‖ (t)")
    ax[1, 1].set_xlabel("t"); ax[1, 1].legend(); ax[1, 1].grid(alpha=.3)
    plt.suptitle(f"Champs de gradient : x1 vs vitesse — couplage {args.coupling}", fontsize=13)
    plt.tight_layout(); plt.savefig(f"grad_fields{tag}.png", dpi=130, bbox_inches="tight")

    # ---- metrics ----
    lines = [f"coupling={args.coupling} epochs={args.epochs} lr={args.lr}",
             "t\tgθ_vel_init\t" + "\t".join(f"gθ_x1_{w}_init" for w in args.x1_weights) +
             "\tgθ_vel_fin\t" + "\t".join(f"gθ_x1_{w}_fin" for w in args.x1_weights) +
             "\t" + "\t".join(f"reldiv_{w}" for w in args.x1_weights)]
    for j, tv in enumerate(tgrid):
        row = [f"{tv:.3f}", f"{gt_v0[j]:.3e}"] + [f"{init_x1[w][0][j]:.3e}" for w in args.x1_weights]
        row += [f"{gt_v[j]:.3e}"] + [f"{fin_x1[w][0][j]:.3e}" for w in args.x1_weights]
        row += [f"{divs[w][0][j]:.3f}" for w in args.x1_weights]
        lines.append("\t".join(row))
    # résumé : ratio d'explosion à t→1
    hi = tgrid > 0.9
    summ = ["", "RÉSUMÉ (t>0.9, signal params moyen) :",
            f"  vitesse       = {gt_v0[hi].mean():.3e}"]
    for w in args.x1_weights:
        summ.append(f"  x1/{w:8s} = {init_x1[w][0][hi].mean():.3e}  "
                    f"(×{init_x1[w][0][hi].mean()/max(gt_v0[hi].mean(),1e-12):.1f} vs vitesse)")
    txt = "\n".join(lines + summ); log(txt)
    open(f"grad_fields_metrics{tag}.txt", "w").write(txt + "\n")
    log(f"saved -> grad_fields{tag}.png , grad_fields_metrics{tag}.txt")


if __name__ == "__main__":
    main()
