"""Le collapse x1-pred vient-il de la FONCTION APPRISE (g≈μ) ou de l'ÉCHANTILLONNAGE
(conversion v=(g-x)/(1-t) → attracteur) ? Uniform collapse AUSSI → ce n'est pas la loss.

Entraîne SmallUNetX1 (x1, poids uniform) et SmallUNet (vitesse) en OT, puis :
  (A) PRÉDICTION x1 : pour de vrais chiffres bruités à t=0.2/0.5/0.8, affiche le g prédit
      (x1-model) et le x1 implicite (v-model). Si g est déjà la moyenne floue → hypothèse (i).
  (B) 3 ÉCHANTILLONNEURS depuis le même bruit :
      - vitesse (ODE)                     : référence, ne collapse pas
      - x1 via vitesse (g-x)/(1-t) (ODE)  : la voie qui collapse
      - x1 via DÉBRUITEUR direct (DDIM)   : g prédit, step déterministe, PAS de /(1-t) explosif
      Si le débruiteur direct donne des chiffres variés → collapse = (ii) échantillonnage, SAUVABLE.
      Si lui aussi donne la moyenne → collapse = (i) fonction, il faut la vitesse.

Sortie : x1_diag_<tag>.png , logge dans claude.log.
Usage : python x1_sampler_diag.py --epochs 30 --tag ot
"""
import argparse, time, torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper
from models.architectures import SmallUNet, SmallUNetX1

dev = "cuda" if torch.cuda.is_available() else "cpu"
DIM, S = 784, 28
LOG = open("claude.log", "a")
def log(*a): print(*a, flush=True); print(*a, file=LOG, flush=True)


def loader(batch):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True), ds


def raw_g(model, xt, t):
    """sortie brute (= x1 prédit) de SmallUNetX1, sans conversion vitesse."""
    return SmallUNet.forward(model, torch.cat([xt, t.view(-1, 1)], -1))


def vel(model, xt, t):
    return model(torch.cat([xt, t.view(-1, 1)], -1))


def train(model, kind, FM, ld, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr); t0 = time.time()
    for ep in range(epochs):
        model.train(); tot = 0.0; n = 0
        for xb, _ in ld:
            x1 = xb.to(dev).view(xb.size(0), -1); x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            out = model(torch.cat([xt, t.view(-1, 1)], -1))
            if kind == "x1":
                x1_tgt = xt + (1 - t.view(-1, 1)) * ut          # FIX: x1 réordonné cohérent avec xt (OT permute!)
                loss = torch.mean((out - x1_tgt) ** 2)          # UNIFORM (borné)
            else:
                loss = torch.mean((out - ut) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); n += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            log(f"  [{kind}] ep {ep+1}/{epochs} loss={tot/n:.4g} ({time.time()-t0:.0f}s)")
    model.eval(); return model


@torch.no_grad()
def sample_ode(model, x0, nsteps=100):
    node = NeuralODE(torch_wrapper(model), solver="dopri5", atol=1e-4, rtol=1e-4)
    return node.trajectory(x0, t_span=torch.linspace(0, 1, 2, device=dev))[-1]


@torch.no_grad()
def sample_denoiser(model_x1, x0, nsteps=50):
    """DDIM déterministe avec le x1 prédit : jamais de /(1-t) explosif, sort g à la fin."""
    ts = torch.linspace(0, 1, nsteps + 1, device=dev)
    x = x0.clone()
    for i in range(nsteps):
        t = ts[i] * torch.ones(x.size(0), device=dev)
        g = raw_g(model_x1, x, t)                                # x1 prédit
        if i == nsteps - 1:
            x = g; break
        x0h = (x - ts[i] * g) / max(1 - ts[i].item(), 0.05)      # x0 prédit
        x = (1 - ts[i + 1]) * x0h + ts[i + 1] * g                # step DDIM
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--nseed", type=int, default=8)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    tag = ("_" + args.tag) if args.tag else ""
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)
    ld, ds = loader(args.batch)
    log(f"=== x1_sampler_diag epochs={args.epochs} (OT) ===")

    torch.manual_seed(0); m_x1 = SmallUNetX1(base_ch=32).to(dev)
    torch.manual_seed(0); m_v = SmallUNet(base_ch=32).to(dev)
    log("[train] x1/uniform"); m_x1 = train(m_x1, "x1", FM, ld, args.epochs, args.lr)
    log("[train] vitesse");    m_v = train(m_v, "vel", FM, ld, args.epochs, args.lr)

    # μ (chiffre moyen) pour référence visuelle
    mu = torch.stack([ds[i][0] for i in range(2000)]).view(2000, DIM).mean(0).to(dev)

    # (A) prédiction x1 à t=0.2/0.5/0.8 sur de vrais chiffres bruités
    reals = torch.stack([ds[i][0] for i in range(4)]).view(4, DIM).to(dev)
    ts_probe = [0.2, 0.5, 0.8]
    dn = lambda v: (v.view(S, S).cpu().numpy() + 1) / 2
    figA, axA = plt.subplots(4, 2 + 2 * len(ts_probe), figsize=(2 * (2 + 2 * len(ts_probe)), 8))
    for r in range(4):
        x1 = reals[r:r+1]
        axA[r, 0].imshow(dn(x1[0]), cmap="gray", vmin=0, vmax=1); axA[r, 0].axis("off")
        axA[r, 1].imshow(dn(mu), cmap="gray", vmin=0, vmax=1); axA[r, 1].axis("off")
        if r == 0: axA[r, 0].set_title("vrai x1", fontsize=9); axA[r, 1].set_title("μ (moyenne)", fontsize=9)
        for k, tv in enumerate(ts_probe):
            x0 = torch.randn_like(x1); t = torch.tensor([tv], device=dev)
            xt = (1 - tv) * x0 + tv * x1
            with torch.no_grad():
                g = raw_g(m_x1, xt, t)                          # x1 prédit par x1-model
                gv = xt + (1 - tv) * vel(m_v, xt, t)            # x1 implicite du v-model
            axA[r, 2 + 2*k].imshow(dn(g[0]), cmap="gray", vmin=0, vmax=1); axA[r, 2+2*k].axis("off")
            axA[r, 3 + 2*k].imshow(dn(gv[0]), cmap="gray", vmin=0, vmax=1); axA[r, 3+2*k].axis("off")
            if r == 0:
                axA[r, 2+2*k].set_title(f"x1-model\nt={tv}", fontsize=8)
                axA[r, 3+2*k].set_title(f"v-model\nt={tv}", fontsize=8)
    figA.suptitle("(A) x1 prédit : le g du x1-model est-il déjà la moyenne μ ?", fontsize=12)
    figA.tight_layout(); figA.savefig(f"x1_diag_predict{tag}.png", dpi=130, bbox_inches="tight")

    # (B) 3 échantillonneurs depuis le MÊME bruit
    torch.manual_seed(1); x0 = torch.randn(args.nseed, DIM, device=dev)
    g_v   = sample_ode(m_v, x0.clone()).clamp(-1, 1)
    g_x1v = sample_ode(m_x1, x0.clone()).clamp(-1, 1)
    g_x1d = sample_denoiser(m_x1, x0.clone()).clamp(-1, 1)
    figB, axB = plt.subplots(args.nseed, 3, figsize=(6, 2 * args.nseed))
    for i in range(args.nseed):
        for j, (img, ttl) in enumerate([(g_v, "vitesse (ODE)"),
                                        (g_x1v, "x1 via vitesse\n(collapse ?)"),
                                        (g_x1d, "x1 débruiteur\ndirect (DDIM)")]):
            axB[i, j].imshow(dn(img[i]), cmap="gray", vmin=0, vmax=1); axB[i, j].axis("off")
            if i == 0: axB[i, j].set_title(ttl, fontsize=9)
    figB.suptitle("(B) 3 échantillonneurs (même bruit) — le débruiteur direct sauve-t-il x1 ?", fontsize=11)
    figB.tight_layout(); figB.savefig(f"x1_diag_sample{tag}.png", dpi=130, bbox_inches="tight")

    # mesure de diversité (std inter-échantillons) pour chiffrer le collapse
    def diversity(g): return float(g.std(0).mean())
    log(f"diversité inter-échantillons (std moy) : vitesse={diversity(g_v):.4f}  "
        f"x1-vitesse={diversity(g_x1v):.4f}  x1-débruiteur={diversity(g_x1d):.4f}")
    log(f"saved -> x1_diag_predict{tag}.png , x1_diag_sample{tag}.png")


if __name__ == "__main__":
    main()
