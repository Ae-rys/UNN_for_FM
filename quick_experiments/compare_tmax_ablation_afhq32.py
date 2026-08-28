# -*- coding: utf-8 -*-
"""
compare_tmax_ablation_afhq32.py

ABLATION du domaine temporel, a tout le reste egal.

Deux runs UNet_torchcfm_ch32, meme archi (1.11M), meme recette torchcfm, meme
couplage OT, meme budget (200k steps). SEULE difference : le domaine ou t est tire.

    results_afhq32/UNet_torchcfm_ch32          t ~ U(0, 1)
    results_afhq32_tmax095/UNet_torchcfm_ch32  t ~ U(0, 0.95)

Le UNet est v-pred : sa loss n'a JAMAIS eu de clamp. Le tirage de t est donc
litteralement la seule variable — c'est le controle propre de "couper a t_max
fait-il perdre en performance ?".

Deux questions, deux mesures
----------------------------
1. NMSE par t, sur une grille qui DEPASSE 0.95 (jusqu'a 0.99). Sur [0, 0.95] les
   deux modeles sont dans leur domaine : qui gagne, et de combien ? Au-dela, le
   modele t_max EXTRAPOLE : combien coute cette zone jamais vue ?

2. Echantillons avec trois solveurs, meme graine :
       Euler-20   grille 0, .05, ..., .95   -> le modele t_max reste DANS son domaine
       Euler-100  grille 0, .01, ..., .99   -> il en sort sur les 4 derniers pas
       dopri5     adaptatif, t -> 1         -> il en sort franchement
   Si couper a t_max est sans danger, les trois colonnes doivent se ressembler.

Sorties -> compare_tmax_ablation_afhq32.png  (NMSE)
           compare_tmax_ablation_samples.png (echantillons)
           compare_tmax_ablation_afhq32.txt

Usage
-----
    source ~/.venvs/unn/bin/activate
    python compare_tmax_ablation_afhq32.py --device cuda:0
"""

import argparse
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchdyn.core import NeuralODE

from compute_fid_cifar10 import build_from_name

IMG_SIZE, CHANNELS = 32, 3
DIM = CHANNELS * IMG_SIZE * IMG_SIZE
N_VAL, SEED = 512, 0

# Deux paires ancien/nouveau regime. Dans chaque paire, TOUT est identique sauf le
# domaine de tirage de t — meme archi, meme recette, meme couplage, meme budget.
PAIRS = {
    "unet": ("UNet_torchcfm_ch32", 200000),
    # Le ScCP est x-pred : sa loss a change AUSSI (le clamp a saute). La paire
    # n'isole donc pas le seul domaine de t, contrairement au UNet — a lire en
    # gardant ca en tete.
    "sccp": ("ConvScCP_UNN_rgb_k9_K10_ic128_L1_LFO", 120000),
}
T_GRID = [round(0.05 * i, 2) for i in range(1, 20)] + [0.96, 0.97, 0.98, 0.99]
T_CUT = 0.95


class VF(torch.nn.Module):
    """Champ de vitesse image -> image, signature attendue par NeuralODE.
    Gere les deux conventions d'appel : UNet (t, image) et x-pred ([x aplati, t])."""
    def __init__(self, m, is_unet=True):
        super().__init__()
        self.m = m
        self.is_unet = is_unet
        self.nfe = 0

    def forward(self, t, x, *a, **kw):
        self.nfe += 1
        if self.is_unet:
            return self.m(t.expand(x.shape[0]), x)
        b = x.shape[0]
        xt_t = torch.cat([x.view(b, -1), t.expand(b).view(b, 1)], dim=-1)
        return self.m(xt_t).view_as(x)


@torch.no_grad()
def sample(vf, n, device, solver, steps=20, seed=0):
    """Retourne (images, nfe). Euler : dernier pas jusqu'a t=1 depuis 1-1/steps."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, CHANNELS, IMG_SIZE, IMG_SIZE, generator=g).to(device)
    vf.nfe = 0
    if solver == "euler":
        grid = [i / steps for i in range(steps)]
        for i, t in enumerate(grid):
            t_next = grid[i + 1] if i + 1 < len(grid) else 1.0
            x = x + vf(torch.full((1,), t, device=device), x) * (t_next - t)
        return x, vf.nfe
    node = NeuralODE(vf, solver="dopri5", atol=1e-5, rtol=1e-5)
    x = node.trajectory(x, t_span=torch.linspace(0, 1, 2, device=device))[-1]
    return x, vf.nfe


def to_img(x):
    return (x.detach().cpu().float() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--cache", type=str, default="./data/afhq_cat32_train.pt")
    p.add_argument("--n-show", type=int, default=8)
    p.add_argument("--pair", type=str, default="unet", choices=sorted(PAIRS),
                   help="quelle paire ancien/nouveau regime comparer")
    p.add_argument("--step", type=int, default=None,
                   help="budget apparie (defaut : celui de la paire)")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    run_name, dflt_step = PAIRS[args.pair]
    step = args.step or dflt_step
    runs_spec = [
        ("t ~ U(0, 1)",    f"results_afhq32/{run_name}/ckpt_step_{step}.pt",        "#7A3E9D"),
        ("t ~ U(0, 0.95)", f"results_afhq32_tmax095/{run_name}/ckpt_step_{step}.pt", "#1B5E8C"),
    ]

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    d = torch.load(args.cache, map_location="cpu", weights_only=False)
    x = d["data"].float().div_(127.5).sub_(1.0)
    g = torch.Generator().manual_seed(args.seed)
    x = x[torch.randperm(x.shape[0], generator=g)]
    xva = x[:N_VAL].reshape(N_VAL, -1)
    var_x1 = float(((xva - xva.mean(0, keepdim=True)) ** 2).mean())
    gv = torch.Generator().manual_seed(args.seed + 1234)
    x0_va = torch.randn(N_VAL, DIM, generator=gv)
    var_ut = float(((x0_va - xva) ** 2).mean())
    x1, x0 = xva.to(device), x0_va.to(device)
    ut = x1 - x0
    print(f"eval sur {N_VAL} images | var(x1)={var_x1:.4f} var(ut)={var_ut:.4f}", flush=True)

    t0 = time.perf_counter()
    models, curves, samples = [], {}, {}
    for label, path, col in runs_spec:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m, is_unet = build_from_name(ck["name"], device)
        m.load_state_dict(ck["ema_model"], strict=True)
        m.t_max = ck.get("t_max")
        # x-pred : .train() fait sortir x1_pred brut (aucun dropout/BN dans ces
        # archis), ce qui evite le clamp d'eval et rend les deux runs comparables.
        m.eval() if is_unet else m.train()
        models.append((label, m, col, is_unet))
        print(f"  {label:<16} {path.split('/')[0]:<24} step {ck['step']:,} "
              f"| t_max={ck.get('t_max')}", flush=True)

        nx, nv = [], []
        with torch.no_grad():
            for tv in T_GRID:
                t = torch.full((N_VAL, 1), tv, device=device)
                xt = (1 - t) * x0 + t * x1
                if is_unet:
                    v = m(t.view(-1),
                          xt.view(-1, CHANNELS, IMG_SIZE, IMG_SIZE)).reshape(N_VAL, -1)
                    x1p = xt + (1 - t) * v
                else:
                    # x-pred en .train() : sortie = x1_pred brut, sans clamp, donc
                    # la conversion en vitesse est exacte pour les DEUX runs.
                    x1p = m(torch.cat([xt, t], dim=-1))
                    v = (x1p - xt) / (1 - t)
                nx.append(float(((x1p - x1) ** 2).mean()) / var_x1)
                nv.append(float(((v - ut) ** 2).mean()) / var_ut)
        curves[label] = (nx, nv)

        vf = VF(m, is_unet).to(device).eval()
        m.eval()                       # l'echantillonnage veut la VITESSE
        for tag, kw in (("Euler-20", dict(solver="euler", steps=20)),
                        ("Euler-100", dict(solver="euler", steps=100)),
                        ("dopri5", dict(solver="dopri5"))):
            try:
                im, nfe = sample(vf, args.n_show, device, seed=args.seed, **kw)
            except AssertionError:
                # modele t_max=0.95 pousse hors de son domaine : fm_velocity_denom
                # REFUSE. C'est un resultat, pas un plantage.
                samples[(label, tag)] = (None, 0)
                print(f"    {tag:<10} refuse (t hors du domaine appris)", flush=True)
                continue
            samples[(label, tag)] = (to_img(im), nfe)
            print(f"    {tag:<10} {nfe:>4} NFE", flush=True)
        if not is_unet:
            m.train()                  # remettre en x-pred pour la suite

    # ------------------------------------------------------------------ texte
    lo = [i for i, t in enumerate(T_GRID) if t <= T_CUT + 1e-9]
    hi = [i for i, t in enumerate(T_GRID) if t > T_CUT + 1e-9]
    L = ["=" * 78,
         "Ablation du domaine temporel — UNet_torchcfm_ch32, 200k steps, tout egal",
         "=" * 78, "",
         f"{'t':>6}" + "".join(f"{lab:>22}" for lab, _, _, _u in models) + "     ecart",
         "-- nmse_x1(t) = MSE(x1_pred, x1) / var(x1) " + "-" * 26]
    for i, tv in enumerate(T_GRID):
        a, b = curves[models[0][0]][0][i], curves[models[1][0]][0][i]
        flag = "   <- hors domaine du modele coupe" if tv > T_CUT else ""
        L.append(f"{tv:>6.2f}{a:>22.4f}{b:>22.4f}{100*(b/a-1):>9.1f}%{flag}")
    L += ["", "-- nmse_v(t) = MSE(v_pred, ut) / var(ut) " + "-" * 29]
    for i, tv in enumerate(T_GRID):
        a, b = curves[models[0][0]][1][i], curves[models[1][0]][1][i]
        flag = "   <- hors domaine du modele coupe" if tv > T_CUT else ""
        L.append(f"{tv:>6.2f}{a:>22.4f}{b:>22.4f}{100*(b/a-1):>9.1f}%{flag}")

    L += ["", "=" * 78, "Moyennes par region", "=" * 78,
          f"{'region':<34}" + "".join(f"{lab:>22}" for lab, _, _, _u in models)]
    for nm, idx, k in (("t <= 0.95  (domaine partage)", lo, 0),
                       ("t >  0.95  (extrapolation)", hi, 0)):
        a = np.mean([curves[models[0][0]][k][i] for i in idx])
        b = np.mean([curves[models[1][0]][k][i] for i in idx])
        L.append(f"{'nmse_x1  ' + nm:<34}{a:>22.4f}{b:>22.4f}   ({100*(b/a-1):+.1f}%)")
    for nm, idx, k in (("t <= 0.95  (domaine partage)", lo, 1),
                       ("t >  0.95  (extrapolation)", hi, 1)):
        a = np.mean([curves[models[0][0]][k][i] for i in idx])
        b = np.mean([curves[models[1][0]][k][i] for i in idx])
        L.append(f"{'nmse_v   ' + nm:<34}{a:>22.4f}{b:>22.4f}   ({100*(b/a-1):+.1f}%)")

    # --- combien de trajectoire se joue APRES t_max, et les solveurs changent-ils
    #     vraiment quelque chose ? On mesure au lieu de juger a l'oeil.
    L += ["", "=" * 78,
          "Part de la trajectoire parcourue APRES t = 0.95  (Euler-100, meme bruit)",
          "=" * 78]
    for lab, m, col, is_unet in models:
        vf = VF(m, is_unet).to(device).eval()
        m.eval()
        g2 = torch.Generator().manual_seed(args.seed)
        xx = torch.randn(args.n_show, CHANNELS, IMG_SIZE, IMG_SIZE, generator=g2).to(device)
        x_start = xx.clone()
        x_at_cut = None
        try:
            with torch.no_grad():
                grid = [i / 100 for i in range(100)]
                for i, tv in enumerate(grid):
                    if x_at_cut is None and tv > T_CUT + 1e-9:
                        x_at_cut = xx.clone()
                    tn = grid[i + 1] if i + 1 < len(grid) else 1.0
                    xx = xx + vf(torch.full((1,), tv, device=device), xx) * (tn - tv)
        except AssertionError:
            L.append(f"  {lab:<18} refuse d'aller au-dela de t_max (garde active)")
            if not is_unet:
                m.train()
            continue
        tot = float((xx - x_start).pow(2).sum(dim=(1, 2, 3)).sqrt().mean())
        tail = float((xx - x_at_cut).pow(2).sum(dim=(1, 2, 3)).sqrt().mean())
        L.append(f"  {lab:<18} ||x(1)-x(0.95)|| / ||x(1)-x(0)||  =  {tail/tot:.4f}")
        if not is_unet:
            m.train()
    L += ["", "Autrement dit : l'essentiel de l'image est deja fixe avant t_max ; la zone",
          "hors domaine ne porte qu'une fraction du chemin.", ""]

    L += ["=" * 78,
          "Distance RMS entre echantillons, meme bruit initial  (images dans [-1,1])",
          "=" * 78]
    keys = [(lab, tag) for lab, _, _, _u in models for tag in ("Euler-20", "Euler-100", "dopri5")]
    names = [f"{'coupe' if '0.95' in l else 'complet'}/{t}" for l, t in keys]
    L.append(f"{'':>22}" + "".join(f"{n:>18}" for n in names))
    for i, ki in enumerate(keys):
        row = f"{names[i]:>22}"
        for kj in keys:
            if samples[ki][0] is None or samples[kj][0] is None:
                row += f"{'-':>18}"
            else:
                dv = float(np.sqrt(((samples[ki][0] - samples[kj][0]) * 2) ** 2).mean())
                row += f"{dv:>18.4f}"
        L.append(row)
    L += ["", "Reference : l'ecart RMS entre deux images AFHQ tirees au hasard vaut "
          f"{float(np.sqrt(((x[:64].reshape(64,-1) - x[64:128].reshape(64,-1))**2).mean())):.4f}.",
          "Un ecart tres inferieur signifie que les deux reglages donnent la MEME image.", ""]

    L += ["", f"paire : {args.pair}  |  budget apparie : {step:,} steps",
          "NFE par solveur : " + ", ".join(
        f"{tag} {samples[(models[0][0], tag)][1]}"
        for tag in ("Euler-20", "Euler-100", "dopri5"))]
    txt = "\n".join(L)
    print("\n" + txt, flush=True)
    open(f"compare_tmax_ablation_{args.pair}.txt", "w").write(txt + "\n")

    # ------------------------------------------------------------------ figure NMSE
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for lab, _, col, _u in models:
        ax[0].plot(T_GRID, curves[lab][0], "-o", ms=3.5, color=col, label=lab)
        ax[1].plot(T_GRID, curves[lab][1], "-o", ms=3.5, color=col, label=lab)
    for a, ttl, yl in ((ax[0], "nmse$_{x_1}$(t) — debruitage", "MSE / var($x_1$)"),
                       (ax[1], "nmse$_v$(t) — objectif", "MSE / var($u_t$)")):
        a.axvline(T_CUT, color="#C05621", ls=":", lw=1.4)
        a.annotate("t_max = 0.95", (T_CUT, a.get_ylim()[1]), color="#C05621",
                   fontsize=8, ha="right", va="top", rotation=90, xytext=(-4, -4),
                   textcoords="offset points")
        a.set_title(ttl); a.set_xlabel("t"); a.set_ylabel(yl)
        a.set_yscale("log"); a.grid(alpha=0.3); a.legend(fontsize=9)
    sub = ("seule la borne de t change" if args.pair == "unet" else
           "la borne de t ET le clamp de la loss changent")
    fig.suptitle(f"Couper le tirage de t a 0.95 — paire {args.pair}\n"
                 f"{run_name}, {step//1000}k steps, {sub}", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"compare_tmax_ablation_{args.pair}.png", dpi=130)

    # ------------------------------------------------------------- figure samples
    tags = ["Euler-20", "Euler-100", "dopri5"]
    nr, nc = len(models) * len(tags), args.n_show
    fig = plt.figure(figsize=(1.35 * nc + 3.0, 1.45 * nr))
    gs = fig.add_gridspec(nr, nc + 2, wspace=0.05, hspace=0.05)
    r = 0
    for lab, _, _, _u in models:
        for tag in tags:
            imgs, nfe = samples[(lab, tag)]
            if imgs is None:
                a = fig.add_subplot(gs[r, 0:2]); a.axis("off")
                a.text(0.97, 0.5, f"{lab}\n{tag}\nREFUSE : hors domaine",
                       ha="right", va="center", fontsize=8.5, color="#C05621",
                       family="monospace")
                r += 1
                continue
            for c in range(nc):
                a = fig.add_subplot(gs[r, c + 2])
                a.imshow(imgs[c]); a.set_xticks([]); a.set_yticks([])
            a = fig.add_subplot(gs[r, 0:2]); a.axis("off")
            oob = (tag != "Euler-20" and "0.95" in lab)
            a.text(0.97, 0.5, f"{lab}\n{tag}  ({nfe} NFE)" + ("\nhors domaine" if oob else ""),
                   ha="right", va="center", fontsize=8.5,
                   color="#C05621" if oob else "#333333",
                   family="monospace")
            r += 1
    fig.suptitle("Meme bruit initial, meme modele, trois solveurs\n"
                 "en orange : le solveur evalue des t que le modele n'a jamais vus",
                 fontsize=11)
    fig.savefig(f"compare_tmax_ablation_{args.pair}_samples.png", dpi=130, bbox_inches="tight")
    print(f"\n-> compare_tmax_ablation_{args.pair}.{{png,txt}} / _samples.png  "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
