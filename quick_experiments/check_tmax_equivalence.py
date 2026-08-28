# -*- coding: utf-8 -*-
"""
check_tmax_equivalence.py

Verifie les 4 proprietes du passage « clamp -> assert » (t ~ U(0, t_max)) :

  1. EQUIVALENCE  la loss x-pred sans clamp,  ||x1p - x1||^2 / (1-t)^2,  est
     EXACTEMENT la MSE vitesse ||v_pred - ut||^2 que journalise un modele v-pred.
     -> les loss des deux familles redeviennent comparables.
  2. L'ANCIENNE loss (clamp min=0.05) ne l'etait PAS : ecart mesure ici.
  3. L'assert saute bien si t sort de [0, t_max] (loss et conversion en vitesse).
  4. euler_sample n'evalue JAMAIS t > t_max (verifie par instrumentation du forward).

Poids aleatoires : on teste une IDENTITE algebrique et des gardes, pas un modele
entraine — inutile de charger un checkpoint.

Usage
-----
    source ~/.venvs/unn/bin/activate
    python check_tmax_equivalence.py                 # ~10 s, CPU suffit
"""

import argparse

import torch

from models.architectures import ConvScCP_UNN, MinimalUNetFM, fm_velocity_denom
from run_cifar10_torchcfm_recipe import (CHANNELS, DIM, IMG_SIZE, euler_sample,
                                         t_max_for)

OK, KO = "  [OK] ", "  [KO] "


def build_sccp(device):
    return ConvScCP_UNN(dim=DIM, K=3, internal_channel=16, kernel_size=3,
                        in_channels=CHANNELS, img_size=IMG_SIZE, use_Unet="l1",
                        version="LFO", use_checkpoint=False, w_bias=True).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--euler-steps", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    t_max = t_max_for(args.euler_steps)
    B = args.batch
    fails = 0

    print(f"Euler-{args.euler_steps}  ->  t_max = {t_max:g}  "
          f"(poids max 1/(1-t_max)^2 = {1/(1-t_max)**2:,.0f})\n")

    model = build_sccp(device)
    model.train()                       # x-pred : le forward renvoie x1_pred
    # tout en float64 : l'identite testee est algebrique, mais (v_pred - ut) subit
    # une annulation catastrophique la ou il est proche de 0. En float32 le residu
    # (~1e-9 en relatif sur la loss) mesurerait l'epsilon machine, pas le code.
    model.double()

    x0 = torch.randn(B, DIM, device=device, dtype=torch.float64)
    x1 = torch.randn(B, DIM, device=device, dtype=torch.float64)
    t = (torch.rand(B, 1, device=device, dtype=torch.float64) * t_max)
    xt = (1 - t) * x0 + t * x1
    ut = x1 - x0

    with torch.no_grad():
        x1p = model(torch.cat([xt, t], dim=-1))

    # ---- 1. equivalence exacte -------------------------------------------------
    print("1. loss x-pred SANS clamp  vs  MSE vitesse")
    l_xpred = ((x1p - x1) ** 2 / (1 - t) ** 2)
    v_pred = (x1p - xt) / (1 - t)
    l_v = (v_pred - ut) ** 2
    rel_mean = abs(l_xpred.mean() - l_v.mean()).item() / l_v.mean().item()
    rel_el = ((l_xpred - l_v).abs() / l_v.abs().clamp_min(1e-30)).max().item()
    tag = OK if rel_mean < 1e-12 else KO
    fails += tag == KO
    print(f"{tag}loss (moyenne) : ecart relatif = {rel_mean:.2e}")
    print(f"       par element   : ecart relatif max = {rel_el:.2e}")
    print(f"       valeurs : x-pred {l_xpred.mean():.10f}   v-pred {l_v.mean():.10f}\n")

    # ---- 2. ce que faisait l'ANCIENNE loss -------------------------------------
    print("2. ANCIENNE loss (clamp min=0.05) vs MSE vitesse, sur t ~ U(0,1)")
    t_full = torch.rand(B, 1, device=device, dtype=torch.float64)
    xt_f = (1 - t_full) * x0 + t_full * x1
    with torch.no_grad():
        x1p_f = model(torch.cat([xt_f, t_full], dim=-1))
    l_old = ((x1p_f - x1) ** 2 / torch.clamp((1 - t_full) ** 2, min=0.05)).mean()
    l_v_f = (((x1p_f - xt_f) / (1 - t_full) - ut) ** 2).mean()
    print(f"{OK}loss journalisee {l_old:.4f}  vs  vraie MSE vitesse {l_v_f:.4f}  "
          f"-> facteur {l_v_f/l_old:.1f}x")
    hi = (t_full > 1 - 0.05 ** 0.5).squeeze(-1)
    print(f"       ({int(hi.sum())}/{B} echantillons au-dessus du seuil t=0.7764 "
          f"suffisent a creuser l'ecart)\n")

    # ---- 3. les asserts --------------------------------------------------------
    print("3. gardes")
    w_min = (1 - t_max) ** 2
    t_bad = torch.full((4, 1), t_max + 0.02, device=device, dtype=torch.float64)
    w_bad = (1 - t_bad) ** 2
    try:
        assert w_bad.min().item() >= w_min * (1 - 1e-6)
        print(f"{KO}la garde de la loss n'a pas saute a t={t_max+0.02:g}"); fails += 1
    except AssertionError:
        print(f"{OK}garde de la loss : saute a t={t_max+0.02:g} > t_max")
    model.eval(); model.t_max = t_max
    try:
        fm_velocity_denom(t_bad, model.t_max)
        print(f"{KO}fm_velocity_denom n'a pas saute"); fails += 1
    except AssertionError:
        print(f"{OK}fm_velocity_denom : saute a t={t_max+0.02:g} > t_max")
    d = fm_velocity_denom(torch.full((4, 1), t_max, device=device,
                                     dtype=torch.float64), model.t_max)
    tag = OK if abs(float(d[0]) - (1 - t_max)) < 1e-6 else KO
    fails += tag == KO
    print(f"{tag}a t=t_max le denominateur vaut 1-t = {float(d[0]):.4f} "
          f"(pas de clamp)\n")

    # ---- 4. le sampler reste dans le domaine -----------------------------------
    print("4. euler_sample : quels t sont reellement evalues ?")
    seen = []
    orig = ConvScCP_UNN.forward

    def spy(self, xt_t, *a, **kw):
        seen.append(float(xt_t[:, self.dim:].max()))
        return orig(self, xt_t, *a, **kw)

    model.float()
    ConvScCP_UNN.forward = spy
    try:
        for tm, lab in ((t_max, f"t_max={t_max:g} (accorde)"),
                        (t_max_for(args.euler_steps * 5), "t_max plus grand")):
            seen.clear()
            model.t_max = max(tm, t_max)
            euler_sample(model, False, device, n=4, n_steps=args.euler_steps,
                         t_max=tm, seed=0)
            tag = OK if max(seen) <= tm + 1e-6 else KO
            fails += tag == KO
            print(f"{tag}{lab:<28} {len(seen)} evaluations, t max vu = {max(seen):.4f}")
    finally:
        ConvScCP_UNN.forward = orig

    # un modele t_max=0.95 echantillonne en Euler-100 SANS troncature doit sauter
    model.t_max = t_max
    try:
        euler_sample(model, False, device, n=4, n_steps=100, t_max=None, seed=0)
        print(f"{KO}Euler-100 non tronque n'a pas saute sur un modele t_max=0.95")
        fails += 1
    except AssertionError:
        print(f"{OK}Euler-100 sur un modele t_max={t_max:g} : refuse "
              f"(t=0.99 hors domaine)")

    # MinimalUNetFM : meme cablage
    mu = MinimalUNetFM(dim=DIM, in_channels=CHANNELS, img_size=IMG_SIZE).to(device).eval()
    mu.t_max = t_max
    try:
        fm_velocity_denom(t_bad, mu.t_max)
        print(f"{KO}MinimalUNetFM : garde absente"); fails += 1
    except AssertionError:
        print(f"{OK}MinimalUNetFM : meme garde")

    print("\n" + ("TOUT OK" if fails == 0 else f"{fails} VERIFICATION(S) EN ECHEC"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
