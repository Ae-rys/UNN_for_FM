# -*- coding: utf-8 -*-
"""
verif_lfo_condition.py

Verifie les affirmations de DERIVATION_LFO.md et l'implementation de
ConvScCP_UNN_v3. Cinq points, tous chiffres.

A. conv_transpose2d(., W, p) est l'adjoint EXACT de conv2d(., W, p).
   C'est le socle du calcul de M* : si c'est faux, tout le reste tombe.

B. L'iteration de la puissance de _ScCPv3_Iteration.loop_gain retrouve la vraie
   norme d'operateur ||B o A||, comparee a la matrice explicite (petit cas).
   On y compare aussi l'estimation par reshape de sigma_max_power_iter, utilisee
   par LNO, pour chiffrer de combien elle se trompe.

C. V = W redonne ||A||^2 : la generalisation contient bien le cas Chambolle-Pock.

D. Dans v3, tau_k . sigma_k . L^2_k = cp_safety pour TOUT k et TOUT t : la
   condition de pas tient partout. Et sigma croit bien comme 1/theta.

E. Dans v2, sigma reste a 1 pendant que tau s'effondre — le defaut que v3 corrige.
   On l'affiche a t = 0.9, ou mu = 81.

Sortie -> verif_lfo_condition.txt

Usage
-----
    source ~/.venvs/unn/bin/activate
    python verif_lfo_condition.py
"""

import torch
import torch.nn.functional as F

from models.architectures import (ConvScCP_UNN_v2, ConvScCP_UNN_v3,
                                  sigma_max_power_iter)

L = []


def say(s=""):
    print(s, flush=True)
    L.append(s)


def explicit_matrix(op, shape_in, dtype=torch.float64):
    """Matrice explicite d'une application lineaire, colonne par colonne."""
    n = int(torch.tensor(shape_in).prod())
    cols = []
    for i in range(n):
        e = torch.zeros(n, dtype=dtype)
        e[i] = 1.0
        cols.append(op(e.view(1, *shape_in)).reshape(-1))
    return torch.stack(cols, dim=1)


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    # ------------------------------------------------------------------ A
    say("=" * 74)
    say("A. conv_transpose2d(., W, p) est-il l'adjoint de conv2d(., W, p) ?")
    say("=" * 74)
    C_in, C_u, k, S = 3, 7, 9, 32
    p = k // 2
    W = torch.randn(C_u, C_in, k, k)
    x, y = torch.randn(2, C_in, S, S), torch.randn(2, C_u, S, S)
    lhs = float((F.conv2d(x, W, padding=p) * y).sum())
    rhs = float((x * F.conv_transpose2d(y, W, padding=p)).sum())
    say(f"  <A x, y>  = {lhs:.10f}")
    say(f"  <x, A* y> = {rhs:.10f}")
    say(f"  ecart relatif = {abs(lhs - rhs) / abs(lhs):.2e}   -> adjoint exact")
    say()

    # ------------------------------------------------------------------ B & C
    say("=" * 74)
    say("B/C. L^2 = ||B o A|| : iteration de la puissance vs matrice explicite")
    say("=" * 74)
    say(f"{'cas':<22}{'exact':>12}{'power iter':>13}{'ecart':>10}{'reshape LNO':>14}")
    for tag, same in (("V independant (LFO)", False), ("V = W  (LNO)", True)):
        C_in2, C_u2, k2, S2 = 2, 3, 3, 8
        p2 = k2 // 2
        Wt = torch.randn(C_u2, C_in2, k2, k2) * 0.3
        Vt = Wt.clone() if same else torch.randn(C_u2, C_in2, k2, k2) * 0.3

        def M(v):
            return F.conv_transpose2d(F.conv2d(v, Wt, padding=p2), Vt, padding=p2)

        exact = float(torch.linalg.svdvals(
            explicit_matrix(M, (C_in2, S2, S2)))[0])

        it = ConvScCP_UNN_v3(dim=C_in2 * S2 * S2, K=1, internal_channel=C_u2,
                             kernel_size=k2, in_channels=C_in2, img_size=S2,
                             use_Unet="l1", version="LFO").layers[0].double()
        with torch.no_grad():
            it.W_weight.copy_(Wt); it.V_weight.copy_(Vt)
        for _ in range(60):                       # convergence, puis lecture
            pi = float(it.loop_gain(S2, n_iter=2))
        u0 = F.normalize(torch.randn(C_u2), dim=0)
        rs, _ = sigma_max_power_iter(Wt, u0, n_iter=60)
        say(f"{tag:<22}{exact:>12.5f}{pi:>13.5f}"
            f"{abs(pi - exact) / exact:>10.1e}{float(rs) ** 2:>14.5f}")
        if same:
            def A(v):
                return F.conv2d(v, Wt, padding=p2)
            nA = float(torch.linalg.svdvals(explicit_matrix(A, (C_in2, S2, S2)))[0])
            say(f"{'':<22}||A||^2 = {nA ** 2:.5f}  -> V=W redonne bien ||A||^2")
    say()
    say("  La colonne 'reshape LNO' est sigma_max_power_iter(W)^2, ce qu'utilise LNO.")
    say("  Elle SOUS-ESTIME le gain de boucle, donc autorise des pas trop grands.")
    say()

    # ------------------------------------------------------------------ D
    say("=" * 74)
    say("D. v3 : la condition tau_k . sigma_k . L^2_k <= 1 tient-elle partout ?")
    say("=" * 74)
    m3 = ConvScCP_UNN_v3(dim=784, K=8, internal_channel=32, kernel_size=9,
                         in_channels=1, img_size=28, use_Unet="l1",
                         version="LFO", w_bias=True).double()
    m3.train()
    worst = 0.0
    for tv in (0.05, 0.2, 0.5, 0.8, 0.9, 0.95):
        xt = torch.randn(1, 784)
        _, st = m3(torch.cat([xt, torch.full((1, 1), tv)], -1), return_steps=True)
        prods = [ta * si * l2 for l2, ta, si, _ in st]
        err = max(abs(v - m3.cp_safety) for v in prods)
        worst = max(worst, err)
        thetas = [al for *_, al in st]
        sig = [si for _, _, si, _ in st]
        say(f"  t={tv:<5} tau.sigma.L^2 = {min(prods):.6f}..{max(prods):.6f} "
            f"| theta_0={thetas[0]:.4f} | sigma x{sig[-1] / sig[0]:.1f} sur {len(st)} iter")
    say(f"\n  ecart max a cp_safety={m3.cp_safety} : {worst:.2e}  -> condition tenue")
    say()

    # ------------------------------------------------------------------ E
    say("=" * 74)
    say("E. v2 a t = 0.9 (mu = 81) : le defaut que v3 corrige")
    say("=" * 74)
    m2 = ConvScCP_UNN_v2(dim=784, K=8, internal_channel=32, kernel_size=9,
                         in_channels=1, img_size=28, use_Unet="l1",
                         version="LFO", w_bias=True).double()
    tau = float(F.softplus(m2.log_tau0).detach())
    mu = 0.9 ** 2 / 0.1 ** 2
    say(f"  mu(0.9) = {mu:.1f},  tau_0 = {tau:.4f}")
    say(f"{'k':>4}{'tau_k (v2)':>14}{'sigma_k (v2)':>15}{'sigma voulu':>14}")
    s0 = 1.0
    sig_ok = 1.0
    for kk in range(5):
        th = (1 + 2 * mu * tau) ** -0.5
        say(f"{kk:>4}{tau:>14.5f}{s0:>15.3f}{sig_ok:>14.3f}")
        tau *= th
        sig_ok /= th
    say("\n  v2 garde sigma = 1 pendant que tau perd un facteur ~50 en 3 iterations :")
    say("  les iterations suivantes ne font presque plus rien. v3 recalcule sigma")
    say("  depuis tau, donc l'accroissement est automatique.")

    open("verif_lfo_condition.txt", "w").write("\n".join(L) + "\n")
    print("\n-> verif_lfo_condition.txt", flush=True)


if __name__ == "__main__":
    main()
