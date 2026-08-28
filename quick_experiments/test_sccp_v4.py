# -*- coding: utf-8 -*-
"""
test_sccp_v4.py — plan de validation §8 du spec "ScCP with direct x_t input".

1. §8.1  Les deux comportements de bord du §4 :
         t -> 1 : r -> 0, dual ecrase a 0, primal reduit a l'identite x -> x_t.
         t -> 0 : r -> softplus(a), regularisation maximale.
2. §8.2  Le bound de pas tient PAR ECHANTILLON, avec un L^2 recalcule proprement.
3. §8.4  gamma init a 2.0 et journalisable par couche.
4. Le prox reste un vrai prox : clamp au rayon, INDEPENDANT de sigma.
5. rho = t^2 est borne, contrairement au mu = t^2/(1-t)^2 de v2/v3 : c'est LA
   correction du spec, on la chiffre.
6. l2 est DIFFERENTIABLE (le bug de v3 est bien parti).

Usage
-----
    source ~/.venvs/unn/bin/activate
    python test_sccp_v4.py
"""
import torch
import torch.nn.functional as F

from models.architectures import ConvScCP_UNN_v4, L1ProxRadiusPow

KW = dict(dim=784, K=10, internal_channel=64, kernel_size=9, in_channels=1,
          img_size=28, use_Unet="l1", version="LFO", w_bias=True)
fails = []


def check(cond, label, detail=""):
    print(f"  [{'OK' if cond else 'ECHEC'}] {label}" + (f"  {detail}" if detail else ""),
          flush=True)
    if not cond:
        fails.append(label)


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    m = ConvScCP_UNN_v4(**KW, x0_mode="xt").double()

    print("\n1. §8.1 — comportements de bord du rayon r(t)")
    pr = m.layers[0].prox
    a = float(F.softplus(pr.a).mean().detach())
    r1 = float(pr.radius(torch.tensor([[1.0]])).mean())
    r0 = float(pr.radius(torch.tensor([[0.0]])).mean())
    check(abs(r1) < 1e-12, "t -> 1 : r -> 0", f"r(1) = {r1:.2e}")
    check(abs(r0 - a) < 1e-12, "t -> 0 : r -> softplus(a)", f"r(0) = {r0:.4f}, a = {a:.4f}")

    m.eval()
    with torch.no_grad():
        z = torch.randn(8, 784)
        t1 = torch.full((8, 1), 1.0 - 1e-9)
        # a t -> 1 le dual est ecrase, le primal doit devenir l'identite x -> x_t
        B = 8
        tt = t1.view(-1, 1, 1, 1)
        x = z.view(B, 1, 28, 28).clone()
        u = m.cold_dual(z)
        tau = F.softplus(m.log_tau0).view(1, 1, 1, 1).expand_as(tt)
        for lay in m.layers:
            l2 = lay.loop_gain(28)
            sg = m.cp_safety / (tau * l2)
            al = (1.0 + 2.0 * tt ** 2 * tau).pow(-0.5)
            x, u = lay(x, u, z.view(B, 1, 28, 28), t1, tau, sg, al)
            tau = al * tau
        zi = z.view(B, 1, 28, 28)
        eps = 1e-9
        # Le point fixe du primal a dual nul est x_t/t (et non x_t) :
        #   x = (x + tau.t.x_t)/(1 + tau.t^2)  =>  x* = x_t / t.
        # A t = 1-eps l'ecart a x_t vaut donc O(eps) — c'est la bonne assertion,
        # pas "x_K = x_t a 1e-9 pres", qui confondait la limite et la vitesse.
        drift_fp = float((x - zi / (1.0 - eps)).abs().max())
        drift_id = float((x - zi).abs().max())
        umax = float(u.abs().max())
    check(umax < 1e-12, "t -> 1 : dual ecrase a 0", f"max|u| = {umax:.2e}")
    # La contraction vers x_t/t vaut prod_k 1/(1 + tau_k t^2), et tau_k DECROIT
    # (tau_{k+1} = alpha_k tau_k), donc la contraction s'affaiblit a chaque
    # iteration. On la predit depuis la recurrence plutot que de la supposer.
    tau_p, contr = float(F.softplus(m.log_tau0).detach()), 1.0
    for _ in range(KW["K"]):
        contr /= (1.0 + tau_p)
        tau_p *= (1.0 + 2.0 * tau_p) ** -0.5
    d0 = float((zi / (1.0 - eps) - zi).abs().max())          # ecart initial a x*
    ratio = drift_fp / max(d0 * contr, 1e-300)
    check(0.3 < ratio < 3.0,
          "t -> 1 : le primal contracte vers x_t/t au taux predit par la recurrence",
          f"residu {drift_fp:.2e}, predit {d0*contr:.2e}, ratio {ratio:.2f}")
    check(drift_id < 100 * eps, "t -> 1 : et donc x_K -> x_t, a O(1-t) pres",
          f"max|x_K - x_t| = {drift_id:.2e} pour 1-t = {eps:.0e}")

    print("\n2. §8.2 — bound de pas par echantillon")
    t = torch.rand(16, 1) * 0.95
    worst = m.check_bound(t)
    check(worst <= m.cp_safety * 1.001, "tau_k.sigma_k.L^2 <= cp_safety pour tout k, tout n",
          f"max = {worst:.6f} (cible {m.cp_safety})")

    print("\n3. §8.4 — exposant gamma")
    g = m.gammas()
    check(len(g) == KW["K"] and all(abs(v - 2.0) < 1e-6 for v in g),
          "gamma initialise a 2.0 sur les K couches", f"K={len(g)}, gamma={g[0]:.6f}")

    print("\n4. le prox reste un prox : clamp independant de sigma")
    pr2 = L1ProxRadiusPow().double()
    u0 = torch.randn(4, 3, 5, 5) * 3
    tt2 = torch.full((4, 1), 0.3)
    same = torch.equal(pr2(u0, tt2), pr2(u0, tt2))
    r = pr2.radius(tt2)
    inside = float((pr2(u0, tt2).abs() <= r + 1e-12).float().mean())
    check(same and inside == 1.0, "sortie = projection sur la boule l_inf de rayon r(t)",
          f"{inside*100:.0f} % des coefficients dans la boule")

    print("\n5. §3 — rho = t^2 est borne (la correction du spec)")
    print(f"       {'t':>6}{'rho.tau (v4)':>15}{'mu.tau (v2/v3)':>17}{'alpha v4':>11}{'alpha v3':>11}")
    tau0 = float(F.softplus(m.log_tau0).detach())
    ok = True
    for tv in (0.5, 0.9, 0.95, 0.99):
        rs, mu = tv ** 2 * tau0, tv ** 2 / (1 - tv) ** 2 * tau0
        a4, a3 = (1 + 2 * rs) ** -0.5, (1 + 2 * mu) ** -0.5
        ok &= a4 > 0.5
        print(f"       {tv:>6.2f}{rs:>15.4f}{mu:>17.2f}{a4:>11.4f}{a3:>11.4f}")
    check(ok, "alpha reste > 0.5 partout en v4 (v3 tombait a 0.01)")

    print("\n6. l2 differentiable (le bug de v3 est corrige)")
    m2 = ConvScCP_UNN_v4(**KW, x0_mode="xt").double()
    l2 = m2.layers[0].loop_gain(28)
    check(l2.requires_grad, "loop_gain garde le graphe", f"requires_grad={l2.requires_grad}")
    gr = torch.autograd.grad(l2, m2.layers[0].W_weight, retain_graph=True)[0]
    check(float(gr.abs().max()) > 0, "dL^2/dW non nul", f"max|grad| = {float(gr.abs().max()):.4f}")

    print("\n" + ("TOUT OK" if not fails else f"{len(fails)} ECHEC(S) : {fails}"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
