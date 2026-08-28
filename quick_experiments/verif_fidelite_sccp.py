# -*- coding: utf-8 -*-
"""
verif_fidelite_sccp.py

Le terme d'attache du ScCP est-il le bon ?

Le code passe z = x_t au terme f_z(x) = 1/2||z - x||^2, qui est le MAP d'un modele
d'observation z = x + bruit, a GAIN UNITE. Or le vrai modele direct du Flow
Matching est

    x_t = t.x1 + (1-t).eps        eps ~ N(0, I)

soit un probleme inverse d'operateur A = t.I et de bruit sigma = 1-t. Le terme
d'attache honnete est donc

    f(x) = ||x_t - t.x||^2 / (2 (1-t)^2)                                     (*)

Ce script etablit trois choses.

A. La forme fermee du prox de (*) — celle qui remplacerait la mise a jour primale
   actuelle x_next = (v + tau.z)/(1+tau) :

       x_next = [ (1-t)^2 . v  +  tau . t . x_t ]  /  [ (1-t)^2 + tau . t^2 ]

   Verifiee par la condition d'optimalite (residu machine), pas par un argmin
   numerique approche. Aucune division par t ni par (1-t).

B. Son conditionnement : les deux poids restent bornes sur tout ]0,1[. A t->0
   l'attache s'eteint d'elle-meme (x_t ne dit rien de x1), a t->1 elle domine
   (x_t = x1). Le x_t/t "naturel" apparait bien, mais PONDERE par mu(t), donc sa
   divergence est guerie par la theorie et non par une normalisation ad hoc.

C. La constante de forte convexite mu(t) = t^2/(1-t)^2, qui pilote le momentum
   accelere : alpha_k = (1 + 2 mu tau_k)^(-1/2). Le code fixe mu = 1 (cf. le
   rapport : "the strong convexity constant of f_z is exactly 1"). C'est vrai
   pour f_z, faux pour (*) : mu = 1 SEULEMENT a t = 0.5.

   C'est le point qui distingue ce changement du simple rescaling d'entree : une
   erreur d'echelle sur l'OBJECTIF se reporte sur g, qui est appris et conditionne
   en temps (on a mesure que le reseau compensait, alpha* = 1.0055 dans
   analyse_input_scaling.py). Le calendrier d'acceleration, lui, est dans la
   DYNAMIQUE de l'algorithme, et tau_k est appris mais independant de t : aucun
   mecanisme ne peut le corriger.

Sortie -> verif_fidelite_sccp.txt

Usage
-----
    source ~/.venvs/unn/bin/activate
    python verif_fidelite_sccp.py
"""

import torch

torch.set_default_dtype(torch.float64)
T_LIST = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
TAUS = [0.1, 0.6, 3.0]


def prox_closed(v, xt, t, tau):
    """prox_{tau.f} pour f(x) = ||xt - t x||^2 / (2(1-t)^2)."""
    s2 = (1.0 - t) ** 2
    return (s2 * v + tau * t * xt) / (s2 + tau * t ** 2)


def main():
    torch.manual_seed(0)
    L = []

    L += ["=" * 76,
          "A. La forme fermee est-elle exacte ?",
          "=" * 76,
          "On evalue le gradient de  tau.f(x) + 1/2||x - v||^2  au point donne par la",
          "forme fermee : il doit etre nul.", "",
          f"{'t':>7}" + "".join(f"{'tau=' + str(tau):>16}" for tau in TAUS)]
    worst = 0.0
    for t in T_LIST:
        row = f"{t:>7.2f}"
        for tau in TAUS:
            v, xt = torch.randn(4096), torch.randn(4096)
            s2 = (1.0 - t) ** 2
            x = prox_closed(v, xt, t, tau)
            grad = tau * (-t) * (xt - t * x) / s2 + (x - v)
            r = float(grad.abs().max()); worst = max(worst, r)
            row += f"{r:>16.2e}"
        L.append(row)
    L += ["", f"residu max : {worst:.2e}  -> forme fermee exacte", ""]

    L += ["=" * 76,
          "B. Conditionnement de la nouvelle mise a jour primale (tau = 0.6)",
          "=" * 76,
          "x_next = [ (1-t)^2 . v + tau.t.x_t ] / [ (1-t)^2 + tau.t^2 ]", "",
          f"{'t':>7}{'poids sur v':>14}{'poids sur x_t':>16}"
          f"{'mu(t)':>13}{'poids/mu':>12}"]
    for t in T_LIST:
        s2, tau = (1.0 - t) ** 2, 0.6
        den = s2 + tau * t ** 2
        mu = t ** 2 / s2
        # tau.t.x_t = tau.mu.(1-t)^2 . (x_t/t) : l'accroche porte sur x_t/t, ponderee
        L.append(f"{t:>7.2f}{s2/den:>14.4f}{tau*t/den:>16.4f}{mu:>13.4f}"
                 f"{tau*mu*s2/den:>12.4f}")
    L += ["",
          "La derniere colonne est le poids porte par x_t/t (observation a gain unite).",
          "Il tend vers 0 quand t -> 0 : la divergence de x_t/t est annulee par mu(t).",
          "Aucun coefficient ne diverge sur ]0,1[.", ""]

    L += ["=" * 76,
          "C. Le momentum : mu(t) contre l'hypothese mu = 1 du code",
          "=" * 76,
          "alpha_k = (1 + 2 mu tau_k)^(-1/2), ici a tau_k = 0.6", "",
          f"{'t':>7}{'mu(t)':>13}{'alpha correct':>16}{'alpha code':>13}"
          f"{'erreur sur mu':>16}"]
    for t in T_LIST:
        mu = t ** 2 / (1.0 - t) ** 2
        a_ok = (1 + 2 * mu * 0.6) ** -0.5
        a_code = (1 + 2 * 0.6) ** -0.5
        L.append(f"{t:>7.2f}{mu:>13.4f}{a_ok:>16.4f}{a_code:>13.4f}{mu:>15.1f}x")
    L += ["",
          "mu = 1 exactement a t = 0.5, et nulle part ailleurs. Le deroule actuel est",
          "donc la methode acceleree du bon probleme en UN SEUL point de l'axe des temps.",
          "",
          "tau_k est appris (log_tau0) mais INDEPENDANT de t : alpha_k est le meme pour",
          "tous les t. Il n'existe aucun mecanisme, appris ou non, pour retrouver la",
          "variation de mu(t) — a la difference d'une erreur d'echelle sur l'objectif,",
          "que g(t) absorbe (cf. analyse_input_scaling.py).", ""]

    txt = "\n".join(L)
    print(txt, flush=True)
    open("verif_fidelite_sccp.txt", "w").write(txt + "\n")
    print("-> verif_fidelite_sccp.txt", flush=True)


if __name__ == "__main__":
    main()
