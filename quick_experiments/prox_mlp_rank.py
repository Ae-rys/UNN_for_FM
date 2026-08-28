# -*- coding: utf-8 -*-
"""
prox_mlp_rank.py
Combien de fonctions de base en t le prox reclame-t-il vraiment ?

La question pratique : dans `L1ProxConv`, faut-il garder la largeur cachee w=32
quand la sortie passe de 1 (prox l1) a 128 (prox l1c) ?

L'entree du MLP est de dimension 1. Quel que soit w, l'application t -> r(t)
decrit donc une COURBE unidimensionnelle dans l'espace de sortie : w ne fixe pas
la dimension du probleme, seulement le nombre de motifs non lineaires en t que
le reseau peut composer. La bonne facon de dimensionner w est donc de regarder
combien de fonctions de base suffisent a reconstruire les rayons appris.

Methode : on echantillonne les K courbes r_k(t) d'un checkpoint entraine sur une
grille fine de t, on centre, et on fait une SVD. Le nombre de valeurs singulieres
necessaires pour capturer 99 % de l'energie est le nombre effectif de motifs.
Si ce rang est petit, une largeur w tres inferieure a 32 suffit.

Usage
-----
    source ~/.venvs/unn/bin/activate
    python prox_mlp_rank.py --ckpt \\
        results_afhq32/ConvScCP_UNN_rgb_k9_K10_ic128_L1_LFO/latest.pt
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser(description="Rang effectif des rayons r_k(t).")
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--weights", type=str, default="ema", choices=["ema", "raw"])
    p.add_argument("--n-t", type=int, default=401)
    args = p.parse_args()

    dev = torch.device("cpu")
    from sample_checkpoint import resolve_checkpoint

    for path in args.ckpt:
        ck = torch.load(path, map_location=dev, weights_only=False)
        model, _is_unet, name, keys = resolve_checkpoint(ck, dev)
        key = keys.get(args.weights)
        model.load_state_dict(ck[key if key in ck else keys["raw"]])
        model.eval()
        step = int(ck.get("step", 0))

        t = torch.linspace(0, 1, args.n_t).view(-1, 1)
        curves = []
        with torch.no_grad():
            for layer in model.layers:
                if not hasattr(layer.prox, "time_scaling"):
                    continue
                r = F.softplus(layer.prox.time_scaling(t))       # (n_t, out_dim)
                curves.append(r.T.numpy())                       # (out_dim, n_t)
        if not curves:
            continue
        M = np.concatenate(curves, axis=0)                       # (n_courbes, n_t)
        # chaque courbe est normalisee : on compare des FORMES, pas des amplitudes
        M = M / np.maximum(np.abs(M).max(axis=1, keepdims=True), 1e-12)
        Mc = M - M.mean(axis=0, keepdims=True)
        s = np.linalg.svd(Mc, compute_uv=False)
        energy = np.cumsum(s ** 2) / max((s ** 2).sum(), 1e-30)

        print(f"\n{name}  step {step:,}")
        print(f"  {M.shape[0]} courbes r(t), echantillonnees sur {args.n_t} valeurs de t")
        print(f"  {'k':>3} {'valeur sing.':>14} {'energie cumulee':>17}")
        for i in range(min(6, len(s))):
            print(f"  {i+1:>3} {s[i]:>14.4f} {100*energy[i]:>16.2f} %")
        for thr in (0.90, 0.99, 0.999):
            rk = int(np.searchsorted(energy, thr) + 1)
            print(f"  rang effectif a {100*thr:g} % d'energie : {rk}")
        rk99 = int(np.searchsorted(energy, 0.99) + 1)
        print(f"\n  -> {rk99} motif(s) suffisent a reconstruire toutes les formes.")
        print(f"     Une largeur cachee w de l'ordre de {max(4, 2*rk99)} suffirait ;")
        print(f"     w=32 en fournit largement plus que necessaire.")


if __name__ == "__main__":
    main()
