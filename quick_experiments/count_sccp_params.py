# -*- coding: utf-8 -*-
"""
count_sccp_params.py
Decompose le nombre de parametres de ScCP_UNN (version FLAT, 2-moons) couche par
couche, et verifie la formule analytique :

  LNO : K * (dual*dim  +  dual  +  P_prox)  +  K          (K log_tau)
  LFO : K * (2*dual*dim +  dual  +  P_prox)  +  1          (1 log_tau0)

avec, pour prox_type="l1", P_prox = params de L1ProxFlat = (1*w + w) + (w*1 + 1)
= 3w + 1 (w = 32 par defaut, HARDCODE dans L1ProxFlat, independant de dual_dim :
le prox n'apprend qu'un rayon scalaire r(t)).

Usage
-----
    python count_sccp_params.py                 # dim=2, K=10, dual=32
    python count_sccp_params.py --dual 32 64 128 256
"""
import argparse
import collections

from models.architectures import ScCP_UNN


def breakdown(dim, K, dual, version, prox_type="l1", w_bias=True):
    m = ScCP_UNN(dim=dim, K=K, dual_dim=dual, version=version,
                 prox_type=prox_type, w_bias=w_bias)
    agg = collections.Counter()
    for name, p in m.named_parameters():
        key = name.split(".", 2)[-1] if name.startswith(("layers", "prox_list")) else name
        agg[key] += p.numel()
    return sum(p.numel() for p in m.parameters()), agg


def formula(dim, K, dual, version, w=32, w_bias=True):
    p_prox = 3 * w + 1
    per_layer = dual * dim + (dual if w_bias else 0) + p_prox
    if version == "LFO":
        per_layer += dim * dual
    return K * per_layer + (K if version == "LNO" else 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--dual", type=int, nargs="+", default=[32])
    p.add_argument("--prox-type", default="l1", choices=["l1", "mlp"])
    args = p.parse_args()

    for dual in args.dual:
        print(f"\n=== dim={args.dim}  K={args.K}  dual_dim={dual}  prox={args.prox_type} ===")
        for version in ("LNO", "LFO"):
            total, agg = breakdown(args.dim, args.K, dual, version, args.prox_type)
            line = "  ".join(f"{k}={v}" for k, v in agg.items())
            print(f"  {version}: total={total}   [{line}]")
            if args.prox_type == "l1":
                th = formula(args.dim, args.K, dual, version)
                ok = "OK" if th == total else f"MISMATCH (formule={th})"
                print(f"        formule -> {th}  {ok}")


if __name__ == "__main__":
    main()
