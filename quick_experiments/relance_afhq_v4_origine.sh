#!/usr/bin/env bash
# Reprise A L'IDENTIQUE du run AFHQ v4 d'origine (aucun --compile, ckpt=1 par
# defaut), seule difference : --device cuda:1 au lieu de cuda:0.
# La config 1 est deja au budget (20 000 steps) et sera seulement re-evaluee ;
# la config 2 repart de zero avec la meme graine.
set -u
cd /home/ec4036/UNN_for_FM
exec /home/ec4036/.venvs/unn/bin/python denoise_probe.py \
  --steps 20000 --device cuda:1 --results-dir results_denoise_afhq_v4 \
  --configs "arch=sccp,k=9,K=10,ic=128; arch=sccp_v4,k=9,K=10,ic=128,x0=xt; arch=sccp,ver=LNO,k=9,K=10,ic=128; arch=sccp_v4,ver=LNO,k=9,K=10,ic=128,x0=xt"
