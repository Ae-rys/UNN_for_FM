#!/usr/bin/env bash
# Relance du probe AFHQ v4 en "mode rapide" : identique au run d'origine
# (memes 4 configs, meme results-dir, meme steps) + --compile et ckpt=0.
#   - --compile        : ~1.9x (cf. bench_speedup_levers.py)
#   - ckpt=0           : coupe le gradient checkpointing, +26 % (la VRAM est libre)
# La reprise est automatique : chaque config repart de son results_denoise_afhq_v4/
# <nom>/ckpt.pt (le nom de config ne depend PAS de ckpt, donc les checkpoints
# ecrits par le run precedent sont bien retrouves), et une config deja au budget
# est simplement re-evaluee.
set -u
cd /home/ec4036/UNN_for_FM
exec ~/.venvs/unn/bin/python denoise_probe.py \
  --steps 20000 --device cuda:0 --results-dir results_denoise_afhq_v4 --compile \
  --configs "arch=sccp,k=9,K=10,ic=128,ckpt=0; arch=sccp_v4,k=9,K=10,ic=128,x0=xt,ckpt=0; arch=sccp,ver=LNO,k=9,K=10,ic=128,ckpt=0; arch=sccp_v4,ver=LNO,k=9,K=10,ic=128,x0=xt,ckpt=0"
