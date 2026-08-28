#!/usr/bin/env bash
# run_sccp_l1c_fm.sh
# ScCP k9/K20/ic128 avec prox l1c (rayon par canal dual) en Flow Matching complet,
# 200 000 steps, meme protocole que les autres runs AFHQ (couplage OT, recette
# torchcfm). Compagnon FM du test de debruitage, qui donnait +1,8 % sur l1 —
# ecart trop petit pour etre distingue du bruit de graine sur le banc.
#
# Lancer DETACHE, sinon la fin de la session tue le run :
#     setsid nohup ./run_sccp_l1c_fm.sh > /dev/null 2>&1 < /dev/null & disown
#
# Reprise : relancer la meme commande, run_afhq32 repart de latest.pt.
set -u
cd "$(dirname "$0")" || exit 1
exec ~/.venvs/unn/bin/python run_afhq32.py \
    --only ConvScCP --prox l1c --K 20 --kernel 9 --ic 128 \
    --steps 200000 --sample-every 10000 --save-every 10000 --keep-every 10000 \
    >> claude.log 2>&1
