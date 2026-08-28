#!/bin/bash
# Entraîne le set 50 ep pour les grilles ELS : ScCP indep+OT, ResNet OT, UNet OT.
# (ResNet/UNet indep réutilisés de temp-5, déjà 50 ep.) Tous chiffres, save-model.
set -u
cd ~/UNN_for_FM
PY=~/.venvs/unn/bin/python
LOG=claude.log

run() {  # $1=name  $2=coupling  $3=resultsdir
    echo "===== [grid50] $1 ($2) start $(date '+%F %T') =====" | tee -a "$LOG"
    $PY run_mnist.py --only "$1" --coupling "$2" --digit -1 --epochs 50 \
        --save-model --results-dir "$3" 2>&1 \
        | grep --line-buffered -E "loss:|Coupling|Params|ERROR|Model :" | tee -a "$LOG"
}

run ConvScCP_k3_K6_ic128_L1_LNO indep results/grid50
run ConvScCP_k3_K6_ic128_L1_LNO ot    results/grid50_ot
run MinimalResNetFM_L6_ic256    ot    results/grid50_ot
run SmallUNet_baseline          ot    results/grid50_ot
echo "===== [grid50] ALL DONE $(date '+%F %T') =====" | tee -a "$LOG"
