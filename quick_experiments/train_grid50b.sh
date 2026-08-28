#!/bin/bash
# Suite corrigée du set 50 ep (lr PAR modèle). ScCP indep+OT déjà faits (lr 1e-1).
# Ici : ResNet OT + UNet(MinimalUNetFM, vrai UNet de Kamb) indep+OT, tous à lr 1e-4
# + schedule kamb (via kwargs des entrées). Tous chiffres, save-model.
set -u
cd ~/UNN_for_FM
PY=~/.venvs/unn/bin/python
LOG=claude.log

run() {  # $1=name  $2=coupling  $3=resultsdir  $4=lr
    echo "===== [grid50b] $1 ($2) lr=$4 start $(date '+%F %T') =====" | tee -a "$LOG"
    $PY run_mnist.py --only "$1" --coupling "$2" --digit -1 --epochs 50 --lr "$4" \
        --save-model --results-dir "$3" 2>&1 \
        | grep --line-buffered -E "loss:|Coupling|Params|ERROR|Model :|LR schedule" | tee -a "$LOG"
}

run MinimalResNetFM_L6_ic256 ot    results/grid50_ot 1e-4
run MinimalUNetFM_kamb       indep results/grid50    1e-4
run MinimalUNetFM_kamb       ot    results/grid50_ot 1e-4
echo "===== [grid50b] ALL DONE $(date '+%F %T') =====" | tee -a "$LOG"
