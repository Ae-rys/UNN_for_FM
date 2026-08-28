#!/bin/bash
# Test ELS-OT : la prédictibilité ELS (dérivée en couplage INDÉPENDANT, Kamb/NIFTY)
# tient-elle pour un modèle entraîné en couplage OT (bug OT corrigé dans train.py) ?
# Entraîne ScCP k3 en OT (MÊME config que le checkpoint indep temp-5 : tous chiffres,
# K=6 ic=128 kernel=3, 100 ep, lr défaut), répertoire séparé, puis éval ELS/IS.
set -u
cd ~/UNN_for_FM
PY=~/.venvs/unn/bin/python
LOG=claude.log
CKPT=results/ot_els_test/ConvScCP_k3_K6_ic128_L1_LNO/model.pt

echo "===== [ELS-OT] TRAIN start $(date '+%F %T') =====" | tee -a "$LOG"
$PY run_mnist.py --only ConvScCP_k3_K6_ic128_L1_LNO --coupling ot --digit -1 \
    --epochs 100 --save-model --results-dir results/ot_els_test 2>&1 | tee -a "$LOG"

if [ -f "$CKPT" ]; then
    echo "===== [ELS-OT] EVAL ELS start $(date '+%F %T') =====" | tee -a "$LOG"
    $PY nifty_els_fm.py --ckpt "$CKPT" --K 6 --ic 128 --kernel 3 --tag sccp_k3_ot 2>&1 | tee -a "$LOG"
    echo "===== [ELS-OT] DONE $(date '+%F %T') — voir nifty_els_metrics_sccp_k3_ot.txt =====" | tee -a "$LOG"
else
    echo "===== [ELS-OT] ÉCHEC : pas de checkpoint $CKPT =====" | tee -a "$LOG"
fi
