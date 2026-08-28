#!/bin/bash
# Warm-start du dual ConvScCP a K=15 — complete le balayage K = {3, 6, 20}.
#
# Reproduit EXACTEMENT le protocole de K=6 (le seul K deja depouille avec sc1) :
#   1. entrainement de la paire baseline / warmstart   (run_warmstart_mnist.py)
#   2. depouillement 6 lignes avec le controle sc1     (analyze_warmstart_runs.py)
#   3. diagnostic d'oubli de u^(0) sur les 2 modeles   (diag_u0_forgetting.py)
#
# L'etape 2 est celle qui produit le chiffre qui compte : sc1 = u recalcule par une
# passe FROIDE sur l'etat precedent (condition d'entrainement, sans recursion). C'est
# lui qui separe "conditionner sur un dual" de "chainer le long de la trajectoire".
#
# Usage :  ./run_warmstart_mnist_k15.sh [GPU] [EPOCHS]
#          ./run_warmstart_mnist_k15.sh 1 200
# Suivi  :  tail -f ~/UNN_for_FM/claude.log
set -u
cd ~/UNN_for_FM
GPU=${1:-1}
EPOCHS=${2:-200}
PY=~/.venvs/unn/bin/python
LOG=~/UNN_for_FM/claude.log
OUT=results/warmstart_mnist_k15

{
  echo "===== [warmstart k15] start $(date '+%F %T') GPU=$GPU epochs=$EPOCHS ====="

  T0=$(date +%s)
  echo "--- [1/3] entrainement de la paire (baseline + warmstart), K=15 ---"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" run_warmstart_mnist.py \
      --runs both --epochs "$EPOCHS" --self-cond-rate 0.9 \
      --K 15 --ic 64 --kernel 9 --version LNO --digit 0 --coupling ot \
      --n-eval 2000 --outdir "$OUT" || { echo "[k15] ECHEC etape 1"; exit 1; }
  T1=$(date +%s); echo "--- [1/3] fini en $((T1-T0)) s ---"

  echo "--- [2/3] depouillement cold / warm / sc1 ---"
  CUDA_VISIBLE_DEVICES=$GPU "$PY" analyze_warmstart_runs.py \
      --outdir "$OUT" --modes cold,warm,sc1 \
      --n-eval 2000 --eval-batch 500 --digit 0 --seed 0 \
      || { echo "[k15] ECHEC etape 2"; exit 1; }
  T2=$(date +%s); echo "--- [2/3] fini en $((T2-T1)) s ---"

  echo "--- [3/3] diagnostic d'oubli de u^(0) ---"
  for TAG in baseline warmstart; do
    CUDA_VISIBLE_DEVICES=$GPU "$PY" diag_u0_forgetting.py \
        --ckpt "$OUT/$TAG/model.pt" --outdir "$OUT/diag_$TAG" \
        --steps 100 --seed 0 || echo "[k15] diag $TAG en echec (non bloquant)"
  done
  T3=$(date +%s); echo "--- [3/3] fini en $((T3-T2)) s ---"

  echo "===== [warmstart k15] fini $(date '+%F %T') — total $((T3-T0)) s ====="
  echo "Table a reporter dans le rapport : $OUT/analysis_table.md"
  echo "Diagnostic (canal ouvert / ferme) : $OUT/diag_{baseline,warmstart}/u0_forgetting.txt"
} >> "$LOG" 2>&1 &

echo "PID $! -> suivi : tail -f $LOG"
