#!/bin/bash
# MNIST 300 epochs (140 700 iterations) — les 3 modeles de tab:quantitative.
# Motivation : a 100 epochs, ScCP LNO descendait encore a -0.092/100ep, soit 10x la
# pente de LFO (-0.009) et du SmallUNet (-0.006). Son FID de 80 etait donc en partie
# du sous-entrainement. LFO et SmallUNet sont deja quasi plats : on les prolonge
# quand meme, pour que la colonne "Iterations" reste comparable entre les 3 lignes.
# Pas de reprise possible (train_mnist ne recharge pas de checkpoint) : on repart de zero.
set -u
cd ~/UNN_for_FM
PY=~/.venvs/unn/bin/python
LOG=~/UNN_for_FM/claude.log
OUT=results/mnist_report_300ep
COMMON="--epochs 300 --digit -1 --coupling ot --lr 1e-3 --x1_weight invsq --batch-size 128 --save-model --results-dir $OUT"
{
  echo "===== [mnist 300ep] start $(date '+%F %T') GPU=1 ====="
  T0=$(date +%s)
  CUDA_VISIBLE_DEVICES=1 $PY run_mnist.py --only ConvScCP_UNN_L1 --K 20 $COMMON
  echo "--- ScCP (LFO+LNO) fini en $(( $(date +%s) - T0 )) s ---"
  T1=$(date +%s)
  # summary.txt est ECRASE par chaque invocation de run_mnist.py : on le met de cote
  cp "$OUT/summary.txt" "$OUT/summary_sccp.txt" 2>/dev/null
  CUDA_VISIBLE_DEVICES=1 $PY run_mnist.py --only SmallUNet_baseline $COMMON
  cp "$OUT/summary.txt" "$OUT/summary_smallunet.txt" 2>/dev/null
  echo "--- SmallUNet fini en $(( $(date +%s) - T1 )) s ---"
  echo "===== [mnist 300ep] fini $(date '+%F %T') — total $(( $(date +%s) - T0 )) s ====="
} >> "$LOG" 2>&1 &
echo "PID $!"
