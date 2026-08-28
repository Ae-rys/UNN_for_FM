#!/bin/bash
# Les 3 lignes MNIST de tab:quantitative : ScCP LNO, ScCP LFO, SmallUNet.
# 100 epochs sur MNIST COMPLET (469 batches/epoch -> 46 900 iterations, le chiffre
# du tableau), couplage OT, loss x-pred ponderee en espace vitesse (invsq), lr 1e-3
# (la valeur annoncee dans subsec:hyperparams). --save-model est indispensable :
# les figures de trajectoire se regenerent depuis les poids.
set -u
cd ~/UNN_for_FM
PY=~/.venvs/unn/bin/python
LOG=~/UNN_for_FM/claude.log
OUT=results/mnist_report_100ep
COMMON="--epochs 100 --digit -1 --coupling ot --lr 1e-3 --x1_weight invsq --batch-size 128 --save-model --results-dir $OUT"
{
  echo "===== [mnist rapport] start $(date '+%F %T') GPU=1 ====="
  T0=$(date +%s)
  CUDA_VISIBLE_DEVICES=1 $PY run_mnist.py --only ConvScCP_UNN_L1 $COMMON
  echo "--- ScCP fini en $(( $(date +%s) - T0 )) s ---"
  T1=$(date +%s)
  CUDA_VISIBLE_DEVICES=1 $PY run_mnist.py --only SmallUNet_baseline $COMMON
  echo "--- SmallUNet fini en $(( $(date +%s) - T1 )) s ---"
  echo "===== [mnist rapport] fini $(date '+%F %T') — total $(( $(date +%s) - T0 )) s ====="
} >> "$LOG" 2>&1 &
echo "PID $!"
