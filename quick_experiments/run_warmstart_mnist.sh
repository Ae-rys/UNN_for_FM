#!/bin/bash
# Lance la paire baseline / warmstart (warm-start du dual ConvScCP) en tache de fond.
#
# NOTE : tmux n'est PAS installe sur hades et /scratch n'est pas accessible en ecriture
# (root:root, 755). On utilise donc setsid+nohup (survit a la deconnexion, comme tmux
# pour ce besoin) et les checkpoints vont dans results/warmstart_mnist/.
# Suivi :  tail -f ~/UNN_for_FM/claude.log
#
# Usage :  ./run_warmstart_mnist.sh [GPU] [EPOCHS]
#          ./run_warmstart_mnist.sh 1 200
set -u
cd ~/UNN_for_FM
GPU=${1:-1}
EPOCHS=${2:-200}
PY=~/.venvs/unn/bin/python
LOG=~/UNN_for_FM/claude.log

echo "===== [warmstart] start $(date '+%F %T') GPU=$GPU epochs=$EPOCHS =====" >> "$LOG"
CUDA_VISIBLE_DEVICES=$GPU setsid nohup "$PY" run_warmstart_mnist.py \
    --runs both --epochs "$EPOCHS" --self-cond-rate 0.9 \
    --K 20 --ic 64 --kernel 9 --version LNO --digit 0 --coupling ot \
    --n-eval 2000 --outdir results/warmstart_mnist \
    >> "$LOG" 2>&1 &
echo "PID $! -> suivi : tail -f $LOG"
