#!/bin/bash
# ScCP LFO, K=15, 400 epochs — controle de convergence : la loss bouge-t-elle encore
# apres 200 epochs ? (100 ep -> 0.1380 ; 200 ep -> 0.1345)
#
# S'enchaine APRES la file run_mnist_K15_200ep_all.sh : on attend la disparition de
# son wrapper (PID passe en argument), ce qui couvre aussi le cas ou elle echoue en
# cours de route — dans les deux cas le GPU est libre a la sortie de la boucle.
#
# Usage :  ./run_mnist_K15_400ep_lfo.sh <PID_de_la_chaine>
set -u
cd ~/UNN_for_FM
WAIT_PID=${1:-0}
PY=~/.venvs/unn/bin/python
LOG=~/UNN_for_FM/claude.log
OUT=results/mnist_report_400ep_K15

{
  echo "===== [K15 400ep LFO] en attente de la file (PID $WAIT_PID) $(date '+%F %T') ====="
  while [ "$WAIT_PID" != "0" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
  # ceinture et bretelles : plus aucun entrainement ne doit tourner
  while pgrep -f "run_mnist.py --only" > /dev/null; do sleep 60; done
  echo "--- file terminee, demarrage du 400 epochs $(date '+%H:%M') ---"

  T=$(date +%s)
  CUDA_VISIBLE_DEVICES=1 $PY run_mnist.py --only ConvScCP_UNN_L1_LFO \
      --K 15 --ic 64 --epochs 400 --digit -1 --coupling ot --lr 1e-3 \
      --x1_weight invsq --batch-size 128 --save-model --results-dir "$OUT" \
    || { echo "!!! [400ep] ECHEC"; exit 1; }
  echo "--- entrainement fini en $(( $(date +%s) - T )) s ---"

  CUDA_VISIBLE_DEVICES=1 $PY trajectory_convsccp.py \
      --ckpt "$OUT/ConvScCP_UNN_L1_LFO/model.pt" --solver euler --steps 10 --n 4 \
      --n-iter-samples 2 --iter-norm per-image || echo "(trajectoire en echec)"
  CUDA_VISIBLE_DEVICES=1 $PY make_mnist_report_figs.py --run "$OUT" --suffix _K15_400ep

  echo "===== [K15 400ep LFO] TOUT EST FINI $(date '+%F %T') ====="
} >> "$LOG" 2>&1 &
echo "PID $!"
