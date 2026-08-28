#!/bin/bash
# MNIST K=15, 200 epochs, les 5 modeles de comparaison + depouillement complet.
#
# Enchaine, sans surveillance :
#   0. attend la fin du ScCP LFO deja lance (meme dossier de sortie)
#   1. ScCP LNO, DFB LFO, DFB LNO, SmallUNet   (sequentiel : les temps
#      d'entrainement restent comparables entre modeles, ce qui ne serait
#      pas le cas en parallele sur un meme GPU)
#   2. trajectoires des 4 UNN (iterates_sample0.png)
#   3. mini-FID + grilles de generation + courbes de loss
#
# ic=64 pour les 4 UNN : DFB LNO etait code en dur a ic=32, il aurait tourne
# avec deux fois moins de canaux duaux que les autres.
set -u
cd ~/UNN_for_FM
PY=~/.venvs/unn/bin/python
LOG=~/UNN_for_FM/claude.log
OUT=results/mnist_report_200ep_K15
COMMON="--K 15 --ic 64 --epochs 200 --digit -1 --coupling ot --lr 1e-3 --x1_weight invsq --batch-size 128 --save-model --results-dir $OUT"

{
  echo "===== [K15 200ep, chaine complete] start $(date '+%F %T') ====="
  # --- 0. attendre le ScCP LFO en cours -------------------------------------
  while pgrep -f "run_mnist.py --only ConvScCP_UNN_L1_LFO --K 15 --epochs 200" > /dev/null; do
    sleep 60
  done
  echo "--- ScCP LFO termine, demarrage de la file $(date '+%H:%M') ---"

  # --- 1. les 4 modeles restants -------------------------------------------
  for M in ConvScCP_UNN_L1_LNO ConvDFB_UNN_L1_LFO ConvDFB_UNN_L1_LNO SmallUNet_baseline; do
    T=$(date +%s)
    echo "--- [$M] debut $(date '+%H:%M') ---"
    CUDA_VISIBLE_DEVICES=1 $PY run_mnist.py --only "$M" $COMMON \
      || echo "!!! [$M] ECHEC (on continue)"
    # run_mnist.py ECRASE summary.txt a chaque appel : on garde une copie par modele
    cp "$OUT/summary.txt" "$OUT/summary_$M.txt" 2>/dev/null
    echo "--- [$M] fini en $(( $(date +%s) - T )) s ---"
  done

  # --- 2. trajectoires des UNN ---------------------------------------------
  for M in ConvScCP_UNN_L1_LFO ConvScCP_UNN_L1_LNO ConvDFB_UNN_L1_LFO ConvDFB_UNN_L1_LNO; do
    [ -f "$OUT/$M/model.pt" ] || continue
    CUDA_VISIBLE_DEVICES=1 $PY trajectory_convsccp.py --ckpt "$OUT/$M/model.pt" \
      --solver euler --steps 10 --n 4 --n-iter-samples 2 --iter-norm per-image \
      || echo "!!! trajectoire $M en echec (non bloquant)"
  done

  # --- 3. FID + figures -----------------------------------------------------
  echo "--- depouillement (mini-FID, grilles, courbes) $(date '+%H:%M') ---"
  CUDA_VISIBLE_DEVICES=1 $PY make_mnist_report_figs.py --run "$OUT" --suffix _K15_200ep

  echo "===== [K15 200ep] TOUT EST FINI $(date '+%F %T') ====="
} >> "$LOG" 2>&1 &
echo "PID $!"
