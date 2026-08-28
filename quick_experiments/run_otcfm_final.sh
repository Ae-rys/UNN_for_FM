#!/bin/bash
# Les 3 runs de reference "propres" pour la comparaison FID sur MNIST :
#   1. SmallUNet_baseline        (baseline UNet)
#   2. ConvScCP_UNN_L1_LFO       (K=20, ic=64)
#   3. ConvScCP_UNN_L1_LNO       (K=20, ic=64)
# Tous en couplage OT, sur TOUS les chiffres, meme nombre d'epochs, meme lr,
# meme dossier -> les mini-FID seront enfin comparables entre eux.
#
# Les 3 tournent en SEQUENTIEL sur un seul GPU. Duree estimee a 100 epochs :
#   SmallUNet ~1.3 h, chaque ScCP ~1.8 h  =>  ~5 h au total.
#   (extrapole de results/OT-CFM (50 ep) et results/temp-4 (100 ep))
#
# --digit -1 est INDISPENSABLE : le defaut de run_mnist.py est --digit 0
# (zeros uniquement), ce qui ne correspondrait pas aux runs de reference.
# --save-model aussi : sans lui, pas de model.pt, donc pas de FID possible.
#
# Suivi :  tail -f ~/UNN_for_FM/claude.log
#
# Usage :  ./run_otcfm_final.sh [GPU] [EPOCHS]
#          ./run_otcfm_final.sh 0 100
set -u
cd ~/UNN_for_FM
GPU=${1:-0}
EPOCHS=${2:-100}
PY=~/.venvs/unn/bin/python
LOG=~/UNN_for_FM/claude.log
OUT=results/OT-CFM-final

mkdir -p "$OUT"

run_one () {
    local name="$1"
    echo "===== [OT-CFM-final] $name — start $(date '+%F %T') =====" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" run_mnist.py \
        --only "$name" \
        --results-dir "$OUT" \
        --epochs "$EPOCHS" \
        --digit -1 \
        --coupling ot \
        --save-model \
        >> "$LOG" 2>&1
    # run_mnist.py reecrit summary.txt a chaque appel : on le met de cote.
    [ -f "$OUT/summary.txt" ] && mv "$OUT/summary.txt" "$OUT/summary_$name.txt"
    echo "===== [OT-CFM-final] $name — done  $(date '+%F %T') =====" >> "$LOG"
}

{
    run_one SmallUNet_baseline
    run_one ConvScCP_UNN_L1_LFO
    run_one ConvScCP_UNN_L1_LNO

    # recompose un summary global a partir des trois
    head -1 "$OUT"/summary_SmallUNet_baseline.txt > "$OUT/summary.txt"
    for f in "$OUT"/summary_*.txt; do tail -n +2 "$f" >> "$OUT/summary.txt"; done

    echo "===== [OT-CFM-final] TOUT FINI $(date '+%F %T') =====" >> "$LOG"
    echo "FID :  for d in $OUT/*/; do CUDA_VISIBLE_DEVICES=$GPU $PY fid_ckpt_mnist.py --run-dir \"\$d\" --n 10000; done" >> "$LOG"
} &

echo "PID $! — 3 runs en sequentiel sur le GPU $GPU, $EPOCHS epochs, ~5 h"
echo "suivi : tail -f $LOG"
