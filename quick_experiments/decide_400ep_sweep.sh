#!/bin/bash
# Decision AUTOMATIQUE : faut-il refaire tous les modeles en 400 epochs ?
#
# Critere : sur le run ScCP LFO 400 epochs, on compare sa PROPRE loss a l'epoch 200
# et a l'epoch 400. Comparer a l'interieur d'un seul run (et non entre le run 200ep
# et le run 400ep) elimine la variance d'initialisation et de tirage.
#
#   gain relatif = (loss@200 - loss@400) / loss@200
#     >= SEUIL  -> la loss bouge encore : on relance les 4 autres modeles en 400 ep
#     <  SEUIL  -> converge : on ne lance rien, on ecrit la conclusion dans le log
#
# SEUIL = 1 % par defaut. Repere : le passage de 100 a 200 epochs avait donne 2.5 %
# (0.1380 -> 0.1345). Un gain deux fois moindre en doublant encore le budget veut dire
# que la queue est logarithmique et que 200 suffit.
#
# Usage :  ./decide_400ep_sweep.sh <PID_du_run_400ep> [SEUIL]
set -u
cd ~/UNN_for_FM
WAIT_PID=${1:-0}
SEUIL=${2:-0.01}
PY=~/.venvs/unn/bin/python
LOG=~/UNN_for_FM/claude.log
SRC=results/mnist_report_400ep_K15
OUT=results/mnist_report_400ep_K15

{
  echo "===== [decision 400ep] en attente du run LFO 400ep (PID $WAIT_PID) $(date '+%F %T') ====="
  while [ "$WAIT_PID" != "0" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 180; done
  while pgrep -f "run_mnist.py --only" > /dev/null; do sleep 60; done

  F="$SRC/ConvScCP_UNN_L1_LFO/loss.txt"
  if [ ! -f "$F" ]; then
    echo "!!! [decision] $F absent — le run 400ep a echoue. Rien n'est lance."
    exit 1
  fi

  VERDICT=$($PY - "$F" "$SEUIL" <<'PYEOF'
import sys
rows = {}
for line in open(sys.argv[1]):
    p = line.split()
    if len(p) == 2:
        try: rows[int(p[0])] = float(p[1])
        except ValueError: pass
seuil = float(sys.argv[2])
l200, l400 = rows.get(200), rows.get(400)
if l200 is None or l400 is None:
    print(f"ERREUR epochs 200/400 absentes de loss.txt (max={max(rows) if rows else '?'})")
    sys.exit(2)
gain = (l200 - l400) / l200
print(f"loss@200={l200:.4f}  loss@400={l400:.4f}  gain={gain*100:+.2f}%  seuil={seuil*100:.1f}%")
sys.exit(0 if gain >= seuil else 1)
PYEOF
)
  CODE=$?
  echo "--- [decision] $VERDICT ---"

  if [ $CODE -ne 0 ]; then
    echo "===== [decision] la loss a CONVERGE (ou erreur de lecture) : aucun run relance. ====="
    echo "      Les resultats de reference restent ceux a 200 epochs."
    exit 0
  fi

  echo "===== [decision] la loss BOUGE ENCORE : relance des 4 autres modeles en 400 epochs ====="
  echo "      (ScCP LFO est deja fait, il n'est pas refait) — ~13 h estimees"
  COMMON="--K 15 --ic 64 --epochs 400 --digit -1 --coupling ot --lr 1e-3 --x1_weight invsq --batch-size 128 --save-model --results-dir $OUT"
  for M in ConvScCP_UNN_L1_LNO ConvDFB_UNN_L1_LFO ConvDFB_UNN_L1_LNO SmallUNet_baseline; do
    T=$(date +%s)
    echo "--- [400ep/$M] debut $(date '+%H:%M') ---"
    CUDA_VISIBLE_DEVICES=1 $PY run_mnist.py --only "$M" $COMMON || echo "!!! [$M] ECHEC (on continue)"
    cp "$OUT/summary.txt" "$OUT/summary_$M.txt" 2>/dev/null
    echo "--- [400ep/$M] fini en $(( $(date +%s) - T )) s ---"
  done
  for M in ConvScCP_UNN_L1_LNO ConvDFB_UNN_L1_LFO ConvDFB_UNN_L1_LNO; do
    [ -f "$OUT/$M/model.pt" ] || continue
    CUDA_VISIBLE_DEVICES=1 $PY trajectory_convsccp.py --ckpt "$OUT/$M/model.pt" \
      --solver euler --steps 10 --n 4 --n-iter-samples 2 --iter-norm per-image \
      || echo "(trajectoire $M en echec)"
  done
  CUDA_VISIBLE_DEVICES=1 $PY make_mnist_report_figs.py --run "$OUT" --suffix _K15_400ep
  echo "===== [decision 400ep] BALAYAGE COMPLET TERMINE $(date '+%F %T') ====="
} >> "$LOG" 2>&1 &
echo "PID $!"
