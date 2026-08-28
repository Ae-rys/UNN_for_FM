#!/usr/bin/env bash
# run_unet_baselines_afhq.sh
# Entraine les deux baselines UNet en Flow Matching complet sur AFHQ-32, au MEME
# budget que le run ScCP k9/K10/ic128 (200 000 steps) et sous le meme protocole
# (couplage OT, recette torchcfm), pour obtenir enfin des comparatifs GENERATIFS
# et pas seulement de debruitage.
#
#   UNet_torchcfm_ch32   1.11M params — la baseline a capacite comparable aux ScCP
#   MinimalUNetFM_kamb   2.11M params — x-pred, meme objectif que le ScCP, sans attention
#
# ORDONNANCEMENT. Le script attend d'abord la fin d'un run deja en cours (PID
# passe en 1er argument), puis enchaine les deux entrainements SEQUENTIELLEMENT.
# Lancer quatre jobs concurrents sur un seul GPU divise le debit de chacun sans
# rien terminer plus tot ; en serie, chaque run garde sa vitesse et les runs
# deja lances ne sont pas ralentis.
#
# Usage
#   ./run_unet_baselines_afhq.sh [PID_A_ATTENDRE] [STEPS]
#   ./run_unet_baselines_afhq.sh              # demarre tout de suite, 200k steps
#   ./run_unet_baselines_afhq.sh 2427758      # attend ce PID, puis enchaine
#
# Sorties -> results_afhq32/UNet_torchcfm_ch32/ et results_afhq32/MinimalUNetFM_kamb/
# Journal  -> claude.log (suivable en tail -f)

set -u
cd "$(dirname "$0")" || exit 1
source ~/.venvs/unn/bin/activate

WAIT_PID="${1:-}"
STEPS="${2:-200000}"
LOG=claude.log

if [ -n "$WAIT_PID" ]; then
  echo "[queue] attente de la fin du PID $WAIT_PID avant de demarrer..." | tee -a "$LOG"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[queue] PID $WAIT_PID termine, demarrage." | tee -a "$LOG"
fi

for MODEL in UNet_torchcfm_ch32 MinimalUNetFM_kamb; do
  echo "[queue] === $MODEL, $STEPS steps ===" | tee -a "$LOG"
  python run_afhq32.py --only "$MODEL" --steps "$STEPS" \
      --sample-every 10000 --save-every 10000 --keep-every 10000 >> "$LOG" 2>&1
  echo "[queue] $MODEL termine (code $?)" | tee -a "$LOG"
done

echo "[queue] tout est fini." | tee -a "$LOG"
