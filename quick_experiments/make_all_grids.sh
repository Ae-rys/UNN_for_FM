#!/bin/bash
# Génère les 6 grilles comparison_mnist_<model>_<coupling>.png (50 ep) :
# {ScCP, ResNet(Kamb), UNet(Kamb=MinimalUNetFM)} × {indep, OT}.
set -u
cd ~/UNN_for_FM
PY=~/.venvs/unn/bin/python
LOG=claude.log

g() {  # $1=name $2=ckpt $3=tag $4=title
    echo "===== [grid] $3 =====" | tee -a "$LOG"
    $PY make_comparison_grid.py --name "$1" --ckpt "$2" --tag "$3" --title "$4" 2>&1 \
        | grep -E "best P|saved|ERROR|Traceback|mismatch" | tee -a "$LOG"
}

g ConvScCP_k3_K6_ic128_L1_LNO results/grid50/ConvScCP_k3_K6_ic128_L1_LNO/model.pt        sccp_indep   "ScCP-FM, indep"
g ConvScCP_k3_K6_ic128_L1_LNO results/grid50_ot/ConvScCP_k3_K6_ic128_L1_LNO/model.pt     sccp_ot      "ScCP-FM, OT"
g MinimalResNetFM_L6_ic256    results/temp-5/MinimalResNetFM_L6_ic256/model.pt           resnet_indep "ResNet-FM (Kamb), indep"
g MinimalResNetFM_L6_ic256    results/grid50_ot/MinimalResNetFM_L6_ic256/model.pt        resnet_ot    "ResNet-FM (Kamb), OT"
g MinimalUNetFM_kamb          results/grid50/MinimalUNetFM_kamb/model.pt                 unet_indep   "UNet (Kamb), indep"
g MinimalUNetFM_kamb          results/grid50_ot/MinimalUNetFM_kamb/model.pt             unet_ot      "UNet (Kamb), OT"
echo "===== [grid] ALL GRIDS DONE $(date '+%F %T') =====" | tee -a "$LOG"
