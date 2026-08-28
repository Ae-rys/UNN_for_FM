# -*- coding: utf-8 -*-
"""repare_metrics_temps.py — restaure it_s / train_time_s dans le metrics.txt
d'une config DEJA AU BUDGET.

Quand denoise_probe.py reprend une config dont le ckpt est deja a --steps, la
boucle d'entrainement est vide : dt ~ 0, donc il reecrit train_time_s=0.0 et un
it_s absurde (1e10). Les vraies valeurs sont recuperables : le champ `elapsed`
du checkpoint contient le temps d'entrainement cumule reel.

Usage : python repare_metrics_temps.py results_denoise_afhq_v4/<config>
"""
import re, sys, os, torch

run_dir = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
d = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location="cpu", weights_only=False)
dt = float(d["elapsed"])
p = os.path.join(run_dir, "metrics.txt")
s = open(p).read()
s = re.sub(r"it_s=[\d.e+]+", f"it_s={steps/dt:.3f}", s)
s = re.sub(r"train_time_s=[\d.e+]+", f"train_time_s={dt:.1f}", s)
open(p, "w").write(s)
print(f"{p} : it_s={steps/dt:.3f}  train_time_s={dt:.1f} ({dt/60:.1f} min)")
