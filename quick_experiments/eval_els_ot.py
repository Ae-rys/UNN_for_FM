"""Éval ELS du checkpoint OT — construit le modèle EXACTEMENT comme à l'entraînement
(via build_experiments de run_mnist), pour éviter le mismatch d'architecture de
nifty.load_model (hardcodé, désync depuis que L1ProxConv w a changé). Réutilise
toute la machine ELS/IS de nifty_els_fm."""
import sys, torch
import models.architectures as A
# Le checkpoint OT a été entraîné avec L1ProxConv w=8 ; architectures.py a depuis
# changé le défaut à 32. On remet le défaut à 8 pour reconstruire l'archi w=8 et
# charger le checkpoint tel quel (AUCUN réentraînement, juste éval).
A.L1ProxConv.__init__.__defaults__ = (8, None)             # (w=8, channels=None)
import nifty_els_fm as N
from run_mnist import build_experiments

CKPT = "results/ot_els_test/ConvScCP_k3_K6_ic128_L1_LNO/model.pt"
NAME = "ConvScCP_k3_K6_ic128_L1_LNO"

entry = [e for e in build_experiments(N.dev) if e["name"] == NAME][0]

def load_model_fixed(ckpt, K, ic, kernel):
    m = entry["build"]()                                   # MÊME archi que l'entraînement
    sd = torch.load(ckpt, map_location=N.dev, weights_only=True)
    m.load_state_dict(sd); m.eval()
    print(f"[load fixed] {NAME} reconstruit via build_experiments (match checkpoint)")
    return m

N.load_model = load_model_fixed                            # monkeypatch
sys.argv = ["nifty_els_fm.py", "--ckpt", CKPT, "--K", "6", "--ic", "128",
            "--kernel", "3", "--tag", "sccp_k3_ot"]
N.main()
