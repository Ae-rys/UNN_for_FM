"""Replot propre de locality_expansivity (lit le .txt, écarte le régime t>0.55 confondu
par l'artefact différences-finies à bas bruit). Sortie : locality_expansivity_clean.png"""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

Ps, els, glob = [], {}, []
tA, blockA = [], False
tB, k3, k9 = [], [], []
mode = None
for ln in open("locality_expansivity_metrics.txt"):
    ln = ln.rstrip("\n")
    if ln.startswith("# A. L(t)"): mode = "A"; continue
    if ln.startswith("# B."): mode = "B"; continue
    if mode == "A" and ln.startswith("t\t"):
        Ps = [int(x[1:]) for x in ln.split("\t")[1:-1]]; els = {P: [] for P in Ps}; continue
    if mode == "B" and ln.startswith("t\t"): continue
    if not ln or ln.startswith("#"): continue
    v = ln.split("\t")
    if mode == "A" and len(v) == len(Ps) + 2:
        tA.append(float(v[0]))
        for i, P in enumerate(Ps): els[P].append(float(v[1 + i]))
        glob.append(float(v[-1]))
    elif mode == "B" and len(v) == 3:
        tB.append(float(v[0])); k3.append(float(v[1])); k9.append(float(v[2]))

tA = np.array(tA); tB = np.array(tB)
keep = tA <= 0.55                                    # plage fiable (hors artefact FD bas bruit)
tR = 0.275                                           # t représentatif (bande expansive)
iR = int(np.argmin(np.abs(tA - tR)))

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
cmap = plt.cm.viridis(np.linspace(0, .9, len(Ps)))
for c, P in zip(cmap, Ps):
    ax[0].plot(tA[keep], np.array(els[P])[keep], "-o", color=c, ms=4, label=f"ELS P={P}")
ax[0].plot(tA[keep], np.array(glob)[keep], "--", color="C3", lw=2, label="GLOBAL (mémo)")
ax[0].axhline(1, color="k", ls=":", lw=1.2)
ax[0].set_yscale("log"); ax[0].set_xlabel("t"); ax[0].set_ylabel("Lipschitz local du débruiteur")
ax[0].set_title("A. Cible ELS : L(t) s'ordonne par taille de patch P\n(t≤0.55, plage fiable)")
ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=.3)

# L vs P au t représentatif -> le bouton de localité
LP = [els[P][iR] for P in Ps]
ax[1].plot(Ps, LP, "-o", color="C0", ms=7, label=f"ELS patch P (t={tA[iR]:.2f})")
ax[1].axhline(glob[iR], color="C3", ls="--", lw=2, label=f"GLOBAL mémo ({glob[iR]:.1f})")
ax[1].axhline(1, color="k", ls=":", lw=1.2, label="seuil non-expansif")
ax[1].set_yscale("log"); ax[1].set_xlabel("taille de patch P  (← plus local | plus global →)")
ax[1].set_ylabel(f"Lipschitz à t={tA[iR]:.2f}")
ax[1].set_title(f"A. Bouton de localité : expansivité croît\nmonotone P3→P27→global (×{glob[iR]/els[Ps[0]][iR]:.0f} à t={tA[iR]:.2f})")
ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both")

ax[2].plot(tB, k3, "-o", color="C0", ms=5, label="ScCP k=3 (RF local)")
ax[2].plot(tB, k9, "-s", color="C3", ms=5, label="ScCP k=9 (RF global)")
ax[2].axvspan(0.12, 0.42, color="gray", alpha=.12)
ax[2].axhline(1, color="k", ls=":", lw=1.2)
ax[2].set_xlabel("t"); ax[2].set_ylabel("Lipschitz local du débruiteur")
ax[2].set_title("B. ScCP ENTRAÎNÉ : RF global > RF local\n(même bande t que A, zone grisée)")
ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

plt.tight_layout(); plt.savefig("locality_expansivity_clean.png", dpi=120)
print("saved locality_expansivity_clean.png")
print(f"ELS L(t={tA[iR]:.2f}) vs P :", {P: round(els[P][iR], 2) for P in Ps}, "GLOBAL", round(glob[iR], 2))
print("ratio global/P3 =", round(glob[iR] / els[Ps[0]][iR], 1))
