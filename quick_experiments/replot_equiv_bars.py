"""Barres corrigées : la distance-au-train SEULE est trompeuse sous équivariance
(un mémoriseur translaté paraît 'nouveau'). On montre dist->train ET dist->dict_augmenté.
NOUVEAU vrai <=> loin des DEUX. Lit equivariance_control_metrics.txt."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np

names, dtr, daug, base = [], [], [], None
for ln in open("equivariance_control_metrics.txt"):
    if ln.startswith("#"):
        base = float(ln.split("baseline test->train=")[1].split(" ")[0]); continue
    if ln.startswith("arm"): continue
    v = ln.rstrip("\n").split("\t")
    names.append(v[0]); dtr.append(float(v[1])); daug.append(float(v[3]))

x = np.arange(len(names)); w = 0.38
fig, ax = plt.subplots(figsize=(12, 5.2))
ax.bar(x - w/2, np.array(dtr)/base, w, label="dist→TRAIN de base / baseline", color="C0")
ax.bar(x + w/2, np.array(daug)/base, w, label="dist→dict AUGMENTÉ (translations) / baseline", color="C1")
ax.axhline(1.0, color="k", ls=":", label="baseline (=1)")
for i in range(len(names)):
    ax.annotate(f"{dtr[i]/base:.2f}", (i-w/2, dtr[i]/base), ha="center", va="bottom", fontsize=8)
    ax.annotate(f"{daug[i]/base:.2f}", (i+w/2, daug[i]/base), ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("distance / baseline");
ax.set_title("NOUVEAU (créatif) ⟺ loin des DEUX dictionnaires.\n"
             "Arm 5 : loin du train (paraît nouveau) MAIS collé au dict augmenté ⇒ mémorise des translations")
ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig("equivariance_control_bars.png", dpi=120)
print("saved equivariance_control_bars.png")
for n, a, b in zip(names, dtr, daug):
    verdict = "NOUVEAU" if (a > 0.9*base and b > 0.9*base) else "mémorise"
    print(f"{n:28s} train={a:6.2f} aug={b:6.2f} -> {verdict}")
