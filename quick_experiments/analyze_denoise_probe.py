# -*- coding: utf-8 -*-
"""
analyze_denoise_probe.py
Depouille TOUS les runs de results_denoise_probe/ et en extrait des regles.

Le piege que ce script existe pour eviter
-----------------------------------------
Le dossier melange des runs qui n'ont PAS ete entraines sous le meme protocole :

  famille A  couplage indep, loss x1 (MSE brute), t = [0.2, 0.4, 0.6, 0.8]
  famille B  couplage ot,    loss v  (ponderee),  t = [0.8]  <- un SEUL niveau

Un classement global sur `nmse_mean` melangerait architecture et protocole, et
condamnerait la famille B pour une raison qui n'a rien a voir avec son archi :
elle n'a jamais vu t < 0.8, donc elle y est evidemment mauvaise. On regroupe
donc par (coupling, loss, train_t) et on ne compare QUE dans une famille — sauf
sur nmse(t=0.8), le seul niveau que les deux ont vu a l'entrainement, et encore
en sachant que B y a concentre 100 % de sa capacite contre 25 % pour A.

Ce qu'on en tire
----------------
  * les exposants d'echelle de la famille A (plan factoriel complet
    k x K x ic = 2 x 3 x 2) par moindres carres sur log(nmse) ;
  * la frontiere de Pareto qualite / parametres / temps GPU ;
  * la loi en profondeur sur nmse(t=0.8), le seul point commun aux deux familles.

Entrees : results_denoise_probe/*/metrics.txt (metadonnees + protocole) et
curve_all.csv (courbes denses, produit par denoise_curve.py — le relancer si de
nouveaux runs sont apparus).

Usage
-----
    source ~/.venvs/unn/bin/activate
    python denoise_curve.py            # (re)genere curve_all.csv si besoin
    python analyze_denoise_probe.py

Sorties -> results_denoise_probe/analyse_*.png + analyse_regles.txt
"""

import argparse
import csv
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wiener_floor(cache, n_val, ts, seed=0):
    """Le MEILLEUR debruiteur LINEAIRE — la reference qui manque pour interpreter
    les nmse : elle dit ce qu'on obtient en n'exploitant QUE les correlations du
    second ordre des images. Un reseau qui ne la bat pas n'a rien appris de la
    structure non gaussienne.

    x_t = (1-t) x0 + t x1, x0 ~ N(0,I). Filtre optimal (covariance C estimee sur
    le MEME split train que denoise_probe.py) :

        x1_hat = mu + t C (t^2 C + (1-t)^2 I)^-1 (x_t - t mu)

    On renvoie DEUX chiffres, et l'ecart entre les deux est instructif :

      empirique   MSE mesuree sur les 512 images de validation, avec le bruit
                  FIXE de denoise_probe.py -> directement comparable aux reseaux.
                  C'est celui a citer.
      analytique  MSE predite par la formule sum_i lam_i (1-t)^2/(t^2 lam_i +
                  (1-t)^2). Avec n=5141 echantillons en dimension 3072, les
                  valeurs propres empiriques sont biaisees : cette formule est
                  OPTIMISTE (elle suppose C exacte et les donnees gaussiennes).
                  A ne pas presenter comme un plancher.
    """
    import torch
    d = torch.load(cache)
    x = d["data"].float().div_(127.5).sub_(1.0)
    g = torch.Generator().manual_seed(seed)
    x = x[torch.randperm(x.shape[0], generator=g)]
    flat = x.reshape(x.shape[0], -1).double()
    xva, xtr = flat[:n_val], flat[n_val:]
    mu = xtr.mean(dim=0, keepdim=True)
    xc = xtr - mu
    n, dim = xc.shape
    C = xc.T @ xc / n
    lam, U = torch.linalg.eigh(C)
    lam = lam.clamp_min(0)

    # meme bruit de validation que make_val_set (meme generateur, meme ordre de t)
    gv = torch.Generator().manual_seed(seed + 1234)
    # MEME denominateur que denoise_probe.reference_mse : variance de la validation
    # autour de SA propre moyenne. Normaliser par la moyenne du train gonflerait le
    # denominateur et flatterait artificiellement la reference lineaire.
    var = float(((xva - xva.mean(dim=0, keepdim=True)) ** 2).mean())
    emp, ana = {}, {}
    for t in sorted(ts):
        x0 = torch.randn(xva.shape[0], dim, generator=gv).double()
        xt = (1 - t) * x0 + t * xva
        gain = t * lam / (t ** 2 * lam + (1 - t) ** 2)          # dans la base propre
        pred = mu + ((xt - t * mu) @ U * gain) @ U.T
        emp[t] = float(((pred - xva) ** 2).mean())
        ana[t] = float((lam * (1 - t) ** 2 / (t ** 2 * lam + (1 - t) ** 2)).mean())
    return emp, ana, var


def read_metrics(path):
    meta, curve = {}, {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if "=" in line and "\t" not in line:
                k, v = line.split("=", 1)
                meta[k] = v
            elif "\t" in line and not line.startswith("t\t"):
                parts = line.split("\t")
                m = re.match(r"t?=?([0-9.]+)$", parts[0])
                if m:
                    curve[float(m.group(1))] = float(parts[1].split("=")[-1])
    return meta, curve


def load_runs(results_dir):
    runs = []
    for name in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, name, "metrics.txt")
        if not os.path.exists(path):
            continue
        m = re.fullmatch(r"ScCP_k(\d+)_K(\d+)_ic(\d+)_(\w+)_(\w+)", name)
        is_ref = name.startswith("unet")
        if not m and not is_ref:
            continue
        meta, _ = read_metrics(path)
        # les runs ecrits AVANT l'ajout des cles loss/coupling sont, par
        # construction, ceux de la famille A (indep + x1, 4 niveaux de bruit)
        train_t = meta.get("train_t", "[0.2, 0.4, 0.6, 0.8]")
        runs.append(dict(
            name=name, is_ref=is_ref,
            k=0 if is_ref else int(m.group(1)),
            K=0 if is_ref else int(m.group(2)),
            ic=0 if is_ref else int(m.group(3)),
            n_params=int(meta["n_params"]), it_s=float(meta["it_s"]),
            time_s=float(meta["train_time_s"]), selected=meta.get("selected", "?"),
            loss=meta.get("loss", "x1"), coupling=meta.get("coupling", "indep"),
            train_t=[float(s) for s in train_t.strip("[]").split(",")],
        ))
    return runs


def load_curves(results_dir, fname="curve_all.csv"):
    path = os.path.join(results_dir, fname)
    if not os.path.exists(path):
        return {}, []
    curves, ts = {}, []
    with open(path) as f:
        for row in csv.DictReader(f):
            ts = ts or sorted(float(c[6:]) for c in row if c.startswith("nmse_t"))
            curves[row["name"]] = {t: float(row[f"nmse_t{t:g}"]) for t in ts}
    return curves, ts


def family_of(r):
    return f"coupling={r['coupling']}, loss={r['loss']}, t={r['train_t']}"


def scaling_exponents(runs, curves, t_ref):
    """Moindres carres sur log(nmse) = c + a log(K) + b log(ic) + d log(k).

    Les exposants se lisent : "doubler K multiplie la nmse par 2^a". Sur un plan
    factoriel complet (2 x 3 x 2 = 12 points) c'est identifiable sans ambiguite."""
    rows = [r for r in runs if r["name"] in curves]
    if len(rows) < 6:
        return None
    y = np.log([np.mean([curves[r["name"]][t] for t in t_ref]) for r in rows])
    X = np.column_stack([np.ones(len(rows))] + [
        np.log([r[key] for r in rows]) for key in ("K", "ic", "k")])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return dict(zip(("const", "K", "ic", "k"), coef)), (1 - ss_res / ss_tot if ss_tot else 0.0)


def main():
    p = argparse.ArgumentParser(description="Regles extraites des runs denoise_probe.")
    p.add_argument("--results-dir", type=str, default="results_denoise_probe")
    p.add_argument("--cache", type=str, default="./data/afhq_cat32_train.pt")
    p.add_argument("--n-val", type=int, default=512)
    args = p.parse_args()
    rd = args.results_dir

    runs = load_runs(rd)
    curves, ts = load_curves(rd)
    if not runs:
        print(f"Aucun run exploitable dans {rd}/", flush=True)
        return

    fams = {}
    for r in runs:
        fams.setdefault(family_of(r), []).append(r)
    famA = max(fams.values(), key=len)                   # la plus peuplee = plan factoriel
    A = [r for r in famA if not r["is_ref"]]             # les ScCP du plan
    refs = [r for r in famA if r["is_ref"]]              # les baselines (plafond)
    B = [r for r in runs if r not in famA]
    t_ref = sorted(A[0]["train_t"])
    wiener, wiener_ana, var_data = wiener_floor(
        args.cache, args.n_val,
        sorted({t for r in runs for t in r["train_t"]} | set(ts or t_ref)))

    out = []
    def say(s=""):
        print(s, flush=True); out.append(s)

    say("=" * 78)
    say(f"{len(runs)} runs exploitables, {len(fams)} protocole(s) distinct(s)")
    say("=" * 78)
    for fam, rs in sorted(fams.items(), key=lambda kv: -len(kv[1])):
        say(f"  [{len(rs):>2} runs] {fam}")
        say(f"           {', '.join(sorted(r['name'].replace('ScCP_','').replace('_l1_LFO','') for r in rs))}")
    say()
    say("Comparaisons valides UNIQUEMENT a l'interieur d'une famille : le protocole")
    say("d'entrainement (couplage, ponderation de la loss, niveaux de bruit vus)")
    say("change ce qui est optimise, pas seulement l'architecture.")
    say()

    # ---------------- famille A : plan factoriel ----------------
    say("=" * 78)
    say(f"FAMILLE A ({len(A)} runs) — {family_of(A[0])}")
    say("=" * 78)
    for r in A:
        r["nmse"] = np.mean([curves[r["name"]][t] for t in t_ref]) if r["name"] in curves else np.nan
    A.sort(key=lambda r: r["nmse"])
    say(f"{'config':<28}{'params':>11}{'nmse':>8}{'min GPU':>9}   nmse par t")
    for r in A:
        c = curves.get(r["name"], {})
        say(f"{r['name'].replace('ScCP_','').replace('_l1_LFO',''):<28}"
            f"{r['n_params']:>11,}{r['nmse']:>8.4f}{r['time_s']/60:>9.1f}   " +
            " ".join(f"{t:g}:{c.get(t, float('nan')):.3f}" for t in t_ref))
    for r in refs:
        c = curves.get(r["name"], {})
        r["nmse"] = np.mean([c[t] for t in t_ref]) if c else np.nan
        say(f"{'[PLAFOND] ' + r['name']:<28}{r['n_params']:>11,}{r['nmse']:>8.4f}"
            f"{r['time_s']/60:>9.1f}   " +
            " ".join(f"{t:g}:{c.get(t, float('nan')):.3f}" for t in t_ref))
    say(f"{'[REF] lineaire opt. (mesure)':<28}{'':>11}"
        f"{np.mean([wiener[t]/var_data for t in t_ref]):>8.4f}{'':>9}   " +
        " ".join(f"{t:g}:{wiener[t]/var_data:.3f}" for t in t_ref))
    say(f"{'[REF] lineaire, formule':<28}{'':>11}"
        f"{np.mean([wiener_ana[t]/var_data for t in t_ref]):>8.4f}{'':>9}   " +
        " ".join(f"{t:g}:{wiener_ana[t]/var_data:.3f}" for t in t_ref) +
        "   <- optimiste (biais de plug-in)")
    best, worst = A[0], A[-1]
    cheap = min(A, key=lambda r: r["time_s"])
    say()
    say(f"  meilleure : {best['name'].replace('ScCP_','').replace('_l1_LFO','')} "
        f"nmse {best['nmse']:.4f} ({best['n_params']/1e6:.2f}M, {best['time_s']/60:.1f} min)")
    say(f"  pire      : {worst['name'].replace('ScCP_','').replace('_l1_LFO','')} "
        f"nmse {worst['nmse']:.4f} ({worst['n_params']/1e6:.2f}M, {worst['time_s']/60:.1f} min)")
    say(f"  -> toute la famille tient dans {100*(worst['nmse']/best['nmse']-1):.1f} % de nmse, "
        f"alors que les parametres varient de {min(r['n_params'] for r in A)/1e6:.2f}M a "
        f"{max(r['n_params'] for r in A)/1e6:.2f}M ({max(r['n_params'] for r in A)/min(r['n_params'] for r in A):.0f}x) "
        f"et le temps GPU de {cheap['time_s']/60:.1f} a "
        f"{max(r['time_s'] for r in A)/60:.1f} min ({max(r['time_s'] for r in A)/cheap['time_s']:.0f}x).")

    exps = scaling_exponents(A, curves, t_ref)
    if exps:
        coef, r2 = exps
        say()
        say(f"  Exposants d'echelle  log(nmse) = c + a.log(K) + b.log(ic) + d.log(k)   (R2={r2:.3f})")
        for key, label in (("K", "profondeur K "), ("ic", "dual ic     "),
                           ("k", "noyau k     ")):
            say(f"    {label} exposant {coef[key]:+.4f}  ->  doubler = "
                f"x{2**coef[key]:.3f} sur la nmse ({100*(2**coef[key]-1):+.1f} %)")

    # ---------------- famille B ----------------
    if B:
        say()
        say("=" * 78)
        say(f"FAMILLE B ({len(B)} runs) — {family_of(B[0])}")
        say("=" * 78)
        tb = sorted(B[0]["train_t"])
        for r in sorted(B, key=lambda r: np.mean([curves[r["name"]][t] for t in tb])
                        if r["name"] in curves else 9):
            c = curves.get(r["name"], {})
            say(f"{r['name'].replace('ScCP_','').replace('_l1_LFO',''):<28}"
                f"{r['n_params']:>11,}{np.mean([c[t] for t in tb]):>8.4f}"
                f"{r['time_s']/60:>9.1f}   (entraine seulement a t={tb})")

    # ---------------- le seul pont entre les familles ----------------
    say()
    say("=" * 78)
    say("PONT ENTRE FAMILLES — nmse a t=0.8, le seul niveau vu par les deux")
    say("=" * 78)
    say("(la famille B y consacre 100 % de sa capacite, la famille A 25 % : tout")
    say(" avantage de B en dessous de ce facteur est une NON-amelioration)")
    bridge = [(r, curves[r["name"]][0.8]) for r in runs
              if r["name"] in curves and 0.8 in curves[r["name"]]]
    for r, v in sorted(bridge, key=lambda x: x[1]):
        fam = "REF" if r["is_ref"] else ("A" if r in A else "B")
        say(f"  [{fam:>3}] {r['name'].replace('ScCP_','').replace('_l1_LFO',''):<28}"
            f"K={r['K']:<3} ic={r['ic']:<4} nmse(0.8)={v:.4f}")

    # ---------------- figures ----------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    for r in A:
        c = curves.get(r["name"], {})
        if c:
            ax.plot(ts, [c[t] for t in ts], "-", lw=1.1, alpha=0.7, color="#1f77b4")
    for r in B:
        c = curves.get(r["name"], {})
        if c:
            ax.plot(ts, [c[t] for t in ts], "--", lw=1.1, alpha=0.7, color="#2ca02c")
    for r in refs:
        c = curves.get(r["name"], {})
        if c:
            ax.plot(ts, [c[t] for t in ts], "-", lw=2.4, color="k", label=r["name"])
    ax.plot(ts, [wiener[t] / var_data for t in ts], ":", lw=2.2, color="#d62728",
            label="optimum LINEAIRE (Wiener)")
    for t in t_ref:
        ax.axvline(t, color="gray", lw=0.6, ls=":")
    ax.axhline(1.0, color="gray", lw=1, ls="-.")
    ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_ylabel("nmse")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(f"a) Courbes nmse(t) — bleu = famille A ({len(A)} ScCP),\n"
                 f"vert tirets = famille B ({len(B)}, t=0.8 seul), noir = plafond UNet",
                 fontsize=9)

    ax = axes[0, 1]
    for k in sorted({r["k"] for r in A}):
        for ic in sorted({r["ic"] for r in A}):
            sel = sorted([r for r in A if r["k"] == k and r["ic"] == ic],
                         key=lambda r: r["K"])
            if len(sel) > 1:
                ax.plot([r["K"] for r in sel], [r["nmse"] for r in sel], "o-",
                        label=f"k={k}, ic={ic}")
    ax.set_xlabel("profondeur K"); ax.set_ylabel("nmse moyenne")
    ax.set_title("b) Effet de la profondeur (famille A)\nnoter l'echelle verticale",
                 fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    for r in A:
        ax.scatter(r["n_params"] / 1e6, r["nmse"], s=42,
                   c="#1f77b4" if r["k"] == 9 else "#ff7f0e")
        ax.annotate(f"K{r['K']}", (r["n_params"] / 1e6, r["nmse"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    for r in refs:
        ax.scatter(r["n_params"] / 1e6, r["nmse"], s=90, marker="*", c="k",
                   zorder=5, label=r["name"])
    ax.axhline(np.mean([wiener[t] / var_data for t in t_ref]), color="#d62728",
               ls=":", lw=2, label="optimum lineaire")
    ax.set_xscale("log"); ax.set_xlabel("parametres (M, echelle log)")
    ax.set_ylabel("nmse moyenne"); ax.legend(fontsize=7)
    ax.set_title("c) Qualite vs capacite (bleu k=9, orange k=15)\n"
                 "nuage plat = la capacite n'est pas le facteur limitant",
                 fontsize=9)

    ax = axes[1, 1]
    for r in A:
        ax.scatter(r["time_s"] / 60, r["nmse"], s=42,
                   c="#1f77b4" if r["k"] == 9 else "#ff7f0e")
        ax.annotate(f"K{r['K']}ic{r['ic']}", (r["time_s"] / 60, r["nmse"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    for r in refs:
        ax.scatter(r["time_s"] / 60, r["nmse"], s=90, marker="*", c="k", zorder=5,
                   label=r["name"])
    ax.set_xscale("log")
    ax.set_xlabel("temps GPU (min, 3000 steps, echelle log)")
    ax.set_ylabel("nmse moyenne"); ax.legend(fontsize=7)
    ax.set_title("d) Qualite vs cout GPU\nle bon coin est en bas a gauche", fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(rd, "analyse_denoise_probe.png"), dpi=110)
    plt.close(fig)

    with open(os.path.join(rd, "analyse_regles.txt"), "w") as f:
        f.write("\n".join(out) + "\n")
    say()
    say(f"-> {rd}/analyse_denoise_probe.png")
    say(f"-> {rd}/analyse_regles.txt")


if __name__ == "__main__":
    main()
