# Synthèse — Expansivité du débruiteur, localité et créativité

*Série d'expériences du 2026-07-13 (ScCP / ConvScCP vs UNet, cadre Flow Matching sur MNIST).*
*Point de départ : deux hypothèses proposées (Fable + moi) sur ce qui distingue ScCP du UNet et
sur le mécanisme de créativité des modèles convolutifs de diffusion (Kamb & Ganguli 2412.20292,
NIFTY 2509.22318). Objectif : les tester proprement, sans surinterpréter.*

---

## Résumé en une page

On a suivi trois hypothèses successives ; **les deux premières sont réfutées par la mesure**, la
troisième donne un résultat propre.

| # | Hypothèse | Verdict | Preuve |
|---|-----------|---------|--------|
| H1 | ScCP = prox convexe (fermement non-expansif, L≤1) → « prix de la convexité » vs cible ELS | **RÉFUTÉE** | ScCP entraîné atteint L≈3.9 ; profil ≈ ResNet/UNet libres |
| H2 | Localité règle l'expansivité L, qui règle la mémorisation | **½** : localité→L **confirmée** ; L→mémorisation **réfutée** | patchs L×6 mais tous généralisent ; seul le global mémorise |
| H3 | Le levier créatif est l'agrégation locale par *fold* (recombinaison), pas L/RF/équivariance | **CONFIRMÉE** (contrôle) | patch+fold crée ; sans fold, ou support global (même équivariant), mémorise |

**Conclusion nette** : le levier de créativité n'est ni la convexité, ni la constante de Lipschitz du
débruiteur, ni la taille de champ récepteur, ni l'équivariance-translation du dictionnaire — c'est
l'**agrégation locale overlappante (fold) qui recombine des patchs de plusieurs images d'entraînement
en une mosaïque** ne correspondant à aucune image du train (même translatée). L'expansivité est un
*corrélat* de la localité, mais elle est **découplée** de la créativité.

**Bonus méthodologique** : la distance-au-train (NN L2) n'est **pas** une métrique de mémorisation
robuste sous équivariance — un mémoriseur translaté paraît « nouveau ». Mesurer une distance
invariante par translation (ou au dictionnaire augmenté).

---

## Cadre et objet mesuré

Toutes les mesures portent sur le **débruiteur** `E[x1 | xt]` (l'objet de l'éval ELS de Kamb), en
**couplage indépendant** (terrain sain, immunisé au bug OT corrigé plus tôt — cf. mémoire
`track_ot_coupling_bug`). Quantité centrale : la **norme d'opérateur du Jacobien**
`L(xt) = ‖∂E[x1|xt]/∂xt‖` = constante de Lipschitz locale.
Rappel théorique : un opérateur **fermement non-expansif** (classe d'un prox convexe, = ScCP-ℓ1 idéalisé)
vérifie `L ≤ 1` **partout**. Donc `L > 1` ⟺ cible/modèle **hors** de la classe convexe.

Cibles analytiques (formes closes, `nifty_els_fm.py`) :
- **ELS patch-local** (NIFTY) : mosaïque de patches d'entraînement, patch entier + *fold* gaussien.
- **GLOBAL / IS** : `E[x1|xt] = Σ_i x1_i softmax(...)` sur images entières = **mémorisation**.

---

## Mesure 0 (échec instructif) — kNN en espace pixel

`target_jacobian_spectrum.py` → **méthode inadéquate**, gardée comme négatif documenté.
En 784-D avec ~10⁴ points, les plus proches voisins sont si éloignés que `Cov(x1|xt) ≈ Cov(x1)`
(marginale) à **tout** t : le kNN ne conditionne rien (malédiction de la dimension). D'où le pivot
vers les débruiteurs **analytiques** différentiables (mesures suivantes).

---

## Mesure 1 — Lipschitz exact de la cible (ELS vs global)

**Script** `target_lipschitz.py` · **Figure** `target_lipschitz.png`
Power-iteration exacte (global, `Cov` pondérée symétrique PSD) / différences finies (ELS).

- **Global (mémorisation)** : `L` = 17–52 dans la bande t∈[0.2,0.4], puis s'effondre à ~0 à haut t
  (verrouillage sur l'image la plus proche). Massivement hors classe non-expansive.
- **ELS patch-local** (la cible que ScCP reproduit) : `L` franchit 1 dès t≈0.13, plateau ~3,
  remonte à ~10–20 à t→0.95. **ELS n'est PAS non-expansif** — mais ~5–10× moins que la mémorisation.

→ La localité « dompte » l'expansivité d'un ordre de grandeur, sans l'annuler.

---

## Mesure 2 — Lipschitz des modèles entraînés (réfute H1)

**Script** `model_lipschitz.py` · **Figure** `model_lipschitz.png`
Débruiteur `x + (1-t)·v_model(x,t)`, mêmes points xt, checkpoints indep `temp-5/`.

| t | ScCP (convexe) | ResNet (libre) | UNet (libre) | ELS cible |
|---|---|---|---|---|
| 0.05 | **0.99** | 0.53 | 0.65 | 0.29 |
| 0.32 | 3.48 | 5.30 | 4.26 | 2.79 |
| 0.50 | 3.59 | 3.65 | 2.97 | 3.04 |
| 0.95 | 1.29 | 1.25 | 1.05 | 9.95 |

- **Le ScCP entraîné (K=6, w-bias) atteint L≈3.9 — pas plafonné à 1.** Le w-bias + K fini + param
  vitesse le **sortent** de la classe fermement non-expansive. L'identification Gribonval
  « ScCP = prox convexe » vaut pour l'opérateur idéalisé (K→∞), pas pour le réseau entraîné.
  *(NB : couches indépendantes ⇒ l'unrolling n'est pas une itération contractante de point fixe,
  donc « grand K → prox exact » ne tient pas non plus — d'où l'inutilité de tester K=30.)*
- **ScCP, ResNet, UNet : profils quasi identiques** (ScCP ~25–35 % moins expansif au pic — empreinte
  convexe **faible**, trop petite pour un écart r²). → cohérent avec r²(ScCP)=0.83 ≈ r²(ResNet)=0.85 :
  pas d'écart parce qu'il n'y a pas de contrainte opérante.
- À t→1 les trois modèles → ~1 (skip identité de `x+(1-t)v`) alors qu'ELS explose : divergence
  **partagée** (pas la convexité).

---

## Mesure 3 — Localité → expansivité (confirme la 1re moitié de H2)

**Scripts** `locality_expansivity.py`, `replot_locality.py` · **Figure** `locality_expansivity_clean.png`

**A. Cible ELS** : à t fixé (t≤0.5, plage fiable), `L` est **monotone en taille de patch P**.
À t=0.275 : P3=1.5 < P5=2.1 < P7=2.4 < P11=2.8 < P15=3.6 < P21=5.4 < P27=8.5 < GLOBAL=18.5 (**×12**).

**B. ScCP entraîné** (mêmes xt) : ScCP k=9/K=20 (RF global) > ScCP k=3/K=6 (RF local) dans la même
bande t∈[0.12,0.42] (t=0.16 : 3.44 vs 1.93). La localité côté **architecture** (champ récepteur)
reproduit la hiérarchie d'expansivité de la cible.

→ La taille de patch / champ récepteur est un **bouton continu** d'expansivité.

---

## Mesure 4 — Expansivité → mémorisation ? (réfute la 2e moitié de H2)

**Script** `memorization_vs_locality.py` · **Figure** `memorization_vs_locality.png`
Sweep P du débruiteur ELS → génération (Euler 30 pas) → distance-au-train
(convention `overfit_test_convsccp.py` : dist sample→train vs baseline test→train = 12.2).

| P | 3 | 5 | 7 | 11 | 19 | 27 | **global** |
|---|---|---|---|---|---|---|---|
| L | 1.5 | 2.1 | 2.4 | 2.6 | 4.1 | 8.5 | **13.0** |
| dist→train | 17.2 | 17.2 | 16.3 | 14.4 | 15.6 | 16.7 | **0.42** |
| ratio/baseline | 1.41 | 1.41 | 1.33 | 1.18 | 1.27 | 1.37 | **0.03** |

- **`L` ne pilote PAS la mémorisation.** Tous les débruiteurs à patchs — **même P=27 (L=8.5,
  patch 27×27 ≈ image entière)** — génèrent plus loin du train qu'un vrai chiffre inédit (ratio 1.2–1.4).
  L monte ×6 sans que la mémorisation bouge.
- La mémorisation n'apparaît qu'au débruiteur **global** — une **falaise**, pas une pente.

→ L'expansivité est un corrélat de la localité mais **découplée** de la créativité.

---

## Mesure 5 (contrôle) — Qu'est-ce qui gate la mémorisation ? (confirme H3)

**Scripts** `equivariance_control.py`, `replot_equiv_bars.py`
**Figures** `equivariance_control.png` (barres + grille visuelle), `equivariance_control_bars.png`

Déconfusion des trois ingrédients qui séparent « global » de « patch P27 » (support, augmentation, fold).
5 débruiteurs, mêmes seeds ; on mesure dist→train de base **ET** dist→dict **augmenté** (translations
±3 px, 14 700 images). **Nouveau ⟺ loin des DEUX.**

| débruiteur | dist→train | dist→dict augmenté | verdict |
|---|---|---|---|
| 1. patch P7 + fold | 16.8 | 15.7 | **nouveau** |
| 2. patch P27 + fold | 15.5 | 11.4 | **nouveau** |
| 3. patch P27 **sans fold** (pixel-central) | 9.5 | **0.45** | mémorise |
| 4. **global** (image entière) | 0.43 | 0.43 | mémorise |
| 5. **global + équivariance translation** | 13.7 | **0.42** | mémorise |

- **Piège méthodo** : dist→train *seule* est trompeuse — l'arm 5 paraît nouveau (ratio 1.12) mais colle
  au dict augmenté (0.42) = mémorise des chiffres **translatés** (grille visuelle : chiffres nets ≈ arm 4).
- **Équivariance-translation seule ≠ créativité** (arm 5 mémorise).
- **Le fold est le gate** : P27 avec fold crée (arm 2), sans fold mémorise (arm 3).
- **Pas la taille de RF** : P27 (≈ image entière) reste créatif avec fold.

→ Le levier créatif est l'**agrégation locale overlappante (fold)** : sélection de patch **indépendante
par position** + **mélange** gaussien ⇒ recombinaison mosaïque de plusieurs images.

---

## Conclusion et lecture vis-à-vis de Kamb / NIFTY

Kamb pose « localité + équivariance ⇒ créativité ». Nos contrôles **désagrègent** ce couple et montrent
que l'ingrédient opérant n'est ni la localité de champ récepteur (P27 est non-local mais créatif), ni
l'équivariance-translation prise isolément (arm 5 mémorise), mais l'**agrégation locale overlappante**
qui autorise la **recombinaison** de patches issus d'images différentes. C'est une lecture plus fine,
et testable, du mécanisme mosaïque.

En parallèle, côté ScCP : sa structure convexe (prox ℓ1) **n'est pas une contrainte opérante** sur le
réseau entraîné (L≈3.9, ≈ modèles libres), ce qui explique proprement pourquoi le ScCP-FM reproduit ELS
aussi bien qu'un ResNet libre (r² 0.83 ≈ 0.85) **sans** invoquer de « prix de la convexité ».

## Caveats honnêtes

- Tout est mesuré sur les débruiteurs **analytiques** (mesures 4–5) ou sur des checkpoints **K=6**
  (mesure 2). Le pont vers ScCP entraîné n'est bouclé que pour l'**expansivité**, pas pour la
  recombinaison-fold : le ScCP n'a pas de fold explicite (sa recombinaison passe par les conv transposées).
- La FD à très bas bruit (t→0.95) est confondue par le facteur `t/(1-t)²` + le skip identité →
  région écartée des lectures.
- Le contrôle arm 5 vs P27 confond encore taille de dictionnaire (300 images entières vs 235k patches) ;
  l'attribution au fold s'appuie surtout sur l'arm 3 (même dict que P27, fold retiré → mémorise).

## Prochain maillon proposé

Vérifier que le **ScCP entraîné recombine** (et ne mémorise pas) : générer des échantillons ScCP et
mesurer la distance au dictionnaire **augmenté** (métrique invariante par translation, comme ici),
+ inspection visuelle mosaïque vs chiffre. Réutilise `overfit_test_convsccp.py` + la logique
`equivariance_control.py`. Aucun entraînement requis (checkpoints k3/k9 existants).

---

### Fichiers produits (dans `~/UNN_for_FM/`)

| Mesure | Script(s) | Figure(s) | Metrics |
|---|---|---|---|
| 0 (échec) | `target_jacobian_spectrum.py` | `target_jacobian_spectrum.png` | `target_jacobian_metrics.txt` |
| 1 | `target_lipschitz.py` | `target_lipschitz.png` | `target_lipschitz_metrics.txt` |
| 2 | `model_lipschitz.py` | `model_lipschitz.png` | `model_lipschitz_metrics.txt` |
| 3 | `locality_expansivity.py`, `replot_locality.py` | `locality_expansivity_clean.png` | `locality_expansivity_metrics.txt` |
| 4 | `memorization_vs_locality.py` | `memorization_vs_locality.png` | `memorization_vs_locality_metrics.txt` |
| 5 | `equivariance_control.py`, `replot_equiv_bars.py` | `equivariance_control.png`, `equivariance_control_bars.png` | `equivariance_control_metrics.txt` |

*Log d'exécution complet : `claude.log`.*
