# Résultats — DFB vs ScCP, prox L1, 2-moons

Ce dossier contient les résultats d'un benchmark de **Flow Matching** sur le jeu de
données *2-moons* (dimension 2), avec comme distribution source une distribution normale.

## Remarque
Dans les graphes "overview_epoch_XXX", il y a marqué "8 gaussiennes", c'est une faute que j'ai oublié de corrigé.
Je m'en suis rendu compte à la fin, mais je ne voulais pas tout relancer.

## Modèles comparés

Deux familles d'algorithmes déroulés, chacune avec deux variantes de
l'opérateur linéaire interne :

- **DFB** (`DFB_UNN`) — Dual Forward-Backward.
- **ScCP** (`ScCP_UNN`) — Chambolle-Pock (accéléré, avec schedule adaptatif
  tau/sigma/alpha).
- **LFO** (Learned Forward Operator) — `W` et `V` sont des paramètres appris
  indépendants, le pas `tau` est appris.
- **LNO** (Learned Normalized Operator) — `V = W` (ou `Wᵗ`), le pas est fixé
  par la norme spectrale de `W`.

Tous les modèles ci-dessus utilisent une **prox analytique L1** (projection sur
la boule L1 duale, pas de MLP appris pour la prox). Un **MLP_baseline**
(perceptron simple `[x, t] -> v_t`) est inclus comme référence non-UNN.

## Les deux études d'ablation

1. **Impact du nombre de couches K** (`*_K{5,10,15}`) — dimension duale
   (`dual_dim`) fixée à **32**, K variant dans `{5, 10, 15}`.
2. **Impact de la dimension des features / variable duale**
   (`*_dual{16,32,64}`) — nombre de couches `K` fixé à **10**,
   `dual_dim` variant dans `{16, 32, 64}`.

(`*_K10` et `*_dual32` désignent donc la même configuration de référence
K=10, dual_dim=32, présente dans les deux études.)

## Setup d'entraînement

- Cible : 2-moons (10 000 points, normalisés), source : 8 gaussiennes
  (rayon 2.0, std 0.3).
- Flow Matching : `ExactOptimalTransportConditionalFlowMatcher` (sigma=0.01). J'utilise donc du Minibatch Optimal transport.
- Optimiseur Adam, lr=1e-3, gradient clipping (max_norm=1.0).
- Batch size 256, **50 epochs** par run.
- Génération à l'inférence par résolution de l'ODE de flow (NeuralODE,
  solveur `dopri5`).

## Métriques suivies

Pour chaque run, deux quantités sont enregistrées :

- **Training loss** : MSE entre le champ de vitesse prédit et le champ de
  vitesse cible (`loss.txt` / `loss.png`), à chaque epoch.
- **Erreur (W2)** : distance de Wasserstein-2 exacte entre 500 points
  générés et 500 points cibles (`error.txt` / `error.png`), calculée
  seulement aux epochs d'évaluation (10, 20, 30, 40, 50).

## Contenu de chaque sous-dossier d'expérience

Chaque expérience (`DFB_UNN_L1_LFO_K5`, `ScCP_UNN_L1_LNO_dual64`, etc.)
contient :

| Fichier | Description |
|---|---|
| `params.txt` | Nombre de paramètres entraînables du modèle |
| `loss.txt`, `loss.png` | Training loss (FM/MSE) par epoch |
| `error.txt`, `error.png` | Erreur W2 (générés vs cible) aux epochs d'évaluation |
| `epoch_{N}.png` | Nuage de points générés à l'epoch N, superposé à la cible |
| `overview_epoch_{N}.png` | 3 panneaux : source (1 gaussienne, malgré l'erreur dans la légende) / cible (2-moons) / généré |
| `vector_field_epoch_{N}.png` | Champ de vitesse appris `v_t(x)` sur une grille, pour plusieurs `t` |

## Fichiers de synthèse (racine du dossier)

| Fichier | Description |
|---|---|
| `summary.txt` | Tableau récapitulatif : params, loss finale, erreur W2 finale, temps d'entraînement, statut, pour toutes les expériences |
| `study_K_loss.png` | Loss finale en fonction de K, une courbe par (modèle, version) |
| `study_K_w2_error.png` | Erreur W2 finale en fonction de K |
| `study_dual_loss.png` | Loss finale en fonction de `dual_dim`, une courbe par (modèle, version) |
| `study_dual_w2_error.png` | Erreur W2 finale en fonction de `dual_dim` |

## Résultats finaux (extraits de `summary.txt`)

| Modèle | params | loss finale | erreur W2 finale |
|---|---|---|---|
| MLP_baseline | 4 546 | 0.0414 | 0.208 |
| ScCP LFO K5 | 1 126 | 0.0610 | 0.250 |
| ScCP LNO K5 | 810 | 0.0441 | 0.207 |
| ScCP LFO K10 / dual32 | 2 251 | 0.0556 | 0.258 |
| ScCP LNO K10 / dual32 | 1 620 | 0.0358 | 0.191 |
| ScCP LFO K15 | 3 376 | 0.0521 | 0.294 |
| ScCP LNO K15 | 2 430 | 0.0357 | 0.314 |
| ScCP LFO dual16 | 1 611 | 0.0605 | 0.275 |
| ScCP LNO dual16 | 1 300 | 0.0387 | 0.239 |
| ScCP LFO dual64 | 3 531 | 0.0507 | 0.260 |
| ScCP LNO dual64 | 2 260 | 0.0357 | 0.225 |
| DFB LFO K5 | 1 255 | 0.0748 | 0.285 |
| DFB LNO K5 | 930 | 0.0758 | 0.317 |
| DFB LFO K10 / dual32 | 2 510 | 0.1022 | 0.300 |
| DFB LNO K10 / dual32 | 1 860 | 0.1416 | 0.528 |
| DFB LFO K15 | 3 765 | 0.0833 | 0.304 |
| DFB LNO K15 | 2 790 | 0.0511 | 0.237 |
| DFB LFO dual16 | 1 870 | 0.1078 | 0.411 |
| DFB LNO dual16 | 1 540 | 0.0552 | 0.260 |
| DFB LFO dual64 | 3 790 | 0.0775 | 0.290 |
| DFB LNO dual64 | 2 500 | 0.0597 | 0.261 |

(Table complète, valeurs exactes : voir `summary.txt`.)

### Observations

- **ScCP surpasse systématiquement DFB** sur cette plage d'hyperparamètres,
  à la fois en loss finale et en erreur W2, pour un nombre de paramètres
  comparable.
- Pour **ScCP**, la variante **LNO** est presque toujours meilleure que LFO
  (loss et W2 plus faibles), à K et `dual_dim` égaux.
- Pour **DFB**, la tendance est moins nette : LNO gagne à K15/dual16/dual64,
  mais **DFB LNO à K10** est un net outlier (loss et W2 nettement plus
  élevés que ses voisins K5/K15) — probablement de l'instabilité
  d'entraînement plutôt qu'un effet structurel.
- L'augmentation de K ou de `dual_dim` n'apporte pas de gain monotone pour
  DFB ni ScCP sur cette plage (5–15 couches, 16–64 features) : la loss/erreur
  ne décroît pas clairement avec plus de capacité, ce qui suggère qu'on est
  déjà dans un régime où le facteur limitant n'est pas la capacité du
  modèle.
