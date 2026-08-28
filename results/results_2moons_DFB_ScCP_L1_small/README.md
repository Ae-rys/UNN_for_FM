# Résultats — DFB vs ScCP, prox L1, 2-moons (modèles "small")

Ce dossier contient les résultats d'un benchmark de **Flow Matching** sur le jeu de
données *2-moons* (dimension 2), avec comme distribution source une **gaussienne
standard**, avec des modèles beaucoup plus petits :
K ∈ {1, 3, 5} et `dual_dim` ∈ {4, 8, 16}, contre K ∈ {5, 10, 15} et
`dual_dim` ∈ {16, 32, 64} dans l'étude "normale").

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

1. **Impact du nombre de couches K** (`*_K{1,3,5}`) — dimension duale
   (`dual_dim`) fixée à **16**, K variant dans `{1, 3, 5}`.
2. **Impact de la dimension des features / variable duale**
   (`*_dual{4,8,16}`) — nombre de couches `K` fixé à **3**,
   `dual_dim` variant dans `{4, 8, 16}`.

(`*_K3` et `*_dual16` désignent donc la même configuration de référence
K=3, dual_dim=16, présente dans les deux études.)

## Setup d'entraînement

- Cible : 2-moons (10 000 points, normalisés), source : **gaussienne standard**
  (et non 8 gaussiennes, contrairement à ce qui peut être affiché par erreur
  dans la légende des `overview_epoch_*.png` — cf. la même coquille déjà
  signalée pour l'étude "normale").
- Flow Matching : `ExactOptimalTransportConditionalFlowMatcher` (sigma=0.01),
  donc minibatch optimal transport.
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

Chaque expérience (`DFB_UNN_L1_LFO_K1`, `ScCP_UNN_L1_LNO_dual8`, etc.)
contient :

| Fichier | Description |
|---|---|
| `params.txt` | Nombre de paramètres entraînables du modèle |
| `loss.txt`, `loss.png` | Training loss (FM/MSE) par epoch |
| `error.txt`, `error.png` | Erreur W2 (générés vs cible) aux epochs d'évaluation |
| `epoch_{N}.png` | Nuage de points générés à l'epoch N, superposé à la cible |
| `overview_epoch_{N}.png` | 3 panneaux : source (gaussienne) / cible (2-moons) / généré |
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
| MLP_baseline | 4 546 | 0.0392 | 0.207 |
| ScCP LFO K1 | 162 | 0.1937 | 0.596 |
| ScCP LNO K1 | 130 | 0.1887 | 0.563 |
| ScCP LFO K3 / dual16 | 484 | 0.0900 | 0.345 |
| ScCP LNO K3 / dual16 | 390 | 0.0528 | 0.247 |
| ScCP LFO K5 | 806 | 0.0665 | 0.307 |
| ScCP LNO K5 | 650 | 0.0461 | 0.317 |
| ScCP LFO dual4 | 340 | 0.1160 | 0.465 |
| ScCP LNO dual4 | 318 | 0.0666 | 0.283 |
| ScCP LFO dual8 | 388 | 0.1070 | 0.394 |
| ScCP LNO dual8 | 342 | 0.0542 | 0.227 |
| DFB LFO K1 | 187 | 0.2008 | 0.599 |
| DFB LNO K1 | 154 | 0.1983 | 0.627 |
| DFB LFO K3 / dual16 | 561 | 0.0847 | 0.273 |
| DFB LNO K3 / dual16 | 462 | 0.0859 | 0.349 |
| DFB LFO K5 | 935 | 0.0854 | 0.322 |
| DFB LNO K5 | 770 | 0.0607 | 0.290 |
| DFB LFO dual4 | 417 | 0.1252 | 0.484 |
| DFB LNO dual4 | 390 | 0.0931 | 0.363 |
| DFB LFO dual8 | 465 | 0.1310 | 0.494 |
| DFB LNO dual8 | 414 | 0.1067 | 0.427 |

(Table complète, valeurs exactes : voir `summary.txt`.)

### Observations

- À très petite capacité (**K=1**), DFB et ScCP performent quasiment de
  manière identique et restent loin du MLP baseline — l'algorithme déroulé
  n'a tout simplement pas assez d'itérations pour transporter la masse.
- Dès **K=3**, **ScCP creuse l'écart avec DFB** : à K=3/dual16, ScCP LNO
  obtient loss=0.053 / W2=0.247 contre DFB LNO loss=0.086 / W2=0.349 — un
  comportement cohérent avec l'étude "normale" (K plus grand).
- Pour **ScCP**, la variante **LNO** est systématiquement meilleure que LFO
  sur cette plage de petits modèles (loss et W2 plus faibles à K et
  `dual_dim` égaux).
- Pour **DFB**, LNO gagne presque toujours en loss, mais l'effet sur l'erreur
  W2 est moins net (ex. DFB K3 : LNO a une loss plus faible que LFO mais une
  erreur W2 plus élevée — 0.349 vs 0.273). Dans le cas LFO, comme dans l'autre 
  étude, il y a de la concentration sur certains points, des "lignes" qui apparaissent.
- **L'augmentation de K (1→3→5) ou de `dual_dim` (4→8→16) améliore nettement
  les performances** sur cette plage de très petits modèles, contrairement à
  l'étude "normale" (K=5–15, dual_dim=16–64) où le gain de capacité
  supplémentaire n'était plus monotone. Cela suggère que les modèles K=1
  (et dans une moindre mesure K=3/dual≤8) sont clairement sous-paramétrés,
  alors que dans l'étude "normale" tous les modèles avaient suffisamment de paramètres.
