# Distribution générée — grille (nombre de couches K) x (nombre de features / dual_dim)

Comparaison croisée des distributions générées (epoch 50, nuage de points généré superposé à la cible 2-moons) en faisant varier le nombre de couches `K` (lignes) et la dimension des features / variable duale `dual_dim` (colonnes), pour chaque combinaison modèle x version (prox L1). Données issues de `results_2moons_DFB_ScCP_L1` (étude "normale", source = 8 gaussiennes) et `results_2moons_DFB_ScCP_L1_small` (étude "small", source = gaussienne standard).

Seules les cases correspondant à une expérience réellement lancée sont remplies (les deux études forment chacune une croix dans l'espace (K, dual_dim), pas une grille complète) — les cases vides (`—`) n'ont pas été testées.


## DFB — LFO

| K \ dual_dim | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| **K=1** | — | — | <img src="results_2moons_DFB_ScCP_L1_small/DFB_UNN_L1_LFO_K1/epoch_50.png" width="140"> | — | — |
| **K=3** | <img src="results_2moons_DFB_ScCP_L1_small/DFB_UNN_L1_LFO_dual4/epoch_50.png" width="140"> | <img src="results_2moons_DFB_ScCP_L1_small/DFB_UNN_L1_LFO_dual8/epoch_50.png" width="140"> | <img src="results_2moons_DFB_ScCP_L1_small/DFB_UNN_L1_LFO_K3/epoch_50.png" width="140"> | — | — |
| **K=5** | — | — | <img src="results_2moons_DFB_ScCP_L1_small/DFB_UNN_L1_LFO_K5/epoch_50.png" width="140"> | <img src="results_2moons_DFB_ScCP_L1/DFB_UNN_L1_LFO_K5/epoch_50.png" width="140"> | — |
| **K=10** | — | — | <img src="results_2moons_DFB_ScCP_L1/DFB_UNN_L1_LFO_dual16/epoch_50.png" width="140"> | <img src="results_2moons_DFB_ScCP_L1/DFB_UNN_L1_LFO_K10/epoch_50.png" width="140"> | <img src="results_2moons_DFB_ScCP_L1/DFB_UNN_L1_LFO_dual64/epoch_50.png" width="140"> |
| **K=15** | — | — | — | <img src="results_2moons_DFB_ScCP_L1/DFB_UNN_L1_LFO_K15/epoch_50.png" width="140"> | — |

## DFB — LNO

| K \ dual_dim | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| **K=1** | — | — | <img src="results_2moons_DFB_ScCP_L1_small/DFB_UNN_L1_LNO_K1/epoch_50.png" width="140"> | — | — |
| **K=3** | <img src="results_2moons_DFB_ScCP_L1_small/DFB_UNN_L1_LNO_dual4/epoch_50.png" width="140"> | <img src="results_2moons_DFB_ScCP_L1_small/DFB_UNN_L1_LNO_dual8/epoch_50.png" width="140"> | <img src="results_2moons_DFB_ScCP_L1_small/DFB_UNN_L1_LNO_K3/epoch_50.png" width="140"> | — | — |
| **K=5** | — | — | <img src="results_2moons_DFB_ScCP_L1_small/DFB_UNN_L1_LNO_K5/epoch_50.png" width="140"> | <img src="results_2moons_DFB_ScCP_L1/DFB_UNN_L1_LNO_K5/epoch_50.png" width="140"> | — |
| **K=10** | — | — | <img src="results_2moons_DFB_ScCP_L1/DFB_UNN_L1_LNO_dual16/epoch_50.png" width="140"> | <img src="results_2moons_DFB_ScCP_L1/DFB_UNN_L1_LNO_K10/epoch_50.png" width="140"> | <img src="results_2moons_DFB_ScCP_L1/DFB_UNN_L1_LNO_dual64/epoch_50.png" width="140"> |
| **K=15** | — | — | — | <img src="results_2moons_DFB_ScCP_L1/DFB_UNN_L1_LNO_K15/epoch_50.png" width="140"> | — |

## ScCP — LFO

| K \ dual_dim | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| **K=1** | — | — | <img src="results_2moons_DFB_ScCP_L1_small/ScCP_UNN_L1_LFO_K1/epoch_50.png" width="140"> | — | — |
| **K=3** | <img src="results_2moons_DFB_ScCP_L1_small/ScCP_UNN_L1_LFO_dual4/epoch_50.png" width="140"> | <img src="results_2moons_DFB_ScCP_L1_small/ScCP_UNN_L1_LFO_dual8/epoch_50.png" width="140"> | <img src="results_2moons_DFB_ScCP_L1_small/ScCP_UNN_L1_LFO_K3/epoch_50.png" width="140"> | — | — |
| **K=5** | — | — | <img src="results_2moons_DFB_ScCP_L1_small/ScCP_UNN_L1_LFO_K5/epoch_50.png" width="140"> | <img src="results_2moons_DFB_ScCP_L1/ScCP_UNN_L1_LFO_K5/epoch_50.png" width="140"> | — |
| **K=10** | — | — | <img src="results_2moons_DFB_ScCP_L1/ScCP_UNN_L1_LFO_dual16/epoch_50.png" width="140"> | <img src="results_2moons_DFB_ScCP_L1/ScCP_UNN_L1_LFO_K10/epoch_50.png" width="140"> | <img src="results_2moons_DFB_ScCP_L1/ScCP_UNN_L1_LFO_dual64/epoch_50.png" width="140"> |
| **K=15** | — | — | — | <img src="results_2moons_DFB_ScCP_L1/ScCP_UNN_L1_LFO_K15/epoch_50.png" width="140"> | — |

## ScCP — LNO

| K \ dual_dim | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| **K=1** | — | — | <img src="results_2moons_DFB_ScCP_L1_small/ScCP_UNN_L1_LNO_K1/epoch_50.png" width="140"> | — | — |
| **K=3** | <img src="results_2moons_DFB_ScCP_L1_small/ScCP_UNN_L1_LNO_dual4/epoch_50.png" width="140"> | <img src="results_2moons_DFB_ScCP_L1_small/ScCP_UNN_L1_LNO_dual8/epoch_50.png" width="140"> | <img src="results_2moons_DFB_ScCP_L1_small/ScCP_UNN_L1_LNO_K3/epoch_50.png" width="140"> | — | — |
| **K=5** | — | — | <img src="results_2moons_DFB_ScCP_L1_small/ScCP_UNN_L1_LNO_K5/epoch_50.png" width="140"> | <img src="results_2moons_DFB_ScCP_L1/ScCP_UNN_L1_LNO_K5/epoch_50.png" width="140"> | — |
| **K=10** | — | — | <img src="results_2moons_DFB_ScCP_L1/ScCP_UNN_L1_LNO_dual16/epoch_50.png" width="140"> | <img src="results_2moons_DFB_ScCP_L1/ScCP_UNN_L1_LNO_K10/epoch_50.png" width="140"> | <img src="results_2moons_DFB_ScCP_L1/ScCP_UNN_L1_LNO_dual64/epoch_50.png" width="140"> |
| **K=15** | — | — | — | <img src="results_2moons_DFB_ScCP_L1/ScCP_UNN_L1_LNO_K15/epoch_50.png" width="140"> | — |
