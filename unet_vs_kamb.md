# UNet — Kamb & Ganguli vs ce repo

Comparaison statique (aucun run). Chiffres de params/normalisation/activation
introspectés depuis les modèles réels : le backbone `MinimalUNet` entraîné
(`~/repro_kamb/files/backbone_MNIST_UNet_zeros.pt`) et les classes `UNet` /
`SmallUNet` de `models/architectures.py`.

| Propriété | Kamb `MinimalUNet` | mon `UNet` | mon `SmallUNet` |
|---|---|---|---|
| # params | 2 026 946 | 3 049 729 | 176 705 |
| scales (nb maxpool) | 3 (2) | 3 (2) | 2 (1) |
| canaux | [64, 128, 256] | [64, 128, 256] | [32, 64] |
| kernel | 3×3 | 3×3 | 3×3 |
| **normalisation** | **AUCUNE** | **GroupNorm(4)** | **GroupNorm(4)** |
| activation | ReLU | SiLU | SiLU |
| conditionnement temps | chaque bloc (biais additif) | 1× après `inc` | 1× après `inc` |
| taille image | 32×32 (MNIST resize) | 28×28 | 28×28 |
| objectif | score-matching / DDIM (predict ε) | Flow Matching (predict x1 / vitesse) | idem |
| échantillonnage | ODE déterministe (DDIM = probability-flow) | ODE déterministe | idem |
| RF effectif vs temps | **rétrécit** grand→~3×3 (Fig. 4, coarse-to-fine) | reste **global** à tout t | idem |
| padding | zeros **et** circular (étudie les deux) | zeros | zeros |
| upsampling | ConvTranspose2d (2, s2) | ConvTranspose2d (2, s2) | idem |

## Différences qui comptent (par ordre d'impact)

1. **Normalisation — le facteur #1.** Eux : *aucune*, choix délibéré pour préserver
   la localité (Def. 3.2 : la sortie d'un pixel ne dépend que de son voisinage).
   Toi : `GroupNorm(4)` dans chaque `DoubleConv`. GroupNorm normalise par la
   moyenne/variance calculées sur **toute l'étendue spatiale** du groupe de canaux
   → mélange l'info de tous les pixels dès la 1ʳᵉ couche, **quelle que soit la
   taille du kernel**. Ça casse la localité que toute la théorie ELS suppose. Une
   meilleure perf de ton UNet peut venir en partie de ce canal non-local (analogue
   au rôle de l'attention, §6 du papier), pas seulement d'une meilleure expressivité.

2. **Le champ récepteur effectif en fonction du temps (coarse-to-fine).**
   Développé en détail dans la section dédiée ci-dessous — c'est le point le plus
   subtil et celui qui explique les gribouillis vs tes chiffres nets.

3. **Conditionnement temps** injecté dans *chaque* bloc (down/bottleneck/up) chez
   eux, vs **une seule fois** après `inc` chez toi.

4. **Secondaire** : ReLU vs SiLU ; image 32 vs 28 ; padding zeros seul vs
   zeros+circular (eux testent circular pour sonder l'équivariance).

5. **`SmallUNet`** est en plus une variante 2-scales / 177 k params — bien plus
   petite que leur 2.0 M — donc écart de capacité, pas seulement de principe.

## Point 2 développé — le champ récepteur effectif en fonction du temps

### 2.0 — D'abord, ce que ce n'est *pas*
Contrairement à ce que ma 1ʳᵉ formulation laissait croire, la différence n'est
**ni** « diffusion vs ODE » **ni** « prédire ε vs prédire la vitesse » :

- DDIM **est** déjà un ODE déterministe (le *probability-flow ODE*). Comme ton Flow
  Matching, il intègre un champ de vecteurs déterministe de bruit → données. Pas de
  stochasticité qui les distinguerait.
- Sur des chemins gaussiens (`x_t = √ᾱ_t x_0 + √(1-ᾱ_t) η` d'un côté,
  `x_t = t·x_1 + (1-t)·x_0` de l'autre), le **score**, le **bruit ε** et la
  **vitesse FM** sont des fonctions **affines** les uns des autres à `(x_t, t)`
  fixés : `v_t = a_t x_t + b_t·ε_t`, `score_t = -ε_t/σ_t`. Prédire l'un ou l'autre
  est une **reparamétrisation** — la classe de fonctions apprise est essentiellement
  la même. Donc « score-matching vs Flow Matching » n'est **pas** le moteur de la
  différence de comportement.

### 2.1 — Le vrai moteur : un RF effectif qui dépend du niveau de bruit
Fait empirique central de Kamb (Fig. 4a, App. C.2) : sur leurs réseaux entraînés,
le **champ récepteur effectif du score rétrécit** au fil du process inverse —
large quand `t` est grand (fort bruit), puis ~3×3 quand `t → 0` (faible bruit).
C'est **émergent de l'objectif de débruitage**, pas un choix d'archi :

- **Fort bruit** : `x_t` est presque du bruit pur. Le débruiteur optimal
  `E[x_0 | x_t]` doit intégrer un **contexte large** pour deviner ne serait-ce que
  la structure grossière → RF grand.
- **Faible bruit** : `x_t ≈ x_0`. La correction optimale est juste d'enlever le
  résidu de bruit, **pixel par pixel** → RF quasi local (3×3).

Le réseau apprend donc un **planning grand→petit**. C'est exactement pourquoi
l'ELS local ne marche qu'avec une **échelle P_t variable** (grande tôt, petite
tard — c'est ce que contient le fichier `scales`, ordonné 15…3), et pourquoi les
**inconsistances fines** apparaissent : la structure grossière (quel objet, où)
est figée **tôt** sous grand RF ; les détails fins sont remplis **tard** sous RF
~3×3, donc des régions éloignées décident « bras ? jambe ? » **sans se coordonner**
→ mauvais nombre de membres (Fig. 5c). C'est un mécanisme *temporel*, pas
seulement spatial.

### 2.2 — MESURÉ : ton ConvScCP entraîné rétrécit AUSSI (coarse-to-fine)
**Correction importante.** J'avais supposé que ton modèle gardait un RF *global à
tout t*. C'est **faux** : la réplication de la Fig. 4a sur le checkpoint entraîné
`results/temp-4/ConvScCP_UNN_L1_LNO/model.pt` (K=20, ic=64, kernel 9×9) donne un RF
effectif qui **se contracte** exactement comme chez Kamb
(`erf_vs_time_checkpoint.py`, `erf_vs_time_checkpoint.png`) :

| t (0=bruit, 1=data) | 0.05 | 0.15 | 0.30 | 0.50 | 0.70 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|
| rayon effectif (px) | 7.1 | 7.1 | 6.9 | 6.2 | 4.9 | 4.1 | 3.6 |
| masse hors 17×17 | 30% | 30% | 30% | 24% | 15% | 10% | 8% |

Deux enseignements :
- **Coarse-to-fine confirmé** : grand RF à fort bruit (7 px), petit à faible bruit
  (3.6 px). Même mécanisme temporel que leur Fig. 4a.
- **L'entraînement rend le modèle LOCAL** : le RF architectural du 9×9×K était
  ~global (~9.9 px, mesuré sur poids aléatoires dans `erf_by_kernel.png`), mais le
  modèle *entraîné* n'en utilise que 3.6–7 px. Le « 9×9 → global » était une borne
  sup, pas ce que le réseau fait réellement.

### 2.3 — Ce que ça change (et la question qui reste ouverte)
- **Implication forte** : ton ConvScCP est en réalité un modèle **local +
  coarse-to-fine + (≈)équivariant** — donc *dans* le cadre de la théorie ELS, pas en
  dehors. Il devrait être **prédictible par une machine ELS calibrée** sur ses
  propres scales. C'est LA prochaine expérience à faire (calibrer P_t sur ce
  checkpoint, puis mesurer le r² ELS vs ConvScCP, comme pour leurs ResNet/UNet).
- **Question RÉSOLUE (expérience k=3).** « Pourquoi tes chiffres sont-ils plus nets
  que leurs gribouillis ? » → c'est bien **la magnitude du RF**. En ré-entraînant un
  ConvScCP **kernel 3×3, K=6** (au lieu de 9×9), le RF effectif tombe à **1.7–3.8 px**
  (`erf_vs_time_k3.png`) et **les sorties deviennent des mosaïques fragmentées**
  (`overfit_grid_k3.png`, colonne 1) — exactement les gribouillis de Kamb.

  | | RF effectif | sorties | r²(IS-FM) | ratio dist-train |
  |---|---|---|---|---|
  | k=9 K=20 | 3.6–7 px | chiffres nets | 0.70 | 1.22 |
  | k=3 K=6 | 1.7–3.8 px | mosaïques | 0.54 | 1.52 |

  Donc : ~7 px (k=9) suffit à garder un chiffre cohérent sur MNIST ; ~3 px (k=3)
  est trop local → ça fragmente. Le seuil cohérence↔mosaïque est là. Plus local ⇒
  plus créatif/combinatoire ⇒ plus loin du train ⇒ moins expliqué par l'IS :
  la thèse locality→creativity de Kamb, démontrée sur ce modèle.
- **NB méthodo** : ceci ne touche pas le point 1 (la GroupNorm de ton `UNet` reste
  un canal non-local *architectural*, indépendant de `t`). La correction concerne le
  ConvScCP-L1 (sans norm), pour lequel l'argument « global » était erroné.

## Ce qui est déjà aligné

Kernel 3×3, structure encoder/decoder à skip-connections, MaxPool + ConvTranspose,
3 scales [64,128,256] (pour mon `UNet`), pas de weight decay implicite particulier.
Le `MaxPool2d(2)` casse l'équivariance translationnelle exacte des deux côtés —
mais c'est **partagé** avec leur UNet (Ronneberger utilise aussi du pooling), donc
ce n'est pas un écart introduit par ton implémentation.
