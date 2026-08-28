# Condition de pas du ScCP déroulé — le cas LFO

Document de travail. Établit ce que doivent valoir `tau_k` et `sigma_k` dans
`ConvScCP_UNN`, pourquoi la version LFO actuelle s'écarte de l'algorithme de
référence, et ce que ça a coûté à la variante `ConvScCP_UNN_v2`.

Vérifications numériques : `verif_lfo_condition.py`.

---

## 1. L'algorithme de référence

Chambolle & Pock (2011), **Algorithme 2** (ALG2), pour

```
min_x  f(x) + g(K x)
```

avec `f` **mu-fortement convexe**, `K` linéaire, `L = ||K||`.

Initialisation : `tau_0`, `sigma_0` tels que **`tau_0 · sigma_0 · L^2 <= 1`**.

Pour `k >= 0` :

```
u_{k+1} = prox_{sigma_k g*}( u_k + sigma_k K xbar_k )
x_{k+1} = prox_{tau_k f}  ( x_k - tau_k K* u_{k+1} )
theta_k = (1 + 2 mu tau_k)^(-1/2)
tau_{k+1}  = theta_k · tau_k
sigma_{k+1} = sigma_k / theta_k
xbar_{k+1} = x_{k+1} + theta_k (x_{k+1} - x_k)
```

### L'invariant qui porte tout

```
tau_{k+1} · sigma_{k+1} = (theta_k tau_k) · (sigma_k / theta_k) = tau_k · sigma_k
```

**Le produit `tau·sigma` est constant.** La condition, vérifiée à `k = 0`, vaut donc
pour tout `k`. C'est la raison d'être de `sigma_{k+1} = sigma_k / theta_k` : cette
ligne n'est pas cosmétique, c'est elle qui maintient la condition pendant que
`tau` décroît. Retirer la mise à jour de `sigma` ne rend pas l'algorithme
« légèrement moins accéléré » — elle casse l'équilibre primal/dual.

---

## 2. Correspondance avec le code

`_OrigConvScCP_Iteration.forward` calcule, dans cet ordre :

```
x_next = prox_{tau f}( x - tau · B u )          # B = conv_transpose2d(., V)
xbar   = x_next + alpha · (x_next - x)
u_next = prox_{sigma g*}( u + sigma · A xbar )  # A = conv2d(., W)
```

C'est ALG2 avec le dual réindexé d'un demi-pas : le `u_k` du code est le
`u_{k+1}` de CP. Les deux étapes utilisent bien le même indice `k` pour
`tau_k` / `sigma_k`, et le `alpha` de l'extrapolation est bien le `theta_k` qui
met `tau` à jour. **La structure est correcte.**

### LNO — correct

```python
tau_k   = softplus(log_tau[k])            # libre, un par itération
sigma_k = 0.99 / (tau_k · sn^2)
```

donne `tau_k · sigma_k = 0.99 / sn^2` **à chaque k**. La condition est satisfaite
par construction, et `sigma` croît automatiquement quand `tau` décroît. La
récurrence de `tau` n'est pas celle d'ALG2 (`tau_k` est appris librement plutôt
que dérivé de `theta`), mais la condition, elle, tient.

### LFO — la ligne manquante

```python
tau_k   = softplus(log_tau0)
sigma_k = 1.0                              # <-- jamais mis à jour
for layer in layers:
    alpha_k = (1 + 2 tau_k)^(-1/2)         # mu = 1 codé en dur
    ...
    tau_k = alpha_k · tau_k                # tau décroît
```

`tau_k · sigma_k = tau_k` **décroît**. La condition reste satisfaite — elle devient
même plus facile — mais l'**accélération est perdue** : le pas dual n'augmente pas
alors que le pas primal s'effondre. À `mu = 1` et `tau_0 = softplus(-0.5) ~ 0.474`,
`theta ~ 0.72` : `sigma` devrait croître d'un facteur `1/0.72^10 ~ 28` sur dix
itérations. Il reste à 1.

---

## 3. Pourquoi ça a tué `ConvScCP_UNN_v2`

v2 remplace `mu = 1` par la vraie constante de forte convexité du terme d'attache
du problème inverse :

```
mu(t) = t^2 / (1-t)^2
```

À `t = 0.9`, `mu = 81`, donc `theta_0 = (1 + 2·81·0.474)^(-1/2) = 0.114`. La suite,
à `sigma` figé :

| k | tau_k | sigma_k voulu | sigma_k du code |
|---|-------|---------------|-----------------|
| 0 | 0.474 | 1.00 | 1 |
| 1 | 0.054 | 8.8 | 1 |
| 2 | 0.017 | 27 | 1 |
| 3 | 0.009 | 53 | 1 |

Le pas primal perd un facteur 50 en trois itérations pendant que le pas dual reste
à 1 : les itérations suivantes ne font quasiment plus rien, le déroulé se réduit à
un pas utile. C'est **exactement** la bosse mesurée sur MNIST à `t = 0.85–0.90`
(+69 % et +180 % de nmse par rapport à v1), alors que l'écart n'est que de 2–5 %
au milieu de la plage.

Avec `mu = 1` le même défaut existe, mais trente fois plus doux — et l'apprentissage
bout-à-bout l'absorbe. C'est pourquoi il était invisible dans v1.

---

## 4. Que vaut `L` quand `V != W` ?

**La difficulté du cas LFO.** L'opérateur du pas primal (`V`) et celui du pas dual
(`W`) sont indépendants. L'algorithme n'est donc plus littéralement CP, qui exige
`K` et son adjoint `K*`. Aucun théorème de convergence ne s'applique tel quel.

Notons

```
A : X -> U,   A x = conv2d(x, W, padding=p)
B : U -> X,   B u = conv_transpose2d(u, V, padding=p)
```

En CP, `B = A*`. En LFO, `B` est libre.

Ce qui gouverne la stabilité est le **gain de boucle** : partant de `x`, un tour
complet applique `sigma A` puis `tau B`, soit `tau·sigma·(B o A)`. La condition
naturelle est donc

```
tau · sigma · ||B o A|| <= 1        soit    L^2 := ||B o A||_op
```

**Cohérence.** Si `B = A*`, alors `||A* A|| = ||A||^2` et on retrouve
`tau·sigma·||A||^2 <= 1`. La généralisation contient bien le cas standard.

C'est une **condition naturelle, pas un théorème** : elle redonne CP quand
`V = W`, et contrôle le gain de la boucle linéarisée sinon. Rien de plus.

---

## 5. Calculer `L^2` correctement

`sigma_max_power_iter` réduit `W` de forme `(C_out, C_in, k, k)` à une matrice
`(C_out, C_in·k^2)` et en prend la plus grande valeur singulière. **Ce n'est pas la
norme d'opérateur de la convolution.** Mesuré (`verif_lfo_condition.py`, cas
`C_in=2, C_u=3, k=3, S=8`) :

```
norme d'operateur vraie : 2.607
estimation par reshape  : 1.538      ratio 1.70
```

Le reshape **sous-estime**, donc `L^2` est trop petit et les pas autorisés trop
grands. LNO utilise cette estimation optimiste.

**Le calcul juste**, aux vraies dimensions spatiales et pour le padding réel :

```
M  (x) = conv_transpose2d( conv2d(x, W, p), V, p )       # M = B o A : X -> X
M* (y) = conv_transpose2d( conv2d(y, V, p), W, p )       # M* = A* o B*
```

en utilisant que `conv_transpose2d(., W, p)` est **exactement** l'adjoint de
`conv2d(., W, p)` à stride 1 et padding égal (vérifié à 0 près, en float64).
Itération de la puissance sur `M* M` :

```
v <- normalize( M*(M v) )        L^2 = ||M v|| / ||v||   a convergence
```

Pour `V = W` : `M = A* A`, auto-adjoint, et `||M|| = ||A||^2`. Le même code
redonne donc LNO exactement.

---

## 6. Ce que fait `ConvScCP_UNN_v3`

Une fois `tau·sigma` fixé par la condition, il reste **un** degré de liberté :
l'équilibre primal/dual. On l'apprend.

```
L^2   = loop_gain()                    # iteration de la puissance, buffer persistant
gamma = softplus(log_gamma)            # appris, un scalaire — l'equilibre
tau_0 = gamma / L
sigma_k = c / (tau_k · L^2)            # c = 0.99
```

Propriétés, toutes voulues :

- `tau_k · sigma_k = c / L^2` **à chaque k** : la condition tient partout.
- `sigma` croît **exactement** comme `1/theta` quand `tau` décroît, puisqu'il est
  recalculé depuis `tau_k`. L'accélération est rétablie sans code dédié.
- `tau_0 = gamma/L` rend le paramètre appris sans dimension : il ne dérive pas
  quand l'échelle de `W` bouge pendant l'entraînement.
- `V = W` redonne LNO avec un `L` correct au lieu du reshape.

La récurrence reste celle d'ALG2 avec la vraie constante :

```
theta_k = (1 + 2 mu(t) tau_k)^(-1/2)         mu(t) = t^2/(1-t)^2
tau_{k+1} = theta_k · tau_k
```

`tau_k`, `sigma_k`, `theta_k` sont des tenseurs `(B,1,1,1)` : ils dépendent de `t`.

---

## 7. Ce qui reste non garanti

À énoncer tel quel, sans l'enjoliver. **v3 corrige la discipline de pas et le
terme d'attache ; elle ne rend pas le modèle « mathématiquement correct ».** Les
deux premiers points ci-dessous sont structurels et hors de portée d'un réglage.

1. **`V != W` : il n'y a pas de problème variationnel du tout.** C'est plus fort
   que « pas de preuve de convergence », et c'est le point important.

   Le point fixe de l'itération vérifie

   ```
   0 ∈ ∂f(x) + B ∂g(A x)
   ```

   Ce serait la condition d'optimalité de `min f(x) + g(Ax)` **seulement si
   `B = A*`**. Sinon, `∇f + B∘A` devrait être symétrique pour être un champ de
   gradients. Mesuré (`verif_lfo_condition.py`, cas `C_in=1, C_u=2, k=3, S=6`) :

   | | asymétrie relative de `B∘A` |
   |---|---|
   | LFO, `V` indépendant | **1.10** |
   | LNO, `V = W` | 0.00 |

   En LFO, le point fixe n'est donc le minimiseur d'**aucune** fonction convexe :
   il n'y a pas de solution vers laquelle converger. `ConvScCP_UNN` en LFO est un
   **réseau récurrent inspiré de Chambolle–Pock**, pas un algorithme déroulé.
   Aucun réglage de pas ne change ça — seul `V = W` le change.

   Conséquence directe pour la rédaction : la description « déroulé de ScCP
   résolvant `min f_z(x) + g(Wx)` » est exacte pour LNO et ne l'est pas pour LFO.
2. **Le déroulé est fini** (`K` itérations). Les garanties de CP sont asymptotiques ;
   elles disent que l'itération est bien posée, pas que `K = 10` suffit.
3. **L'objectif change d'une itération à l'autre — y compris en LNO.** Chaque
   couche a son propre `W_k` et son propre prox (rayon `r_k(t)` appris). La
   théorie suppose `f`, `g` et `K` **fixes** le long des itérations. Ici la suite
   des itérés ne résout pas un problème unique mais en enchaîne `K` différents,
   donc son point fixe n'est la solution d'aucun d'entre eux.

   C'est un écart indépendant du précédent : il subsiste même avec `V = W`. Le
   niveau réellement fondé serait LNO **plus** un `W` et un prox partagés entre
   itérations — ce que fait `SharedConvCP_UNN`, pas `ConvScCP_UNN`.

   Contrepartie utile : que le rayon soit appris **par itération** est ce qui
   permet au modèle d'encaisser la croissance de `sigma` d'un facteur ~150 le long
   du déroulé (cf. §6).
4. `mu(t)` diverge en `t -> 1`. Le code le protège par `clamp_min(1e-8)` sur
   `(1-t)^2`, inerte pour `t <= 1 - 1e-4`. Avec `t ~ U(0, t_max)` et
   `t_max = 0.95`, on reste très en deçà.
