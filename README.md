# UNN for Flow Matching

**Optimisation-inspired network architectures for image generation with normalising flows.**

This repository studies **unrolled (unfolded) neural networks (UNNs)** — architectures obtained by
truncating a convergent proximal algorithm and learning its parameters end to end — as the
**velocity estimator of a Flow Matching (FM) generative model**.

The starting point is the equivalence between a single FM time step and a Gaussian denoising
problem. Since the signal-processing community has spent a decade building light, interpretable,
Lipschitz-certifiable *proximal denoisers*, the question is whether one of them can replace the
UNet backbone of a flow model, and at what cost in sample quality.

Internship of **Estéban Collas** (ENS de Lyon) supervised by **Audrey Repetti** (Heriot-Watt
University, Edinburgh). The full write-up is [`internship_report/main.pdf`](internship_report/main.pdf).

---

## What is in here

The main architecture is **`ConvScCP_UNN`**: an unrolled *strongly-convex Chambolle–Pock* network,
conditioned on the flow time `t`, run directly in pixel space. Each of its `K` unrolled iterations is

```
x_{k+1} = (x_k − τ_k V_k u_k + τ_k z) / (1 + τ_k)     # primal step, re-injects the observation z
y_{k+1} = x_{k+1} + α_k (x_{k+1} − x_k)               # extrapolation
u_{k+1} = prox(u_k + σ_k W_k y_{k+1}, t)              # dual step, time-conditioned prox
```

with `x_0 = z = x_t`, `u_0 = 0`, and the accelerated schedule `α_k = (1+2τ_k)^{-1/2}`,
`τ_{k+1} = α_k τ_k`, `σ_{k+1} = σ_k / α_k`.

Two consequences matter for the whole project:

* **Interpretability.** The primal iterate `x_k` never leaves image space (no pooling, no
  multiscale), so *every intermediate state of the network is a displayable image*. You can watch
  the denoising happen layer by layer, inside a single ODE step — something a UNet cannot show you.
* **Expressivity trade-off.** All channel mixing has to pass back through the one/three-channel
  primal image, and the receptive field only grows linearly in `K`. This is the suspected reason
  the model lags a UNet on harder data (see [Limitations](#limitations)).

## Headline results

All runs use OT couplings and the *x*-prediction / velocity-loss parameterisation.

| Dataset | Model | Config | #Params | Steps | Val. loss | FID / W2 |
|---|---|---|---:|---:|---:|---:|
| 2-moons | ScCP LNO | K=7, m=32 | 1 358 | 1M | 0.03 | 0.20 |
| 2-moons | ScCP LFO | K=7, m=32 | 1 800 | 1M | 0.03 | 0.20 |
| 2-moons | DFB LNO | K=7, m=32 | 1 351 | 1M | 0.03 | 0.20 |
| 2-moons | MLP baseline | — | 4 546 | 1M | 0.03 | 0.19 |
| MNIST | **ScCP LNO** | K=15, m=64 | 80 190 | 93 800 | 0.13 | **41.4** |
| MNIST | ScCP LFO | K=15, m=64 | 157 936 | 93 800 | 0.13 | 53.3 |
| MNIST | DFB LNO | K=15, m=64 | 80 175 | 93 800 | 0.21 | 275.2 |
| MNIST | DFB LFO | K=15, m=64 | 157 950 | 93 800 | 0.16 | 121.0 |
| MNIST | SmallUNet baseline | — | 176 705 | 93 800 | 0.12 | 27.6 |
| MNIST | *train vs. test reference* | 2k/2k imgs | — | — | — | *5.41* |
| AFHQ-32 | ScCP LFO | K=20, m=256 | 6 919 061 | 165k | n/a | 46.65 |
| AFHQ-32 | UNet (torchcfm) | ch=64 | 4 437 699 | 200k | 0.14 | 10.95 |
| AFHQ-32 | *train vs. test reference* | 3653/2k imgs | — | — | — | *7.67* |

The MNIST "FID" is a **mini-FID** computed in the feature space of a small MNIST classifier
(not InceptionV3) on 2 000 samples; its real-vs-real floor is ≈ 4, so gaps of a few units are noise.
2-moons uses a 2-Wasserstein distance instead.

**Read as:** ScCP generates convincingly on 2-moons and MNIST, gets close on AFHQ-32, and clearly
beats the DFB family at equal budget — but does not catch a UNet, and costs roughly **10× more
training time** for it.

---

## Installation

```bash
python -m venv ~/.venvs/unn && source ~/.venvs/unn/bin/activate
pip install torch torchvision torchdyn torchcfm matplotlib tqdm scikit-learn pot
```

There is no build step, test suite, or linter.

### `PYTHONPATH` (required)

The runnable entry points live at the repository root, but several modules they import
(`train.py`, `flops_utils.py`, `run_imagenet32.py`, …) currently live in `quick_experiments/`.
Until that is consolidated, export:

```bash
export PYTHONPATH="$PWD/quick_experiments:$PYTHONPATH"
```

Without it, every root script except `generate_digits.py` fails with `ModuleNotFoundError`.

## Quickstart

```bash
# 1. Two moons — proof of concept, CPU-friendly, minutes
python run_2moons.py --epochs 50 --results-dir results_2moons

# 2. MNIST — the main image experiment (digit 0 only by default)
python run_mnist.py --epochs 200 --K 15 --ic 64 --coupling ot --save-model \
                    --results-dir results/mnist_report
python run_mnist.py --only ConvScCP --epochs 2        # smoke test
python run_mnist.py --digit -1                        # full MNIST instead of a single class

# 3. AFHQ-cats 32×32 — RGB, the hardest setting reached
python quick_experiments/prepare_afhq_cats.py         # builds ./data/afhq_cat32_train.pt
python run_afhq32.py --only ConvScCP --steps 200 --results-dir /tmp/smoke   # ETA check, ~2 min
nohup python run_afhq32.py --only ConvScCP >> claude.log 2>&1 &             # the real run

# 4. Sample from a saved checkpoint
python generate_digits.py --ckpt results/mnist_report/ConvScCP_UNN_L1_LNO/model.pt --n 256
```

Every runner accepts `--only NAME` / `--skip NAME` (substring match on the experiment name) and
`--results-dir DIR`. The step-driven runners (`run_afhq32.py`, `run_cifar10_torchcfm_recipe.py`)
**resume automatically** from `latest.pt` after a crash — just re-issue the same command.

⚠️ `run_cifar10_torchcfm_recipe.py` reproduces the official torchcfm recipe at 400k steps: several
days of GPU for the UNet baseline, and far more for a ScCP. Check the ETA printed after the first
100 steps before committing.

## Repository layout

```
models/architectures.py    all model definitions (UNNs, prox operators, UNet baselines)
train.py *                 train_mnist() — the CFM training loop
flops_utils.py *           per-eval FLOP counting + wall-clock accounting

run_2moons.py              2-moons benchmark
run_mnist.py               MNIST benchmark
run_afhq32.py              AFHQ-cats 32×32 (RGB), torchcfm recipe
run_cifar10_torchcfm_recipe.py   CIFAR-10 / ImageNet-32 under the official recipe
generate_digits.py         sampling from a checkpoint (config auto-detected from the state_dict)

quick_experiments/         ~110 one-off analysis, ablation, diagnostic and figure scripts
results/                   run outputs (git-ignored)
internship_report/         LaTeX sources and the compiled report
```

`*` = currently under `quick_experiments/`, see the `PYTHONPATH` note above.

### Anatomy of a run directory

```
<results-dir>/<experiment_name>/
├── model.pt          weights (only with --save-model)
├── latest.pt         resumable checkpoint (step-driven runners)
├── loss.txt/.png     training curve
├── epoch_*.png       samples over training
├── trajectory/       per-layer intermediate states — the interpretability figures
├── params.txt        raw parameter count
└── parametres.txt    key=value: model_class, K, dual_dim, version,
                      velocity_flops, train_time_s, …
```

`velocity_flops` counts conv+matmul for **one** velocity evaluation at batch 1. Compare it across
architectures — do **not** convert it to time: ScCP's GPU efficiency is ~5× worse than its FLOP
count suggests, because it issues many small full-resolution kernels.

---

## Design notes

**Model families.** `ScCP_*` (strongly-convex Chambolle–Pock, primal + extrapolated + dual) and
`DFB_*`/`DiFB_*` (Dual Forward–Backward, the latter with Nesterov momentum on the dual). Each exists
in a flat variant (dense `W`, MLP prox, for 2-moons) and a `Conv*` variant (`conv2d` /
`conv_transpose2d`, 9×9 kernel with padding 4 so the transpose is the exact adjoint). Baselines:
`MLP_baseline`, `SmallUNet`, a custom `UNet`, and torchcfm's `UNetModel`.

**LFO vs. LNO.** `version="LFO"` (Learned Forward Operator) lets `W` and `V` be independent
learnable parameters with a learnable step size — a *mismatched adjoint*. `version="LNO"` (Learned
Normalized Operator) ties `V = Wᵀ` and fixes the step size from the spectral norm
(`σ_k = 0.99 (τ_k‖W_k‖²)⁻¹`, power iteration), preserving the convergence conditions. LNO is
cheaper in parameters and, on MNIST, also better.

**Proximity operators.** The default is the analytical ℓ1 dual prox (`L1ProxConv`): a per-channel
clipping with a *learned, time-dependent radius*. That keeps the only black box in the model down to
the linear analysis operator `W_k`. `DoubleConvTime`, `SmallUNet` and `SiLUProxConv` are available as
learned alternatives.

**x-prediction and the `t_max` domain.** The UNN's primal iterate is by construction an estimate of
the clean image, so the models are trained *x*-pred and the velocity is reconstructed as
`v = (x̂₁ − x_t)/(1 − t)`. Because `‖v − (x₁−x₀)‖² = ‖x̂₁ − x₁‖²/(1−t)²`, the *x*-pred loss weighted
by `1/(1−t)²` **is** the velocity MSE — provided no clamp bites. Hence `t ~ U(0, t_max)` with
`t_max = 1 − 1/N` (N = Euler steps at inference, default 20 → 0.95): an N-step Euler never evaluates
past `1 − 1/N`, so no useful `t` is dropped and the weight stays bounded. Two asserts enforce this
(in the loss, and in `fm_velocity_denom`). Consequences:

* `model.t_max = None` restores the historical `clamp(min=0.05)` so pre-`t_max` checkpoints reload identically.
* `t_max` is stored in the checkpoint; resuming with a different one is refused (`--allow-tmax-change`).
* **Sample a `t_max=T` run only with a solver truncated at T.** `dopri5` evaluates `t` arbitrarily
  close to 1 and leaves the learned domain. Verify with `quick_experiments/check_tmax_equivalence.py`.
* **Losses across parameterisations are not comparable** when a clamp is active — check the clamp
  threshold before reading two numbers side by side.

**Odd symmetry, and the `w_bias` fix.** With `u₀ = 0`, warm start `z = x_t`, and no convolutional
bias, every ScCP update is *linear* and the ℓ1 clipping satisfies `prox(−a) = −prox(a)`. By induction
the velocity field is **odd**: `v(−x_t, t) = −v(x_t, t)`, hence `gen(−x₀) = −gen(x₀)`. On normalised
MNIST, negating a digit inverts its colours, so the generator is forced to output a distribution
symmetric about 0 — about half the samples come out colour-inverted. The fix is a zero-initialised
per-dual-channel bias in the analysis operator (`w_bias`), i.e. penalising `g(W_k · − b)`: it breaks
the parity while preserving the proximal reading, and starts exactly at the symmetric operator.

---

## Limitations

* **Training cost.** ScCP is deeply sequential and trains ~10× slower than a comparable UNet
  (~50 h vs. ~5 h on AFHQ-32). This is a GPU-efficiency problem, not a FLOP problem.
* **Expressivity.** The primal variable appears *forced* to remain a denoised image, and the dual is
  only ever updated through a linear map of it — so channel mixing must round-trip through a
  few-channel image. Whether this is a real ceiling, or whether ScCP simply stores information in
  intermediate "sketch" images, is left open. Replacing `W_k, V_k` by a learned encoder/decoder pair
  would relax it, at the cost of the convergence guarantees.
* No attention, no batchnorm — unlike the torchcfm UNet used as the target baseline.

## Explored directions (negative results)

Documented so they are not re-run:

* **Latent-space FM.** Running the flow in a light convolutional autoencoder's latent space gave
  consistently blurrier digits and worse ScCP convergence. Likely cause: the ℓ1 prox is a meaningful
  prior on pixels or wavelets, not on dense entangled codes. It also moves credit for sample quality
  to the decoder. *Abandoned — all reported results are in pixel space.* A structured
  (wavelet / sparse-coding) latent would be the way to retry.
* **Weight sharing across the K iterations.** Enables a variable inference depth, but costs a large,
  consistent amount of final loss on 2-moons (shared stagnates above 2.1 vs. 1.75–1.95 per-layer) and
  does not recover on MNIST. *Abandoned.*
* **Asymmetric two-threshold dual prox** as an alternative parity fix. Works, but `w_bias` was kept:
  it acts on the input dependence rather than on a static threshold. *Dropped for simplicity.*
* **Warm-starting the dual `u` across ODE steps** (trained via the RIN self-conditioning trick).
  Something a UNet structurally cannot do — but the effect depends entirely on `K`: on MNIST it helps
  only when the model is layer-starved (K=6: mini-FID 86 → 64) and *hurts* otherwise
  (K=20: 24 → 57; K=3: 302 → 1059). It never beat simply running more iterations. *Not adopted.*

## Report

```bash
cd internship_report && pdflatex -shell-escape main && bibtex main && pdflatex -shell-escape main
```

> Collas, E. and Repetti, A. *Study of optimisation-inspired network architectures for image
> generation with normalising flows.* Internship report, ENS de Lyon / Heriot-Watt University, 2026.

Key references: Chambolle & Pock (2011); Le et al. (2024), *Unfolded proximal neural networks*;
Lipman et al. (2023), *Flow Matching*; Tong et al. (2024), *Improving and generalizing flow-based
generative models* (torchcfm); Jabri, Fleet & Chen (2023), *RIN*.
