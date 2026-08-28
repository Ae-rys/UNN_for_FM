# UNN for Flow Matching — MNIST Benchmark

## Structure

```
unn_mnist/
├── models/
│   ├── __init__.py
│   └── architectures.py   ← all model definitions (proximal blocks + UNNs)
├── train.py               ← training loop (saves checkpoints, images, losses)
├── run_mnist.py           ← main script — loops over every experiment
├── results/               ← created at runtime
│   └── <experiment_name>/
|   |    └── <model_name>/
|   |    │   ├── epoch_N.pt          (checkpoint)
|   |    │   ├── epoch_N_t0.png      (generated images at t=0)
|   |    │   ├── epoch_N_t1.png      (generated images at t=1)
|   |    │   ├── epoch_N_niter_K.png (multi-iter models)
|   |    │   ├── loss.png            (loss curve)
|   |    │   └── loss.txt            (epoch \t loss, tab-separated)
|   |    └── summary.txt            ← final loss + status for every run
```

## Usage

```bash
# All experiments, 5 epochs each (default)
python run_mnist.py

# Fewer epochs for a quick test
python run_mnist.py --epochs 2

# Only shared models
python run_mnist.py --only Shared

# Skip UNet-based models
python run_mnist.py --skip UNet

# Custom results directory
python run_mnist.py --results-dir /path/to/my/results
```

## Models included

### Baselines

| Name                    | Notes                              |
|-------------------------|------------------------------------|
| MLP_baseline            | Large MLP (w=1024), time-varying   |
| UNet_torchCFM_baseline  | torchcfm built-in UNet             |
| UNet_baseline           | Custom UNet (base_ch=32)           |
| SmallUNet_baseline      | Lighter UNet (base_ch=32)          |

### Standard UNNs (per-layer W, flat)

| Name          | Family | Prox | Version |
|---------------|--------|------|---------|
| DiFB_UNN      | DiFB   | MLP  | LFO/LNO |
| ScCP_UNN      | ScCP   | MLP  | LFO/LNO |

### Convolutional UNNs (per-layer W)

| Name          | Family  | Prox        | Version |
|---------------|---------|-------------|---------|
| ConvDFB_UNN   | ConvDFB | DoubleConvT | LFO/LNO |

### Shared flat UNNs (shared W, linear operator)

| Name          | Family | Prox | Version | Training |
|---------------|--------|------|---------|----------|
| SharedDFB_UNN | DFB    | MLP  | LFO/LNO | rand / fixed |

### Shared convolutional UNNs (shared W, conv operator)

| Name                    | Family  | Prox              | Version | Training     |
|-------------------------|---------|-------------------|---------|--------------|
| SharedConvDFB_DCT       | ConvDFB | DoubleConvT       | LFO/LNO | fixed        |
| SharedConvDFB_UNet      | ConvDFB | SmallUNet         | LFO/LNO | rand / fixed |
| SharedConvDFB_CFMUNet   | ConvDFB | torchcfm UNet     | LFO/LNO | rand / fixed |
| SharedConvScCP_DCT      | ScCP    | DoubleConvT       | LFO/LNO | rand         |
| SharedConvScCP_UNet     | ScCP    | SmallUNet         | LFO/LNO | rand / fixed |
| SharedConvScCP_CFMUNet  | ScCP    | torchcfm UNet     | LFO/LNO | rand / fixed |

**Shared models** tie W across all K iterations, enabling variable iteration count at inference: `model(xt_t, n_iter=N)`.

---

`LFO` = Learned Forward Operator — W and V are independent learned parameters, step size `tau` is learnable.  
`LNO` = Learned Normalized Operator — V = W (or Wᵀ for flat), step size fixed by spectral norm.  
`rand` = iteration count randomised U[5, 15] during training; evaluated at 5, 10, 20, 30 iterations.  
`fixed` = fixed iteration count K=5 throughout training and evaluation.  
`DCT` = DoubleConvTime prox (two conv layers with time injection).  
`ScCP` = Strongly-convex accelerated Chambolle-Pock (learnable momentum schedule).
