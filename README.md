# UNN for Flow Matching — MNIST Benchmark

Refactored from the original Colab notebook. All MNIST experiments run
sequentially from a single entry point.

## Structure

```
unn_mnist/
├── models/
│   ├── __init__.py
│   └── architectures.py   ← all model definitions (proximal blocks + UNNs)
├── train.py               ← training loop (saves checkpoints, images, losses)
├── run_mnist.py           ← main script — loops over every experiment
├── results/               ← created at runtime
│   └── <model_name>/
│       ├── epoch_N.pt          (checkpoint)
│       ├── epoch_N_t0.png      (generated images at t=0)
│       ├── epoch_N_t1.png      (generated images at t=1)
│       ├── epoch_N_niter_K.png (multi-iter models)
│       ├── loss.png            (loss curve)
│       └── loss.txt            (epoch \t loss, tab-separated)
└── summary.txt            ← final loss + status for every run
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

| Name                      | Family       | Prox         | Version | Shared |
|---------------------------|--------------|--------------|---------|--------|
| MLP_baseline              | MLP          | —            | —       | —      |
| SmallUNet_baseline        | UNet         | —            | —       | —      |
| DiFB_UNN_LFO/LNO          | DiFB         | MLP          | LFO/LNO | No    |
| CP_UNN_LFO/LNO            | CP           | MLP          | LFO/LNO | No    |
| ConvDFB_UNN_LFO/LNO       | ConvDFB      | DoubleConvT  | LFO/LNO | No    |
| ConvCP_UNN_LFO/LNO        | ConvCP       | DoubleConvT  | LFO/LNO | No    |
| SharedConvDFB_UNet_*/DCT_*| ConvDFB      | UNet / DCT   | LFO/LNO | Yes   |
| SharedConvCP_UNet_*/DCT_* | ConvCP       | UNet / DCT   | LFO/LNO | Yes   |

`LFO` = Learned Forward Operator (W and V are independent learned parameters).  
`LNO` = Learned Normalized Operator (V = W, step size fixed by spectral norm).  
`rand` = iteration count randomised U[5,15] during training.  
`fixed` = fixed iteration count K=5.

## Bugs fixed vs. original notebook

1. `model.get_model` → `model.__class__.__name__` (attribute did not exist).
2. `SharedConvCP_UNN.__init__`: `internal_channels` → `internal_channel`
   (undefined variable in the `torch.randn(...)` call for `shared_V`).
3. `SharedConvCP_UNN.forward`: `t_scaled` → `t`
   (variable was never defined; used in both the UNet and DoubleConvTime branches).
4. `CP_Iteration` (LNO branch): `spectral_norm(W_weight)` → `spectral_norm(self.W_weight)`
   (`W_weight` is not a local variable in `__init__`).
5. `ConvCP_Iteration` (LNO branch): same fix as above.
