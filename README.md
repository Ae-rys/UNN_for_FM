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
| DiFB_UNN                  | DiFB         | MLP          | LFO/LNO | No     |
| CP_UNN_                   | CP           | MLP          | LFO/LNO | No     |
| ConvDFB_UNN               | ConvDFB      | DoubleConvT  | LFO/LNO | No     |
| ConvCP_UNN                | ConvCP       | DoubleConvT  | LFO/LNO | No     |
| SharedConvDFB             | ConvDFB      | UNet / DCT   | LFO/LNO | Yes    |
| SharedConvCP              | ConvCP       | UNet / DCT   | LFO/LNO | Yes    |

`LFO` = Learned Forward Operator (W and V are independent learned parameters).  
`LNO` = Learned Normalized Operator (V = W, step size fixed by spectral norm).  
`rand` = iteration count randomised U[5,15] during training.  
`fixed` = fixed iteration count K=5.
