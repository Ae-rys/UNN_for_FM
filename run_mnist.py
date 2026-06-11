# -*- coding: utf-8 -*-
"""
run_mnist.py
Launch all MNIST Flow Matching experiments in sequence.

Usage
-----
    python run_mnist.py [--epochs N] [--results-dir DIR] [--skip PATTERN]

Each experiment is defined by a dictionary in EXPERIMENTS below.
Results (checkpoints, generated images, loss curves) are saved under
    <results_dir>/<experiment_name>/
"""

import argparse
import gc
import os
import sys

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchcfm.models.unet import UNetModel

from models import (
    DiFB_UNN, DFB_UNN,
    SharedDFB_UNN,
    ConvDFB_UNN, ConvDiFB_UNN, ConvScCP_UNN, SharedConvDFB_UNN,
    ScCP_UNN, SharedConvScCP_UNN,
    SmallUNet, small_MLP, UNet,
)
from train import train_mnist

# ---------------------------------------------------------------------------
# Experiment registry
# Each entry:
#   name         : str   — used as sub-folder name
#   build        : callable(device) → nn.Module
#   kwargs       : dict  — extra keyword arguments forwarded to train_mnist
# ---------------------------------------------------------------------------

def build_experiments(device):
    return [
        # ---- Baselines ----
        dict(
            name  = "MLP_baseline",
            build = lambda: small_MLP(dim=784, w=1024, time_varying=True).to(device),
            kwargs= {},
        ),
        dict(
            name  = "UNet_torchCFM_baseline",
            build = lambda: UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1).to(device),
            kwargs= {},
        ),
        dict(
            name  = "UNet_baseline",
            build = lambda: UNet(base_ch=32).to(device),
            kwargs= {},
        ),
        dict(
            name  = "SmallUNet_baseline",
            build = lambda: SmallUNet(base_ch=32).to(device),
            kwargs= {},
        ),
        # ---- Convolutional UNNs ----
        dict(
            name  = "ConvDFB_UNN_LFO",
            build = lambda: ConvDFB_UNN(dim=784, K=20, internal_channel=64, version="LFO").to(device),
            kwargs= {},
        ),
        dict(
            name  = "ConvDFB_UNN_LNO",
            build = lambda: ConvDFB_UNN(dim=784, K=20, internal_channel=64, version="LNO").to(device),
            kwargs= {},
        ),
        # ---- Convolutional UNNs + L1 prox (non-shared, per-layer W) ----
        dict(
            name  = "ConvDFB_UNN_L1_LFO",
            build = lambda: ConvDFB_UNN(dim=784, K=32, internal_channel=1024, use_Unet="l1", version="LFO", use_checkpoint=True).to(device),
            kwargs= {},
        ),
        dict(
            name  = "ConvDFB_UNN_L1_LNO",
            build = lambda: ConvDFB_UNN(dim=784, K=20, internal_channel=256, use_Unet="l1", version="LNO", use_checkpoint=True).to(device),
            kwargs= {},
        ),
        dict(
            name  = "ConvDiFB_UNN_L1_LFO",
            build = lambda: ConvDiFB_UNN(dim=784, K=20, internal_channel=256, use_Unet="l1", version="LFO").to(device),
            kwargs= {},
        ),
        dict(
            name  = "ConvDiFB_UNN_L1_LNO",
            build = lambda: ConvDiFB_UNN(dim=784, K=20, internal_channel=256, use_Unet="l1", version="LNO").to(device),
            kwargs= {},
        ),
        dict(
            name  = "ConvScCP_UNN_L1_LFO",
            build = lambda: ConvScCP_UNN(dim=784, K=20, internal_channel=512, use_Unet="l1", version="LFO").to(device),
            kwargs= {},
        ),
        dict(
            name  = "ConvScCP_UNN_L1_LNO",
            build = lambda: ConvScCP_UNN(dim=784, K=20, internal_channel=256, use_Unet="l1", version="LNO").to(device),
            kwargs= {},
        ),
        # ---- Shared ConvDFB ----
        dict(
            name  = "SharedConvDFB_UNet_LFO_rand",
            build = lambda: SharedConvDFB_UNN(dim=784, K=5, internal_channel=64, use_Unet=True,  version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedConvDFB_UNet_LNO_rand",
            build = lambda: SharedConvDFB_UNN(dim=784, K=5, internal_channel=64, use_Unet=True,  version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedConvDFB_UNet_LFO_fixed",
            build = lambda: SharedConvDFB_UNN(dim=784, K=5, internal_channel=64, use_Unet=True,  version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        dict(
            name  = "SharedConvDFB_UNet_LNO_fixed",
            build = lambda: SharedConvDFB_UNN(dim=784, K=5, internal_channel=64, use_Unet=True,  version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        dict(
            name  = "SharedConvDFB_DCT_LFO_fixed",
            build = lambda: SharedConvDFB_UNN(dim=784, K=5, internal_channel=64, use_Unet=False, version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        dict(
            name  = "SharedConvDFB_DCT_LNO_fixed",
            build = lambda: SharedConvDFB_UNN(dim=784, K=5, internal_channel=64, use_Unet=False, version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        # ---- ScCP (Accelerated Chambolle-Pock, strongly convex), Shared ConvDFB + torchcfm UNet prox ----
        dict(
            name  = "SharedConvDFB_CFMUNet_LFO_rand",
            build = lambda: SharedConvDFB_UNN(dim=784, K=5, internal_channel=64, use_Unet="cfm", version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedConvDFB_CFMUNet_LNO_rand",
            build = lambda: SharedConvDFB_UNN(dim=784, K=5, internal_channel=64, use_Unet="cfm", version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedConvDFB_CFMUNet_LFO_fixed",
            build = lambda: SharedConvDFB_UNN(dim=784, K=5, internal_channel=64, use_Unet="cfm", version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        dict(
            name  = "SharedConvDFB_CFMUNet_LNO_fixed",
            build = lambda: SharedConvDFB_UNN(dim=784, K=5, internal_channel=64, use_Unet="cfm", version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        # ---- Shared ConvDFB + L1 prox ----
        dict(
            name  = "SharedConvDFB_L1_LFO_rand",
            build = lambda: SharedConvDFB_UNN(dim=784, K=5, internal_channel=64, use_Unet="l1", version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedConvDFB_L1_LNO_rand",
            build = lambda: SharedConvDFB_UNN(dim=784, K=5, internal_channel=64, use_Unet="l1", version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedConvDFB_L1_LFO_fixed",
            build = lambda: SharedConvDFB_UNN(dim=784, K=30, internal_channel=128, use_Unet="l1", version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=False),
        ),
        dict(
            name  = "SharedConvDFB_L1_LNO_fixed",
            build = lambda: SharedConvDFB_UNN(dim=784, K=30, internal_channel=128, use_Unet="l1", version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        # ---- Shared ConvScCP + L1 prox ----
        dict(
            name  = "SharedConvScCP_L1_LFO_rand",
            build = lambda: SharedConvScCP_UNN(dim=784, K=30, internal_channel=128, use_Unet="l1", version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedConvScCP_L1_LNO_rand",
            build = lambda: SharedConvScCP_UNN(dim=784, K=30, internal_channel=128, use_Unet="l1", version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedConvScCP_L1_LFO_fixed",
            build = lambda: SharedConvScCP_UNN(dim=784, K=30, internal_channel=128, use_Unet="l1", version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        dict(
            name  = "SharedConvScCP_L1_LNO_fixed",
            build = lambda: SharedConvScCP_UNN(dim=784, K=30, internal_channel=128, use_Unet="l1", version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        # ---- Shared ConvScCP + torchcfm UNet prox ----
        dict(
            name  = "SharedConvScCP_CFMUNet_LFO_rand",
            build = lambda: SharedConvScCP_UNN(dim=784, K=5, internal_channel=64, use_Unet="cfm", version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedConvScCP_CFMUNet_LNO_rand",
            build = lambda: SharedConvScCP_UNN(dim=784, K=5, internal_channel=64, use_Unet="cfm", version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedConvScCP_CFMUNet_LFO_fixed",
            build = lambda: SharedConvScCP_UNN(dim=784, K=5, internal_channel=64, use_Unet="cfm", version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        dict(
            name  = "SharedConvScCP_CFMUNet_LNO_fixed",
            build = lambda: SharedConvScCP_UNN(dim=784, K=5, internal_channel=64, use_Unet="cfm", version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        # ---- Shared ConvScCP ----
        dict(
            name  = "SharedConvScCP_UNet_LFO_rand",
            build = lambda: SharedConvScCP_UNN(dim=784, K=6, internal_channel=64, use_Unet=True,  version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedConvScCP_UNet_LNO_rand",
            build = lambda: SharedConvScCP_UNN(dim=784, K=5, internal_channel=64, use_Unet=True,  version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedConvScCP_UNet_LFO_fixed",
            build = lambda: SharedConvScCP_UNN(dim=784, K=5, internal_channel=64, use_Unet=True,  version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        dict(
            name  = "SharedConvScCP_UNet_LNO_fixed",
            build = lambda: SharedConvScCP_UNN(dim=784, K=5, internal_channel=64, use_Unet=True,  version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        dict(
            name  = "SharedConvScCP_DCT_LFO_rand",
            build = lambda: SharedConvScCP_UNN(dim=784, K=5, internal_channel=64, use_Unet=False, version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedConvScCP_DCT_LNO_rand",
            build = lambda: SharedConvScCP_UNN(dim=784, K=5, internal_channel=64, use_Unet=False, version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
    ]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_train_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run all UNN-FM experiments on MNIST.")
    parser.add_argument("--epochs",      type=int,   default=5,         help="Number of training epochs per model.")
    parser.add_argument("--results-dir", type=str,   default="results", help="Root directory for outputs.")
    parser.add_argument("--skip",        type=str,   default="",        help="Skip experiments whose name contains this substring.")
    parser.add_argument("--only",        type=str,   default="",        help="Run only experiments whose name contains this substring.")
    parser.add_argument("--batch-size",  type=int,   default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.results_dir, exist_ok=True)
    train_loader = get_train_loader(batch_size=args.batch_size)

    experiments = build_experiments(device)

    # Filtering
    if args.only:
        experiments = [e for e in experiments if args.only in e["name"]]
    if args.skip:
        experiments = [e for e in experiments if args.skip not in e["name"]]

    print(f"\n{len(experiments)} experiment(s) to run.\n")

    summary = []
    for i, exp in enumerate(experiments, 1):
        name = exp["name"]
        print(f"\n[{i}/{len(experiments)}] {name}")
        try:
            model  = exp["build"]()
            losses, n_params, train_time = train_mnist(
                model        = model,
                train_loader = train_loader,
                device       = device,
                results_dir  = args.results_dir,
                model_name   = name,
                nb_epochs    = args.epochs,
                **exp.get("kwargs", {}),
            )
            summary.append((name, losses[-1], n_params, train_time, "OK"))
        except Exception as exc:
            print(f"  ERROR: {exc}")
            summary.append((name, float("nan"), 0, 0.0, str(exc)))
        finally:
            model = None
            gc.collect()
            torch.cuda.empty_cache()

    # ---- Final summary ----
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, final_loss, n_params, train_time, status in summary:
        print(f"  {name:<45} params={n_params:>10,}  final_loss={final_loss:.4f}  time={train_time:6.0f}s  [{status}]")

    summary_path = os.path.join(args.results_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("model_name\tn_params\tfinal_loss\ttrain_time_s\tstatus\n")
        for name, final_loss, n_params, train_time, status in summary:
            f.write(f"{name}\t{n_params}\t{final_loss:.6f}\t{train_time:.1f}\t{status}\n")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
