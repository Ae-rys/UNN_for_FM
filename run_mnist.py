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
import os
import sys

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from models import (
    DiFB_UNN, DFB_UNN,
    ConvDFB_UNN, SharedConvDFB_UNN,
    CP_UNN, ConvCP_UNN, SharedConvCP_UNN,
    SmallUNet, small_MLP,
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
    alpha = torch.tensor(1.0)
    return [
        # ---- Baselines ----
        dict(
            name  = "MLP_baseline",
            build = lambda: small_MLP(dim=784, w=1024, time_varying=True).to(device),
            kwargs= {},
        ),
        dict(
            name  = "SmallUNet_baseline",
            build = lambda: SmallUNet(base_ch=64).to(device),
            kwargs= {},
        ),
        # ---- Standard UNNs ----
        dict(
            name  = "DiFB_UNN_LFO",
            build = lambda: DiFB_UNN(dim=784, K=15, w=512, version="LFO").to(device),
            kwargs= {},
        ),
        dict(
            name  = "DiFB_UNN_LNO",
            build = lambda: DiFB_UNN(dim=784, K=15, w=512, version="LNO").to(device),
            kwargs= {},
        ),
        dict(
            name  = "CP_UNN_LFO",
            build = lambda: CP_UNN(dim=784, K=3, w=1024, version="LFO").to(device),
            kwargs= {},
        ),
        dict(
            name  = "CP_UNN_LNO",
            build = lambda: CP_UNN(dim=784, K=3, w=1024, version="LNO").to(device),
            kwargs= {},
        ),
        # ---- Convolutional UNNs ----
        dict(
            name  = "ConvDFB_UNN_LFO",
            build = lambda: ConvDFB_UNN(dim=784, K=10, internal_channel=64, version="LFO").to(device),
            kwargs= {},
        ),
        dict(
            name  = "ConvDFB_UNN_LNO",
            build = lambda: ConvDFB_UNN(dim=784, K=10, internal_channel=64, version="LNO").to(device),
            kwargs= {},
        ),
        dict(
            name  = "ConvCP_UNN_LFO",
            build = lambda: ConvCP_UNN(dim=784, K=3, internal_channels=64, version="LFO").to(device),
            kwargs= {},
        ),
        dict(
            name  = "ConvCP_UNN_LNO",
            build = lambda: ConvCP_UNN(dim=784, K=3, internal_channels=64, version="LNO").to(device),
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
        # ---- Shared ConvCP ----
        dict(
            name  = "SharedConvCP_UNet_LNO_rand",
            build = lambda: SharedConvCP_UNN(dim=784, K=5, internal_channel=64, use_Unet=True,  version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedConvCP_UNet_LFO_rand",
            build = lambda: SharedConvCP_UNN(dim=784, K=5, internal_channel=64, use_Unet=True,  version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedConvCP_UNet_LFO_fixed",
            build = lambda: SharedConvCP_UNN(dim=784, K=5, internal_channel=64, use_Unet=True,  version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        dict(
            name  = "SharedConvCP_UNet_LNO_fixed",
            build = lambda: SharedConvCP_UNN(dim=784, K=5, internal_channel=64, use_Unet=True,  version="LNO").to(device),
            kwargs= dict(randomized_layer_nb=False, multi_iter=True),
        ),
        dict(
            name  = "SharedConvCP_DCT_LFO_rand",
            build = lambda: SharedConvCP_UNN(dim=784, K=5, internal_channel=64, use_Unet=False, version="LFO").to(device),
            kwargs= dict(randomized_layer_nb=True, multi_iter=True),
        ),
        dict(
            name  = "SharedConvCP_DCT_LNO_rand",
            build = lambda: SharedConvCP_UNN(dim=784, K=5, internal_channel=64, use_Unet=False, version="LNO").to(device),
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
            losses = train_mnist(
                model        = model,
                train_loader = train_loader,
                device       = device,
                results_dir  = args.results_dir,
                model_name   = name,
                nb_epochs    = args.epochs,
                **exp.get("kwargs", {}),
            )
            summary.append((name, losses[-1], "OK"))
        except Exception as exc:
            print(f"  ERROR: {exc}")
            summary.append((name, float("nan"), str(exc)))

    # ---- Final summary ----
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, final_loss, status in summary:
        print(f"  {name:<45} final_loss={final_loss:.4f}  [{status}]")

    summary_path = os.path.join(args.results_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("model_name\tfinal_loss\tstatus\n")
        for name, final_loss, status in summary:
            f.write(f"{name}\t{final_loss:.6f}\t{status}\n")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
