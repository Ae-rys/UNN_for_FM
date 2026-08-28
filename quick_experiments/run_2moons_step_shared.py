# -*- coding: utf-8 -*-
"""
run_2moons_step_shared.py
Flow Matching on 2-moons with an auxiliary loss that encourages the unrolled
primal iterations x^[0], ..., x^[K] of a SharedDFB_UNN to take steps of equal
size:

    L_step = Var_k( ||x^[k+1] - x^[k]|| )   averaged over the batch

Total loss: L = L_v (FM loss) + lambda_step * L_step

By default this trains both a lambda_step=0 baseline and the regularized
model (same seed) so the two can be compared directly.

Usage
-----
    python run_2moons_step_shared.py [--epochs N] [--lambda-step L] [--results-dir DIR]
"""

import argparse
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.utils import torch_wrapper
from torchdyn.core import NeuralODE
from tqdm import tqdm

from models import SharedDFB_UNN
from run_2moons import (
    DIM, BATCH_SIZE, sample_8gaussians, get_moons_loader,
    save_scatter, save_overview, save_unn_paths, save_vector_field,
)
from run_2moons_step import step_uniformity_loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_2moons_step_shared(
    model,
    train_loader,
    device,
    results_dir,
    model_name,
    nb_epochs=100,
    lr=1e-3,
    lambda_step=1.0,
):
    run_dir = os.path.join(results_dir, model_name)
    os.makedirs(run_dir, exist_ok=True)

    ref_data = torch.cat([x for (x,) in train_loader])[:2000].numpy()

    FM        = ConditionalFlowMatcher(sigma=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}\nModel : {model_name}\nParams: {total_params:,}\nlambda_step: {lambda_step}\n{'='*60}")

    with open(os.path.join(run_dir, "params.txt"), "w") as f:
        f.write(f"{total_params}\n")

    loss_history, loss_v_history, loss_step_history = [], [], []
    train_start = time.perf_counter()
    eval_every  = max(1, nb_epochs // 5)

    for epoch in range(nb_epochs):
        model.train()
        total_loss, total_loss_v, total_loss_step = 0.0, 0.0, 0.0
        t0 = time.perf_counter()

        for (x1_batch,) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{nb_epochs}", leave=False):
            x1 = x1_batch.to(device)
            B  = x1.shape[0]
            x0 = sample_8gaussians(B, device=device)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            xt_t = torch.cat([xt, t.view(B, 1)], dim=-1)

            vt, x_traj, _ = model(xt_t, return_traj=True)

            loss_v    = torch.mean((vt - ut) ** 2)
            loss_step = step_uniformity_loss(x_traj)
            loss      = loss_v + lambda_step * loss_step

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss      += loss.item()
            total_loss_v    += loss_v.item()
            total_loss_step += loss_step.item()

        n_batches = len(train_loader)
        avg_loss, avg_loss_v, avg_loss_step = (
            total_loss / n_batches, total_loss_v / n_batches, total_loss_step / n_batches,
        )
        elapsed = time.perf_counter() - t0
        loss_history.append(avg_loss)
        loss_v_history.append(avg_loss_v)
        loss_step_history.append(avg_loss_step)
        print(f"  Epoch {epoch+1}/{nb_epochs} — loss: {avg_loss:.4f}  "
              f"(v: {avg_loss_v:.4f}, step: {avg_loss_step:.4f})  ({elapsed:.1f}s)")

        if (epoch + 1) % eval_every == 0 or epoch == nb_epochs - 1:
            model.eval()
            with torch.no_grad():
                x0_test = sample_8gaussians(2000, device=device)
                t_span  = torch.linspace(0, 1, 2, device=device)

                node = NeuralODE(torch_wrapper(model), solver="dopri5", atol=1e-4, rtol=1e-4)
                traj = node.trajectory(x0_test, t_span=t_span)
                gen  = traj[-1].cpu().numpy()
                save_scatter(
                    gen,
                    title=f"{model_name} — epoch {epoch+1}",
                    path=os.path.join(run_dir, f"epoch_{epoch+1}.png"),
                    ref=ref_data,
                )
                save_overview(
                    x0_test.cpu().numpy(), ref_data, gen,
                    title=f"{model_name} — epoch {epoch+1}",
                    path=os.path.join(run_dir, f"overview_epoch_{epoch+1}.png"),
                )
                save_unn_paths(
                    model, train_loader, device,
                    title=f"{model_name} — primal trajectory $x^{{[0..K]}}$ — epoch {epoch+1}",
                    path=os.path.join(run_dir, f"unn_paths_epoch_{epoch+1}.png"),
                )
                save_vector_field(
                    model, train_loader, device,
                    title=f"{model_name} — predicted velocity field $v_t(x)$ — epoch {epoch+1}",
                    path=os.path.join(run_dir, f"vector_field_epoch_{epoch+1}.png"),
                )
            model.train()

    # Loss curves
    plt.figure()
    plt.plot(range(1, nb_epochs + 1), loss_history,      marker="o", markersize=3, label="total")
    plt.plot(range(1, nb_epochs + 1), loss_v_history,    marker="o", markersize=3, label="v (FM)")
    plt.plot(range(1, nb_epochs + 1), loss_step_history, marker="o", markersize=3, label="step uniformity (Var_k)")
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.title(f"Training loss — {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss.png"))
    plt.close()

    with open(os.path.join(run_dir, "loss.txt"), "w") as f:
        f.write("epoch\ttotal\tloss_v\tloss_step\n")
        for ep, (l, lv, ls) in enumerate(zip(loss_history, loss_v_history, loss_step_history), 1):
            f.write(f"{ep}\t{l:.6f}\t{lv:.6f}\t{ls:.6f}\n")

    total_time = time.perf_counter() - train_start
    print(f"  Results saved to: {run_dir}")
    return loss_history, total_params, total_time, run_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FM on 2-moons with primal step-uniformity regularization (SharedDFB_UNN).")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--results-dir", type=str,   default="results_2moons_step_shared")
    parser.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lambda-step", type=float, default=1.0)
    parser.add_argument("--K",           type=int,   default=10)
    parser.add_argument("--w",           type=int,   default=32)
    parser.add_argument("--dual-dim",    type=int,   default=64)
    parser.add_argument("--version",     type=str,   default="LFO", choices=["LFO", "LNO"])
    parser.add_argument("--prox-type",   type=str,   default="l1", choices=["mlp", "l1"])
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--no-baseline", action="store_true",
                         help="skip training the lambda_step=0 baseline for comparison")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.results_dir, exist_ok=True)
    train_loader = get_moons_loader(batch_size=args.batch_size)

    configs = []
    if not args.no_baseline:
        configs.append((f"SharedDFB_UNN_{args.version}_step0", 0.0))
    configs.append((f"SharedDFB_UNN_{args.version}_step{args.lambda_step}", args.lambda_step))

    for model_name, lam in configs:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        model = SharedDFB_UNN(
            dim=DIM, K=args.K, w=args.w, dual_dim=args.dual_dim,
            version=args.version, prox_type=args.prox_type,
        ).to(device)

        train_2moons_step_shared(
            model        = model,
            train_loader = train_loader,
            device       = device,
            results_dir  = args.results_dir,
            model_name   = model_name,
            nb_epochs    = args.epochs,
            lambda_step  = lam,
        )


if __name__ == "__main__":
    main()
