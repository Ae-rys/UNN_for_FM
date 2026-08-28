# -*- coding: utf-8 -*-
"""
run_2moons_dual.py
Quick experiment: Flow Matching on 2-moons (dim=2, source = 8 Gaussians) where the
DFB_UNN's internal dual variable u is also trained to follow a linear interpolation
u_t = (1-t) * u_0 + t * u_1, with u_0 / u_1 obtained by passing x0 / x1 through the DFB.

Usage
-----
    python run_2moons_dual.py [--epochs N] [--results-dir DIR] [--lambda-u L]
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

from models import DFB_UNN
from run_2moons import DIM, BATCH_SIZE, sample_8gaussians, get_moons_loader, save_scatter, save_overview


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_2moons_dual(
    model,
    train_loader,
    device,
    results_dir,
    model_name,
    nb_epochs=100,
    lr=1e-3,
    lambda_u=1.0,
):
    run_dir = os.path.join(results_dir, model_name)
    os.makedirs(run_dir, exist_ok=True)

    ref_data = torch.cat([x for (x,) in train_loader])[:2000].numpy()

    FM        = ConditionalFlowMatcher(sigma=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}\nModel : {model_name}\nParams: {total_params:,}\n{'='*60}")

    with open(os.path.join(run_dir, "params.txt"), "w") as f:
        f.write(f"{total_params}\n")

    loss_history   = []
    loss_v_history = []
    loss_u_history = []
    train_start    = time.perf_counter()
    eval_every     = max(1, nb_epochs // 5)

    for epoch in range(nb_epochs):
        model.train()
        total_loss, total_loss_v, total_loss_u = 0.0, 0.0, 0.0
        t0 = time.perf_counter()

        for (x1_batch,) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{nb_epochs}", leave=False):
            x1 = x1_batch.to(device)
            B  = x1.shape[0]
            x0 = sample_8gaussians(B, device=device)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            t = t.view(B, 1)

            # u_0, u_1: dual variables obtained by running the DFB on x0 (t=0) and x1 (t=1)
            zeros_t = torch.zeros(B, 1, device=device)
            ones_t  = torch.ones(B, 1, device=device)
            with torch.no_grad():
                _, u0 = model(torch.cat([x0, zeros_t], dim=-1), return_u=True)
                _, u1 = model(torch.cat([x1, ones_t], dim=-1), return_u=True)
            u_target = (1 - t) * u0 + t * u1
            # print("u_target:", u_target)

            xt_t = torch.cat([xt, t], dim=-1)
            vt, u_pred = model(xt_t, return_u=True)

            loss_v = torch.mean((vt - ut) ** 2)
            loss_u = torch.mean(((u_target - u_pred ) - (u1 - u0)) ** 2)
            loss   = loss_v + lambda_u * loss_u

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss   += loss.item()
            total_loss_v += loss_v.item()
            total_loss_u += loss_u.item()

        n_batches = len(train_loader)
        avg_loss, avg_loss_v, avg_loss_u = (
            total_loss / n_batches, total_loss_v / n_batches, total_loss_u / n_batches,
        )
        elapsed = time.perf_counter() - t0
        loss_history.append(avg_loss)
        loss_v_history.append(avg_loss_v)
        loss_u_history.append(avg_loss_u)
        print(f"  Epoch {epoch+1}/{nb_epochs} — loss: {avg_loss:.4f}  "
              f"(v: {avg_loss_v:.4f}, u: {avg_loss_u:.4f})  ({elapsed:.1f}s)")

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
            model.train()

    # Loss curves
    plt.figure()
    plt.plot(range(1, nb_epochs + 1), loss_history,   marker="o", markersize=3, label="total")
    plt.plot(range(1, nb_epochs + 1), loss_v_history, marker="o", markersize=3, label="v (FM)")
    plt.plot(range(1, nb_epochs + 1), loss_u_history, marker="o", markersize=3, label="u (dual interp.)")
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.title(f"Training loss — {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss.png"))
    plt.close()

    with open(os.path.join(run_dir, "loss.txt"), "w") as f:
        f.write("epoch\ttotal\tloss_v\tloss_u\n")
        for ep, (l, lv, lu) in enumerate(zip(loss_history, loss_v_history, loss_u_history), 1):
            f.write(f"{ep}\t{l:.6f}\t{lv:.6f}\t{lu:.6f}\n")

    total_time = time.perf_counter() - train_start
    print(f"  Results saved to: {run_dir}")
    return loss_history, total_params, total_time


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FM on 2-moons with auxiliary dual-variable interpolation loss.")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--results-dir", type=str,   default="results_2moons_dual")
    parser.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lambda-u",    type=float, default=1.0)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.results_dir, exist_ok=True)
    train_loader = get_moons_loader(batch_size=args.batch_size)

    model = DFB_UNN(dim=DIM, K=10, dual_dim=64, version="LFO", learned_prox=False).to(device)

    train_2moons_dual(
        model        = model,
        train_loader = train_loader,
        device       = device,
        results_dir  = args.results_dir,
        model_name   = "DFB_UNN_LFO_dual",
        nb_epochs    = args.epochs,
        lambda_u     = args.lambda_u,
    )


if __name__ == "__main__":
    main()
