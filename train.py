# -*- coding: utf-8 -*-
"""
train.py
Training loop for Flow Matching on MNIST.
"""

import os
import random
import time

import matplotlib
matplotlib.use("Agg")  # headless — save figures instead of displaying them
import matplotlib.pyplot as plt
import torch
from torchdyn.core import NeuralODE
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher
from torchcfm.utils import torch_wrapper
from tqdm import tqdm


def plot_images(images, title, save_path):
    """Save a strip of 5 generated images to disk."""
    n = 5
    images = images.cpu().view(-1, 28, 28)
    fig, axes = plt.subplots(1, n, figsize=(10, 2))
    fig.suptitle(title, fontsize=9)
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=80)
    plt.close(fig)


def train_mnist(
    model,
    train_loader,
    device,
    results_dir,
    model_name,
    nb_epochs=5,
    lr=1e-3,
    randomized_layer_nb=False,
    multi_iter=False,
):
    """
    Train `model` with Flow Matching on MNIST and save results.

    Parameters
    ----------
    model            : nn.Module  — the model to train
    train_loader     : DataLoader — MNIST train loader
    device           : str/device
    results_dir      : str        — root directory where outputs are stored
    model_name       : str        — used for sub-folder and checkpoint names
    nb_epochs        : int
    lr               : float
    randomized_layer_nb : bool   — randomly draw n_iter ∈ [5, 15] at each batch
    multi_iter       : bool       — generate with several iteration counts at eval time
    """
    run_dir = os.path.join(results_dir, model_name)
    os.makedirs(run_dir, exist_ok=True)
    
    #model = torch.compile(model).to(device)
    
    FM        = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Model : {model_name}")
    print(f"Params: {total_params:,}")
    print(f"{'='*60}")

    with open(os.path.join(run_dir, "params.txt"), "w") as f:
        f.write(f"{total_params}\n")

    loss_history = []
    train_start = time.perf_counter()

    for epoch in range(nb_epochs):
        model.train()
        total_loss = 0.0
        t0 = time.perf_counter()

        for x1_batch, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{nb_epochs}", leave=False):
            batch_size = x1_batch.size(0)
            if model_name == "UNet_torchCFM_baseline":
                x1 = x1_batch.to(device)
            else:
                x1 = x1_batch.to(device).view(batch_size, -1)
            x0 = torch.randn_like(x1)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

            if randomized_layer_nb:
                xt_t = torch.cat([xt, t.view(batch_size, 1)], dim=-1)
                n_iter = random.randint(5, 15)
                vt = model(xt_t, n_iter=n_iter)
            elif model_name == "UNet_torchCFM_baseline":
                vt = model(t, xt)
            else:
                xt_t = torch.cat([xt, t.view(batch_size, 1)], dim=-1)
                vt = model(xt_t)

            loss = torch.mean((vt - ut) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        elapsed  = time.perf_counter() - t0
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch+1}/{nb_epochs} — loss: {avg_loss:.4f}  ({elapsed:.1f}s)")

        # ---- Periodic image  and model saving ----
        if epoch % 2 == 0:
            
            # checkpoint_path = os.path.join(run_dir, f"model_epoch_{epoch+1}.pt")
            # torch.save(model.state_dict(), checkpoint_path)
            
            model.eval()
            with torch.no_grad():
                if model_name == "UNet_torchCFM_baseline":
                    x0_test = torch.randn(10, 1, 28, 28, device=device)
                else:
                    x0_test = torch.randn(10, 784, device=device)
                t_span  = torch.linspace(0, 1, 2, device=device)

                if multi_iter:
                    for n_it in [5, 10, 20, 30]:
                        ode_func = lambda t, x, ni=n_it, **kw: model(
                            torch.cat([x, t.expand(x.shape[0], 1)], dim=-1),
                            n_iter=ni,
                        )
                        node = NeuralODE(
                            ode_func, solver="dopri5",
                            atol=1e-3, rtol=1e-3,
                        )
                        traj = node.trajectory(x0_test, t_span=t_span)
                        img_path = os.path.join(run_dir, f"epoch_{epoch+1}_niter_{n_it}.png")
                        plot_images(traj[-1], title=f"{model_name} — epoch {epoch+1}, {n_it} iters", save_path=img_path)
                else:
                    if model_name == "UNet_torchCFM_baseline":
                        class _Wrapper(torch.nn.Module):
                            def __init__(self, m): super().__init__(); self.m = m
                            def forward(self, t, x, **kw): return self.m(t.expand(x.shape[0]), x)
                        node = NeuralODE(_Wrapper(model), solver="dopri5", atol=1e-4, rtol=1e-4)
                    else:
                        node = NeuralODE(torch_wrapper(model), solver="dopri5", atol=1e-4, rtol=1e-4)
                    traj = node.trajectory(x0_test, t_span=t_span)

                    for tag, frames in [("t0", traj[0]), ("t1", traj[-1])]:
                        img_path = os.path.join(run_dir, f"epoch_{epoch+1}_{tag}.png")
                        plot_images(frames, title=f"{model_name} — epoch {epoch+1} ({tag})", save_path=img_path)

            model.train()

    # ---- Save loss curve ----
    loss_path = os.path.join(run_dir, "loss.png")
    plt.figure()
    plt.plot(range(1, nb_epochs + 1), loss_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average FM loss")
    plt.title(f"Training loss — {model_name}")
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    # ---- Save loss values as text ----
    with open(os.path.join(run_dir, "loss.txt"), "w") as f:
        for ep, l in enumerate(loss_history, 1):
            f.write(f"{ep}\t{l:.6f}\n")

    total_time = time.perf_counter() - train_start
    print(f"  Results saved to: {run_dir}")
    return loss_history, total_params, total_time
