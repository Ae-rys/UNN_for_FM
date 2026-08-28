# -*- coding: utf-8 -*-
"""
train.py
Training loop for Flow Matching on MNIST.
"""

import math
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
import torch._inductor.config as ind


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


def write_param_file(model, run_dir):
    """Write "parametres.txt" in run_dir with the architecture hyperparameters
    of `model`, read directly off its attributes. Mirrors run_2moons.py's
    write_param_file so make_grille.py's K x dual_dim grid also works on
    MNIST results — `internal_channel` (the MNIST equivalent of dual_dim,
    the conv UNNs' internal channel count) is stored under the "dual_dim"
    key for that reason.
    """
    fields = {
        "model_class": type(model).__name__,
        "K":           getattr(model, "K", None),
        "dual_dim":    getattr(model, "internal_channel", None),
        "version":     getattr(model, "version", None),
    }
    with open(os.path.join(run_dir, "parametres.txt"), "w") as f:
        for key, value in fields.items():
            if value is not None:
                f.write(f"{key}={value}\n")


# =========================================================================== #
#  (B) train_mnist modifié
# =========================================================================== #
def train_mnist(
    model,
    train_loader,
    device,
    results_dir,
    model_name,
    nb_epochs=5,
    lr=1e-2,
    randomized_layer_nb=False,
    multi_iter=False,
    ae=None,                       # <-- NOUVEAU : autoencodeur gelé (None = espace image)
    lr_schedule="constant",        # "constant" (defaut) | "cosine" (warmup+decay) | "kamb" (exp par step)
    warmup_frac=0.05,              # fraction des steps totaux en warmup lineaire (si lr_schedule="cosine")
    min_lr_ratio=0.0,              # LR final = min_lr_ratio * lr (plancher du cosinus)
    exp_gamma=0.999965,            # facteur multiplicatif par step pour lr_schedule="kamb" (Kamb & Ganguli App C.1)
    coupling="indep",              # "indep" (ConditionalFlowMatcher, defaut) | "ot" (ExactOptimalTransport)
    save_model=False,              # si True : sauvegarde run_dir/model.pt (state_dict)
    x1_weight="invsq",             # pondération loss x1-pred : "invsq" (legacy 1/(1-t)², EXPLOSE) |
                                   # "uniform" (MSE simple, borné, casse le collapse) | "minsnr" (1/(1-t)² plafonné à gamma)
    min_snr_gamma=5.0,             # plafond du poids pour x1_weight="minsnr"
):
    """
    Si `ae` est fourni : Flow Matching DANS LE LATENT.
      - x1 est encodé une fois (ae.encode, gelé), le bruit x0 est latent,
        la loss est latente.
      - la génération intègre l'ODE dans le latent puis décode (ae.decode)
        seulement pour sauvegarder les images.
    Si `ae is None` : comportement image d'origine (inchangé).
    """
    run_dir = os.path.join(results_dir, model_name)
    os.makedirs(run_dir, exist_ok=True)
    write_param_file(model, run_dir)                       # défini ailleurs

    # model = torch.compile(model)

    latent = ae is not None
    if latent:
        ae = ae.to(device).eval()
        for p in ae.parameters():
            p.requires_grad_(False)
        c_lat = ae.c_lat
        Hl    = ae.latent_spatial
        lat_dim = c_lat * Hl * Hl                          # = model.dim (196 par défaut)

    FM        = (ExactOptimalTransportConditionalFlowMatcher(sigma=0.1) if coupling == "ot"
                 else ConditionalFlowMatcher(sigma=0.1))
    print(f"Coupling    : {coupling}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -------- learning-rate schedule (par step de descente de gradient) --------
    total_steps  = nb_epochs * len(train_loader)
    warmup_steps = int(warmup_frac * total_steps)
    def lr_lambda(step):
        if lr_schedule == "constant":
            return 1.0
        if lr_schedule == "kamb":                     # exp decay par step : lr * gamma^step
            return exp_gamma ** step
        if step < warmup_steps:                       # warmup lineaire 0 -> 1
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)   # 0 -> 1
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))                     # 1 -> 0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if lr_schedule == "kamb":
        _hl = math.log(0.5) / math.log(exp_gamma) / max(1, len(train_loader))   # epochs pour /2
        print(f"LR schedule : kamb (exp gamma={exp_gamma} par step, lr0={lr:.1e}, "
              f"demi-vie ~{_hl:.0f} epochs)")
    else:
        print(f"LR schedule : {lr_schedule}" + (f" (warmup {warmup_steps}/{total_steps} steps, "
              f"min_lr={min_lr_ratio*lr:.2e})" if lr_schedule != "constant" else ""))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}\nModel : {model_name}\nParams: {total_params:,}\n{'='*60}")
    with open(os.path.join(run_dir, "params.txt"), "w") as f:
        f.write(f"{total_params}\n")

    loss_history = []
    train_start  = time.perf_counter()

    for epoch in range(nb_epochs):
        model.train()
        total_loss = 0.0
        t0 = time.perf_counter()

        for x1_batch, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{nb_epochs}", leave=False):
            batch_size = x1_batch.size(0)

            # -------- cible x1 : latente (encodée, gelée) ou image --------
            if latent:
                x1_img = x1_batch.to(device).view(batch_size, 1, 28, 28)
                with torch.no_grad():
                    x1 = ae.encode(x1_img).flatten(1)      # (B, lat_dim)
            elif model_name == "UNet_torchCFM_baseline":
                x1 = x1_batch.to(device)
            else:
                x1 = x1_batch.to(device).view(batch_size, -1)

            x0 = torch.randn_like(x1)                      # bruit dans le bon espace
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

            # -------- forward --------
            if randomized_layer_nb and not latent:
                xt_t = torch.cat([xt, t.view(batch_size, 1)], dim=-1)
                out  = model(xt_t, n_iter=random.randint(5, 15))
            elif model_name == "UNet_torchCFM_baseline" and not latent:
                out = model(t, xt)
            else:
                xt_t = torch.cat([xt, t.view(batch_size, 1)], dim=-1)
                out  = model(xt_t)

            # -------- loss --------
            if getattr(model, "predicts_x1", False):
                # cible = x1 ; pondération sur ‖g-x1‖² (t remis en (B,1) pour broadcaster)
                omt2 = (1 - t.view(-1, 1)) ** 2
                if x1_weight == "uniform":
                    w = 1.0                                              # MSE simple, borné
                elif x1_weight == "minsnr":
                    w = torch.clamp(1.0 / omt2, max=min_snr_gamma)       # poids vitesse plafonné à gamma
                else:  # "invsq" : legacy 1/(1-t)² (EXPLOSE près de t=1)
                    w = 1.0 / torch.clamp(omt2, min=0.000005 ** 2)
                loss = torch.mean(w * (out - x1) ** 2)
            else:
                loss = torch.mean((out - ut) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1}/{nb_epochs} — loss: {avg_loss:.4f}  lr: {cur_lr:.2e}  ({time.perf_counter()-t0:.1f}s)")

        # ---- génération périodique ----
        if epoch % 2 == 0 and epoch > 40:
            model.eval()
            with torch.no_grad():
                if latent:
                    # bruit latent -> intégration ODE dans le latent -> décodage
                    x0_test = torch.randn(10, lat_dim, device=device)
                    node = NeuralODE(torch_wrapper(model), solver="dopri5",
                                     atol=1e-5, rtol=1e-5)
                    traj = node.trajectory(x0_test, t_span=torch.linspace(0, 1, 2, device=device))
                    lat_final = traj[-1].view(10, c_lat, Hl, Hl)
                    imgs = ae.decode(lat_final)            # (10,1,28,28)
                    plot_images(imgs, title=f"{model_name} — epoch {epoch+1}",
                                save_path=os.path.join(run_dir, f"epoch_{epoch+1}.png"))
                elif model_name == "UNet_torchCFM_baseline":
                    x0_test = torch.randn(10, 1, 28, 28, device=device)
                    class _W(torch.nn.Module):
                        def __init__(s, m): super().__init__(); s.m = m
                        def forward(s, t, x, **kw): return s.m(t.expand(x.shape[0]), x)
                    node = NeuralODE(_W(model), solver="dopri5", atol=1e-5, rtol=1e-5)
                    traj = node.trajectory(x0_test, t_span=torch.linspace(0, 1, 2, device=device))
                    plot_images(traj[-1], title=f"{model_name} — epoch {epoch+1}",
                                save_path=os.path.join(run_dir, f"epoch_{epoch+1}.png"))
                else:
                    x0_test = torch.randn(10, 784, device=device)
                    node = NeuralODE(torch_wrapper(model), solver="dopri5", atol=1e-5, rtol=1e-5)
                    traj = node.trajectory(x0_test, t_span=torch.linspace(0, 1, 2, device=device))
                    plot_images(traj[-1], title=f"{model_name} — epoch {epoch+1}",
                                save_path=os.path.join(run_dir, f"epoch_{epoch+1}.png"))
            model.train()

    # ---- courbes / log ----
    plt.figure(); plt.plot(range(1, nb_epochs + 1), loss_history, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Average FM loss"); plt.title(f"Training loss — {model_name}")
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "loss.png")); plt.close()
    with open(os.path.join(run_dir, "loss.txt"), "w") as f:
        for ep, l in enumerate(loss_history, 1):
            f.write(f"{ep}\t{l:.6f}\n")

    # ---- checkpoint : poids du modele (meme convention que run_2moons -> make_grille) ----
    if save_model:
        torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    print(f"  Results saved to: {run_dir}")
    return loss_history, total_params, time.perf_counter() - train_start













































# def train_mnist(
#     model,
#     train_loader,
#     device,
#     results_dir,
#     model_name,
#     nb_epochs=5,
#     lr=1e-3,
#     randomized_layer_nb=False,
#     multi_iter=False,
# ):
#     """
#     Train `model` with Flow Matching on MNIST and save results.

#     Parameters
#     ----------
#     model            : nn.Module  — the model to train
#     train_loader     : DataLoader — MNIST train loader
#     device           : str/device
#     results_dir      : str        — root directory where outputs are stored
#     model_name       : str        — used for sub-folder and checkpoint names
#     nb_epochs        : int
#     lr               : float
#     randomized_layer_nb : bool   — randomly draw n_iter ∈ [5, 15] at each batch
#     multi_iter       : bool       — generate with several iteration counts at eval time
#     """
#     run_dir = os.path.join(results_dir, model_name)
#     os.makedirs(run_dir, exist_ok=True)
#     write_param_file(model, run_dir)

#     # ind.layout_optimization = False

#     # model = torch.compile(model).to(device)

#     FM        = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"\n{'='*60}")
#     print(f"Model : {model_name}")
#     print(f"Params: {total_params:,}")
#     print(f"{'='*60}")

#     with open(os.path.join(run_dir, "params.txt"), "w") as f:
#         f.write(f"{total_params}\n")

#     loss_history = []
#     train_start = time.perf_counter()

#     for epoch in range(nb_epochs):
#         model.train()
#         total_loss = 0.0
#         t0 = time.perf_counter()

#         for x1_batch, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{nb_epochs}", leave=False):
#             batch_size = x1_batch.size(0)
#             if model_name == "UNet_torchCFM_baseline":
#                 x1 = x1_batch.to(device)
#             else:
#                 x1 = x1_batch.to(device).view(batch_size, -1)
#             x0 = torch.randn_like(x1)

#             t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

#             if randomized_layer_nb:
#                 xt_t = torch.cat([xt, t.view(batch_size, 1)], dim=-1)
#                 n_iter = random.randint(5, 15)
#                 out = model(xt_t, n_iter=n_iter)
#             elif model_name == "UNet_torchCFM_baseline":
#                 out = model(t, xt)
#             else:
#                 xt_t = torch.cat([xt, t.view(batch_size, 1)], dim=-1)
#                 out = model(xt_t)

#             # Models flagged `predicts_x1` output D_theta(x_t, t) ≈ x1 directly
#             # during training (the velocity division by (1-t) only happens at
#             # eval/generation time), so the loss target is x1, not the FM
#             # velocity ut.
#             if getattr(model, "predicts_x1", False):
#                 loss = torch.mean((out - x1) ** 2)
#             else:
#                 loss = torch.mean((out - ut) ** 2)
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             total_loss += loss.item()

#         avg_loss = total_loss / len(train_loader)
#         elapsed  = time.perf_counter() - t0
#         loss_history.append(avg_loss)
#         print(f"  Epoch {epoch+1}/{nb_epochs} — loss: {avg_loss:.4f}  ({elapsed:.1f}s)")

#         # ---- Periodic image  and model saving ----
#         if epoch % 2 == 0 and epoch > 50:
            
#             # checkpoint_path = os.path.join(run_dir, f"model_epoch_{epoch+1}.pt")
#             # torch.save(model.state_dict(), checkpoint_path)
            
#             model.eval()
#             with torch.no_grad():
#                 if model_name == "UNet_torchCFM_baseline":
#                     x0_test = torch.randn(10, 1, 28, 28, device=device)
#                 else:
#                     x0_test = torch.randn(10, 784, device=device)
#                 t_span  = torch.linspace(0, 1, 2, device=device)

#                 if multi_iter:
#                     for n_it in [5, 10, 20, 30]:
#                         ode_func = lambda t, x, ni=n_it, **kw: model(
#                             torch.cat([x, t.expand(x.shape[0], 1)], dim=-1),
#                             n_iter=ni,
#                         )
#                         node = NeuralODE(
#                             ode_func, solver="dopri5",
#                             atol=1e-5, rtol=1e-5,
#                         )
#                         traj = node.trajectory(x0_test, t_span=t_span)
#                         img_path = os.path.join(run_dir, f"epoch_{epoch+1}_niter_{n_it}.png")
#                         plot_images(traj[-1], title=f"{model_name} — epoch {epoch+1}, {n_it} iters", save_path=img_path)
#                 else:
#                     if model_name == "UNet_torchCFM_baseline":
#                         class _Wrapper(torch.nn.Module):
#                             def __init__(self, m): super().__init__(); self.m = m
#                             def forward(self, t, x, **kw): return self.m(t.expand(x.shape[0]), x)
#                         node = NeuralODE(_Wrapper(model), solver="dopri5", atol=1e-5, rtol=1e-5)
#                     else:
#                         node = NeuralODE(torch_wrapper(model), solver="dopri5", atol=1e-5, rtol=1e-5)
#                     traj = node.trajectory(x0_test, t_span=t_span)

#                     # Alias of the final (t1) frame as "epoch_{N}.png", matching
#                     # run_2moons.py's naming convention so make_grille.py's K x
#                     # dual_dim grid also works on MNIST results.
#                     final_path = os.path.join(run_dir, f"epoch_{epoch+1}.png")
#                     plot_images(traj[-1], title=f"{model_name} — epoch {epoch+1}", save_path=final_path)

#             model.train()

#     # ---- Save loss curve ----
#     loss_path = os.path.join(run_dir, "loss.png")
#     plt.figure()
#     plt.plot(range(1, nb_epochs + 1), loss_history, marker="o")
#     plt.xlabel("Epoch")
#     plt.ylabel("Average FM loss")
#     plt.title(f"Training loss — {model_name}")
#     plt.tight_layout()
#     plt.savefig(loss_path)
#     plt.close()

#     # ---- Save loss values as text ----
#     with open(os.path.join(run_dir, "loss.txt"), "w") as f:
#         for ep, l in enumerate(loss_history, 1):
#             f.write(f"{ep}\t{l:.6f}\n")

#     total_time = time.perf_counter() - train_start
#     print(f"  Results saved to: {run_dir}")
#     return loss_history, total_params, total_time
