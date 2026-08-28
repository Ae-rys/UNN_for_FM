# -*- coding: utf-8 -*-
"""
gaussian_prox_experiment.py

Expérience de validation sur cas gaussien à gaussien.
Sous P0 = N(0,I) et P1 = N(mu, Sigma), le transport optimal est linéaire
et le champ de vecteurs exact est calculable en forme close.
On compare la convergence vers v_t^exact d'un UNN vs un MLP.
"""

from models.architectures import ScCP_UNN, sigma_max_power_iter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint

from models import DFB_UNN, CP_UNN, small_MLP

# ---------------------------------------------------------------------------
# 0. Shared DFB-UNN vectoriel (pour comparaison) : poids partagés sur toutes les itérations
# ---------------------------------------------------------------------------

class SharedDFB_UNN_vec(nn.Module):
    """
    Version vectorielle (non-image) du SharedConvDFB_UNN.
    Poids partagés sur toutes les itérations — discrétisation
    la plus fidèle de l'ODE continue de Fukumizu et al.
    """
    def __init__(self, dim, K=10, w=32, version="LFO"):
        super().__init__()
        self.dim     = dim
        self.K       = K
        self.version = version

        # Poids partagés
        self.shared_W = nn.Parameter(
            torch.eye(dim) + torch.randn(dim, dim) * 0.01
        )
        if version == "LFO":
            self.shared_V = nn.Parameter(
                torch.eye(dim) + torch.randn(dim, dim) * 0.01
            )
            self.tau = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer(
                '_sigma_u', F.normalize(torch.randn(dim), dim=0)
            )

        # Proximal partagé (time-varying)
        self.shared_prox = small_MLP(dim=dim, w=w, time_varying=True)

    def forward(self, xt_t, n_iter=None):
        iters = n_iter if n_iter is not None else self.K
        xt = xt_t[:, :self.dim]
        t  = xt_t[:, self.dim:]
        z  = xt
        u  = torch.zeros_like(xt)

        if self.version == "LFO":
            tau = F.softplus(self.tau)
            V   = self.shared_V
        else:
            _s, self._sigma_u = sigma_max_power_iter(
                self.shared_W, self._sigma_u
            )
            tau = 1.99 / _s ** 2
            V   = self.shared_W.T

        for _ in range(iters):
            x_next = z - F.linear(u, V, None)
            step   = tau * F.linear(x_next, self.shared_W, None)
            u      = self.shared_prox(torch.cat([u + step, t], dim=-1))

        return x_next - xt

# ---------------------------------------------------------------------------
# 1. Transport optimal gaussien : forme close
# ---------------------------------------------------------------------------

def compute_gaussian_OT_map(mu1: torch.Tensor, Sigma1: torch.Tensor):
    """
    Calcule la matrice A du transport optimal T(x) = A(x - 0) + mu1
    entre P0 = N(0, I_d) et P1 = N(mu1, Sigma1).
    
    La carte OT est T(x) = A @ x + mu1 où
        A = Sigma1^{1/2}  (cas P0 = N(0,I)).
    
    Retourne A, A_inv (pour le prox).
    """
    # Sigma1 = U D U^T  (eigendecomposition)
    eigvals, U = torch.linalg.eigh(Sigma1)
    # A = Sigma1^{1/2} = U D^{1/2} U^T
    D_sqrt = torch.diag(eigvals.clamp(min=1e-8).sqrt())
    A = U @ D_sqrt @ U.T
    A_inv = torch.linalg.inv(A)
    return A, A_inv


def brenier_potential_quadratic(A: torch.Tensor):
    """
    Pour T = A (linéaire), le potentiel de Brenier forward (P0->P1) est
        phi(x0) = (1/2) x0^T A x0
    (à une constante près).
    Le potentiel backward (P1->P0) est
        phi*(x1) = (1/2) x1^T A^{-1} x1.
    
    Retourne phi_star = potentiel de Brenier P1->P0.
    """
    # phi*(x1) = (1/2) ||A^{-1/2} x1||^2
    # son sous-différentiel est A^{-1} x1 = x0
    pass  # on n'en a pas besoin directement


def exact_vector_field(
    xt: torch.Tensor,
    t: torch.Tensor,
    mu1: torch.Tensor,
    A: torch.Tensor,
    A_inv: torch.Tensor,
) -> torch.Tensor:
    """
    Champ de vecteurs exact pour OT-CFM entre N(0,I) et N(mu1, Sigma1).
    
    Schedule standard : alpha_t = 1-t, beta_t = t.
    
    Sous couplage OT, x1 est déterminé par x_t :
        x_t = (1-t) x0 + t x1
        x0 = A^{-1}(x1 - mu1)   (transport OT backward)
    
    En substituant et résolvant pour x1 :
        x_t = (1-t) A^{-1}(x1 - mu1) + t x1
        x_t = [(1-t) A^{-1} + t I] x1 - (1-t) A^{-1} mu1
    
    Donc :
        x1 = M_t^{-1} (x_t + (1-t) A^{-1} mu1)
    avec M_t = (1-t) A^{-1} + t I.
    
    Et le champ de vecteurs est :
        v_t(x_t) = (x1 - x0) = x1 - A^{-1}(x1 - mu1).
    """
    t_col = t.view(-1, 1)
    d = xt.shape[1]
    I = torch.eye(d, device=xt.device, dtype=xt.dtype)
    
    # M_t = (1-t) A^{-1} + t I  — varie selon t dans le batch
    # Pour chaque sample i : M_t[i] = (1-t[i]) * A_inv + t[i] * I
    # On vectorise sur le batch
    
    v_t_list = []
    for i in range(xt.shape[0]):
        ti = t[i].item()
        M_t = (1 - ti) * A_inv + ti * I
        rhs = xt[i] + (1 - ti) * (A_inv @ mu1)
        x1_i = torch.linalg.solve(M_t, rhs)
        x0_i = A_inv @ (x1_i - mu1)
        v_t_list.append(x1_i - x0_i)
    
    return torch.stack(v_t_list, dim=0)


def exact_vector_field_batched(
    xt: torch.Tensor,
    t: torch.Tensor,
    mu1: torch.Tensor,
    A_inv: torch.Tensor,
) -> torch.Tensor:
    """
    Version batched et vectorisée de exact_vector_field.
    Exploite le fait que M_t est diagonalisable conjointement à A_inv.
    """
    t_col = t.view(-1, 1)          # (B, 1)
    d = xt.shape[1]
    I = torch.eye(d, device=xt.device, dtype=xt.dtype)

    # Eigendecomposition de A_inv (symétrique) : A_inv = U Λ U^T
    eigvals, U = torch.linalg.eigh(A_inv)  # eigvals : (d,), U : (d, d)

    # Dans la base propre, M_t est diagonal :
    # M_t = (1-t) A_inv + t I
    # => dans la base U : diag((1-t)*lambda_j + t)
    # Λ_t[i, j] = (1-t[i]) * eigvals[j] + t[i]
    Lambda_t = (1 - t_col) * eigvals.unsqueeze(0) + t_col  # (B, d)

    # rhs = x_t + (1-t) * A_inv @ mu1
    Ainv_mu1 = A_inv @ mu1  # (d,)
    rhs = xt + (1 - t_col) * Ainv_mu1.unsqueeze(0)  # (B, d)

    # Résolution : x1 = U (Λ_t^{-1}) U^T rhs
    rhs_rotated = rhs @ U          # (B, d)
    x1_rotated  = rhs_rotated / Lambda_t
    x1 = x1_rotated @ U.T         # (B, d)

    # x0 = A_inv @ (x1 - mu1)
    x0 = (x1 - mu1.unsqueeze(0)) @ A_inv.T  # (B, d)

    return x1 - x0   # v_t = x1 - x0


# ---------------------------------------------------------------------------
# 2. Prox exact pour le cas gaussien (pour référence)
# ---------------------------------------------------------------------------

def prox_exact_gaussian(
    xt: torch.Tensor,
    t: torch.Tensor,
    A_inv: torch.Tensor,
    mu1: torch.Tensor,
) -> torch.Tensor:
    """
    Calcule x1 = prox_{lambda_t phi}(z_t) exactement.
    
    z_t = x_t / t  (variable rescalée de Fukumizu et al.)
    lambda_t = (1-t) / t
    phi(x1) = (1/2)(x1 - mu1)^T A^{-1} (x1 - mu1)
    
    Condition 1er ordre :
        (lambda_t * A_inv + I) x1 = z_t + lambda_t * A_inv @ mu1
    """
    eps = 1e-6
    t_col = t.view(-1, 1).clamp(min=eps)
    lam   = (1 - t_col) / t_col           # (B, 1)  lambda_t
    z_t   = xt / t_col                    # (B, d)  z_t = x_t / t

    d = xt.shape[1]
    device = xt.device

    # Eigendecomposition de A_inv = U Λ U^T
    eigvals, U = torch.linalg.eigh(A_inv)  # eigvals : (d,), U : (d, d)

    # Dans la base propre :
    # (lambda_t * eigvals[j] + 1) * x1_rot[i,j] = z_rot[i,j] + lambda_t * eigvals[j] * mu1_rot[j]
    z_rot   = z_t @ U                           # (B, d)
    mu1_rot = mu1 @ U                           # (d,)
    Ainv_mu1_rot = eigvals * mu1_rot            # (d,)  = (A_inv @ mu1) rotated

    # coeff[i, j] = lam[i] * eigvals[j] + 1
    coeff   = lam * eigvals.unsqueeze(0) + 1.0  # (B, d)
    rhs_rot = z_rot + lam * Ainv_mu1_rot.unsqueeze(0)  # (B, d)

    x1_rot = rhs_rot / coeff                    # (B, d)
    x1     = x1_rot @ U.T                       # (B, d)
    return x1


# ---------------------------------------------------------------------------
# 3. Dataset : échantillons OT-CFM gaussien -> gaussien
# ---------------------------------------------------------------------------

def sample_OT_CFM_gaussian(
    batch_size: int,
    mu1: torch.Tensor,
    A: torch.Tensor,
    device: torch.device,
):
    """
    Echantillonne (x_t, t, v_t_exact) pour OT-CFM entre N(0,I) et N(mu1, Sigma1).
    
    Sous couplage OT : x1 = A @ x0 + mu1.
    """
    d = mu1.shape[0]
    x0 = torch.randn(batch_size, d, device=device)
    x1 = x0 @ A.T + mu1.unsqueeze(0)       # T(x0) = A x0 + mu1
    t  = torch.rand(batch_size, device=device)
    t_col = t.view(-1, 1)
    
    xt = (1 - t_col) * x0 + t_col * x1
    vt = x1 - x0                            # champ cible OT-CFM standard
    return xt, t, vt, x0, x1


# ---------------------------------------------------------------------------
# 4. Entraînement
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    mu1: torch.Tensor,
    A: torch.Tensor,
    A_inv: torch.Tensor,
    device: torch.device,
    n_iter: int = 10_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    log_every: int = 1000,
):
    """
    Entraîne `model` sur la loss CFM (MSE sur le champ de vecteurs).
    Retourne l'historique des losses et des erreurs vs v_exact.
    """
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Params: {total_params:,}")
    print(f"{'='*60}")
    
    loss_history  = []
    error_history = []   # ||v_theta - v_exact||^2 moyen

    for step in range(n_iter):
        optimizer.zero_grad()
        
        xt, t, vt_exact, _, _ = sample_OT_CFM_gaussian(batch_size, mu1, A, device)
        
        # Input du modèle : (xt, t) concaténés
        inp = torch.cat([xt, t.unsqueeze(1)], dim=1)
        
        # Prédiction
        vt_pred = model(inp)
        
        # Loss CFM = MSE sur v_t
        loss = F.mse_loss(vt_pred, vt_exact)
        loss.backward()
        optimizer.step()
        
        if step % log_every == 0 or step == n_iter - 1:
            with torch.no_grad():
                # Erreur vs champ exact (sur un batch de validation plus grand)
                xt_val, t_val, _, _, _ = sample_OT_CFM_gaussian(2048, mu1, A, device)
                v_exact_val = exact_vector_field_batched(xt_val, t_val, mu1, A_inv)
                inp_val = torch.cat([xt_val, t_val.unsqueeze(1)], dim=1)
                v_pred_val = model(inp_val)
                err = F.mse_loss(v_pred_val, v_exact_val).item()
                
            loss_history.append(loss.item())
            error_history.append(err)
            print(f"Step {step:5d} | CFM loss: {loss.item():.4e} | ||v_pred - v_exact||^2: {err:.4e}")
    
    return loss_history, error_history


# ---------------------------------------------------------------------------
# 5. Expérience principale
# ---------------------------------------------------------------------------

def run_experiment(d=10, n_iter=10_000, K_list=(3, 5, 10), device_str="cpu"):
    device = torch.device(device_str)
    
    # --- Définition du problème gaussien ---
    torch.manual_seed(42)
    mu1    = torch.randn((d,), device=device)
    mu1[:2] = torch.tensor([3.0, 2.0], device=device)
    # Sigma1 quelconque définie positive
    L      = torch.randn(d, d, device=device) * 0.5
    Sigma1 = L @ L.T + torch.eye(d, device=device)
    
    A, A_inv = compute_gaussian_OT_map(mu1, Sigma1)
    A     = A.to(device)
    A_inv = A_inv.to(device)
    mu1   = mu1.to(device)
    
    print(f"Problème : P0 = N(0,I_{d}), P1 = N(mu1, Sigma1)")
    print(f"A = {A.cpu().numpy()}")
    
    # --- Vérification numérique : le prox exact doit donner x1 ---
    print("\n--- Vérification du prox exact ---")
    xt_test, t_test, _, x0_test, x1_test = sample_OT_CFM_gaussian(100, mu1, A, device)
    x1_prox = prox_exact_gaussian(xt_test, t_test, A_inv, mu1)
    err_prox = (x1_prox - x1_test).norm() / x1_test.norm()
    print(f"Erreur relative prox exact vs x1 vrai : {err_prox:.2e}  (doit être ~0)")
    
    # --- Modèles à comparer ---
    results = {}
    
    # # MLP baseline
    # print("\n--- Entraînement MLP ---")
    # mlp = nn.Sequential(
    #     nn.Linear(d + 1, 64), nn.SiLU(),
    #     nn.Linear(64, 64),    nn.SiLU(),
    #     nn.Linear(64, d),
    # ).to(device)
    # results["MLP"] = train_model(mlp, mu1, A, A_inv, device, n_iter=n_iter)
    
    # # DFB-UNN pour différents K
    # for K in K_list:
    #     for version in ["LFO", "LNO"]:
    #         name = f"DFB_UNN_K{K}_{version}"
    #         print(f"\n--- Entraînement {name} ---")
    #         model = DFB_UNN(dim=d, K=K, version=version, w=32, learned_prox=True).to(device)
    #         results[name] = train_model(model, mu1, A, A_inv, device, n_iter=n_iter)
    
    # # CP-UNN pour K intermédiaire
    # for K in K_list:
    #     for version in ["LFO", "LNO"]:
    #         name = f"CP_UNN_K{K}_{version}"
    #         print(f"\n--- Entraînement {name} ---")
    #         model = CP_UNN(dim=d, K=K, version=version, w=32).to(device)
    #         results[name] = train_model(model, mu1, A, A_inv, device, n_iter=n_iter)
    
    # ScCP-UNN pour K intermédiaire
    for K in K_list:
        for version in ["LFO", "LNO"]:
            name = f"ScCP_UNN_K{K}_{version}"
            print(f"\n--- Entraînement {name} ---")
            model = ScCP_UNN(dim=d, K=K, version=version, w=32).to(device)
            results[name] = train_model(model, mu1, A, A_inv, device, n_iter=n_iter)

    return results


# ---------------------------------------------------------------------------
# 6. Visualisation
# ---------------------------------------------------------------------------

def _plot_K_ablation(K_list, final_errors, param_counts):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    styles = {
        "LNO":        ('o-',  "DFB-UNN LNO"),
        "LFO":        ('s-',  "DFB-UNN LFO"),
        "Shared_LNO": ('^-',  "Shared DFB LNO"),
        "Shared_LFO": ('v-',  "Shared DFB LFO"),
        "MLP_matched":('x--', "MLP (params égaux)"),
    }

    for key, (style, label) in styles.items():
        axes[0].semilogy(K_list,              final_errors[key], style, label=label)
        axes[1].semilogy(param_counts[key],   final_errors[key], style, label=label)

    axes[0].set_xlabel("K (nombre d'itérations)")
    axes[0].set_ylabel("$\\|v_\\theta - v^*\\|^2$ final")
    axes[0].set_title("Erreur vs $v^{exact}$ en fonction de K")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Nombre de paramètres")
    axes[1].set_ylabel("$\\|v_\\theta - v^*\\|^2$ final")
    axes[1].set_title("Erreur vs $v^{exact}$ en fonction du budget")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("K_ablation.png", dpi=150)
    plt.show()

def plot_results(results: dict, log_every: int = 1000, n_iter: int = 10_000):
    steps = np.arange(0, n_iter, log_every)
    if len(steps) < len(next(iter(results.values()))[1]):
        steps = np.append(steps, n_iter - 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for (name, (loss_hist, err_hist)), color in zip(results.items(), colors):
        x = steps[:len(err_hist)]
        axes[0].semilogy(x, loss_hist, label=name, color=color)
        axes[1].semilogy(x, err_hist,  label=name, color=color)
    
    axes[0].set_title("CFM Loss (MSE sur $v_t$)")
    axes[0].set_xlabel("Itération")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title("Erreur vs $v_t^{\\mathrm{exact}}$")
    axes[1].set_xlabel("Itération")
    axes[1].set_ylabel("$\\|v_\\theta - v^*\\|^2$")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("gaussian_prox_experiment.png", dpi=150)
    plt.show()
    print("Figure sauvegardée : gaussian_prox_experiment.png")


def plot_vector_fields(mu1, A, A_inv, model, device, title=""):
    """
    Visualise le champ de vecteurs exact vs prédit pour un t fixé.
    Utile pour d=2.
    """
    t_fixed = 0.5
    n_grid  = 15
    x_range = np.linspace(-3, 5, n_grid)
    y_range = np.linspace(-2, 4, n_grid)
    XX, YY  = np.meshgrid(x_range, y_range)
    grid    = torch.tensor(
        np.stack([XX.ravel(), YY.ravel()], axis=1), dtype=torch.float32, device=device
    )
    t_vec = torch.full((grid.shape[0],), t_fixed, device=device)
    
    # Champ exact
    with torch.no_grad():
        v_exact = exact_vector_field_batched(grid, t_vec, mu1, A_inv).cpu().numpy()
        inp     = torch.cat([grid, t_vec.unsqueeze(1)], dim=1)
        v_pred  = model(inp).cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, v, ttl in zip(axes, [v_exact, v_pred], ["$v_t^{exact}$", f"$v_\\theta$ {title}"]):
        ax.quiver(XX, YY, v[:, 0].reshape(n_grid, n_grid), v[:, 1].reshape(n_grid, n_grid),
                  scale=20, alpha=0.8)
        ax.set_title(f"{ttl}  (t={t_fixed})")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"vfield_{title.replace(' ','_')}.png", dpi=120)
    plt.show()


# ---------------------------------------------------------------------------
# 7. Expérience supplémentaire : erreur en fonction de K
# ---------------------------------------------------------------------------

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def make_mlp_with_budget(d, n_params_target, device):
    """
    Construit un MLP à 2 couches cachées avec ~n_params_target paramètres.
    (d+1)*w + w*w + w*d = n_params  =>  w^2 + w*(2d+1) - n_params = 0
    """
    # Résolution quadratique
    a = 1
    b = 2 * d + 1
    c = -n_params_target
    w = int((-b + np.sqrt(b**2 - 4*a*c)) / (2*a))
    w = max(w, 2)  # au moins 2 neurones
    mlp = nn.Sequential(
        nn.Linear(d + 1, w), nn.SiLU(),
        nn.Linear(w, w),     nn.SiLU(),
        nn.Linear(w, d),
    ).to(device)
    return mlp, count_params(mlp)

def K_ablation(mu1, A, A_inv, device, K_list=(1, 2, 3, 5, 10, 15), n_iter=5000):
    """
    Compare DFB-UNN LNO, LFO, Shared, et MLP à budget paramètres égal.
    """
    final_errors = {
        "LNO": [], "LFO": [],
        "Shared_LNO": [], "Shared_LFO": [],
        "MLP_matched": []
    }
    param_counts = {
        "LNO": [], "LFO": [],
        "Shared_LNO": [], "Shared_LFO": [],
        "MLP_matched": []
    }

    d = mu1.shape[0]

    for K in K_list:
        # --- DFB non-partagé ---
        for version in ["LNO", "LFO"]:
            model = DFB_UNN(dim=d, K=K, version=version).to(device)
            n_params = count_params(model)
            param_counts[version].append(n_params)
            _, err_hist = train_model(
                model, mu1, A, A_inv, device, n_iter=n_iter, log_every=n_iter
            )
            final_errors[version].append(err_hist[-1])
            print(f"K={K:2d} DFB {version:3s} ({n_params:5d} params): {err_hist[-1]:.4e}")

        # --- DFB partagé ---
        # Pour le cas 2D, on utilise DFB_UNN avec shared=True si tu l'implémentes,
        # ou on adapte SharedConvDFB_UNN pour dim général.
        # Ici on fait une version partagée simple adaptée au cas vectoriel (pas image).
        for version in ["LNO", "LFO"]:
            model = SharedDFB_UNN_vec(dim=d, K=K, version=version).to(device)
            n_params = count_params(model)
            param_counts[f"Shared_{version}"].append(n_params)
            _, err_hist = train_model(
                model, mu1, A, A_inv, device, n_iter=n_iter, log_every=n_iter
            )
            final_errors[f"Shared_{version}"].append(err_hist[-1])
            print(f"K={K:2d} Shared {version:3s} ({n_params:5d} params): {err_hist[-1]:.4e}")

        # --- MLP à budget égal (référence : même budget que DFB LFO) ---
        n_params_ref = count_params(DFB_UNN(dim=d, K=K, version="LFO").to(device))
        mlp, actual_params = make_mlp_with_budget(d, n_params_ref, device)
        param_counts["MLP_matched"].append(actual_params)
        _, err_hist_mlp = train_model(
            mlp, mu1, A, A_inv, device, n_iter=n_iter, log_every=n_iter
        )
        final_errors["MLP_matched"].append(err_hist_mlp[-1])
        print(f"K={K:2d} MLP ({actual_params:5d} params, cible {n_params_ref}): {err_hist_mlp[-1]:.4e}")

    _plot_K_ablation(K_list, final_errors, param_counts)
    return final_errors, param_counts


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device_str}")
    
    # Expérience principale
    results = run_experiment(
        d=10,
        n_iter=10_000,
        K_list=(3, 5),
        device_str=device_str,
    )
    plot_results(results, log_every=1000, n_iter=10_000)
    
    # Ablation sur K
    device = torch.device(device_str)
    mu1    = torch.tensor([2.0, 1.0], device=device)
    torch.manual_seed(42)
    L      = torch.randn(2, 2, device=device) * 0.5
    Sigma1 = L @ L.T + torch.eye(2, device=device)
    A, A_inv = compute_gaussian_OT_map(mu1, Sigma1)
    
    K_ablation(mu1, A, A_inv, device, K_list=(1, 2, 3, 5, 10, 15), n_iter=5000)
    
    if d==2:
        # Visualisation du champ de vecteurs (d=2 seulement)
        best_model = DFB_UNN(dim=2, K=10, version="LNO", w=32).to(device)
        train_model(best_model, mu1, A, A_inv, device, n_iter=10_000, log_every=10_000)
        plot_vector_fields(mu1, A, A_inv, best_model, device, title="DFB-UNN LNO K=10")