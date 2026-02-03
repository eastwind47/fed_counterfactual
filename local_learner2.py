# -*- coding: utf-8 -*-
"""
local_learner2.py

Client-side learner with optional IMEX (semi-implicit) updates and
compatibility with Augmented Lagrangian (AL/ADMM) signals from the server.

This module mirrors the behavior of your existing local_learner1, while adding:
  • A stable **IMEX** update mode for (θ, φ)
  • Compatibility with server-side **AL/ADMM** (no client duals required)

API (drop-in spirit):
  - local_forward_pass(y_prev, y_curr, h_c_tm1, u_tm1) → dict payload to server
  - gradient_update(gradx, gradx_est) → in-place updates of θ, φ

Notation (for a fixed client m):
  p = p_m (state dim), s = s_m (input dim), d = d_m (measurement dim)
  A ∈ R^{p×p}, B ∈ R^{p×s}, C ∈ R^{d×p}
  θ ∈ R^{p×d}, φ ∈ R^{p}
  For a batch of T steps (column-major time):
    y_prev ∈ R^{d×T}  holds y^{t-1}
    y_curr ∈ R^{d×T}  holds y^{t}
    h_c_tm1 ∈ R^{p×T} holds \hat h_{c}^{t-1}
    u_tm1 ∈ R^{s×T}   holds u^{t-1}

Model (per column t):
  \hat h_{a}^{t-1} = h_c_tm1[:,t] + θ y_prev[:,t]
  h_{a}^{t}        = A \hat h_{a}^{t-1} + B u_tm1[:,t] + φ
  r_{a}^{t}        = y_curr[:,t] - C h_{a}^{t}

Server upstreams to client:
  gradx     = 2 r_{s}  ∈ R^{p×T}
  gradx_est = 2 A^T( ξ d )     [penalty mode]  or  2 A^T( ρ u^{k+1} ) [AL mode, scaled dual],
  where u^{k+1} = u^k + d^k.

Client gradients used here:
  ∇_φ L_local = -(2/T) Σ_t C^T r_a^{t}
  ∇_θ L_local = -(2/T) Σ_t A^T C^T r_a^{t} (y_prev^{t})^T
  ∇_θ L_server = (1/T) [ A^T gradx + gradx_est ]  @ (y_prev)^T,
  where gradx = 2 r_s and (AL) gradx_est = 2 A^T( ρ u^{k+1} ).
  ∇_φ L_server = (2/T) Σ_t r_{s}^{t} = (1/T) sum_t gradx[:,t]

IMEX (semi-implicit) option:
  Solve for increments Δθ, Δφ using local SPD approximations:
    (I + τ_θ η_1 H_θ) vec(Δθ) = - vec( η_1 ∇_θ L_local + η_2 ∇_θ L_server )
    (I + τ_φ γ_1 H_φ)    Δφ   = -      γ_1 ∇_φ L_local - γ_2 ∇_φ L_server
  where  H_θ = 2 ( Σ_{yy}^T ⊗ A^T C^T C A )  and  H_φ = 2 C^T C.

If IMEX is disabled, we perform the explicit updates used previously:
  θ ← θ - η_1 ∇_θ L_local - η_2 ∇_θ L_server
  φ ← φ - γ_1 ∇_φ L_local - γ_2 ∇_φ L_server
"""
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

Array = np.ndarray


# ---------------------------- Helpers ---------------------------- #
def _vec(M: Array) -> Array:
    return M.reshape(-1, 1, order="F")


def _mat(v: Array, shape: Tuple[int, int]) -> Array:
    return v.reshape(shape, order="F")


def _I(n: int) -> Array:
    return np.eye(n, dtype=float)


def _spd_solve(L: Array, b: Array) -> Array:
    """Solve L x = b with Cholesky if SPD; fallback to np.linalg.solve."""
    try:
        R = np.linalg.cholesky(L)
        y = np.linalg.solve(R.T, b)
        x = np.linalg.solve(R, y)
        return x
    except np.linalg.LinAlgError:
        return np.linalg.solve(L, b)


class LocalLearner2:
    """Client-side learner (one client m).

    Parameters
    ----------
    A : Array (p×p)
        Known diagonal A_{mm} for this client.
    B : Array (p×s)
        Known diagonal B_{mm} for this client.
    C : Array (d×p)
        Known diagonal C_{mm} for this client.
    eta_1, eta_2 : float
        Stepsizes for θ from local/server terms.
    gamma_1, gamma_2 : float
        Stepsizes for φ from local/server terms.
    use_imex_local : bool
        If True, apply semi-implicit (proximal) updates for θ, φ.
    tau_theta, tau_phi : float
        IMEX damping multipliers (≥ 0). Default 1.0.
    """

    def __init__(
        self,
        A: Array,
        B: Array,
        C: Array,
        *,
        eta_1: float = 1e-2,
        eta_2: float = 1e-2,
        gamma_1: float = 1e-2,
        gamma_2: float = 1e-2,
        use_imex_local: bool = False,
        tau_theta: float = 1.0,
        tau_phi: float = 1.0,
    ) -> None:
        # Fixed local system matrices
        self.A: Array = A.copy()
        self.B: Array = B.copy()
        self.C: Array = C.copy()

        # Dimensions
        self.p: int = self.A.shape[0]
        self.s: int = self.B.shape[1]
        self.d: int = self.C.shape[0]

        # Learnable params
        self.theta: Array = np.zeros((self.p, self.d))
        self.phi: Array = np.zeros((self.p,))

        # Stepsizes
        self.eta_1 = float(eta_1)
        self.eta_2 = float(eta_2)
        self.gamma_1 = float(gamma_1)
        self.gamma_2 = float(gamma_2)

        # IMEX controls
        self.use_imex_local = bool(use_imex_local)
        self.tau_theta = float(tau_theta)
        self.tau_phi = float(tau_phi)

        # Caches for a round
        self._cache: Dict[str, Array] = {}

    # ----------------------- Public setters/getters ----------------------- #
    def set_params(self, theta: Array, phi: Array) -> None:
        assert theta.shape == (self.p, self.d)
        assert phi.shape == (self.p,)
        self.theta = theta.copy()
        self.phi = phi.copy()

    def get_params(self) -> Tuple[Array, Array]:
        return self.theta.copy(), self.phi.copy()

    # ------------------------- Forward pass ------------------------- #
    def local_forward_pass(
        self,
        y_prev: Array,   # (d×T)  y^{t-1}
        y_curr: Array,   # (d×T)  y^{t}
        h_c_tm1: Array,  # (p×T)  \hat h_{c}^{t-1}
        u_tm1: Array,    # (s×T)  u^{t-1}
    ) -> Dict[str, Array]:
        """Compute augmented states and local residuals, cache stats, and
        return payload for the server.

        Returns dict with keys:
          'h_aug_est_tm1' : \hat h_{a}^{t-1}  (p×T)
          'h_aug_pred'    : h_{a}^{t}          (p×T)
          'r_local'       : r_{a}^{t}          (d×T)
          'Sigma_yy'      : Σ_{y_m y_m}        (d×d)
        """
        p, d, s = self.p, self.d, self.s
        assert y_prev.shape[0] == d and y_curr.shape[0] == d
        assert h_c_tm1.shape[0] == p and u_tm1.shape[0] == s
        T = y_prev.shape[1]
        assert y_curr.shape[1] == T and h_c_tm1.shape[1] == T and u_tm1.shape[1] == T

        # Augmented prior and state
        h_hat_a_tm1 = h_c_tm1 + self.theta @ y_prev            # (p×T)
        h_a = self.A @ h_hat_a_tm1 + self.B @ u_tm1 + self.phi.reshape(-1, 1)  # (p×T)

        # Local output residuals
        r_local = y_curr - self.C @ h_a                         # (d×T)

        # Time-averaged stats used in IMEX/Hessian
        Sigma_yy = (y_prev @ y_prev.T) / float(T)               # (d×d)
        # Note: Sigma_yy is symmetric PSD; Sigma_yy.T == Sigma_yy.

        # Cache everything needed for gradients
        self._cache = {
            "y_prev": y_prev.copy(),
            "y_curr": y_curr.copy(),
            "h_c_tm1": h_c_tm1.copy(),
            "u_tm1": u_tm1.copy(),
            "h_hat_a_tm1": h_hat_a_tm1.copy(),
            "h_a": h_a.copy(),
            "r_local": r_local.copy(),
            "Sigma_yy": Sigma_yy.copy(),
            "T": T,
        }

        return {
            "h_aug_est_tm1": h_hat_a_tm1,
            "h_aug_pred": h_a,
            "r_local": r_local,
            "Sigma_yy": Sigma_yy,
        }

    # -------------------------- Update step -------------------------- #
    def gradient_update(self, gradx: Array, gradx_est: Array) -> None:
        """Update θ and φ using server upstreams and local residuals.

        Parameters
        ----------
        gradx : Array (p×T)
            Server upstream 2 r_{s}.
        gradx_est : Array (p×T)
            Server upstream 2 A^T(ξ d) or 2 A^T(ρ u^{k+1}) [scaled-dual AL] depending on server mode.
        """
        # Pull from cache
        y_prev: Array = self._cache["y_prev"]
        r_local: Array = self._cache["r_local"]
        T: int = int(self._cache["T"])  # number of columns

        p = self.p
        assert gradx.shape[0] == p and gradx_est.shape[0] == p, "gradx/gradx_est must have p rows"
        assert gradx.shape[1] == T and gradx_est.shape[1] == T, "gradx/gradx_est must have T columns"

        A, C = self.A, self.C

        # ---------- Gradients ---------- #
        inv_T = 1.0 / float(T)

        # Local grads
        g_phi_local = -(2.0 * inv_T) * (C.T @ r_local.sum(axis=1).reshape(-1, 1)).reshape(-1)  # (p,)
        g_theta_local = -(2.0 * inv_T) * (A.T @ C.T @ r_local) @ y_prev.T                      # (p×d)

        # Server uses scaled duals: after server_forward_pass, u is updated to u^{k+1},
        # and we receive gradx_est = 2 A^T(ρ u^{k+1}). This equals 2 A^T[ρ(u^k + d^k)].

        # Server grads
        # ∇_φ L_s = (1/T) sum_t gradx[:,t]
        g_phi_server = (inv_T) * gradx.sum(axis=1).reshape(-1)                                 # (p,)
        # ∇_θ L_s = (1/T) [ A^T gradx + gradx_est ] @ y_prev^T
        g_theta_server = (inv_T) * (A.T @ gradx + gradx_est) @ y_prev.T                        # (p×d)

        # Combine (explicit gradient direction)
        G_theta = self.eta_1 * g_theta_local + self.eta_2 * g_theta_server                     # (p×d)
        G_phi = self.gamma_1 * g_phi_local + self.gamma_2 * g_phi_server                       # (p,)

        if not self.use_imex_local:
            # -------- Explicit GD (original behavior) -------- #
            self.theta = self.theta - G_theta
            self.phi = self.phi - G_phi

            metrics = {
                "norm_r_local": float(np.linalg.norm(r_local) / np.sqrt(T)),
                "norm_gradx": float(np.linalg.norm(gradx) / np.sqrt(T)),
                "norm_gradx_est": float(np.linalg.norm(gradx_est) / np.sqrt(T)),
                "delta_theta_fro": float(np.linalg.norm(-G_theta, ord='fro')),
                "delta_phi": float(np.linalg.norm(-G_phi)),
            }
            self._cache["metrics"] = metrics

            return

        # -------- IMEX: semi-implicit (proximal) step -------- #
        # Hessian approximations (local, SPD/PSD):
        #   H_theta = 2 ( Σ_{yy}^T ⊗ A^T C^T C A )  (size: (p d)×(p d))
        #   H_phi   = 2 C^T C                         (size: p×p)
        Sigma_yy: Array = self._cache["Sigma_yy"]
        H_theta_left = A.T @ C.T @ C @ A                       # (p×p)
        H_theta = np.kron(Sigma_yy.T, H_theta_left) * 2.0      # (p d × p d)
        H_phi = (C.T @ C) * 2.0                                # (p×p)

        eps = 1e-10
        # Solve for Δθ, Δφ:
        # (I + τ_θ η_1 H_θ) vec(Δθ) = - vec(G_theta)
        # (I + τ_φ γ_1 H_φ)    Δφ   = - G_phi
        L_theta = _I(self.p * self.d) + (self.tau_theta * self.eta_1) * H_theta + eps * _I(self.p * self.d)
        rhs_theta = -_vec(G_theta)
        dtheta = _spd_solve(L_theta, rhs_theta)
        dtheta = _mat(dtheta, (self.p, self.d))

        L_phi = _I(self.p) + (self.tau_phi * self.gamma_1) * H_phi + eps * _I(self.p)
        rhs_phi = -G_phi.reshape(-1, 1)
        dphi = _spd_solve(L_phi, rhs_phi).reshape(-1)

        # Apply updates
        self.theta = self.theta + dtheta
        self.phi = self.phi + dphi

        metrics = {
            "norm_r_local": float(np.linalg.norm(r_local) / np.sqrt(T)),
            "norm_gradx": float(np.linalg.norm(gradx) / np.sqrt(T)),
            "norm_gradx_est": float(np.linalg.norm(gradx_est) / np.sqrt(T)),
            "delta_theta_fro": float(np.linalg.norm(dtheta, ord='fro')),
            "delta_phi": float(np.linalg.norm(dphi)),
        }
        self._cache["metrics"] = metrics
