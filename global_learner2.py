# -*- coding: utf-8 -*-
"""
global_learner2.py

A drop-in, extended server learner that supports:
  • Augmented Lagrangian (AL/ADMM) hard-constraint handling (λ, ρ)
  • IMEX (semi-implicit / proximal) server updates for off-diagonal A,B

It is designed to be API-compatible with your existing training loop:
  - server_forward_pass(...)  : compute server states/residuals, upstream terms
  - apply_server_updates()    : update off-diagonal blocks (explicit GD or IMEX)
  - get_gradients_for_clients(): return upstream signals to local clients

Assumptions:
  • States and inputs are provided as dictionaries of arrays per component m,
    with 1-based string keys: "1","2",...,"M" to match your previous code.
  • Time proceeds along the second dimension: shape (dim, T).
  • Off-diagonal parameter blocks are addressed by concatenated 1-based keys:
      key_off = f"{m}{n}" for m != n.
  • Diagonal blocks A_mm, B_mm, C_mm are provided at construction and fixed.

Notes:
  - If your code uses a different key convention (e.g., ints or tuples), change
    the _key() / _offkey() helpers accordingly.
  - This module uses only numpy; no external deps.

Author: global_learner2 (proposed by ChatGPT)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np


Array = np.ndarray


def _vec(M: Array) -> Array:
    """Column-major vectorization to a column vector (n,1)."""
    return M.reshape(-1, 1, order="F")


def _mat(v: Array, shape: Tuple[int, int]) -> Array:
    """Inverse of _vec (column-major). Accepts (n,) or (n,1)."""
    return v.reshape(shape, order="F")


def _I(n: int) -> Array:
    return np.eye(n, dtype=float)


def _key(m: int) -> str:
    """1-based component key for diagonal blocks."""
    return f"{m}"


def _offkey(m: int, n: int) -> str:
    """1-based key for off-diagonal blocks (m != n)."""
    return f"{m}{n}"


class GlobalLearner2:
    """
    Extended global learner with AL/ADMM and IMEX options.

    Parameters
    ----------
    M : int
        Number of components/clients.
    A_mm : Dict[str, Array]
        Fixed diagonal A blocks: A_mm[key_m] ∈ R^{p_m × p_m}.
    B_mm : Dict[str, Array]
        Fixed diagonal B blocks: B_mm[key_m] ∈ R^{p_m × s_m}.
    C_mm : Dict[str, Array]
        Fixed diagonal C blocks: C_mm[key_m] ∈ R^{d_m × p_m}.
    p_dims, s_dims, d_dims : Dict[str, int]
        Dimensions per component m, keyed by "1".. "M".
    T : int
        Number of time steps per round.
    xi : float
        Penalty coefficient for penalty-method mode (kept for compatibility).
    use_aug_lagrangian : bool
        If True, use AL with duals λ and penalty ρ instead of ξ.
    rho : float
        Augmented-Lagrangian penalty. Typical 1–10.
    use_imex : bool
        If True, apply semi-implicit (proximal) server updates on A,B.
    alpha_a, alpha_b : float
        Server step sizes for A and B updates (used both in explicit and IMEX).
    """

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        M: int,
        A_mm: Dict[str, Array],
        B_mm: Dict[str, Array],
        C_mm: Dict[str, Array],
        p_dims: Dict[str, int],
        s_dims: Dict[str, int],
        d_dims: Dict[str, int],
        T: int,
        *,
        xi: float = 0.0,
        use_aug_lagrangian: bool = False,
        rho: float = 1.0,
        use_imex: bool = False,
        alpha_a: float = 1e-2,
        alpha_b: float = 1e-2,
    ) -> None:
        self.M = M
        self.A_mm = A_mm
        self.B_mm = B_mm
        self.C_mm = C_mm
        self.p_dims = p_dims
        self.s_dims = s_dims
        self.d_dims = d_dims
        self.T = T

        # Optimization knobs
        self.xi = float(xi)
        self.use_aug_lagrangian = bool(use_aug_lagrangian)
        self.rho = float(rho)
        self.use_imex = bool(use_imex)
        self.alpha_a = float(alpha_a)
        self.alpha_b = float(alpha_b)

        # Off-diagonal parameter dictionaries (to be set via set_offdiag)
        self.A_mn: Dict[str, Array] = {}
        self.B_mn: Dict[str, Array] = {}

        # Gradient accumulators (explicit path)
        self.grad_A: Dict[str, Array] = {}
        self.grad_B: Dict[str, Array] = {}

        # Flattened views (optional logging)
        self.A_mn_vec: Dict[str, Array] = {}
        self.B_mn_vec: Dict[str, Array] = {}

        # AL/ADMM scaled duals u_m^t = λ_m^t / ρ (per component m, per time t)
        # Shape per m: (p_m, T)
        self.u_dual: Dict[str, Array] = {
            _key(m): np.zeros((self.p_dims[_key(m)], self.T), dtype=float)
            for m in range(1, self.M + 1)
        }
        # Optional: track ρ adaptation history
        self.rho_history = []  # list of dict reports from adapt_rho

        # Cached per-round signals / stats for IMEX RHS and LHS formation
        self._cache = {
            "h_a": {},      # h_{m,a}^t
            "h_c": {},      # \hat h_{m,c}^{t-1}
            "u": {},        # u_m^{t-1}
            "h_a_prev": {}, # \hat h_{m,a}^{t-1} (optional; may be needed for d)
            "r": {},        # r_{m,s}^t
            "d": {},        # d_{m,s}^t
        }
        # Time-averaged stats (Σ, μ) for IMEX
        self.stats = {
            "Sigma_hnhn": {},  # key_off -> (p_n, p_n)
            "Sigma_unun": {},  # key_off -> (s_n, s_n)
            "mu_hn": {},       # key_off -> (p_n, 1)
            "mu_un": {},       # key_off -> (s_n, 1)
        }

    # --------------------------------------------------------------------- #
    # Off-diagonal initialization
    # --------------------------------------------------------------------- #
    def set_offdiag(self, A_mn_init: Dict[str, Array], B_mn_init: Dict[str, Array]) -> None:
        """Set initial off-diagonal parameter blocks and reset vec views."""
        self.A_mn = {k: v.copy() for k, v in A_mn_init.items()}
        self.B_mn = {k: v.copy() for k, v in B_mn_init.items()}
        self.A_mn_vec = {k: _vec(v).ravel(order="F") for k, v in self.A_mn.items()}
        self.B_mn_vec = {k: _vec(v).ravel(order="F") for k, v in self.B_mn.items()}

    # --------------------------------------------------------------------- #
    # Forward pass: compute server states, residuals, AL duals, and grads
    # --------------------------------------------------------------------- #
    def server_forward_pass(
        self,
        h_aug_pred: Dict[str, Array],
        h_c: Dict[str, Array],
        u: Dict[str, Array],
        *,
        h_a_prev: Optional[Dict[str, Array]] = None,
        build_stats: bool = True,
    ) -> Tuple[Dict[str, Array], Dict[str, Array]]:
        """
        Compute server states h_{m,s}, residuals r_{m,s}, constraint vectors d_{m,s},
        update AL duals (if enabled), and accumulate explicit-path gradients.

        Parameters
        ----------
        h_aug_pred : dict m -> (p_m, T)
            Local augmented states h_{m,a}^t sent by clients.
        h_c : dict m -> (p_m, T)
            Central prior states \hat h_{m,c}^{t-1}.
        u : dict m -> (s_m, T)
            Inputs u_m^{t-1} per component.
        h_a_prev : optional dict m -> (p_m, T)
            Augmented prior \hat h_{m,a}^{t-1}. If None, uses zeros for Δh in d.
        build_stats : bool
            If True, accumulate Σ and μ for IMEX LHS; otherwise skip.

        Returns
        -------
        gradx, gradx_est : dict m -> Array
            Upstream partials to send to clients:
              gradx[m][:, t]     = 2 r_{m,s}^t
              gradx_est[m][:, t] = 2 A_mm^T ( ξ d_{m,s}^t )   [penalty mode]
                                   2 A_mm^T ( λ_m^t + ρ d^t) [AL mode]
        """
        # Cache signals
        self._cache["h_a"] = {k: v.copy() for k, v in h_aug_pred.items()}
        self._cache["h_c"] = {k: v.copy() for k, v in h_c.items()}
        self._cache["u"] = {k: v.copy() for k, v in u.items()}
        if h_a_prev is not None:
            self._cache["h_a_prev"] = {k: v.copy() for k, v in h_a_prev.items()}
        else:
            # If not provided, default Δh = 0 in d (we will warn once)
            self._cache["h_a_prev"] = {
                k: np.zeros_like(self._cache["h_c"][k]) for k in h_c.keys()
            }

        # Reset grads
        self.grad_A = {k: np.zeros_like(V) for k, V in self.A_mn.items()}
        self.grad_B = {k: np.zeros_like(V) for k, V in self.B_mn.items()}

        gradx: Dict[str, Array] = {}
        gradx_est: Dict[str, Array] = {}

        # Per-time accumulation for stats
        stats_acc = {
            "Sigma_hnhn": {},
            "Sigma_unun": {},
            "mu_hn": {},
            "mu_un": {},
        }

        # Iterate over components m
        for m in range(1, self.M + 1):
            key_m = _key(m)
            p_m = self.p_dims[key_m]  # dimension of component m (optional helper)
            # Convenience views
            A_mm = self.A_mm[key_m]
            B_mm = self.B_mm[key_m]

            h_a_m = self._cache["h_a"][key_m]          # (p_m, T)
            h_c_m = self._cache["h_c"][key_m]          # (p_m, T)
            u_m = self._cache["u"][key_m]              # (s_m, T)
            h_a_prev_m = self._cache["h_a_prev"][key_m]# (p_m, T)

            # Prepare outputs per m
            gradx[key_m] = np.zeros_like(h_a_m)
            gradx_est[key_m] = np.zeros_like(h_a_m)

            # Time loop
            for t in range(self.T):
                # Server state using current off-diagonal estimates
                h_s = A_mm @ h_c_m[:, [t]]
                for n in range(1, self.M + 1):
                    if n == m:
                        continue
                    key_off = _offkey(m, n)
                    h_s += self.A_mn[key_off] @ self._cache["h_c"][_key(n)][:, [t]]
                h_s += B_mm @ u_m[:, [t]]
                for n in range(1, self.M + 1):
                    if n == m:
                        continue
                    key_off = _offkey(m, n)
                    h_s += self.B_mn[key_off] @ self._cache["u"][_key(n)][:, [t]]

                # Residuals
                r = h_a_m[:, [t]] - h_s  # r_{m,s}^t
                d = A_mm @ (h_a_prev_m[:, [t]] - h_c_m[:, [t]])  # start with A_mm (Δh)
                for n in range(1, self.M + 1):
                    if n == m:
                        continue
                    key_off = _offkey(m, n)
                    d -= self.A_mn[key_off] @ self._cache["h_c"][_key(n)][:, [t]]

                # Store residuals
                if t == 0:
                    self._cache["r"][key_m] = np.zeros_like(h_a_m)
                    self._cache["d"][key_m] = np.zeros_like(h_a_m)
                self._cache["r"][key_m][:, [t]] = r
                self._cache["d"][key_m][:, [t]] = d

                # AL (scaled dual) / Penalty combination
                if self.use_aug_lagrangian:
                    u_t = self.u_dual[key_m][:, [t]]            # u^k
                    combined = r + self.rho * (u_t + d)         # r + ρ(u^k + d^k)
                    # Dual update (per-time): u^{k+1} = u^k + d^k
                    self.u_dual[key_m][:, [t]] = u_t + d
                    # Upstream estimate gradient: 2 A_mm^T [ ρ (u^k + d^k) ] = 2 A_mm^T [ ρ u^{k+1} ]
                    gradx_est[key_m][:, [t]] = 2.0 * (self.A_mm[key_m].T @ (self.rho * (u_t + d)))
                else:
                    combined = r + self.xi * d
                    gradx_est[key_m][:, [t]] = 2.0 * (self.xi * self.A_mm[key_m].T @ d)

                # Upstream to clients
                gradx[key_m][:, [t]] = 2.0 * r

                # Accumulate explicit-path server grads
                # ∇_{Â_{mn}} L_s = -(2/T) Σ combined h_{n,c}^T
                # ∇_{ B̂_{mn}} L_s = -(2/T) Σ r u_n^T
                for n in range(1, self.M + 1):
                    if n == m:
                        continue
                    key_off = _offkey(m, n)
                    h_n = self._cache["h_c"][_key(n)][:, [t]]
                    u_n = self._cache["u"][_key(n)][:, [t]]
                    self.grad_A[key_off] += -(2.0 / self.T) * (combined @ h_n.T)
                    self.grad_B[key_off] += -(2.0 / self.T) * (r @ u_n.T)

                    # Stats accumulation for IMEX (per (m,n))
                    if build_stats:
                        # Σ_{h_n h_n}, μ_{h_n}
                        S_h = stats_acc["Sigma_hnhn"].setdefault(key_off, np.zeros((h_n.size, h_n.size)))
                        stats_acc["Sigma_hnhn"][key_off] = S_h + (h_n @ h_n.T)
                        mu_h = stats_acc["mu_hn"].setdefault(key_off, np.zeros((h_n.size, 1)))
                        stats_acc["mu_hn"][key_off] = mu_h + h_n
                        # Σ_{u_n u_n}, μ_{u_n}
                        S_u = stats_acc["Sigma_unun"].setdefault(key_off, np.zeros((u_n.size, u_n.size)))
                        stats_acc["Sigma_unun"][key_off] = S_u + (u_n @ u_n.T)
                        mu_u = stats_acc["mu_un"].setdefault(key_off, np.zeros((u_n.size, 1)))
                        stats_acc["mu_un"][key_off] = mu_u + u_n

        # Finalize stats (time average)
        if build_stats:
            for key_off in stats_acc["Sigma_hnhn"].keys():
                self.stats["Sigma_hnhn"][key_off] = stats_acc["Sigma_hnhn"][key_off] / self.T
                self.stats["mu_hn"][key_off] = stats_acc["mu_hn"][key_off] / self.T
                self.stats["Sigma_unun"][key_off] = stats_acc["Sigma_unun"][key_off] / self.T
                self.stats["mu_un"][key_off] = stats_acc["mu_un"][key_off] / self.T

        return gradx, gradx_est

    # --------------------------------------------------------------------- #
    # Server updates: explicit GD (default) or IMEX proximal solves
    # --------------------------------------------------------------------- #
    def apply_server_updates(self) -> None:
        """
        Update off-diagonal blocks Â_{mn}, B̂_{mn}.
        - If use_imex=False: explicit gradient descent (current behavior).
        - If use_imex=True : semi-implicit (proximal) solves per off-diagonal block.
        """
        if not self.use_imex:
            # Explicit gradient descent updates (unchanged behavior)
            for key_off, G in self.grad_A.items():
                self.A_mn[key_off] = self.A_mn[key_off] - self.alpha_a * G
                self.A_mn_vec[key_off] = _vec(self.A_mn[key_off]).ravel(order="F")
            for key_off, G in self.grad_B.items():
                self.B_mn[key_off] = self.B_mn[key_off] - self.alpha_b * G
                self.B_mn_vec[key_off] = _vec(self.B_mn[key_off]).ravel(order="F")
            return

        # IMEX updates
        # Requirements: last forward-pass caches are available
        h_a = self._cache["h_a"]
        h_c = self._cache["h_c"]
        u = self._cache["u"]
        h_a_prev = self._cache["h_a_prev"]
        r_cache = self._cache["r"]
        d_cache = self._cache["d"]

        # Solve for each off-diagonal block separately
        for m in range(1, self.M + 1):
            key_m = _key(m)
            A_mm = self.A_mm[key_m]
            # Precompute Δh_m(t) = \hat h_{a,m}^{t-1} - \hat h_{c,m}^{t-1}
            Delta_h_m = h_a_prev[key_m] - h_c[key_m]  # (p_m, T)

            for n in range(1, self.M + 1):
                if n == m:
                    continue
                key_off = _offkey(m, n)
                key_n = _key(n)

                p_m = self.p_dims[key_m]
                p_n = self.p_dims[key_n]
                s_n = self.s_dims[key_n]

                # LHS matrices (SPD)
                Sigma_hnhn = self.stats["Sigma_hnhn"].get(key_off, None)
                if Sigma_hnhn is None:
                    # Fallback: compute from cache if not present
                    HN = np.zeros((p_n, p_n))
                    for t in range(self.T):
                        h_n = h_c[key_n][:, [t]]
                        HN += h_n @ h_n.T
                    Sigma_hnhn = HN / self.T
                    self.stats["Sigma_hnhn"][key_off] = Sigma_hnhn

                Sigma_unun = self.stats["Sigma_unun"].get(key_off, None)
                if Sigma_unun is None:
                    UN = np.zeros((s_n, s_n))
                    for t in range(self.T):
                        u_n = u[key_n][:, [t]]
                        UN += u_n @ u_n.T
                    Sigma_unun = UN / self.T
                    self.stats["Sigma_unun"][key_off] = Sigma_unun

                # Penalty scale used in IMEX RHS (penalty vs AL)
                penalty_scale = self.xi
                use_AL = self.use_aug_lagrangian
                if use_AL:
                    penalty_scale = self.rho  # use ρ for the d-path; λ appears as a separate RHS term

                # ------------------------------ #
                # IMEX for A_{mn}
                # (I + 2 α_A (1+penalty_scale) (Σ_{h_n h_n}^T ⊗ I)) vec(A_new) = vec(A_old) + RHS_A
                # Build RHS_A from signals, excluding self-coupling (p = n); AL adds λ term.
                # ------------------------------ #
                L_A = np.kron(Sigma_hnhn.T, _I(p_m))
                L_A = _I(p_m * p_n) + 2.0 * self.alpha_a * (1.0 + penalty_scale) * L_A

                # Build RHS_A
                RHS_A = np.zeros((p_m * p_n, 1))
                A_old_vec = _vec(self.A_mn[key_off])

                for t in range(self.T):
                    # r contributes fully to RHS (explicit)
                    r_m_t = r_cache[key_m][:, [t]]

                    # AL only (scaled duals): use ρ * u^{k+1}; u was updated in server_forward_pass
                    lam_m_t = (self.rho * self.u_dual[key_m][:, [t]]) if use_AL else 0.0

                    # d-path (only cross terms and Δh part; self-coupling moved to LHS)
                    # ρ * (A_mm Δh_m) h_n^T  -  ρ * Σ_{p≠m, p≠n} A_mp h_p h_n^T
                    d_cross = 0.0
                    if use_AL:
                        d_cross = A_mm @ Delta_h_m[:, [t]]

                    # Aggregate cross A terms: sum over p != m and p != n
                    cross_A = np.zeros((self.p_dims[key_m], 1))
                    for p in range(1, self.M + 1):
                        if p == m or p == n:
                            continue
                        key_mp = _offkey(m, p)
                        h_p_t = h_c[_key(p)][:, [t]]
                        cross_A += self.A_mn[key_mp] @ h_p_t  # (p_m,1)

                    # Form per-time RHS contribution:
                    #   term_A = r_m_t + lam_m_t + ρ*(A_mm Δh_m - cross_A)
                    # No B-terms in d; B enters through r only.
                    term_A = r_m_t + (lam_m_t if use_AL else 0.0)
                    if use_AL:
                        term_A = term_A + self.rho * (d_cross - cross_A)

                    # + (1+ξ)*(h_a - A_mm h_c - B_mm u_m) path (penalty mode only)  —— maps to same structure as above.
                    if not use_AL and (1.0 + penalty_scale) != 1.0:
                        # use compact identity: A_mm θ y + φ = h_a - A_mm h_c - B_mm u_m
                        h_a_minus = h_a[key_m][:, [t]] - A_mm @ h_c[key_m][:, [t]] - self.B_mm[key_m] @ u[key_m][:, [t]]
                        term_A = term_A + penalty_scale * (h_a_minus)  # adds ξ * (A_mm θ y + φ)

                        # subtract ξ * sum_{p≠m, p≠n} A_mp h_p (already in -cross_A multiplied by ρ above for AL – not here)
                        term_A = term_A - penalty_scale * cross_A

                    h_n_t = h_c[key_n][:, [t]]  # (p_n,1)
                    RHS_A += (2.0 * self.alpha_a / self.T) * _vec(term_A @ h_n_t.T)

                A_new_vec = np.linalg.solve(L_A, A_old_vec + RHS_A)
                self.A_mn[key_off] = _mat(A_new_vec, (p_m, p_n))
                self.A_mn_vec[key_off] = A_new_vec.ravel(order="F")

                # ------------------------------ #
                # IMEX for B_{mn}
                # (I + 2 α_B (Σ_{u_n u_n}^T ⊗ I)) vec(B_new) = vec(B_old) + RHS_B
                # RHS_B uses r only; self-coupling removed to LHS.
                # ------------------------------ #
                L_B = np.kron(Sigma_unun.T, _I(p_m))
                L_B = _I(p_m * s_n) + 2.0 * self.alpha_b * L_B

                RHS_B = np.zeros((p_m * s_n, 1))
                B_old_vec = _vec(self.B_mn[key_off])

                for t in range(self.T):
                    # Build S^{(B)}_{mn}(t) = r excluding the B_{mn} u_n part (handled implicitly)
                    # Using residual definition r = h_a - (A_mm h_c + Σ A_mp h_p + B_mm u_m + Σ B_mp u_p)
                    # The explicit contribution to RHS_B is simply r_m^t u_n^T, since the self B_{mn} term is implicit in L_B.
                    r_m_t = r_cache[key_m][:, [t]]
                    u_n_t = u[key_n][:, [t]]
                    RHS_B += (2.0 * self.alpha_b / self.T) * _vec(r_m_t @ u_n_t.T)

                B_new_vec = np.linalg.solve(L_B, B_old_vec + RHS_B)
                self.B_mn[key_off] = _mat(B_new_vec, (p_m, s_n))
                self.B_mn_vec[key_off] = B_new_vec.ravel(order="F")

    # --------------------------------------------------------------------- #
    # Upstream getters (to clients)
    # --------------------------------------------------------------------- #
    def get_gradients_for_clients(
        self, include_est: bool = True
    ) -> Tuple[Dict[str, Array], Optional[Dict[str, Array]]]:
        """
        Return the upstream partials for each client:
          gradx[m]     = 2 r_{m,s}^t
          gradx_est[m] = 2 A_mm^T ( ξ d_{m,s}^t )   [penalty mode]
                       = 2 A_mm^T ( ρ (u_m^t + d^t) ) = 2 A_mm^T ( ρ u^{k+1} ) [AL mode, scaled dual]
        """
        gradx = {k: 2.0 * self._cache["r"][k] for k in self._cache["r"].keys()}
        if not include_est:
            return gradx, None

        gradx_est: Dict[str, Array] = {}
        for m in range(1, self.M + 1):
            key_m = _key(m)
            A_mm = self.A_mm[key_m]
            if self.use_aug_lagrangian:
                # With scaled duals, after server_forward_pass we have u^{k+1} = u^k + d^k.
                # The upstream term is 2 A_mm^T [ ρ (u^k + d^k) ] = 2 A_mm^T [ ρ u^{k+1} ].
                val = np.zeros_like(self._cache["r"][key_m])
                for t in range(self.T):
                    u_t = self.u_dual[key_m][:, [t]]  # already updated to u^{k+1}
                    val[:, [t]] = 2.0 * (A_mm.T @ (self.rho * u_t))
                gradx_est[key_m] = val
            else:
                gradx_est[key_m] = 2.0 * (self.xi * A_mm.T @ self._cache["d"][key_m])

        return gradx, gradx_est

    def adapt_rho(
        self,
        mu: float = 10.0,
        tau_inc: float = 2.0,
        tau_dec: float = 2.0,
        rho_min: float = 1e-6,
        rho_max: float = 1e6,
        use_A_weight: bool = True,
        logger: Optional[object] = None,
    ) -> Dict[str, float]:
        """
        Residual-balanced ρ adaptation (Boyd/ADMM heuristic).
        Should be called *after* server_forward_pass(...) for the current round,
        when self._cache["d"] (constraint residuals) and self.u_dual are up-to-date.

        r_p  := sqrt(sum_m || \bar d_m ||_2^2),   where \bar d_m = mean_t d_m^t
        r_d  := ρ * sqrt(sum_m || A_mm^T \bar d_m ||_2^2) if use_A_weight else ρ * sqrt(sum_m || \bar d_m ||_2^2)

        If r_p > μ r_d:   ρ ← min(ρ_max, τ_inc * ρ)
        elif r_d > μ r_p: ρ ← max(ρ_min, ρ / τ_dec)
        else:             hold ρ

        Returns a dict with metrics and action; also appends to self.rho_history.
        """
        # Guard
        if "d" not in self._cache or not self._cache["d"]:
            raise RuntimeError("adapt_rho: residual cache 'd' is empty; call server_forward_pass(...) first.")

        rp_sq = 0.0
        rd_sq = 0.0
        for m in range(1, self.M + 1):
            key_m = _key(m)
            d_m = self._cache["d"][key_m]  # (p_m, T)
            # average over time
            d_bar = np.mean(d_m, axis=1, keepdims=True)  # (p_m, 1)
            rp_sq += float(d_bar.T @ d_bar)
            if use_A_weight:
                A_mm = self.A_mm[key_m]
                Ad = A_mm.T @ d_bar
                rd_sq += float(Ad.T @ Ad)
            else:
                rd_sq += float(d_bar.T @ d_bar)

        r_p = float(np.sqrt(rp_sq))
        r_d = float(self.rho * np.sqrt(rd_sq))

        old_rho = float(self.rho)
        action = "hold"
        if r_p > mu * r_d:
            self.rho = float(min(rho_max, self.rho * tau_inc))
            action = "inc"
        elif r_d > mu * r_p and self.rho > rho_min:
            self.rho = float(max(rho_min, self.rho / tau_dec))
            action = "dec"

        report = {"r_p": r_p, "r_d": r_d, "old_rho": old_rho, "new_rho": float(self.rho), "action": action}
        self.rho_history.append(report)

        msg = f"[AL] adapt_rho: r_p={r_p:.3e}, r_d={r_d:.3e}, action={action}, rho: {old_rho:.3e} -> {self.rho:.3e}"
        if logger is None:
            print(msg)
        else:
            try:
                logger.info(msg)
            except Exception:
                print(msg)
        return report