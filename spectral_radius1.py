from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import scipy.linalg

from data_retriever import RetrieveData


@dataclass
class ComponentDimensions:
    p: int  # state dimension
    s: int  # input dimension
    d: int  # output dimension


class SpectralRadiusAnalyzer:
    """Construct the linear recursion matrix ``H_m`` for each component.

    The implementation mirrors the notation supplied in the LaTeX snippet. All
    statistics are computed over the training horizon ``T`` using the DKF
    trajectories ``\hat{h}_{m,c}``, the inputs ``u_m`` and the outputs ``y_m``.
    """

    def __init__(
        self,
        data: RetrieveData,
        dkf_trajectories: Dict[str, np.ndarray],
    ) -> None:
        self.data = data
        self.dkf_trajectories = {
            int(key): np.asarray(value, dtype=float)
            for key, value in dkf_trajectories.items()
        }

        self.M = data.num_components
        self.T = data.training_time

        # Server hyper-parameters
        gl_pack = data.global_learners_pack
        self.alpha_a = gl_pack["alpha_a"]
        self.alpha_b = gl_pack["alpha_b"]
        self.xi = gl_pack["xi"]
        self.A_diag = self._extract_diag_blocks(gl_pack["A_mm"], label="A_mm")
        self.B_diag = self._extract_diag_blocks(gl_pack["B_mm"], label="B_mm")

        # Local bundles per component
        self.local_packs = {
            m: data.local_learners_pack[f"comp_{m}"] for m in range(1, self.M + 1)
        }
        self.dimensions: Dict[int, ComponentDimensions] = {}
        self.Y: Dict[int, np.ndarray] = {}
        self.U: Dict[int, np.ndarray] = {}
        self.C: Dict[int, np.ndarray] = {}
        self.mu_y: Dict[int, np.ndarray] = {}
        self.Sigma_yy: Dict[int, np.ndarray] = {}
        self.mu_h: Dict[int, np.ndarray] = {}
        self.mu_u: Dict[int, np.ndarray] = {}

        # Statistics that depend on component pairs
        self.Sigma_hh: Dict[Tuple[int, int], np.ndarray] = {}
        self.Sigma_uu: Dict[Tuple[int, int], np.ndarray] = {}
        self.Sigma_uh: Dict[Tuple[int, int], np.ndarray] = {}
        self.Sigma_hu: Dict[Tuple[int, int], np.ndarray] = {}
        self.Sigma_yh: Dict[Tuple[int, int], np.ndarray] = {}
        self.Sigma_yu: Dict[Tuple[int, int], np.ndarray] = {}
        self.Sigma_hy: Dict[Tuple[int, int], np.ndarray] = {}
        self.Sigma_uy: Dict[Tuple[int, int], np.ndarray] = {}

        self._prepare_component_summaries()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_H_blocks(self, component: int) -> Dict[str, np.ndarray]:
        """Return the block matrices that compose ``H_m`` for ``component``."""
        if component < 1 or component > self.M:
            raise ValueError(f"component index {component} out of range [1, {self.M}]")

        dims = self.dimensions[component]
        A_mm = self.A_diag[component]
        B_mm = self.B_diag[component]
        C_mm = self.C[component]

        peers = [idx for idx in range(1, self.M + 1) if idx != component]
        if not peers:
            raise ValueError("Spectral radius requires at least two components")

        # Stepsizes and local hyper-parameters
        pack = self.local_packs[component]
        eta_1 = pack["eta_1"]
        eta_2 = pack["eta_2"]
        gamma_1 = pack["gamma_1"]
        gamma_2 = pack["gamma_2"]

        p_m, d_m = dims.p, dims.d
        I_pm = np.eye(p_m)

        # Helper indices for block placement
        peer_A_sizes = {n: p_m * self.dimensions[n].p for n in peers}
        peer_B_sizes = {n: p_m * self.dimensions[n].s for n in peers}
        offset_A = 0
        offset_B = offset_A + sum(peer_A_sizes.values())
        offset_phi = offset_B + sum(peer_B_sizes.values())
        offset_theta = offset_phi + p_m

        total_dim = offset_theta + p_m * d_m
        H = np.zeros((total_dim, total_dim))

        # ------------------------------
        # Fill A rows
        # ------------------------------
        row_start = offset_A
        for n in peers:
            p_n = self.dimensions[n].p
            row_end = row_start + peer_A_sizes[n]

            # Columns in A block
            col_start = offset_A
            for p in peers:
                p_p = self.dimensions[p].p
                col_end = col_start + peer_A_sizes[p]

                block = np.zeros((peer_A_sizes[n], peer_A_sizes[p]))
                if n == p:
                    block += np.eye(peer_A_sizes[n])
                Sigma_hp_hn_T = self.Sigma_hh[(p, n)].T
                block -= 2.0 * self.alpha_a * (1.0 + self.xi) * np.kron(Sigma_hp_hn_T, I_pm)
                H[row_start:row_end, col_start:col_end] = block
                col_start = col_end

            # Columns in B block
            col_start = offset_B
            for p in peers:
                col_end = col_start + peer_B_sizes[p]
                Sigma_up_hn_T = self.Sigma_uh[(p, n)].T
                block = -2.0 * self.alpha_a * np.kron(Sigma_up_hn_T, I_pm)
                H[row_start:row_end, col_start:col_end] = block
                col_start = col_end

            # Column for phi
            col_start = offset_phi
            col_end = offset_phi + p_m
            block = 2.0 * self.alpha_a * np.kron(self.mu_h[n], I_pm)
            H[row_start:row_end, col_start:col_end] = block

            # Column for theta
            col_start = offset_theta
            col_end = total_dim
            Sigma_y_hn_T = self.Sigma_yh[(component, n)].T
            block = 2.0 * self.alpha_a * (1.0 + self.xi) * np.kron(Sigma_y_hn_T, A_mm)
            H[row_start:row_end, col_start:col_end] = block

            row_start = row_end

        # ------------------------------
        # Fill B rows
        # ------------------------------
        row_start = offset_B
        for n in peers:
            s_n = self.dimensions[n].s
            row_end = row_start + peer_B_sizes[n]

            # Columns in A block
            col_start = offset_A
            for p in peers:
                col_end = col_start + peer_A_sizes[p]
                Sigma_hp_un_T = self.Sigma_hu[(p, n)].T
                block = -2.0 * self.alpha_b * np.kron(Sigma_hp_un_T, I_pm)
                H[row_start:row_end, col_start:col_end] = block
                col_start = col_end

            # Columns in B block
            col_start = offset_B
            for p in peers:
                col_end = col_start + peer_B_sizes[p]
                block = np.zeros((peer_B_sizes[n], peer_B_sizes[p]))
                if n == p:
                    block += np.eye(peer_B_sizes[n])
                Sigma_up_un_T = self.Sigma_uu[(p, n)].T
                block -= 2.0 * self.alpha_b * np.kron(Sigma_up_un_T, I_pm)
                H[row_start:row_end, col_start:col_end] = block
                col_start = col_end

            # Column for phi
            col_start = offset_phi
            col_end = offset_phi + p_m
            block = 2.0 * self.alpha_b * np.kron(self.mu_u[n], I_pm)
            H[row_start:row_end, col_start:col_end] = block

            # Column for theta
            col_start = offset_theta
            col_end = total_dim
            Sigma_y_un_T = self.Sigma_yu[(component, n)].T
            block = 2.0 * self.alpha_b * np.kron(Sigma_y_un_T, A_mm)
            H[row_start:row_end, col_start:col_end] = block

            row_start = row_end

        # ------------------------------
        # Fill phi row
        # ------------------------------
        row_start = offset_phi
        row_end = row_start + p_m

        # Columns from A block
        col_start = offset_A
        for p in peers:
            col_end = col_start + peer_A_sizes[p]
            block = 2.0 * gamma_2 * np.kron(self.mu_h[p].T, I_pm)
            H[row_start:row_end, col_start:col_end] = block
            col_start = col_end

        # Columns from B block
        col_start = offset_B
        for p in peers:
            col_end = col_start + peer_B_sizes[p]
            block = 2.0 * gamma_2 * np.kron(self.mu_u[p].T, I_pm)
            H[row_start:row_end, col_start:col_end] = block
            col_start = col_end

        # Diagonal phi block
        col_start = offset_phi
        col_end = col_start + p_m
        H_phi_phi = (
            np.eye(p_m)
            - 2.0 * gamma_1 * (C_mm.T @ C_mm)
            - 2.0 * gamma_2 * np.eye(p_m)
        )
        H[row_start:row_end, col_start:col_end] = H_phi_phi

        # Phi-theta block
        col_start = offset_theta
        col_end = total_dim
        term_theta = 2.0 * gamma_1 * (C_mm.T @ C_mm @ A_mm) + 2.0 * gamma_2 * A_mm
        H[row_start:row_end, col_start:col_end] = -np.kron(
            self.mu_y[component].T, term_theta
        )

        # ------------------------------
        # Fill theta row
        # ------------------------------
        row_start = offset_theta
        row_end = total_dim

        # Columns from A block
        col_start = offset_A
        for p in peers:
            col_end = col_start + peer_A_sizes[p]
            Sigma_hp_y_T = self.Sigma_hy[(p, component)].T
            block = 2.0 * eta_2 * (1.0 + self.xi) * np.kron(Sigma_hp_y_T, A_mm.T)
            H[row_start:row_end, col_start:col_end] = block
            col_start = col_end

        # Columns from B block
        col_start = offset_B
        for p in peers:
            col_end = col_start + peer_B_sizes[p]
            Sigma_up_y_T = self.Sigma_uy[(p, component)].T
            block = 2.0 * eta_2 * np.kron(Sigma_up_y_T, A_mm.T)
            H[row_start:row_end, col_start:col_end] = block
            col_start = col_end

        # Column from phi block
        col_start = offset_phi
        col_end = col_start + p_m
        first = 2.0 * eta_1 * np.kron(self.mu_y[component], A_mm.T @ C_mm.T @ C_mm)
        second = 2.0 * eta_2 * np.kron(self.mu_y[component], A_mm.T)
        H[row_start:row_end, col_start:col_end] = -(first + second)

        # Diagonal theta block
        col_start = offset_theta
        col_end = total_dim
        Sigma_yy_T = self.Sigma_yy[component].T
        term1 = np.kron(Sigma_yy_T, A_mm.T @ C_mm.T @ C_mm @ A_mm)
        term2 = np.kron(Sigma_yy_T, A_mm.T @ A_mm)
        H_theta_theta = (
            np.eye(p_m * d_m)
            - 2.0 * eta_1 * term1
            - 2.0 * eta_2 * (1.0 + self.xi) * term2
        )
        H[row_start:row_end, col_start:col_end] = H_theta_theta

        return {
            "H": H,
            "H_AA": H[offset_A:offset_B, offset_A:offset_B],
            "H_BB": H[offset_B:offset_phi, offset_B:offset_phi],
            "H_phiphi": H_phi_phi,
            "H_thetatheta": H_theta_theta,
        }

    def spectral_radii(self, component: int) -> Dict[str, float]:
        """Return spectral radii for the diagonal blocks and the full ``H_m``."""
        blocks = self.build_H_blocks(component)
        rho_full = _spectral_radius(blocks["H"])
        rho_AA = _spectral_radius(blocks["H_AA"])
        rho_BB = _spectral_radius(blocks["H_BB"])
        rho_phi = _spectral_radius(blocks["H_phiphi"])
        rho_theta = _spectral_radius(blocks["H_thetatheta"])
        return {
            "rho_full": rho_full,
            "rho_H_AA": rho_AA,
            "rho_H_BB": rho_BB,
            "rho_H_phiphi": rho_phi,
            "rho_H_thetatheta": rho_theta,
        }

    def print_spectral_summary(self) -> None:
        """Print spectral radii for every component."""
        for m in range(1, self.M + 1):
            radii = self.spectral_radii(m)
            readable = ", ".join(
                f"{key}={value:.6f}" for key, value in radii.items()
            )
            print(f"Component {m}: {readable}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_diag_blocks(self, blocks: Dict[str, np.ndarray], *, label: str) -> Dict[int, np.ndarray]:
        """Convert diagonal block dictionary keyed by 'mm' strings into int-keyed."""
        diag: Dict[int, np.ndarray] = {}
        for key, value in blocks.items():
            mid = len(key) // 2
            left, right = key[:mid], key[mid:]
            if left and left == right:
                idx = int(left)
                diag[idx] = np.asarray(value, dtype=float)
        if len(diag) != self.M:
            missing = sorted({m for m in range(1, self.M + 1)} - set(diag.keys()))
            raise KeyError(f"{label} is missing diagonal blocks for components {missing}")
        return diag

    def _prepare_component_summaries(self) -> None:
        for m in range(1, self.M + 1):
            pack = self.local_packs[m]
            p_m = int(pack["p_m"])
            s_m = int(pack["s_m"])
            d_m = int(pack["d_m"])
            dims = ComponentDimensions(p=p_m, s=s_m, d=d_m)
            self.dimensions[m] = dims

            Y_m = np.asarray(pack["Y"][:, : self.T], dtype=float)
            U_m = np.asarray(pack["U"][:, : self.T], dtype=float)
            C_m = np.asarray(pack["C"], dtype=float)
            self.Y[m] = Y_m
            self.U[m] = U_m
            self.C[m] = C_m

            self.mu_y[m] = np.mean(Y_m, axis=1, keepdims=True)
            self.Sigma_yy[m] = (Y_m @ Y_m.T) / self.T
            self.mu_u[m] = np.mean(U_m, axis=1, keepdims=True)
            self.mu_h[m] = np.mean(self.dkf_trajectories[m], axis=1, keepdims=True)

        # Pairwise statistics
        for n in range(1, self.M + 1):
            h_n = self.dkf_trajectories[n]
            u_n = self.U[n]
            for p in range(1, self.M + 1):
                h_p = self.dkf_trajectories[p]
                u_p = self.U[p]
                self.Sigma_hh[(n, p)] = (h_n @ h_p.T) / self.T
                self.Sigma_uu[(n, p)] = (u_n @ u_p.T) / self.T
                self.Sigma_uh[(n, p)] = (u_n @ h_p.T) / self.T
                self.Sigma_hu[(n, p)] = (h_n @ u_p.T) / self.T

        for m in range(1, self.M + 1):
            y_m = self.Y[m]
            for n in range(1, self.M + 1):
                h_n = self.dkf_trajectories[n]
                u_n = self.U[n]
                self.Sigma_yh[(m, n)] = (y_m @ h_n.T) / self.T
                self.Sigma_yu[(m, n)] = (y_m @ u_n.T) / self.T
                self.Sigma_hy[(n, m)] = (h_n @ y_m.T) / self.T
                self.Sigma_uy[(n, m)] = (u_n @ y_m.T) / self.T


def _spectral_radius(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    eigvals = scipy.linalg.eigvals(matrix, check_finite=False)
    return float(np.max(np.abs(eigvals)))
