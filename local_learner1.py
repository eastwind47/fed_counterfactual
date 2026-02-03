"""Local learner for the revised communication protocol.

This module mirrors the structure of ``local_learner.py`` while implementing the
client-side computations of Algorithm 1 (lines 9–14). Each instance of
``LocalModel`` corresponds to a single client/component and is responsible for
computing the local trajectories that are transmitted to the server after a
complete pass over the time horizon.
"""

from __future__ import annotations

import numpy as np

import kalman_filter


class LocalModel:
    """Local learner that keeps track of client-side trajectories.

    Parameters
    ----------
    T : int
        Horizon length for the current training window.
    component_data : dict
        Client-specific data bundle produced by ``data_retriever``. The bundle
        must contain the keys ``A``, ``B``, ``C``, ``Q``, ``R``, ``P0``, ``x0``,
        ``Y``, ``U``, ``p_m``, ``d_m``, ``s_m`` and the local hyper-parameters.
    theta : np.ndarray, optional
        Initial value for the local augmentation matrix ``\theta_m``. When not
        provided a zero matrix with shape ``(p_m, d_m)`` is used.
    phi : np.ndarray, optional
        Initial value for the local state bias ``\varphi_m``. Defaults to the
        zero vector supplied in ``component_data['phi0']`` when available.
    """

    def __init__(self, T: int, component_data: dict) -> None:
        # Horizon length
        self.T = int(T)

        # System matrices and observations
        self.A = component_data['A']
        self.B = component_data['B']
        self.C = component_data['C']
        self.Q = component_data['Q']
        self.R = component_data['R']
        self.P0 = component_data['P0']
        self.x0 = component_data['x0']
        self.Y = component_data['Y']
        self.U = component_data['U']

        # Dimensions
        self.p_m = component_data['p_m']
        self.d_m = component_data['d_m']
        self.s_m = component_data['s_m']

        # Client-side hyper-parameters
        self.lambda_l = component_data['lambda_l']
        self.eta_1 = component_data['eta_1']
        self.eta_2 = component_data['eta_2']
        self.gamma_1 = component_data['gamma_1']
        self.gamma_2 = component_data['gamma_2']

        # Local parameters initialisation
        self.theta = component_data['theta_init']
        self.phi  = component_data['phi_init']



        h_aug_init = component_data.get('h_aug_init')
        if h_aug_init is None:
            self.h_aug_init = self.x0.copy()
        else:
            base = np.array(h_aug_init, copy=True)
            if base.ndim == 2 and base.shape[1] >= 1:
                self.h_aug_init = base[:, :1]
            else:
                self.h_aug_init = base.reshape(self.p_m, 1)

        y_init = component_data.get('y_init')
        if y_init is None:
            self.y_init = self.C @ self.h_aug_init
        else:
            base = np.array(y_init, copy=True)
            if base.ndim == 2 and base.shape[1] >= 1:
                self.y_init = base[:, :1]
            else:
                self.y_init = base.reshape(self.d_m, 1)

        # Buffers for the DKF trajectories used as the cooperative baseline
        self.X_dkf = np.zeros((self.p_m, self.T))
        self.X_dkf_pred = np.zeros((self.p_m, self.T))
        self.X_dkf_resd = np.zeros((1, self.T))

        # Buffers for the augmented trajectories shared with the server
        self.h_aug_pred = np.zeros((self.p_m, self.T))  # h_{m,a}^{k,t}
        self.h_aug_est = np.zeros((self.p_m, self.T))   # \hat{h}_{m,a}^{k,t}
        self.y_residuals = np.zeros((self.d_m, self.T)) # r_{m,a}^{k,t}

        self.local_loss = 0.0
        self._forward_ran = False

    # ------------------------------------------------------------------
    # Utilities for producing the cooperative baseline (DKF)
    # ------------------------------------------------------------------
    def run_DKF(self) -> None:
        """Run the distributed Kalman filter to obtain ``\hat{h}_{m,c}``."""
        dkf = kalman_filter.KalmanFilter(self.A, self.B, self.C, self.Q, self.R, self.P0, self.x0)

        for t in range(self.T):
            dkf.predict()
            self.X_dkf_pred[:, t:t + 1] = dkf.get_state()
            residual = dkf.residual(self.Y[:, t:t + 1])
            self.X_dkf_resd[0, t:t + 1] = np.linalg.norm(residual)
            dkf.update(self.Y[:, t:t + 1])
            self.X_dkf[:, t:t + 1] = dkf.get_state()

    def ensure_dkf_rollout(self) -> None:
        """Ensure DKF buffers are populated before a local forward pass."""
        if not np.any(self.X_dkf):
            self.run_DKF()

    # ------------------------------------------------------------------
    # Algorithm 1 (client) – lines 9–14
    # ------------------------------------------------------------------
    def local_forward_pass(self) -> dict[str, np.ndarray | float]:
        """Compute local trajectories as described in Algorithm 1 (lines 9–14).

        Returns
        -------
        dict
            A payload containing the rollout ``h_aug_pred``, the corrected
            estimate ``h_aug_est``, the cooperative baseline ``h_coop``, the
            measurement residuals, and the scalar local loss.
        """
        self.ensure_dkf_rollout()

        # Reset buffers for the new pass
        self.h_aug_pred.fill(0.0)
        self.h_aug_est.fill(0.0)
        self.y_residuals.fill(0.0)
        self.local_loss = 0.0
        self._forward_ran = False

        prev_estimate = self.h_aug_init.copy()

        for t in range(self.T):
            y_t = self.Y[:, t:t + 1]
            u_prev = self.U[:, t - 1:t] if t > 0 else np.zeros((self.s_m, 1))

            # Line 10: local model rollout using previous augmented estimate
            h_pred_t = self.A @ prev_estimate + self.B @ u_prev + self.phi
            self.h_aug_pred[:, t:t + 1] = h_pred_t

            # Line 12: residual with current observations
            residual = y_t - self.C @ h_pred_t
            self.y_residuals[:, t:t + 1] = residual
            self.local_loss += float(np.linalg.norm(residual) ** 2)

            # Line 11 provides cooperative estimate via DKF (fixed across rounds)
            h_coop_t = self.X_dkf[:, t:t + 1]

            # Line 13: augmented estimate incorporating learned correction
            h_est_t = h_coop_t + self.theta @ y_t
            self.h_aug_est[:, t:t + 1] = h_est_t

            # Prepare for next step sampling h_{m,a}^{k,t}
            prev_estimate = h_est_t

        # self.local_loss = self.local_loss / self.T + self.lambda_l * np.linalg.norm(self.theta, 'fro') ** 2
        self.local_loss = self.local_loss / self.T
        self._forward_ran = True

        return {
            'h_aug_pred': self.h_aug_pred.copy(),
            'h_aug_est': self.h_aug_est.copy(),
            'h_coop': self.X_dkf[:, :self.T].copy(),
            'residuals': self.y_residuals.copy(),
            'local_loss': float(self.local_loss),
        }

    def gradient_descent_update(self, gradx: np.ndarray, gradx_est: np.ndarray) -> None:
        """Update ``theta`` and ``phi`` using local and server gradients (Alg. 1, lines 35–39).

        Parameters
        ----------
        gradx : np.ndarray
            Matrix of shape ``(p_m, T)`` containing ``∇_{h_{m,a}} L_s^k`` for
            each time step ``t`` provided by the server.
        gradx_est : np.ndarray
            Matrix of shape ``(p_m, T)`` with ``∇_{\hat{h}_{m,a}} L_s^k`` aligned so
            column ``t`` corresponds to ``∂L_s/∂\hat{h}_{m,a}^{t}``.
        """
        if gradx.shape != (self.p_m, self.T):
            raise ValueError(
                f"gradx has shape {gradx.shape}; expected {(self.p_m, self.T)}"
            )
        if gradx_est.shape != (self.p_m, self.T):
            raise ValueError(
                f"gradx_est has shape {gradx_est.shape}; expected {(self.p_m, self.T)}"
            )
        if not self._forward_ran:
            raise RuntimeError(
                "local_forward_pass must be executed before gradient updates"
            )

        residuals = self.y_residuals[:, :self.T]
        y_prev = np.zeros((self.d_m, self.T))
        y_prev[:, :1] = self.y_init
        if self.T > 1:
            y_prev[:, 1:] = self.Y[:, :-1]

        # Local gradients (lines 12–14 derivatives)
        cT_residuals = self.C.T @ residuals
        grad_phi_local = -(2.0 / self.T) * np.sum(cT_residuals, axis=1, keepdims=True)
        grad_theta_local = -(2.0 / self.T) * (self.A.T @ cT_residuals) @ y_prev.T

        # Server-induced gradients (line 36 accumulation)
        grad_theta_server = (self.A.T @ gradx) @ y_prev.T
        grad_theta_server += gradx_est @ y_prev.T
        grad_phi_server = np.sum(gradx, axis=1, keepdims=True)

        # Parameter updates (lines 39–40)
        self.theta = self.theta - self.eta_1 * grad_theta_local - self.eta_2 * grad_theta_server
        self.phi = self.phi - self.gamma_1 * grad_phi_local - self.gamma_2 * grad_phi_server

    # ------------------------------------------------------------------
    # Helpers to update local parameters between communication rounds
    # ------------------------------------------------------------------
    def set_theta(self, theta: np.ndarray) -> None:
        if theta.shape != (self.p_m, self.d_m):
            raise ValueError(
                f"theta has shape {theta.shape}; expected {(self.p_m, self.d_m)}"
            )
        self.theta = theta.copy()

    def set_phi(self, phi: np.ndarray) -> None:
        if phi.shape != (self.p_m, 1):
            raise ValueError(
                f"phi has shape {phi.shape}; expected {(self.p_m, 1)}"
            )
        self.phi = phi.copy()

    def get_local_payload(self) -> dict[str, np.ndarray | float]:
        """Return the most recent forward-pass data without recomputing."""
        return {
            'h_aug_pred': self.h_aug_pred.copy(),
            'h_aug_est': self.h_aug_est.copy(),
            'h_coop': self.X_dkf[:, :self.T].copy(),
            'residuals': self.y_residuals.copy(),
            'local_loss': float(self.local_loss),
        }
