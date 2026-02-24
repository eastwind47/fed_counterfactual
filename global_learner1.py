import numpy as np


class GlobalModel:
    """Global learner mirroring the original structure with B couplings."""

    def __init__(self, M, T, global_learner_data, x_dkf):
        # Core dimensions
        self.M = M
        self.T = T

        # Reference trajectories (h_{m,c} in the pseudocode)
        self.x_dkf = {key: value.copy() for key, value in x_dkf.items()}

        # Server-side hyper-parameters
        self.alpha_a = global_learner_data["alpha_a"]
        self.alpha_b = global_learner_data["alpha_b"]
        self.lambda_g = global_learner_data["lambda_g"]
        self.xi = global_learner_data["xi"]

        # Diagonal blocks
        self.A_mm = global_learner_data["A_mm"]
        self.B_mm = global_learner_data["B_mm"]

        # Off-diagonal initialisations
        self.A_mn = {
            key: value.copy() for key, value in global_learner_data["A_init_offdiag"].items()
        }
        self.B_mn = {
            key: value.copy() for key, value in global_learner_data["B_init_offdiag"].items()
        }

        # Vectorised copies (useful for logging)
        self.A_mn_vec = {key: mat.flatten() for key, mat in self.A_mn.items()}
        self.B_mn_vec = {key: mat.flatten() for key, mat in self.B_mn.items()}

        # Optional dimension helpers (may be absent in some configs)
        self.p_vec = global_learner_data.get("p_vec", [])
        self.s_vec = global_learner_data.get("s_vec", [])

        # Tracking containers for diagnostics
        self.server_loss = 0.0
        self.gradx_history = {
            f"{m+1}": np.zeros_like(self.x_dkf[f"{m+1}"])
            for m in range(self.M)
        }
        self.gradx_est_history = {
            f"{m+1}": np.zeros_like(self.x_dkf[f"{m+1}"])
            for m in range(self.M)
        }

        self.grad_A = {key: np.zeros_like(val) for key, val in self.A_mn.items()}
        self.grad_B = {key: np.zeros_like(val) for key, val in self.B_mn.items()}
        self.grad_A_norms = {key: 0.0 for key in self.A_mn}
        self.grad_B_norms = {key: 0.0 for key in self.B_mn}
        self.grad_A_total = 0.0
        self.grad_B_total = 0.0
        self.loss_align = 0.0
        self.loss_consensus = 0.0

        # Server-side trajectory trackers
        self.h_server = {
            key: np.zeros_like(self.x_dkf[key]) for key in self.x_dkf
        }
        self.align_residuals = {
            key: np.zeros_like(self.x_dkf[key]) for key in self.x_dkf
        }
        self.consensus_history = {
            key: np.zeros_like(self.x_dkf[key]) for key in self.x_dkf
        }

    # ------------------------------------------------------------------
    # Algorithm 1 (server) – lines 23–32
    # ------------------------------------------------------------------
    def server_forward_pass(self, h_aug_pred, h_aug_est, u_global):
        """Compute ``h_{m,s}^{k,t}``, ``L_s`` and gradients (Alg. 1 lines 23–32).

        Parameters
        ----------
        h_aug_pred : dict
            Mapping component id → ``h_{m,a}^{k,t}`` rollout (line 10 output).
        h_aug_est : dict
            Mapping component id → ``\hat{h}_{m,a}^{k,t}`` augmented estimate.
        u_global : dict
            Mapping component id → inputs ``u_m`` with shape ``(s_m, T)``.
        """

        self.h_server = {
            key: np.zeros_like(self.x_dkf[key]) for key in self.x_dkf
        }
        self.align_residuals = {
            key: np.zeros_like(self.x_dkf[key]) for key in self.x_dkf
        }
        self.consensus_history = {
            key: np.zeros_like(self.x_dkf[key]) for key in self.x_dkf
        }
        self.gradx_est_history = {
            key: np.zeros_like(self.x_dkf[key]) for key in self.x_dkf
        }

        total_loss = 0.0
        loss_align_total = 0.0
        loss_consensus_total = 0.0
        gradx = {
            key: np.zeros_like(h_aug_pred[key]) for key in h_aug_pred
        }
        gradx_est = {
            key: np.zeros_like(h_aug_est[key]) for key in h_aug_est
        }
        grad_A_accum = {key: np.zeros_like(val) for key, val in self.grad_A.items()}
        grad_B_accum = {key: np.zeros_like(val) for key, val in self.grad_B.items()}

        for t in range(1, self.T):
            for m in range(self.M):
                key_m = f"{m+1}"
                key_diag = f"{m+1}{m+1}"

                h_c_prev = self.x_dkf[key_m][:, t-1:t]
                u_prev_m = u_global[key_m][:, t-1:t]

                h_s_t = self.A_mm[key_diag] @ h_c_prev + self.B_mm[key_diag] @ u_prev_m
                consensus_res = self.A_mm[key_diag] @ (h_aug_est[key_m][:, t-1:t] - h_c_prev)

                for n in range(self.M):
                    if n == m:
                        continue
                    key_n = f"{n+1}"
                    key_off = f"{m+1}{n+1}"
                    h_c_peer = self.x_dkf[key_n][:, t-1:t]
                    u_peer = u_global[key_n][:, t-1:t]

                    A_off = self.A_mn.get(key_off)
                    if A_off is not None:
                        h_s_t += A_off @ h_c_peer
                        consensus_res -= A_off @ h_c_peer

                    B_off = self.B_mn.get(key_off)
                    if B_off is not None:
                        h_s_t += B_off @ u_peer

                h_pred_t = h_aug_pred[key_m][:, t:t+1]
                residual_curr = h_pred_t - h_s_t

                self.h_server[key_m][:, t:t+1] = h_s_t
                self.align_residuals[key_m][:, t:t+1] = residual_curr
                self.consensus_history[key_m][:, t:t+1] = consensus_res

                term_align = np.linalg.norm(residual_curr) ** 2
                term_reg = self.xi * np.linalg.norm(consensus_res) ** 2
                total_loss += term_align + term_reg
                loss_align_total += term_align
                loss_consensus_total += term_reg

                gradx[key_m][:, t:t+1] = 2.0 * residual_curr
                combined = residual_curr + self.xi * consensus_res
                grad_hat_prev = 2.0 * self.xi * (self.A_mm[key_diag].T @ consensus_res)
                gradx_est[key_m][:, t-1:t] += grad_hat_prev

                for n in range(self.M):
                    if n == m:
                        continue
                    key_n = f"{n+1}"
                    key_off = f"{m+1}{n+1}"
                    h_c_peer = self.x_dkf[key_n][:, t-1:t]
                    u_peer = u_global[key_n][:, t-1:t]

                    if key_off in grad_A_accum:
                        grad_A_accum[key_off] += -2.0 * (combined @ h_c_peer.T)
                    if key_off in grad_B_accum:
                        grad_B_accum[key_off] += -2.0 * (residual_curr @ u_peer.T)

        loss_value = total_loss / self.T
        for key in gradx:
            gradx[key] /= self.T
            self.gradx_history[key][:, :] = gradx[key]
            self.gradx_est_history[key][:, :] = gradx_est[key] / self.T

        for key, value in grad_A_accum.items():
            self.grad_A[key][:, :] = value / self.T
        for key, value in grad_B_accum.items():
            self.grad_B[key][:, :] = value / self.T

        self.server_loss = loss_value
        self.loss_align = float(loss_align_total / self.T)
        self.loss_consensus = float(loss_consensus_total / self.T)
        self.grad_A_norms = {
            key: float(np.linalg.norm(val))
            for key, val in self.grad_A.items()
        }
        self.grad_B_norms = {
            key: float(np.linalg.norm(val))
            for key, val in self.grad_B.items()
        }
        self.grad_A_total = float(
            np.sqrt(sum(val * val for val in self.grad_A_norms.values()))
        )
        self.grad_B_total = float(
            np.sqrt(sum(val * val for val in self.grad_B_norms.values()))
        )

        return {
            "loss": loss_value,
            "loss_align": self.loss_align,
            "loss_consensus": self.loss_consensus,
            "gradx": {key: gradx[key].copy() for key in gradx},
            "gradx_est": {key: gradx_est[key].copy() for key in gradx_est},
            "grad_A_norms": self.grad_A_norms.copy(),
            "grad_B_norms": self.grad_B_norms.copy(),
            "grad_A_total": self.grad_A_total,
            "grad_B_total": self.grad_B_total,
        }

    def apply_server_updates(self):
        """Gradient step on ``A_mn`` and ``B_mn`` (Alg. 1 lines 29–31)."""
        for key, grad in self.grad_A.items():
            self.A_mn[key] -= self.alpha_a * grad
            self.A_mn_vec[key] = self.A_mn[key].flatten()

        for key, grad in self.grad_B.items():
            self.B_mn[key] -= self.alpha_b * grad
            self.B_mn_vec[key] = self.B_mn[key].flatten()

    def get_gradients_for_clients(self, include_est: bool = False):
        """Return ``∇_{h_{m,a}} L_s`` and optionally ``∇_{\hat{h}_{m,a}} L_s``."""
        primary = {
            key: value.copy() for key, value in self.gradx_history.items()
        }
        if include_est:
            secondary = {
                key: value.copy()
                for key, value in self.gradx_est_history.items()
            }
            return primary, secondary
        return primary
