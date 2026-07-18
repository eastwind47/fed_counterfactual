"""
KalmanFilter.py
---------------
Standard discrete-time linear Kalman filter with a control input, used as the
per-client distributed Kalman filter (DKF) baseline in the federated pipeline.

Per-client model (using only locally-known diagonal blocks):
    x_t = A x_{t-1} + B u_{t-1} + w,   w ~ N(0, Q)   process noise
    y_t = C x_t                + v,    v ~ N(0, R)   measurement noise

Why it is needed:
    Each client filters its own observations with only its local matrices
    (A_mm, B_mm, C_mm, Q_mm, R_mm) and its own control u_m, ignoring cross-component
    coupling. The filtered/predicted states form the cooperative baseline the VFL
    correction is added to. Run both at data-generation time (to precompute and cache
    the DKF trajectories) and inside the local learner at training time.

Inputs:
    System matrices A, B, C, Q, R; and per run an observation trajectory Y, a control
    trajectory U, and optionally an initial state x0 and error covariance P0.

Outputs:
    Filtered state trajectory X_dkf and one-step-ahead predicted trajectory X_dkf_pred,
    both column-major (P, T).
"""

import numpy as np


class KalmanFilter:
    """
    Discrete-time linear Kalman filter with control input.

    Convention (matches the legacy pipeline): each step first PREDICTS using the
    previous control u_{t-1} (zero at t=0) and records the a-priori (predicted) state,
    then UPDATES with the current measurement y_t and records the a-posteriori
    (filtered) state.

    Attributes
    ----------
    A : ndarray, shape (P, P)
        State-transition matrix (local block A_mm).
    B : ndarray, shape (P, S)
        Control matrix (local block B_mm).
    C : ndarray, shape (D, P)
        Observation matrix (local block C_mm).
    Q : ndarray, shape (P, P)
        Process-noise covariance.
    R : ndarray, shape (D, D)
        Measurement-noise covariance.
    x : ndarray, shape (P, 1)
        Current state estimate; set by reset()/predict()/update().
    P : ndarray, shape (P, P)
        Current estimate-error covariance; set by reset()/predict()/update().
    """

    def __init__(self, A, B, C, Q, R):
        """
        Parameters
        ----------
        A, B, C, Q, R : ndarray
            Local system matrices with shapes (P,P), (P,S), (D,P), (P,P), (D,D).
        """
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.x = None   # state estimate,   set on reset()
        self.P = None   # error covariance, set on reset()

    def reset(self, x0, P0):
        """
        Initialize the filter state and covariance before a run.

        Parameters
        ----------
        x0 : ndarray, shape (P,) or (P, 1)
            Initial state estimate.
        P0 : ndarray, shape (P, P)
            Initial estimate-error covariance.
        """
        self.x = np.asarray(x0, dtype=float).reshape(self.A.shape[0], 1)
        self.P = np.asarray(P0, dtype=float).copy()

    def predict(self, u):
        """
        A-priori step: x <- A x + B u ; P <- A P A^T + Q.

        Parameters
        ----------
        u : ndarray, shape (S, 1)
            Control input applied over the step being predicted.
        """
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q

    def residual(self, y):
        """
        Innovation y - C x (measurement residual against the current state estimate).

        Parameters
        ----------
        y : ndarray, shape (D, 1)
            Measurement.

        Returns
        -------
        r : ndarray, shape (D, 1)
            The residual y - C x.
        """
        return y - self.C @ self.x

    def update(self, y):
        """
        A-posteriori step: correct the state with measurement y via the Kalman gain.

        Parameters
        ----------
        y : ndarray, shape (D, 1)
            Measurement at the current timestep.
        """
        innovation = self.residual(y)                     # y - C x, shape (D, 1)
        S = self.C @ self.P @ self.C.T + self.R           # innovation covariance (D, D)
        K = self.P @ self.C.T @ np.linalg.inv(S)          # Kalman gain (P, D)
        self.x = self.x + K @ innovation
        self.P = self.P - K @ self.C @ self.P

    def run(self, Y, U, x0=None, P0=None):
        """
        Run the filter over a full trajectory (column-major).

        Uses the legacy stepping convention: at time t, predict with the previous
        control u_{t-1} (zero at t=0), store the predicted state, then update with y_t
        and store the filtered state.

        Parameters
        ----------
        Y : ndarray, shape (D, T)
            Observation trajectory (one column per timestep).
        U : ndarray, shape (S, T)
            Control trajectory (one column per timestep).
        x0 : ndarray, shape (P,) or (P, 1) or None
            Initial state. Defaults to zeros — the client does not know its true
            initial state (the DKF init convention shared with the uncert pipeline).
        P0 : ndarray, shape (P, P) or None
            Initial error covariance. Defaults to the identity I_P (neutral unit
            initial uncertainty).

        Returns
        -------
        X_dkf : ndarray, shape (P, T)
            Filtered (a-posteriori) state estimates.
        X_dkf_pred : ndarray, shape (P, T)
            One-step-ahead predicted (a-priori) state estimates.
        """
        P_dim = self.A.shape[0]
        S_dim = self.B.shape[1]
        T = Y.shape[1]

        # DKF init: zeros state, identity covariance (client is agnostic to true x0).
        if x0 is None:
            x0 = np.zeros((P_dim, 1))
        if P0 is None:
            P0 = np.eye(P_dim)
        self.reset(x0, P0)

        X_dkf = np.zeros((P_dim, T))
        X_dkf_pred = np.zeros((P_dim, T))
        for t in range(T):
            # Predict with the previous control u_{t-1}; zero at t=0 (no prior input).
            u_prev = U[:, t - 1:t] if t > 0 else np.zeros((S_dim, 1))
            self.predict(u_prev)
            X_dkf_pred[:, t:t + 1] = self.x               # a-priori (predicted) state
            self.update(Y[:, t:t + 1])
            X_dkf[:, t:t + 1] = self.x                    # a-posteriori (filtered) state
        return X_dkf, X_dkf_pred
