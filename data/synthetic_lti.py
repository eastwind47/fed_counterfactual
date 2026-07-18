"""
synthetic_lti.py
----------------
Generates synthetic controlled Linear Time-Invariant (LTI) state-space data for the
federated counterfactual experiments, saved as separate train and test splits.

The system model is:
    x_{t+1} = A x_t + B u_t + w_t,    w_t ~ N(0, Q)   process noise
    y_t     = C x_t         + v_t,    v_t ~ N(0, R)   measurement noise

where (P_total = sum P_m, D_total = sum D_m, S_total = sum S_m over N components):
    x_t in R^P_total  — global state
    y_t in R^D_total  — global observation
    u_t in R^S_total  — persistently-exciting control input

Diagonal blocks A_mm, B_mm and the block-diagonal C_mm are the locally-known matrices.
Off-diagonal blocks A_mn (state coupling) and B_mn (control coupling) are the unknown
inter-component coupling the federated learner estimates. Both splits share the same
true matrices A, B, C, Q, R; only the control realization, noise, and x0 differ.

Why it is needed:
    Produces the benchmark data for the whole project. The train split is used to learn
    A_mn, B_mn, theta, phi; the test split is a held-out factual trajectory used as the
    ground-truth factual for counterfactual experiments and for validating estimation.

Inputs:
    configs/data_linear.yaml — system dimensions, coupling graph, matrix/noise/control
    parameters, and per-split (train/test) horizons and initial-state distributions.

Outputs (saved under output_path/train/ and output_path/test/):
    Global files (full stacked system):
        A_complete.csv, B_complete.csv, C_complete.csv, Q_complete.csv, R_complete.csv,
        x0_complete.csv, U_complete.csv, X_complete.csv, Y_complete.csv
    Per-component subdirectories C1/, C2/, ... (the federated view — each client sees
    only its own slice and its own diagonal blocks):
        X.csv, Y.csv, U.csv, x0.csv,
        A.csv (= A_mm), B.csv (= B_mm), C.csv (= C_mm), Q.csv (= Q_mm), R.csv (= R_mm),
        X_dkf.csv, X_dkf_pred.csv   — precomputed distributed-KF filtered and
                                      one-step-predicted states for this component,
                                      from a filter using only its local blocks.
    shift_labels.csv (split root; only when controls.type == mean_shift): (T,) 0/1
        vector marking timesteps inside any mean-shift window (ground-truth event mask).
    dataset_params.json (at output_path root): config snapshot + diagnostics
    (spectral radius of A, per-split persistence-of-excitation eigenvalues).

    NOTE: all trajectory CSVs are row-major (T, dim) — one row per timestep. The data
    loader transposes them to the (dim, T) column-major layout used by the algorithms.

Entry point: generate_data.py (project root). Imports core.KalmanFilter for the
per-client DKF precompute.
"""

import json
from pathlib import Path

import numpy as np
import yaml
from scipy.linalg import block_diag, eigvals

from core.KalmanFilter import KalmanFilter

# Config path resolved relative to this file so generate_data.py works from any cwd.
CONFIG_PATH = Path(__file__).parent.parent / "configs" / "data_linear.yaml"


def _load_config():
    """
    Load and return the YAML config as a nested Python dictionary.

    Returns
    -------
    cfg : dict
        Parsed contents of configs/data_linear.yaml.
    """
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _block_starts(dims):
    """
    Cumulative start index of each component's block in a stacked global vector.

    Parameters
    ----------
    dims : list of int
        Per-component dimensions, e.g. state_dims [2,2,2] -> starts [0,2,4].

    Returns
    -------
    starts : list of int
        starts[m] is the first global index of component m's block.
    """
    return [sum(dims[:m]) for m in range(len(dims))]


def _build_A(N, state_dims, dependencies, cfg_A, rng):
    """
    Build a stable block-structured state matrix A.

    Steps:
      1. Diagonal blocks A_mm: uniform entries in [0.1, 1.0] (local dynamics, known).
      2. Off-diagonal blocks A_mn: for each declared edge (src, dst), fill block
         A[dst_rows, src_cols] with U[0.1, 1.0] * off_diagonal_scale (coupling to learn).
      3. Stability: rescale the full A so its spectral radius equals
         spectral_radius_target, placing all eigenvalues strictly inside the unit circle.

    Parameters
    ----------
    N : int
        Number of components.
    state_dims : list of int
        State dimension P_m per component.
    dependencies : list of [src, dst]
        Directed coupling edges (0-based). Edge (src, dst) => A[dst, src] nonzero.
    cfg_A : dict
        Keys: off_diagonal_scale (float, default 1.0),
              spectral_radius_target (float, default 1/1.1 ~= 0.909).
    rng : np.random.RandomState
        Seeded RNG for reproducibility.

    Returns
    -------
    A : ndarray, shape (P_total, P_total)
        Stable state matrix, spectral radius == spectral_radius_target.
    """
    P_starts = _block_starts(state_dims)
    alpha = cfg_A.get("off_diagonal_scale", 1.0)   # A_mn magnitude vs A_mm

    # Step 1: diagonal blocks A_mm, entries in [0.1, 1.0] (strictly positive).
    diag_blocks = [rng.rand(state_dims[m], state_dims[m]) * 0.9 + 0.1
                   for m in range(N)]
    A = block_diag(*diag_blocks)   # shape (P_total, P_total); off-diagonals still 0

    # Step 2: off-diagonal coupling blocks A_mn on the declared edges only.
    for src, dst in dependencies:
        r0, r1 = P_starts[dst], P_starts[dst] + state_dims[dst]   # dst state rows
        c0, c1 = P_starts[src], P_starts[src] + state_dims[src]   # src state cols
        A[r0:r1, c0:c1] = (rng.rand(state_dims[dst], state_dims[src]) * 0.9 + 0.1) * alpha

    # Step 3: stability scaling. rho = max|eig(A)|; scale so max|eig| == target.
    target_rho = cfg_A.get("spectral_radius_target", 1.0 / 1.1)
    rho = np.max(np.abs(eigvals(A)))
    A *= (target_rho / rho)
    return A


def _resolve_B_edges(N, coupling, system_deps, custom_deps):
    """
    Resolve which off-diagonal B_mn blocks are nonzero, as a list of (src, dst) edges.

    Parameters
    ----------
    N : int
        Number of components.
    coupling : str
        One of "dense", "shared", "custom", "none":
          "dense"  -> every ordered pair (src, dst), src != dst
          "shared" -> the state-coupling edges system_deps
          "custom" -> the explicit edges custom_deps
          "none"   -> no off-diagonal blocks (block-diagonal B)
    system_deps : list of [src, dst]
        The state-coupling edges from system.dependencies (used by "shared").
    custom_deps : list of [src, dst]
        The B-specific edges from matrices.B.dependencies (used by "custom").

    Returns
    -------
    edges : list of (int, int)
        Directed (src, dst) edges; edge (src, dst) => B[dst_rows, src_cols] nonzero.
    """
    if coupling == "dense":
        return [(src, dst) for dst in range(N) for src in range(N) if src != dst]
    if coupling == "shared":
        return [tuple(e) for e in system_deps]
    if coupling == "custom":
        return [tuple(e) for e in custom_deps]
    if coupling == "none":
        return []
    raise ValueError(f"Unknown B coupling '{coupling}'. "
                     f"Choose: dense | shared | custom | none.")


def _build_B(N, state_dims, input_dims, system_deps, cfg_B, rng):
    """
    Build the block input matrix B (diagonal B_mm known; off-diagonal B_mn to learn).

    Diagonal blocks B_mm ~ U(diag_low, diag_high), shape (P_m, S_m). Off-diagonal
    blocks B_mn are placed on the edges selected by cfg_B["coupling"] (see
    _resolve_B_edges), drawn from U(diag_low, diag_high) * off_diagonal_scale.
    B is NOT stability-scaled — it does not affect the state-transition spectrum.

    Parameters
    ----------
    N : int
        Number of components.
    state_dims : list of int
        State dimension P_m per component (B block rows).
    input_dims : list of int
        Input dimension S_m per component (B block cols).
    system_deps : list of [src, dst]
        system.dependencies, used when coupling == "shared".
    cfg_B : dict
        Keys: diag_low (float), diag_high (float), off_diagonal_scale (float),
              coupling (str), dependencies (list, used when coupling == "custom").
    rng : np.random.RandomState
        Seeded RNG for reproducibility.

    Returns
    -------
    B : ndarray, shape (P_total, S_total)
        Block input matrix. P_total = sum(state_dims), S_total = sum(input_dims).
    """
    low = cfg_B["diag_low"]
    high = cfg_B["diag_high"]
    beta = cfg_B.get("off_diagonal_scale", 1.0)   # B_mn magnitude vs B_mm

    P_starts = _block_starts(state_dims)
    S_starts = _block_starts(input_dims)

    # Diagonal blocks B_mm assembled block-diagonally.
    diag_blocks = [rng.uniform(low, high, size=(state_dims[m], input_dims[m]))
                   for m in range(N)]
    B = block_diag(*diag_blocks)   # shape (P_total, S_total); off-diagonals still 0

    # Off-diagonal coupling blocks B_mn on the resolved edges.
    edges = _resolve_B_edges(N, cfg_B["coupling"], system_deps, cfg_B.get("dependencies", []))
    for src, dst in edges:
        r0, r1 = P_starts[dst], P_starts[dst] + state_dims[dst]   # dst state rows
        c0, c1 = S_starts[src], S_starts[src] + input_dims[src]   # src input cols
        B[r0:r1, c0:c1] = rng.uniform(low, high, size=(state_dims[dst], input_dims[src])) * beta
    return B


def _build_C(N, state_dims, obs_dims, cfg_C, rng):
    """
    Build the block-diagonal observation matrix C (each client observes its own state).

    Supported types:
        "normal"         : C_m[i,j] ~ N(mean, std^2). Full column rank a.s. when D_m>=P_m.
        "uniform"        : C_m[i,j] ~ U(low, high). Use a zero-centered interval to avoid
                           near-collinear columns.
        "scaled_uniform" : C_m = scale_m * ones((D_m, P_m)). Rank 1 (kept for compatibility).

    Parameters
    ----------
    N : int
        Number of components.
    state_dims : list of int
        State dimension P_m per component.
    obs_dims : list of int
        Observation dimension D_m per component.
    cfg_C : dict
        type (str) and its parameters (mean/std, or low/high, or scales).
    rng : np.random.RandomState
        Seeded RNG for reproducibility.

    Returns
    -------
    C : ndarray, shape (D_total, P_total)
        Block-diagonal observation matrix.
    """
    if cfg_C["type"] == "normal":
        mean, std = cfg_C["mean"], cfg_C["std"]
        blocks = [rng.normal(mean, std, size=(obs_dims[m], state_dims[m])) for m in range(N)]
    elif cfg_C["type"] == "uniform":
        low, high = cfg_C["low"], cfg_C["high"]
        blocks = [rng.uniform(low, high, size=(obs_dims[m], state_dims[m])) for m in range(N)]
    elif cfg_C["type"] == "scaled_uniform":
        scales = cfg_C["scales"]
        blocks = [scales[m] * np.ones((obs_dims[m], state_dims[m])) for m in range(N)]
    else:
        raise NotImplementedError(f"C type '{cfg_C['type']}' not supported.")
    return block_diag(*blocks)


def _build_noise_covs(N, state_dims, obs_dims, cfg_Q, cfg_R):
    """
    Build block-diagonal process (Q) and measurement (R) noise covariances.

    Each component block is isotropic: Q_m = cfg_Q[m] * I_{P_m}, R_m = cfg_R[m] * I_{D_m}.

    Parameters
    ----------
    N : int
        Number of components.
    state_dims : list of int
        State dimension P_m per component.
    obs_dims : list of int
        Observation dimension D_m per component.
    cfg_Q : list of float
        Process-noise variance per component.
    cfg_R : list of float
        Measurement-noise variance per component.

    Returns
    -------
    Q : ndarray, shape (P_total, P_total)
        Block-diagonal process-noise covariance.
    R : ndarray, shape (D_total, D_total)
        Block-diagonal measurement-noise covariance.
    """
    Q_blocks = [cfg_Q[m] * np.eye(state_dims[m]) for m in range(N)]
    R_blocks = [cfg_R[m] * np.eye(obs_dims[m]) for m in range(N)]
    return block_diag(*Q_blocks), block_diag(*R_blocks)


def _build_controls(T, N, input_dims, cfg_controls, rng):
    """
    Build a control trajectory u according to the split's controls config.

    Two types:
      "gaussian"   : u[t] ~ N(mu, sigma^2) i.i.d. at every step (persistently exciting).
      "mean_shift" : u = 0 except inside declared windows; inside each window the
                     targeted components get u ~ N(mu, sigma^2). Also returns a label
                     vector marking window timesteps.

    Parameters
    ----------
    T : int
        Number of timesteps.
    N : int
        Number of components.
    input_dims : list of int
        Control dimension S_m per component (S_total = sum).
    cfg_controls : dict
        type (str) and its parameters:
          gaussian   -> mu (float, default 0.0), sigma (float)
          mean_shift -> windows (list of dicts: t_start, t_end, components, mu, sigma)
    rng : np.random.RandomState
        Seeded RNG for reproducibility.

    Returns
    -------
    u : ndarray, shape (T, S_total)
        Control-input trajectory (row-major).
    shift_labels : ndarray, shape (T,), dtype int, or None
        1 at timesteps inside any window (mean_shift only); None for gaussian.
    """
    S_total = sum(input_dims)
    ctype = cfg_controls["type"]

    if ctype == "gaussian":
        mu = cfg_controls.get("mu", 0.0)          # mean defaults to 0 if omitted
        sigma = cfg_controls["sigma"]
        return rng.normal(mu, sigma, size=(T, S_total)), None

    if ctype == "mean_shift":
        S_starts = _block_starts(input_dims)
        u = np.zeros((T, S_total))                 # zero baseline outside all windows
        shift_labels = np.zeros(T, dtype=int)
        for w in cfg_controls["windows"]:
            t0, t1 = w["t_start"], w["t_end"]
            mu, sigma = w["mu"], w["sigma"]
            # "all" targets every component; otherwise a list of 0-based component ids.
            comps = list(range(N)) if w["components"] == "all" else w["components"]
            for m in comps:
                s0, s1 = S_starts[m], S_starts[m] + input_dims[m]
                # Targeted component's input over [t0, t1): u ~ N(mu, sigma^2).
                u[t0:t1, s0:s1] = rng.normal(mu, sigma, size=(t1 - t0, input_dims[m]))
            shift_labels[t0:t1] = 1                 # mark this window's timesteps
        return u, shift_labels

    raise NotImplementedError(f"controls type '{ctype}' not supported.")


def check_persistence_of_excitation(u, tol=1e-6):
    """
    Test whether a control trajectory satisfies the persistence-of-excitation (PE) condition.

    PE requires the input Gramian G = sum_t u(t) u(t)^T = u^T u to be positive definite,
    i.e. all its eigenvalues strictly positive. Non-PE inputs leave directions of the
    input space unexcited, making the corresponding B columns unidentifiable.

    Parameters
    ----------
    u : ndarray, shape (T, S_total)
        Control trajectory (row-major).
    tol : float
        Threshold below which an eigenvalue is treated as zero. Default 1e-6.

    Returns
    -------
    is_pe : bool
        True iff every eigenvalue of the input Gramian exceeds tol.
    eigenvalues : ndarray, shape (S_total,)
        Sorted (ascending) eigenvalues of the input Gramian.
    """
    gramian = u.T @ u                          # shape (S_total, S_total)
    eigenvalues = np.sort(np.linalg.eigvalsh(gramian))
    return bool(np.all(eigenvalues > tol)), eigenvalues


def _build_x0(P_total, x0_mean, x0_std, rng):
    """
    Build the initial state x0.

    Parameters
    ----------
    P_total : int
        Total state dimension.
    x0_mean : float
        Mean of each initial-state element.
    x0_std : float
        Std of each initial-state element. If 0, x0 is the constant x0_mean.
    rng : np.random.RandomState
        Seeded RNG for reproducibility.

    Returns
    -------
    x0 : ndarray, shape (P_total,)
        Initial state vector.
    """
    if x0_std > 0:
        return rng.normal(x0_mean, x0_std, size=P_total)
    return np.full(P_total, x0_mean, dtype=float)


def _simulate(A, B, C, Q, R, T, u, x0, rng):
    """
    Simulate the controlled LTI trajectory for T timesteps.

    State update : x_{t+1} = A x_t + B u_t + w_t,   w_t ~ N(0, Q)
    Measurement  : y_t     = C x_t + v_t,           v_t ~ N(0, R)

    Noise is drawn in bulk before the loop so the trajectory is fully determined by the
    rng state at call time.

    Parameters
    ----------
    A : ndarray, shape (P_total, P_total)
        State matrix.
    B : ndarray, shape (P_total, S_total)
        Input matrix.
    C : ndarray, shape (D_total, P_total)
        Observation matrix.
    Q : ndarray, shape (P_total, P_total)
        Process-noise covariance.
    R : ndarray, shape (D_total, D_total)
        Measurement-noise covariance.
    T : int
        Number of timesteps.
    u : ndarray, shape (T, S_total)
        Control-input trajectory.
    x0 : ndarray, shape (P_total,)
        Initial state (x[0]).
    rng : np.random.RandomState
        Seeded RNG for the noise streams.

    Returns
    -------
    x : ndarray, shape (T, P_total)
        State trajectory (row-major). x[0] = x0.
    y : ndarray, shape (T, D_total)
        Observation trajectory (row-major).
    """
    n_x, n_y = A.shape[0], C.shape[0]
    w = rng.multivariate_normal(np.zeros(n_x), Q, size=T)   # shape (T, P_total)
    v = rng.multivariate_normal(np.zeros(n_y), R, size=T)   # shape (T, D_total)

    x = np.zeros((T, n_x))
    y = np.zeros((T, n_y))
    x[0] = np.asarray(x0, dtype=float)

    # y[t] uses x[t]; the final measurement y[T-1] is computed after the loop.
    for t in range(T - 1):
        x[t + 1] = A @ x[t] + B @ u[t] + w[t]
        y[t] = C @ x[t] + v[t]
    y[-1] = C @ x[-1] + v[-1]
    return x, y


def _run_dkf_precompute(N, state_dims, obs_dims, input_dims, A, B, C, Q, R, y, u):
    """
    Precompute each client's distributed Kalman filter (DKF) trajectory.

    For each component m, a Kalman filter using only the LOCAL diagonal blocks
    (A_mm, B_mm, C_mm, Q_mm, R_mm) is run on that component's REAL observations y_m
    and controls u_m (from the full coupled system) — the filter ignores cross-component
    coupling, exactly as at training time. It is initialized at zeros with unit covariance
    (it does not know the true x0). Caching these here lets the training loop read a fixed
    DKF reference from disk instead of re-running the filter on each realization.

    Parameters
    ----------
    N : int
        Number of components.
    state_dims, obs_dims, input_dims : list of int
        Per-component P_m, D_m, S_m.
    A : ndarray, shape (P_total, P_total)
        Full state matrix (only its diagonal blocks A_mm are used).
    B : ndarray, shape (P_total, S_total)
        Full input matrix (only its diagonal blocks B_mm are used).
    C : ndarray, shape (D_total, P_total)
        Block-diagonal observation matrix.
    Q : ndarray, shape (P_total, P_total)
        Block-diagonal process-noise covariance.
    R : ndarray, shape (D_total, D_total)
        Block-diagonal measurement-noise covariance.
    y : ndarray, shape (T, D_total)
        Full observation trajectory (row-major) from the coupled system.
    u : ndarray, shape (T, S_total)
        Full control trajectory (row-major).

    Returns
    -------
    X_dkf_list : list of ndarray, each shape (T, P_m)
        Filtered DKF states per component (row-major, ready to save).
    X_dkf_pred_list : list of ndarray, each shape (T, P_m)
        One-step-ahead predicted DKF states per component (row-major).
    """
    P_starts = _block_starts(state_dims)
    D_starts = _block_starts(obs_dims)
    S_starts = _block_starts(input_dims)

    X_dkf_list, X_dkf_pred_list = [], []
    for m in range(N):
        p0, p1 = P_starts[m], P_starts[m] + state_dims[m]
        d0, d1 = D_starts[m], D_starts[m] + obs_dims[m]
        s0, s1 = S_starts[m], S_starts[m] + input_dims[m]

        # Local (diagonal) blocks — the only matrices client m knows.
        A_mm = A[p0:p1, p0:p1]
        B_mm = B[p0:p1, s0:s1]
        C_mm = C[d0:d1, p0:p1]
        Q_mm = Q[p0:p1, p0:p1]
        R_mm = R[d0:d1, d0:d1]

        # Transpose this client's REAL observation/control slices to column-major (dim, T)
        # for the filter, which follows the (dim, T) in-memory algorithm convention.
        Y_m = y[:, d0:d1].T          # shape (D_m, T)
        U_m = u[:, s0:s1].T          # shape (S_m, T)

        kf = KalmanFilter(A_mm, B_mm, C_mm, Q_mm, R_mm)
        X_dkf_m, X_dkf_pred_m = kf.run(Y_m, U_m)   # both (P_m, T); x0=zeros, P0=I

        # Transpose back to row-major (T, P_m) for CSV storage.
        X_dkf_list.append(X_dkf_m.T)
        X_dkf_pred_list.append(X_dkf_pred_m.T)
    return X_dkf_list, X_dkf_pred_list


def _save_split(split_dir, N, state_dims, obs_dims, input_dims, A, B, C, Q, R, x, y, u,
                X_dkf_list, X_dkf_pred_list, shift_labels=None):
    """
    Save one split's global matrices/trajectories and per-component slices to disk.

    Global files hold the full stacked system; per-component subdirectories C1/, C2/, ...
    hold each client's own slice and its own DIAGONAL blocks only (A_mm, B_mm, C_mm) —
    the off-diagonal coupling A_mn, B_mn are withheld (they are what the learner estimates)
    — plus the precomputed DKF reference trajectories.

    Parameters
    ----------
    split_dir : Path
        Directory for this split (e.g. output_path/train).
    N : int
        Number of components.
    state_dims, obs_dims, input_dims : list of int
        Per-component P_m, D_m, S_m.
    A : ndarray, shape (P_total, P_total)
        State matrix.
    B : ndarray, shape (P_total, S_total)
        Input matrix.
    C : ndarray, shape (D_total, P_total)
        Observation matrix.
    Q : ndarray, shape (P_total, P_total)
        Process-noise covariance.
    R : ndarray, shape (D_total, D_total)
        Measurement-noise covariance.
    x : ndarray, shape (T, P_total)
        State trajectory (row-major).
    y : ndarray, shape (T, D_total)
        Observation trajectory (row-major).
    u : ndarray, shape (T, S_total)
        Control trajectory (row-major).
    X_dkf_list : list of ndarray, each shape (T, P_m)
        Precomputed filtered DKF states per component.
    X_dkf_pred_list : list of ndarray, each shape (T, P_m)
        Precomputed one-step-predicted DKF states per component.
    shift_labels : ndarray, shape (T,) or None
        Mean-shift event mask (1 inside any window); saved only when not None.
    """
    split_dir.mkdir(parents=True, exist_ok=True)

    # --- Global files ---
    np.savetxt(split_dir / "A_complete.csv", A, delimiter=",")
    np.savetxt(split_dir / "B_complete.csv", B, delimiter=",")
    np.savetxt(split_dir / "C_complete.csv", C, delimiter=",")
    np.savetxt(split_dir / "Q_complete.csv", Q, delimiter=",")
    np.savetxt(split_dir / "R_complete.csv", R, delimiter=",")
    np.savetxt(split_dir / "X_complete.csv", x, delimiter=",")
    np.savetxt(split_dir / "Y_complete.csv", y, delimiter=",")
    np.savetxt(split_dir / "U_complete.csv", u, delimiter=",")
    np.savetxt(split_dir / "x0_complete.csv", x[0:1, :], delimiter=",")   # shape (1, P_total)

    # Mean-shift event mask (present only for mean_shift controls).
    if shift_labels is not None:
        np.savetxt(split_dir / "shift_labels.csv", shift_labels, delimiter=",", fmt="%d")

    P_starts = _block_starts(state_dims)
    D_starts = _block_starts(obs_dims)
    S_starts = _block_starts(input_dims)

    # --- Per-component files ---
    for m in range(N):
        comp_dir = split_dir / f"C{m + 1}"
        comp_dir.mkdir(exist_ok=True)
        p0, p1 = P_starts[m], P_starts[m] + state_dims[m]
        d0, d1 = D_starts[m], D_starts[m] + obs_dims[m]
        s0, s1 = S_starts[m], S_starts[m] + input_dims[m]

        # Trajectories for component m
        np.savetxt(comp_dir / "X.csv", x[:, p0:p1], delimiter=",")
        np.savetxt(comp_dir / "Y.csv", y[:, d0:d1], delimiter=",")
        np.savetxt(comp_dir / "U.csv", u[:, s0:s1], delimiter=",")
        np.savetxt(comp_dir / "x0.csv", x[0:1, p0:p1], delimiter=",")

        # Locally-known DIAGONAL blocks only (off-diagonal A_mn/B_mn withheld).
        np.savetxt(comp_dir / "A.csv", A[p0:p1, p0:p1], delimiter=",")
        np.savetxt(comp_dir / "B.csv", B[p0:p1, s0:s1], delimiter=",")
        np.savetxt(comp_dir / "C.csv", C[d0:d1, p0:p1], delimiter=",")
        np.savetxt(comp_dir / "Q.csv", Q[p0:p1, p0:p1], delimiter=",")
        np.savetxt(comp_dir / "R.csv", R[d0:d1, d0:d1], delimiter=",")

        # Precomputed distributed-KF reference for component m (filtered + predicted).
        np.savetxt(comp_dir / "X_dkf.csv", X_dkf_list[m], delimiter=",")
        np.savetxt(comp_dir / "X_dkf_pred.csv", X_dkf_pred_list[m], delimiter=",")


def generate_and_save(output_path):
    """
    Main entry point for controlled-LTI data generation.

    Builds the shared true matrices A, B, C, Q, R once, then generates and saves the
    train and test splits (each with its own controls, noise, x0, and precomputed DKF).
    Prints diagnostics (spectral radius of A, persistence-of-excitation of each split's
    controls) and writes dataset_params.json for provenance.

    Parameters
    ----------
    output_path : str
        Root directory for the dataset. Created if absent. Splits are written to
        output_path/train/ and output_path/test/.
    """
    cfg = _load_config()
    out = Path(output_path)

    # --- System dimensions and coupling graph ---
    N = cfg["system"]["num_components"]
    state_dims = cfg["system"]["state_dims"]
    obs_dims = cfg["system"]["obs_dims"]
    input_dims = cfg["system"]["input_dims"]
    seed = cfg["system"]["seed"]
    deps = cfg["system"]["dependencies"]
    P_total = sum(state_dims)

    # Dedicated RNGs so streams stay independent: changing the controls/noise of one
    # split (or the matrices) never shifts another. Offsets are arbitrary but fixed.
    rng_mat = np.random.RandomState(seed)              # matrix construction
    rng_train = np.random.RandomState(seed + 1000)     # train controls + x0 + noise
    rng_test = np.random.RandomState(seed + 2000)      # test controls + x0 + noise

    # --- Shared true matrices (identical across splits) ---
    A = _build_A(N, state_dims, deps, cfg["matrices"]["A"], rng_mat)
    B = _build_B(N, state_dims, input_dims, deps, cfg["matrices"]["B"], rng_mat)
    C = _build_C(N, state_dims, obs_dims, cfg["matrices"]["C"], rng_mat)
    Q, R = _build_noise_covs(N, state_dims, obs_dims,
                             cfg["matrices"]["Q"], cfg["matrices"]["R"])

    rho = float(np.max(np.abs(eigvals(A))))

    # --- Generate each split ---
    pe_report = {}
    for split, rng_split in (("train", rng_train), ("test", rng_test)):
        T = cfg[split]["total_time"]
        x0 = _build_x0(P_total, cfg[split]["x0_mean"], cfg[split]["x0_std"], rng_split)
        u, shift_labels = _build_controls(T, N, input_dims, cfg[split]["controls"], rng_split)
        x, y = _simulate(A, B, C, Q, R, T, u, x0, rng_split)
        X_dkf_list, X_dkf_pred_list = _run_dkf_precompute(
            N, state_dims, obs_dims, input_dims, A, B, C, Q, R, y, u)
        _save_split(out / split, N, state_dims, obs_dims, input_dims,
                    A, B, C, Q, R, x, y, u, X_dkf_list, X_dkf_pred_list, shift_labels=shift_labels)

        is_pe, pe_eigs = check_persistence_of_excitation(u)
        pe_report[split] = {"control_type": cfg[split]["controls"]["type"],
                            "is_pe": is_pe, "min_gramian_eig": float(pe_eigs[0])}
        print(f"{split} data saved  : {(out / split).resolve()}  "
              f"(T={T}, controls={cfg[split]['controls']['type']}, PE={is_pe})")

    # --- Provenance ---
    params = {
        "generator": "data/synthetic_lti.py",
        "config": cfg,
        "diagnostics": {"spectral_radius_A": rho, "persistence_of_excitation": pe_report},
    }
    with open(out / "dataset_params.json", "w") as f:
        json.dump(params, f, indent=2)

    print(f"Spectral radius of A : {rho:.6f}")
    print(f"dataset_params.json  : {(out / 'dataset_params.json').resolve()}")
